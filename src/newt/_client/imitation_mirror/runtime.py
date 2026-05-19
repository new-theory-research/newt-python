"""Robot inference client runtime — hardware setup and control loop."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import asyncio
import select
import socket
import sys
import termios
import time
import tty
from typing import Any

import numpy as np

# Hardware-only imports: torch and lerobot. Wrapped so the module can be imported
# without T4 deps on the X path (no hardware). Functions that call into lerobot
# will raise NameError / RuntimeError at call time when unavailable.
try:
    import torch
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
    from lerobot.robots.bi_widowx_ai import BiWidowXAIFollowerConfig
    from lerobot.robots.widowx_ai import WidowXAIFollowerConfig
    _LEROBOT_AVAILABLE = True
except ImportError:
    _LEROBOT_AVAILABLE = False

from nt._client.imitation_mirror.network_client import NetworkClient
from nt._client.imitation_mirror.robot import TrossenRobotClient
from nt._client.imitation_mirror.config import (
    InferenceClientConfig,
    InferenceRobotConfig,
)


# ---------------------------------------------------------------------------
# Pure data-transform helpers
# ---------------------------------------------------------------------------


_MAX_STDIN_DRAIN_BYTES = 4096


def _to_numpy(tree: Any) -> Any:
    """Recursively convert tensors in nested containers to numpy arrays."""
    if isinstance(tree, torch.Tensor):
        return tree.cpu().numpy()
    if isinstance(tree, dict):
        return {k: _to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [_to_numpy(v) for v in tree]
    return tree


def normalize_camera_name(name: str) -> str:
    """Normalize camera feature keys into bare camera names.

    Strips the ``observation.images.`` prefix when present.
    """
    if "." in name:
        return name.split(".")[-1]
    return name


def _drain_stdin_buffer() -> None:
    """Discard pending stdin input so resume waits for a fresh keypress.

    Uses ``termios.tcflush`` when available and falls back to non-blocking reads
    with a bounded iteration count if flushing is unsupported.
    """
    stdin_fd = sys.stdin.fileno()
    try:
        termios.tcflush(stdin_fd, termios.TCIFLUSH)
    except termios.error:
        for _ in range(_MAX_STDIN_DRAIN_BYTES):
            if not select.select([sys.stdin], [], [], 0)[0]:
                break
            sys.stdin.read(1)


def _image_to_chw(image: np.ndarray) -> np.ndarray:
    """Convert color image array to CHW layout."""
    if image.ndim == 3 and image.shape[-1] in (1, 3):
        return np.transpose(image, (2, 0, 1))
    if image.ndim == 3 and image.shape[0] in (1, 3):
        return image
    raise ValueError(f"Unexpected image shape {image.shape}; expected HWC or CHW.")


def _depth_to_chw(depth: np.ndarray) -> np.ndarray:
    """Convert depth image array to CHW layout."""
    if depth.ndim == 2:
        return depth[None, ...]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        return np.transpose(depth, (2, 0, 1))
    if depth.ndim == 3 and depth.shape[0] == 1:
        return depth
    raise ValueError(f"Unexpected depth shape {depth.shape}; expected HW or 1xHW.")


def _stack_camera_matrices(
    matrices: dict,
    camera_names: list[str],
    shape: tuple[int, int],
) -> np.ndarray:
    """Stack per-camera matrices in a deterministic camera order."""
    stacked: list[np.ndarray] = []
    for name in camera_names:
        matrix = matrices.get(name)
        if matrix is None:
            matrix = matrices.get(f"observation.images.{name}")
        if matrix is None:
            raise KeyError(f"Missing camera matrix for '{name}'.")
        matrix_array = np.asarray(matrix, dtype=np.float32)
        if matrix_array.shape != shape:
            raise ValueError(
                f"Camera matrix for '{name}' must be {shape}, got {matrix_array.shape}."
            )
        stacked.append(matrix_array)
    return np.stack(stacked, axis=0)


def _build_sample_from_buffer(
    obs_buffer: list[dict],
    camera_names: list[str],
    intrinsics_map: dict[str, list[list[float]]],
    extrinsics_map: dict,
    depth_scales: dict[str, float],
    language_text: str | None,
    language_text_key: str | None,
    visual_context_length: int | None = None,
    proprio_context_length: int | None = None,
) -> dict:
    """Build a batched wire-format sample from an observation history."""
    image_keys = [f"observation.images.{name}" for name in camera_names]
    depth_keys = [f"{key}.depth" for key in image_keys]

    images_seq: list[np.ndarray] = []
    depths_seq: list[np.ndarray] = []
    states_seq: list[np.ndarray] = []

    for frame in obs_buffer:
        images_per_frame: list[np.ndarray] = []
        depths_per_frame: list[np.ndarray] = []
        for image_key, depth_key in zip(image_keys, depth_keys):
            image = _image_to_chw(np.asarray(frame[image_key]))
            depth = _depth_to_chw(np.asarray(frame[depth_key])).astype(np.float32)
            scale = depth_scales[normalize_camera_name(image_key)]
            depth = depth * scale
            images_per_frame.append(image)
            depths_per_frame.append(depth)
        images_seq.append(np.stack(images_per_frame, axis=0))
        depths_seq.append(np.stack(depths_per_frame, axis=0))
        states_seq.append(np.asarray(frame["observation.state"], dtype=np.float32))

    vis_slice = (
        slice(-visual_context_length, None) if visual_context_length else slice(None)
    )
    proprio_slice = (
        slice(-proprio_context_length, None) if proprio_context_length else slice(None)
    )

    images = np.stack(images_seq[vis_slice], axis=0)[None, ...]
    depths = np.stack(depths_seq[vis_slice], axis=0)[None, ...]
    state = np.stack(states_seq[proprio_slice], axis=0)[None, ...]

    intrinsics = _stack_camera_matrices(intrinsics_map, camera_names, (3, 3))
    extrinsics = _stack_camera_matrices(extrinsics_map, camera_names, (4, 4))

    sample = {
        "observation": {
            "images": images,
            "depth_maps": depths,
            "state": state,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }
    }
    if language_text is not None and language_text != "":
        sample["metadata"] = {language_text_key: language_text}
    return sample


# ---------------------------------------------------------------------------
# Hardware setup helpers
# ---------------------------------------------------------------------------


def _load_extrinsics(config: InferenceClientConfig) -> dict:
    """Load camera extrinsics from site config (nt.toml)."""
    from nt._client.imitation_mirror.site_config import load_site_config

    site_path = getattr(config, "site_config_path", None)
    if site_path != "":
        site = load_site_config(site_path)
        if site is not None and site.cameras:
            return site.extrinsics_map()

    raise ValueError(
        "No camera extrinsics available. "
        "Define camera extrinsics in ~/.config/nt/nt.toml."
    )


def _build_intrinsics_matrices(calibration: dict) -> dict[str, list[list[float]]]:
    """Convert flat camera calibration dictionaries into 3x3 intrinsics."""
    intrinsics: dict[str, list[list[float]]] = {}
    for name, calib in calibration.items():
        fx = calib.get("fx")
        fy = calib.get("fy")
        cx = calib.get("cx")
        cy = calib.get("cy")
        if None in (fx, fy, cx, cy):
            raise ValueError(f"Calibration for camera '{name}' missing fx/fy/cx/cy.")
        intrinsics[name] = [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ]
    return intrinsics


def _build_cameras_config(
    robot_config: InferenceRobotConfig,
) -> dict[str, RealSenseCameraConfig]:
    """Build RealSense camera config map from runtime config."""
    if str(robot_config.type).lower() != "trossen":
        raise ValueError(
            f"Unsupported robot.type={robot_config.type!r}; expected 'trossen'."
        )

    camera_map: dict[str, RealSenseCameraConfig] = {}
    for name, camera in robot_config.cameras.items():
        camera_name = str(name)
        if camera_name in camera_map:
            raise ValueError(
                f"Duplicate camera name in robot.cameras: {camera_name!r}."
            )
        camera_map[camera_name] = RealSenseCameraConfig(
            serial_number_or_name=str(camera.serial_number_or_name),
            fps=int(camera.fps),
            width=int(camera.width),
            height=int(camera.height),
            use_depth=bool(camera.use_depth),
            use_depth_postprocessing=bool(camera.use_depth_postprocessing),
        )

    if not camera_map:
        raise ValueError("robot.cameras must contain at least one camera config.")
    return camera_map


def _build_wrist_camera_names(robot_config: InferenceRobotConfig) -> set[str]:
    """Collect wrist camera names from robot camera runtime config."""
    return {
        str(name)
        for name, camera in robot_config.cameras.items()
        if bool(getattr(camera, "is_wrist_camera", False))
    }


def _build_robot_config(
    cameras_config: dict[str, RealSenseCameraConfig],
    robot_config: InferenceRobotConfig,
):
    """Build low-level robot config from the arm config set during config resolution."""
    from nt._client.imitation_mirror.config import (
        TrossenBimanualConfig,
        TrossenSingleArmConfig,
    )

    if str(robot_config.type).lower() != "trossen":
        raise ValueError(
            f"Unsupported robot.type={robot_config.type!r}; expected 'trossen'."
        )

    arm = robot_config.arm
    if arm is None:
        raise ValueError(
            "robot.arm is not set. Ensure the arm profile was applied "
            "during config resolution."
        )

    if isinstance(arm, TrossenSingleArmConfig):
        single_ip = str(arm.ip or "").strip()
        if single_ip == "":
            raise ValueError("robot.arm.ip must be set for single-arm inference.")
        return WidowXAIFollowerConfig(
            ip=single_ip,
            goal_time_multiplier=float(arm.goal_time_multiplier),
            cameras=cameras_config,
        )

    if isinstance(arm, TrossenBimanualConfig):
        left_ip = str(arm.left_arm_ip or "").strip()
        right_ip = str(arm.right_arm_ip or "").strip()
        if left_ip == "" or right_ip == "":
            raise ValueError(
                "robot.arm.left_arm_ip and robot.arm.right_arm_ip "
                "must be set for bimanual inference."
            )
        return BiWidowXAIFollowerConfig(
            left_arm_ip=left_ip,
            right_arm_ip=right_ip,
            left_arm_goal_time_multiplier=float(arm.left_arm_goal_time_multiplier),
            right_arm_goal_time_multiplier=float(arm.right_arm_goal_time_multiplier),
            cameras=cameras_config,
        )

    raise TypeError(f"Unsupported arm config type: {type(arm).__name__}.")


# ---------------------------------------------------------------------------
# Main entry point — hardware setup + control loop
# ---------------------------------------------------------------------------


async def run_inference_client(
    config: InferenceClientConfig,
    client: NetworkClient,
    server_config: dict,
) -> None:
    """Run the online robot inference loop.

    Expects a fully resolved config (cameras already filtered, arm set)
    and a pre-connected ``NetworkClient`` with its server config.

    Args:
        config: Fully resolved client runtime configuration.
        client: Pre-connected network client for inference requests.
        server_config: Server runtime metadata from ``/config``.
    """
    # Read server metadata.
    visual_context_length = server_config.get("visual_context_length")
    proprio_context_length = server_config.get("proprio_context_length")
    arm_count = server_config.get("arm_count")
    action_horizon = server_config.get("action_horizon")
    target_fps = server_config.get("target_fps")
    language_text_key = server_config.get("language_text_key")
    if visual_context_length is None:
        raise ValueError("Server config missing visual_context_length.")
    if proprio_context_length is None:
        raise ValueError("Server config missing proprio_context_length.")
    if arm_count is None:
        raise ValueError("Server config missing arm_count.")
    if language_text_key is None and config.language_text:
        raise ValueError(
            "Server config missing language_text_key but client has language_text set."
        )

    # --- Hardware setup ---------------------------------------------------

    cameras_config = _build_cameras_config(config.robot)
    extrinsics = _load_extrinsics(config)

    for wrist_camera_name in _build_wrist_camera_names(config.robot):
        short_key = wrist_camera_name
        full_key = f"observation.images.{wrist_camera_name}"
        if short_key not in extrinsics and full_key not in extrinsics:
            extrinsics[short_key] = np.eye(4, dtype=np.float32)

    robot_impl_config = _build_robot_config(cameras_config, config.robot)
    robot = TrossenRobotClient(robot_config=robot_impl_config)
    robot.connect()
    robot.set_action_contract(arm_count=int(arm_count))

    # --- Validation and diagnostics ---------------------------------------

    action_interval = float(config.action_interval)
    max_actions_per_chunk = int(config.max_actions_per_chunk)
    if max_actions_per_chunk < 0:
        raise ValueError("max_actions_per_chunk must be >= 0.")
    if max_actions_per_chunk > 0:
        print(f"Limiting executed actions per chunk to first {max_actions_per_chunk}.")
    else:
        print(
            "Warning: max_actions_per_chunk=0 executes full action horizon open-loop "
            "before replanning. Consider 1<max_actions_per_chunk<action_horizon for receding-horizon control."
        )

    target_fps_value: int | None = None
    if target_fps is not None:
        target_fps_value = int(target_fps)
    if target_fps_value is not None and target_fps_value > 0:
        train_dt = 1.0 / float(target_fps_value)
        dt_ratio = action_interval / train_dt
        if dt_ratio < 0.9 or dt_ratio > 1.1:
            print(
                "Warning: action_interval does not match checkpoint training cadence. "
                f"action_interval={action_interval:.4f}s vs train_dt={train_dt:.4f}s "
                f"(target_fps={target_fps_value}, ratio={dt_ratio:.3f})."
            )

        horizon_value = int(action_horizon) if action_horizon is not None else None
        if horizon_value is not None and horizon_value > 0:
            replan_steps = (
                horizon_value if max_actions_per_chunk == 0 else max_actions_per_chunk
            )
            train_replan_time = horizon_value * train_dt
            runtime_replan_time = replan_steps * action_interval
            ratio = runtime_replan_time / max(train_replan_time, 1e-8)
            if ratio < 0.9 or ratio > 1.1:
                print(
                    "Warning: effective replanning horizon duration differs from training. "
                    f"runtime={runtime_replan_time:.3f}s vs train={train_replan_time:.3f}s "
                    f"(ratio={ratio:.3f})."
                )

    if bool(config.dry_run):
        print("Dry run enabled: actions will be computed but not sent to the robot.")

    calibration = robot.get_calibration()
    intrinsics = _build_intrinsics_matrices(calibration)
    depth_scales = {
        normalize_camera_name(name): float(cam_cfg["depth_scale"])
        for name, cam_cfg in calibration.items()
    }

    camera_names = [str(name) for name in config.robot.cameras.keys()]
    for name in camera_names:
        if name not in depth_scales:
            raise KeyError(f"Missing depth_scale for camera '{name}'.")
        if name not in intrinsics and f"observation.images.{name}" not in intrinsics:
            raise KeyError(f"Missing intrinsics for camera '{name}'.")
        if name not in extrinsics and f"observation.images.{name}" not in extrinsics:
            raise KeyError(f"Missing extrinsics for camera '{name}'.")

    # --- Control loop -----------------------------------------------------

    _max_context = max(int(visual_context_length), int(proprio_context_length))
    obs_buffer: deque[dict] = deque(maxlen=_max_context)
    action_queue: deque[np.ndarray] = deque()

    robot_features = robot.observation_features
    dataset_features = hw_to_dataset_features(
        robot_features,
        "observation",
        use_video=True,
    )

    # --- Eval recording session (opt-in) ---------------------------------

    recording_cfg = getattr(config, "recording", None)
    record_enabled = bool(getattr(recording_cfg, "request_recording", False))
    record_session_id = ""
    record_client_id = ""
    record_base_metadata: dict = {}
    if record_enabled:
        label = getattr(recording_cfg, "session_label", None)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        record_session_id = f"{timestamp}_{label}" if label else timestamp
        record_client_id = socket.gethostname()
        record_base_metadata = {
            "session_label": label,
            "language_text": config.language_text,
            "action_interval": float(config.action_interval),
            "max_actions_per_chunk": int(config.max_actions_per_chunk),
            "dry_run": bool(config.dry_run),
            "read_proprio_from_follower": bool(config.read_proprio_from_follower),
            "camera_names": list(config.robot.cameras.keys()),
        }
        record_base_metadata = {
            k: v for k, v in record_base_metadata.items() if v is not None
        }
        print(
            f"[recorder] requesting server eval recording session="
            f"{record_session_id} client={record_client_id}"
        )
        await client.start_episode(
            record_session_id,
            record_client_id,
            metadata=record_base_metadata,
        )

    print("Client running. Press Ctrl+C to stop. Press Ctrl+H to send robot home.")
    print("Pre-filling observation buffer...")
    while len(obs_buffer) < _max_context:
        loop_start = time.perf_counter()
        observation = robot.get_observation()
        observation_np = _to_numpy(observation)
        observation_dataset = build_dataset_frame(
            dataset_features,
            observation_np,
            prefix="observation",
        )
        observation_dataset = _to_numpy(observation_dataset)
        obs_buffer.append(observation_dataset)
        elapsed = time.perf_counter() - loop_start
        rest = action_interval - elapsed
        if rest > 0:
            await asyncio.sleep(rest)

    action_count = 0
    infer_samples = 0
    infer_time_ms = 0.0

    # Raw-mode setup only works when stdin is a real TTY (interactive
    # session). Skip it when the client is launched over a pipelined SSH
    # `bash -s` where stdin is a pipe — there's no keyboard to read Ctrl+H
    # from anyway.
    stdin_is_tty = sys.stdin.isatty()
    old_settings = termios.tcgetattr(sys.stdin) if stdin_is_tty else None
    if stdin_is_tty:
        tty.setcbreak(sys.stdin.fileno())

    try:
        action_history: list[np.ndarray] = []

        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == "\x08":
                    print("\r\n[Interrupt] Sending robot home... (Release controls)")
                    robot.send_home()
                    print("\r\n[Interrupt] Robot homed. Clearing buffers.")
                    if record_enabled:
                        await client.end_episode()
                    obs_buffer.clear()
                    action_queue.clear()
                    action_history.clear()
                    _drain_stdin_buffer()
                    print(
                        "\r\n[Interrupt] Set up the scene, then press any key to resume..."
                    )
                    while not select.select([sys.stdin], [], [], 0)[0]:
                        await asyncio.sleep(0.05)
                    sys.stdin.read(1)
                    if record_enabled:
                        await client.start_episode(
                            record_session_id,
                            record_client_id,
                            metadata=record_base_metadata,
                        )
                    print("\r\n[Interrupt] Pre-filling observation buffer...")
                    while len(obs_buffer) < _max_context:
                        observation = robot.get_observation()
                        observation_np = _to_numpy(observation)
                        observation_dataset = build_dataset_frame(
                            dataset_features,
                            observation_np,
                            prefix="observation",
                        )
                        observation_dataset = _to_numpy(observation_dataset)
                        obs_buffer.append(observation_dataset)
                        await asyncio.sleep(action_interval)
                    print("\r\n[Interrupt] Resuming control.")

            loop_start = time.perf_counter()

            observation = robot.get_observation()
            observation_np = _to_numpy(observation)
            observation_dataset = build_dataset_frame(
                dataset_features,
                observation_np,
                prefix="observation",
            )
            observation_dataset = _to_numpy(observation_dataset)

            if not bool(config.read_proprio_from_follower) and len(action_history) > 0:
                observation_dataset["observation.state"] = action_history[-1]
            obs_buffer.append(observation_dataset)

            if not action_queue:
                infer_start = time.perf_counter()
                sample = _build_sample_from_buffer(
                    list(obs_buffer),
                    camera_names,
                    intrinsics,
                    extrinsics,
                    depth_scales,
                    config.language_text,
                    language_text_key,
                    visual_context_length=int(visual_context_length),
                    proprio_context_length=int(proprio_context_length),
                )
                actions_batch, _ = await client.infer(sample, run_inference=True)
                infer_elapsed = (time.perf_counter() - infer_start) * 1e3
                infer_time_ms += infer_elapsed
                infer_samples += 1

                if actions_batch is not None and len(actions_batch) > 0:
                    actions = actions_batch[0]
                    if max_actions_per_chunk > 0:
                        actions = actions[:max_actions_per_chunk]
                    for action_vector in actions:
                        action_queue.append(action_vector)
                else:
                    print("Warning: Inference returned None actions.")

            if action_queue:
                action_vector = action_queue.popleft()
                if not bool(config.dry_run):
                    robot.send_action(action_vector)
                    if not bool(config.read_proprio_from_follower):
                        action_history.append(
                            robot.vector_to_robot_action_vector(action_vector)
                        )
                action_count += 1
            else:
                print("Buffer empty, skipping step...")

            elapsed_loop = time.perf_counter() - loop_start
            rest = action_interval - elapsed_loop
            if rest > 0:
                await asyncio.sleep(rest)

            if (
                int(config.profile_every) > 0
                and infer_samples > 0
                and action_count % int(config.profile_every) == 0
            ):
                avg_infer = infer_time_ms / infer_samples
                print(f"[profile] actions={action_count}, avg_infer={avg_infer:.2f}ms")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if record_enabled:
            try:
                await client.end_episode()
            except Exception as e:
                print(f"[recorder] end_episode on shutdown failed: {e}")
        robot.disconnect()
        if stdin_is_tty:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
