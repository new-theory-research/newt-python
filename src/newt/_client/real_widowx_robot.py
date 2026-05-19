"""Real WidowX-AI robot adapter — single right-arm + RealSense cameras.

Implements the same RobotClient Protocol surface as MockRobot
(`nt._client.mock_robot.MockRobot`), backed by lerobot's WidowXAIFollower
for the arm and lerobot's RealSense camera adapter for vision. Reads site
config from ``~/.config/nt/nt.toml``.

T1: this file is OURS — Andrii's ``run_robot_client.py`` is the canonical
reference, not extended. We re-derive the toml-reading + arm/camera open
patterns here rather than importing from imitation_mirror.

T3/T4: lerobot, trossen_arm, and pyrealsense2 imports are runtime-only
(inside methods) — keeps the nt-runway package importable in client-only
envs that don't have those installed.

Known issue (handoff 2026-05-14, motor-6 fault):
    ``cartesian_action`` on this rig faults motor 6 (likely wrist_rotate /
    gripper drive) when the IK-driven target is rejected. ``send_action``
    catches ``trossen_arm.RuntimeError``, logs the fault, and continues so
    the wire stays alive for the v0 demo path. NOT a silent fallback —
    every fault is logged with the offending row + motor index when the
    library reports one.

Action mapping (provisional — flagged for the first smoke):
    The model returns a (T, 14) bimanual chunk. brief-203's existing slice
    (`slice_aloha_to_widowx`) takes `actions[:, :7]` (left-arm slot, where
    we padded our single-arm obs in). For a right-arm-only rig this is a
    convention call — the model has been seeing our state in the left-arm
    slot, so we read actions back from that same slot.

    Per the 2026-05-14 handoff (cartesian-action finding):
        action[0] → effector.x.pos
        action[1] → effector.y.pos
        action[2] → effector.z.pos
        action[3] → effector.alpha.pos
        action[4] → effector.beta.pos
        action[5] → effector.gamma.pos
        action[6] → gripper.pos

    Still provisional — could end up being joint-space (waist/shoulder/...).
    First closed-loop smoke against the rig will confirm or invalidate.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_SITE_CONFIG_PATH = Path("~/.config/nt/nt.toml")

# Target image shape expected by pad_widowx_to_aloha (channel-first, 224x224, uint8).
_IMAGE_H = 224
_IMAGE_W = 224


# ---------------------------------------------------------------------------
# Site config (minimal, local parse — no import from imitation_mirror)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SiteArm:
    id: str
    follower_ip: str | None


@dataclass(frozen=True)
class _SiteCamera:
    id: str
    serial_number: str
    is_wrist: bool


@dataclass(frozen=True)
class _Site:
    arms: list[_SiteArm]
    cameras: list[_SiteCamera]


def _load_site(path: Path) -> _Site:
    """Parse arms + cameras from ``nt.toml``. Stdlib tomllib only."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Site config not found at '{resolved}'. "
            "Define arm IPs + camera serials in ~/.config/nt/nt.toml."
        )

    with open(resolved, "rb") as f:
        raw = tomllib.load(f)

    arms_raw = raw.get("robot_config", {}).get("arms", [])
    arms = [
        _SiteArm(id=a["id"], follower_ip=a.get("follower_ip"))
        for a in arms_raw
    ]

    cameras_raw = raw.get("camera_config", {}).get("cameras", [])
    cameras = []
    for c in cameras_raw:
        if "id" not in c or "serial_number" not in c:
            raise ValueError(
                "Each camera entry must have 'id' and 'serial_number' "
                f"(got: {c})."
            )
        frame = c.get("frame", "world")
        is_wrist = isinstance(frame, dict) and "arm" in frame
        cameras.append(
            _SiteCamera(
                id=c["id"],
                serial_number=str(c["serial_number"]),
                is_wrist=is_wrist,
            )
        )

    return _Site(arms=arms, cameras=cameras)


# ---------------------------------------------------------------------------
# Image helpers (HxWxC uint8 from RealSense → 3x224x224 uint8 channel-first)
# ---------------------------------------------------------------------------


def _to_channel_first_224(img: np.ndarray) -> np.ndarray:
    """Coerce an arbitrary RealSense frame to (3, 224, 224) uint8.

    Accepts:
        - (H, W, 3) uint8 BGR/RGB — channel-last; we don't swap RGB↔BGR
          because the model was trained on lerobot's default capture which
          matches what we hand back here.
        - (3, H, W) uint8 — already channel-first.

    Resizing uses simple stride-based subsample/repeat — keeps the file
    dep-free (no cv2/Pillow). If a sharper resize is needed later, swap
    in a real bilinear via lerobot's image utils.
    """
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)

    if img.ndim == 3 and img.shape[-1] == 3:
        # channel-last → channel-first
        img = np.transpose(img, (2, 0, 1))
    elif img.ndim == 3 and img.shape[0] == 3:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    _, h, w = img.shape
    if (h, w) == (_IMAGE_H, _IMAGE_W):
        return img

    # Nearest-neighbour resize (cheap; no extra deps). v0 demo only —
    # quality-sensitive callers should swap in lerobot's bilinear.
    row_idx = np.linspace(0, h - 1, _IMAGE_H).astype(np.int64)
    col_idx = np.linspace(0, w - 1, _IMAGE_W).astype(np.int64)
    return img[:, row_idx][:, :, col_idx]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class RealWidowXRobot:
    """Real WidowX-AI adapter — single right-arm + RealSense cameras.

    Mirrors MockRobot's 3-method Protocol surface (``connect``,
    ``get_observation``, ``send_action``) plus a ``disconnect`` cleanup
    hook. Lerobot + trossen_arm + pyrealsense2 are imported lazily inside
    methods so this module is importable in client-only envs.
    """

    def __init__(
        self,
        site_config_path: str | None = None,
        arm_name: str | None = None,
    ) -> None:
        """Resolve the arm + cameras from the site config.

        Args:
            site_config_path: Path to ``nt.toml``. Defaults to
                ``~/.config/nt/nt.toml``.
            arm_name: Explicit arm id to use. Defaults to the first arm
                whose id contains "right" (this rig is right-arm-only
                even though the toml may declare a bimanual layout).
        """
        path = Path(site_config_path) if site_config_path else _DEFAULT_SITE_CONFIG_PATH
        self._site = _load_site(path)

        if not self._site.arms:
            raise ValueError(
                "Site config has no arms in [[robot_config.arms]]."
            )

        if arm_name is not None:
            selected = next((a for a in self._site.arms if a.id == arm_name), None)
            if selected is None:
                available = [a.id for a in self._site.arms]
                raise ValueError(
                    f"Arm '{arm_name}' not found in site config. "
                    f"Available: {available}"
                )
        else:
            selected = next(
                (a for a in self._site.arms if "right" in a.id.lower()),
                None,
            )
            if selected is None:
                available = [a.id for a in self._site.arms]
                raise ValueError(
                    "No arm with 'right' in its id found in site config. "
                    f"Available: {available}. Pass arm_name= to override."
                )

        if not selected.follower_ip:
            raise ValueError(
                f"Site arm '{selected.id}' is missing follower_ip."
            )

        self._arm_spec = selected

        if not self._site.cameras:
            raise ValueError(
                "Site config has no cameras in [[camera_config.cameras]]."
            )

        # Built lazily on connect — keeps __init__ free of lerobot import.
        self._arm = None
        self._cameras: dict[str, object] = {}

    def connect(self) -> None:
        """Connect to the arm + open all RealSense cameras."""
        # Lazy imports — module top-level must stay lerobot-free (T4).
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import (
            RealSenseCameraConfig,
        )
        from lerobot.robots.widowx_ai.widowx_ai_follower import WidowXAIFollower
        from lerobot.robots.widowx_ai.config_widowx_ai import (
            WidowXAIFollowerConfig,
        )

        print(
            f"[real_widowx] connecting arm '{self._arm_spec.id}' at "
            f"{self._arm_spec.follower_ip} ...",
            flush=True,
        )
        self._arm = WidowXAIFollower(
            WidowXAIFollowerConfig(
                ip=self._arm_spec.follower_ip,
                id=self._arm_spec.id,
            )
        )
        self._arm.connect()

        for cam in self._site.cameras:
            print(
                f"[real_widowx] opening camera '{cam.id}' "
                f"(serial={cam.serial_number}, wrist={cam.is_wrist}) ...",
                flush=True,
            )
            try:
                handle = RealSenseCamera(
                    RealSenseCameraConfig(serial_number_or_name=cam.serial_number)
                )
                handle.connect()
                self._cameras[cam.id] = handle
            except Exception as exc:
                # Camera not physically connected (or in use). Skip — the wire
                # stays alive; _grab_image returns zeros for missing cameras.
                # Honest log so the demo doesn't claim signal it doesn't have.
                print(
                    f"[real_widowx] camera '{cam.id}' not available "
                    f"({type(exc).__name__}: {exc}); using zero placeholder",
                    flush=True,
                )

        real = sorted(self._cameras.keys())
        missing = [c.id for c in self._site.cameras if c.id not in self._cameras]
        print(
            f"[real_widowx] connected — real cameras: {real or '(none)'}; "
            f"placeholder: {missing or '(none)'}",
            flush=True,
        )

    def _grab_image(self, prefer_wrist: bool) -> np.ndarray:
        """Grab one frame from the first camera matching the wrist/world axis.

        Falls back to any available camera if no exact match is found —
        better to send a real frame for the off-slot than to zero-fill and
        silently degrade signal.
        """
        if not self._cameras:
            return np.zeros((3, _IMAGE_H, _IMAGE_W), dtype=np.uint8)

        target = None
        for cam in self._site.cameras:
            if cam.is_wrist == prefer_wrist and cam.id in self._cameras:
                target = cam
                break
        if target is None:
            # No wrist/world match in connected cameras — fall back to ANY
            # connected camera so observation carries real signal rather than
            # zeros. Per-camera connect status already logged loudly above.
            for cam in self._site.cameras:
                if cam.id in self._cameras:
                    target = cam
                    break
        if target is None:
            return np.zeros((3, _IMAGE_H, _IMAGE_W), dtype=np.uint8)

        frame = self._cameras[target.id].async_read()
        return _to_channel_first_224(frame)

    def get_observation(self) -> tuple:
        """Return a WidowX-shaped observation tuple.

        Returns (joints, gripper, ext_image, wrist_image):
            joints:      (6,) float32 — 6 arm joint positions
            gripper:     float32       — gripper position
            ext_image:   (3, 224, 224) uint8 — external/overhead camera
            wrist_image: (3, 224, 224) uint8 — wrist-mounted camera

        Joint keys read from the arm's observation dict, in WidowX order:
        waist, shoulder, elbow, wrist_tilt, wrist_yaw, wrist_rotate.
        Per the 2026-05-14 connect-read, these `.pos` keys exist alongside
        the cartesian effector keys; we read joint-space here because
        pad_widowx_to_aloha expects the WidowX joint convention.
        """
        if self._arm is None:
            raise RuntimeError("RealWidowXRobot.connect() not called.")

        obs = self._arm.get_observation()

        joint_keys = (
            "waist.pos",
            "shoulder.pos",
            "elbow.pos",
            "wrist_tilt.pos",
            "wrist_yaw.pos",
            "wrist_rotate.pos",
        )
        joints = np.array(
            [float(obs[k]) for k in joint_keys],
            dtype=np.float32,
        )
        gripper = np.float32(float(obs["gripper.pos"]))

        ext_image = self._grab_image(prefer_wrist=False)
        wrist_image = self._grab_image(prefer_wrist=True)

        return joints, gripper, ext_image, wrist_image

    def send_action(self, chunk: np.ndarray) -> None:
        """Apply a (T, 7) action chunk row-by-row as cartesian commands.

        Each row maps to:
            row[0..5] → effector.{x, y, z, alpha, beta, gamma}.pos
            row[6]    → gripper.pos

        Known issue: cartesian_action faults motor 6 (likely wrist_rotate /
        gripper drive) on this rig under some IK-rejected targets. We
        catch ``trossen_arm.RuntimeError``, log it, and continue — the
        wire stays alive through the fault.

        NOT a silent fallback: every fault is logged with the row index
        and the exception message before continuing.
        """
        if self._arm is None:
            raise RuntimeError("RealWidowXRobot.connect() not called.")

        # Lazy import so the trossen_arm dep stays out of module top-level.
        import trossen_arm

        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise ValueError(
                f"Expected (T, 7) action chunk, got shape {chunk.shape}."
            )

        first_row = "  ".join(f"{v:+.4f}" for v in chunk[0])
        print(
            f"[real_widowx] applying chunk: shape={chunk.shape} "
            f"first row=[{first_row}]",
            flush=True,
        )

        # Joint-mode action: bypass lerobot's cartesian-only send_action +
        # call the underlying trossen_arm driver's set_all_positions
        # directly. pi05_base + trossen norm-stats output values per row:
        # [waist, shoulder, elbow, wrist_tilt, wrist_yaw, wrist_rotate, gripper].
        #
        # Gripper clipping: model outputs gripper in roughly normalized
        # [0, 1], but joint 6 (gripper) hardware range is [-0.004, 0.044]
        # meters (~44mm physical opening). Clip the gripper target into
        # the hardware range. Arm joints 0-5 pass through unchanged —
        # tonight's smoke confirmed they accept the model's rad outputs.
        driver = self._arm.driver  # trossen_arm.TrossenArmDriver
        _GRIPPER_MIN, _GRIPPER_MAX = -0.004, 0.044
        for t, row in enumerate(chunk):
            try:
                arm_joints = [float(v) for v in row[:6]]
                gripper = float(row[6])
                gripper_clipped = max(_GRIPPER_MIN, min(_GRIPPER_MAX, gripper))
                driver.set_all_positions(arm_joints + [gripper_clipped])
            except (trossen_arm.RuntimeError, trossen_arm.LogicError, TypeError, AttributeError) as exc:
                # Wire stays alive; every fault is logged loudly. Fault
                # classes observed/expected on this rig:
                # - RuntimeError: motor-side CAN failure
                # - LogicError: out-of-range joint position rejection
                # - TypeError: API signature mismatch (e.g. 7 vs 6 vals)
                print(
                    f"[real_widowx] set_all_positions faulted at row {t} "
                    f"({type(exc).__name__}, continuing): {exc}",
                    flush=True,
                )
                if isinstance(exc, (TypeError, AttributeError)):
                    # Once we see this, no point looping through 49 more
                    # of the same error.
                    print(
                        "[real_widowx] TypeError suggests API mismatch — "
                        "aborting chunk early.",
                        flush=True,
                    )
                    break

    def disconnect(self) -> None:
        """Release arm torque + close all cameras."""
        if self._arm is not None:
            try:
                self._arm.disconnect()
            except Exception as exc:
                print(f"[real_widowx] arm.disconnect failed: {exc}", flush=True)
            self._arm = None

        for cam_id, handle in self._cameras.items():
            try:
                handle.disconnect()
            except Exception as exc:
                print(
                    f"[real_widowx] camera '{cam_id}' disconnect failed: {exc}",
                    flush=True,
                )
        self._cameras.clear()
