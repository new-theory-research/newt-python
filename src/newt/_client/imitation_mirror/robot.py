from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

# Hardware-only imports: torch, lerobot, and geometry helpers that need torch.
# Wrapped so the module can be imported without T4 deps on the X path (no hardware).
# Functions that call into lerobot will raise RuntimeError at call time when unavailable.
try:
    import torch
    import lerobot
    from lerobot.robots import RobotConfig
    from nt._client.imitation_mirror.utils.geometry import axis_angle_to_matrix, matrix_to_axis_angle
    from nt._client.imitation_mirror.utils.trossen_bimanual_frames import left_from_right_base
    _LEROBOT_AVAILABLE = True
except ImportError:
    _LEROBOT_AVAILABLE = False


# Keep left-arm cartesian key order explicit for frame remapping.
_LEFT_CARTESIAN_KEYS = (
    "left_effector.x.pos",
    "left_effector.y.pos",
    "left_effector.z.pos",
    "left_effector.alpha.pos",
    "left_effector.beta.pos",
    "left_effector.gamma.pos",
)


@runtime_checkable
class RobotClient(Protocol):
    """Minimal robot client interface for inference."""

    action_features: Sequence[str]
    is_connected: bool

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_observation(self) -> dict[str, Any]: ...

    def set_action_contract(self, *, arm_count: int) -> None: ...

    def send_action(self, action: dict[str, float] | Sequence[float]) -> None: ...

    def send_home(self) -> None: ...

    def teleop_safety_stop(self) -> None: ...

    def get_calibration(self) -> dict[str, Any]:
        """Return camera calibration data (intrinsics, depth scale)."""
        ...


class TrossenRobotClient(RobotClient):
    """Minimal Trossen robot client interface for inference. Stores config internally.

    Wanted to hard-code this for now for simplicity.
    """

    def __init__(self, robot_config: RobotConfig):
        if not _LEROBOT_AVAILABLE:
            raise RuntimeError(
                "lerobot is not installed. TrossenRobotClient requires lerobot at runtime."
            )
        super().__init__()

        # Robot configuration.
        self.robot_config = robot_config

        # Backing robot instance.
        self._robot = lerobot.robots.make_robot_from_config(self.robot_config)
        # Capture action feature names in deterministic order.
        raw_action_features = getattr(self._robot, "action_features", [])
        if isinstance(raw_action_features, dict):
            self.action_features = tuple(raw_action_features.keys())
        else:
            self.action_features = tuple(raw_action_features)

        # Capture observation features for dataset-frame conversion.
        self.observation_features = getattr(self._robot, "observation_features", [])

        # Default to single-arm until the server config is read.
        self._arm_count = 1

        # Cache right->left base transform for bimanual robot egress mapping.
        self._left_from_right_base: torch.Tensor | None = None

    # Delegate to the underlying robot instance.
    @property
    def is_connected(self) -> bool:
        return bool(getattr(self._robot, "is_connected", False))

    def connect(self) -> None:
        self._robot.connect()

    def disconnect(self) -> None:
        self._robot.disconnect()

    def get_observation(self) -> dict[str, Any]:
        return self._robot.get_observation()

    def set_action_contract(self, *, arm_count: int) -> None:
        """Set action contract metadata for this client.

        Args:
            arm_count: Number of arm blocks in policy action vectors.
        """
        # Fail fast on invalid arm counts.
        if int(arm_count) <= 0:
            raise ValueError(f"arm_count must be > 0, got {arm_count}.")

        # Persist arm count for diagnostics.
        self._arm_count = int(arm_count)

        # Resolve bimanual right->left base transform when needed.
        if self._arm_count == 2:
            self._left_from_right_base = torch.tensor(
                left_from_right_base(),
                dtype=torch.float64,
            )  # (4, 4)
        else:
            self._left_from_right_base = None

    def _remap_bimanual_action_to_robot_frame(
        self, action_dict: dict[str, float]
    ) -> dict[str, float]:
        """Map left-arm action pose from policy frame back to robot-native frame.

        Args:
            action_dict: Robot action dict built from policy output.

        Returns:
            dict[str, float]: Action dict with left-arm pose remapped to left base.
        """
        # Require a configured right->left frame transform in bimanual mode.
        if self._left_from_right_base is None:
            raise ValueError(
                "Bimanual action remap requires configured left_from_right_base transform."
            )

        # Enforce required left-arm cartesian keys for remapping.
        missing_keys = [key for key in _LEFT_CARTESIAN_KEYS if key not in action_dict]
        if missing_keys:
            raise KeyError(f"Bimanual action remap missing keys: {missing_keys}.")

        # Read left-arm position channels currently expressed in right-base frame.
        pos_right = torch.tensor(
            [
                action_dict["left_effector.x.pos"],
                action_dict["left_effector.y.pos"],
                action_dict["left_effector.z.pos"],
            ],
            dtype=torch.float64,
        ).view(1, 3)  # (1, 3)

        # Read left-arm axis-angle orientation in right-base frame.
        aa_right = torch.tensor(
            [
                action_dict["left_effector.alpha.pos"],
                action_dict["left_effector.beta.pos"],
                action_dict["left_effector.gamma.pos"],
            ],
            dtype=torch.float64,
        ).view(1, 3)  # (1, 3)

        # Decode axis-angle orientation to rotation matrix form.
        rot_right = axis_angle_to_matrix(aa_right)  # (1, 3, 3)

        # Split right->left rigid transform into rotation and translation.
        transform = self._left_from_right_base  # (4, 4)
        rot = transform[:3, :3]  # (3, 3)
        trans = transform[:3, 3]  # (3,)

        # Map left-arm position from right-base frame into left-base frame.
        pos_left = torch.matmul(pos_right, rot.transpose(0, 1)) + trans  # (1, 3)

        # Map left-arm orientation from right-base frame into left-base frame.
        rot_left = torch.matmul(rot, rot_right)  # (1, 3, 3)
        aa_left = matrix_to_axis_angle(rot_left)  # (1, 3)

        # Copy and update only remapped left-arm cartesian channels.
        remapped = dict(action_dict)
        remapped["left_effector.x.pos"] = float(pos_left[0, 0].item())
        remapped["left_effector.y.pos"] = float(pos_left[0, 1].item())
        remapped["left_effector.z.pos"] = float(pos_left[0, 2].item())
        remapped["left_effector.alpha.pos"] = float(aa_left[0, 0].item())
        remapped["left_effector.beta.pos"] = float(aa_left[0, 1].item())
        remapped["left_effector.gamma.pos"] = float(aa_left[0, 2].item())
        return remapped

    def _vector_to_action_dict(self, action: Sequence[float]) -> dict[str, float]:
        """Convert a flat action vector into robot action dict.

        Args:
            action: Flat model output vector with shape ``(D_action,)``.

        Returns:
            dict[str, float]: Robot-native action dictionary.
        """
        # Convert incoming vector into a flat float64 numpy array.
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)  # (D_action,)

        # Resolve expected action width from robot action feature names.
        expected_dim = len(self.action_features)
        actual_dim = int(action_array.shape[0])

        # Fail hard when policy output width does not match robot action features.
        if actual_dim != expected_dim:
            raise ValueError(
                "Action vector width mismatch. "
                f"Got {actual_dim}, expected {expected_dim} from robot.action_features. "
                f"Configured arm_count={self._arm_count}."
            )

        # Build a deterministic key/value action dictionary from ordered features.
        return {
            feature_name: float(action_array[idx])
            for idx, feature_name in enumerate(self.action_features)
        }

    def _action_dict_to_vector(self, action_dict: dict[str, float]) -> np.ndarray:
        """Convert a robot action dictionary into ordered vector form.

        Args:
            action_dict: Action payload keyed by ``self.action_features``.

        Returns:
            np.ndarray: Flat action vector in robot feature order.
        """
        # Validate that all required action keys are present.
        missing_keys = [key for key in self.action_features if key not in action_dict]
        if missing_keys:
            raise KeyError(f"Action dictionary missing keys: {missing_keys}.")

        # Pack values in the exact robot action feature order.
        return np.asarray(
            [float(action_dict[key]) for key in self.action_features],
            dtype=np.float32,
        )  # (D_action,)

    def vector_to_robot_action_dict(self, action: Sequence[float]) -> dict[str, float]:
        """Convert one policy action vector into robot-native action dictionary.

        Args:
            action: Flat policy output vector with shape ``(D_action,)``.

        Returns:
            dict[str, float]: Robot-native action dictionary.
        """
        # Convert vector payload into key/value action dictionary.
        action_dict = self._vector_to_action_dict(action)

        # Remap bimanual left-arm pose from policy frame to robot frame.
        if self._arm_count == 2:
            action_dict = self._remap_bimanual_action_to_robot_frame(action_dict)

        return action_dict

    def vector_to_robot_action_vector(self, action: Sequence[float]) -> np.ndarray:
        """Convert one policy action vector into robot-frame vector channels.

        This helper is used by the client-side proprio override path so that
        fed-back state remains in the same frame as real follower observations.

        Args:
            action: Flat policy output vector with shape ``(D_action,)``.

        Returns:
            np.ndarray: Robot-frame vector in ``self.action_features`` order.
        """
        action_dict = self.vector_to_robot_action_dict(action)
        return self._action_dict_to_vector(action_dict)

    def send_action(self, action: dict[str, float] | Sequence[float]) -> None:
        """Send an action to the robot.

        Args:
            action: Robot-native action dict or flat model output vector.
        """
        # Pass through robot-native dict actions directly.
        if isinstance(action, dict):
            self._robot.send_action(action)
            return

        # Convert model output vector to robot-native action dictionary.
        action_dict = self.vector_to_robot_action_dict(action)

        # Send converted action through the backing robot API.
        self._robot.send_action(action_dict)

    def send_home(self) -> None:
        self._robot.send_home()

    def teleop_safety_stop(self) -> None:
        if hasattr(self._robot, "teleop_safety_stop"):
            self._robot.teleop_safety_stop()

    def get_calibration(self) -> dict[str, Any]:
        """Fetch intrinsics, depth scale, and shape from RealSense cameras."""
        calibration = {}
        for name, camera in self._robot.cameras.items():
            if not hasattr(camera, "intrinsics"):
                # Skip if not available/connected
                raise ValueError(f"Camera {name} does not have intrinsics attribute.")

            # RealSenseCamera.intrinsics is ALREADY the dictionary we want (fx, fy, depth_scale).
            # We just want to add shape to it.
            # Convert/Copy to dict to be safe and mutable
            calib_data = dict(camera.intrinsics)

            # Extract shape if available in config
            if hasattr(camera, "config"):
                h = camera.config.height
                w = camera.config.width
                # Assume 3 channels for RGB cameras
                calib_data["shape"] = (h, w, 3)

            calibration[name] = calib_data

        return calibration

    def __getattr__(self, name: str):
        # Fallback to the underlying robot for any other attributes.
        return getattr(self._robot, name)
