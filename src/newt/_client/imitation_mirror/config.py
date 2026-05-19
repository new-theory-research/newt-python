from __future__ import annotations

from ml_collections import config_dict

from nt._client.imitation_mirror.config_sections import ConfigSection


class InferenceSshConfig(ConfigSection):
    """Configuration for optional SSH tunneling in robot client runtime.

    Attributes:
        enabled: Whether to open an SSH tunnel before connecting websocket.
        host: SSH host used by sshtunnel.
        user: SSH username used by sshtunnel.
        identity_file: Optional private key path for SSH authentication.
        forward_viewer: Whether to forward the remote Viser port.
        viewer_remote_port: Remote Viser port to forward when enabled.
    """

    enabled: bool = True
    host: str | None = None
    user: str = "ubuntu"
    identity_file: str = "~/.ssh/nt_amsterdam.pem"
    forward_viewer: bool = True
    viewer_remote_port: int = 8079


class InferenceCameraConfig(ConfigSection):
    """Configuration for one RealSense camera in robot runtime.

    Attributes:
        serial_number_or_name: RealSense serial number or OS camera name.
        fps: Camera framerate.
        width: Camera image width.
        height: Camera image height.
        use_depth: Whether to stream depth.
        use_depth_postprocessing: Whether to enable depth post-processing.
        is_wrist_camera: Whether missing extrinsics should default to identity.
    """

    serial_number_or_name: str
    fps: int = 30
    width: int = 640
    height: int = 480
    use_depth: bool = True
    use_depth_postprocessing: bool = True
    is_wrist_camera: bool = False


class TrossenSingleArmConfig(ConfigSection):
    """Configuration for single-arm Trossen follower setup.

    Attributes:
        arm_name: Name of the arm in nt.toml to use (e.g. ``"right-arm"``).
        ip: Robot controller IP for single-arm setup.
        goal_time_multiplier: Goal-time scaling multiplier.
    """

    arm_name: str = "right-arm"
    ip: str | None = None
    goal_time_multiplier: float = 20.0


class TrossenBimanualConfig(ConfigSection):
    """Configuration for bimanual Trossen follower setup.

    Attributes:
        left_arm_ip: Left-arm robot controller IP.
        right_arm_ip: Right-arm robot controller IP.
        left_arm_goal_time_multiplier: Left-arm goal-time scaling multiplier.
        right_arm_goal_time_multiplier: Right-arm goal-time scaling multiplier.
    """

    left_arm_ip: str | None = None
    right_arm_ip: str | None = None
    left_arm_goal_time_multiplier: float = 20.0
    right_arm_goal_time_multiplier: float = 20.0


class InferenceRobotConfig(ConfigSection):
    """Robot runtime configuration used by the inference client.

    Attributes:
        type: Robot backend type. Currently supports only ``"trossen"``.
        cameras: Camera-name mapping to camera runtime configs.
        arm: Arm-specific config, set by the arm profile after server
            arm-count is known. Either ``TrossenSingleArmConfig`` or
            ``TrossenBimanualConfig``.
    """

    type: str = "trossen"
    cameras: config_dict.ConfigDict
    arm: TrossenSingleArmConfig | TrossenBimanualConfig | None = None


class RecordingClientConfig(ConfigSection):
    """Client-side opt-in for server eval recording.

    Attributes:
        request_recording: When ``True``, the client emits ``episode_start`` /
            ``episode_end`` control messages at the start/end of each rollout
            so the server records a ``.rrd`` and uploads it to GCS. When
            ``False`` (default), no control messages are sent and the client
            behaves exactly as before.
        session_label: Optional human tag mixed into the server-side session
            id (``{timestamp}_{label}``). Useful to group related rollouts.
    """

    request_recording: bool = False
    session_label: str | None = None


class InferenceClientConfig(ConfigSection):
    """Configuration for robot-side inference client runtime.

    Attributes:
        site_config_path: Path to site TOML. Defaults to ``~/.config/nt/nt.toml``.
            Set to empty string to skip site config loading.
        camera_mapping: Optional explicit mapping from server camera names to local
            camera names, e.g. ``{"surrounding1": "cam_left", "surrounding3": "cam_right"}``.
            When set, this mapping is used directly instead of auto-matching by name.
            When ``None`` (default), cameras are matched by name and a ``ValueError``
            is raised if any server camera has no local match.
        server_url: WebSocket endpoint for inference server.
        action_interval: Target control-loop period in seconds.
        max_actions_per_chunk: Maximum actions executed per inference chunk (0 = full horizon open-loop).
        profile_every: Print latency profile every N actions (0 disables).
        read_proprio_from_follower: Use follower proprio state directly.
        dry_run: Compute and queue actions without sending to robot.
        language_text: Optional text instruction sent as metadata.
        ssh: Optional SSH tunnel configuration.
        robot: Robot backend runtime configuration.
        recording: Opt-in server-side eval recording settings.
    """

    site_config_path: str | None = None
    camera_mapping: dict[str, str] | None = None
    server_url: str
    action_interval: float
    max_actions_per_chunk: int
    profile_every: int
    read_proprio_from_follower: bool
    dry_run: bool
    language_text: str | None
    ssh: InferenceSshConfig
    robot: InferenceRobotConfig
    recording: RecordingClientConfig


def build_default_inference_client_config() -> InferenceClientConfig:
    """Build default inference client config.

    Returns:
        InferenceClientConfig: Client config with repository defaults.
    """

    # Construct nested sections explicitly to avoid shared mutable defaults.
    return InferenceClientConfig(
        ssh=InferenceSshConfig(),
        robot=InferenceRobotConfig(
            cameras=config_dict.ConfigDict({}),
        ),
        recording=RecordingClientConfig(),
    )


__all__ = [
    "InferenceSshConfig",
    "InferenceCameraConfig",
    "TrossenSingleArmConfig",
    "TrossenBimanualConfig",
    "InferenceRobotConfig",
    "RecordingClientConfig",
    "InferenceClientConfig",
    "build_default_inference_client_config",
]
