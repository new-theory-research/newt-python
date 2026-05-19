"""Load robot site configuration from ``~/.config/nt/nt.toml``.

The site config provides per-machine robot, camera, and workspace
settings so that inference client profiles don't need to hardcode
serial numbers, IPs, or extrinsics.

Typical TOML layout::

    [workspace]
    width  = 0.840
    height = 0.718
    robot_offset = 0.0375

    [[robot_config.arms]]
    id = "left-arm"
    leader_ip = "192.168.1.3"
    follower_ip = "192.168.1.5"

    [[camera_config.cameras]]
    id = "surrounding1"
    serial_number = "324422301880"
    frame = "world"
    extrinsics = [
      [ 1.0, 0.0, 0.0, 0.0 ],
      [ 0.0, 1.0, 0.0, 0.0 ],
      [ 0.0, 0.0, 1.0, 0.0 ],
      [ 0.0, 0.0, 0.0, 1.0 ],
    ]

Camera ``frame`` can be ``"world"`` (static extrinsics) or
``{ arm = "<arm-id>" }`` (wrist-mounted, identity fallback).
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SITE_CONFIG_PATH = Path("~/.config/nt/nt.toml")


@dataclass(frozen=True)
class SiteArmConfig:
    """A single robot arm from the site config."""

    id: str
    leader_ip: str | None = None
    follower_ip: str | None = None


@dataclass(frozen=True)
class SiteCameraConfig:
    """A single camera from the site config."""

    id: str
    serial_number: str
    frame: str | dict
    extrinsics: np.ndarray  # (4, 4)

    @property
    def is_wrist_camera(self) -> bool:
        """Camera is wrist-mounted when frame references an arm."""
        return isinstance(self.frame, dict) and "arm" in self.frame


@dataclass(frozen=True)
class SiteWorkspaceConfig:
    """Workspace physical dimensions."""

    width: float | None = None
    height: float | None = None
    robot_offset: float | None = None


@dataclass(frozen=True)
class SiteConfig:
    """Parsed site configuration from ``nt.toml``."""

    workspace: SiteWorkspaceConfig = field(default_factory=SiteWorkspaceConfig)
    arms: list[SiteArmConfig] = field(default_factory=list)
    cameras: list[SiteCameraConfig] = field(default_factory=list)

    def camera_by_id(self, camera_id: str) -> SiteCameraConfig | None:
        """Look up a camera by its id."""
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        return None

    def arm_by_id(self, arm_id: str) -> SiteArmConfig | None:
        """Look up an arm by its id."""
        for arm in self.arms:
            if arm.id == arm_id:
                return arm
        return None

    def extrinsics_map(self) -> dict[str, np.ndarray]:
        """Build camera-name → 4x4 extrinsics mapping for all cameras."""
        return {cam.id: cam.extrinsics for cam in self.cameras}


def _parse_extrinsics(raw: list[list[float]]) -> np.ndarray:
    """Parse a 4x4 extrinsics matrix from nested TOML arrays."""
    arr = np.array(raw, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError(f"Extrinsics must be 4x4, got shape {arr.shape}.")
    return arr


def load_site_config(
    path: Path | str | None = None,
) -> SiteConfig | None:
    """Load site configuration from a TOML file.

    Args:
        path: Path to the TOML file. Defaults to ``~/.config/nt/nt.toml``.
              If the file doesn't exist, returns ``None`` (not an error).

    Returns:
        Parsed ``SiteConfig`` or ``None`` when the file is absent.
    """
    resolved = Path(path or DEFAULT_SITE_CONFIG_PATH).expanduser().resolve()
    if not resolved.exists():
        logger.debug("Site config not found at %s; skipping.", resolved)
        return None

    logger.info("Loading site config from %s", resolved)
    with open(resolved, "rb") as f:
        raw = tomllib.load(f)

    # Parse workspace
    ws_raw = raw.get("workspace", {})
    workspace = SiteWorkspaceConfig(
        width=ws_raw.get("width"),
        height=ws_raw.get("height"),
        robot_offset=ws_raw.get("robot_offset"),
    )

    # Parse arms
    arms_raw = raw.get("robot_config", {}).get("arms", [])
    arms = [
        SiteArmConfig(
            id=a["id"],
            leader_ip=a.get("leader_ip"),
            follower_ip=a.get("follower_ip"),
        )
        for a in arms_raw
    ]

    # Parse cameras
    cameras_raw = raw.get("camera_config", {}).get("cameras", [])
    cameras = []
    for c in cameras_raw:
        if "id" not in c:
            raise ValueError("Each camera entry must have an 'id' field.")
        if "serial_number" not in c:
            raise ValueError(f"Camera '{c['id']}' missing 'serial_number'.")
        if "extrinsics" not in c:
            raise ValueError(f"Camera '{c['id']}' missing 'extrinsics'.")

        cameras.append(
            SiteCameraConfig(
                id=c["id"],
                serial_number=str(c["serial_number"]),
                frame=c.get("frame", "world"),
                extrinsics=_parse_extrinsics(c["extrinsics"]),
            )
        )

    return SiteConfig(workspace=workspace, arms=arms, cameras=cameras)


__all__ = [
    "SiteConfig",
    "SiteArmConfig",
    "SiteCameraConfig",
    "SiteWorkspaceConfig",
    "load_site_config",
    "DEFAULT_SITE_CONFIG_PATH",
]
