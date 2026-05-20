"""Mirrored imitation_learning client support modules.

Mirror of src/infra/inference/client/* from imitation_learning @ 9a18027d.
Mechanical adaptations: import paths rewritten, pack/unpack provided inline
(from imitation_learning/src/infra/inference/msgpack_numpy.py, same pattern
as nt-runway/dry_run.py), hardware-only exports guarded for X-path compat.
"""

import functools

import msgpack
import numpy as np


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


pack = functools.partial(msgpack.packb, default=_pack_array)
unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


from .config import (
    InferenceSshConfig,
    InferenceCameraConfig,
    TrossenSingleArmConfig,
    TrossenBimanualConfig,
    InferenceRobotConfig,
    InferenceClientConfig,
    build_default_inference_client_config,
)
from .network_client import NetworkClient
from .site_config import SiteConfig, load_site_config

# Hardware-only re-exports: robot.py and runtime.py need lerobot/torch at call time.
# Guarded so `import newt._client.imitation_mirror` succeeds on the X path.
try:
    from .robot import RobotClient, TrossenRobotClient
    from .runtime import run_inference_client, _build_robot_config, normalize_camera_name
    _HARDWARE_EXPORTS_AVAILABLE = True
except ImportError:
    _HARDWARE_EXPORTS_AVAILABLE = False

__all__ = [
    "pack",
    "unpack",
    "InferenceSshConfig",
    "InferenceCameraConfig",
    "TrossenSingleArmConfig",
    "TrossenBimanualConfig",
    "InferenceRobotConfig",
    "InferenceClientConfig",
    "build_default_inference_client_config",
    "NetworkClient",
    "RobotClient",
    "TrossenRobotClient",
    "run_inference_client",
    "_build_robot_config",
    "normalize_camera_name",
    "SiteConfig",
    "load_site_config",
]
