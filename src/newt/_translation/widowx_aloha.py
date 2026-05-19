"""WidowX ↔ pi05_aloha bimanual translation.

Pure functions — no IO, no side effects, no hardware deps.
Sits at the seam between single-arm WidowX and the bimanual model server.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


def pad_widowx_to_aloha(
    joints: ndarray,
    gripper: ndarray,
    ext_image: ndarray,
    wrist_image: ndarray,
    prompt: str,
) -> dict:
    """Pad single-arm WidowX observation to pi05_aloha bimanual schema.

    WidowX occupies the left-arm slot. Right arm and missing cameras are zero-filled.

    State layout: [L-joints(6), L-grip(1), R-joints(6), R-grip(1)] = 14 dims.
    Cameras: cam_high=ext_image, cam_low=zeros, cam_left_wrist=wrist_image, cam_right_wrist=zeros.

    Args:
        joints: (6,) float32 — WidowX joint angles
        gripper: scalar or (1,) float32 — WidowX gripper position
        ext_image: (3, H, W) uint8 — external/overhead camera → cam_high
        wrist_image: (3, H, W) uint8 — wrist camera → cam_left_wrist
        prompt: natural-language task description

    Returns:
        dict with 'state' (14,), 'images' (4 cameras), 'prompt' — ready for /infer.
    """
    joints = np.asarray(joints, dtype=np.float32).reshape(6)
    gripper_scalar = float(np.asarray(gripper, dtype=np.float32).flat[0])

    state = np.zeros(14, dtype=np.float32)
    state[0:6] = joints
    state[6] = gripper_scalar
    # state[7:14] stay zero — right arm + right gripper

    ext_image = np.asarray(ext_image, dtype=np.uint8)
    wrist_image = np.asarray(wrist_image, dtype=np.uint8)

    return {
        "state": state,
        "images": {
            "cam_high": ext_image,
            "cam_low": np.zeros_like(ext_image),
            "cam_left_wrist": wrist_image,
            "cam_right_wrist": np.zeros_like(ext_image),
        },
        "prompt": prompt,
    }


def slice_aloha_to_widowx(actions: ndarray) -> ndarray:
    """Slice bimanual action chunk down to single-arm WidowX.

    Args:
        actions: (T, 14) — bimanual action chunk from /infer

    Returns:
        (T, 7) — left arm joints (dims 0:6) + gripper (dim 6)

    Raises:
        ValueError: if actions shape is not (T, 14)
    """
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected (T, 14) actions, got shape {actions.shape}")
    return actions[:, :7]
