"""
Mock robot adapter for nt-runway end-to-end smoke testing.

Satisfies the minimum RobotClient Protocol surface — 3 methods: connect,
get_observation, send_action — without any hardware, lerobot, or torch deps.

WidowX-shaped: get_observation returns the same tuple shape as _fake_widowx_obs()
in dry_run_widowx.py. send_action accepts a (T, 7) ndarray and logs it.

Brief-203 only. Brief-204 inherits this surface and swaps real Interbotix calls in.
"""

from __future__ import annotations

import numpy as np


class MockRobot:
    """Mock WidowX adapter — logs, never moves anything."""

    def __init__(self) -> None:
        self._chunks_applied = 0

    def connect(self) -> None:
        """Initialize the mock robot (no hardware — logs ready signal)."""
        print("[mock_robot] connected (no hardware)", flush=True)

    def get_observation(self) -> tuple:
        """Return fake WidowX observation — zeros, correctly shaped.

        Returns (joints, gripper, ext_image, wrist_image):
            joints:      (6,) float32
            gripper:     float32
            ext_image:   (3, 224, 224) uint8
            wrist_image: (3, 224, 224) uint8
        """
        return (
            np.zeros(6, dtype=np.float32),
            np.float32(0.5),
            np.zeros((3, 224, 224), dtype=np.uint8),
            np.zeros((3, 224, 224), dtype=np.uint8),
        )

    def send_action(self, chunk: np.ndarray) -> None:
        """Accept a (T, 7) action chunk — log it, don't move anything.

        T is the action horizon (50 for pi05_base defaults). The 7 dims are
        the WidowX single-arm layout: [6 joints, 1 gripper].
        """
        self._chunks_applied += 1
        first_row = "  ".join(f"{v:+.4f}" for v in chunk[0])
        print(
            f"[mock_robot] applied chunk: shape={chunk.shape} first row=[{first_row}]",
            flush=True,
        )
