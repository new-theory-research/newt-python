"""
Run a mock inference loop against the New Theory NT0-FP3 model.

This script opens a WebSocket stream to NT0-FP3, sends mock observations
(zero 8-dim state vector + blank RGB camera frames for three cameras), and
prints the action chunks that come back. No robot hardware is required —
the server's firehose coercion handles structurally-valid zero inputs. Expect
a few "received chunk: shape=(50, 8) ..." lines per inference cycle, then a
"stop_reason: max_duration" and "chunks_received: N" summary. Requires
NT_API_KEY in env: export NT_API_KEY=nt_...
"""

from __future__ import annotations

import os
import sys

import numpy as np

import newt


_CAMERA_KEYS = ["right-wrist-camera", "surrounding1", "surrounding2"]


def main() -> None:
    api_key = os.environ.get("NT_API_KEY")
    if not api_key:
        print("Error: NT_API_KEY is not set. Export your key and re-run.", file=sys.stderr)
        sys.exit(1)

    chunks_received: list[np.ndarray] = []

    def read_state() -> dict:
        return {
            "state": np.zeros(8, dtype=np.float32),
            "images": {cam: np.zeros((3, 240, 320), dtype=np.uint8) for cam in _CAMERA_KEYS},
        }

    def execute(chunk: np.ndarray) -> None:
        chunks_received.append(chunk)
        print(f"received chunk: shape={chunk.shape}  first_pose={chunk[0]}", flush=True)

    robot = newt.Robot(
        api_key=api_key,
        read_state=read_state,
        execute=execute,
        model="nt0-fp3",
    )

    result = robot.run("pick up the cup", max_duration=10.0)
    print(f"stop_reason: {result.stop_reason}")
    print(f"chunks_received: {len(chunks_received)}")


if __name__ == "__main__":
    main()
