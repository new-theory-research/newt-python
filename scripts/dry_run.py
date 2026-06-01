"""
End-to-end smoke for the WS /stream endpoint — newt.Robot path.

Uses mock read_state / execute callables (no hardware required).
Drives the full obs→action loop: mock state → newt.Robot → WS /stream → chunks applied.

Usage:
    export NT_API_KEY=<key>
    python scripts/dry_run.py wss://newtheory--ntdeva-openpi-serve-serve.modal.run/stream

    # Custom prompt or duration:
    python scripts/dry_run.py <url> --prompt "grasp the block" --max-duration 15

Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import newt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end smoke: mock callables → newt.Robot → WS /stream"
    )
    parser.add_argument("url", help="WebSocket endpoint URL (wss://...modal.run/stream)")
    parser.add_argument("--prompt", default="pick up the cup", help="Language prompt")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Max run seconds")
    args = parser.parse_args()

    api_key = os.environ.get("NT_API_KEY")
    if not api_key:
        print("[smoke] FAIL: NT_API_KEY not set in env.", file=sys.stderr)
        sys.exit(1)

    # Point the library at the requested endpoint via the env override. v0
    # Robot resolves URL from _MODEL_ENDPOINTS[model]; NT_INFERENCE_URL exists
    # as the test/smoke escape hatch.
    os.environ["NT_INFERENCE_URL"] = args.url

    # NT0-FP3 camera keys (matches serve_nt0._CAMERA_KEYS).
    _NT0_CAMERA_KEYS = ["right-wrist-camera", "surrounding1", "surrounding2"]

    # --- mock callables ---
    chunks_applied: list[np.ndarray] = []
    read_state_calls = [0]

    def mock_read_state() -> dict:
        read_state_calls[0] += 1
        if read_state_calls[0] == 1:
            print("[smoke] WS connect /stream OK", flush=True)
            print("[smoke] sent obs frame with depth+matrices (zero-fill / identity)", flush=True)
        # NT0-FP3 observation: include depth + camera matrices so UnprojectPoints runs.
        # Zero-fill depth + identity intrinsics/extrinsics → degraded (not real) point cloud,
        # but the encoder receives a valid (non-None) cloud and produces action chunks.
        return {
            "state": np.zeros(14, dtype=np.float32),
            "depth_maps": {cam: np.zeros((240, 320), dtype=np.float32) for cam in _NT0_CAMERA_KEYS},
            "intrinsics":  {cam: np.eye(3, dtype=np.float32) for cam in _NT0_CAMERA_KEYS},
            "extrinsics":  {cam: np.eye(4, dtype=np.float32) for cam in _NT0_CAMERA_KEYS},
        }

    def mock_execute(chunk: np.ndarray) -> None:
        chunks_applied.append(chunk)

    # --- connect + run ---
    print(f"[smoke] connecting to {args.url} ...", flush=True)
    robot = newt.Robot(
        api_key=api_key,
        read_state=mock_read_state,
        execute=mock_execute,
    )

    try:
        result = robot.run(args.prompt, max_duration=args.max_duration)
    except newt.AuthError as exc:
        print(f"[smoke] FAIL auth: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[smoke] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[smoke] received action chunks: {len(chunks_applied)}", flush=True)
    print(
        f"[smoke] terminal stop_reason={result.stop_reason}"
        " (expected on pi0.5: max_duration)",
        flush=True,
    )
    print(f"[smoke] [mock_execute] applied {len(chunks_applied)} chunks", flush=True)

    # --- assertions ---
    _VALID_STOP_REASONS = {"task_complete", "max_duration", "interrupted", "error"}
    if result.stop_reason not in _VALID_STOP_REASONS:
        print(
            f"[smoke] FAIL: stop_reason={result.stop_reason!r} not in {_VALID_STOP_REASONS}",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(chunks_applied) < 1:
        print("[smoke] FAIL: no action chunks were applied (expected >= 1)", file=sys.stderr)
        sys.exit(1)

    if read_state_calls[0] < 1:
        print("[smoke] FAIL: read_state() was never called", file=sys.stderr)
        sys.exit(1)

    print("[smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
