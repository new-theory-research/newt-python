"""Smoke: Robot.infer(obs) against the live NT0-FP3 endpoint (brief-228).

Proves the one-shot infer() call path end-to-end and that InferenceResponse.__repr__
prints the NT0-FP3 semantic axis labels (x, y, z, qw, qx, qy, qz, gripper) rather than
float indices.

Uses a ZERO-FILLED observation: the server firehose-coerces missing fields, so this
returns a real (if degraded) action chunk — enough to validate the request/response
contract + the labeled repr without recorded episode data.

Runs via DISCOVERY, not the NT_INFERENCE_URL override:
  - With discovery, Robot fetches /v1/models at construction and InferenceResponse picks
    up the real action_axes from the registry contract -> the 8 labels appear.
  - With NT_INFERENCE_URL set, discovery is skipped, the registry is empty, and infer()
    falls back to dim_0..dim_7. That defeats the point of this smoke, so it asserts the
    override is NOT set.

Required env:
    NT_API_KEY — valid NT API key (nt_...).

Run:
    NT_API_KEY=nt_... uv run python scripts/smoke_infer.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

import newt

_EXPECTED_AXES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]


def main() -> int:
    api_key = os.environ.get("NT_API_KEY")
    if not api_key:
        print("PENDING: NT_API_KEY not set. Set it and re-run to smoke against live NT0-FP3.")
        return 2

    if os.environ.get("NT_INFERENCE_URL"):
        print(
            "ERROR: NT_INFERENCE_URL is set — discovery is skipped and the registry is "
            "empty, so axis labels fall back to dim_N. Unset NT_INFERENCE_URL so this "
            "smoke exercises the real label path via /v1/models discovery."
        )
        return 1

    # read_state/execute are unused by infer() but required by the Robot constructor.
    robot = newt.Robot(
        api_key=api_key,
        read_state=lambda: {},
        execute=lambda chunk: None,
        model="nt0-fp3",
    )

    # Zero-filled observation — server zero-fills the rest. Prompt rides on the kwarg.
    obs = {"state": np.zeros(8, dtype=np.float32)}
    resp = robot.infer(obs, prompt="clean_table")

    print("repr:", repr(resp))
    print("model:", resp.model)
    print("latency_ms:", f"{resp.latency_ms:.0f}")
    print("action_chunk shape:", resp.action_chunk.shape)
    print("axes:", resp.axes)

    assert isinstance(resp.action_chunk, np.ndarray), "action_chunk must be an ndarray"
    assert resp.action_chunk.shape[-1] == 8, (
        f"NT0-FP3 chunk last dim must be 8, got {resp.action_chunk.shape}"
    )
    assert resp.axes == _EXPECTED_AXES, (
        f"expected NT0-FP3 axis labels {_EXPECTED_AXES}, got {resp.axes} — "
        "registry contract.action_axes did not reach the SDK"
    )
    assert resp.latency_ms > 0, "latency_ms must be positive"

    print("\nSMOKE PASS: infer() returned a labeled NT0-FP3 chunk via the shared run() wire.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
