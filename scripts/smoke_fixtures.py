"""Smoke: both bundled fixtures against live NT0-FP3 (brief-228 STEP 1).

Polls /health until 200 (cold start ~50s), then runs robot.infer(fixtures.load(name))
for each fixture and asserts:
  - the labeled repr shows the 8 NT0-FP3 axis labels
  - the returned chunk is non-degenerate (real cross-dim AND cross-timestep variance,
    not zero-clustered)

cup_stacking now carries the cleaned prompt "Stack one cup into another cup." — this
smoke must confirm it still returns a non-degenerate chunk.

Required env:
    NT_API_KEY        — valid NT API key (nt_...).
    NT_INFERENCE_URL  — MUST be unset (discovery path is needed for axis labels).

Run:
    NT_API_KEY=$(cat /tmp/.nt_key) uv run python scripts/smoke_fixtures.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np

import newt
from newt import fixtures
from newt._client.robot import _DEFAULT_BOOTSTRAP_URL

_EXPECTED_AXES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]


def _poll_health(api_key: str, timeout_s: int = 120) -> None:
    url = _DEFAULT_BOOTSTRAP_URL.rstrip("/") + "/health"
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    print(f"/health 200 after {attempt} attempt(s)")
                    return
        except Exception as exc:
            print(f"  /health not ready (attempt {attempt}): {exc}")
        time.sleep(5)
    raise TimeoutError(f"/health never returned 200 within {timeout_s}s")


def _non_degenerate(chunk: np.ndarray) -> tuple[bool, str]:
    """Real variance both across dims (per timestep) and across timesteps (per dim)."""
    cross_dim = float(np.std(chunk, axis=1).mean())      # variance among the 8 axes
    cross_time = float(np.std(chunk, axis=0).mean())     # variance over the horizon
    abs_max = float(np.abs(chunk).max())
    ok = cross_dim > 1e-3 and cross_time > 1e-3 and abs_max > 1e-2
    detail = (
        f"mean std across dims={cross_dim:.4f}, mean std across timesteps={cross_time:.4f}, "
        f"abs max={abs_max:.4f}"
    )
    return ok, detail


def main() -> int:
    api_key = os.environ.get("NT_API_KEY")
    if not api_key:
        print("PENDING: NT_API_KEY not set.")
        return 2
    if os.environ.get("NT_INFERENCE_URL"):
        print("ERROR: NT_INFERENCE_URL is set — unset it so discovery populates axis labels.")
        return 1

    _poll_health(api_key)

    robot = newt.Robot(
        api_key=api_key,
        read_state=lambda: {},
        execute=lambda chunk: None,
        model="nt0-fp3",
    )

    failures = []
    for name in fixtures.available():
        obs = fixtures.load(name)
        print(f"\n=== {name} | prompt: {obs['prompt']!r} ===")
        resp = robot.infer(obs)
        print("repr:", repr(resp))
        print("shape:", resp.action_chunk.shape, "axes:", resp.axes)

        if resp.axes != _EXPECTED_AXES:
            failures.append(f"{name}: axes {resp.axes} != {_EXPECTED_AXES}")
        ok, detail = _non_degenerate(resp.action_chunk)
        print("non-degenerate:", ok, "|", detail)
        if not ok:
            failures.append(f"{name}: DEGENERATE chunk — {detail}")

    if failures:
        print("\nSMOKE FAIL:")
        for f in failures:
            print("  -", f)
        return 1

    print("\nSMOKE PASS: both fixtures returned labeled, non-degenerate NT0-FP3 chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
