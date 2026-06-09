"""Live cold-start smoke: registry resolution must be instant even when Modal is cold.

Golden test #1: "my first call resolves instantly, even when the model is cold."
The registry bootstrap now defaults to the always-on Railway registry, so
Robot() construction (one GET /v1/models + model→endpoint resolution) should
complete in well under a second regardless of the NT0-FP3 Modal app's state.
This script proves it against a verified-cold Modal app:

  1. Reports the NT0-FP3 Modal app's running-task count (`modal app list`) so
     the operator can confirm the app is actually cold. If the modal CLI isn't
     installed, prints the command to run elsewhere and proceeds.
  2. Constructs newt.Robot(api_key=...) on the DEFAULT bootstrap path — no env
     overrides allowed.
  3. Times the registry-resolution portion (all of Robot() construction: the
     registry fetch + endpoint resolution are the only network work in it).
  4. PASS iff resolution took < 1s AND real labeled action axes resolved from
     the registry payload. FAIL otherwise, with timing and error.

No robot rig required — Robot() without read_state/execute never touches hardware.

Required env:
    NT_API_KEY        — valid NT API key (nt_...).
    NT_BOOTSTRAP_URL  — MUST be unset (this smoke tests the default).
    NT_INFERENCE_URL  — MUST be unset (would skip discovery entirely).

Run:
    NT_API_KEY=$(cat /tmp/.nt_key) python scripts/smoke_cold_e2e.py

Exit codes: 0 PASS, 1 FAIL, 2 misconfigured environment.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

_MODAL_APP_NAME = "ntdeva-nt0-fp3-serve"
_BUDGET_S = 1.0


def _check_env() -> str:
    api_key = os.environ.get("NT_API_KEY")
    if not api_key:
        print("ERROR: NT_API_KEY is not set. Run: NT_API_KEY=nt_... python scripts/smoke_cold_e2e.py")
        sys.exit(2)
    blockers = [v for v in ("NT_BOOTSTRAP_URL", "NT_INFERENCE_URL") if os.environ.get(v)]
    if blockers:
        print(
            f"ERROR: {', '.join(blockers)} set — this smoke tests the DEFAULT bootstrap "
            "path and env overrides would mask it. Unset and re-run."
        )
        sys.exit(2)
    return api_key


def _report_modal_state() -> None:
    """Best-effort visibility into whether the Modal app is cold. Never fails the smoke."""
    print(f"--- Modal app state ({_MODAL_APP_NAME}) ---")
    if shutil.which("modal") is None:
        print("modal CLI not found. To verify the app is cold, run elsewhere:")
        print("    modal app list")
        print("Proceeding anyway — the PASS condition does not depend on this report.")
        return
    try:
        # --json because the table view truncates app names ("ntdeva-nt0…").
        out = subprocess.run(
            ["modal", "app", "list", "--json"], capture_output=True, text=True, timeout=60
        )
        import json

        rows = [
            app
            for app in json.loads(out.stdout)
            if _MODAL_APP_NAME in (app.get("Description") or "")
        ]
        if rows:
            for app in rows:
                print(
                    f"{app.get('Description')}: state={app.get('State')} "
                    f"tasks={app.get('Tasks')}"
                )
            print("(0 tasks = cold)")
        else:
            print(f"`modal app list --json` returned no row for {_MODAL_APP_NAME}.")
    except Exception as exc:
        print(f"`modal app list --json` failed ({exc}); proceeding without the cold report.")


def main() -> int:
    api_key = _check_env()
    _report_modal_state()

    import newt
    from newt._client.robot import _resolve_action_axes, _resolve_bootstrap_url

    print("--- Registry resolution (default bootstrap path) ---")
    print(f"bootstrap URL: {_resolve_bootstrap_url()}")

    start = time.perf_counter()
    try:
        robot = newt.Robot(api_key=api_key)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"FAIL: Robot() raised after {elapsed:.3f}s: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.perf_counter() - start

    axes = _resolve_action_axes(robot._registry, robot._model)
    print(f"registry resolution: {elapsed:.3f}s (budget {_BUDGET_S:.1f}s)")
    print(f"resolved endpoint:   {robot._url}")
    print(f"action axes:         {axes}")

    if elapsed >= _BUDGET_S:
        print(f"FAIL: registry resolution took {elapsed:.3f}s (>= {_BUDGET_S:.1f}s budget)")
        return 1
    if not axes:
        print("FAIL: no labeled action axes in the registry payload (dim_N fallback would apply)")
        return 1
    print(f"PASS: registry resolved in {elapsed:.3f}s with {len(axes)} labeled axes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
