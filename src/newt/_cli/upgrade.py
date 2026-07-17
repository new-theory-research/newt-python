"""newt upgrade — get the latest CLI, plus a quiet once-a-day "you're stale" nudge.

Two pieces (console-014):

  ``newt upgrade``  runs the one documented upgrade command (``uv tool upgrade newt``)
  when it can CONFIRM this ``newt`` was installed as a ``uv tool`` — the documented
  install (README Quickstart: ``uv tool install "git+ssh://…"``). When the install
  method can't be confirmed, it PRINTS the command and does NOT run it, with a one-line
  why — never guess-executing a package manager against the wrong environment (Rule 10).
  ``--print`` always prints and never runs.

  ``run_update_check`` is the passive nudge, called from ``main()`` AFTER a command
  succeeds. It does a single ~1s-capped GET to the console's ``/api/cli/version``, gated
  to at most once a day via ``~/.nt/update-check.json``, and prints ONE stderr line when
  the installed version differs from ``latest``:

      newt <latest> available — run 'newt upgrade'

  Every constraint is hard: it never blocks or slows a command (post-output, ~1s cap,
  once/day), is silent on ANY failure (a dead endpoint costs nothing), is stderr-only,
  is skipped entirely under ``--json``, and is disabled by ``NEWT_NO_UPDATE_CHECK=1``.

Featherweight like the rest of the CLI: stdlib ``urllib`` only, cache under ``~/.nt/``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.request import Request, urlopen

from newt import _credentials

# Same default console base + resolution as finetune.py's _console_url() — the version
# GET is public (no auth), simpler than finetune's Bearer calls.
_DEFAULT_CONSOLE = "https://newtheory-console.vercel.app"

# The documented upgrade command for a `uv tool install` (console-014 Finding 3). Used
# as the offline default; the endpoint's `upgrade` string is preferred when reachable so
# the command lives in one place (single source of truth).
_DEFAULT_UPGRADE_COMMAND = "uv tool upgrade newt"

# Hard cap on the PASSIVE check's GET so a dead/slow endpoint never delays a command
# (Rule 11). The explicit verb (user-initiated) may wait a little longer.
_CHECK_TIMEOUT_S = 1.0
_VERB_TIMEOUT_S = 3.0


def _console_url() -> str:
    return os.environ.get("NT_CONSOLE_URL", _DEFAULT_CONSOLE).rstrip("/")


def _cache_path() -> Path:
    """The once-a-day timestamp cache, beside the credentials under ~/.nt/ (the
    _credentials.py convention). Read live off the module so tests that repoint
    CREDENTIALS_DIR are honored."""
    return _credentials.CREDENTIALS_DIR / "update-check.json"


def _installed_version() -> str | None:
    """This install's own version via importlib.metadata. Returns None when the package
    metadata can't be read (e.g. some editable/dev layouts) — the check then fails silent
    rather than fabricate a version to compare (Rule 10)."""
    try:
        return version("newt")
    except PackageNotFoundError:
        return None


def _fetch_latest(console: str, *, timeout: float) -> dict | None:
    """GET {console}/api/cli/version. Returns {"latest": str, "upgrade": str|None}, or
    None on ANY failure — timeout, dead endpoint, offline, non-200, bad JSON, missing
    `latest`. None means "say nothing"; the check never fabricates a staleness verdict
    it can't source (Rule 10). A dead endpoint costs nothing."""
    try:
        req = Request(f"{console}/api/cli/version", method="GET")
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed console URL
            body = json.loads(resp.read())
    except Exception:  # noqa: BLE001 — silent on every error is the whole point
        return None
    if not isinstance(body, dict):
        return None
    latest = body.get("latest")
    if not isinstance(latest, str) or not latest:
        return None
    upgrade = body.get("upgrade")
    return {
        "latest": latest,
        "upgrade": upgrade if isinstance(upgrade, str) and upgrade else None,
    }


def _load_cache() -> dict:
    try:
        data = json.loads(_cache_path().read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — missing/corrupt cache == "never checked"
        return {}


def _write_cache(today: str, latest: str) -> None:
    """Record today's check under ~/.nt/, mirroring _credentials.py's 0o700 dir
    discipline. A cache we can't write just means we check again tomorrow — never fatal,
    never noisy (the passive check must not surface an error, Rule 10/11)."""
    path = _cache_path()
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text(json.dumps({"last_check": today, "latest": latest}))
    except OSError:
        pass


def run_update_check(args: list[str]) -> None:
    """Post-command, at most once a day: print one stderr line if a newer CLI exists.

    HARD constraints (console-014): never blocks or slows the command — runs after the
    command's output, with a ~1s cap on the GET, gated to once/day via the ~/.nt/ cache.
    Silent on ANY failure. stderr-only. Skipped under --json and NEWT_NO_UPDATE_CHECK=1.
    Never raises — a failure here must never change a command's exit code or leak noise.
    """
    try:
        if os.environ.get("NEWT_NO_UPDATE_CHECK") == "1":
            return
        if "--json" in args:
            # Agents parse stdout; skip the check entirely (the notice is stderr-only
            # anyway, but skipping is the belt-and-suspenders the card asks for).
            return

        installed = _installed_version()
        if not installed:
            return  # can't read our own version → nothing honest to compare against

        today = date.today().isoformat()
        cache = _load_cache()
        if cache.get("last_check") == today:
            return  # already checked today — no fetch, no notice (the daily window)

        info = _fetch_latest(_console_url(), timeout=_CHECK_TIMEOUT_S)
        # Burn the daily budget on EVERY attempt, success or not, so a dead endpoint is
        # hit at most once/day (not once per command until it answers) — Rule 11.
        _write_cache(today, info["latest"] if info else installed)
        if not info:
            return

        if info["latest"] != installed:
            print(f"newt {info['latest']} available — run 'newt upgrade'", file=sys.stderr)
    except Exception:  # noqa: BLE001 — the check must never affect the command
        return


def _is_uv_tool_install() -> bool:
    """Best-effort: is this ``newt`` a ``uv tool install`` (the documented global CLI)?

    A uv-tool install runs from an interpreter under uv's tools directory
    (``…/uv/tools/<name>/…``); we look for the ``uv/tools`` segment in ``sys.prefix``.
    Conservative on purpose: a project ``uv add`` venv or an editable/dev checkout won't
    match, so we return False and the caller PRINTS the command instead of running it —
    never guess-executing a package manager against the wrong environment (Rule 10)."""
    parts = Path(sys.prefix).resolve().parts
    for i in range(len(parts) - 1):
        if parts[i] == "uv" and parts[i + 1] == "tools":
            return True
    return False


def _resolve_upgrade_command() -> str:
    """The upgrade command, preferring the endpoint's `upgrade` string when reachable
    (single source of truth), else the documented local default for offline use."""
    info = _fetch_latest(_console_url(), timeout=_VERB_TIMEOUT_S)
    if info and info.get("upgrade"):
        return info["upgrade"]
    return _DEFAULT_UPGRADE_COMMAND


def _usage() -> None:
    print("Usage: newt upgrade [--print]")
    print("")
    print("  Upgrade the newt CLI to the latest version. When newt was installed as a")
    print("  uv tool (the documented install), this runs `uv tool upgrade newt` for you.")
    print("  When the install method can't be confirmed, it PRINTS the command instead")
    print("  of running it — it never runs a package manager against the wrong install.")
    print("")
    print("Options:")
    print("  --print  Print the upgrade command and exit — never runs it.")
    print("")
    print("Notice: after any command succeeds, newt may print one quiet stderr line")
    print("  when a newer version exists (at most once a day). Disable it entirely with")
    print("  NEWT_NO_UPDATE_CHECK=1.")
    print("")
    print("Environment:")
    print("  NEWT_NO_UPDATE_CHECK  Set to 1 to disable the once-a-day update notice.")
    print("  NT_CONSOLE_URL        Console URL (default: https://newtheory-console.vercel.app)")


def cmd_upgrade(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    command = _resolve_upgrade_command()

    if "--print" in args:
        print(command)
        return 0

    # Only RUN when we can confirm this is a uv-tool install; otherwise print-don't-run
    # with a one-line why (Finding 3, Rule 10).
    if not _is_uv_tool_install():
        print(command)
        print(
            "  (couldn't confirm how `newt` was installed — run the command above "
            "yourself)",
            file=sys.stderr,
        )
        return 0

    print(f"$ {command}")
    try:
        proc = subprocess.run(command.split())  # noqa: S603 — fixed, documented command
    except FileNotFoundError:
        print(
            "newt upgrade: `uv` was not found on your PATH — install uv, or run the "
            "command above yourself.",
            file=sys.stderr,
        )
        return 1
    return proc.returncode
