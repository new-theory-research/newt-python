"""newt episodes — inspect and validate recorded NT v0.0.3 episodes.

Frontend only: ``validate`` calls ``newt.recording.validate`` and renders the
verdict. No invariant logic lives here — the checks are the library's.

    newt episodes validate <dir>          human-readable PASS/FAIL per check
    newt episodes validate <dir> --json   structured verdict
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _usage() -> None:
    print("Usage: newt episodes <subcommand> [options]")
    print("")
    print("Subcommands:")
    print("  validate <dir>   Validate an episode_<id> directory against the NT v0.0.3 invariants.")
    print("")
    print("Options:")
    print("  --json           Emit a machine-readable JSON verdict.")
    print("")
    print("  Validation needs the extra:  pip install \"newt[recording]\"")


def cmd_episodes(args: list[str]) -> int:
    if not args or any(a in ("-h", "--help") for a in args) and not (args and args[0] == "validate"):
        # bare `newt episodes` or `-h` at the top level prints usage
        if not args or args[0] in ("-h", "--help"):
            _usage()
            return 0

    sub = args[0]
    if sub != "validate":
        print(f"newt episodes: unknown subcommand {sub!r}", file=sys.stderr)
        print("Run 'newt episodes --help' for usage.", file=sys.stderr)
        return 1

    rest = args[1:]
    if any(a in ("-h", "--help") for a in rest):
        _usage()
        return 0

    as_json = "--json" in rest
    positional = [a for a in rest if not a.startswith("-")]
    if not positional:
        print("newt episodes validate: a directory is required.", file=sys.stderr)
        print("        Fix: newt episodes validate ./episodes/episode_abcd1234", file=sys.stderr)
        return 1

    episode_dir = Path(positional[0])

    try:
        from newt.recording import validate
    except Exception as exc:
        print(f"[newt episodes] {exc}", file=sys.stderr)
        return 1

    try:
        result = validate(episode_dir)
    except Exception as exc:
        # Lantern (missing extra) or a read error — surface it, don't trace.
        print(f"[newt episodes] {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        mark = "PASS" if result["valid"] else "FAIL"
        print(f"[{mark}] {result['episode']}")
        for c in result["checks"]:
            print(f"  {'ok ' if c['ok'] else 'BAD'} {c['check']}: {c['detail']}")

    return 0 if result["valid"] else 1
