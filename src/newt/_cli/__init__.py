"""newt CLI entry point.

Installed as the `newt` console script via pyproject.toml [project.scripts].
"""
from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _usage()
        sys.exit(0)

    cmd = args[0]
    if cmd == "login":
        from newt._cli.login import cmd_login
        sys.exit(cmd_login(args[1:]))

    if cmd == "models":
        from newt._cli.models import cmd_models
        sys.exit(cmd_models(args[1:]))

    if cmd == "status":
        from newt._cli.status import cmd_status
        sys.exit(cmd_status(args[1:]))

    print(f"newt: unknown command '{cmd}'", file=sys.stderr)
    print("Run 'newt --help' for usage.", file=sys.stderr)
    sys.exit(1)


def _usage() -> None:
    print("Usage: newt <command> [options]")
    print("")
    print("Commands:")
    print("  login    Authenticate and store credentials in ~/.nt/credentials")
    print("  models   List every model your key can drive")
    print("  status   Show your current key, identity, and registry connectivity")
    print("")
    print("Options:")
    print("  --json   Emit machine-readable JSON (supported by models, status)")
    print("")
    print("Environment:")
    print("  NT_API_KEY        API key override (overrides ~/.nt/credentials)")
    print("  NT_BOOTSTRAP_URL  Override registry discovery base URL")
    print("  NT_INFERENCE_URL  Override inference endpoint directly (skips discovery)")
    print("  NT_CONSOLE_URL    Console URL (default: https://console.newtheory.ai)")
