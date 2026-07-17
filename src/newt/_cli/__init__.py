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

    if args[0] in ("--version", "-V"):
        _print_version()
        sys.exit(0)

    rc = _dispatch(args)

    # Passive, once-a-day, post-output staleness nudge — only AFTER a command SUCCEEDED,
    # never for `upgrade` itself (it just changed the version). Fully silent on any
    # failure and never able to change `rc` or slow the command (console-014).
    if rc == 0 and args[0] != "upgrade":
        from newt._cli.upgrade import run_update_check
        run_update_check(args)

    sys.exit(rc)


def _dispatch(args: list[str]) -> int:
    cmd = args[0]
    if cmd == "login":
        from newt._cli.login import cmd_login
        return cmd_login(args[1:])

    if cmd == "logout":
        from newt._cli.logout import cmd_logout
        return cmd_logout(args[1:])

    if cmd == "models":
        from newt._cli.models import cmd_models
        return cmd_models(args[1:])

    if cmd == "status":
        from newt._cli.status import cmd_status
        return cmd_status(args[1:])

    if cmd == "run":
        from newt._cli.run import cmd_run
        return cmd_run(args[1:])

    if cmd == "skill":
        from newt._cli.skill import cmd_skill
        return cmd_skill(args[1:])

    if cmd == "record":
        from newt._cli.record import cmd_record
        return cmd_record(args[1:])

    if cmd == "finetune":
        from newt._cli.finetune import cmd_finetune
        return cmd_finetune(args[1:])

    if cmd == "promote":
        from newt._cli.promote import cmd_promote
        return cmd_promote(args[1:])

    if cmd == "episodes":
        from newt._cli.episodes import cmd_episodes
        return cmd_episodes(args[1:])

    if cmd == "upgrade":
        from newt._cli.upgrade import cmd_upgrade
        return cmd_upgrade(args[1:])

    if cmd == "version":
        _print_version()
        return 0

    print(f"newt: unknown command '{cmd}'", file=sys.stderr)
    print("Run 'newt --help' for usage.", file=sys.stderr)
    return 1


def _print_version() -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("newt")
    except PackageNotFoundError:
        from newt import __version__ as v
    print(f"newt {v}")


def _usage() -> None:
    print("Usage: newt <command> [options]")
    print("")
    print("Commands:")
    print("  login    Authenticate and store credentials in ~/.nt/credentials")
    print("  logout   Remove local credentials (key remains valid until revoked on the console)")
    print("  models   List every model your key can drive")
    print("  status   Show your current key, identity, and registry connectivity")
    print("  run      Run one real inference against your model (try: newt run <tag>)")
    print("  skill    Manage built-in skills (try: newt skill install)")
    print("  record   Record NT episodes from an embodiment (needs the [recording] extra)")
    print("  episodes Validate recorded episodes (try: newt episodes validate <dir>)")
    print("  finetune Launch a training run on NT's GPUs and watch it (try: newt finetune --dataset <name>)")
    print("  promote  Keep a fine-tune's checkpoint band and serve it (try: newt promote <job-handle> --band <n>)")
    print("  upgrade  Upgrade the CLI to the latest version (try: newt upgrade)")
    print("  version  Show the installed newt version (also: --version, -V)")
    print("")
    print("Options:")
    print("  --json   Emit machine-readable JSON (supported by logout, models, status,")
    print("           run, record, episodes, finetune, promote)")
    print("  --print  (login only) Print the key to stdout; do not write credentials.")
    print("           Compose with: KEY=$(newt login --print)")
    print("")
    print("Environment:")
    print("  NT_API_KEY        API key override (overrides ~/.nt/credentials)")
    print("  NT_BOOTSTRAP_URL  Override registry discovery base URL")
    print("  NT_INFERENCE_URL  Override inference endpoint directly (skips discovery)")
    print("  NT_CONSOLE_URL    Console URL (default: https://newtheory-console.vercel.app)")
    print("  NEWT_NO_UPDATE_CHECK  Set to 1 to disable the once-a-day 'update available' notice")
