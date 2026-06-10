"""newt skill — install built-in skills into the current project."""
from __future__ import annotations

import importlib.resources
import json
import os
import sys


def cmd_skill(args: list[str]) -> int:
    if not args or args[0] in ("-h", "--help"):
        _usage()
        return 0

    sub = args[0]
    if sub == "install":
        return _cmd_skill_install(args[1:])

    print(f"newt skill: unknown subcommand '{sub}'", file=sys.stderr)
    print("Run 'newt skill --help' for usage.", file=sys.stderr)
    return 1


def _usage() -> None:
    print("Usage: newt skill <subcommand> [options]")
    print("")
    print("Subcommands:")
    print("  install  Copy the newt-onboarding guide into .claude/skills/ in the current directory")
    print("")
    print("Options:")
    print("  --json   Emit machine-readable JSON")


def _cmd_skill_install(args: list[str]) -> int:
    as_json = "--json" in args

    try:
        ref = importlib.resources.files("newt") / "skills" / "newt-onboarding" / "SKILL.md"
        skill_text = ref.read_text(encoding="utf-8")
    except Exception as exc:
        if as_json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"newt: could not read skill data — {exc}", file=sys.stderr)
        return 1

    dest_dir = os.path.join(os.getcwd(), ".claude", "skills", "newt-onboarding")
    dest_file = os.path.join(dest_dir, "SKILL.md")
    overwrite = os.path.exists(dest_file)

    try:
        os.makedirs(dest_dir, exist_ok=True)
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(skill_text)
    except OSError as exc:
        if as_json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"newt: failed to write skill — {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps({"ok": True, "path": dest_file, "overwrite": overwrite}))
        return 0

    if overwrite:
        print(f"Skill updated — {dest_file}")
    else:
        print(f"Skill installed — {dest_file}")
    return 0
