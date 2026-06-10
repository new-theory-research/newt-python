"""newt models — list available models your key can drive."""
from __future__ import annotations

import json
import os
import sys

from newt._credentials import read_api_key

# ANSI color codes — semantic roles per the CLI aesthetic doc
_RESET = "\033[0m"
_GREEN = "\033[92m"   # pop-green: where you are / UID headline
_MINT = "\033[96m"    # dim mint: static facts (tags)
_GRAY = "\033[90m"    # warm gray: neutral / secondary info


def _c(code: str, text: str) -> str:
    """Wrap text in an ANSI escape when stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


def cmd_models(args: list[str]) -> int:
    as_json = "--json" in args

    api_key = os.environ.get("NT_API_KEY") or read_api_key()
    if not api_key:
        print(
            "newt: no API key found.\n"
            "  Run `newt login` to authenticate, or set NT_API_KEY.",
            file=sys.stderr,
        )
        return 1

    import newt

    try:
        models = newt.list_models(api_key)
    except newt.AuthError as exc:
        print(f"newt: authentication failed — {exc}", file=sys.stderr)
        print("  Rotate your key at the NT console, or run `newt login` again.", file=sys.stderr)
        return 1
    except newt.RegistryUnavailable as exc:
        print(f"newt: registry unreachable — {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"newt: unexpected error — {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(models))
        return 0

    if not models:
        print("No models available for your key.")
        return 0

    for model in models:
        uid = model.get("uid") or "—"
        model_type = model.get("type") or ""
        base = model.get("base") or ""
        tags = [t for t in (model.get("tags") or []) if t]
        axes = model.get("axes") or []

        # UID is the headline — pop-green
        print(_c(_GREEN, uid), end="")

        parts: list[str] = []
        if model_type:
            parts.append(model_type)
        if base and base != uid:
            parts.append(f"base {base}")
        if axes:
            parts.append(f"axes [{', '.join(str(a) for a in axes)}]")

        if parts:
            print("  " + _c(_GRAY, " • ".join(parts)), end="")

        if tags:
            print("  " + _c(_MINT, "  ".join(tags)), end="")

        print()

    return 0
