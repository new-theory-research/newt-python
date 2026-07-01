"""newt models — list available models your key can drive."""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from newt._credentials import read_api_key

# ANSI color codes — semantic roles per the CLI aesthetic doc
_RESET = "\033[0m"
_GREEN = "\033[92m"   # pop-green: display name headline
_MINT = "\033[96m"    # dim mint: uids (secondary identity)
_GRAY = "\033[90m"    # warm gray: axes (muted facts)


def _c(code: str, text: str) -> str:
    """Wrap text in an ANSI escape when stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Name derivation
# ---------------------------------------------------------------------------

def _display_name(model: dict[str, Any]) -> str:
    """Primary display name: first tag if any, else uid."""
    tags = [t for t in (model.get("tags") or []) if t]
    if tags:
        return tags[0]
    return model.get("uid") or "—"


def _task_name(fine_tune: dict[str, Any], base_display: str) -> str:
    """Short task label for a fine-tune line.

    Prefer a tag that does NOT start with '{base_display}-' (most readable).
    If all tags share the base prefix, strip it and return the longest result
    (more descriptive > shorter). Falls back to uid.
    """
    tags = [t for t in (fine_tune.get("tags") or []) if t]
    if not tags:
        return fine_tune.get("uid") or "—"

    prefix = base_display + "-"
    non_base_tags = [t for t in tags if not t.startswith(prefix)]
    if non_base_tags:
        # Pick shortest among tags that already stand alone as task names
        return min(non_base_tags, key=len)

    # All tags share the base prefix — strip it, return the longest remainder
    stripped = [t[len(prefix):] for t in tags]
    return max(stripped, key=len)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _group_models(models: list[dict[str, Any]]) -> tuple[
    list[tuple[dict[str, Any], list[dict[str, Any]]]],  # families: (base, [fine_tunes])
    list[dict[str, Any]],                               # orphan fine-tunes
]:
    """Split models into (base, fine_tunes) families plus orphans.

    A fine-tune is an orphan when its base uid is absent from the payload.
    Bases with no fine-tunes produce an empty list.
    Family order follows first appearance of each base in the payload.
    """
    bases: list[dict[str, Any]] = []
    base_seen: set[str] = set()

    for m in models:
        if m.get("type") != "fine_tune":
            uid = m.get("uid") or ""
            if uid not in base_seen:
                bases.append(m)
                base_seen.add(uid)

    fine_tunes_by_base: dict[str, list[dict[str, Any]]] = {b["uid"]: [] for b in bases}
    orphans: list[dict[str, Any]] = []

    for m in models:
        if m.get("type") == "fine_tune":
            base_uid = m.get("base") or ""
            if base_uid in fine_tunes_by_base:
                fine_tunes_by_base[base_uid].append(m)
            else:
                orphans.append(m)

    families = [(b, fine_tunes_by_base[b["uid"]]) for b in bases]
    return families, orphans


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _axes_fragment(model: dict[str, Any]) -> str:
    """Return the axes string fragment, or '' if none."""
    axes = (model.get("contract") or {}).get("action_axes") or []
    if not axes:
        return ""
    return " ".join(str(a) for a in axes)


def _render_base_line(base: dict[str, Any]) -> str:
    """Render the family header line for a base model."""
    name = _display_name(base)
    uid = base.get("uid") or "—"
    axes = _axes_fragment(base)

    parts = [_c(_GREEN, name)]

    if uid != name:
        parts.append(_c(_MINT, uid))

    if axes:
        parts.append(_c(_GRAY, "· axes " + axes))

    return "  ".join(parts)


def _render_fine_tune_line(ft: dict[str, Any], base_display: str, task_col_width: int) -> str:
    """Render a single fine-tune as an indented line under its base."""
    task = _task_name(ft, base_display)
    uid = ft.get("uid") or "—"

    task_padded = task.ljust(task_col_width)
    return "    " + _c(_GREEN, task_padded) + "  " + _c(_MINT, uid)


def _render_orphan_line(ft: dict[str, Any]) -> str:
    """Render an orphan fine-tune whose base isn't in the payload."""
    task = _display_name(ft)
    uid = ft.get("uid") or "—"
    base_ref = ft.get("base") or "unknown"
    return _c(_GREEN, task) + "  " + _c(_MINT, uid) + "  " + _c(_GRAY, f"[base: {base_ref}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _render_models(models: list[dict[str, Any]]) -> str:
    """Render the grouped human-readable model listing. Returns a string."""
    families, orphans = _group_models(models)
    lines: list[str] = []

    # Orphans first (edge case — no base to group under)
    for ft in orphans:
        lines.append(_render_orphan_line(ft))

    if orphans and families:
        lines.append("")

    for i, (base, fine_tunes) in enumerate(families):
        if i > 0:
            lines.append("")

        lines.append(_render_base_line(base))

        if fine_tunes:
            base_display = _display_name(base)
            # Compute task column width for alignment within this family
            task_names = [_task_name(ft, base_display) for ft in fine_tunes]
            task_col_width = max(len(t) for t in task_names) if task_names else 0

            for ft in fine_tunes:
                lines.append(_render_fine_tune_line(ft, base_display, task_col_width))

    return "\n".join(lines)


def _usage() -> None:
    print("Usage: newt models [options]")
    print("")
    print("  List every model your API key can drive.")
    print("")
    print("Options:")
    print("  --json  Emit machine-readable JSON")
    print("")
    print("Environment:")
    print("  NT_API_KEY        API key override (overrides ~/.nt/credentials).")
    print("  NT_BOOTSTRAP_URL  Override registry discovery base URL.")
    print("  NT_INFERENCE_URL  Override inference endpoint directly (skips discovery).")


def cmd_models(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

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

    print(_render_models(models))
    return 0
