"""Tests for `newt skill install` — skill self-equip mechanism.

Each test encodes WHY the behavior matters, not just what it does.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

from newt._cli.skill import cmd_skill, _cmd_skill_install


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_install(args: list[str], monkeypatch, tmp_path):
    """Run cmd_skill ['install', ...] capturing stdout/stderr, cwd set to tmp_path."""
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_out)
    monkeypatch.setattr(sys, "stderr", captured_err)
    monkeypatch.chdir(tmp_path)
    exit_code = cmd_skill(["install"] + args)
    return exit_code, captured_out.getvalue(), captured_err.getvalue()


# ---------------------------------------------------------------------------
# Golden: an SDK-only Claude equips itself
# ---------------------------------------------------------------------------

def test_install_creates_skill_file(monkeypatch, tmp_path):
    """An SDK-only Claude runs `newt skill install` and gets the guide.

    The SKILL.md must exist in .claude/skills/newt-onboarding/ under cwd.
    This is the primary outcome: any directory becomes equipped with one command.
    """
    exit_code, out, err = _run_install([], monkeypatch, tmp_path)

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    skill_file = tmp_path / ".claude" / "skills" / "newt-onboarding" / "SKILL.md"
    assert skill_file.exists(), "SKILL.md must exist in .claude/skills/newt-onboarding/"
    assert err == "", f"no stderr expected on success: {err!r}"


def test_install_skill_has_valid_frontmatter(monkeypatch, tmp_path):
    """The installed skill has valid Claude skill frontmatter (name: + description:).

    Claude Code reads the frontmatter to register the skill. A skill with missing
    or malformed frontmatter silently fails to load — no error, no guide.
    """
    _run_install([], monkeypatch, tmp_path)
    skill_file = tmp_path / ".claude" / "skills" / "newt-onboarding" / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8")

    assert text.startswith("---"), "skill must begin with YAML frontmatter block"
    assert "name:" in text, "frontmatter must contain 'name:'"
    assert "description:" in text, "frontmatter must contain 'description:'"


def test_install_skill_content_matches_canonical(monkeypatch, tmp_path):
    """Installed content matches the canonical package data exactly.

    If content diverges (e.g., truncated write or encoding mismatch), the agent
    gets a partial or corrupted guide. Exact match guards against silent corruption.
    """
    import importlib.resources
    canonical = (
        importlib.resources.files("newt") / "skills" / "newt-onboarding" / "SKILL.md"
    ).read_text(encoding="utf-8")

    _run_install([], monkeypatch, tmp_path)
    skill_file = tmp_path / ".claude" / "skills" / "newt-onboarding" / "SKILL.md"
    installed = skill_file.read_text(encoding="utf-8")

    assert installed == canonical, "installed content must match canonical package data byte-for-byte"


def test_install_idempotent_overwrites_without_error(monkeypatch, tmp_path):
    """Re-running `newt skill install` exits 0 and overwrites — no error, no abort.

    An agent that re-runs the command during environment setup must not get stuck
    or see a failure because the file already exists.
    """
    # First install
    exit_code1, _, _ = _run_install([], monkeypatch, tmp_path)
    assert exit_code1 == 0

    # Second install — must still succeed
    exit_code2, out2, err2 = _run_install([], monkeypatch, tmp_path)
    assert exit_code2 == 0, f"re-run must exit 0; stderr={err2!r}"
    assert err2 == "", f"no stderr on re-run: {err2!r}"


def test_install_overwrite_emits_notice(monkeypatch, tmp_path):
    """Re-run prints an 'updated' notice, not the same first-install message.

    The overwrite notice tells the agent the file was already present and has
    been refreshed — distinguishable from a first install.
    """
    # First install
    _run_install([], monkeypatch, tmp_path)
    # Second install
    _, out2, _ = _run_install([], monkeypatch, tmp_path)

    assert "updated" in out2.lower() or "overwrite" in out2.lower() or "overwrit" in out2.lower(), (
        f"re-run must emit an overwrite/update notice, got: {out2!r}"
    )


def test_install_json_emits_structured_result(monkeypatch, tmp_path):
    """`--json` emits a machine-readable result; no human-readable text on stdout.

    Agents and scripts that parse stdout rely on this contract. Any non-JSON
    text breaks the parse.
    """
    exit_code, out, err = _run_install(["--json"], monkeypatch, tmp_path)

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    parsed = json.loads(out)
    assert parsed["ok"] is True, f"json must have ok=true: {parsed!r}"
    assert "path" in parsed, f"json must include 'path': {parsed!r}"


def test_install_json_contains_no_announcement(monkeypatch, tmp_path):
    """`--json` stdout contains no human announcement copy.

    Announcement copy in `--json` output breaks scripted consumers. The brief
    explicitly requires announcements never appear in `--json`.
    """
    _, out, _ = _run_install(["--json"], monkeypatch, tmp_path)

    parsed = json.loads(out)  # must parse cleanly — no trailing text
    # The parsed output is the entire stdout; if it has announcement text the
    # parse would have already failed. Explicitly confirm no prose.
    assert "Claude" not in out or out.strip().startswith("{"), (
        f"no Claude-Code announcement in --json stdout: {out!r}"
    )


def test_install_json_overwrite_flag_set_on_second_run(monkeypatch, tmp_path):
    """`--json` reports overwrite=true on re-run, false on first install.

    Scripts that want to know if a file was fresh vs updated can read this flag.
    """
    _, out1, _ = _run_install(["--json"], monkeypatch, tmp_path)
    assert json.loads(out1)["overwrite"] is False

    _, out2, _ = _run_install(["--json"], monkeypatch, tmp_path)
    assert json.loads(out2)["overwrite"] is True


def test_install_path_is_under_cwd(monkeypatch, tmp_path):
    """`newt skill install` writes only inside cwd — never a home-dir or global path.

    Writing to a fixed path outside cwd would equip the wrong project or
    overwrite another project's skill unexpectedly.
    """
    _, out, _ = _run_install(["--json"], monkeypatch, tmp_path)

    parsed = json.loads(out)
    written_path = parsed["path"]
    # Path must start with tmp_path (cwd)
    assert written_path.startswith(str(tmp_path)), (
        f"written path must be under cwd ({tmp_path}), got: {written_path!r}"
    )


# ---------------------------------------------------------------------------
# Error + edge cases
# ---------------------------------------------------------------------------

def test_skill_unknown_subcommand_exits_nonzero(monkeypatch, tmp_path):
    """Unknown subcommand exits non-zero with a helpful error.

    An agent that typos the subcommand must see an error, not silent success.
    """
    captured_err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured_err)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.chdir(tmp_path)

    exit_code = cmd_skill(["bogus"])
    assert exit_code != 0
    assert "bogus" in captured_err.getvalue() or "unknown" in captured_err.getvalue().lower()


def test_skill_help_exits_zero(monkeypatch, tmp_path):
    """`newt skill --help` exits 0 and prints usage."""
    captured_out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.chdir(tmp_path)

    exit_code = cmd_skill(["--help"])
    assert exit_code == 0
    assert "install" in captured_out.getvalue()


def test_skill_no_args_exits_zero_with_usage(monkeypatch, tmp_path):
    """`newt skill` (no subcommand) exits 0 and prints usage — not an error."""
    captured_out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.chdir(tmp_path)

    exit_code = cmd_skill([])
    assert exit_code == 0
    assert "install" in captured_out.getvalue()
