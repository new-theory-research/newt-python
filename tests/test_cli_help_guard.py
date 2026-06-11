"""Tests for the uniform -h/--help guard across every newt CLI verb.

Each verb must:
  - print verb-specific usage (description + flags) and exit 0
  - perform NO action when -h or --help is present in its args
  - global help (bare newt / newt -h / newt --help) is byte-identical to baseline

AC1 — newt login --help: prints usage with login description and --print flag,
      exits 0, zero network calls, zero credential writes.
AC2 — every other verb (logout, models, status, skill install) with -h/--help:
      prints usage, exits 0, no action performed.
AC3 — global help: bare newt / newt -h / newt --help unchanged.
"""
from __future__ import annotations

import io
import sys
from unittest.mock import patch

import pytest

from newt._cli.login import cmd_login
from newt._cli.logout import cmd_logout
from newt._cli.models import cmd_models
from newt._cli.status import cmd_status
from newt._cli.skill import cmd_skill
from newt._cli import main, _usage as global_usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture(fn, args, monkeypatch):
    """Run a cmd_* function with args, capturing stdout/stderr. Returns (rc, out, err)."""
    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    rc = fn(args)
    return rc, out.getvalue(), err.getvalue()


def _capture_main(argv, monkeypatch):
    """Run main() with sys.argv = ['newt'] + argv, capturing stdout/stderr and SystemExit."""
    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "argv", ["newt"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        main()
    return exc_info.value.code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# AC1 — newt login --help / -h: usage, exit 0, no network, no credential writes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_login_help_exits_zero(flag, monkeypatch):
    """login with -h/--help exits 0 without running the login flow."""
    rc, out, err = _capture(cmd_login, [flag], monkeypatch)
    assert rc == 0, f"expected exit 0; rc={rc}, stderr={err!r}"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_login_help_mentions_print_flag(flag, monkeypatch):
    """login --help must document --print so agents know about the composable mode."""
    rc, out, err = _capture(cmd_login, [flag], monkeypatch)
    assert "--print" in out, f"--print must appear in login help: {out!r}"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_login_help_no_network_calls(flag, monkeypatch):
    """login --help must make zero network calls — the login flow must not start."""
    urlopen_calls = []
    monkeypatch.setattr("newt._cli.login.urlopen", lambda *a, **kw: urlopen_calls.append(a) or None)

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    rc = cmd_login([flag])

    assert rc == 0
    assert urlopen_calls == [], (
        f"urlopen must not be called with {flag}; got {len(urlopen_calls)} call(s)"
    )


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_login_help_no_credential_writes(flag, monkeypatch):
    """login --help must never call write_api_key — no file side effects."""
    write_calls = []
    monkeypatch.setattr("newt._cli.login.write_api_key", lambda key: write_calls.append(key))

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    rc = cmd_login([flag])

    assert rc == 0
    assert write_calls == [], (
        f"write_api_key must not be called with {flag}; called with: {write_calls}"
    )


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_login_help_describes_login(flag, monkeypatch):
    """login --help output must name the login flow (authentication / pairing)."""
    rc, out, err = _capture(cmd_login, [flag], monkeypatch)
    lowered = out.lower()
    assert "login" in lowered or "authenticat" in lowered or "pairing" in lowered, (
        f"login help must describe the login action: {out!r}"
    )


# ---------------------------------------------------------------------------
# AC2 — logout with -h/--help: usage, exit 0, credentials file untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_logout_help_exits_zero(flag, monkeypatch):
    """logout with -h/--help exits 0 without touching credentials."""
    rc, out, err = _capture(cmd_logout, [flag], monkeypatch)
    assert rc == 0, f"expected exit 0; stderr={err!r}"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_logout_help_leaves_credentials_untouched(flag, monkeypatch, tmp_path):
    """logout --help must not delete credentials, even if the file exists."""
    import stat
    import newt._cli.logout as logout_mod

    cred_dir = tmp_path / ".nt"
    cred_file = cred_dir / "credentials"
    cred_dir.mkdir(mode=0o700, parents=True)
    cred_file.write_text("api_key = nt_testkey12345\n")
    cred_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

    monkeypatch.setattr(logout_mod, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(logout_mod, "CREDENTIALS_PATH", cred_file)

    rc, out, err = _capture(cmd_logout, [flag], monkeypatch)

    assert rc == 0
    assert cred_file.exists(), f"credentials must not be deleted by {flag}"
    content = cred_file.read_text()
    assert "nt_testkey12345" in content, f"credentials must be byte-identical after {flag}"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_logout_help_describes_action(flag, monkeypatch):
    """logout --help must describe what the command does."""
    rc, out, err = _capture(cmd_logout, [flag], monkeypatch)
    lowered = out.lower()
    assert "logout" in lowered or "credential" in lowered or "remove" in lowered, (
        f"logout help must describe the action: {out!r}"
    )


# ---------------------------------------------------------------------------
# AC2 — models with -h/--help: usage, exit 0, no API call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_models_help_exits_zero(flag, monkeypatch):
    """models with -h/--help exits 0 without making an API call."""
    rc, out, err = _capture(cmd_models, [flag], monkeypatch)
    assert rc == 0, f"expected exit 0; stderr={err!r}"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_models_help_no_api_call(flag, monkeypatch):
    """models --help must not call newt.list_models."""
    list_calls = []
    monkeypatch.setattr("newt.list_models", lambda *a, **kw: list_calls.append(a) or [])

    rc, out, err = _capture(cmd_models, [flag], monkeypatch)

    assert rc == 0
    assert list_calls == [], (
        f"newt.list_models must not be called with {flag}; got {len(list_calls)} call(s)"
    )


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_models_help_describes_action(flag, monkeypatch):
    """models --help must describe what the command does."""
    rc, out, err = _capture(cmd_models, [flag], monkeypatch)
    lowered = out.lower()
    assert "model" in lowered or "list" in lowered, (
        f"models help must describe the action: {out!r}"
    )


# ---------------------------------------------------------------------------
# AC2 — status with -h/--help: usage, exit 0, no API call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_status_help_exits_zero(flag, monkeypatch):
    """status with -h/--help exits 0 without making an API call."""
    rc, out, err = _capture(cmd_status, [flag], monkeypatch)
    assert rc == 0, f"expected exit 0; stderr={err!r}"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_status_help_no_api_call(flag, monkeypatch):
    """status --help must not call newt.list_models (the registry probe)."""
    list_calls = []
    monkeypatch.setattr("newt.list_models", lambda *a, **kw: list_calls.append(a) or [])

    rc, out, err = _capture(cmd_status, [flag], monkeypatch)

    assert rc == 0
    assert list_calls == [], (
        f"newt.list_models must not be called with {flag}; got {len(list_calls)} call(s)"
    )


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_status_help_describes_action(flag, monkeypatch):
    """status --help must describe what the command does."""
    rc, out, err = _capture(cmd_status, [flag], monkeypatch)
    lowered = out.lower()
    assert "status" in lowered or "key" in lowered or "registry" in lowered, (
        f"status help must describe the action: {out!r}"
    )


# ---------------------------------------------------------------------------
# AC2 — skill with -h/--help (already had it; regression guard + install sub-help)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_skill_help_exits_zero(flag, monkeypatch):
    """skill -h/--help exits 0 (already implemented — regression guard)."""
    rc, out, err = _capture(cmd_skill, [flag], monkeypatch)
    assert rc == 0, f"expected exit 0; stderr={err!r}"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_skill_help_no_install(flag, monkeypatch, tmp_path):
    """skill -h/--help must not write any files to disk."""
    monkeypatch.chdir(tmp_path)
    rc, out, err = _capture(cmd_skill, [flag], monkeypatch)
    assert rc == 0
    claude_dir = tmp_path / ".claude"
    assert not claude_dir.exists(), (
        f".claude/ must not be created by {flag}; found: {list(tmp_path.iterdir())}"
    )


# ---------------------------------------------------------------------------
# AC3 — global help: bare newt / newt -h / newt --help behavior unchanged
# ---------------------------------------------------------------------------

def test_global_bare_newt_exits_zero(monkeypatch):
    """Bare `newt` (no args) prints usage and exits 0."""
    rc, out, err = _capture_main([], monkeypatch)
    assert rc == 0, f"expected exit 0; stderr={err!r}"
    assert "Usage" in out or "usage" in out.lower(), f"usage must appear: {out!r}"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_global_help_exits_zero(flag, monkeypatch):
    """newt --help / newt -h prints usage and exits 0."""
    rc, out, err = _capture_main([flag], monkeypatch)
    assert rc == 0, f"expected exit 0; stderr={err!r}"
    assert "Usage" in out or "usage" in out.lower(), f"usage must appear: {out!r}"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_global_help_lists_all_verbs(flag, monkeypatch):
    """Global help must list all verbs so developers can discover the CLI."""
    rc, out, err = _capture_main([flag], monkeypatch)
    for verb in ("login", "logout", "models", "status", "skill"):
        assert verb in out, f"global help must list '{verb}': {out!r}"


def test_global_help_content_consistent(monkeypatch):
    """bare newt and newt --help produce the same output."""
    rc_bare, out_bare, _ = _capture_main([], monkeypatch)
    rc_help, out_help, _ = _capture_main(["--help"], monkeypatch)
    assert out_bare == out_help, (
        f"bare newt and newt --help must produce identical output\n"
        f"bare: {out_bare!r}\n--help: {out_help!r}"
    )
