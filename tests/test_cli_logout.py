"""Offline unit tests for `newt logout` CLI command.

Tests cover the developer-visible contract WITHOUT touching the network.
Each test encodes why the behavior matters, not just what it does.
"""
from __future__ import annotations

import io
import json
import sys

import pytest

import newt._cli.logout as logout_mod
from newt._cli.logout import cmd_logout
from newt._credentials import CREDENTIALS_PATH, write_api_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: list[str], monkeypatch, *, tmp_path, env_key: str | None = None):
    """Run cmd_logout with a fake credentials dir, capturing stdout/stderr."""
    fake_cred_dir = tmp_path / ".nt"
    fake_cred_file = fake_cred_dir / "credentials"

    monkeypatch.setattr(logout_mod, "CREDENTIALS_DIR", fake_cred_dir)
    monkeypatch.setattr(logout_mod, "CREDENTIALS_PATH", fake_cred_file)

    if env_key is not None:
        monkeypatch.setenv("NT_API_KEY", env_key)
    else:
        monkeypatch.delenv("NT_API_KEY", raising=False)

    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    exit_code = cmd_logout(args)
    return exit_code, out.getvalue(), err.getvalue(), fake_cred_dir, fake_cred_file


def _write_fake_credentials(cred_dir, cred_file, key="nt_testkey12345"):
    """Create a fake credentials file as if written by write_api_key."""
    import stat
    cred_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    cred_file.write_text(f"api_key = {key}\n")
    cred_file.chmod(stat.S_IRUSR | stat.S_IWUSR)


# ---------------------------------------------------------------------------
# AC1: credentials file exists — delete it, print path + revocation pointer
# ---------------------------------------------------------------------------

def test_logout_deletes_credentials_and_prints_path(monkeypatch, tmp_path):
    """logout removes the credentials file and tells the user exactly what was deleted.

    A developer who just ran `newt logout` must be able to verify the file is gone
    without guessing. The output names the path so there is no ambiguity.
    """
    exit_code, out, err, cred_dir, cred_file = _run(
        [], monkeypatch, tmp_path=tmp_path
    )
    # Pre-state: no file yet — just confirming the path setup.
    # Now create and re-run.
    _write_fake_credentials(cred_dir, cred_file)
    assert cred_file.exists(), "test setup: credentials must exist before logout"

    exit_code, out, err, cred_dir, cred_file = _run(
        [], monkeypatch, tmp_path=tmp_path
    )

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    assert not cred_file.exists(), "credentials file must be gone after logout"
    assert str(cred_file) in out, f"deleted path must appear in output: {out!r}"


def test_logout_mentions_console_revocation(monkeypatch, tmp_path):
    """logout tells the developer that the key itself is still valid and where to revoke it.

    Deleting the local file is not the same as revoking the key. A developer who
    ran logout on a shared machine still has a live key — they must be told where
    to revoke it.
    """
    exit_code, out, err, cred_dir, cred_file = _run(
        [], monkeypatch, tmp_path=tmp_path
    )
    _write_fake_credentials(cred_dir, cred_file)

    exit_code, out, err, cred_dir, cred_file = _run(
        [], monkeypatch, tmp_path=tmp_path
    )

    assert exit_code == 0
    combined = out.lower()
    assert "revoke" in combined or "console" in combined, (
        f"must mention revocation or the console: {out!r}"
    )


def test_logout_does_not_touch_other_files_in_nt_dir(monkeypatch, tmp_path):
    """logout deletes only the credentials file; sibling files in ~/.nt/ are untouched.

    A developer may have other files in ~/.nt/ — logout must leave them alone.
    This guards against accidental recursive deletion or directory removal.
    """
    exit_code, out, err, cred_dir, cred_file = _run(
        [], monkeypatch, tmp_path=tmp_path
    )
    cred_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write_fake_credentials(cred_dir, cred_file)
    # Create a sibling file
    sibling = cred_dir / "config"
    sibling.write_text("some_setting = true\n")

    exit_code, out, err, cred_dir, cred_file = _run(
        [], monkeypatch, tmp_path=tmp_path
    )

    assert exit_code == 0
    assert not cred_file.exists(), "credentials must be removed"
    assert sibling.exists(), "sibling file in ~/.nt/ must not be removed"
    # Dir must still exist since sibling is present
    assert cred_dir.exists(), "~/.nt/ dir must not be removed when it still has files"


# ---------------------------------------------------------------------------
# AC2: no credentials file — idempotent exit 0
# ---------------------------------------------------------------------------

def test_logout_idempotent_when_no_credentials(monkeypatch, tmp_path):
    """logout exits 0 with an 'already logged out' message when no file exists.

    Running `newt logout` twice must succeed silently — this is the idempotent shape,
    not an error. Scripts and agents should be able to call it unconditionally.
    """
    exit_code, out, err, _, _ = _run([], monkeypatch, tmp_path=tmp_path)

    assert exit_code == 0, f"expected exit 0 when no credentials; stderr={err!r}"
    assert "already" in out.lower() or "logged out" in out.lower(), (
        f"must say 'already logged out' shape: {out!r}"
    )
    assert err == "", f"no stderr expected on clean no-op: {err!r}"


def test_logout_idempotent_second_call(monkeypatch, tmp_path):
    """Two consecutive logout calls both succeed — neither errors on the second call.

    Any tool that retries logout (deploy scripts, CI cleanup) must not fail on the
    second invocation.
    """
    exit_code, out, err, cred_dir, cred_file = _run(
        [], monkeypatch, tmp_path=tmp_path
    )
    _write_fake_credentials(cred_dir, cred_file)

    # First call
    exit_code1, out1, err1, cred_dir, cred_file = _run(
        [], monkeypatch, tmp_path=tmp_path
    )
    assert exit_code1 == 0
    assert not cred_file.exists()

    # Second call — file is gone; must still exit 0
    exit_code2, out2, err2, _, _ = _run([], monkeypatch, tmp_path=tmp_path)
    assert exit_code2 == 0, f"second logout must be exit 0; stderr={err2!r}"


# ---------------------------------------------------------------------------
# AC3: NT_API_KEY env var warning
# ---------------------------------------------------------------------------

def test_logout_warns_when_env_key_is_set(monkeypatch, tmp_path):
    """When NT_API_KEY is set, logout warns the user loudly after removing the file.

    logout cannot unset a parent shell's environment variable. A developer who
    ran `export NT_API_KEY=...` is still authenticated even after logout — they
    must be told, so they know to unset it for a full sign-out.
    """
    exit_code, out, err, cred_dir, cred_file = _run(
        [], monkeypatch, tmp_path=tmp_path, env_key="nt_envkey123"
    )
    _write_fake_credentials(cred_dir, cred_file)

    exit_code, out, err, _, _ = _run(
        [], monkeypatch, tmp_path=tmp_path, env_key="nt_envkey123"
    )

    assert exit_code == 0
    assert "NT_API_KEY" in out, (
        f"must name NT_API_KEY in the env-var warning: {out!r}"
    )
    combined = out.lower()
    assert "unset" in combined or "environment" in combined, (
        f"must tell developer to unset the variable: {out!r}"
    )


def test_logout_no_env_warning_when_env_key_absent(monkeypatch, tmp_path):
    """Without NT_API_KEY in the environment, no env-var warning appears.

    The warning is signal, not noise. Printing it when there is no env key
    trains developers to ignore it.
    """
    exit_code, out, err, cred_dir, cred_file = _run(
        [], monkeypatch, tmp_path=tmp_path, env_key=None
    )
    _write_fake_credentials(cred_dir, cred_file)

    exit_code, out, err, _, _ = _run(
        [], monkeypatch, tmp_path=tmp_path, env_key=None
    )

    assert exit_code == 0
    assert "NT_API_KEY" not in out, (
        f"no env-var warning expected when NT_API_KEY is unset: {out!r}"
    )


def test_logout_already_out_with_env_key_still_warns(monkeypatch, tmp_path):
    """Even in the already-logged-out path, NT_API_KEY warning fires if env var is set.

    A developer who has no credentials file but still has NT_API_KEY set is still
    authenticated — the warning applies regardless of whether a file was present.
    """
    # No credentials file, but env key set
    exit_code, out, err, _, _ = _run(
        [], monkeypatch, tmp_path=tmp_path, env_key="nt_envkey123"
    )

    assert exit_code == 0
    assert "NT_API_KEY" in out, (
        f"env-var warning must fire even on already-logged-out path: {out!r}"
    )


# ---------------------------------------------------------------------------
# AC4: --json output
# ---------------------------------------------------------------------------

def test_logout_json_removed_action(monkeypatch, tmp_path):
    """`newt logout --json` emits action='removed' with the credentials_path and env_var_warning.

    Agents and scripts calling `newt logout --json` get a machine-readable result
    with all necessary fields to branch on.
    """
    exit_code, out, err, cred_dir, cred_file = _run(
        ["--json"], monkeypatch, tmp_path=tmp_path
    )
    _write_fake_credentials(cred_dir, cred_file)

    exit_code, out, err, _, _ = _run(
        ["--json"], monkeypatch, tmp_path=tmp_path
    )

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    data = json.loads(out)
    assert data["action"] == "removed", f"expected action=removed: {data!r}"
    assert "credentials_path" in data, f"credentials_path must be present: {data!r}"
    assert "env_var_warning" in data, f"env_var_warning must be present: {data!r}"
    assert data["env_var_warning"] is False  # env key not set in this call


def test_logout_json_already_logged_out_action(monkeypatch, tmp_path):
    """`newt logout --json` emits action='already_logged_out' when no file exists.

    An agent that calls logout idempotently can branch on action without special-casing
    the exit code.
    """
    exit_code, out, err, _, _ = _run(
        ["--json"], monkeypatch, tmp_path=tmp_path
    )

    assert exit_code == 0
    data = json.loads(out)
    assert data["action"] == "already_logged_out", f"expected already_logged_out: {data!r}"


def test_logout_json_env_var_warning_true_when_set(monkeypatch, tmp_path):
    """`newt logout --json` sets env_var_warning=true when NT_API_KEY is in the environment."""
    exit_code, out, err, cred_dir, cred_file = _run(
        ["--json"], monkeypatch, tmp_path=tmp_path, env_key="nt_envkey123"
    )
    _write_fake_credentials(cred_dir, cred_file)

    exit_code, out, err, _, _ = _run(
        ["--json"], monkeypatch, tmp_path=tmp_path, env_key="nt_envkey123"
    )

    data = json.loads(out)
    assert data["env_var_warning"] is True, (
        f"env_var_warning must be true when NT_API_KEY is set: {data!r}"
    )


# ---------------------------------------------------------------------------
# AC5: round-trip test (mocked login write → logout → second logout)
# ---------------------------------------------------------------------------

def test_logout_round_trip_login_then_logout_then_idempotent(monkeypatch, tmp_path):
    """A mock login writes credentials; logout removes them; second logout is idempotent.

    This is the full lifecycle: authenticate, sign out, confirm sign-out is clean.
    All offline — no network.
    """
    cred_dir = tmp_path / ".nt"
    cred_file = cred_dir / "credentials"

    monkeypatch.setattr(logout_mod, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(logout_mod, "CREDENTIALS_PATH", cred_file)
    monkeypatch.delenv("NT_API_KEY", raising=False)

    # -- "login" phase: write credentials as write_api_key would --
    import newt._credentials as cred_mod
    orig_path = cred_mod.CREDENTIALS_PATH
    orig_dir = cred_mod.CREDENTIALS_DIR
    monkeypatch.setattr(cred_mod, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(cred_mod, "CREDENTIALS_PATH", cred_file)
    cred_mod.write_api_key("nt_roundtripkey123")
    assert cred_file.exists(), "login step: credentials must exist after write_api_key"
    assert cred_file.read_text().strip() == "api_key = nt_roundtripkey123"

    # -- first logout --
    out1 = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out1)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    exit_code1 = cmd_logout([])
    assert exit_code1 == 0
    assert not cred_file.exists(), "credentials must be gone after first logout"

    # -- second logout (idempotent path) --
    out2 = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out2)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    exit_code2 = cmd_logout([])
    assert exit_code2 == 0, "second logout must be exit 0 (idempotent)"
    assert "already" in out2.getvalue().lower() or "logged out" in out2.getvalue().lower(), (
        f"second logout must say already-logged-out shape: {out2.getvalue()!r}"
    )
