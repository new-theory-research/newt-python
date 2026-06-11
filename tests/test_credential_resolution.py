"""Golden test matrix — credential resolution contract.

Pins the env-first precedence rule across Robot construction and the `models` CLI verb.
No network calls — registry discovery functions are patched to return an empty list.

Why this test exists: three field receipts on 2026-06-10 showed the SDK was file-first
while the CLI was env-first. The behavior change (SDK flipped to env-first) is only
observable when env and file both exist and disagree — that case MUST stay red if it
regresses.

Matrix:
    env only  → env key used
    file only → file key used
    both      → env wins  ← the behavior change; this row is the critical one
    neither   → AuthError raised

Covers:
    Robot construction (newt.Robot, api_key=None triggers auto-resolution)
    CLI verb (newt._cli.models.cmd_models)

Run:
    uv run pytest tests/test_credential_resolution.py -v
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

import newt
from newt._client.robot import AuthError


_ENV_KEY = "nt_envkeyenvkeyenvkeyenvkeyenvkeyenvkeyenv0"
_FILE_KEY = "nt_filekeyfikeyfilekeyfikeyfilekeyfil0000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_creds(path: Path, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"api_key = {key}\n")


def _make_robot(monkeypatch) -> newt.Robot:
    """Construct Robot(api_key=None) with discovery patched out."""
    monkeypatch.setattr("newt._client.robot._fetch_registry", lambda url, key: [])
    monkeypatch.setattr(
        "newt._client.robot._resolve_model_endpoint",
        lambda registry, model, url: "wss://dummy.test/stream",
    )
    return newt.Robot(
        read_state=lambda: {},
        execute=lambda chunk: None,
    )


# ---------------------------------------------------------------------------
# Robot: env / file / both / neither
# ---------------------------------------------------------------------------


def test_robot_env_only(monkeypatch, tmp_path):
    """Robot with only NT_API_KEY set resolves to the env key.

    Developer sets the env var in their shell; no credentials file present.
    Robot must use that key — not raise, not guess.
    """
    monkeypatch.setenv("NT_API_KEY", _ENV_KEY)
    monkeypatch.setattr("newt._credentials.CREDENTIALS_PATH", tmp_path / "credentials")

    robot = _make_robot(monkeypatch)

    assert robot._api_key == _ENV_KEY


def test_robot_file_only(monkeypatch, tmp_path):
    """Robot with only ~/.nt/credentials present resolves to the file key.

    Developer ran `newt login`; NT_API_KEY is not set in the environment.
    Robot must read the credentials file and use that key.
    """
    monkeypatch.delenv("NT_API_KEY", raising=False)
    cred_path = tmp_path / "credentials"
    _write_creds(cred_path, _FILE_KEY)
    monkeypatch.setattr("newt._credentials.CREDENTIALS_PATH", cred_path)

    robot = _make_robot(monkeypatch)

    assert robot._api_key == _FILE_KEY


def test_robot_both_env_wins(monkeypatch, tmp_path):
    """When both NT_API_KEY and ~/.nt/credentials are set, env key wins.

    This is the behavior change: SDK was file-first before brief-247. Now env wins.
    An agent or CI job with NT_API_KEY set must not be overridden by a stale
    credentials file on the same machine.
    """
    monkeypatch.setenv("NT_API_KEY", _ENV_KEY)
    cred_path = tmp_path / "credentials"
    _write_creds(cred_path, _FILE_KEY)
    monkeypatch.setattr("newt._credentials.CREDENTIALS_PATH", cred_path)

    robot = _make_robot(monkeypatch)

    assert robot._api_key == _ENV_KEY, (
        f"env key must win over file key; got {robot._api_key!r} "
        f"(env={_ENV_KEY!r}, file={_FILE_KEY!r})"
    )
    assert robot._api_key != _FILE_KEY


def test_robot_neither_raises_auth_error(monkeypatch, tmp_path):
    """Robot with no key anywhere raises AuthError — not a silent hang.

    A new developer who hasn't logged in and hasn't set the env var must get
    a clear error pointing them toward `newt login` or NT_API_KEY.
    """
    monkeypatch.delenv("NT_API_KEY", raising=False)
    monkeypatch.setattr("newt._credentials.CREDENTIALS_PATH", tmp_path / "credentials")
    monkeypatch.setattr("newt._client.robot._fetch_registry", lambda url, key: [])
    monkeypatch.setattr(
        "newt._client.robot._resolve_model_endpoint",
        lambda registry, model, url: "wss://dummy.test/stream",
    )

    with pytest.raises(AuthError) as exc_info:
        newt.Robot(read_state=lambda: {}, execute=lambda chunk: None)

    msg = str(exc_info.value).lower()
    assert "login" in msg or "nt_api_key" in msg.upper() or "api key" in msg, (
        f"AuthError must point toward a fix (login / NT_API_KEY); got: {exc_info.value!r}"
    )


# ---------------------------------------------------------------------------
# CLI verb (models): env / file / both
# ---------------------------------------------------------------------------


def _run_models(monkeypatch, *, env_key: str | None, file_key: str | None):
    """Run cmd_models with controlled credential state; return (exit_code, captured_api_key)."""
    from newt._cli.models import cmd_models

    if env_key is not None:
        monkeypatch.setenv("NT_API_KEY", env_key)
    else:
        monkeypatch.delenv("NT_API_KEY", raising=False)

    monkeypatch.setattr(
        "newt._cli.models.read_api_key",
        lambda: file_key,
    )

    captured: list[str] = []

    def fake_list_models(api_key, *args, **kwargs):
        captured.append(api_key)
        return []

    monkeypatch.setattr(newt, "list_models", fake_list_models)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    exit_code = cmd_models([])
    return exit_code, captured[0] if captured else None


def test_models_cli_env_only(monkeypatch):
    """newt models with only NT_API_KEY set calls list_models with the env key.

    Developer set NT_API_KEY; no credentials file. The command must authenticate
    with that key — not fail, not fall through to a non-existent file key.
    """
    exit_code, used_key = _run_models(monkeypatch, env_key=_ENV_KEY, file_key=None)

    assert exit_code == 0
    assert used_key == _ENV_KEY


def test_models_cli_file_only(monkeypatch):
    """newt models with only ~/.nt/credentials set calls list_models with the file key.

    Developer ran `newt login`; NT_API_KEY not set. The command must read the
    credentials file and authenticate — not fail with 'no key found'.
    """
    exit_code, used_key = _run_models(monkeypatch, env_key=None, file_key=_FILE_KEY)

    assert exit_code == 0
    assert used_key == _FILE_KEY


def test_models_cli_both_env_wins(monkeypatch):
    """newt models with both NT_API_KEY and credentials file uses the env key.

    Env wins, same rule as Robot. A CI job with NT_API_KEY must not be overridden
    by whatever key a previous `newt login` wrote to disk on the same machine.
    """
    exit_code, used_key = _run_models(monkeypatch, env_key=_ENV_KEY, file_key=_FILE_KEY)

    assert exit_code == 0
    assert used_key == _ENV_KEY, (
        f"env key must win over file key; got {used_key!r} "
        f"(env={_ENV_KEY!r}, file={_FILE_KEY!r})"
    )
    assert used_key != _FILE_KEY
