"""Offline unit tests for `newt models` CLI command.

Tests cover the developer-visible contract WITHOUT touching the network.
Each test encodes why the behavior matters, not just what it does.
"""
from __future__ import annotations

import io
import json
import sys

import pytest

import newt
from newt._client.robot import AuthError, RegistryUnavailable
from newt._cli.models import cmd_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: list[str], monkeypatch, models_return=None, side_effect=None):
    """Run cmd_models with a mocked newt.list_models, capturing stdout/stderr."""
    captured_out = io.StringIO()
    captured_err = io.StringIO()

    if side_effect is not None:
        def fake_list_models(*a, **kw):
            raise side_effect
    else:
        def fake_list_models(*a, **kw):
            return models_return if models_return is not None else []

    monkeypatch.setattr(newt, "list_models", fake_list_models)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")
    # Ensure stdout/stderr are captured
    monkeypatch.setattr(sys, "stdout", captured_out)
    monkeypatch.setattr(sys, "stderr", captured_err)

    exit_code = cmd_models(args)
    return exit_code, captured_out.getvalue(), captured_err.getvalue()


_SAMPLE_MODELS = [
    {
        "uid": "ft_base_nt0fp3",
        "type": "fine_tune",
        "base": "base_nt0fp3",
        "tags": ["nt0-fp3"],
        "contract": {
            "action_axes": [
                "shoulder_pan", "shoulder_lift", "elbow", "forearm_roll",
                "wrist_angle", "wrist_rotate", "gripper",
            ]
        },
    },
    {"uid": "base_nt0fp3", "type": "base", "base": None, "tags": []},
]


# ---------------------------------------------------------------------------
# Golden: developer who just logged in types `newt models` and sees what
# their key can drive — catalog renders compactly, exit 0.
# ---------------------------------------------------------------------------

def test_models_renders_catalog(monkeypatch):
    """A developer who just logged in types `newt models` and sees the catalog.

    The command must exit 0 and print each model's UID, type, and base on its
    own line. The developer can scan the list to find the model they want.
    """
    exit_code, out, err = _run([], monkeypatch, models_return=_SAMPLE_MODELS)

    assert exit_code == 0, f"expected exit 0, got {exit_code}; stderr={err!r}"
    assert "ft_base_nt0fp3" in out
    assert "base_nt0fp3" in out
    assert err == ""


def test_models_renders_axes_from_contract(monkeypatch):
    """A developer scanning the catalog sees each model's labeled action axes.

    The registry payload carries axes at contract.action_axes — the human
    render must surface them (the axes are part of the reward moment: you see
    what the model actually drives). A model without a contract renders cleanly
    with no axes fragment. This test exists because the renderer once read a
    nonexistent top-level "axes" key and the column silently never printed.
    """
    exit_code, out, _ = _run([], monkeypatch, models_return=_SAMPLE_MODELS)

    assert exit_code == 0
    lines = [l for l in out.splitlines() if l.strip()]
    ft_line = next(l for l in lines if "ft_base_nt0fp3" in l)
    base_line = next(l for l in lines if l.strip().startswith("base_nt0fp3"))

    assert "axes [" in ft_line, f"axes must render for a model with a contract: {ft_line!r}"
    assert "shoulder_pan" in ft_line and "gripper" in ft_line, (
        f"axis labels from contract.action_axes must appear: {ft_line!r}"
    )
    assert "axes [" not in base_line, (
        f"model without a contract must not render an axes fragment: {base_line!r}"
    )


def test_models_one_line_per_model(monkeypatch):
    """Each model appears on its own line — so a developer can scan or grep."""
    exit_code, out, _ = _run([], monkeypatch, models_return=_SAMPLE_MODELS)

    assert exit_code == 0
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == len(_SAMPLE_MODELS), (
        f"expected {len(_SAMPLE_MODELS)} non-empty lines, got {len(lines)}: {lines!r}"
    )


# ---------------------------------------------------------------------------
# Golden: `--json` emits a valid JSON array
# ---------------------------------------------------------------------------

def test_models_json_flag_emits_valid_array(monkeypatch):
    """`--json` emits a JSON array so agents and scripts can parse the catalog.

    Every model dict must be present. The output must parse as a list.
    """
    exit_code, out, err = _run(["--json"], monkeypatch, models_return=_SAMPLE_MODELS)

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    parsed = json.loads(out)
    assert isinstance(parsed, list), f"expected JSON array, got {type(parsed)}"
    assert len(parsed) == len(_SAMPLE_MODELS)
    uids = [m["uid"] for m in parsed]
    assert "ft_base_nt0fp3" in uids
    assert "base_nt0fp3" in uids


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_models_bad_key_exits_nonzero_and_blames_key(monkeypatch):
    """A developer with a revoked key sees an auth failure, not a registry outage.

    This is the affordance guard: the command must say the KEY is bad, not
    report that the registry is down. Otherwise the developer chases phantom
    infra problems.
    """
    exc = AuthError(
        code=4001,
        type="auth.invalid_key",
        message="Authentication failed: API key rejected by registry /v1/models. Rotate your key.",
        context={"key_prefix": "nt_testke"},
    )
    exit_code, out, err = _run([], monkeypatch, side_effect=exc)

    assert exit_code != 0, "bad key must produce non-zero exit"
    assert "authentication" in err.lower() or "key" in err.lower(), (
        f"error must mention authentication or key, got: {err!r}"
    )
    # Must NOT say the registry is down when the key is the problem
    assert "registry" not in err.lower() or "key" in err.lower(), (
        f"must not only blame registry when key is bad: {err!r}"
    )


def test_models_registry_unavailable_exits_nonzero(monkeypatch):
    """When the registry is genuinely unreachable, the command fails with context."""
    exc = RegistryUnavailable(
        bootstrap_url="https://nt-registry-production.up.railway.app",
        reason="connection refused",
    )
    exit_code, out, err = _run([], monkeypatch, side_effect=exc)

    assert exit_code != 0
    assert "registry" in err.lower() or "unreachable" in err.lower()


def test_models_no_key_exits_nonzero_and_tells_user_what_to_do(monkeypatch, tmp_path):
    """Without a key, the developer gets a clear 'run newt login' message."""
    import io as _io
    import sys as _sys

    # Remove NT_API_KEY from env and point credentials to an empty tmp dir
    monkeypatch.delenv("NT_API_KEY", raising=False)

    # Patch read_api_key to return None (no credentials file)
    import newt._cli.models as models_mod
    monkeypatch.setattr(models_mod, "read_api_key", lambda: None)

    captured_err = _io.StringIO()
    monkeypatch.setattr(_sys, "stderr", captured_err)
    monkeypatch.setattr(_sys, "stdout", _io.StringIO())

    exit_code = cmd_models([])

    assert exit_code != 0
    err = captured_err.getvalue()
    assert "login" in err.lower() or "NT_API_KEY" in err, (
        f"must tell the developer how to fix: {err!r}"
    )


def test_models_empty_catalog(monkeypatch):
    """An empty catalog prints a clean message rather than silently outputting nothing."""
    exit_code, out, err = _run([], monkeypatch, models_return=[])

    assert exit_code == 0
    assert out.strip()  # some output, even if empty


def test_models_json_empty_catalog_is_valid_array(monkeypatch):
    """--json with an empty catalog still emits a valid JSON array, not nothing."""
    exit_code, out, err = _run(["--json"], monkeypatch, models_return=[])

    assert exit_code == 0
    parsed = json.loads(out)
    assert parsed == []
