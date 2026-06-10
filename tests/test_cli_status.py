"""Offline unit tests for `newt status` CLI command.

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
from newt._cli.status import cmd_status


_SAMPLE_MODELS = [
    {"uid": "ft_base_nt0fp3", "type": "fine_tune", "base": "base_nt0fp3", "tags": ["nt0-fp3"]},
    {"uid": "base_nt0fp3", "type": "base", "base": None, "tags": []},
]


def _run(args: list[str], monkeypatch, models_return=None, side_effect=None):
    """Run cmd_status with mocked newt.list_models, capturing stdout/stderr."""
    out = io.StringIO()
    err = io.StringIO()

    if side_effect is not None:
        def fake_list_models(*a, **kw):
            raise side_effect
    else:
        def fake_list_models(*a, **kw):
            return models_return if models_return is not None else _SAMPLE_MODELS

    monkeypatch.setattr(newt, "list_models", fake_list_models)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey12345")
    monkeypatch.delenv("NT_BOOTSTRAP_URL", raising=False)
    monkeypatch.delenv("NT_INFERENCE_URL", raising=False)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    exit_code = cmd_status(args)
    return exit_code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# Golden: `newt status` tells me my situation in one screen
# ---------------------------------------------------------------------------

def test_status_shows_key_source_and_registry(monkeypatch):
    """A developer types `newt status` and sees their key source and registry state.

    The output must name the key source (env var or credentials file) and confirm
    the registry is reachable with latency — one screen covers the situation.
    """
    exit_code, out, err = _run([], monkeypatch, models_return=_SAMPLE_MODELS)

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    assert "NT_API_KEY" in out or "nt_testke" in out, (
        f"key source must appear in output: {out!r}"
    )
    assert "reachable" in out.lower(), f"registry status must show 'reachable': {out!r}"


def test_status_bad_key_blames_key_not_registry(monkeypatch):
    """With a bad key, status names the KEY as the problem — not a registry outage.

    Misattributing a rejected key to a registry outage sends developers chasing
    phantom infra problems. The affordance guard: bad key → say key, not registry.
    """
    exc = AuthError(
        code=4001,
        type="auth.invalid_key",
        message="Authentication failed: API key rejected.",
        context={"key_prefix": "nt_testke"},
    )
    exit_code, out, err = _run([], monkeypatch, side_effect=exc)

    assert exit_code != 0, "bad key must produce non-zero exit"
    combined = (out + err).lower()
    assert "key" in combined, f"must mention 'key': {combined!r}"
    assert "unreachable" not in combined, (
        f"must not say registry unreachable when key is the problem: {combined!r}"
    )


def test_status_no_key_exits_nonzero_and_guides_user(monkeypatch):
    """Without any API key, status exits non-zero and tells the developer what to do."""
    import newt._cli.status as status_mod

    monkeypatch.delenv("NT_API_KEY", raising=False)
    monkeypatch.setattr(status_mod, "read_api_key", lambda: None)

    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    exit_code = cmd_status([])

    assert exit_code != 0
    combined = (out.getvalue() + err.getvalue()).lower()
    assert "login" in combined or "nt_api_key" in combined, (
        f"must tell user how to fix: {combined!r}"
    )


def test_status_override_vars_shown_loudly(monkeypatch):
    """When NT_BOOTSTRAP_URL is set, status surfaces it prominently in the output.

    Silent overrides are the classic footgun — a developer running against a staging
    endpoint must see it immediately without digging through their env.
    """
    # Set up env and mock directly (can't use _run helper — it clears override vars)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey12345")
    monkeypatch.setenv("NT_BOOTSTRAP_URL", "https://staging.example.com")
    monkeypatch.delenv("NT_INFERENCE_URL", raising=False)
    monkeypatch.setattr(newt, "list_models", lambda *a, **kw: _SAMPLE_MODELS)

    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    exit_code = cmd_status([])

    assert exit_code == 0
    output = out.getvalue()
    assert "NT_BOOTSTRAP_URL" in output, f"override must appear in output: {output!r}"
    assert "staging.example.com" in output, f"override URL must be shown: {output!r}"


def test_status_registry_down_is_distinct_from_bad_key(monkeypatch):
    """A genuinely unreachable registry produces a different message than a bad key.

    The developer needs to know: debug the key, or debug connectivity.
    """
    exc = RegistryUnavailable(
        bootstrap_url="https://nt-registry-production.up.railway.app",
        reason="connection refused",
    )
    exit_code, out, err = _run([], monkeypatch, side_effect=exc)

    assert exit_code != 0
    combined = (out + err).lower()
    assert "unreachable" in combined or "registry" in combined, (
        f"must mention registry unreachable: {combined!r}"
    )
    # key_rejected wording must not appear when the issue is connectivity
    assert "key rejected" not in combined, (
        f"must not say key rejected when issue is connectivity: {combined!r}"
    )


# ---------------------------------------------------------------------------
# Golden: agent self-diagnosis via `newt status --json`
# ---------------------------------------------------------------------------

def test_status_json_has_all_required_fields(monkeypatch):
    """An agent calls `newt status --json` and gets all required machine-readable fields.

    All four fields must be present — agents self-diagnose by parsing these.
    """
    exit_code, out, err = _run(["--json"], monkeypatch, models_return=_SAMPLE_MODELS)

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    data = json.loads(out)
    for field in ("key_source", "overrides", "registry_reachable", "latency_ms"):
        assert field in data, f"required field '{field}' missing: {data!r}"


def test_status_json_values_are_correct_for_good_key(monkeypatch):
    """JSON carries correct key_source='env_var', registry_reachable=True, and latency_ms int."""
    exit_code, out, err = _run(["--json"], monkeypatch, models_return=_SAMPLE_MODELS)

    data = json.loads(out)
    assert data["key_source"] == "env_var", f"expected 'env_var': {data!r}"
    assert data["registry_reachable"] is True
    assert isinstance(data["latency_ms"], int)
    assert data["latency_ms"] >= 0
    assert data["overrides"]["NT_BOOTSTRAP_URL"] is None
    assert data["overrides"]["NT_INFERENCE_URL"] is None


def test_status_json_bad_key_exit_nonzero_registry_still_reachable(monkeypatch):
    """With a bad key, `--json` exits non-zero but marks registry_reachable=True.

    The registry responded (401 is a response) — it is reachable.
    An agent parsing this distinguishes: bad key → rotate vs. outage → check infra.
    """
    exc = AuthError(
        code=4001,
        type="auth.invalid_key",
        message="Authentication failed: API key rejected.",
        context={"key_prefix": "nt_testke"},
    )
    exit_code, out, err = _run(["--json"], monkeypatch, side_effect=exc)

    assert exit_code != 0, "bad key must be non-zero exit"
    data = json.loads(out)
    assert data["registry_reachable"] is True, (
        f"registry was reachable (it sent back 401): {data!r}"
    )
    assert data["key_source"] == "env_var"


def test_status_json_no_key_fields_correct(monkeypatch):
    """Without any key, `--json` exits non-zero with key_source='none' and correct nulls."""
    import newt._cli.status as status_mod

    monkeypatch.delenv("NT_API_KEY", raising=False)
    monkeypatch.setattr(status_mod, "read_api_key", lambda: None)

    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    exit_code = cmd_status(["--json"])

    assert exit_code != 0
    data = json.loads(out.getvalue())
    assert data["key_source"] == "none"
    assert data["registry_reachable"] is False
    assert data["latency_ms"] is None
