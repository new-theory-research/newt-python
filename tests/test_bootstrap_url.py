"""Offline unit tests for registry-bootstrap URL resolution (brief-233).

Why this matters: the registry bootstrap is the FIRST network call every Robot()
makes. When the default pointed at the NT0-FP3 Modal app, a cold Modal container
turned "import newt; Robot(...)" into a multi-second (sometimes failing) first
experience — golden test #1 ("my first call resolves instantly, even when the
model is cold") hinged on it. The default now points at the always-on Railway
registry; these tests pin that default AND the env-override semantics so neither
silently regresses. No network: only _resolve_bootstrap_url() (pure env logic)
is exercised.
"""
from __future__ import annotations

import pytest

from newt._client.robot import _DEFAULT_BOOTSTRAP_URL, _resolve_bootstrap_url

_RAILWAY_URL = "https://nt-registry-production.up.railway.app"


@pytest.fixture()
def clean_env(monkeypatch):
    """No bootstrap-related env vars — what a fresh developer's shell looks like."""
    monkeypatch.delenv("NT_BOOTSTRAP_URL", raising=False)
    monkeypatch.delenv("NT_INFERENCE_URL", raising=False)


def test_default_bootstrap_is_railway_registry(clean_env):
    """With no env overrides, discovery hits the always-on Railway registry.

    If this regresses to a Modal URL, every fresh developer's first Robot() call
    pays a Modal cold start (or times out) just to READ the registry — the exact
    failure golden test #1 exists to prevent.
    """
    assert _resolve_bootstrap_url() == _RAILWAY_URL
    assert _DEFAULT_BOOTSTRAP_URL == _RAILWAY_URL
    # No Modal app may sit behind registry discovery by default.
    assert "modal.run" not in _resolve_bootstrap_url()


def test_nt_bootstrap_url_env_still_overrides_default(clean_env, monkeypatch):
    """NT_BOOTSTRAP_URL overrides the default, verbatim — same semantics as before.

    Smokes and local registry servers depend on this escape hatch; the Railway
    flip must not have narrowed it.
    """
    monkeypatch.setenv("NT_BOOTSTRAP_URL", "http://localhost:8000")
    assert _resolve_bootstrap_url() == "http://localhost:8000"


def test_nt_bootstrap_url_wins_over_nt_inference_url(clean_env, monkeypatch):
    """Precedence is unchanged: NT_BOOTSTRAP_URL > NT_INFERENCE_URL-derived > default.

    Pinned because "identical semantics" means the ORDER, not just each var in
    isolation — a reordering would silently repoint smokes that set both.
    """
    monkeypatch.setenv("NT_BOOTSTRAP_URL", "https://explicit.example")
    monkeypatch.setenv("NT_INFERENCE_URL", "wss://derived.example/stream")
    assert _resolve_bootstrap_url() == "https://explicit.example"


def test_nt_inference_url_derivation_unchanged(clean_env, monkeypatch):
    """NT_INFERENCE_URL still derives the HTTPS bootstrap host (strip WS scheme + path).

    The middle rung of the resolution order — untouched by the Railway flip, and
    this test proves it stayed untouched.
    """
    monkeypatch.setenv("NT_INFERENCE_URL", "wss://some-host.example/stream")
    assert _resolve_bootstrap_url() == "https://some-host.example"
