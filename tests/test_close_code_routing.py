"""Unit tests for WS close-code routing (newt-error-routing-4403).

Ported from the failure-matrix audit's repro_sdk_surface.py: deterministically
drive the real robot.py routing (`_check_error_envelope_frame` and
`_check_close_error`) with synthetic server frames — one per failure class — and
assert exactly what the SDK surfaces. No network, no GPU, no spend.

Two things this gate protects:
  1. Refusals a hacker must SEE — 4403 owner-forbidden and codeless mid-session
     drops (1006/1011) — now raise typed exceptions carrying the server's
     message, instead of a silent RunResult(stop_reason="error").
  2. T1 (public SDK, additive only) — every pre-existing close code still raises
     the SAME class with an UNCHANGED message. The envelope shape here IS the
     wire contract, so the SDK renders identically to live.
"""

from __future__ import annotations

import pytest

from newt._client.robot import (
    AuthError,
    ConnectionDroppedError,
    ContractMismatchError,
    ForbiddenError,
    ModelNotFoundError,
    NewTheoryError,
    ProtocolError,
    ServerError,
    VerifierError,
    _CLOSE_CODE_TO_EXCEPTION,
    _DEFAULT_TYPE_FOR_CODE,
    _check_close_error,
    _check_error_envelope_frame,
)


# --- synthetic close-frame objects matching websockets' ConnectionClosed.rcvd ---
class _Rcvd:
    def __init__(self, code, reason=""):
        self.code = code
        self.reason = reason


class _ConnClosed(Exception):
    """Stand-in for websockets.exceptions.ConnectionClosed (only .rcvd is read)."""

    def __init__(self, code, reason=""):
        self.rcvd = _Rcvd(code, reason)
        super().__init__(f"close {code}")


class _NoRcvd(Exception):
    """A ConnectionClosed with no `.rcvd` frame at all — a truly codeless drop."""

    rcvd = None


def _envelope(code, type_, message, context=None):
    """The six-field error envelope exactly as the server's
    _build_error_envelope_from_catalog emits it before closing the socket."""
    return {
        "code": code,
        "type": type_,
        "message": message,
        "context": context or {},
        "docs": None,
        "trace_id": "tr_demo123",
    }


# Server catalog default messages (verbatim) for the classes where the server
# passes no message_override. These strings are the T1 contract: they must not
# drift, because downstream code and docs quote them.
_FORBIDDEN_MSG = (
    "That model belongs to another team. Your key can only serve models your "
    "team owns."
)
_AUTH_MSG = (
    "API key rejected. The key was revoked, never issued, or malformed. "
    "Generate a new key in the New Theory console."
)
_MODEL_NOT_FOUND_MSG = (
    "Model not found. Check spelling or call newt.list_models() to see "
    "available models."
)
_CONTRACT_MSG = (
    "Model 'so101-ft' expects state shape [6]; got [14]. Adjust read_state() to "
    "return [6]-dim state, or switch to a matching model."
)
_SERVER_MSG = (
    "The model raised an error during inference. Retry the request. If this "
    "persists, contact support with the trace_id."
)
_VERIFIER_MSG = (
    "The API key verification service is temporarily unavailable. Retry the "
    "request in a few seconds."
)


# ---------------------------------------------------------------------------
# The two fixes: 4403 forbidden is surfaced, codeless drops are not silent.
# ---------------------------------------------------------------------------

def test_forbidden_envelope_raises_forbidden_error_with_server_message():
    """4403 owner-forbidden envelope → ForbiddenError, server message preserved.

    WHY: the server sends a perfect auth.forbidden envelope explaining the key
    can only serve the team's own models. Pre-fix the SDK looked up 4403, got
    None, and returned — throwing the message on the floor. The hacker must see
    it.
    """
    env = _envelope(4403, "auth.forbidden", _FORBIDDEN_MSG, {"model": "run-one"})
    with pytest.raises(ForbiddenError) as excinfo:
        _check_error_envelope_frame(env)
    assert str(excinfo.value) == _FORBIDDEN_MSG
    assert excinfo.value.type == "auth.forbidden"
    assert excinfo.value.code == 4403


def test_forbidden_bare_close_raises_forbidden_error():
    """4403 bare WS close (no envelope) → ForbiddenError via the fallback path.

    WHY: the run loop can hit the close before any envelope frame arrives. Both
    routing paths must agree; the bare-close path must also raise ForbiddenError.
    """
    exc = _ConnClosed(4403, "forbidden")
    with pytest.raises(ForbiddenError) as excinfo:
        _check_close_error(exc, model="run-one")
    assert excinfo.value.code == 4403
    assert excinfo.value.type == "auth.forbidden"


@pytest.mark.parametrize("code", [1006, 1011])
def test_codeless_drop_raises_not_silent(code):
    """1006 / 1011 mid-session drop → ConnectionDroppedError with retry hint.

    WHY: an unmapped abnormal close is the SDK-side face of a cold-start ping
    timeout, an OOM kill, or a container restart. Pre-fix it returned silently
    → stop_reason="error", the hacker got nothing. It must raise an honest,
    typed error that names the cold-load-retry escape hatch — the Rule 10
    silent-fallback this card closes.
    """
    exc = _ConnClosed(code, "")
    with pytest.raises(ConnectionDroppedError) as excinfo:
        _check_close_error(exc, model="run-one")
    msg = str(excinfo.value).lower()
    assert "connection dropped" in msg
    assert "cold-loading" in msg and "retry" in msg
    assert isinstance(excinfo.value, NewTheoryError)


def test_truly_codeless_drop_raises_not_silent():
    """A ConnectionClosed with no `.rcvd` frame at all still raises, not returns."""
    with pytest.raises(ConnectionDroppedError):
        _check_close_error(_NoRcvd(), model="run-one")


# ---------------------------------------------------------------------------
# Normal completion must NOT raise.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", [1000, 1005])
def test_normal_close_does_not_raise(code):
    """A clean 1000/1005 close is normal completion — never an error.

    WHY: the not-silent fallback must fire only on abnormal drops. A normal
    terminal/1000 completion returns None (the run loop then reports the real
    stop_reason). Raising here would turn every successful run into a spurious
    exception.
    """
    assert _check_close_error(_ConnClosed(code, ""), model="run-one") is None


# ---------------------------------------------------------------------------
# T1 regression guard: every pre-existing code → SAME class, UNCHANGED message.
# ---------------------------------------------------------------------------

_ENVELOPE_REGRESSION = [
    (4001, "auth.invalid_key", _AUTH_MSG, AuthError),
    (4404, "model_not_found.unknown_identifier", _MODEL_NOT_FOUND_MSG, ModelNotFoundError),
    (4422, "contract_mismatch.state_shape", _CONTRACT_MSG, ContractMismatchError),
    (4500, "server.inference_error", _SERVER_MSG, ServerError),
    (4503, "verifier.unavailable", _VERIFIER_MSG, VerifierError),
]


@pytest.mark.parametrize("code,type_,message,exc_class", _ENVELOPE_REGRESSION)
def test_pre_existing_envelope_classes_unchanged(code, type_, message, exc_class):
    """T1: adding 4403 and the codeless fallback must not touch existing routing.

    Each pre-existing envelope still raises its exact class with its exact
    message (ServerError overrides __str__, so we assert on .message, the raw
    field, not str()).
    """
    env = _envelope(code, type_, message)
    with pytest.raises(exc_class) as excinfo:
        _check_error_envelope_frame(env)
    assert type(excinfo.value) is exc_class
    assert excinfo.value.type == type_
    assert excinfo.value.message == message
    assert excinfo.value.code == code


@pytest.mark.parametrize(
    "code,exc_class",
    [
        (4001, AuthError),
        (4400, ProtocolError),
        (4404, ModelNotFoundError),
        (4422, ContractMismatchError),
        (4500, ServerError),
        (4503, VerifierError),
    ],
)
def test_pre_existing_bare_close_classes_unchanged(code, exc_class):
    """T1: the bare-close fallback still raises each pre-existing class."""
    with pytest.raises(exc_class) as excinfo:
        _check_close_error(_ConnClosed(code, ""), model="m")
    assert type(excinfo.value) is exc_class


# ---------------------------------------------------------------------------
# The routing maps themselves — the wiring the two paths share.
# ---------------------------------------------------------------------------

def test_maps_gained_4403_only():
    """4403 is now mapped on both maps; no pre-existing key was removed."""
    assert _CLOSE_CODE_TO_EXCEPTION[4403] is ForbiddenError
    assert _DEFAULT_TYPE_FOR_CODE[4403] == "auth.forbidden"
    for code in (4001, 4400, 4404, 4422, 4500, 4503):
        assert code in _CLOSE_CODE_TO_EXCEPTION
        assert code in _DEFAULT_TYPE_FOR_CODE
