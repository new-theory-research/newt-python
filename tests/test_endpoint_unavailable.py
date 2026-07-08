"""WS-upgrade HTTP rejections map to the *honest* error, not a fake auth failure.

Receipt (2026-06-29): a tester hit a 404 during the WS upgrade — the stable
inference endpoint was down — and the SDK told him "Authentication failed …
Check your API key and rotate it." He chased a working key for nothing. A 404 is
an endpoint-availability problem, not an auth problem; only 401/403 is auth.

These tests pin that boundary so the lie can't come back: a 404/5xx must raise
EndpointUnavailableError and must NOT tell the developer to touch their key.
"""

import warnings
from unittest.mock import patch

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

import newt


def _make_robot() -> newt.Robot:
    """Robot with NT_INFERENCE_URL set so registry discovery is skipped."""
    with patch.dict("os.environ", {"NT_INFERENCE_URL": "wss://fake.invalid/stream"}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", newt.EnvOverrideWarning)
            return newt.Robot(api_key="nt_fakekey")


def _invalid_status(code: int) -> InvalidStatus:
    return InvalidStatus(Response(code, "x", Headers()))


def _raise_status(code: int):
    def _connect(*_args, **_kwargs):
        raise _invalid_status(code)

    return _connect


def test_404_is_endpoint_error_not_auth(monkeypatch):
    """A 404 during the WS upgrade → EndpointUnavailableError, never AuthError.

    If this regresses, the SDK is back to blaming the developer's key for a
    server-side outage — the exact failure that cost a tester ~30 minutes.
    """
    robot = _make_robot()
    monkeypatch.setattr("newt._client.robot.connect", _raise_status(404))

    with pytest.raises(newt.EndpointUnavailableError) as exc_info:
        robot.infer({})

    exc = exc_info.value
    assert not isinstance(exc, newt.AuthError), "404 must not be an AuthError"
    assert exc.type == "endpoint.not_found"
    assert exc.code == 404
    assert exc.context["http_status"] == 404
    assert "fake.invalid" in exc.context["url"]

    # The anti-lie assertion: a 404 message must never send the dev to their key.
    msg = exc.message.lower()
    assert "api key" not in msg and "rotate" not in msg, (
        "404 message must not blame the API key — that is the regression we are fencing"
    )
    assert "not an api-key problem" in msg


def test_503_is_endpoint_unavailable(monkeypatch):
    """A 5xx during the upgrade → EndpointUnavailableError(endpoint.unavailable)."""
    robot = _make_robot()
    monkeypatch.setattr("newt._client.robot.connect", _raise_status(503))

    with pytest.raises(newt.EndpointUnavailableError) as exc_info:
        robot.infer({})

    assert exc_info.value.type == "endpoint.unavailable"
    assert exc_info.value.code == 503
    assert "rotate" not in exc_info.value.message.lower()


@pytest.mark.parametrize("code", [401, 403])
def test_upgrade_401_403_is_endpoint_not_auth(monkeypatch, code):
    """401/403 at the *upgrade* is a routing/proxy problem, not an NT-key problem.

    The NT key is validated after the socket is accepted (WS close 4001), so no
    upgrade-level HTTP status reflects key validity. A 403 in particular is what
    Starlette returns when a path has no WebSocket route — observed live against a
    real Modal endpoint. Mapping it to "rotate your API key" is the same lie in a
    different costume.
    """
    robot = _make_robot()
    monkeypatch.setattr("newt._client.robot.connect", _raise_status(code))

    with pytest.raises(newt.EndpointUnavailableError) as exc_info:
        robot.infer({})

    exc = exc_info.value
    assert not isinstance(exc, newt.AuthError)
    assert exc.code == code
    msg = exc.message.lower()
    assert "rotate" not in msg, "upgrade 401/403 must not tell the dev to rotate their key"
    assert "not an nt api-key problem" in msg


def test_endpoint_error_not_retried(monkeypatch):
    """A 404 raises immediately — no verifier-retry budget burned on a dead endpoint."""
    robot = _make_robot()
    calls = 0

    def _connect(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise _invalid_status(404)

    monkeypatch.setattr("newt._client.robot.connect", _connect)
    monkeypatch.setattr("newt._client.robot.time.sleep", lambda _s: None)

    with pytest.raises(newt.EndpointUnavailableError):
        robot.infer({})
    assert calls == 1, "endpoint errors must not be retried"
