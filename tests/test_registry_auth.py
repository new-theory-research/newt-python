"""Offline unit tests for registry-discovery error mapping (brief-229).

The incident: a wrong-format key (an `ak_…` key from Clerk's console-profile panel
instead of an `nt_…` key) reached `/v1/models`, the server wrapped the console's 400
as a 503, and the SDK mapped any 5xx on `/v1/models` to RegistryUnavailable. A
rejected key read as a registry OUTAGE — sending a beta tester (and us) chasing a
phantom infra problem for ~30 min.

These tests pin the fix at the SDK boundary WITHOUT touching the network: they
simulate exactly what the server returns and assert which exception the developer
sees. Each test maps to one rung of the repro table in the brief.

Why these matter (not just what they do):
  - A 401 (the server's post-fix response for a bad/wrong-format key) MUST become
    AuthError, never RegistryUnavailable. If this regresses, "your key is wrong"
    silently becomes "our registry is down" again — the exact misdiagnosis.
  - A genuine 5xx (verifier actually down) MUST stay RegistryUnavailable. The fix is
    a discrimination, not a blanket reclassification — losing the outage path would
    mask real outages as auth failures, the opposite failure.
  - The AuthError message must carry the key-format hint so the developer learns the
    nt_/ak_ distinction without a support round-trip.
"""
from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest

import newt
from newt._client.robot import RegistryUnavailable, _fetch_registry

_BOOTSTRAP = "https://example.invalid"


def _http_error(code: int, detail: str | None = None) -> HTTPError:
    """Build an HTTPError that mimics urllib's view of a FastAPI error response.

    urllib raises HTTPError as a file-like object; _http_error_detail() calls
    .read() on it to recover the JSON body. We replicate that: a JSON
    {"detail": ...} body when detail is given, empty otherwise.
    """
    body = json.dumps({"detail": detail}).encode() if detail is not None else b""
    return HTTPError(
        url=_BOOTSTRAP + "/v1/models",
        code=code,
        msg="error",
        hdrs=None,
        fp=io.BytesIO(body),
    )


def _patch_urlopen(monkeypatch, raiser):
    """Make urllib.request.urlopen (as imported inside _fetch_registry) raise `raiser`."""
    import urllib.request

    def fake_urlopen(*args, **kwargs):
        raise raiser

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


# --- The core discrimination: 401 → AuthError, 5xx → RegistryUnavailable ---------


def test_console_400_rejected_key_yields_autherror_not_registry_unavailable(monkeypatch):
    """Server 401 (console rejected the key) → AuthError. The headline regression.

    Post-fix, a console 4xx becomes a server 401 carrying the key-format hint. The
    SDK must raise AuthError — NOT RegistryUnavailable. This is the exact path that
    made a bad `ak_` key look like a registry outage.
    """
    detail = (
        "API key rejected. Keys start with nt_; an ak_ key comes from the wrong "
        "flow — create one in the console Create-key flow."
    )
    _patch_urlopen(monkeypatch, _http_error(401, detail))

    with pytest.raises(newt.AuthError) as exc_info:
        _fetch_registry(_BOOTSTRAP, "ak_000deadbeef")

    # Must NOT be a registry-outage error.
    assert not isinstance(exc_info.value, RegistryUnavailable)
    # The server's hint must reach the developer verbatim.
    assert "nt_" in str(exc_info.value)
    assert "ak_" in str(exc_info.value)
    assert exc_info.value.type == "auth.invalid_key"


def test_console_5xx_genuine_outage_stays_registry_unavailable(monkeypatch):
    """Server 503 (verifier genuinely down) → RegistryUnavailable. The outage path.

    The fix must NOT swallow real outages into AuthError. A 5xx on /v1/models still
    means "couldn't reach the registry" — the developer should retry, not rotate keys.
    """
    _patch_urlopen(monkeypatch, _http_error(503, "Auth verifier unavailable: timeout"))

    with pytest.raises(RegistryUnavailable) as exc_info:
        _fetch_registry(_BOOTSTRAP, "nt_realkey")

    assert not isinstance(exc_info.value, newt.AuthError)
    assert exc_info.value.type == "registry.unavailable"


def test_connection_error_stays_registry_unavailable(monkeypatch):
    """A transport-level failure (verifier unreachable) → RegistryUnavailable.

    URLError covers DNS/connection-refused/timeout — genuinely "can't reach the
    registry," distinct from "your key is bad." Must not become AuthError.
    """
    _patch_urlopen(monkeypatch, URLError("connection refused"))

    with pytest.raises(RegistryUnavailable):
        _fetch_registry(_BOOTSTRAP, "nt_realkey")


# --- The key-format hint, even without a server body -----------------------------


def test_autherror_hint_derived_locally_when_server_body_absent(monkeypatch):
    """A 401 with no body still names the nt_/ak_ distinction for a wrong-format key.

    Defense in depth: if the server's detail doesn't arrive (older server, stripped
    body), the SDK derives the hint from the key the developer actually sent. A bad
    key must never read as a bare "authentication failed" with no next action.
    """
    _patch_urlopen(monkeypatch, _http_error(401, detail=None))

    with pytest.raises(newt.AuthError) as exc_info:
        _fetch_registry(_BOOTSTRAP, "ak_000deadbeef")

    msg = str(exc_info.value)
    assert "nt_" in msg
    assert "ak_" in msg


def test_autherror_for_valid_prefix_key_does_not_misattribute_format(monkeypatch):
    """An `nt_`-prefixed key that's still rejected (401, no body) → rotate-key hint.

    A revoked/invalid nt_ key is a different story from a wrong-format key — the hint
    must not falsely tell the developer their key format is wrong when it isn't.
    """
    _patch_urlopen(monkeypatch, _http_error(401, detail=None))

    with pytest.raises(newt.AuthError) as exc_info:
        _fetch_registry(_BOOTSTRAP, "nt_000revoked")

    msg = str(exc_info.value)
    assert "Rotate" in msg
    # Must not claim the key is the wrong format — it has the right prefix.
    assert "wrong flow" not in msg
