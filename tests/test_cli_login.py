"""Offline unit tests for `newt login` CLI command.

Tests cover the three acceptance criteria WITHOUT touching the network or
waiting 10 minutes:

  AC1 — bounded poll: a pairing never confirmed terminates at the TTL bound,
        exits non-zero, and prints the expiry message naming both `newt login`
        and NT_API_KEY.

  AC2 — immediate exit on server-signalled expiry: a poll response that
        indicates the pairing is expired/gone causes immediate exit with the
        same message and no further poll calls.

  AC3 — happy path regression: confirm-on-second-poll still returns the key
        and writes credentials exactly as before.

TTL source (AC4): the server currently does NOT return expires_in / expires_at
(documented in the module docstring).  The 10-minute fallback is always active
in production.  The deadline derivation is tested here by injecting a short
deadline via monkeypatching time.monotonic.
"""
from __future__ import annotations

import io
import json
import sys
from typing import Iterator
from unittest.mock import MagicMock, call, patch

import pytest

from newt._cli.login import _EXPIRY_MSG, cmd_login


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_START = {
    "browser_url": "https://console.newtheory.ai/cli-auth/AAAA",
    "user_code": "AAAA-BBBB",
    "poll_url": "https://console.newtheory.ai/api/cli/auth/poll/token-123",
}

_FAKE_KEY = "nt_" + "a" * 40


def _start_resp_bytes(extra: dict | None = None) -> bytes:
    payload = dict(_FAKE_START)
    if extra:
        payload.update(extra)
    return json.dumps(payload).encode()


def _poll_resp_bytes(status: str, key: str | None = None) -> bytes:
    payload: dict = {"status": status}
    if key is not None:
        payload["key"] = key
    return json.dumps(payload).encode()


class _FakeHTTPResp:
    """Minimal file-like object returned by urlopen context manager."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResp":
        return self

    def __exit__(self, *_) -> None:
        pass


# ---------------------------------------------------------------------------
# AC1 — bounded poll terminates at TTL; message names newt login + NT_API_KEY
# ---------------------------------------------------------------------------

def test_deadline_expiry_exits_nonzero_with_correct_message(monkeypatch):
    """A pairing that is never confirmed must exit non-zero at the TTL deadline.

    This is the CI/agent safety guard: without a deadline, the CLI hangs
    forever in environments where nobody will ever open a browser.

    The test injects a tiny deadline (via mocked time.monotonic) so the loop
    exits after the first sleep, no real waiting required.
    """
    # Monotonic sequence: first call sets 'now' for deadline calc; subsequent
    # calls (inside the while condition) return a value past the deadline so
    # the loop exits immediately after one iteration.
    #
    # The code path is:
    #   now = time.monotonic()          → tick 0
    #   deadline = now + _MAX_WAIT_S
    #   while time.monotonic() < deadline:   → tick 1 (past deadline → exits)
    tick = iter([0.0, 10 * 60 + 1.0])
    monkeypatch.setattr("newt._cli.login.time.monotonic", lambda: next(tick))
    monkeypatch.setattr("newt._cli.login.time.sleep", lambda _: None)
    monkeypatch.setattr("newt._cli.login.webbrowser.open", lambda _: False)

    urlopen_calls: list[str] = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        urlopen_calls.append(url)
        return _FakeHTTPResp(_start_resp_bytes())

    monkeypatch.setattr("newt._cli.login.urlopen", fake_urlopen)

    captured_err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured_err)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    rc = cmd_login([])

    assert rc != 0, "must exit non-zero when deadline passes without confirmation"

    err = captured_err.getvalue()
    assert "newt login" in err, f"expiry message must name 'newt login': {err!r}"
    assert "NT_API_KEY" in err, (
        f"expiry message must name NT_API_KEY for headless environments: {err!r}"
    )
    # The exact canonical message must appear
    assert _EXPIRY_MSG in err, f"full expiry message not in stderr: {err!r}"


def test_deadline_expiry_no_poll_calls_after_deadline(monkeypatch):
    """After the deadline passes the loop must not issue any further poll calls.

    This verifies the guard exits the loop rather than making one extra
    network call per iteration.
    """
    tick = iter([0.0, 10 * 60 + 1.0])
    monkeypatch.setattr("newt._cli.login.time.monotonic", lambda: next(tick))
    monkeypatch.setattr("newt._cli.login.time.sleep", lambda _: None)
    monkeypatch.setattr("newt._cli.login.webbrowser.open", lambda _: False)

    poll_calls = 0

    def fake_urlopen(req, timeout=None):
        nonlocal poll_calls
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "poll" in url:
            poll_calls += 1
        return _FakeHTTPResp(_start_resp_bytes())

    monkeypatch.setattr("newt._cli.login.urlopen", fake_urlopen)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    cmd_login([])

    assert poll_calls == 0, (
        f"no poll calls should happen when the deadline is already passed: got {poll_calls}"
    )


def test_expires_in_from_start_response_used_as_deadline(monkeypatch):
    """When the start response contains expires_in, it sets the deadline.

    The server currently does NOT return this field (fallback always applies
    in production). This test documents the contract so that when the server
    adds it the client picks it up correctly.
    """
    # expires_in = 5 seconds.  Monotonic sequence: tick 0 sets 'now = 0.0',
    # deadline = 5.0.  Next monotonic check returns 6.0 → past deadline.
    tick = iter([0.0, 6.0])
    monkeypatch.setattr("newt._cli.login.time.monotonic", lambda: next(tick))
    monkeypatch.setattr("newt._cli.login.time.sleep", lambda _: None)
    monkeypatch.setattr("newt._cli.login.webbrowser.open", lambda _: False)

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResp(_start_resp_bytes({"expires_in": 5}))

    monkeypatch.setattr("newt._cli.login.urlopen", fake_urlopen)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    rc = cmd_login([])

    assert rc != 0
    # The deadline came from the server's 5-second expires_in, not _MAX_WAIT_S.
    # If the code ignored expires_in it would need tick[1] = 601 to expire.


# ---------------------------------------------------------------------------
# AC2 — server-signalled expiry causes immediate exit; no further poll calls
# ---------------------------------------------------------------------------

def test_poll_expired_status_exits_immediately(monkeypatch):
    """A poll response with status='expired' causes immediate exit with the
    expiry message.

    This covers the case where the server is ahead of our local deadline
    (e.g. server-side cleanup ran early) and tells us explicitly.
    """
    monkeypatch.setattr("newt._cli.login.time.monotonic", lambda: 0.0)
    monkeypatch.setattr("newt._cli.login.time.sleep", lambda _: None)
    monkeypatch.setattr("newt._cli.login.webbrowser.open", lambda _: False)

    poll_calls = 0

    def fake_urlopen(req, timeout=None):
        nonlocal poll_calls
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "poll" in url:
            poll_calls += 1
            return _FakeHTTPResp(_poll_resp_bytes("expired"))
        return _FakeHTTPResp(_start_resp_bytes())

    monkeypatch.setattr("newt._cli.login.urlopen", fake_urlopen)

    captured_err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured_err)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    rc = cmd_login([])

    assert rc != 0, "must exit non-zero on expired status"
    err = captured_err.getvalue()
    assert _EXPIRY_MSG in err, f"expiry message must appear: {err!r}"
    assert poll_calls == 1, (
        f"must stop after the first expired response, got {poll_calls} poll calls"
    )


def test_poll_http_410_exits_immediately(monkeypatch):
    """An HTTP 410 from the poll endpoint means the pairing is gone server-side.

    Must exit immediately with the expiry message — not keep polling until
    the local deadline.
    """
    from urllib.error import HTTPError

    monkeypatch.setattr("newt._cli.login.time.monotonic", lambda: 0.0)
    monkeypatch.setattr("newt._cli.login.time.sleep", lambda _: None)
    monkeypatch.setattr("newt._cli.login.webbrowser.open", lambda _: False)

    poll_calls = 0

    def fake_urlopen(req, timeout=None):
        nonlocal poll_calls
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "poll" in url:
            poll_calls += 1
            raise HTTPError(url, 410, "Gone", {}, None)
        return _FakeHTTPResp(_start_resp_bytes())

    monkeypatch.setattr("newt._cli.login.urlopen", fake_urlopen)

    captured_err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured_err)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    rc = cmd_login([])

    assert rc != 0
    err = captured_err.getvalue()
    assert _EXPIRY_MSG in err, f"expiry message must appear on 410: {err!r}"
    assert poll_calls == 1, (
        f"must stop after first 410, got {poll_calls} poll calls"
    )


# ---------------------------------------------------------------------------
# AC3 — happy path regression: confirm on second poll writes key and exits 0
# ---------------------------------------------------------------------------

def test_happy_path_confirm_on_second_poll_writes_key(monkeypatch, tmp_path):
    """The happy path is unchanged: a key arrives on the second poll, gets
    written to credentials, and the command exits 0.

    This is a regression guard — if the deadline or expiry changes accidentally
    affect the success branch, this test catches it.
    """
    from newt._credentials import CREDENTIALS_DIR, CREDENTIALS_PATH

    # Redirect credentials write to tmp_path
    fake_creds_dir = tmp_path / ".nt"
    fake_creds_path = fake_creds_dir / "credentials"
    monkeypatch.setattr("newt._credentials.CREDENTIALS_DIR", fake_creds_dir)
    monkeypatch.setattr("newt._credentials.CREDENTIALS_PATH", fake_creds_path)

    # Monotonic: tick 0 = now for deadline, subsequent ticks well within window.
    tick = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr("newt._cli.login.time.monotonic", lambda: next(tick))
    monkeypatch.setattr("newt._cli.login.time.sleep", lambda _: None)
    monkeypatch.setattr("newt._cli.login.webbrowser.open", lambda _: False)

    poll_calls = 0

    def fake_urlopen(req, timeout=None):
        nonlocal poll_calls
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "poll" in url:
            poll_calls += 1
            if poll_calls == 1:
                return _FakeHTTPResp(_poll_resp_bytes("pending"))
            return _FakeHTTPResp(_poll_resp_bytes("confirmed", key=_FAKE_KEY))
        return _FakeHTTPResp(_start_resp_bytes())

    monkeypatch.setattr("newt._cli.login.urlopen", fake_urlopen)

    captured_out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    rc = cmd_login([])

    assert rc == 0, "happy path must exit 0"
    out = captured_out.getvalue()
    assert "Logged in successfully" in out, f"success message not in stdout: {out!r}"

    # Key must be written to the credentials file
    assert fake_creds_path.exists(), "credentials file must be written"
    creds_content = fake_creds_path.read_text()
    assert _FAKE_KEY in creds_content, (
        f"key must appear in credentials file: {creds_content!r}"
    )

    assert poll_calls == 2, (
        f"expected exactly 2 poll calls (pending then confirmed), got {poll_calls}"
    )
