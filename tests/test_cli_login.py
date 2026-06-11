"""Offline unit tests for `newt login` CLI command.

Tests cover the TTL acceptance criteria WITHOUT touching the network or
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

--print tests cover the developer-visible contract WITHOUT touching the
network.  Each test encodes WHY the behavior matters, not just WHAT it does.
The mock structure replaces `urlopen` so both the /start POST and the
/poll GET are intercepted.  A minimal fake response object matches the
interface urlopen returns (read(), __enter__, __exit__).
"""
from __future__ import annotations

import io
import json
import sys
from unittest.mock import MagicMock

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


_START_RESPONSE = {
    "browser_url": "https://console.newtheory.ai/pair?code=WXYZ",
    "user_code": "WXYZ",
    "poll_url": "https://console.newtheory.ai/api/cli/auth/poll/abc123",
}

_CONFIRMED_RESPONSE = {
    "status": "confirmed",
    "key": _FAKE_KEY,
}

_PENDING_RESPONSE = {"status": "pending"}


def _make_response(payload: dict) -> MagicMock:
    """Return a mock that behaves like urlopen()'s context-manager response."""
    m = MagicMock()
    m.read.return_value = json.dumps(payload).encode()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    return m


def _run(args: list[str], monkeypatch, urlopen_side_effect=None) -> tuple[int, str, str]:
    """
    Run cmd_login with:
    - webbrowser.open patched to no-op (never open a real browser)
    - time.sleep patched to no-op (don't actually wait)
    - urlopen replaced by urlopen_side_effect (a list of return values or exceptions)
    - stdout/stderr captured
    """
    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    call_count = [0]

    def fake_urlopen(req, timeout=None):
        idx = call_count[0]
        call_count[0] += 1
        if urlopen_side_effect is None:
            raise RuntimeError("urlopen_side_effect not set")
        val = urlopen_side_effect[idx]
        if isinstance(val, BaseException):
            raise val
        return val

    monkeypatch.setattr("newt._cli.login.urlopen", fake_urlopen)
    monkeypatch.setattr("newt._cli.login.webbrowser.open", lambda url: False)
    monkeypatch.setattr("newt._cli.login.time.sleep", lambda s: None)

    exit_code = cmd_login(args)
    return exit_code, out.getvalue(), err.getvalue()


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


# ---------------------------------------------------------------------------
# --print AC1: --print routes instructions to stderr; bare key goes to stdout only
# ---------------------------------------------------------------------------

def test_print_flag_key_on_stdout_only(monkeypatch, tmp_path):
    """With --print, stdout is EXACTLY the key (+ trailing newline) — nothing else.

    Composability is the contract: KEY=$(newt login --print) must capture only the
    key string.  The URL, user-code, status lines belong on stderr so the human/agent
    watching the terminal still sees them.
    """
    responses = [
        _make_response(_START_RESPONSE),
        _make_response(_CONFIRMED_RESPONSE),
    ]
    monkeypatch.setenv("HOME", str(tmp_path))

    exit_code, out, err = _run(["--print"], monkeypatch, urlopen_side_effect=responses)

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    assert out.strip() == _FAKE_KEY, (
        f"stdout must be exactly the key; got: {out!r}"
    )
    # Instructions must have gone to stderr, not stdout
    assert _START_RESPONSE["browser_url"] in err, (
        f"browser URL must appear on stderr: {err!r}"
    )
    assert _START_RESPONSE["user_code"] in err, (
        f"user code must appear on stderr: {err!r}"
    )
    # URL and code must NOT appear on stdout
    assert _START_RESPONSE["browser_url"] not in out, (
        f"browser URL must NOT appear on stdout: {out!r}"
    )
    assert _START_RESPONSE["user_code"] not in out, (
        f"user code must NOT appear on stdout: {out!r}"
    )


# ---------------------------------------------------------------------------
# --print AC2a: --print does not call write_api_key (credentials file never touched)
# ---------------------------------------------------------------------------

def test_print_flag_does_not_write_credentials(monkeypatch):
    """With --print, write_api_key is never called.

    The whole point of --print is scripting: the caller manages the key.
    Writing to disk would silently contradict the contract.
    """
    write_calls: list[str] = []
    monkeypatch.setattr("newt._cli.login.write_api_key", lambda key: write_calls.append(key))

    responses = [
        _make_response(_START_RESPONSE),
        _make_response(_CONFIRMED_RESPONSE),
    ]
    exit_code, out, err = _run(["--print"], monkeypatch, urlopen_side_effect=responses)

    assert exit_code == 0
    assert write_calls == [], (
        f"write_api_key must NOT be called with --print; called with: {write_calls}"
    )


# ---------------------------------------------------------------------------
# --print AC2b: --print does not modify an EXISTING credentials file
# ---------------------------------------------------------------------------

def test_print_flag_does_not_modify_existing_credentials(monkeypatch, tmp_path):
    """With --print, an existing credentials file is left byte-identical.

    An agent or script may share a machine with a human who already logged in.
    --print must not overwrite or corrupt their credentials.
    We test this by verifying write_api_key is never called — since _credentials.py
    paths are module-level constants computed at import time, write_api_key is the
    only path that mutates the file.
    """
    write_calls: list[str] = []
    monkeypatch.setattr("newt._cli.login.write_api_key", lambda key: write_calls.append(key))

    responses = [
        _make_response(_START_RESPONSE),
        _make_response(_CONFIRMED_RESPONSE),
    ]
    exit_code, out, err = _run(["--print"], monkeypatch, urlopen_side_effect=responses)

    assert exit_code == 0
    assert write_calls == [], (
        f"write_api_key must NOT be called with --print; called with: {write_calls}"
    )


# ---------------------------------------------------------------------------
# --print AC3: without --print, behavior is unchanged
# ---------------------------------------------------------------------------

def test_normal_login_writes_credentials(monkeypatch):
    """Without --print, a successful pairing calls write_api_key with the key.

    This is the happy path every developer hits on first setup.
    We verify write_api_key is called (not the file itself) since CREDENTIALS_PATH
    is a module-level constant resolved at import time.
    """
    write_calls: list[str] = []
    monkeypatch.setattr("newt._cli.login.write_api_key", lambda key: write_calls.append(key))

    responses = [
        _make_response(_START_RESPONSE),
        _make_response(_CONFIRMED_RESPONSE),
    ]
    exit_code, out, err = _run([], monkeypatch, urlopen_side_effect=responses)

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    assert write_calls == [_FAKE_KEY], (
        f"write_api_key must be called with the key; calls: {write_calls}"
    )


def test_normal_login_key_prefix_in_stdout(monkeypatch, tmp_path):
    """Without --print, the success message with key prefix appears on stdout.

    The developer must see confirmation — the key prefix lets them cross-check
    without exposing the full secret.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    responses = [
        _make_response(_START_RESPONSE),
        _make_response(_CONFIRMED_RESPONSE),
    ]
    exit_code, out, err = _run([], monkeypatch, urlopen_side_effect=responses)

    assert exit_code == 0
    # Key prefix (first 12 chars) must appear in stdout
    assert _FAKE_KEY[:12] in out, (
        f"key prefix must appear in stdout success message: {out!r}"
    )


def test_normal_login_ttl_expiry(monkeypatch, tmp_path):
    """Without --print, a 410 during polling exits non-zero with the expiry message.

    The 10-minute TTL is a server-enforced security boundary.  The developer must
    know to re-run `newt login` rather than keep waiting.
    """
    from urllib.error import HTTPError

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    gone = HTTPError(
        url="https://console.newtheory.ai/api/cli/auth/poll/abc123",
        code=410,
        msg="Gone",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )
    responses = [
        _make_response(_START_RESPONSE),
        gone,
    ]
    exit_code, out, err = _run([], monkeypatch, urlopen_side_effect=responses)

    assert exit_code != 0, "expired pairing must exit non-zero"
    assert "expired" in err.lower() or "10-minute" in err.lower(), (
        f"expiry message must appear on stderr: {err!r}"
    )


# ---------------------------------------------------------------------------
# --print AC4: --print + TTL expiry → non-zero exit, empty stdout
# ---------------------------------------------------------------------------

def test_print_flag_ttl_expiry_empty_stdout(monkeypatch, tmp_path):
    """With --print, a 410 expiry exits non-zero and stdout is empty.

    Composability contract: KEY=$(newt login --print) must never capture
    anything key-like in a failure mode.  An empty stdout tells the caller
    unambiguously that no key was obtained.
    """
    from urllib.error import HTTPError

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    gone = HTTPError(
        url="https://console.newtheory.ai/api/cli/auth/poll/abc123",
        code=410,
        msg="Gone",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )
    responses = [
        _make_response(_START_RESPONSE),
        gone,
    ]
    exit_code, out, err = _run(["--print"], monkeypatch, urlopen_side_effect=responses)

    assert exit_code != 0, "expired pairing must exit non-zero with --print"
    assert out == "", (
        f"stdout must be empty on failure with --print; got: {out!r}"
    )
    assert "expired" in err.lower() or "10-minute" in err.lower(), (
        f"expiry message must appear on stderr: {err!r}"
    )


# ---------------------------------------------------------------------------
# --print AC4 extension: --print + pending → confirmed after one pending poll
# ---------------------------------------------------------------------------

def test_print_flag_pending_then_confirmed(monkeypatch, tmp_path):
    """With --print, a pending poll followed by confirm still delivers the key on stdout.

    The poll loop must keep going through pending responses and only exit at confirmed.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    responses = [
        _make_response(_START_RESPONSE),
        _make_response(_PENDING_RESPONSE),
        _make_response(_CONFIRMED_RESPONSE),
    ]
    exit_code, out, err = _run(["--print"], monkeypatch, urlopen_side_effect=responses)

    assert exit_code == 0
    assert out.strip() == _FAKE_KEY


# ---------------------------------------------------------------------------
# --print AC5: "key not saved" stderr note appears with --print on success
# ---------------------------------------------------------------------------

def test_print_flag_key_not_saved_note_on_stderr(monkeypatch, tmp_path):
    """With --print, the success path prints a 'key not saved' note to stderr.

    Without the note, an agent watching stderr would see no signal that persistence
    was intentionally skipped — they'd be left wondering if something went wrong.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    responses = [
        _make_response(_START_RESPONSE),
        _make_response(_CONFIRMED_RESPONSE),
    ]
    exit_code, out, err = _run(["--print"], monkeypatch, urlopen_side_effect=responses)

    assert exit_code == 0
    assert "not saved" in err.lower() or "key not saved" in err.lower(), (
        f"must emit 'key not saved' note on stderr: {err!r}"
    )


# ---------------------------------------------------------------------------
# New AC1 — self-contained waiting line: URL present, no "above" reference
# ---------------------------------------------------------------------------

def test_waiting_line_is_self_contained(monkeypatch):
    """The 'Waiting' line includes the full confirmation URL so it's readable
    even if the preceding URL block is hidden (folded terminal, late attach).

    AC: waiting line contains browser_url; no output line anywhere says 'above'.
    """
    responses = [
        _make_response(_START_RESPONSE),
        _make_response(_CONFIRMED_RESPONSE),
    ]

    write_calls: list[str] = []
    monkeypatch.setattr("newt._cli.login.write_api_key", lambda key: write_calls.append(key))

    exit_code, out, err = _run([], monkeypatch, urlopen_side_effect=responses)

    combined = out + err
    browser_url = _START_RESPONSE["browser_url"]
    user_code = _START_RESPONSE["user_code"]

    # Find the waiting line (starts with "Waiting")
    waiting_lines = [l for l in combined.splitlines() if "Waiting" in l or "waiting" in l.lower()]
    assert waiting_lines, f"no 'Waiting' line found in output: {combined!r}"
    waiting_line = waiting_lines[0]
    assert browser_url in waiting_line, (
        f"waiting line must contain the full URL; got: {waiting_line!r}"
    )

    # The URL is present in the waiting line — the code is in the URL as a query param
    # or in adjacent text; assert that the URL (which carries the code) is self-contained
    assert user_code in waiting_line or user_code in browser_url, (
        f"waiting line or URL must reference the user code: {waiting_line!r}"
    )

    # No line in the entire output should direct the user to look 'above'
    for line in combined.splitlines():
        assert "above" not in line.lower(), (
            f"output must never reference 'above' (stranded output risk): {line!r}"
        )


# ---------------------------------------------------------------------------
# New AC2 — periodic re-emit during polling: URL+code re-printed at ~30s cadence
# ---------------------------------------------------------------------------

def test_periodic_reemit_during_polling(monkeypatch):
    """While polling, a self-contained URL+code line is re-emitted every ~30 seconds.

    Re-emit protects users whose terminal scrollback was lost after the initial
    block was printed — they can recover the URL without restarting the flow.

    The test advances a fake monotonic clock and counts re-emit lines without
    altering the poll call cadence (_POLL_INTERVAL_S unchanged).
    """
    from newt._cli.login import _POLL_INTERVAL_S, _REEMIT_INTERVAL_S

    reemit_every = max(1, int(_REEMIT_INTERVAL_S / _POLL_INTERVAL_S))

    # Simulate enough iterations for 2 re-emits, then confirm.
    # We need at least 2 * reemit_every poll iterations before confirming.
    total_pending = reemit_every * 2  # triggers re-emit at iteration reemit_every and 2*reemit_every

    # Build monotonic ticks: tick 0 = start, subsequent ticks stay well within deadline
    tick_values = [0.0] + [float(i) for i in range(1, total_pending + 10)]
    tick_iter = iter(tick_values)
    monkeypatch.setattr("newt._cli.login.time.monotonic", lambda: next(tick_iter))
    monkeypatch.setattr("newt._cli.login.time.sleep", lambda _: None)
    monkeypatch.setattr("newt._cli.login.webbrowser.open", lambda _: False)
    monkeypatch.setattr("newt._cli.login.write_api_key", lambda key: None)

    call_count = [0]
    browser_url = _START_RESPONSE["browser_url"]
    user_code = _START_RESPONSE["user_code"]

    def fake_urlopen(req, timeout=None):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            return _make_response(_START_RESPONSE)
        # Return pending for the first total_pending polls, then confirmed
        if idx <= total_pending:
            return _make_response(_PENDING_RESPONSE)
        return _make_response(_CONFIRMED_RESPONSE)

    monkeypatch.setattr("newt._cli.login.urlopen", fake_urlopen)

    out = __import__("io").StringIO()
    err = __import__("io").StringIO()
    monkeypatch.setattr(__import__("sys"), "stdout", out)
    monkeypatch.setattr(__import__("sys"), "stderr", err)

    exit_code = __import__("newt._cli.login", fromlist=["cmd_login"]).cmd_login([])

    assert exit_code == 0, f"expected success; stderr={err.getvalue()!r}"

    combined = out.getvalue() + err.getvalue()

    # Count re-emit lines (the "Still waiting" lines)
    reemit_lines = [l for l in combined.splitlines() if "Still waiting" in l]
    assert len(reemit_lines) >= 2, (
        f"expected at least 2 re-emit lines at ~30s cadence; got {len(reemit_lines)}: {reemit_lines}"
    )

    # Each re-emit line must be self-contained: URL and code both present
    for line in reemit_lines:
        assert browser_url in line, f"re-emit line must contain URL: {line!r}"
        assert user_code in line, f"re-emit line must contain code: {line!r}"

    # Poll call count: 1 (start) + total_pending (pending) + 1 (confirmed) = total_pending + 2
    expected_polls = total_pending  # poll calls only (start call is index 0)
    actual_polls = call_count[0] - 1  # subtract the start call
    assert actual_polls == total_pending + 1, (
        f"poll timing must be unchanged: expected {total_pending + 1} poll calls, got {actual_polls}"
    )


# ---------------------------------------------------------------------------
# New AC3 — no-browser fallback is self-contained with agent signpost
# ---------------------------------------------------------------------------

def test_no_browser_fallback_self_contained_with_signpost(monkeypatch):
    """When no browser opens, the fallback message names the URL inline (no
    'above') and includes the agent signpost mentioning both `newt login --print`
    and NT_API_KEY.
    """
    responses = [
        _make_response(_START_RESPONSE),
        _make_response(_CONFIRMED_RESPONSE),
    ]
    monkeypatch.setattr("newt._cli.login.write_api_key", lambda key: None)
    # webbrowser.open returns False → no-browser branch
    monkeypatch.setattr("newt._cli.login.webbrowser.open", lambda _: False)

    out = __import__("io").StringIO()
    err = __import__("io").StringIO()
    monkeypatch.setattr(__import__("sys"), "stdout", out)
    monkeypatch.setattr(__import__("sys"), "stderr", err)
    monkeypatch.setattr("newt._cli.login.time.sleep", lambda _: None)

    call_count = [0]
    def fake_urlopen(req, timeout=None):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            return _make_response(_START_RESPONSE)
        return _make_response(_CONFIRMED_RESPONSE)
    monkeypatch.setattr("newt._cli.login.urlopen", fake_urlopen)

    exit_code = __import__("newt._cli.login", fromlist=["cmd_login"]).cmd_login([])

    combined = out.getvalue() + err.getvalue()
    browser_url = _START_RESPONSE["browser_url"]

    # Find the no-browser line
    no_browser_lines = [l for l in combined.splitlines() if "No browser" in l]
    assert no_browser_lines, f"no 'No browser' line found: {combined!r}"
    no_browser_line = no_browser_lines[0]

    # Must name the URL inline
    assert browser_url in no_browser_line, (
        f"no-browser line must contain the URL: {no_browser_line!r}"
    )
    # Must not say 'above'
    assert "above" not in no_browser_line.lower(), (
        f"no-browser line must not reference 'above': {no_browser_line!r}"
    )

    # Signpost line must mention both `newt login --print` and NT_API_KEY
    assert "newt login --print" in combined, (
        f"signpost must mention `newt login --print`: {combined!r}"
    )
    assert "NT_API_KEY" in combined, (
        f"signpost must mention NT_API_KEY: {combined!r}"
    )


# ---------------------------------------------------------------------------
# New AC4 — --print mode: new lines route to stderr; stdout stays bare key only
# ---------------------------------------------------------------------------

def test_print_mode_new_lines_route_to_stderr(monkeypatch):
    """With --print, the new self-contained waiting line and no-browser signpost
    must appear on stderr, not stdout. Stdout must remain exactly the bare key.
    """
    responses = [
        _make_response(_START_RESPONSE),
        _make_response(_CONFIRMED_RESPONSE),
    ]

    exit_code, out, err = _run(["--print"], monkeypatch, urlopen_side_effect=responses)

    assert exit_code == 0, f"expected success; err={err!r}"

    # stdout: exactly the bare key
    assert out.strip() == _FAKE_KEY, (
        f"stdout must be exactly the key in --print mode; got: {out!r}"
    )

    browser_url = _START_RESPONSE["browser_url"]

    # The waiting line (with URL) and signpost must appear on stderr
    assert browser_url in err, (
        f"browser URL must appear on stderr in --print mode: {err!r}"
    )
    assert "Waiting" in err, (
        f"waiting line must appear on stderr in --print mode: {err!r}"
    )

    # Nothing from our new lines should land on stdout
    assert "Waiting" not in out, (
        f"'Waiting' line must not appear on stdout: {out!r}"
    )
    assert "No browser" not in out, (
        f"no-browser line must not appear on stdout: {out!r}"
    )
