"""Offline unit tests for transient verifier-unavailability retry (SDK layer).

The incident: a cold verifier raised VerifierError('The API key verification
service is temporarily unavailable. Retry the request in a few seconds.')
straight through to the user on their first documented call — they needed a
hand-written retry loop (succeeded on attempt 2). The error message prescribes
its own fix; the SDK now does it.

These tests pin the three behaviours the fix must guarantee WITHOUT touching
the network. Each test names what would break if it regressed.

AC mapping:
  AC1 — test_transient_verifier_failure_retries_and_succeeds:
           one/two failures then success → InferenceResponse, warning emitted
  AC2 — test_autherror_never_retried:
           AuthError must propagate immediately, call count == 1
  AC3 — test_budget_exhausted_raises_verifier_error:
           verifier never recovers → VerifierError after ≤4 retries, budget ≤45s

The mock strategy: replace Robot._ws_connect so each "attempt" either raises
VerifierError (transient) or returns a fake WS that replies with a valid
action frame. We also freeze time.sleep so tests run instantly.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import newt
from newt._client.robot import (
    VerifierError,
    VerifierTransientRetry,
    _with_verifier_retry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOOTSTRAP = "https://example.invalid"


def _make_verifier_error(msg: str = "The API key verification service is temporarily unavailable. Retry the request in a few seconds.") -> VerifierError:
    """Build the canonical transient VerifierError the server emits on close 4503."""
    inst = VerifierError.__new__(VerifierError)
    newt.NewTheoryError.__init__(
        inst,
        code=4503,
        type="verifier.unavailable",
        message=msg,
        context={},
        docs=None,
        trace_id="",
    )
    return inst


def _make_action_frame() -> bytes:
    """Minimal valid msgpack action frame for infer() to accept."""
    import msgpack

    chunk = np.zeros((4, 8), dtype=np.float32)
    payload = {
        "type": "action",
        "chunk": {
            b"__ndarray__": True,
            b"data": chunk.tobytes(),
            b"dtype": chunk.dtype.str,
            b"shape": chunk.shape,
        },
    }
    return msgpack.packb(payload, use_bin_type=True)


def _fake_ws_that_returns_action() -> MagicMock:
    """Fake websocket object: send() is a no-op, recv() returns one action frame."""
    ws = MagicMock()
    ws.send.return_value = None
    ws.recv.return_value = _make_action_frame()
    ws.close.return_value = None
    return ws


def _make_robot() -> newt.Robot:
    """Construct a Robot with NT_INFERENCE_URL set so registry discovery is skipped."""
    with patch.dict("os.environ", {"NT_INFERENCE_URL": "wss://fake.invalid/stream"}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", newt.EnvOverrideWarning)
            robot = newt.Robot(api_key="nt_fakekey")
    return robot


# ---------------------------------------------------------------------------
# AC1 — transient failure: retry transparently, warning emitted on first retry
# ---------------------------------------------------------------------------


def test_transient_verifier_failure_retries_and_succeeds(monkeypatch):
    """One transient failure then success → InferenceResponse returned; warning emitted.

    If this regresses: a cold verifier on the first call raises VerifierError
    to the developer instead of being absorbed. They'd need a hand-written retry
    loop — the documented snippet breaks on a cold system.
    """
    robot = _make_robot()

    call_count = 0

    def fake_ws_connect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_verifier_error()
        return _fake_ws_that_returns_action()

    monkeypatch.setattr(robot, "_ws_connect", fake_ws_connect)

    # Freeze sleep so the test runs instantly.
    slept: list[float] = []
    monkeypatch.setattr("newt._client.robot.time.sleep", lambda s: slept.append(s))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = robot.infer({})

    assert isinstance(result, newt.InferenceResponse)
    assert call_count == 2, "SDK must have retried exactly once after the transient failure"

    # Warning emitted exactly once, naming the wait.
    retry_warnings = [w for w in caught if issubclass(w.category, VerifierTransientRetry)]
    assert len(retry_warnings) == 1, "VerifierTransientRetry must be emitted exactly once"
    assert "verifier" in str(retry_warnings[0].message).lower()
    assert "retry" in str(retry_warnings[0].message).lower() or "retrying" in str(retry_warnings[0].message).lower()

    # A sleep happened (backoff was applied).
    assert len(slept) == 1 and slept[0] > 0


def test_two_transient_failures_then_success(monkeypatch):
    """Two transient failures then success → InferenceResponse; warning emitted once.

    The retry budget allows up to 4 retries; two failures is well within budget.
    Warning fires only on the first retry, not on subsequent ones (mirrors
    ColdStartRetry's pattern).
    """
    robot = _make_robot()
    call_count = 0

    def fake_ws_connect():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise _make_verifier_error()
        return _fake_ws_that_returns_action()

    monkeypatch.setattr(robot, "_ws_connect", fake_ws_connect)
    monkeypatch.setattr("newt._client.robot.time.sleep", lambda s: None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = robot.infer({})

    assert isinstance(result, newt.InferenceResponse)
    assert call_count == 3

    retry_warnings = [w for w in caught if issubclass(w.category, VerifierTransientRetry)]
    assert len(retry_warnings) == 1, "Warning must be emitted once, not once per retry"


# ---------------------------------------------------------------------------
# AC2 — definitively-bad key: AuthError propagates immediately, zero retries
# ---------------------------------------------------------------------------


def test_autherror_never_retried(monkeypatch):
    """AuthError on a bad key propagates immediately; call count is exactly 1.

    If this regresses: a developer with a wrong key would wait up to 45s for
    retries before getting the "invalid key" error — they'd think the system is
    broken rather than getting instant, actionable feedback.
    """
    robot = _make_robot()
    call_count = 0

    def fake_ws_connect():
        nonlocal call_count
        call_count += 1
        raise newt.AuthError(
            code=4001,
            type="auth.invalid_key",
            message="Authentication failed: API key rejected.",
            context={"key_prefix": "nt_fake"},
        )

    monkeypatch.setattr(robot, "_ws_connect", fake_ws_connect)
    slept: list[float] = []
    monkeypatch.setattr("newt._client.robot.time.sleep", lambda s: slept.append(s))

    with pytest.raises(newt.AuthError) as exc_info:
        robot.infer({})

    assert call_count == 1, "AuthError must not be retried — zero retry attempts"
    assert len(slept) == 0, "No sleep on auth failure path"
    assert exc_info.value.type == "auth.invalid_key"


# ---------------------------------------------------------------------------
# AC3 — verifier never recovers: VerifierError raised after budget, budget ≤45s
# ---------------------------------------------------------------------------


def test_budget_exhausted_raises_verifier_error(monkeypatch):
    """Verifier never recovers → VerifierError raised after retry budget, original message intact.

    If this regresses: the SDK would loop forever on a permanently-down verifier,
    hanging the developer's process with no way out (no exception, no timeout).
    """
    from newt._client.robot import _VERIFIER_MAX_RETRIES, _VERIFIER_BACKOFF_SECONDS

    robot = _make_robot()
    original_msg = "The API key verification service is temporarily unavailable. Retry the request in a few seconds."
    call_count = 0

    def fake_ws_connect():
        nonlocal call_count
        call_count += 1
        raise _make_verifier_error(original_msg)

    monkeypatch.setattr(robot, "_ws_connect", fake_ws_connect)
    slept: list[float] = []
    monkeypatch.setattr("newt._client.robot.time.sleep", lambda s: slept.append(s))

    with pytest.raises(VerifierError) as exc_info:
        robot.infer({})

    # Must have tried initial + max retries attempts total, then raised.
    assert call_count == _VERIFIER_MAX_RETRIES + 1, (
        f"Expected {_VERIFIER_MAX_RETRIES + 1} attempts total, got {call_count}"
    )

    # Original error message must be preserved — don't rewrite the user's error.
    assert original_msg in str(exc_info.value)
    assert exc_info.value.type == "verifier.unavailable"

    # Budget check: total sleep must be ≤45s (never hangs indefinitely).
    total_sleep = sum(slept)
    assert total_sleep <= 45.0, f"Total backoff {total_sleep}s exceeds 45s budget"

    # Exactly _VERIFIER_MAX_RETRIES sleep calls (one per retry, not one per attempt).
    assert len(slept) == _VERIFIER_MAX_RETRIES


# ---------------------------------------------------------------------------
# Bonus: _with_verifier_retry passes non-verifier VerifierError through immediately
# ---------------------------------------------------------------------------


def test_non_transient_verifier_error_not_retried():
    """A VerifierError with a non-transient type propagates immediately.

    The retry is scoped to type="verifier.unavailable" only. Other verifier
    error types (if the server introduces them) must not be silently swallowed
    by the retry loop.
    """
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        inst = VerifierError.__new__(VerifierError)
        newt.NewTheoryError.__init__(
            inst,
            code=4503,
            type="verifier.config_error",  # not "verifier.unavailable"
            message="Verifier configuration error.",
            context={},
        )
        raise inst

    with pytest.raises(VerifierError) as exc_info:
        _with_verifier_retry(fn)

    assert call_count == 1, "Non-transient VerifierError must not be retried"
    assert exc_info.value.type == "verifier.config_error"


# ---------------------------------------------------------------------------
# SDK audit F1 (2026-07-16): robot.run()'s non-stream path must not spam stdout
# ---------------------------------------------------------------------------


def _make_terminal_frame(stop_reason: str = "task_complete") -> bytes:
    """Minimal valid msgpack terminal frame that ends the run() loop."""
    import msgpack

    return msgpack.packb({"type": "terminal", "stop_reason": stop_reason}, use_bin_type=True)


def test_run_blocking_produces_no_stdout_output(monkeypatch, capsys):
    """robot.run() must drive frames without printing anything to stdout.

    Before the fix, _run_blocking_once had four unconditional print() calls
    ('[newt debug] frame N: sending X bytes', 'send returned...', etc.) that
    fired every frame — a team driving a real arm at 30Hz got their terminal
    flooded with debug chatter they never asked for and had no way to turn off
    (SDK audit F1, 2026-07-16). The fix routes them through the `newt` logger
    at DEBUG/INFO, which produces no output without a handler configured — the
    library never calls basicConfig. This test pins that: frames still flow
    (read_state/execute called, a real stop_reason returned) while stdout (and
    stderr, since logging's lastResort handler only fires at WARNING+) stay
    clean.
    """
    read_state_calls = 0
    execute_calls: list[np.ndarray] = []

    def read_state() -> dict:
        nonlocal read_state_calls
        read_state_calls += 1
        return {"state": np.zeros(14, dtype=np.float32)}

    def execute(chunk: np.ndarray) -> None:
        execute_calls.append(chunk)

    with patch.dict("os.environ", {"NT_INFERENCE_URL": "wss://fake.invalid/stream"}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", newt.EnvOverrideWarning)
            robot = newt.Robot(
                api_key="nt_fakekey",
                read_state=read_state,
                execute=execute,
            )

    ws = MagicMock()
    ws.send.return_value = None
    # One action frame, then a terminal frame to end the loop deterministically.
    ws.recv.side_effect = [_make_action_frame(), _make_terminal_frame()]
    ws.close.return_value = None
    monkeypatch.setattr(robot, "_ws_connect", lambda: ws)

    result = robot.run("pick up the cup", max_duration=5.0)

    assert read_state_calls >= 1, "read_state() must still be called"
    assert len(execute_calls) >= 1, "execute() must still be called with the action chunk"
    assert result.stop_reason == "task_complete"

    captured = capsys.readouterr()
    assert captured.out == "", f"run() must not print to stdout; got {captured.out!r}"
    assert captured.err == "", f"run() must not print to stderr; got {captured.err!r}"
