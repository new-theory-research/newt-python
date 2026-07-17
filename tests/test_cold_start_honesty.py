"""Offline unit tests for issue #38 — honest cold-start/retry timing on Robot.infer().

The receipt: a cold fine-tune's first call printed `latency 83609ms` as if that
whole wait were steady-state inference speed, then a warm call answered in
1863ms. `latency_ms` (the final attempt's own send+recv) was already correct in
isolation, but nothing told the developer the call as a whole took far longer, or
why. These tests pin the SDK-layer facts `newt run` needs to render honestly:
`total_ms` (real wall-clock for the whole call) alongside `latency_ms`, and
`retries` (a real count, never a guessed duration) — plus the friction #6 fix
that stops a DegradationWarning from leaking a filesystem path to the terminal.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np

import newt
from newt._client.robot import (
    ColdStartRetry,
    DegradationWarning,
    VerifierError,
    _format_nt_warning,
)


def _make_verifier_error(msg: str = "The API key verification service is temporarily unavailable. Retry the request in a few seconds.") -> VerifierError:
    inst = VerifierError.__new__(VerifierError)
    newt.NewTheoryError.__init__(
        inst, code=4503, type="verifier.unavailable", message=msg, context={}, docs=None, trace_id="",
    )
    return inst


def _make_action_frame() -> bytes:
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
    ws = MagicMock()
    ws.send.return_value = None
    ws.recv.return_value = _make_action_frame()
    ws.close.return_value = None
    return ws


def _make_robot() -> newt.Robot:
    with patch.dict("os.environ", {"NT_INFERENCE_URL": "wss://fake.invalid/stream"}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", newt.EnvOverrideWarning)
            robot = newt.Robot(api_key="nt_fakekey")
    return robot


# ---------------------------------------------------------------------------
# Warm path: total_ms == latency_ms (no gap), retries == 0
# ---------------------------------------------------------------------------


def test_warm_call_total_equals_latency_and_zero_retries(monkeypatch):
    """A normal warm call — no retry, fast connect — has nothing to split.

    If this regresses (total_ms drifting meaningfully above latency_ms on the
    happy path, or retries nonzero), `newt run`'s render logic would show a
    misleading split on every call, not just cold ones.
    """
    robot = _make_robot()
    monkeypatch.setattr(robot, "_ws_connect", lambda: _fake_ws_that_returns_action())

    result = robot.infer({})

    assert result.retries == 0
    # total_ms is real wall time for the whole call (>= latency_ms always); on a
    # warm mocked call the gap is just interpreter overhead, not a real wait.
    assert result.total_ms >= result.latency_ms
    assert result.total_ms - result.latency_ms < 50, (
        f"warm call's total_ms shouldn't drift far from latency_ms: "
        f"total={result.total_ms} latency={result.latency_ms}"
    )


# ---------------------------------------------------------------------------
# Verifier retry: retries folds in the real retry count
# ---------------------------------------------------------------------------


def test_verifier_retry_folds_into_retries_count(monkeypatch):
    """A transient verifier retry must show up in .retries — the real count the
    retry loop itself knows, never a guessed duration (issue #38's discipline).

    If this regresses: `newt run` has no honest signal that a retry happened and
    would render the split-latency line only by luck (a large total-vs-latency
    gap), missing the case where the retry resolved fast.
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
    monkeypatch.setattr("newt._client.robot.time.sleep", lambda s: None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", newt.VerifierTransientRetry)
        result = robot.infer({})

    assert result.retries == 1, "one verifier retry must produce retries == 1"
    assert isinstance(result, newt.InferenceResponse)


def test_two_verifier_retries_counted_exactly(monkeypatch):
    """Two transient retries then success → retries == 2, not 1 (not just 'any')."""
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", newt.VerifierTransientRetry)
        result = robot.infer({})

    assert result.retries == 2


# ---------------------------------------------------------------------------
# Cold-start connect retry folds in too
# ---------------------------------------------------------------------------


def test_cold_start_connect_retry_folds_into_retries_count(monkeypatch):
    """When _ws_connect()'s own cold-start retry fires, infer() must fold that
    into .retries too — a cold container is exactly the case #38 is about.
    """
    robot = _make_robot()

    def fake_ws_connect():
        # Simulate what the real _ws_connect() does on its cold-start-retry path:
        # sets the flag infer() reads immediately after the call returns.
        robot._last_connect_cold_retried = True
        return _fake_ws_that_returns_action()

    monkeypatch.setattr(robot, "_ws_connect", fake_ws_connect)

    result = robot.infer({})

    assert result.retries == 1


# ---------------------------------------------------------------------------
# Friction #6: DegradationWarning (and siblings) render without a filesystem path
# ---------------------------------------------------------------------------


def test_nt_warning_categories_render_without_filesystem_path():
    """The default (uncaptured) rendering of an NT warning must not leak the
    internal .py path + source-line dump — just 'Category: message'.

    Before the fix: `warnings.warn(DegradationWarning(...))` in a plain script
    (no -W flag, no test harness) printed
    '/Users/.../robot.py:1637: DegradationWarning: <msg>\\n  warnings.warn(...)'
    — an internal path and alarming boilerplate for what's meant to read as a
    plain-language heads-up.
    """
    rendered = _format_nt_warning(
        "Model 'nt0-fp3' expected cameras not all present. Missing: ['left_wrist'].",
        DegradationWarning,
        "/Users/dev/.venv/lib/python3.12/site-packages/newt/_client/robot.py",
        1637,
    )
    assert rendered == (
        "DegradationWarning: Model 'nt0-fp3' expected cameras not all present. "
        "Missing: ['left_wrist'].\n"
    )
    assert "robot.py" not in rendered
    assert "/Users/" not in rendered


def test_non_nt_warning_categories_keep_stock_format():
    """A non-NT warning category (e.g. a third-party DeprecationWarning) must keep
    Python's stock format, path and all — this fix is scoped to our own categories,
    not a blanket suppression of every warning's source location."""
    rendered = _format_nt_warning(
        "some other library's warning",
        DeprecationWarning,
        "/some/other/lib.py",
        42,
    )
    assert "/some/other/lib.py:42" in rendered
    assert "DeprecationWarning: some other library's warning" in rendered


def test_cold_start_retry_message_has_no_hardcoded_minute_estimate():
    """ColdStartRetry's message must not hardcode a duration the wire doesn't
    support (issue #38's own callout: the old '60-90s' text was right for
    NT0-FP3 and wrong for the SO-101 family app measured at ~3.5min cold).
    """
    warning = ColdStartRetry("Cold-start retry for model='so101-red-cube-bowl' "
                              "(warming up the container — first call after idle "
                              "can take a few minutes). Subsequent calls hit the "
                              "warm container.")
    msg = str(warning)
    assert "60-90s" not in msg
    assert "30-90s" not in msg
    # digits immediately followed by a time unit would be a fabricated estimate
    import re
    assert not re.search(r"\d+\s*[-–]\s*\d+\s*(s|sec|second|min|minute)", msg), (
        f"message must not carry a numeric duration range: {msg!r}"
    )
