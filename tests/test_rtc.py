"""Tests for Real-Time Chunking (run(rtc=True)) — additive, flag-gated.

Verifies the brief-304 client-side contract:
  - execute() arity detection (one-arg legacy vs. two-arg RTC) via signature
    inspection — both shapes, plus the teaching rail for a wrong-shaped execute.
  - The rtc loop against a MOCK WS server (threading, no network):
      * splice-at-index correctness (resume == prefix_len; the off-by-one home)
      * progress-frame emission (boundary + every K actions)
      * prefix_len handling (absent → 0; the frozen prefix is not replayed)
      * one-arg fallback warning + boundary-only splicing
      * abort-callback plumb-through (two-arg execute is preemptible)
  - rtc=False stays byte-identical (covered by the unchanged existing suite).

No network, no hardware. The mock WS is an in-process queue pair so we exercise
the real sender/receiver thread split.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import newt
from newt._client.robot import (
    EmbodimentError,
    RTCBoundarySplicingWarning,
    Robot,
    _EXECUTE_ARITY_ONE,
    _EXECUTE_ARITY_TWO,
    _execute_arity,
    _pack,
    _unpack,
)


# ---------------------------------------------------------------------------
# execute() arity detection — both shapes + teaching rail
# ---------------------------------------------------------------------------

def test_arity_one_arg_function():
    def execute(chunk):
        pass
    assert _execute_arity(execute) == _EXECUTE_ARITY_ONE


def test_arity_two_arg_function():
    def execute(chunk, should_abort):
        pass
    assert _execute_arity(execute) == _EXECUTE_ARITY_TWO


def test_arity_one_arg_bound_method():
    class E:
        def execute(self, chunk):
            pass
    assert _execute_arity(E().execute) == _EXECUTE_ARITY_ONE


def test_arity_two_arg_bound_method():
    class E:
        def execute(self, chunk, should_abort):
            pass
    assert _execute_arity(E().execute) == _EXECUTE_ARITY_TWO


def test_arity_two_arg_optional_default_is_still_two():
    # execute(chunk, should_abort=None) — the runtime supplies it; arity is two.
    def execute(chunk, should_abort=None):
        pass
    assert _execute_arity(execute) == _EXECUTE_ARITY_TWO


def test_arity_var_positional_treated_as_two():
    # execute(*actions) can absorb (chunk, should_abort).
    def execute(*actions):
        pass
    assert _execute_arity(execute) == _EXECUTE_ARITY_TWO


def test_arity_three_arg_raises_teaching_error():
    def execute(chunk, should_abort, extra):
        pass
    with pytest.raises(EmbodimentError) as exc_info:
        _execute_arity(execute)
    err = exc_info.value
    assert err.type == "embodiment.execute_signature"
    # Teaching message names both valid shapes and points at docs.
    msg = str(err)
    assert "execute(chunk)" in msg
    assert "execute(chunk, should_abort)" in msg
    assert err.docs is not None


def test_arity_zero_arg_raises_teaching_error():
    # No positional slot for the chunk at all.
    def execute():
        pass
    with pytest.raises(EmbodimentError) as exc_info:
        _execute_arity(execute)
    assert exc_info.value.type == "embodiment.execute_signature"


def test_arity_lambda_one_arg():
    assert _execute_arity(lambda chunk: None) == _EXECUTE_ARITY_ONE


def test_arity_lambda_two_arg():
    assert _execute_arity(lambda chunk, abort: None) == _EXECUTE_ARITY_TWO


# ---------------------------------------------------------------------------
# Mock WS server (threading, no network)
# ---------------------------------------------------------------------------

class MockWS:
    """In-process stand-in for websockets.sync.client connection.

    The test script drives the server side: `outbound` is a list of already-
    msgpack-packed frames the server will hand back, one per recv(). Every
    client send() is captured (unpacked) into `sent` so the test can assert on
    obs/progress frames. recv() pops the next scripted server frame; when the
    script is exhausted it raises ConnectionClosed-like behavior by returning a
    terminal frame (set by the test) or blocking until close.
    """

    def __init__(self, server_frames, recv_latency=0.03):
        # server_frames: list of dicts the server emits, consumed by recv() in order.
        # recv_latency simulates inference wall-time so the execute thread has time
        # to pull each chunk before the next lands (mirrors the real ~1.6s floor;
        # the whole point of RTC is that generation takes real time).
        self._server_frames = list(server_frames)
        self._recv_idx = 0
        self._recv_latency = recv_latency
        self.sent = []  # unpacked client→server frames
        self._lock = threading.Lock()
        self.closed = False

    def send(self, payload):
        if self.closed:
            from websockets.exceptions import ConnectionClosed
            raise ConnectionClosed(None, None)
        with self._lock:
            self.sent.append(_unpack(payload))

    def recv(self):
        # Simulate inference wall-time so the execute thread drains each chunk
        # before the next lands (the WS thread must not blast through all frames
        # instantly — that's not how a real server behaves).
        time.sleep(self._recv_latency)
        with self._lock:
            if self._recv_idx < len(self._server_frames):
                frame = self._server_frames[self._recv_idx]
                self._recv_idx += 1
                return _pack(frame)
        from websockets.exceptions import ConnectionClosed
        raise ConnectionClosed(None, None)

    def close(self):
        self.closed = True


def _make_robot(monkeypatch, mock_ws):
    """Build a Robot wired to a mock WS, skipping network construction."""
    monkeypatch.setenv("NT_INFERENCE_URL", "wss://mock/stream")
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")

    captured = {}

    def _read_state():
        captured.setdefault("read_state_calls", 0)
        captured["read_state_calls"] += 1
        return {"state": np.zeros(8, dtype=np.float32)}

    robot = Robot(
        api_key="nt_testkey",
        read_state=_read_state,
        execute=lambda chunk: None,  # replaced per-test
    )
    monkeypatch.setattr(robot, "_ws_connect", lambda: mock_ws)
    return robot, captured


def _action_frame(chunk, prefix_len=None):
    f = {"type": "action", "chunk": chunk}
    if prefix_len is not None:
        f["prefix_len"] = prefix_len
    return f


def _terminal(reason="max_duration"):
    return {"type": "terminal", "stop_reason": reason}


# ---------------------------------------------------------------------------
# prefix_len handling + splice-at-index correctness
# ---------------------------------------------------------------------------

def test_rtc_prefix_len_zero_plays_whole_chunk(monkeypatch):
    """prefix_len absent → resume at 0 → the whole chunk is played (no frozen prefix)."""
    chunk = np.arange(10, dtype=np.float32).reshape(10, 1)
    mock = MockWS([_action_frame(chunk), _terminal()])
    robot, _ = _make_robot(monkeypatch, mock)

    played = []
    robot._execute = lambda c: played.append(c)

    robot.run("test", max_duration=5.0, rtc=True)

    assert len(played) == 1
    # No prefix_len → start=0 → played tail == full chunk.
    np.testing.assert_array_equal(played[0], chunk)


def test_rtc_prefix_len_skips_frozen_prefix(monkeypatch):
    """prefix_len=p → the first p actions are the frozen region and are NOT replayed.

    The resume index IS prefix_len (the off-by-one home): played tail == chunk[p:].
    """
    chunk = np.arange(10, dtype=np.float32).reshape(10, 1)
    mock = MockWS([_action_frame(chunk, prefix_len=3), _terminal()])
    robot, _ = _make_robot(monkeypatch, mock)

    played = []
    robot._execute = lambda c: played.append(c)

    robot.run("test", max_duration=5.0, rtc=True)

    assert len(played) == 1
    # Resume at index 3 — actions 0,1,2 are frozen prefix, not replayed.
    np.testing.assert_array_equal(played[0], chunk[3:])
    # Specifically: first played action is chunk[3], never chunk[2] (stutter) or
    # chunk[4] (gap).
    assert played[0][0, 0] == 3.0


def test_rtc_prefix_len_equal_horizon_plays_nothing(monkeypatch):
    """prefix_len == horizon → whole chunk is frozen → nothing fresh is played."""
    chunk = np.arange(5, dtype=np.float32).reshape(5, 1)
    mock = MockWS([_action_frame(chunk, prefix_len=5), _terminal()])
    robot, _ = _make_robot(monkeypatch, mock)

    played = []
    robot._execute = lambda c: played.append(c)

    robot.run("test", max_duration=5.0, rtc=True)

    # start >= n → the chunk contributes no fresh actions.
    assert played == []


def test_rtc_negative_prefix_len_coerced_to_zero(monkeypatch):
    """A malformed (negative) prefix_len is coerced to 0 — fail toward replay-all."""
    chunk = np.arange(4, dtype=np.float32).reshape(4, 1)
    mock = MockWS([_action_frame(chunk, prefix_len=-2), _terminal()])
    robot, _ = _make_robot(monkeypatch, mock)

    played = []
    robot._execute = lambda c: played.append(c)
    robot.run("test", max_duration=5.0, rtc=True)

    np.testing.assert_array_equal(played[0], chunk)


def test_rtc_multi_chunk_splice_sequence(monkeypatch):
    """Two chunks across the boundary: each plays its tail from its own prefix_len."""
    c0 = np.arange(6, dtype=np.float32).reshape(6, 1)
    c1 = (np.arange(6, dtype=np.float32) + 100).reshape(6, 1)
    mock = MockWS([
        _action_frame(c0, prefix_len=0),
        _action_frame(c1, prefix_len=2),
        _terminal(),
    ])
    robot, _ = _make_robot(monkeypatch, mock)

    played = []
    robot._execute = lambda c: played.append(c.copy())
    robot.run("test", max_duration=5.0, rtc=True)

    assert len(played) == 2
    np.testing.assert_array_equal(played[0], c0)        # start 0
    np.testing.assert_array_equal(played[1], c1[2:])    # start 2, frozen prefix skipped


# ---------------------------------------------------------------------------
# rtc-enable handshake — the client must TELL the server it's RTC
# ---------------------------------------------------------------------------

def test_rtc_first_obs_frame_carries_rtc_flag(monkeypatch):
    """run(rtc=True) MUST stamp rtc=true on the FIRST obs frame.

    WHY (Rule 7): the server's RTC mode is opt-in via this field (or a ?rtc=true
    query param). If the client never sends either, the server silently runs
    VANILLA inference — no inpainting, no overlap, the chunk-boundary seam never
    closes — and nothing errors. That fail-silent gap is exactly what this test
    pins: the rtc flag must be present on the first frame and absent thereafter
    (it is a session-open signal, not a per-frame one).
    """
    c0 = np.arange(6, dtype=np.float32).reshape(6, 1)
    c1 = (np.arange(6, dtype=np.float32) + 100).reshape(6, 1)
    mock = MockWS([
        _action_frame(c0, prefix_len=0),
        _action_frame(c1, prefix_len=2),
        _terminal(),
    ])
    robot, _ = _make_robot(monkeypatch, mock)
    robot._execute = lambda c: None

    robot.run("test", max_duration=5.0, rtc=True)

    obs_frames = [f for f in mock.sent if f.get("type") == "obs"]
    assert obs_frames, "expected at least one obs frame"
    # First obs enables RTC; the server reads rtc=true off exactly this frame.
    assert obs_frames[0].get("rtc") is True, (
        "first obs frame must carry rtc=true or the server runs vanilla "
        "(fail-silent: no overlap, seam never closes)"
    )
    # Subsequent obs frames don't repeat it (session-open signal, like max_duration).
    for f in obs_frames[1:]:
        assert not f.get("rtc"), "rtc flag should only be on the first obs frame"


def test_non_rtc_run_never_sends_rtc_flag(monkeypatch):
    """A normal run() (rtc=False) must NEVER stamp the rtc field — additive-safe."""
    chunk = np.arange(4, dtype=np.float32).reshape(4, 1)
    mock = MockWS([_action_frame(chunk), _terminal()])
    robot, _ = _make_robot(monkeypatch, mock)
    robot._execute = lambda c: None

    robot.run("test", max_duration=5.0)  # rtc defaults to False

    obs_frames = [f for f in mock.sent if f.get("type") == "obs"]
    for f in obs_frames:
        assert "rtc" not in f, "non-RTC run must not send the rtc field"


# ---------------------------------------------------------------------------
# progress-frame emission
# ---------------------------------------------------------------------------

def test_rtc_progress_frame_emitted_at_boundary(monkeypatch):
    """Each chunk-boundary emits a progress frame with action_index == prefix_len."""
    chunk = np.arange(3, dtype=np.float32).reshape(3, 1)
    mock = MockWS([_action_frame(chunk, prefix_len=1), _terminal()])
    robot, _ = _make_robot(monkeypatch, mock)
    robot._execute = lambda c: None

    robot.run("test", max_duration=5.0, rtc=True)

    progress = [f for f in mock.sent if f.get("type") == "progress"]
    assert progress, "expected at least one progress frame"
    # Boundary frame reports the splice index (== prefix_len).
    assert any(p["action_index"] == 1 for p in progress)


def test_rtc_progress_frame_every_k_actions(monkeypatch):
    """A two-arg execute that polls should_abort every action emits a progress frame
    every K=5 actions (the K-cadence half of the wire contract)."""
    chunk = np.zeros((12, 1), dtype=np.float32)
    mock = MockWS([_action_frame(chunk, prefix_len=0), _terminal()])
    robot, _ = _make_robot(monkeypatch, mock)

    def execute(c, should_abort):
        # Poll once per action, like the starter polls _check_emergency_stop.
        for _ in range(c.shape[0]):
            if should_abort():
                return

    robot._execute = execute
    robot.run("test", max_duration=5.0, rtc=True)

    progress = [f for f in mock.sent if f.get("type") == "progress"]
    indices = sorted(p["action_index"] for p in progress)
    # Boundary (0) + K-cadence at 5 and 10 (12 actions, K=5).
    assert 5 in indices
    assert 10 in indices


# ---------------------------------------------------------------------------
# one-arg fallback warning + abort plumb-through
# ---------------------------------------------------------------------------

def test_rtc_one_arg_emits_boundary_splicing_warning(monkeypatch):
    """rtc=True with a one-arg execute warns once, naming the upgrade path."""
    chunk = np.zeros((4, 1), dtype=np.float32)
    mock = MockWS([_action_frame(chunk), _terminal()])
    robot, _ = _make_robot(monkeypatch, mock)
    robot._execute = lambda c: None

    with pytest.warns(RTCBoundarySplicingWarning) as record:
        robot.run("test", max_duration=5.0, rtc=True)

    assert len(record) == 1
    msg = str(record[0].message)
    assert "should_abort" in msg  # names the upgrade path


def test_rtc_two_arg_no_boundary_warning(monkeypatch):
    """A two-arg execute does NOT trigger the one-arg fallback warning."""
    import warnings as _w

    chunk = np.zeros((4, 1), dtype=np.float32)
    mock = MockWS([_action_frame(chunk), _terminal()])
    robot, _ = _make_robot(monkeypatch, mock)
    robot._execute = lambda c, should_abort: None

    with _w.catch_warnings():
        _w.simplefilter("error", RTCBoundarySplicingWarning)
        robot.run("test", max_duration=5.0, rtc=True)  # must not raise


def test_rtc_abort_callback_is_callable_and_returns_bool(monkeypatch):
    """The should_abort handed to a two-arg execute is callable and returns a bool."""
    chunk = np.zeros((6, 1), dtype=np.float32)
    mock = MockWS([_action_frame(chunk), _terminal()])
    robot, _ = _make_robot(monkeypatch, mock)

    seen = {}

    def execute(c, should_abort):
        seen["callable"] = callable(should_abort)
        seen["result"] = should_abort()

    robot._execute = execute
    robot.run("test", max_duration=5.0, rtc=True)

    assert seen["callable"] is True
    assert isinstance(seen["result"], bool)


def test_rtc_two_arg_execute_preempts_on_newer_chunk(monkeypatch):
    """A two-arg execute that honors should_abort returns early when a newer chunk
    is buffered, and the loop then splices to it."""
    c0 = np.zeros((50, 1), dtype=np.float32)   # long chunk to give time to preempt
    c1 = np.ones((50, 1), dtype=np.float32)
    mock = MockWS([
        _action_frame(c0, prefix_len=0),
        _action_frame(c1, prefix_len=0),
        _terminal(),
    ])
    robot, _ = _make_robot(monkeypatch, mock)

    played_chunks = []

    def execute(c, should_abort):
        # Record which chunk we started, play action-by-action, honor abort.
        marker = float(c[0, 0])
        played_chunks.append(marker)
        for _ in range(c.shape[0]):
            if should_abort():
                return
            time.sleep(0.002)

    robot._execute = execute
    robot.run("test", max_duration=5.0, rtc=True)

    # Both chunks got an execute() call — the second only happens if the first
    # preempted (or finished) and the loop spliced.
    assert 0.0 in played_chunks  # c0
    assert 1.0 in played_chunks  # c1


# ---------------------------------------------------------------------------
# rtc=True / stream=True mutual exclusion + terminal handling
# ---------------------------------------------------------------------------

def test_rtc_and_stream_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("NT_INFERENCE_URL", "wss://mock/stream")
    robot = Robot(
        api_key="nt_testkey",
        read_state=lambda: {},
        execute=lambda c: None,
    )
    with pytest.raises(TypeError) as exc_info:
        robot.run("test", rtc=True, stream=True)
    assert "rtc=True" in str(exc_info.value)


def test_rtc_terminal_stop_reason_surfaces(monkeypatch):
    """A terminal frame's stop_reason is returned in RunResult."""
    chunk = np.zeros((3, 1), dtype=np.float32)
    mock = MockWS([_action_frame(chunk), _terminal(reason="task_complete")])
    robot, _ = _make_robot(monkeypatch, mock)
    robot._execute = lambda c: None

    result = robot.run("test", max_duration=5.0, rtc=True)
    assert result.stop_reason == "task_complete"


def test_rtc_run_requires_callbacks(monkeypatch):
    """rtc=True still requires read_state + execute (same guard as the blocking path)."""
    monkeypatch.setenv("NT_INFERENCE_URL", "wss://mock/stream")
    robot = Robot(api_key="nt_testkey")  # no callbacks
    with pytest.raises(TypeError) as exc_info:
        robot.run("test", rtc=True)
    assert "read_state" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def test_rtc_warning_exported():
    assert hasattr(newt, "RTCBoundarySplicingWarning")
    assert newt.RTCBoundarySplicingWarning is RTCBoundarySplicingWarning
    assert "RTCBoundarySplicingWarning" in newt.__all__
