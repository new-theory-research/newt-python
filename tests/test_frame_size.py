"""Unit tests for the frame-size refusal guard (serve-023).

The gap: `infer()`, `_run_blocking_once`, and `_stream_once` each pack an obs
frame and hand it straight to `ws.send()` with no size check. A frame over
Modal's inbound limit is torn down in transit — no close frame, no error — and
on the streaming path only the FIRST observation is ever pre-flighted against
the model contract, so a run that grows a frame mid-stream (a camera
reconnecting at native resolution, e.g.) put an unchecked frame on the wire
and died silently. These tests pin: an over-limit frame refuses BEFORE any
byte reaches ws.send(); an under-limit frame is untouched; and the reachable
failure this whole brief exists for — frame 1 of a stream passes, frame 7
doesn't, and it's caught locally instead of dying on the wire.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import newt
from newt._client.robot import _MAX_FRAME_BYTES, FrameTooLargeError, _pack


def _make_robot() -> newt.Robot:
    """Robot with registry discovery skipped (no network) — mirrors test_verifier_retry.py."""
    with patch.dict("os.environ", {"NT_INFERENCE_URL": "wss://fake.invalid/stream"}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", newt.EnvOverrideWarning)
            robot = newt.Robot(api_key="nt_fakekey")
    return robot


def _action_frame_bytes() -> bytes:
    """A minimal valid msgpack action frame, round-tripped through the SDK's own codec."""
    chunk = np.zeros((4, 8), dtype=np.float32)
    return _pack({"type": "action", "chunk": chunk})


def _small_obs() -> dict:
    return {"state": np.zeros((6,), dtype=np.float32)}


def _oversized_obs() -> dict:
    """One (3, 1024, 1024) uint8 image, ~3.14 MB raw — well over the 2 MiB limit."""
    return {
        "state": np.zeros((6,), dtype=np.float32),
        "images": {"top": np.zeros((3, 1024, 1024), dtype=np.uint8)},
    }


# ---------------------------------------------------------------------------
# infer() — the one-shot send site
# ---------------------------------------------------------------------------

def test_infer_refuses_oversized_frame_before_ws_send():
    """An over-limit frame raises FrameTooLargeError; ws.send() is never reached."""
    robot = _make_robot()
    ws = MagicMock()
    ws.send.side_effect = AssertionError(
        "ws.send() was called — the size guard must refuse BEFORE any byte hits the wire"
    )
    robot._ws_connect = MagicMock(return_value=ws)

    with pytest.raises(FrameTooLargeError) as exc:
        robot.infer(_oversized_obs())

    assert ws.send.call_count == 0
    assert f"{_MAX_FRAME_BYTES:,}" in str(exc.value)
    assert exc.value.type == "frame_too_large.exceeds_limit"
    assert exc.value.context["frame_index"] == 1
    assert exc.value.context["image_shapes"] == {"top": [3, 1024, 1024]}


def test_infer_frame_under_limit_sends_unchanged():
    """A normal-size frame is untouched — sends and returns exactly as before the guard."""
    robot = _make_robot()
    ws = MagicMock()
    ws.send.return_value = None
    ws.recv.return_value = _action_frame_bytes()
    robot._ws_connect = MagicMock(return_value=ws)

    response = robot.infer(_small_obs())

    assert ws.send.call_count == 1
    assert response.action_chunk.shape == (4, 8)


# ---------------------------------------------------------------------------
# Streaming — the reachable failure: frame 1 passes, frame 7 doesn't
# ---------------------------------------------------------------------------

def test_streaming_frame_1_passes_frame_7_refuses():
    """`_stream_once` pre-flights only the FIRST obs against the model contract;
    frames 2..N are never shape-checked. This is the size guard's reason to
    exist: a read_state() that grows mid-run must be caught locally, not die
    on the wire with no close frame — the exact gap serve-023 was filed for.
    """
    robot = _make_robot()
    ws = MagicMock()
    ws.send.return_value = None
    ws.recv.return_value = _action_frame_bytes()
    robot._ws_connect = MagicMock(return_value=ws)

    call_count = 0

    def read_state():
        nonlocal call_count
        call_count += 1
        return _oversized_obs() if call_count == 7 else _small_obs()

    robot._read_state = read_state
    robot._execute = lambda chunk: None

    gen = robot.run("pick up the cube", stream=True)
    received = []
    with pytest.raises(FrameTooLargeError) as exc:
        for chunk in gen:
            received.append(chunk)

    assert len(received) == 6  # frames 1-6 sent and yielded fine
    assert exc.value.context["frame_index"] == 7
    assert ws.send.call_count == 6  # frame 7's send never happened
