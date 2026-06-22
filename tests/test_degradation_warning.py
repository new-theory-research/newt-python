"""Unit tests for the DegradationWarning surface (brief-258b).

These exercise the SDK rendering of the three degradation signals the server
attaches to the action chunk's `warnings` dict, WITHOUT a live serve. The payload
shapes are the contract; if the server emits them, the SDK must name the missing
fields per camera and the quality-impact sentence (the legibility bar).

The signals tested:
  - missing_expected_cameras   (pre-258b camera signal — must keep working)
  - missing_depth_fields       (declared geometry inputs absent, per camera)
  - mid_session_degradation    (garbled state after frame 1; per-frame)
"""

from __future__ import annotations

import warnings

from newt._client.robot import (
    DegradationWarning,
    _maybe_warn_degradation,
    _maybe_warn_mid_session,
)


def _depth_warning_payload() -> dict:
    """The shape serve_nt0._build_warning_from_catalog('degradation.missing_depth_fields')
    produces, hand-built so the test does not import the server."""
    return {
        "type": "degradation.missing_depth_fields",
        "impact": (
            "Missing geometry was identity/zero-filled; the point cloud the "
            "encoder conditions on is vacuous or geometrically wrong, so actions "
            "will be noticeably degraded versus a run with real depth, intrinsics, "
            "and extrinsics."
        ),
        "context": {
            "model": "nt0-fp3",
            "missing_expected_fields": [
                {"camera": "right-wrist-camera", "fields": ["depth_maps", "intrinsics", "extrinsics"]},
                {"camera": "surrounding1", "fields": ["depth_maps", "intrinsics", "extrinsics"]},
                {"camera": "surrounding2", "fields": ["depth_maps", "intrinsics", "extrinsics"]},
            ],
        },
        "docs": "https://docs.newtheory.ai/api/errors#degradation-missing-depth-fields",
        "trace_id": "tr_test1234",
    }


def _mid_session_payload(frame_index: int = 5) -> dict:
    return {
        "type": "degradation.mid_session_state",
        "impact": (
            "The malformed state was coerced to the expected shape (zero-fill / "
            "pad / truncate); an all-zero or padded pose poisons the proprio "
            "context buffer, so the next several inferences are computed from a "
            "state that does not reflect the robot — expect degraded or unsafe "
            "actions until clean state resumes."
        ),
        "context": {
            "model": "nt0-fp3",
            "frame_index": frame_index,
            "expected_shape": [8],
            "got_shape": [],
            "got_dtype": "absent",
            "expected_dtype": "float32",
        },
        "docs": "https://docs.newtheory.ai/api/errors#degradation-mid-session-state",
        "trace_id": "tr_test5678",
    }


def test_missing_depth_fields_names_fields_and_impact():
    """The geometry warning must name the specific missing fields per camera AND
    carry the quality-impact sentence — both halves of the legibility bar."""
    frame = {"type": "action", "warnings": {"missing_depth_fields": _depth_warning_payload()}}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _maybe_warn_degradation(frame, "nt0-fp3")

    msgs = [str(w.message) for w in rec if issubclass(w.category, DegradationWarning)]
    assert len(msgs) == 1, f"expected exactly one DegradationWarning, got {msgs}"
    msg = msgs[0]
    # Names the specific missing fields (legibility bar #1).
    assert "depth_maps" in msg and "intrinsics" in msg and "extrinsics" in msg
    # Names them PER camera, not as a flat list.
    assert "right-wrist-camera" in msg and "surrounding1" in msg and "surrounding2" in msg
    # Carries the quality-impact sentence (legibility bar #2).
    assert "degraded" in msg and "point cloud" in msg


def test_missing_expected_cameras_still_warns():
    """The pre-258b camera signal must keep firing — no regression."""
    frame = {"type": "action", "warnings": {"missing_expected_cameras": ["surrounding2"]}}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _maybe_warn_degradation(frame, "nt0-fp3")
    msgs = [str(w.message) for w in rec if issubclass(w.category, DegradationWarning)]
    assert len(msgs) == 1 and "surrounding2" in msgs[0]


def test_both_camera_and_depth_warn_independently():
    """A frame missing both cameras and geometry emits two distinct warnings."""
    frame = {
        "type": "action",
        "warnings": {
            "missing_expected_cameras": ["surrounding2"],
            "missing_depth_fields": _depth_warning_payload(),
        },
    }
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _maybe_warn_degradation(frame, "nt0-fp3")
    msgs = [str(w.message) for w in rec if issubclass(w.category, DegradationWarning)]
    assert len(msgs) == 2


def test_no_warning_on_clean_frame():
    """Happy path: a frame with no warnings dict emits nothing."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _maybe_warn_degradation({"type": "action"}, "nt0-fp3")
        _maybe_warn_mid_session({"type": "action"}, "nt0-fp3")
    assert [w for w in rec if issubclass(w.category, DegradationWarning)] == []


def test_mid_session_warns_with_frame_index_and_impact():
    """Mid-session warning names which frame went bad and carries the impact sentence."""
    frame = {"type": "action", "warnings": {"mid_session_degradation": _mid_session_payload(frame_index=7)}}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _maybe_warn_mid_session(frame, "nt0-fp3")
    msgs = [str(w.message) for w in rec if issubclass(w.category, DegradationWarning)]
    assert len(msgs) == 1
    assert "frame 7" in msgs[0]
    assert "proprio" in msgs[0]  # impact sentence present


def test_mid_session_fires_per_frame_not_once():
    """Each garbled frame gets its own warning — this is the every-frame guarantee.

    If the SDK gated mid-session behind a once-per-run flag, two garbled frames
    would produce one warning and the developer would not learn how many frames
    were affected. brief-258b forbids that.
    """
    total = 0
    for idx in (3, 4):
        frame = {"type": "action", "warnings": {"mid_session_degradation": _mid_session_payload(frame_index=idx)}}
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _maybe_warn_mid_session(frame, "nt0-fp3")
        total += len([w for w in rec if issubclass(w.category, DegradationWarning)])
    assert total == 2, "mid-session must warn once per affected frame"


def test_bytes_keys_decoded():
    """msgpack may hand back bytes keys/values; the SDK must still render."""
    frame = {
        b"type": b"action",
        b"warnings": {
            b"missing_depth_fields": {
                b"type": b"degradation.missing_depth_fields",
                b"impact": b"actions will be noticeably degraded; the point cloud is vacuous.",
                b"context": {
                    b"missing_expected_fields": [
                        {b"camera": b"surrounding1", b"fields": [b"extrinsics"]},
                    ],
                },
                b"trace_id": b"tr_bytes",
            },
        },
    }
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _maybe_warn_degradation(frame, "nt0-fp3")
    msgs = [str(w.message) for w in rec if issubclass(w.category, DegradationWarning)]
    assert len(msgs) == 1 and "surrounding1" in msgs[0] and "extrinsics" in msgs[0]
