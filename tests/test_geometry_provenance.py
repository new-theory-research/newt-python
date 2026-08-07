"""Geometry provenance is an explicit, per-camera episode-format claim."""
from __future__ import annotations

import copy
import importlib.util
import json

import pytest


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _entry(**overrides):
    from newt.recording import CameraSpec
    from newt.recording._writer import _camera_entry

    values = {"id": "wrist", "width": 640, "height": 480}
    values.update(overrides)
    return _camera_entry(CameraSpec(**values))


def test_no_geometry_is_written_as_explicit_invalid_not_silence():
    from newt.recording._validate import _geometry_provenance_check

    camera = _entry()
    assert camera["geometry_provenance"] == {
        "kind": "invalid",
        "reason": "not_provided",
    }
    assert "extrinsics" not in camera
    assert _geometry_provenance_check(camera) == (
        True,
        "explicitly invalid: not_provided",
    )


@pytest.mark.parametrize("receipt", [None, "", "   ", "/tmp/calibration.json", "../run.json"])
def test_measured_geometry_cannot_be_constructed_without_a_portable_receipt(receipt):
    from newt.recording import CameraSpec

    with pytest.raises(ValueError, match="portable calibration receipt"):
        CameraSpec(
            "wrist",
            640,
            480,
            extrinsics=copy.deepcopy(IDENTITY),
            geometry_kind="measured",
            geometry_receipt=receipt,
        )


def test_three_marks_are_structurally_distinct_and_name_the_fact_they_assert():
    cameras = {
        "measured": _entry(
            extrinsics=copy.deepcopy(IDENTITY),
            geometry_kind="measured",
            geometry_receipt="calibration/run-17.json",
        ),
        "declared": _entry(
            extrinsics=copy.deepcopy(IDENTITY),
            geometry_kind="declared",
            geometry_source="kit:standard-bracket-v2",
        ),
        "invalid": _entry(geometry_reason="operator_skipped_calibration"),
    }
    claims = [camera["geometry_provenance"] for camera in cameras.values()]
    assert {claim["kind"] for claim in claims} == set(cameras)
    assert len({frozenset(claim) for claim in claims}) == 3
    assert claims[0]["receipt"] == "calibration/run-17.json"
    assert claims[1]["source"] == "kit:standard-bracket-v2"
    assert claims[2]["reason"] == "operator_skipped_calibration"


def test_geometry_claims_are_per_camera_and_do_not_leak():
    measured = _entry(
        id="world",
        extrinsics=copy.deepcopy(IDENTITY),
        geometry_kind="measured",
        geometry_receipt="calibration/run-17.json",
    )
    invalid = _entry(id="wrist", geometry_reason="not_calibrated")
    assert measured["geometry_provenance"] == {
        "kind": "measured",
        "receipt": "calibration/run-17.json",
    }
    assert invalid["geometry_provenance"] == {
        "kind": "invalid",
        "reason": "not_calibrated",
    }
    assert "extrinsics" in measured
    assert "extrinsics" not in invalid


def test_legacy_camera_is_accepted_as_an_absent_claim_not_an_invented_one():
    from newt.recording._validate import _geometry_provenance_check

    ok, detail = _geometry_provenance_check({"id": "legacy", "frame": "world"})
    assert ok is True
    assert detail == "no geometry claim — episode predates explicit geometry provenance"


def test_validator_rejects_unbacked_measured_claim_and_names_owner_and_repair():
    from newt.recording._validate import _geometry_provenance_check

    ok, detail = _geometry_provenance_check(
        {
            "id": "wrist",
            "extrinsics": copy.deepcopy(IDENTITY),
            "geometry_provenance": {"kind": "measured"},
        }
    )
    assert ok is False
    assert "no portable episode-relative receipt" in detail
    assert "recording owner" in detail
    assert "must repair" in detail


@pytest.mark.skipif(importlib.util.find_spec("mcap") is None, reason="needs the recording extra")
def test_real_camera_episode_records_default_claim_and_validates(tmp_path):
    """Exercise the disk format, not merely the camera-dict helper."""
    from newt.recording import Session, validate
    from tests.fixtures.fake_camera_source import make_camera_source

    session = Session(make_camera_source(), task="geometry fixture", output_dir=tmp_path)
    try:
        session.start_episode()
        path = session.end_episode(keep=True)
    finally:
        session.close()

    camera = json.loads((path / "episode.json").read_text())["camera_config"]["cameras"][0]
    assert camera["geometry_provenance"] == {
        "kind": "invalid",
        "reason": "not_provided",
    }
    result = validate(path)
    claim = next(c for c in result["checks"] if c["check"] == "geometry_provenance[cam0]")
    assert claim == {
        "check": "geometry_provenance[cam0]",
        "ok": True,
        "detail": "explicitly invalid: not_provided",
    }
    assert result["valid"] is True
