"""A measured camera's receipt is a file inside the episode, not a name in a field.

``CameraSpec`` already rejects a receipt path that is absolute, empty, or escapes
the episode — the shape. These tests are about the other half: the path resolves
to a file that is really there, and an episode whose pointer dangles is refused
rather than committed.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil

import pytest

# A camera attached to the episode opens a real ffmpeg pipe at start_episode() —
# same gate the rest of the camera-carrying suites use.
_HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(
    not _HAVE_FFMPEG, reason="needs ffmpeg + ffprobe on PATH (the color encoder)"
)
needs_recording = pytest.mark.skipif(
    importlib.util.find_spec("mcap") is None, reason="needs the recording extra"
)

IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

RECEIPT_BODY = json.dumps({"procedure": "checkerboard", "rms_px": 0.21, "passes": 4})


def _rig_receipt(tmp_path, name="run-17.json", body=RECEIPT_BODY):
    """What a kit's calibrate run left on the rig's own disk, outside any episode."""
    rig = tmp_path / "rig-disk"
    rig.mkdir(exist_ok=True)
    path = rig / name
    path.write_text(body)
    return path


def _measured_source(*, receipt_claim, receipt_path=None):
    from newt.recording import CameraSpec
    from tests.fixtures.fake_camera_source import FakeCameraSource

    source = FakeCameraSource(
        cameras=[
            CameraSpec(
                "cam0",
                64,
                48,
                30,
                extrinsics=copy.deepcopy(IDENTITY),
                geometry_kind="measured",
                geometry_receipt=receipt_claim,
            )
        ]
    )
    if receipt_path is not None:
        source.calibration_receipt = receipt_path
    return source


def _record(source, dest):
    from newt.recording import Session

    session = Session(source, task="calibration receipt", output_dir=dest)
    try:
        session.start_episode()
        return session.end_episode(keep=True)
    finally:
        session.close()


def _nothing_landed(dest):
    """No episode, and no half-built temp dir either — abandon() removes it whole."""
    return sorted(p.name for p in dest.iterdir()) == [] if dest.exists() else True


@needs_recording
@needs_ffmpeg
def test_measured_receipt_is_a_real_file_the_episode_pointer_resolves_to(tmp_path):
    dest = tmp_path / "episodes"
    src = _rig_receipt(tmp_path)
    path = _record(
        _measured_source(receipt_claim="calibration/run-17.json", receipt_path=src), dest
    )

    camera = json.loads((path / "episode.json").read_text())["camera_config"]["cameras"][0]
    pointer = camera["geometry_provenance"]["receipt"]
    assert pointer == "calibration/run-17.json"
    # The whole point: follow the pointer from the episode root and land on the
    # bytes the rig wrote, not on a plausible name.
    landed = path / pointer
    assert landed.is_file()
    assert landed.read_text() == RECEIPT_BODY
    assert json.loads(landed.read_text())["rms_px"] == 0.21


@needs_recording
@needs_ffmpeg
def test_receipt_path_that_is_not_a_file_refuses_the_episode(tmp_path):
    dest = tmp_path / "episodes"
    missing = tmp_path / "rig-disk" / "run-17.json"
    with pytest.raises(RuntimeError) as excinfo:
        _record(
            _measured_source(
                receipt_claim="calibration/run-17.json", receipt_path=missing
            ),
            dest,
        )

    message = str(excinfo.value)
    assert str(missing) in message
    assert "must point it at the file its own calibrate run produced" in message
    assert _nothing_landed(dest)


@needs_recording
@needs_ffmpeg
def test_measured_camera_with_no_receipt_at_all_refuses_the_episode(tmp_path):
    dest = tmp_path / "episodes"
    with pytest.raises(RuntimeError) as excinfo:
        _record(_measured_source(receipt_claim="calibration/run-17.json"), dest)

    message = str(excinfo.value)
    assert "camera 'cam0'" in message
    assert "no calibration receipt at all" in message
    assert "must also set calibration_receipt to the file its calibrate run wrote" in message
    assert _nothing_landed(dest)


@needs_recording
@needs_ffmpeg
def test_measured_camera_pointing_at_the_wrong_name_refuses_the_episode(tmp_path):
    dest = tmp_path / "episodes"
    src = _rig_receipt(tmp_path, name="run-18.json")
    with pytest.raises(RuntimeError) as excinfo:
        _record(
            _measured_source(receipt_claim="calibration/run-17.json", receipt_path=src),
            dest,
        )

    message = str(excinfo.value)
    assert "camera 'cam0'" in message
    # Both halves of the mismatch, so the reader does not have to guess which end
    # is wrong: what the pointer says, and what the episode actually carries.
    assert "calibration/run-17.json" in message
    assert "calibration/run-18.json" in message
    assert "must point geometry_receipt at the file it handed over" in message
    assert _nothing_landed(dest)


def test_the_three_refusals_never_share_a_string(tmp_path):
    """Rule 12's hard part: two causes must not read the same. Asserted, not eyeballed."""
    from newt.recording import CameraSpec
    from newt.recording._writer import EpisodeWriter

    def refusal(**kwargs):
        writer = object.__new__(EpisodeWriter)
        writer._tmp = tmp_path / "tmp-episode"
        writer._tmp.mkdir(exist_ok=True)
        writer.abandon = lambda: None
        writer.cameras = [
            CameraSpec(
                "cam0",
                64,
                48,
                30,
                extrinsics=copy.deepcopy(IDENTITY),
                geometry_kind="measured",
                geometry_receipt="calibration/run-17.json",
            )
        ]
        writer.calibration_receipt = kwargs["receipt"]
        with pytest.raises(RuntimeError) as excinfo:
            EpisodeWriter._land_calibration_receipt(writer)
        return str(excinfo.value)

    messages = [
        refusal(receipt=tmp_path / "rig-disk" / "gone.json"),
        refusal(receipt=None),
        refusal(receipt=_rig_receipt(tmp_path, name="run-18.json")),
    ]
    assert len(set(messages)) == 3
    for message in messages:
        assert message.endswith("Episode discarded; nothing written.")


@needs_recording
def test_an_uncalibrated_episode_carries_no_calibration_directory(tmp_path):
    """The skip path costs nothing: no receipt, no measured claim, no refusal."""
    from newt.recording import SINGLE_ARM_DESCRIPTOR, SimulatedSource

    dest = tmp_path / "episodes"
    path = _record(SimulatedSource(SINGLE_ARM_DESCRIPTOR), dest)

    assert (path / "episode.json").is_file()
    assert not (path / "calibration").exists()


@needs_recording
def test_session_takes_the_receipt_off_the_source_not_off_a_flag(tmp_path):
    """The kit is the only thing that knows it calibrated, so the kit's own object
    is where the path comes from — no Session argument, no CLI flag."""
    from newt.recording import SINGLE_ARM_DESCRIPTOR, Session, SimulatedSource

    src = _rig_receipt(tmp_path)
    source = SimulatedSource(SINGLE_ARM_DESCRIPTOR)
    source.calibration_receipt = src

    session = Session(source, task="calibration receipt", output_dir=tmp_path / "episodes")
    assert session._calibration_receipt == src
    try:
        session.start_episode()
        path = session.end_episode(keep=True)
    finally:
        session.close()

    assert (path / "calibration" / "run-17.json").read_text() == RECEIPT_BODY


@needs_recording
def test_a_source_with_no_such_attribute_records_normally(tmp_path):
    from newt.recording import SINGLE_ARM_DESCRIPTOR, Session, SimulatedSource

    source = SimulatedSource(SINGLE_ARM_DESCRIPTOR)
    assert not hasattr(source, "calibration_receipt")

    session = Session(source, task="calibration receipt", output_dir=tmp_path / "episodes")
    assert session._calibration_receipt is None
    try:
        session.start_episode()
        path = session.end_episode(keep=True)
    finally:
        session.close()

    assert (path / "episode.json").is_file()
