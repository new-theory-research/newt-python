"""Goldens for the camera bridge — frames reach the writer, and the frame-count
invariant finally has something to count.

Every episode this library has ever written passed the frame-count check by having
no frames: `no cameras — invariant not applicable`. A check that cannot fail is not
evidence. These tests drive a synthetic camera source end to end so the check runs
for real — MCAP markers against the frames actually encoded into the file — with no
hardware anywhere.

The double is deliberately not a camera library. The library learns nothing about
what kind of camera it is talking to, which is the seam these tests exist to hold.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_HAVE_EXTRA = (
    importlib.util.find_spec("mcap") is not None
    and importlib.util.find_spec("google.protobuf") is not None
    and importlib.util.find_spec("numpy") is not None
)
_HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

needs_extra = pytest.mark.skipif(
    not _HAVE_EXTRA, reason="needs the [recording] extra (mcap/protobuf/numpy)"
)
needs_ffmpeg = pytest.mark.skipif(
    not _HAVE_FFMPEG, reason="needs ffmpeg + ffprobe on PATH (the color encoder)"
)


def _record_one(source, tmp_path, seconds: float = 0.6):
    """Record one kept episode from a source and return its directory."""
    import time

    from newt.recording import Session

    session = Session(source, task="wipe the table", output_dir=tmp_path)
    try:
        session.start_episode()
        time.sleep(seconds)
        return session.end_episode(keep=True)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# The bridge contract — half of it is a refusal (runs everywhere)
# ---------------------------------------------------------------------------


def test_declared_cameras_without_a_reader_are_refused(tmp_path):
    """A source that declares cameras but cannot deliver frames is refused at
    construction, not discovered as an episode full of empty video files.

    `cameras` and `read_frames()` are one contract. Half of it would produce an
    episode.json naming cameras that never delivered a single frame — a file that
    lies about what was recorded, which is the exact failure Rule 10 names.
    """
    from newt.recording import SINGLE_ARM_DESCRIPTOR, CameraSpec, Session, SimulatedSource

    source = SimulatedSource(SINGLE_ARM_DESCRIPTOR)
    source.cameras = [CameraSpec("wrist", 64, 48, 30)]

    with pytest.raises(TypeError) as exc:
        Session(source, task="t", output_dir=tmp_path)
    message = str(exc.value)
    assert "read_frames()" in message
    assert "wrist" in message, "the refusal names which cameras were declared"


def test_a_frame_reader_with_nothing_declared_is_refused(tmp_path):
    """A source that reads frames but declares no cameras is refused, rather than
    quietly recording a state-only episode while pulling frames off hardware and
    throwing every one of them away.
    """
    from newt.recording import SINGLE_ARM_DESCRIPTOR, Session, SimulatedSource

    source = SimulatedSource(SINGLE_ARM_DESCRIPTOR)
    source.read_frames = lambda: {}

    with pytest.raises(TypeError) as exc:
        Session(source, task="t", output_dir=tmp_path)
    assert "cameras" in str(exc.value)


def test_the_two_half_contract_refusals_do_not_share_a_string(tmp_path):
    """Two different mistakes, two different messages (Rule 12). A developer who
    reads one must not be sent to fix the other."""
    from newt.recording import SINGLE_ARM_DESCRIPTOR, CameraSpec, Session, SimulatedSource

    declared_only = SimulatedSource(SINGLE_ARM_DESCRIPTOR)
    declared_only.cameras = [CameraSpec("wrist", 64, 48, 30)]
    reader_only = SimulatedSource(SINGLE_ARM_DESCRIPTOR)
    reader_only.read_frames = lambda: {}

    messages = []
    for source in (declared_only, reader_only):
        with pytest.raises(TypeError) as exc:
            Session(source, task="t", output_dir=tmp_path)
        messages.append(str(exc.value))
    assert messages[0] != messages[1]


# ---------------------------------------------------------------------------
# Frames reach the writer — the invariant runs for real
# ---------------------------------------------------------------------------


@needs_extra
@needs_ffmpeg
def test_a_camera_source_puts_video_in_the_episode(tmp_path):
    """Record from a source that declared a camera and the episode has video in it.

    This is the whole card in one assert: `cameras/<id>/color.mp4` exists beside
    `data.mcap`, `episode.json` names the camera the source opened, and the file
    holds real encoded frames. Before this, every episode was joints only.
    """
    from newt.recording._writer import _ffprobe_frame_count
    from tests.fixtures.fake_camera_source import make_camera_source

    source = make_camera_source()
    path = _record_one(source, tmp_path)

    video = path / "cameras" / "cam0" / "color.mp4"
    assert video.exists(), f"no video written; episode holds {sorted(p.name for p in path.iterdir())}"

    meta = json.loads((path / "episode.json").read_text())
    declared = [c["id"] for c in meta["camera_config"]["cameras"]]
    assert declared == ["cam0"], "episode.json declares the camera the source opened"

    encoded = _ffprobe_frame_count(video)
    assert encoded > 0, "the encoder produced a file with no frames in it"
    assert encoded == source.frames_read["cam0"], (
        f"the source handed over {source.frames_read['cam0']} frames and "
        f"{encoded} were encoded — frames went missing between the bridge and the file"
    )


@needs_extra
@needs_ffmpeg
def test_frame_count_invariant_actually_runs_and_passes(tmp_path):
    """`newt episodes validate` compares MCAP markers to encoded frames for real.

    The check has always passed by exemption — no cameras, nothing to count. Here
    it counts: the per-camera line reads `<N> MCAP markers vs <N> video frames`,
    and `not applicable` must be nowhere in the output. A green check that could
    not have gone red is what this test exists to stop.
    """
    from newt.recording import validate
    from tests.fixtures.fake_camera_source import make_camera_source

    path = _record_one(make_camera_source(), tmp_path)
    report = validate(path)

    assert report["valid"], f"episode did not validate: {report}"
    frame_checks = {
        name: detail
        for name, (ok, detail) in _checks(report).items()
        if name.startswith("frame_count")
    }
    assert "frame_count[cam0]" in frame_checks, (
        f"the per-camera invariant did not run; checks were {sorted(_checks(report))}"
    )
    detail = frame_checks["frame_count[cam0]"]
    assert "not applicable" not in detail
    assert "MCAP markers vs" in detail and "video frames" in detail
    markers, frames = _counts(detail)
    assert markers == frames > 0, f"invariant ran on nothing or disagreed: {detail!r}"


@needs_extra
@needs_ffmpeg
def test_three_cameras_each_get_their_own_video_and_their_own_check(tmp_path):
    """Three cameras produce three video files and three independent invariant
    checks — a rig is not one camera, and one camera's frames must never be
    counted against another's."""
    from newt.recording import validate
    from tests.fixtures.fake_camera_source import make_three_camera_source

    path = _record_one(make_three_camera_source(), tmp_path)
    report = validate(path)
    assert report["valid"], f"episode did not validate: {report}"

    checks = _checks(report)
    for cam_id in ("cam0", "cam1", "cam2"):
        assert (path / "cameras" / cam_id / "color.mp4").exists()
        detail = checks[f"frame_count[{cam_id}]"][1]
        markers, frames = _counts(detail)
        assert markers == frames > 0, f"{cam_id}: {detail!r}"


@needs_extra
@needs_ffmpeg
def test_dropped_frames_are_counted_and_never_substituted(tmp_path):
    """A camera that returns nothing on a read drops that frame — counted, and not
    papered over with a duplicate to keep the count even.

    The invariant still passes, because a dropped frame is a frame that was never
    written and never marked. What must NOT happen is the drop being invisible: the
    encoded count is the number of frames the camera actually delivered, strictly
    fewer than the number of reads.
    """
    from newt.recording._writer import _ffprobe_frame_count
    from tests.fixtures.fake_camera_source import FakeCameraSource

    source = FakeCameraSource(drop_frame_every=3)
    path = _record_one(source, tmp_path)

    encoded = _ffprobe_frame_count(path / "cameras" / "cam0" / "color.mp4")
    assert encoded == source.frames_read["cam0"], (
        "encoded frames must equal the frames the source actually handed over — "
        "a substituted or duplicated frame would make this larger"
    )
    assert encoded > 0


# ---------------------------------------------------------------------------
# The old path is not broken by the new one
# ---------------------------------------------------------------------------


@needs_extra
def test_state_only_episodes_still_record_and_still_excuse_the_invariant(tmp_path):
    """A source with no cameras records exactly as it did before: no cameras
    directory, and check 7 reads `no cameras — invariant not applicable`.

    The camera path is additive. A rig that records joints only must not grow a
    camera thread, a cameras directory, or a different validate verdict.
    """
    from newt.recording import SINGLE_ARM_DESCRIPTOR, SimulatedSource, validate

    path = _record_one(SimulatedSource(SINGLE_ARM_DESCRIPTOR), tmp_path, seconds=0.2)
    assert not (path / "cameras").exists()

    report = validate(path)
    assert report["valid"]
    ok, detail = _checks(report)["frame_count_invariant"]
    assert ok and detail == "no cameras — invariant not applicable"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _checks(report: dict) -> dict:
    """Validator checks as {name: (ok, detail)} — the report's own shape, read once
    here so the assertions above stay about behavior."""
    return {c["check"]: (c["ok"], c["detail"]) for c in report["checks"]}


def _counts(detail: str) -> tuple[int, int]:
    """`<N> MCAP markers vs <M> video frames` -> (N, M)."""
    numbers = [int(tok) for tok in detail.replace("—", " ").split() if tok.isdigit()]
    assert len(numbers) >= 2, f"unreadable invariant line: {detail!r}"
    return numbers[0], numbers[1]
