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
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_SRC = Path(__file__).resolve().parent.parent / "src"

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


@needs_extra
@needs_ffmpeg
def test_a_caller_that_owns_its_bridge_may_declare_cameras_and_push_frames(tmp_path):
    """A frontend holding its own camera bridge declares cameras and pushes with
    `feed_frame` — the pattern that method's docstring has always described.

    The half-contract refusal is about a SOURCE that declared cameras it cannot
    deliver. A caller passing `cameras=` is not that source, and demanding a
    `read_frames()` from it would require a second bridge racing the first over
    the same camera ids.
    """
    import numpy as np

    from newt.recording import SINGLE_ARM_DESCRIPTOR, CameraSpec, Session, SimulatedSource
    from newt.recording._writer import _ffprobe_frame_count

    session = Session(
        SimulatedSource(SINGLE_ARM_DESCRIPTOR),
        task="t",
        output_dir=tmp_path,
        cameras=[CameraSpec("cam0", 64, 48, 30)],
    )
    assert [c["id"] for c in session.describe()["cameras"]] == ["cam0"]
    try:
        session.start_episode()
        for tick in range(5):
            frame = np.full((48, 64, 3), tick * 20, dtype=np.uint8)
            session.feed_frame("cam0", frame)
        path = session.end_episode(keep=True)
    finally:
        session.close()

    assert _ffprobe_frame_count(path / "cameras" / "cam0" / "color.mp4") == 5, (
        "every pushed frame must land; the caller's bridge is the only one running"
    )


# ---------------------------------------------------------------------------
# The four refusals — four causes, four strings (Rule 12)
# ---------------------------------------------------------------------------


def test_the_four_camera_failure_causes_never_share_a_string():
    """Four ways the camera half of a recording fails, four messages.

    A developer who lands on one of these must never be sent to fix a different
    problem. Pairwise distinctness is the mechanical form of Rule 12; the causes
    tuple is the enumeration, so a fifth cause added without a string fails here
    rather than rendering something empty at 2am on a bench.
    """
    from newt.recording._seam import CAMERA_FAILURE_CAUSES, camera_failure_message

    assert set(CAMERA_FAILURE_CAUSES) == {
        "will_not_open",
        "stopped_answering",
        "encoder_refused",
        "resolution_not_declarable",
    }
    messages = {c: camera_failure_message(c, "the particulars") for c in CAMERA_FAILURE_CAUSES}
    assert len(set(messages.values())) == len(CAMERA_FAILURE_CAUSES), (
        f"two causes share a message: {messages}"
    )
    for cause, message in messages.items():
        # Rule 12's three parts: what happened, whose problem it is, what to do.
        assert "Yours:" in message or "Ours:" in message, f"{cause} names no owner"
        assert "Do now:" in message, f"{cause} names no next step"
        assert "the particulars" in message, f"{cause} drops its own detail"


def test_an_unwritten_cause_fails_loudly_rather_than_rendering_empty():
    from newt.recording._seam import camera_failure_message

    with pytest.raises(AssertionError):
        camera_failure_message("a_cause_nobody_wrote", "detail")


@needs_extra
def test_a_camera_that_will_not_open_refuses_the_run_and_names_the_camera(tmp_path):
    """A declared camera that does not come up refuses the whole run.

    The failure mode being refused: a rig that catches the open error, drops that
    camera from its list, and records a state-only episode nobody asked for. The
    refusal names which camera, what the driver said, and what to do next, and no
    episode directory is left behind.
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "newt", "record",
            "--source", "tests.fixtures.fake_camera_source:make_camera_that_will_not_open",
            "--json", "--task", "t", "--dest", str(tmp_path),
        ],
        input="", capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(_SRC.parent)}, timeout=60,
    )
    assert proc.returncode == 1, f"expected the construction-refusal code; stdout={proc.stdout!r}"
    assert "'wrist'" in proc.stderr, "the refusal must name which camera would not open"
    assert "no device on the bus answered" in proc.stderr, "and what the driver said"
    assert "Do now:" in proc.stderr, "and what to do next"
    assert sorted(tmp_path.glob("episode_*")) == [], (
        "a camera that would not open must never leave a state-only episode behind"
    )


@needs_extra
@needs_ffmpeg
def test_a_camera_that_stops_answering_refuses_the_episode(tmp_path):
    """Frames stop arriving half-way through and the episode is refused, not kept.

    Committing it would hand someone joints that run the whole take beside video
    that stops early, with nothing in the file saying where. The episode is
    discarded whole — no directory, no partial video, nothing substituted for the
    frames that never came.
    """
    import time as _time

    from newt.recording import CameraCaptureFailed, Session
    from tests.fixtures.fake_camera_source import make_dying_camera_source

    session = Session(make_dying_camera_source(), task="t", output_dir=tmp_path)
    try:
        session.start_episode()
        _time.sleep(0.4)
        with pytest.raises(CameraCaptureFailed) as exc:
            session.end_episode(keep=True)
    finally:
        session.close()

    assert exc.value.cause == "stopped_answering"
    assert "device disconnected mid-stream" in str(exc.value)
    assert sorted(tmp_path.glob("episode_*")) == [], "a refused episode leaves no directory"


@needs_extra
@needs_ffmpeg
def test_a_frame_the_encoder_refuses_refuses_the_episode(tmp_path):
    """A frame that cannot be encoded kills the episode rather than being skipped.

    Skipping it would be an invented decision: the marker timeline and the video
    would disagree by however many frames were quietly dropped on the floor, and
    the invariant would catch it later with no way to say why.
    """
    import time as _time

    from newt.recording import CameraCaptureFailed, Session
    from tests.fixtures.fake_camera_source import make_unencodable_frame_source

    session = Session(make_unencodable_frame_source(), task="t", output_dir=tmp_path)
    try:
        session.start_episode()
        _time.sleep(0.4)
        with pytest.raises(CameraCaptureFailed) as exc:
            session.end_episode(keep=True)
    finally:
        session.close()

    assert exc.value.cause == "encoder_refused"
    assert sorted(tmp_path.glob("episode_*")) == []


@needs_extra
def test_cameras_that_disagree_on_shape_cannot_be_declared_and_are_refused(tmp_path):
    """A rig whose cameras stream different shapes is refused before it records.

    `episode.json` carries one width/height/fps for the whole camera set, so a
    mixed rig would be described by whichever camera came first and described
    wrongly for the rest — a shaped-right number standing in for one nobody
    measured. The refusal names every declaration so the disagreement is visible.
    """
    from newt.recording import Session
    from tests.fixtures.fake_camera_source import make_mismatched_camera_source

    with pytest.raises(TypeError) as exc:
        Session(make_mismatched_camera_source(), task="t", output_dir=tmp_path)
    message = str(exc.value)
    assert "cam0=64x48@30fps" in message and "cam1=128x96@15fps" in message
    assert "Do now:" in message
    assert sorted(tmp_path.glob("episode_*")) == []


@needs_extra
@needs_ffmpeg
def test_the_cli_exits_three_when_the_camera_bridge_dies_mid_episode(tmp_path):
    """`newt record` gives a dead camera bridge its own exit code.

    Three refusals, three codes: 1 for a rig that would not come up at all, 2 for
    the frontend's own refusal (an unwritable destination), 3 for an episode the
    library refused after recording it. An agent driving `--json` routes on the
    code without parsing prose, and none of the three is a traceback.
    """
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "newt", "record",
            "--source", "tests.fixtures.fake_camera_source:make_dying_camera_source",
            "--json", "--task", "t", "--dest", str(tmp_path),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env={**os.environ, "PYTHONPATH": str(_SRC.parent)},
    )
    proc.stdin.write(json.dumps({"cmd": "start"}) + "\n")
    proc.stdin.flush()
    time.sleep(0.5)  # let the camera thread reach its death; not a timing assertion
    proc.stdin.write(json.dumps({"cmd": "stop", "keep": True}) + "\n")
    proc.stdin.flush()
    stdout, stderr = proc.communicate(timeout=60)

    assert proc.returncode == 3, f"expected the refused-episode code; stdout={stdout!r} stderr={stderr!r}"
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    refused = [e for e in events if e["event"] == "refused"]
    assert refused and refused[0]["cause"] == "stopped_answering", (
        f"the refusal must be a routable event, not a traceback: {events}"
    )
    assert "stopped" not in [e["event"] for e in events], "a refused episode never reports as stopped"
    assert sorted(tmp_path.glob("episode_*")) == []


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


@needs_extra
@needs_ffmpeg
def test_dropped_frames_are_reported_per_camera_and_never_rolled_into_state(tmp_path):
    """A camera that misses frames says so, by name, while it is still recording.

    A count kept and never shown is the same as no count. This asserts the two
    surfaces a frontend actually renders — the live status snapshot and the
    dropped-frame report — and that camera drops are reported per camera rather
    than blended into the state total, because a rig that lost a tenth of one
    camera's frames and none of its joints has a specific problem worth naming.
    """
    import time as _time

    from newt.recording import Session
    from tests.fixtures.fake_camera_source import FakeCameraSource

    source = FakeCameraSource(drop_frame_every=2)
    session = Session(source, task="t", output_dir=tmp_path)
    try:
        session.start_episode()
        _time.sleep(0.5)
        st = session.status()
        report = session.dropped_report()
        path = session.end_episode(keep=True)
    finally:
        session.close()

    assert st.dropped_frames.get("cam0", 0) > 0, "a dropped frame that nothing counted"
    assert st.frame_counts.get("cam0", 0) > 0
    assert st.dropped_state == 0, "the joints never dropped; only the camera did"
    assert report is not None and "camera cam0" in report and "frames dropped" in report, (
        f"the report must name the camera that dropped: {report!r}"
    )

    counts, dropped = session.last_episode_frames
    assert dropped["cam0"] >= st.dropped_frames["cam0"] > 0, (
        "the ended episode's drop count carries forward the one status() showed live"
    )
    assert counts["cam0"] == source.frames_read["cam0"], (
        "the writer wrote exactly the frames the source delivered — no substitutes"
    )
    assert validate_ok(path), "a dropped frame is a gap in the video, not a corrupt episode"


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


def validate_ok(path) -> bool:
    from newt.recording import validate

    return bool(validate(path)["valid"])


def _counts(detail: str) -> tuple[int, int]:
    """`<N> MCAP markers vs <M> video frames` -> (N, M)."""
    numbers = [int(tok) for tok in detail.replace("—", " ").split() if tok.isdigit()]
    assert len(numbers) >= 2, f"unreadable invariant line: {detail!r}"
    return numbers[0], numbers[1]
