"""The seam holds for a body the library has never met.

The camera bridge was built against one rig. A bridge that works for one rig may
just be that rig's shape spelled differently — the way to find out is to point the
same `newt record` at a completely different body and see whether the library
needs anything new.

The second body here is `examples/record_with_cameras.py`: different joints,
different channel, different camera count, different camera ids, different frame
size and rate, and written from scratch against the seam rather than by
subclassing anything the library ships. It records video through the same verb,
and the library gains nothing — no camera conditional, no vendor type, no id it
recognizes. The mechanical form of that claim is the last test in this file: the
names this body chose appear in the episode it wrote and nowhere in the SDK.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_EXAMPLE = _ROOT / "examples" / "record_with_cameras.py"
_SPEC = "examples.record_with_cameras:make_source"

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


@pytest.fixture(scope="module")
def second_body_episode(tmp_path_factory):
    """One episode recorded from the second body through the real CLI.

    Driven as a subprocess over `--json` — the user path, not a library call, so
    nothing about how the frontend loads a source is stubbed out. Recorded once
    and shared: it is a real encode, and the assertions below are about one
    episode rather than one apiece.
    """
    if not (_HAVE_EXTRA and _HAVE_FFMPEG):
        pytest.skip("needs the [recording] extra and ffmpeg on PATH")

    dest = tmp_path_factory.mktemp("second-body")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "newt", "record",
            "--source", _SPEC,
            "--json", "--task", "sort the bin", "--dest", str(dest),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env={**os.environ, "PYTHONPATH": str(_ROOT)},
    )
    proc.stdin.write(json.dumps({"cmd": "start"}) + "\n")
    proc.stdin.flush()
    time.sleep(1.0)
    proc.stdin.write(json.dumps({"cmd": "stop", "keep": True}) + "\n")
    proc.stdin.flush()
    proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
    proc.stdin.flush()
    stdout, stderr = proc.communicate(timeout=120)

    assert proc.returncode == 0, f"record failed ({proc.returncode}); stderr={stderr!r}"
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    stopped = [e for e in events if e["event"] == "stopped"]
    assert stopped, f"no episode was kept: {events}"
    return {"events": events, "stopped": stopped[0], "path": Path(stopped[0]["path"])}


@needs_extra
@needs_ffmpeg
def test_a_second_body_records_video_through_the_unchanged_verb(second_body_episode):
    """A different rig, the same `newt record`, and its cameras are in the file.

    Two cameras this time, not one; 320x240 at 15fps, not the first body's shape;
    ids the body chose for itself. `episode.json` declares exactly what it opened
    and both videos are on disk beside `data.mcap`.
    """
    path = second_body_episode["path"]
    meta = json.loads((path / "episode.json").read_text())

    declared = [c["id"] for c in meta["camera_config"]["cameras"]]
    assert declared == ["overhead", "wrist"], (
        f"episode.json must name the cameras this body opened, got {declared}"
    )
    assert (meta["camera_config"]["width"], meta["camera_config"]["height"]) == (320, 240)
    assert meta["camera_config"]["fps"] == 15

    for cam_id in declared:
        video = path / "cameras" / cam_id / "color.mp4"
        assert video.exists() and video.stat().st_size > 0, (
            f"camera {cam_id} declared but no video written; episode holds "
            f"{sorted(p.name for p in path.iterdir())}"
        )

    frame_counts = second_body_episode["stopped"]["frame_counts"]
    assert set(frame_counts) == {"overhead", "wrist"} and all(n > 0 for n in frame_counts.values()), (
        f"both cameras must have delivered frames: {frame_counts}"
    )
    assert second_body_episode["stopped"]["state_count"] > 0, "joints recorded too, not video alone"


@needs_extra
@needs_ffmpeg
def test_the_invariant_runs_per_camera_on_the_second_bodys_episode(second_body_episode):
    """Check 7 counts, twice, and could have gone red either time.

    A rig with two cameras gets two independent comparisons — one camera's frames
    are never counted against the other's, and neither reads `not applicable`.
    """
    from newt.recording import validate

    report = validate(second_body_episode["path"])
    assert report["valid"], f"the second body's episode did not validate: {report}"

    checks = {c["check"]: c["detail"] for c in report["checks"]}
    for cam_id in ("overhead", "wrist"):
        name = f"frame_count[{cam_id}]"
        assert name in checks, f"the invariant did not run for {cam_id}; checks were {sorted(checks)}"
        detail = checks[name]
        assert "not applicable" not in detail
        counts = [int(tok) for tok in detail.replace("—", " ").split() if tok.isdigit()]
        assert len(counts) >= 2 and counts[0] == counts[1] > 0, (
            f"{name} ran on nothing or disagreed: {detail!r}"
        )


@needs_extra
@needs_ffmpeg
def test_the_second_body_recorded_a_different_shape_than_the_first(second_body_episode):
    """The proof is only worth something if the two bodies actually differ.

    A second body that happens to be the first body renamed proves nothing about
    the seam. This one declares one arm and one state channel of its own naming,
    and the episode carries them — so the library described a rig it was not
    built against.
    """
    from newt.recording import validate

    path = second_body_episode["path"]
    meta = json.loads((path / "episode.json").read_text())
    assert [a["id"] for a in meta["robot_config"]["arms"]] == ["gantry"], (
        f"the episode must carry this body's arm, not a default: {meta['robot_config']}"
    )

    channel = {c["check"]: c["detail"] for c in validate(path)["checks"]}["robot_state_channel"]
    assert channel.startswith("robot_state/gantry"), (
        f"the state channel must be the one this body declared: {channel!r}"
    )


@needs_extra
def test_a_camera_source_needs_no_behavior_from_the_library():
    """Writing a camera source imports data types and nothing else.

    The whole camera contract is two members the library finds by name. If a
    future change makes a source import `Session`, a camera helper, or a base
    class to deliver frames, the seam grew a requirement and this fails — which
    is the point. A rig should not have to inherit anything to hand over pixels.
    """
    tree = ast.parse(_EXAMPLE.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("newt")
        for alias in node.names
    }
    assert imported <= {"CameraSpec", "JointState", "StateDescriptor", "CameraOpenError"}, (
        f"the example reaches into the library for {sorted(imported)} — a camera "
        "source should need only the data types it fills in"
    )


def test_the_library_holds_no_name_this_body_chose():
    """The ids travel as data; the SDK never learns them.

    This body called its cameras `overhead` and `wrist` and its rig a `gantry`.
    Those strings belong in the episode it wrote — the test above reads them back
    out of `episode.json` — and nowhere in the source of the library that wrote
    it. The same claim has to hold for every rig's serials, vendor types and
    camera names, and this body is the one whose names we get to check.

    Looks for the names as string literals, which is what special-casing a rig
    actually looks like (`if cam_id == "overhead"`), rather than as substrings —
    "no overhead" in a docstring about performance is English, not a camera.
    Runs without the extra: it is a read of the tree, not a recording.
    """
    literal = re.compile(r"""['"](?:overhead|gantry)['"]""", re.IGNORECASE)
    hits = [
        f"{path.relative_to(_SRC)}:{i}"
        for path in sorted(_SRC.rglob("*.py"))
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
        if literal.search(line)
    ]
    assert hits == [], (
        "the library learned a name belonging to a rig: " + "; ".join(hits)
    )
