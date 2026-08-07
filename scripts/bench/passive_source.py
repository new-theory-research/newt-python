"""A passive recording source for proving the collect page on a real bench.

**What this is for.** The clay in `apps/collect` has to be shown working against
a real rig — a live view with real video in it, real episodes on disk, real
delivery states in the completed list. It does not have to be shown moving a
robot to do that, and it should not: the kit's live factory energizes both arms
and sweeps them to a home pose on connect, which is a physical act that belongs
to a person standing at the bench with a hand on the kill switch, not to an
agent driving the software.

So this source reads the cameras and nothing else. Nothing here opens a serial
port, constructs a driver, or sends a command to an arm. Run it and the arms sit
exactly where the operator left them.

**What is real and what is not, said plainly.** The video is real: three
RealSense units, read frame by frame off V4L2 through the same OpenCV backend
the starter kit uses. The joints are **not** real — they are `newt`'s own
`SimulatedSource` tracing sinusoids around a non-zero rest pose, and every
surface that can say so does: `source_kind` says it in capitals, the live view's
footer prints it, and the preflight banner carries it into the terminal. A
substitution nobody can see is the thing Rule 10 forbids; a substitution
declared in four places is a bench instrument.

**Why it is committed.** The last proof of this kind was driven by a shell
wrapper that was never committed and is now gone, which left its own receipts
unreproducible — the bench README had to reconstruct the command line from the
run's output. This file is the fix for that: the instrument ships next to the
thing it proves.

**Why it does not use the kit's camera code.** The kit opens cameras through
lerobot's `RealSenseCamera`, and lerobot pins `rerun-sdk<0.23` while the live
view needs `>=0.34.1` — the two cannot share an environment (see the
`[tool.uv] conflicts` note in newt-python's pyproject). Going through OpenCV
directly keeps this instrument installable beside the view it exists to fill.
That is a deliberate divergence from the product path, and it is why this is a
bench instrument and not a source anybody should record training data with.

Usage, from a machine with the cameras attached:

    newt record --source passive_source:cameras_only \
        --task "..." --dest ~/episodes --view --control --push \
        --page-dir /path/to/apps/collect/dist

`NT_BENCH_CAMERAS` selects which V4L2 indices to open, comma-separated. With
nothing set it opens every device that identifies as a RealSense colour stream
and delivers a frame when asked.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from newt.recording import (
    CameraOpenError,
    CameraSpec,
    SimulatedSource,
    StateDescriptor,
)

#: The mode every camera is asked for. One shape for the whole set, because NT
#: episode v0.0.3 declares a single width/height/fps for all of them — a rig whose
#: cameras disagree is refused by the session rather than described wrongly.
WIDTH, HEIGHT, FPS = 640, 480, 30

#: The joint stream's shape. Real names in the real order, so an episode written
#: here has the channel and joint layout a live run would write and only the
#: numbers differ.
DESCRIPTOR = StateDescriptor(
    arms=[{"id": "bench"}],
    channels=["bench/leader", "bench/follower"],
    joint_names=[
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
    ],
    state_fields=["positions", "velocities", "efforts"],
)


#: The ioctl that lists a V4L2 node's pixel formats, and the one format that is
#: unambiguously a colour stream. Depth nodes advertise `Z16`; the infrared nodes
#: advertise `GREY`/`Y8I`/`Y12I` alongside `UYVY`, and on some units that `UYVY`
#: really is colour while on others it is the projector's dot pattern. `YUYV`
#: alone means colour on every unit tried, so it is the only thing scanned for.
_VIDIOC_ENUM_FMT = 0xC0405602
_FMTDESC = "III32sI16s"
_COLOUR_FORMAT = "YUYV"


def _node_formats(index: int) -> list[str]:
    """The four-character pixel formats `/dev/video{index}` advertises."""
    import fcntl
    import struct

    formats: list[str] = []
    try:
        with open(f"/dev/video{index}", "rb") as handle:
            for slot in range(16):
                buffer = bytearray(struct.pack(_FMTDESC, slot, 1, 0, b"", 0, b""))
                try:
                    fcntl.ioctl(handle, _VIDIOC_ENUM_FMT, buffer, True)
                except OSError:
                    break
                *_, pixelformat, _ = struct.unpack(_FMTDESC, bytes(buffer))
                formats.append(pixelformat.to_bytes(4, "little").decode("ascii", "replace"))
    except OSError:
        return []
    return formats


def _realsense_indices() -> tuple[list[int], list[int]]:
    """The colour nodes to open, and the other RealSense nodes that exist.

    A RealSense unit publishes several V4L2 nodes and most of them are not the
    picture you mean: depth, and two or three infrared streams that open cleanly
    and hand back the projector's dot pattern. Recording one of those into a
    dataset as though it were the scene is exactly the silently-wrong data Rule 10
    is about, so the scan picks only nodes whose format list is unambiguously
    colour and *reports the rest* rather than quietly including or excluding them.

    On rigs where a camera's colour stream shares a node with its infrared (the
    short-range units do this), the scan will not find it. That is why the second
    return value exists and why the caller prints it: the fix is to name the
    indices in ``NT_BENCH_CAMERAS`` after looking at a frame from each, not to
    loosen this rule until something appears.
    """
    colour: list[int] = []
    others: list[int] = []
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        index = int(node.name.removeprefix("video"))
        formats = _node_formats(index)
        if not formats:
            continue
        if formats == [_COLOUR_FORMAT]:
            colour.append(index)
        else:
            others.append(index)
    return sorted(colour), sorted(others)


class _Camera:
    """One opened V4L2 node, proven by a frame before it is declared.

    The proof matters: half the nodes a RealSense publishes open cleanly and then
    never deliver anything. Declaring one of those would write a camera into
    `episode.json` that contributes no frames, and the session's frame-count
    invariant would fail at commit — after the take, rather than before it.
    """

    def __init__(self, index: int) -> None:
        import cv2

        self._cv2 = cv2
        self._index = index
        self._capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            raise CameraOpenError(f"video{index}", "VideoCapture would not open it")

        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self._capture.set(cv2.CAP_PROP_FPS, FPS)

        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._capture.release()
            raise CameraOpenError(
                f"video{index}", "it opened but delivered no first frame"
            )

        height, width = frame.shape[:2]
        if (width, height) != (WIDTH, HEIGHT):
            # Declared as delivered, not as requested. A CameraSpec that repeats
            # what we asked for would describe every episode wrongly the moment a
            # driver quietly negotiated something else.
            print(
                f"[bench] video{index} negotiated {width}x{height}, not "
                f"{WIDTH}x{HEIGHT}; declaring what it delivers.",
                file=sys.stderr,
                flush=True,
            )
        self.spec = CameraSpec(
            id=f"video{index}", width=width, height=height, fps=FPS
        )

    def read(self):
        ok, frame = self._capture.read()
        # None is a dropped frame, which the Session counts and reports. It is
        # never a repeat of the last one: a recorded frame is a frame that was read.
        return frame if ok else None

    def release(self) -> None:
        self._capture.release()


class PassiveCameraSource:
    """Real cameras, declared-simulated joints, and no way to move an arm.

    The joint half is delegated to `newt`'s `SimulatedSource` rather than
    reimplemented, so the numbers in a bench episode come from the same generator
    the SDK's own tests use and this file has no opinion about what a joint is.
    """

    def __init__(self, cameras: list[_Camera]) -> None:
        self.descriptor = DESCRIPTOR
        self._joints = SimulatedSource(DESCRIPTOR)
        self._cameras = cameras
        # Bound per instance, the way the seam discovers them: a source with
        # read_frames() and no cameras is refused, and rightly.
        if cameras:
            self.cameras = [camera.spec for camera in cameras]
            self.read_frames = self._read_frames

    @property
    def source_kind(self) -> str:
        """What the preflight banner and the view's footer print.

        Both halves in one sentence, in capitals, because this string is the main
        place a person reading a bench episode months later finds out that half of
        it was generated.
        """
        return (
            f"LIVE CAMERAS ({len(self._cameras)}) + SIMULATED JOINTS "
            f"(passive bench source — no arm was connected or energized)"
        )

    def read_state(self):
        return self._joints.read_state()

    def _read_frames(self) -> dict:
        return {camera.spec.id: camera.read() for camera in self._cameras}

    def disable_all(self) -> None:
        """The kill hook. There is nothing energized, and it says so rather than
        silently succeeding — a kill that quietly does nothing is a kill nobody
        should trust the next time."""
        print(
            "[bench] disable_all(): nothing to torque off — this source never "
            "connected to an arm.",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        for camera in self._cameras:
            camera.release()


def cameras_only():
    """Build the source. `--source passive_source:cameras_only`.

    Every camera that comes up is opened before anything is declared, and one
    that will not come up refuses the whole run — a rig that quietly dropped a
    camera from the list would record a shorter episode and tell nobody.
    """
    named = os.environ.get("NT_BENCH_CAMERAS")
    if named:
        indices = [int(part) for part in named.split(",") if part.strip()]
        others: list[int] = []
    else:
        indices, others = _realsense_indices()

    if not indices:
        raise CameraOpenError(
            "none",
            "no V4L2 node on this machine advertises an unambiguous colour "
            f"stream (the other RealSense nodes here are {others or 'none'}).\n"
            "Yours: the cameras may be attached and still not be found this way "
            "— a short-range unit puts its colour stream on the same node as its "
            "infrared, and this scan will not guess which is which.\n"
            "Do now: capture a frame from each node, look at it, and name the "
            "ones you meant in NT_BENCH_CAMERAS (e.g. NT_BENCH_CAMERAS=4,10,14)",
        )

    opened: list[_Camera] = []
    try:
        for index in indices:
            # Every failure is a refusal, whether the index was named or scanned.
            # A run that quietly opened four of five cameras would write an
            # episode that is short one view and say nothing about it.
            opened.append(_Camera(index))
    except Exception:
        for camera in opened:
            camera.release()
        raise

    print(
        f"[bench] {len(opened)} camera(s) open: "
        f"{', '.join(camera.spec.id for camera in opened)}. No arm was touched.",
        file=sys.stderr,
        flush=True,
    )
    if others:
        print(
            f"[bench] not opened: {', '.join(f'video{i}' for i in others)} "
            f"(depth and infrared nodes). Name them in NT_BENCH_CAMERAS if one "
            f"of them is the colour stream you meant.",
            file=sys.stderr,
            flush=True,
        )
    return PassiveCameraSource(opened)
