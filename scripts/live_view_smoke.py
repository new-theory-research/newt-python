"""Drive a live view off the simulated source, with no hardware anywhere.

A bench tool, not a test: it starts a session with a declared robot, prints the
URL, and holds it open so a browser (or a screenshotter) can look. Everything it
publishes is a sinusoid — nothing here reads or moves an arm.

    python scripts/live_view_smoke.py --urdf path/to/robot.urdf --seconds 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newt.live import LiveView  # noqa: E402
from newt.recording import SINGLE_ARM_DESCRIPTOR, Session, SimulatedSource  # noqa: E402
from newt.recording._seam import JointDrive, ViewDeclaration  # noqa: E402


class _WithFakeCameras:
    """A source with cameras that are a moving test pattern, for proving the view
    on a laptop. Nothing here is a camera and the pattern says so."""

    def __init__(self, inner, count: int) -> None:
        import numpy as np

        from newt.recording import CameraSpec

        self._inner = inner
        self._np = np
        self.descriptor = inner.descriptor
        self._ids = [f"fake-{i}" for i in range(count)]
        self.cameras = [CameraSpec(id=i, width=640, height=480, fps=30) for i in self._ids]
        self._tick = 0
        self.source_kind = f"SIMULATED joints + {count} FAKE camera(s) — a test pattern"

    def read_state(self):
        return self._inner.read_state()

    def read_frames(self):
        np = self._np
        self._tick += 1
        frames = {}
        for index, cam_id in enumerate(self._ids):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            x = (self._tick * 7 + index * 90) % 600
            frame[:, :, index % 3] = 40
            frame[100:380, x : x + 40] = 255
            frames[cam_id] = frame
        return frames

    def close(self):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, help="a robot description to draw")
    parser.add_argument("--joints", default="joint_0,joint_1,joint_2,joint_3,joint_4,joint_5")
    parser.add_argument("--prefix", default="arm")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--dest", default="/tmp/live-view-smoke-episodes")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--record-after", type=float, default=8.0)
    parser.add_argument("--fake-cameras", type=int, default=0)
    args = parser.parse_args()

    declaration = None
    if args.urdf:
        names = [n for n in args.joints.split(",") if n]
        declaration = ViewDeclaration(
            urdf_path=str(args.urdf),
            entity_prefix=args.prefix,
            drives=tuple(
                JointDrive(urdf_joint=name, channel=SINGLE_ARM_DESCRIPTOR.channels[0], index=i)
                for i, name in enumerate(names)
            ),
            joint_convention=(
                "SMOKE RUN — the values driving this are a sinusoid from "
                "newt.recording.SimulatedSource, not an arm. No convention is being "
                "vouched for because nothing physical produced these numbers."
            ),
        )

    source = SimulatedSource(SINGLE_ARM_DESCRIPTOR)
    if args.fake_cameras:
        source = _WithFakeCameras(source, args.fake_cameras)
    session = Session(
        source,
        task="live view smoke — nothing is moving",
        output_dir=args.dest,
    )
    view = LiveView(
        descriptor=session.descriptor,
        task=session.describe()["task"],
        camera_ids=session.camera_ids,
        declaration=declaration,
        port=args.port,
        grpc_port=args.port + 1,
    )
    url = view.start()
    session.attach_observer(view)
    print(f"[view] {url}", flush=True)

    deadline = time.monotonic() + args.seconds
    recording = False
    started = time.monotonic()
    try:
        while time.monotonic() < deadline:
            if not recording and time.monotonic() - started >= args.record_after:
                episode = session.start_episode()
                recording = True
                print(f"[rec] episode {episode} started while the view stays up", flush=True)
            time.sleep(0.5)
        if recording:
            path = session.end_episode(keep=True)
            print(f"[rec] kept {path}", flush=True)
        for name, exc in session.observer_failures():
            print(f"[view] observer {name} dropped: {exc!r}", file=sys.stderr)
    finally:
        session.close()
        view.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
