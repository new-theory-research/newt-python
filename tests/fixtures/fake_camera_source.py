"""A test-double RecordingSource that also delivers camera frames.

Stands in for a rig that opened its own cameras: it declares what it "opened"
through ``cameras`` and hands frames back through ``read_frames()``, which is the
whole camera bridge contract. No hardware, no vendor driver, no camera library —
the point of the double is that the library cannot tell the difference, because
it never learns what kind of camera it is talking to.

Frames are a moving synthetic gradient (bgr24, HxWx3 uint8) rather than a flat
colour, so a frame that silently repeats is visible in the encoded video instead
of hiding in a wall of identical pixels.
"""
from __future__ import annotations

from newt.recording import SINGLE_ARM_DESCRIPTOR, CameraSpec, SimulatedSource


class FakeCameraSource(SimulatedSource):
    """A simulated joint stream plus a synthetic camera bridge.

    ``drop_frame_every`` returns ``None`` for the first camera on that cadence —
    the drop path, so it can be exercised without unplugging anything.
    """

    def __init__(
        self,
        descriptor=SINGLE_ARM_DESCRIPTOR,
        *,
        cameras: "list[CameraSpec] | None" = None,
        drop_frame_every: int = 0,
    ) -> None:
        super().__init__(descriptor)
        self.cameras = list(cameras) if cameras else [CameraSpec("cam0", 64, 48, 30)]
        self._drop_frame_every = drop_frame_every
        self._frame_tick = 0
        self.frames_read: dict[str, int] = {c.id: 0 for c in self.cameras}

    def read_frames(self) -> dict:
        import numpy as np

        self._frame_tick += 1
        out: dict[str, object] = {}
        for i, cam in enumerate(self.cameras):
            if self._drop_frame_every and i == 0 and self._frame_tick % self._drop_frame_every == 0:
                out[cam.id] = None
                continue
            frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            # A ramp that moves with the tick: a stuck or duplicated frame shows up
            # as a still image where the gradient should be sliding.
            frame[:, :, 0] = (self._frame_tick * 7) % 256
            frame[:, :, 1] = np.arange(cam.width, dtype=np.uint8)[None, :]
            frame[:, :, 2] = i * 40
            out[cam.id] = frame
            self.frames_read[cam.id] += 1
        return out

    @property
    def source_kind(self) -> str:
        return f"SIMULATED ({len(self.cameras)} synthetic camera(s), no hardware)"


def make_camera_source():
    """Factory the CLI's --source loader calls with no arguments — one camera."""
    return FakeCameraSource()


def make_three_camera_source():
    """Three cameras, all the same shape. The multi-camera path with nothing
    heterogeneous about it — the heterogeneous case is a format question this
    double deliberately does not pretend to answer."""
    return FakeCameraSource(
        cameras=[
            CameraSpec("cam0", 64, 48, 30),
            CameraSpec("cam1", 64, 48, 30),
            CameraSpec("cam2", 64, 48, 30),
        ]
    )
