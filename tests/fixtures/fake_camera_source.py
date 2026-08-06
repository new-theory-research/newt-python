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

from newt.recording import (
    SINGLE_ARM_DESCRIPTOR,
    CameraOpenError,
    CameraSpec,
    SimulatedSource,
)


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


class DyingCameraSource(FakeCameraSource):
    """A camera that answers a few reads and then stops — the mid-episode death.

    Standing in for a unit that re-enumerates on the USB bus half-way through a
    take. The reads that succeeded were real; the ones after it never arrive, and
    nothing is substituted to cover them.
    """

    def __init__(self, *, raise_after: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self._raise_after = raise_after

    def read_frames(self) -> dict:
        if self._frame_tick >= self._raise_after:
            raise OSError("device disconnected mid-stream")
        return super().read_frames()


class _UnencodableFrame:
    """A frame the encoder path cannot take: it refuses to become pixels.

    A real one is a frame whose layout is not the bgr24 its CameraSpec declared.
    This double reaches the same place — the write that turns a frame into bytes —
    without needing a camera that lies about its own format.
    """

    def __array__(self, *args, **kwargs):
        raise ValueError("frame is not the bgr24 layout its camera declared")


class UnencodableFrameSource(FakeCameraSource):
    """Delivers frames the writer cannot encode. The episode must be refused, not
    committed with the frames that happened to make it in."""

    def read_frames(self) -> dict:
        self._frame_tick += 1
        return {cam.id: _UnencodableFrame() for cam in self.cameras}


def make_camera_that_will_not_open():
    """A rig whose declared camera never comes up. The source refuses the whole
    run rather than returning a shorter camera list — a camera quietly dropped
    from the list is a silently state-only episode."""
    raise CameraOpenError(
        "wrist", "no device on the bus answered (simulated bring-up failure)"
    )


def make_dying_camera_source():
    """Answers one read, then stops answering."""
    return DyingCameraSource(raise_after=1)


def make_unencodable_frame_source():
    return UnencodableFrameSource()


def make_mismatched_camera_source():
    """Two cameras whose streams disagree in shape — the episode format carries
    one width/height/fps for the whole set, so this cannot be declared honestly."""
    return FakeCameraSource(
        cameras=[
            CameraSpec("cam0", 64, 48, 30),
            CameraSpec("cam1", 128, 96, 15),
        ]
    )


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
