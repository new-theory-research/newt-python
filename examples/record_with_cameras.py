"""Record episodes with video from your own rig — a complete camera source.

``newt record`` writes joint traces on its own. To get ``cameras/<id>/color.mp4``
beside ``data.mcap``, your source has to hand it frames. This file is the whole
contract, implemented from scratch on a body that does not exist:

    newt record --task "sort the bin" --source examples.record_with_cameras:make_source

Two optional members are all it takes, and the library finds both by name:

``cameras``
    One :class:`newt.recording.CameraSpec` per camera you actually opened, with
    the id *you* chose. The episode declares exactly these, and one encoder is
    opened per entry. A camera that did not open belongs nowhere in this list —
    raise :class:`newt.recording.CameraOpenError` instead, so a hardware failure
    never turns into an episode that quietly recorded joints only.

``read_frames()``
    One read per declared camera, ``{camera id: frame}``, each frame an HxWx3
    ``bgr24`` array matching the width and height that camera declared. Return
    ``None`` (or omit the key) for a camera that had nothing this read: that is a
    dropped frame, and it is counted and reported. Never substitute, duplicate or
    zero-fill a frame to make a count come out even — the gap is the truth, and
    the markers in the MCAP say exactly where it is.

The frames here are drawn, not captured. Nothing on your machine is a camera, and
this module is not a fallback for one — ``newt record`` will never select it for
you, and a real rig replaces both methods with its own driver's open and read.
What it does prove is that the library holds no camera identity: it never learns
what kind of camera this is, what it is bolted to, or how many there could have
been. Swap in your driver and the SDK does not change.

Needs the recording extra and ffmpeg on PATH::

    pip install "newt[recording]"
"""
from __future__ import annotations

import math

import numpy as np

from newt.recording import CameraSpec, JointState, StateDescriptor

# A benchtop gantry: three linear axes, a rotating head, a gripper. One channel,
# because one thing moves. Your descriptor describes your rig — the library reads
# this and never assumes an arm shape.
DESCRIPTOR = StateDescriptor(
    arms=[{"id": "gantry"}],
    channels=["gantry"],
    joint_names=["x", "y", "z", "yaw", "gripper"],
    state_fields=["positions", "velocities"],
)

# Both cameras run the same width, height and rate. NT episode v0.0.3 carries one
# of each for the whole camera set, so a rig whose cameras disagree is refused at
# construction rather than described wrongly — pick a shape they share.
CAMERAS = [
    CameraSpec("overhead", 320, 240, 15),
    CameraSpec("wrist", 320, 240, 15),
]


class GantrySource:
    """A recording source with cameras, top to bottom.

    Required: ``descriptor`` and ``read_state()``. Optional and both present here:
    ``cameras`` + ``read_frames()`` (the camera bridge) and ``close()``. There is
    deliberately no ``disable_all()`` — this body has no motors to torque off, and
    the library asks for it by name rather than requiring every source to have one.
    """

    def __init__(self) -> None:
        self.descriptor = DESCRIPTOR
        self.cameras = list(CAMERAS)
        self.closed = False
        self._tick = 0
        self._frame_tick = 0

    # --- state: one synchronized snapshot per channel, per read ---------------

    def read_state(self) -> dict:
        """Return one JointState per declared channel. ``None`` for a channel is a
        dropped read — counted by the session, never invented."""
        self._tick += 1
        phase = self._tick * 0.04
        positions = [
            0.30 * math.sin(phase),
            0.20 * math.cos(phase * 0.7),
            0.05 + 0.02 * math.sin(phase * 1.9),
            0.50 * math.sin(phase * 0.3),
            0.02 if (self._tick // 60) % 2 else 0.06,
        ]
        velocities = [
            0.30 * 0.04 * math.cos(phase),
            -0.20 * 0.028 * math.sin(phase * 0.7),
            0.05 * 0.038 * math.cos(phase * 1.9),
            0.50 * 0.012 * math.cos(phase * 0.3),
            0.0,
        ]
        return {"gantry": JointState(positions=positions, velocities=velocities)}

    # --- cameras: one read per declared camera, per frame ---------------------

    def read_frames(self) -> dict:
        """Read every declared camera once. A real driver blocks here for as long
        as the hardware takes; that is fine, this runs on its own thread and never
        delays state capture."""
        self._frame_tick += 1
        return {cam.id: self._draw(cam, i) for i, cam in enumerate(self.cameras)}

    def _draw(self, cam: CameraSpec, index: int):
        """Stand-in for ``driver.read()`` — an HxWx3 bgr24 frame at the declared
        size. The bar sweeps with the tick, so a frame that silently repeats shows
        up as a still image instead of hiding in a wall of identical pixels."""
        frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        frame[:, :, 1] = np.linspace(0, 255, cam.width, dtype=np.uint8)[None, :]
        frame[:, :, 2] = 60 + 60 * index
        bar = (self._frame_tick * 9 + index * 40) % cam.width
        frame[:, bar : bar + 12, :] = 255
        return frame

    # --- the rest of the seam -------------------------------------------------

    @property
    def source_kind(self) -> str:
        """What the preflight prints. Say what it is — an operator reading
        'SYNTHETIC' knows immediately that no lens is pointed at anything."""
        return "SYNTHETIC gantry (2 drawn cameras, no hardware)"

    def close(self) -> None:
        """Release the rig. A real one closes its driver handles here — including
        the cameras it opened, since whoever opened them owns closing them."""
        self.closed = True


def make_source() -> GantrySource:
    """The factory ``--source MODULE:FACTORY`` calls, with no arguments.

    A real one opens the hardware here and raises ``CameraOpenError(camera_id,
    detail)`` if a declared camera will not come up::

        try:
            handle = my_driver.open(serial)
        except MyDriverError as exc:
            raise CameraOpenError("overhead", str(exc)) from exc

    Refusing beats returning a shorter camera list: the run stops with nothing
    written, instead of recording an episode that is missing a camera nobody
    noticed until they went looking for the video months later.
    """
    return GantrySource()
