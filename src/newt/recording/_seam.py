"""The recording seam — what a Session consumes, and the only surface a new
embodiment implements.

Ported from rebot-bench ``embodiment.py`` (the 2026-06-11 bench seed). The seam
is deliberately tiny: read state, describe the stream, optionally torque off.
No session logic, no control, no execution lives here — that behavior belongs to
``newt.recording.Session`` and must not leak into the protocol.

This module is **featherweight on purpose**: stdlib only, no ``mcap`` / ``cv2`` /
``protobuf``. It is safe to import without the ``recording`` extra, so a
descriptor or a simulated source can be constructed (and the layering tested)
even in a core-only install. The heavy machinery (the MCAP writer, the protobuf
schema) lives in sibling modules behind the lantern guard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

# --- the four ways the camera half of a recording fails ---------------------
#
# Four causes, four strings, one place. A reader who lands on one of these must
# never be sent to fix a different problem, so the strings live together where
# they can be read against each other (Rule 12).

CAMERA_FAILURE_CAUSES = (
    "will_not_open",
    "stopped_answering",
    "encoder_refused",
    "resolution_not_declarable",
)


def camera_failure_message(cause: str, detail: str) -> str:
    """The message for one camera failure cause. ``detail`` is the concrete
    particulars — which camera, what the driver said, which shapes disagreed."""
    if cause == "will_not_open":
        return (
            f"A camera the rig declared would not open: {detail}.\n"
            "Yours: the camera never came up. A rig that quietly dropped it from "
            "the list would have recorded a shorter episode and told no one, so "
            "the whole run is refused instead.\n"
            "Do now: nothing was recorded and no episode directory exists. Check "
            "the camera is connected and not already held open by another process, "
            "then run the same command again."
        )
    if cause == "stopped_answering":
        return (
            "A camera stopped answering part-way through the episode: the source's "
            f"read_frames() raised ({detail}).\n"
            "Yours: the camera or its driver quit mid-capture — the frames simply "
            "stopped arriving, and nothing was substituted to cover the gap.\n"
            "Do now: the episode was discarded, so nothing partial was written. "
            "Check the camera's connection (a USB re-enumeration mid-run does this) "
            "and record again."
        )
    if cause == "encoder_refused":
        return (
            f"The video encoder refused a frame part-way through the episode ({detail}).\n"
            "Ours: the frame reached us and we could not encode it — most often a "
            "frame whose size or pixel layout is not the one its camera declared.\n"
            "Do now: the episode was discarded, so nothing partial was written. "
            "Check that every frame read_frames() returns matches the width, height "
            "and 3-channel bgr24 layout its CameraSpec declares."
        )
    if cause == "resolution_not_declarable":
        return (
            "The declared cameras do not agree on width, height and frame rate, so "
            f"the episode cannot honestly declare one of each: {detail}.\n"
            "Ours: NT episode v0.0.3 carries a single width/height/fps for the whole "
            "camera set, so a mixed rig would be described by whichever camera came "
            "first and described wrongly for every other one.\n"
            "Do now: declare the cameras that share a shape (one camera is a valid "
            "rig), or configure the rig so every declared camera streams at the same "
            "width, height and frame rate. Per-camera dimensions are a change to the "
            "episode format, not a setting."
        )
    # Never reached through the call sites above; kept so an added cause that
    # forgets its string fails loudly instead of rendering an empty one.
    raise AssertionError(f"no message written for camera failure cause {cause!r} ({detail})")


class CameraOpenError(RuntimeError):
    """A declared camera would not open — raised by a **source** during bring-up.

    A source that opens cameras owns the failure when one of them does not come
    up. Raising this (rather than returning a shorter ``cameras`` list) is what
    keeps a hardware failure from becoming a silently state-only episode. The
    library supplies the string so every rig says the same thing; it learns
    nothing about the camera beyond the opaque id the source chose.
    """

    def __init__(self, camera_id: str, detail: str) -> None:
        self.camera_id = camera_id
        self.detail = detail
        super().__init__(
            camera_failure_message("will_not_open", f"camera {camera_id!r} — {detail}")
        )


class CameraCaptureFailed(RuntimeError):
    """The camera bridge failed part-way through an episode, so the episode was
    refused rather than committed with video that ends before its joints do.
    Raised by ``Session.end_episode(keep=True)``; carries the cause key so a
    frontend can route on it without matching on prose."""

    def __init__(self, cause: str, message: str) -> None:
        self.cause = cause
        super().__init__(message)


@dataclass(frozen=True)
class StateDescriptor:
    """The static shape of one embodiment's state stream, read by the Session and
    the episode writer so neither has to know which arm it is talking to.

    ``arms`` lists the physical arms; their entries land in ``episode.json`` under
    ``robot_config.arms``. ``channels`` lists the ``robot_state/<suffix>`` topics
    this source emits — one entry per MCAP channel, in the order ``read_state``
    keys them. A single kinesthetic arm has one channel whose suffix is the arm id
    (``"b601"`` -> ``robot_state/b601``); a leader/follower teleop pair emits two
    channels per arm (``"left/leader"`` -> ``robot_state/left/leader``), the
    canonical v0.0.3 bimanual convention.

    ``joint_names`` is the shared joint ordering (gripper last); ``state_fields``
    names the RobotState fields this embodiment actually populates, for the
    preflight contract print. Nothing here is speculative session state.
    """

    arms: list[dict]
    channels: list[str]
    joint_names: list[str]
    state_fields: list[str] = field(
        default_factory=lambda: ["positions", "velocities", "efforts"]
    )


@dataclass
class JointState:
    """One synchronized snapshot of an arm's motors. Arrays are joint-order,
    gripper last. Fields an embodiment does not expose are left empty — the
    format permits omitting them, and an empty array is never an invented value.
    """

    positions: list[float]
    velocities: list[float] = field(default_factory=list)
    efforts: list[float] = field(default_factory=list)
    rotor_temperatures: list[float] = field(default_factory=list)
    driver_temperatures: list[float] = field(default_factory=list)
    status_codes: list[int] = field(default_factory=list)


@runtime_checkable
class RecordingSource(Protocol):
    """The recording seam. What a ``Session`` consumes, and the only surface a new
    embodiment implements. Deliberately minimal — read, describe, optional kill.

    ``read_state`` returns a per-channel dict keyed by the descriptor's channel
    suffixes; a value of ``None`` for a channel is a dropped read for that channel
    (counted by the Session, never swallowed). ``disable_all`` is the optional
    kill-switch hook (torque-off); a source with no actuation may omit it.
    ``close`` releases the connection; a source with nothing to release may omit it.

    **Cameras are the same kind of optional member**, discovered by ``getattr`` the
    way ``disable_all`` and ``close`` are — not part of the required surface, so a
    state-only source implements nothing new. A source that captures video exposes
    BOTH of:

    ``cameras``
        A list of :class:`newt.recording.CameraSpec` — one per camera the source
        actually opened, with the id the source chose. The Session declares exactly
        these in ``episode.json`` and opens one encoder per entry. A camera that
        failed to open belongs nowhere in this list; refusing the whole run is the
        source's call, and a source that quietly drops it from the list has turned
        a hardware failure into a silently shorter episode.

    ``read_frames()``
        One read per declared camera, returned as ``{camera id: frame}`` where a
        frame is an HxWx3 ``bgr24`` array. ``None`` (or an absent key) is a dropped
        frame for that camera — counted, never substituted. The call may block as
        the hardware blocks; it runs on its own thread and never delays state
        capture.

    The division of labor this expresses: **the source opens, reads and closes
    cameras; the library encodes, timestamps and enforces the frame-count
    invariant.** The library holds no camera identity — ids travel as opaque
    strings the source chose, exactly as channel names already do.
    """

    descriptor: StateDescriptor

    def read_state(self) -> dict[str, JointState | None]:
        ...


class SimulatedSource:
    """A hardware-free RecordingSource that emits a deterministic joint stream.

    This is how the validator confirms format conformance in CI and how a user
    drives ``newt record --simulate`` with no arm attached. It is the generalized
    port of rebot-bench's two simulated sources — pass a descriptor and it drives
    however many channels that descriptor declares (single kinesthetic arm or
    bimanual leader/follower).

    Joint positions trace slow sinusoids around a non-zero rest pose, so a
    'go to zero' bug anywhere downstream stays visible. ``drop_every`` injects a
    dropped read (returns ``None`` on one channel) on a fixed cadence so the
    per-episode dropped-frame report is exercised; silence on a drop is the
    named disease.
    """

    # A non-zero rest pose so a 'snap to zero' bug downstream is visible. Length
    # is clamped/padded to the descriptor's joint count at sample time.
    _REST_POSE: ClassVar[list[float]] = [
        0.0, math.pi / 3.0, math.pi / 6.0, math.pi / 5.0, 0.0, 0.0, 0.0
    ]

    def __init__(self, descriptor: StateDescriptor, drop_every: int = 0) -> None:
        self.descriptor = descriptor
        self._t = 0
        self._drop_every = drop_every
        self.disabled = False

    def read_state(self) -> dict[str, JointState | None]:
        self._t += 1
        out: dict[str, JointState | None] = {}
        for i, channel in enumerate(self.descriptor.channels):
            # Inject a dropped read on the first channel at the cadence; the
            # report must name it. A per-channel phase offset keeps each channel's
            # stream distinct and non-invented.
            if self._drop_every and i == 0 and self._t % self._drop_every == 0:
                out[channel] = None
                continue
            out[channel] = self._sample(self._t, i)
        return out

    def _sample(self, tick: int, channel_index: int) -> JointState:
        n = len(self.descriptor.joint_names)
        rest = (self._REST_POSE + [0.0] * n)[:n]
        phase = tick * 0.05 + channel_index * 0.3
        positions = [r + 0.08 * math.sin(phase + j) for j, r in enumerate(rest)]
        velocities = [0.08 * math.cos(phase + j) for j in range(n)]
        return JointState(
            positions=positions,
            velocities=velocities,
            efforts=[0.0] * n,
            rotor_temperatures=[33.0 + j for j in range(n)],
            driver_temperatures=[32.0 + j for j in range(n)],
            status_codes=[0] * n,
        )

    @property
    def source_kind(self) -> str:
        return f"SIMULATED ({len(self.descriptor.channels)} channel(s), no hardware)"

    def disable_all(self) -> None:
        # Observable so the kill path is verifiable in a hardware-free run. Goes
        # to stderr: stdout is reserved for the frontend's data channel (the
        # --json event stream must stay pure JSON).
        import sys

        self.disabled = True
        print("[newt record] (simulate) disable_all() — all motors torque-off.", file=sys.stderr, flush=True)

    def close(self) -> None:
        pass


# Two ready-made descriptors so ``newt record --simulate`` and the validator have
# something to drive with no embodiment module wired. They mirror the two shapes
# rebot-bench validated: one kinesthetic arm, and a bimanual leader/follower pair.

SINGLE_ARM_DESCRIPTOR = StateDescriptor(
    arms=[{"id": "sim-arm"}],
    channels=["sim-arm"],
    joint_names=[
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    ],
    state_fields=["positions", "velocities", "efforts"],
)

BIMANUAL_DESCRIPTOR = StateDescriptor(
    arms=[
        {"id": "left", "leader_ip": "sim", "follower_ip": "sim"},
        {"id": "right", "leader_ip": "sim", "follower_ip": "sim"},
    ],
    channels=["left/leader", "left/follower", "right/leader", "right/follower"],
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
