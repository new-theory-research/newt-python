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


# --- the two ways the driving half of a recording fails ---------------------
#
# Driving does not belong to an episode the way a writer does — it runs from the
# moment the session starts ticking until the session closes, takes and gaps
# alike. So it fails in two places that must never share a string: inside a take
# (an episode exists and is refused) and between takes (no episode exists, and
# what is refused is starting the next one).

DRIVE_FAILURE_CAUSES = (
    "stopped_mid_episode",
    "stopped_between_takes",
)


def drive_failure_message(cause: str, detail: str) -> str:
    """The message for one drive failure cause. ``detail`` is the source's own
    exception, verbatim: what driving means is the source's, and so is the reason
    it stopped."""
    if cause == "stopped_mid_episode":
        return (
            f"The rig stopped being driven part-way through the episode: the source's "
            f"drive_pair() raised ({detail}).\n"
            "Yours: whatever this source drives went unreachable, refused the move, or "
            "faulted. Nothing stale was re-sent to cover the gap, and no tick after it "
            "was recorded.\n"
            "Do now: the episode was discarded, so nothing partial was written, and "
            "capture stopped when driving did. The rig is standing where it stopped "
            "being driven and may still be holding — put it away with `newt rest` "
            "before walking up to it.\n"
            "Then: the message in the parentheses is the source's, not the library's. "
            "That is where the fix is."
        )
    if cause == "stopped_between_takes":
        return (
            "This session stopped being able to drive the rig between takes, so it "
            f"will not open another episode: the source's drive_pair() raised "
            f"({detail}).\n"
            "Yours: driving runs from the moment the session starts ticking, not just "
            "while an episode is open — so it stopped in the gap, with nothing "
            "recording. No episode was started and nothing was written.\n"
            "Do now: the rig is standing where it stopped being driven and may still "
            "be holding — put it away with `newt rest` before walking up to it. This "
            "session will not drive again: close it and start a new one once the "
            "source can drive, rather than recording takes nothing is moving.\n"
            "Then: the message in the parentheses is the source's, not the library's. "
            "That is where the fix is."
        )
    # Never reached through the call sites above; kept so an added cause that
    # forgets its string fails loudly instead of rendering an empty one.
    raise AssertionError(f"no message written for drive failure cause {cause!r} ({detail})")


# --- the five ways a pair source is built wrong -----------------------------
#
# All five fire at class definition, where the kit author is standing and nothing
# is connected — not at a bench with a rig that will not move. Five causes, five
# strings, one place (Rule 12): each one names a different thing the author has to
# go and write, so they may never share a sentence.

PAIR_FAULT_CAUSES = (
    "tick_rewritten",
    "no_driving_half",
    "no_reading_half",
    "says_both",
    "reason_not_a_sentence",
)


def pair_fault_message(cause: str, detail: str) -> str:
    """The message for one way a :class:`PairSource` subclass is built wrong.
    ``detail`` is the subclass's own name."""
    if cause == "tick_rewritten":
        return (
            f"{detail} defines its own read_state(), which is the one method a pair "
            "source does not write.\n"
            "Yours: read_state() is where the two halves of a tick are fused — the "
            "drive runs, then the read, and the reading is not reachable without the "
            "drive having happened. A subclass that writes its own is back to a source "
            "that can be read without being driven, which is what this class exists to "
            "make unbuildable.\n"
            "Do now: nothing was constructed and nothing connected — this fired when "
            "the class was defined.\n"
            "Then: put the moving half in drive_pair() and the reporting half in "
            "read_pair(). Everything your read_state() did belongs in one of the two."
        )
    if cause == "no_driving_half":
        return (
            f"{detail} is a pair source with no driving half: it implements neither "
            "drive_pair() nor a stated reason to command nothing.\n"
            "Yours: a pair is one atom — the part that is moved and the part it is "
            "moved from belong to the same tick. A pair that only reads is a rig that "
            "stands still through a whole session while every readout looks right, and "
            "there is no way to notice from the outside.\n"
            "Do now: nothing was constructed and nothing connected — this fired when "
            "the class was defined.\n"
            "Then: implement drive_pair() — what moving this rig means is yours, the "
            "library only calls it — or, if this source is genuinely meant to command "
            "nothing, set not_driven_because to the sentence saying why. There is no "
            "third state, and no default."
        )
    if cause == "no_reading_half":
        return (
            f"{detail} is a pair source with no reading half: it implements no "
            "read_pair(), so its tick would drive the rig and report nothing.\n"
            "Yours: an episode is written from what read_pair() returns. A source that "
            "drives and reports nothing writes a file of a demonstration nobody can "
            "see.\n"
            "Do now: nothing was constructed and nothing connected — this fired when "
            "the class was defined.\n"
            "Then: implement read_pair(), returning one entry per channel your "
            "descriptor declares — None for a channel that produced nothing this tick."
        )
    if cause == "says_both":
        return (
            f"{detail} implements drive_pair() and also sets not_driven_because, so it "
            "says two things at once.\n"
            "Yours: not_driven_because is the deliberate choice to command nothing. "
            "Beside a drive_pair(), it leaves a method that is never called and a "
            "reader of the class with no way to tell which statement is the true one.\n"
            "Do now: nothing was constructed and nothing connected — this fired when "
            "the class was defined.\n"
            "Then: keep drive_pair() and delete the reason if this source drives; keep "
            "the reason and delete the method if it does not."
        )
    if cause == "reason_not_a_sentence":
        return (
            f"{detail} sets not_driven_because to something that is not a sentence.\n"
            "Yours: that attribute is read by an operator, through the preflight, "
            "before they stand in front of a rig that is about to not move. A True, an "
            "empty string or a bare flag answers the wrong question — the question is "
            "*why*, and nobody but the kit can answer it.\n"
            "Do now: nothing was constructed and nothing connected — this fired when "
            "the class was defined.\n"
            "Then: set not_driven_because to a plain sentence saying what this source "
            "reads and what it deliberately does not command."
        )
    # Never reached through the call sites above; kept so an added cause that
    # forgets its string fails loudly instead of rendering an empty one.
    raise AssertionError(f"no message written for pair fault cause {cause!r} ({detail})")


class DriveStopped(RuntimeError):
    """A pair source's driving half raised, so the tick never completed.

    Raised by :meth:`PairSource.read_state` and by nothing else — a kit raises
    whatever its own driver raised, and this wraps it. It exists so the capture
    loop can tell the two halves of one tick apart: driving that stopped means
    this session is done driving (:class:`DriveFailed`, refused at the episode
    boundary), while a read that raised is a different failure with a different
    fix. The source's own exception is the ``__cause__``; nothing is added to it.
    """

    def __init__(self, exc: BaseException) -> None:
        super().__init__(f"{type(exc).__name__}: {exc}")


class DriveFailed(RuntimeError):
    """The source's driving half raised, so the session stopped driving.

    A tick that could not drive the rig is a tick nobody demonstrated, and an
    episode whose joints go still half-way through — with nothing in the file
    saying the driving stopped — is the camera-that-ends-early failure wearing
    different clothes. The capture loop learns of it as a :class:`DriveStopped`
    out of the source's own tick. Raised by ``Session.end_episode(keep=True)``, which
    abandons the episode rather than committing it, and by
    ``Session.start_episode()``, which refuses to open a take on a session whose
    driving already died.

    Carries the cause key so a frontend can route on it without matching on
    prose, exactly as ``CameraCaptureFailed`` does.
    """

    def __init__(self, cause: str, detail: str) -> None:
        self.cause = cause
        self.detail = detail
        super().__init__(drive_failure_message(cause, detail))


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


@dataclass(frozen=True)
class JointDrive:
    """One joint of a declared robot description, and where its value comes from.

    ``urdf_joint`` is a joint name **in the kit's own description file**. ``channel``
    is one of the descriptor's channel suffixes, and ``index`` is the position this
    joint occupies in that channel's ``positions`` array. Three facts, all of them
    the kit's — ``newt`` reads them and matches nothing by guesswork.

    There is deliberately no sign, scale or offset here. A value is drawn exactly as
    the arm reported it; if a kit's positive direction disagrees with its
    description's, that is a measurement somebody has to take at a bench, and a
    number invented to make a render look right is the fabricated input Rule 10
    exists for. ``ViewDeclaration.joint_convention`` is where a kit says, in words,
    how far it can vouch for the two agreeing.
    """

    urdf_joint: str
    channel: str
    index: int


@dataclass(frozen=True)
class ViewDeclaration:
    """What a kit declares so a live view can draw its robot — the whole of what
    ``newt`` is willing to learn about an embodiment's shape.

    A source exposes this as a ``view_declaration`` attribute and ``newt`` reads it
    at session start. Nothing in this dataclass is optional-with-a-default that
    stands in for a measurement: a kit that cannot name its description file, its
    driven joints, or its own convention has no live robot drawn, and the view says
    so in words rather than drawing a plausible wrong one.

    ``urdf_path``       — the description file, on the machine the session runs on.
    ``entity_prefix``   — where the geometry lands in the entity tree.
    ``drives``          — one entry per joint this source can move (see JointDrive).
    ``joint_convention``— a sentence the view shows verbatim, in which the kit says
                          how the values it reports relate to the directions its
                          description turns, and how far it can vouch for that. This
                          is required and has no default, because "the signs agree"
                          is exactly the claim nobody may make on a kit's behalf.
    ``visual_only``     — draw visual meshes and leave collision hulls out. Superimposed
                          they are unreadable; a kit whose description has no collision
                          geometry is unaffected either way.
    """

    urdf_path: str
    entity_prefix: str
    drives: tuple[JointDrive, ...]
    joint_convention: str
    visual_only: bool = True


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

    **A rig that is moved every tick does that inside ``read_state``**, and there
    is no second method for it. ``read_state()`` is the whole tick: one call, and
    everything the source *is* happens behind it. A source whose rig is driven
    from its own motion subclasses :class:`PairSource`, which writes that tick —
    see there for why the driving is not a separate member of this protocol.
    """

    descriptor: StateDescriptor

    def read_state(self) -> dict[str, JointState | None]:
        ...


class PairSource:
    """A rig one part of which is moved from another part's motion — and the
    reason ``newt`` has a class for it rather than a method: **the pair is the
    atom.**

    Reading such a rig and driving it are not two things a session does in order.
    They are one thing. Any seam where the first can happen without the second is
    a seam where a rig stands still through an entire session while every readout
    on every surface looks correct — and that is not a hypothetical: it shipped
    three times in four days, once per verb, each verb re-losing the driving half
    on its own. A step a caller can forget is a step callers forget.

    So the halves are fused here, once, in a method a kit does not write.
    ``read_state()`` — the session's only tick call — drives and then reads, and
    the reading is not reachable without the drive having run. A subclass
    supplies the two halves:

    ``drive_pair()``
        Move the rig. What moving means is entirely the kit's — which part is
        driven, from what, what an action is — and this library learns only that
        there is a step to take. Takes nothing, returns nothing. Raising from it
        stops the tick before the read (:class:`DriveStopped`), which stops
        capture and refuses the episode rather than recording a rig that quietly
        went still. A value that simply did not arrive this tick is a drop, and
        drops belong in the read half.

    ``read_pair()``
        Report the tick, in exactly the shape ``read_state`` has always returned:
        one entry per channel the descriptor declares, ``None`` for a channel that
        produced nothing.

    Everything else a source may expose — ``cameras`` / ``read_frames``,
    ``disable_all``, ``close``, ``view_declaration`` — is untouched and stays on
    the subclass. This class owns exactly one thing: that a tick is whole.

    **The rare variant, chosen out loud.** A pair that is read and deliberately
    commands nothing sets ``not_driven_because`` to the sentence saying why, and
    implements no ``drive_pair``. That is a line somebody had to write and a
    sentence the preflight prints back to the operator — never a state a rig
    arrives in by a method being left out. A subclass that does neither does not
    define at all: the refusal fires where the kit author is standing, with
    nothing connected and nothing energized.
    """

    descriptor: StateDescriptor

    #: The sentence a source sets to say it reads this pair and commands nothing.
    #: ``None`` — the default — is the ordinary pair, which drives every tick. It
    #: is deliberately not a boolean: "does this drive" is answered by whether
    #: ``drive_pair`` is implemented, and the only thing left to say is *why not*.
    not_driven_because: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        """Refuse a subclass that is not a whole pair, at class-definition time.

        This is the enforcement, and it runs at import: every way of arriving at
        a source that reads without driving is caught here, before a factory has
        connected anything. The alternative — finding out at a bench — is the
        four days this class was written out of.
        """
        super().__init_subclass__(**kwargs)
        if "read_state" in cls.__dict__:
            raise TypeError(pair_fault_message("tick_rewritten", cls.__name__))
        reason = cls.not_driven_because
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise TypeError(pair_fault_message("reason_not_a_sentence", cls.__name__))
        drives = cls.drive_pair is not PairSource.drive_pair
        if drives and reason is not None:
            raise TypeError(pair_fault_message("says_both", cls.__name__))
        if not drives and reason is None:
            raise TypeError(pair_fault_message("no_driving_half", cls.__name__))
        if cls.read_pair is PairSource.read_pair:
            raise TypeError(pair_fault_message("no_reading_half", cls.__name__))

    @property
    def drives(self) -> bool:
        """Whether this source's tick moves the rig — read by the Session for the
        preflight row and the live view's health payload, and by nothing that
        decides whether the rig actually moves. That is the point of the re-cut:
        the driving is inside the tick, so a wrong answer here misdescribes a
        session rather than silencing one."""
        return type(self).not_driven_because is None

    def drive_pair(self) -> None:
        """Move the rig, once. Implemented by the kit; see the class docstring."""
        raise TypeError(pair_fault_message("no_driving_half", type(self).__name__))

    def read_pair(self) -> dict[str, JointState | None]:
        """Report one tick's channels. Implemented by the kit."""
        raise TypeError(pair_fault_message("no_reading_half", type(self).__name__))

    def read_state(self) -> dict[str, JointState | None]:
        """The whole tick: drive, then read. **Not a kit's to override** — a
        subclass that defines it does not define (see ``__init_subclass__``).

        The order is the one ``newt.teleop``'s loop already uses: the state a tick
        records is the state of a rig that has just been told where to go. The
        drive's exception is wrapped rather than passed through so the capture
        loop can tell it from a read that failed; the read never runs after it.
        """
        if type(self).not_driven_because is None:
            try:
                self.drive_pair()
            except Exception as exc:
                # Re-raised, never swallowed: the wrap is what tells the capture
                # loop which half of the tick died, and the source's own
                # exception rides along untouched as __cause__.
                raise DriveStopped(exc) from exc
        return self.read_pair()


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
