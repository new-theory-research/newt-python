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
from typing import Protocol, runtime_checkable


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
    """

    descriptor: StateDescriptor

    def read_state(self) -> dict[str, "JointState | None"]:
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
    _REST_POSE = [0.0, math.pi / 3.0, math.pi / 6.0, math.pi / 5.0, 0.0, 0.0, 0.0]

    def __init__(self, descriptor: StateDescriptor, drop_every: int = 0) -> None:
        self.descriptor = descriptor
        self._t = 0
        self._drop_every = drop_every
        self.disabled = False

    def read_state(self) -> dict[str, "JointState | None"]:
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
