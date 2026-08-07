"""The pair is the atom: one tick reads the rig and drives it, or neither.

The bug these tests hold the line on (2026-08-03 → 08-06, three times in four
days): a rig recorded through `newt record` while the part that should have been
moved sat still. It tracked fine under `newt teleop`, because that verb's loop
sends every tick — and `newt record`'s loop only ever read. Then the composed
verb re-lost it. Then the record+view path re-lost it again.

Three losses, one shape: driving was a step *beside* the read, so every new
caller could take the reading and not take the drive. The fix removes the
choice. A rig moved from its own motion subclasses `newt.recording.PairSource`,
which writes the tick itself — `read_state()` drives and then reads, the reading
is unreachable without the drive, and the session has exactly one call to make.
Reading such a rig while commanding nothing is still possible and still
sometimes right; it is a sentence somebody writes on the class, and the
preflight reads it back to the operator.

The load-bearing tests are the first four: they are shape tests about what can
be *built*, not about whether a method happens to be called. No arm, no vendor
driver, nothing but fake sources counting what they were asked to do.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from newt.recording import (
    SINGLE_ARM_DESCRIPTOR,
    CameraCaptureFailed,
    CameraSpec,
    DriveFailed,
    DriveStopped,
    JointState,
    PairSource,
    Session,
)

_SRC = Path(__file__).resolve().parent.parent / "src"

# Only the tests that open an episode need the extra. The tick itself does not:
# a session with a view attached runs the same loop with no writer in it, which
# is also the shape the bench was in — `--view --control`, watching, between
# takes — so the load-bearing tests run in every CI leg.
_HAVE_EXTRA = (
    importlib.util.find_spec("mcap") is not None
    and importlib.util.find_spec("google.protobuf") is not None
)
needs_extra = pytest.mark.skipif(
    not _HAVE_EXTRA, reason="needs the [recording] extra (mcap/protobuf)"
)


class _Watcher:
    """A view, as far as the Session is concerned: a second sink that starts the
    capture loop and counts what it is offered."""

    def __init__(self) -> None:
        self.states = 0

    def on_state(self, channels, ts_ns) -> None:
        self.states += 1


def _reading() -> dict[str, JointState | None]:
    n = len(SINGLE_ARM_DESCRIPTOR.joint_names)
    return {SINGLE_ARM_DESCRIPTOR.channels[0]: JointState(positions=[0.1 * i for i in range(n)])}


class _ReadOnlySource:
    """A rig that is read and never driven — every source that existed before
    this seam. Not a pair at all, so nothing about its capture may change."""

    descriptor = SINGLE_ARM_DESCRIPTOR
    source_kind = "TEST read-only"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.disabled = False

    def read_state(self) -> dict[str, JointState | None]:
        self.calls.append("read")
        return _reading()

    def disable_all(self) -> None:
        self.disabled = True


class _PairRig(PairSource):
    """The atom: one object whose tick moves the rig and reports what it moved.

    `drive_pair()` stands in for whatever a kit does to move a rig from another
    part's motion. The library never sees inside it; this double proves that by
    doing nothing but writing down that it was called.

    It also carries a `drive()` method that nothing may ever call. That is the
    old, forgettable member, kept here as bait: a session that grew a second
    call back into its tick path trips the test below instead of shipping.
    """

    descriptor = SINGLE_ARM_DESCRIPTOR
    source_kind = "TEST pair"

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls: list[str] = []
        self.driven = 0
        self.disabled = False
        self.legacy_drive_calls = 0
        self._fail_after = fail_after

    def drive_pair(self) -> None:
        self.driven += 1
        self.calls.append("drive")
        if self._fail_after is not None and self.driven > self._fail_after:
            raise ConnectionError("the driven part stopped answering (test double)")

    def read_pair(self) -> dict[str, JointState | None]:
        self.calls.append("read")
        return _reading()

    def drive(self) -> None:  # pragma: no cover — asserted never to run
        self.legacy_drive_calls += 1

    def disable_all(self) -> None:
        self.disabled = True


class _PairRigWithDyingCamera(_PairRig):
    """A real rig's intersection: paired arms and a camera bridge."""

    def __init__(self) -> None:
        super().__init__()
        self.cameras = [CameraSpec("wrist", 64, 48, 30)]

    def read_frames(self):
        raise RuntimeError("wrist camera stopped answering (test double)")


class _WatchingPair(PairSource):
    """The rare variant, chosen out loud: a pair read and deliberately not
    commanded. The reason is the whole declaration — there is no flag."""

    descriptor = SINGLE_ARM_DESCRIPTOR
    source_kind = "TEST pair, commanding nothing"
    not_driven_because = (
        "this source watches both parts of the rig and commands neither; "
        "somebody moves it by hand"
    )

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.disabled = False

    def read_pair(self) -> dict[str, JointState | None]:
        self.calls.append("read")
        return _reading()

    def disable_all(self) -> None:
        self.disabled = True


def _session(source, tmp_path, **kw) -> Session:
    return Session(source, task="pick up the cup", output_dir=tmp_path, state_hz=100, **kw)


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------- #
# The shape: what can be built, and what cannot
# --------------------------------------------------------------------------- #

def test_a_pair_sources_tick_cannot_complete_without_its_drive_half():
    """The atom, asserted as an atom.

    Not "drive gets called" — that is the assertion the old shape passed while
    the bug shipped three times, because a caller who never made the call was
    never in the test. This asserts the stronger thing: there is no way to get a
    reading out of a pair source without its drive having run. Break the fusion
    in `PairSource.read_state` — return the read before the drive, swallow the
    drive's exception, make the drive conditional on anything — and this goes
    red.
    """
    rig = _PairRig(fail_after=0)

    with pytest.raises(DriveStopped) as caught:
        rig.read_state()

    assert rig.calls == ["drive"], (
        f"the read half ran after the drive half died: {rig.calls} — a reading "
        "taken from a rig that was not driven is the whole bug"
    )
    assert isinstance(caught.value.__cause__, ConnectionError), (
        "the source's own exception has to survive the wrap"
    )

    # And the ordinary tick: one call, both halves, drive first.
    fine = _PairRig()
    assert fine.read_state() == _reading()
    assert fine.calls == ["drive", "read"]


def test_a_pair_source_may_not_write_its_own_tick():
    """The fusion is not advice. A subclass that defines `read_state` is a
    subclass that could read without driving, so it does not define at all — and
    the refusal fires at import, where the kit author is standing, rather than at
    a bench in front of a rig that will not move."""
    with pytest.raises(TypeError) as caught:

        class WritesItsOwnTick(PairSource):
            descriptor = SINGLE_ARM_DESCRIPTOR

            def drive_pair(self) -> None:
                pass

            def read_pair(self):
                return _reading()

            def read_state(self):  # the whole point: this may not exist
                return _reading()

    assert "read_state()" in str(caught.value)
    assert "drive_pair()" in str(caught.value), "say where the code belongs instead"


def test_a_pair_that_leaves_its_driving_half_out_does_not_define():
    """The forgettable half-step, refused at the only moment it is free.

    This is the class the bench had: something that reads a rig and commands
    nothing, arrived at by omission and indistinguishable from the real thing
    until an operator watched a part not move for an hour. It is now a class
    that cannot be created — with the two ways out named in the refusal.
    """
    with pytest.raises(TypeError) as caught:

        class ForgotToDrive(PairSource):
            descriptor = SINGLE_ARM_DESCRIPTOR

            def read_pair(self):
                return _reading()

    message = str(caught.value)
    assert "drive_pair()" in message and "not_driven_because" in message, (
        "a refusal has to name both ways out, or the author guesses"
    )
    assert "no third state" in message


def test_reading_a_pair_without_driving_it_is_a_sentence_somebody_wrote():
    """The rare variant exists, and it is chosen — never fallen into.

    A source that says why it commands nothing ticks without driving, reports
    itself as not driving, and hands its own sentence to whatever is about to
    tell the operator. Nothing about it is reachable by leaving a method out.
    """
    rig = _WatchingPair()

    assert rig.drives is False
    assert rig.read_state() == _reading()
    assert rig.calls == ["read"], "the deliberate variant drives nothing"

    # Not a flag, and not both things at once.
    for bad_reason in (True, "", "   "):
        with pytest.raises(TypeError) as caught:
            type("Flagged", (PairSource,), {
                "descriptor": SINGLE_ARM_DESCRIPTOR,
                "not_driven_because": bad_reason,
                "read_pair": lambda self: _reading(),
            })
        assert "not a sentence" in str(caught.value)

    with pytest.raises(TypeError) as caught:
        type("SaysBoth", (PairSource,), {
            "descriptor": SINGLE_ARM_DESCRIPTOR,
            "not_driven_because": "reads only",
            "drive_pair": lambda self: None,
            "read_pair": lambda self: _reading(),
        })
    assert "two things at once" in str(caught.value)


def test_a_pair_with_nothing_to_report_does_not_define():
    """The other half of the atom. A source that drives and reports nothing
    writes a file of a demonstration nobody can see."""
    with pytest.raises(TypeError) as caught:

        class DrivesAndSaysNothing(PairSource):
            descriptor = SINGLE_ARM_DESCRIPTOR

            def drive_pair(self) -> None:
                pass

    assert "read_pair()" in str(caught.value)


def test_the_five_ways_a_pair_is_built_wrong_never_say_the_same_thing():
    """Rule 12, on the surface a kit author actually hits. Five causes, five
    fixes — a shared string sends four of them to the wrong file."""
    from newt.recording._seam import PAIR_FAULT_CAUSES, pair_fault_message

    said = {cause: pair_fault_message(cause, "TheirRig") for cause in PAIR_FAULT_CAUSES}
    assert len(set(said.values())) == len(PAIR_FAULT_CAUSES)
    for cause, message in said.items():
        assert "TheirRig" in message, f"{cause} does not name the class"
        assert "nothing connected" in message, (
            f"{cause} has to say the rig is dark — that is what makes it free"
        )

    with pytest.raises(AssertionError):
        pair_fault_message("a_cause_nobody_wrote", "TheirRig")


# --------------------------------------------------------------------------- #
# The session's tick path: one call, and nothing beside it to forget
# --------------------------------------------------------------------------- #

def test_the_record_tick_drives_the_rig_through_the_sources_own_tick(tmp_path):
    """A session moves the rig. Every tick, not some of them.

    The session here has a view attached and no episode, which is the state the
    bench was in for most of an hour: `--view --control`, watching, between
    takes. Recording adds a file; it is not what makes the rig move.
    """
    source = _PairRig()
    session = _session(source, tmp_path)
    try:
        session.attach_observer(_Watcher())
        assert _wait_for(lambda: source.driven >= 5), (
            f"the record tick never drove the source (driven={source.driven}, "
            f"reads={source.calls.count('read')}) — the rig would sit still while "
            "everything on the page looked right, which is the bench bug"
        )
    finally:
        session.close()

    reads = source.calls.count("read")
    assert abs(source.driven - reads) <= 1, (
        f"{source.driven} drive(s) against {reads} read(s) — the two are one tick"
    )
    pairs = list(zip(source.calls[::2], source.calls[1::2]))
    assert all(pair == ("drive", "read") for pair in pairs[:-1]), (
        f"drive must come before the read of the same tick; saw {source.calls[:8]}"
    )


def test_the_session_has_no_second_call_to_forget(tmp_path):
    """The regression guard for the shape itself.

    The double carries a `drive()` — the old, separately-invokable member — and
    the session must never touch it. If a tick path grows a second call beside
    `read_state()` again, this goes red at the moment it is added rather than at
    a bench four days later.
    """
    source = _PairRig()
    session = _session(source, tmp_path)
    try:
        session.attach_observer(_Watcher())
        assert _wait_for(lambda: source.driven >= 5)
    finally:
        session.close()

    assert source.legacy_drive_calls == 0, (
        "the session called a drive step beside the tick — that is the shape a "
        "caller can omit, and three verbs already did"
    )
    assert set(source.calls) == {"drive", "read"}


def test_a_source_that_is_not_a_pair_is_recorded_exactly_as_before(tmp_path):
    """A rig that only reads — the cameras-only path, the simulated stream, every
    source shipped before this — is polled the way it always was.

    Nothing is inferred and nothing is required: no new method, no error, no
    extra attribute demanded of a source that was never a pair.
    """
    source = _ReadOnlySource()
    session = _session(source, tmp_path)
    watcher = _Watcher()
    try:
        assert session.drives is False
        assert session.describe()["drives"] is False
        assert session.describe()["not_driven_because"] is None
        session.attach_observer(watcher)
        assert _wait_for(lambda: source.calls.count("read") >= 5)
    finally:
        session.close()

    assert set(source.calls) == {"read"}, (
        f"a read-only source was asked to do something else: {set(source.calls)}"
    )
    assert watcher.states >= 5, "and it is still read at the state rate"


@needs_extra
def test_the_tick_that_drives_is_the_tick_that_writes_the_episode(tmp_path):
    """One action, one file. The take a driving session writes is written by the
    same ticks that moved the rig — there is no second loop, and no second read
    of hardware that answers one client at a time."""
    from newt.recording import validate

    source = _PairRig()
    session = _session(source, tmp_path)
    try:
        session.start_episode()
        assert _wait_for(lambda: source.driven >= 10)
        path = session.end_episode(keep=True)
    finally:
        session.close()

    state_count, _dropped = session.last_episode_counts
    assert state_count > 0
    assert state_count <= source.driven, (
        f"{state_count} frame(s) written from {source.driven} driven tick(s) — a "
        "frame the rig was not driven for is a frame nobody demonstrated"
    )
    assert validate(path)["valid"]


@needs_extra
def test_a_driving_session_keeps_driving_between_takes(tmp_path):
    """The rig does not stop being driven because a take ended.

    Driving that stopped at the end of an episode would leave the driven part
    standing while the part it follows kept moving — and the next take would
    command it across that whole gap in a single tick. So an episode adds a
    writer; it does not start or stop the motion.
    """
    source = _PairRig()
    session = _session(source, tmp_path)
    try:
        session.start_episode()
        assert _wait_for(lambda: source.driven >= 3)
        session.end_episode(keep=False)

        between = source.driven
        assert _wait_for(lambda: source.driven >= between + 3), (
            "driving stopped when the take did — the rig would be commanded "
            "across the whole gap when the next take starts"
        )
    finally:
        session.close()

    stopped_at = source.driven
    time.sleep(0.1)
    assert source.driven == stopped_at, "close() must stop the driving too"


@needs_extra
def test_a_camera_failure_does_not_stop_the_rig_but_still_refuses_the_episode(tmp_path):
    """A dead camera costs the take, not control of the paired arms.

    Camera capture and driving share a session but not a stop condition. Once
    the camera bridge reports its failure, the state loop must keep taking the
    indivisible drive-and-read tick even though the episode can no longer be
    kept.
    """
    source = _PairRigWithDyingCamera()
    session = _session(source, tmp_path)
    try:
        session.start_episode()
        assert _wait_for(lambda: session.camera_failure is not None), (
            "the camera thread's failure was not recorded"
        )

        driven_after_failure = source.driven
        assert _wait_for(lambda: source.driven >= driven_after_failure + 3), (
            "a camera failure stopped the paired rig — the follower would freeze "
            "while the leader kept moving"
        )
        with pytest.raises(CameraCaptureFailed) as caught:
            session.end_episode(keep=True)
    finally:
        session.close()

    assert caught.value.cause == "stopped_answering"
    assert "wrist camera stopped answering" in str(caught.value)
    assert sorted(tmp_path.glob("episode_*")) == [], (
        "the rig keeps driving, but the camera-less episode must still be refused"
    )


# --------------------------------------------------------------------------- #
# When the driving stops
# --------------------------------------------------------------------------- #

@needs_extra
def test_a_drive_that_raises_stops_capture_and_refuses_the_episode(tmp_path):
    """A rig that stopped being driven mid-take does not become a kept episode.

    Committing it would hand someone a file whose rig was demonstrated for the
    first ten seconds and sat still for the rest, with nothing in the episode
    saying which part was which. The refusal names the source's own error, and
    no directory is left behind.
    """
    source = _PairRig(fail_after=3)
    session = _session(source, tmp_path)
    try:
        session.start_episode()
        assert _wait_for(lambda: session.drive_failure is not None), (
            "a drive half that raised on a background thread went unnoticed — the "
            "rig would simply stop moving with no signal why"
        )
        with pytest.raises(DriveFailed) as caught:
            session.end_episode(keep=True)
    finally:
        session.close()

    message = str(caught.value)
    assert "drive_pair() raised" in message
    assert "ConnectionError" in message, "the source's own cause has to survive"
    assert "newt rest" in message, "an operator needs the next step, not just the cause"
    assert sorted(tmp_path.glob("episode_*")) == [], "a refused episode leaves no directory"


@needs_extra
def test_a_discarded_episode_after_a_drive_failure_still_leaves_nothing(tmp_path):
    """Discarding is not a refusal — it is what the operator asked for, and it
    works the same whether or not the driving died. The refusal is for `keep`."""
    source = _PairRig(fail_after=2)
    session = _session(source, tmp_path)
    try:
        session.start_episode()
        assert _wait_for(lambda: session.drive_failure is not None)
        assert session.end_episode(keep=False) is None
    finally:
        session.close()
    assert sorted(tmp_path.glob("episode_*")) == []


# --------------------------------------------------------------------------- #
# When the driving stops BETWEEN takes — the window nothing was watching
# --------------------------------------------------------------------------- #
#
# This is the bench's own state: `--view --control`, an observer attached, no
# episode open. Driving runs there (that is the point of keeping it running
# across the gap), so it can die there — with no writer to refuse, no readout
# printing, and nobody looking. The three tests below are the three places that
# silence used to be total.

def test_driving_that_dies_between_takes_survives_the_next_start_episode(tmp_path):
    """A failure in the gap is not erased by the take that follows it.

    The rig stops moving, `start_episode()` clears the record, and the next take
    commits normally — three minutes of a still rig with nothing in the file
    saying so. That is the original bug reached from the other side, so the
    session refuses the take instead: driving is session-scoped, and this session
    will not drive again.

    Needs no recording extra on purpose — the refusal lands before the writer is
    imported, so this holds in the bare-install CI leg too.
    """
    source = _PairRig(fail_after=3)
    session = _session(source, tmp_path)
    try:
        session.attach_observer(_Watcher())  # a view, no episode — the bench state
        assert _wait_for(lambda: session.drive_failure is not None), (
            "the drive half raised between takes and the session noticed nothing"
        )
        with pytest.raises(DriveFailed) as caught:
            session.start_episode()
    finally:
        session.close()

    assert caught.value.cause == "stopped_between_takes"
    assert session.drive_failure is not None, (
        "the failure was consumed by the refusal — a second start_episode() would "
        "then open a take on a rig nobody is driving"
    )
    assert isinstance(session.drive_failure, ConnectionError), (
        "the session remembers the source's own exception, not the wrapper"
    )
    assert sorted(tmp_path.glob("episode_*")) == []


def test_the_two_ways_driving_stops_never_say_the_same_thing(tmp_path):
    """Rule 12: mid-episode and between-takes are different situations for the
    person reading. One had an episode discarded; the other has no episode at all
    and cannot start one. A shared string would send half of them to the wrong
    place."""
    from newt.recording._seam import drive_failure_message

    detail = "ConnectionError: the driven part stopped answering"
    mid = drive_failure_message("stopped_mid_episode", detail)
    between = drive_failure_message("stopped_between_takes", detail)

    assert mid != between
    assert "the episode was discarded" in mid
    assert "No episode was started" in between
    assert detail in mid and detail in between, "the source's own cause survives both"
    for message in (mid, between):
        assert "newt rest" in message, "an operator needs the next step, not just the cause"


def test_the_idle_wait_between_takes_reads_the_drive_failure(tmp_path):
    """The keyboard loop's reader.

    Between takes the frontend is blocked on a keypress, which is why this went
    unseen at the bench: the rig had already stopped when the operator pressed
    SPACE. The wait watches the session as well as the keyboard, so the answer
    arrives before the take does.
    """
    from newt._cli.record import _drive_stopped_between_takes, _wait_for_space

    source = _PairRig(fail_after=2)
    session = _session(source, tmp_path)
    try:
        session.attach_observer(_Watcher())
        assert _wait_for(lambda: session.drive_failure is not None)
        assert _wait_for_space(session) == "drive_stopped", (
            "the wait sat on the keyboard while the rig stood still"
        )
        spoken = _drive_stopped_between_takes(session.drive_failure)
    finally:
        session.close()

    assert "ConnectionError" in spoken, "the source's own cause has to reach the operator"
    assert "no episode is affected" in spoken, "nothing was recording — say so"
    assert "look before you reach" in spoken, "the rig is standing where it stopped"


def test_the_idle_wait_distinguishes_drive_and_both_camera_failures():
    """The keyboard reports which loop died before the operator starts a take."""
    from types import SimpleNamespace

    from newt._cli.record import (
        _camera_stopped_between_takes,
        _drive_stopped_between_takes,
        _wait_for_space,
    )

    drive = ConnectionError("the driven part stopped answering")
    stopped_answering = RuntimeError("camera 0 timed out")
    encoder_refused = ValueError("frame changed shape")
    cases = (
        (
            SimpleNamespace(drive_failure=drive, camera_failure=None),
            "drive_stopped",
            _drive_stopped_between_takes(drive),
        ),
        (
            SimpleNamespace(
                drive_failure=None,
                camera_failure=("stopped_answering", stopped_answering),
            ),
            "camera_stopped",
            _camera_stopped_between_takes(("stopped_answering", stopped_answering)),
        ),
        (
            SimpleNamespace(
                drive_failure=None,
                camera_failure=("encoder_refused", encoder_refused),
            ),
            "camera_stopped",
            _camera_stopped_between_takes(("encoder_refused", encoder_refused)),
        ),
    )

    outcomes = []
    for session, result, message in cases:
        assert _wait_for_space(session) == result
        outcomes.append((result, message))

    assert len(set(outcomes)) == 3, "drive and the two camera causes need distinct outcomes"
    assert "camera 0 timed out" in outcomes[1][1]
    assert "frame changed shape" in outcomes[2][1]


# --------------------------------------------------------------------------- #
# Saying so before anything moves
# --------------------------------------------------------------------------- #

def test_the_preflight_says_whether_this_session_will_move_the_rig(tmp_path, capsys):
    """The line that was missing at the bench.

    A source that reads a rig and drives none of it looks exactly like one that
    drives — right up until nothing moves. The preflight is where an operator
    finds out which one they have, and a rig that commands nothing on purpose
    gets the kit's own sentence rather than a bare `no`.
    """
    from newt._cli.record import _print_preflight

    cases = (
        (_ReadOnlySource(), "no —"),
        (_PairRig(), "YES —"),
        (_WatchingPair(), "somebody moves it by hand"),
    )
    for source, expected in cases:
        session = _session(source, tmp_path)
        try:
            capsys.readouterr()
            _print_preflight(session, as_json=False)
            rows = [ln for ln in capsys.readouterr().out.splitlines() if "drives " in ln]
            assert len(rows) == 1, f"one drives row, got {rows}"
            assert expected in rows[0], rows[0]
        finally:
            session.close()


def test_a_pushed_session_never_drives_the_rig_itself(tmp_path):
    """`newt record --teleop` hands a rig to `newt.teleop`'s loop, which reads
    the action and sends it itself. That Session is built `state_pushed=True`,
    and it must not tick the source at all — two commands per tick on a socket
    that takes one client is worse than none, and it is the failure that would
    arrive quietly as a driven part that jitters."""
    source = _PairRig()
    session = _session(source, tmp_path, state_pushed=True)
    try:
        session.attach_observer(_Watcher())
        time.sleep(0.2)
    finally:
        session.close()

    assert source.driven == 0, (
        f"a pushed session drove the rig {source.driven} time(s) on its own — the "
        "caller's loop is already sending, so this is a second command per tick"
    )
    assert source.calls.count("read") == 0, "and it does not poll one either"


def test_the_verb_that_drives_refuses_a_source_that_drives_itself():
    """The other half of the same hazard, caught before anything moves.

    `newt teleop` reads an action and sends it every tick. Handed a pair source,
    whose own tick also drives, it would command the rig twice per tick — so it
    refuses while the rig is still whatever the factory left it, and names the
    verb that ticks a pair source properly.
    """
    from newt.teleop import TeleopError, _require_source

    class _Part:
        name = "the rig"

        def halt(self) -> None:
            pass

    class DrivesItself(_PairRig):
        def describe(self) -> str:
            return "a pair source that also looks like a teleop source"

        def moving_parts(self):
            return [_Part()]

        def read_action(self):
            return object()

        def send_action(self, action) -> None:
            pass

    source = DrivesItself()
    with pytest.raises(TeleopError) as caught:
        _require_source(source)

    message = str(caught.value)
    assert "twice" in message, "say what would actually happen to the rig"
    assert "newt record" in message, "name the verb that does tick a pair source"
    assert source.driven == 0 and source.calls == [], "and nothing was driven finding out"


def test_an_agent_reading_the_contract_is_told_the_same_thing(tmp_path):
    """`drives` is in the library's own report, so every frontend says the same
    thing — a human reading a row and an agent reading a key cannot disagree."""
    driving = _session(_PairRig(), tmp_path)
    reading = _session(_ReadOnlySource(), tmp_path)
    watching = _session(_WatchingPair(), tmp_path)
    try:
        assert driving.describe()["drives"] is True
        assert reading.describe()["drives"] is False
        assert watching.describe()["drives"] is False
        assert watching.describe()["not_driven_because"] == _WatchingPair.not_driven_because
    finally:
        driving.close()
        reading.close()
        watching.close()


def test_json_mode_refuses_a_driving_source_and_puts_the_rig_away(tmp_path):
    """An agent has no Ctrl+H, and this source moves hardware. Refuse, loudly.

    `--json` is what an agent uses INSTEAD of a keyboard, so a driving session on
    that path would command a rig with no kill key. It is only knowable after the
    factory ran — the rig is up by then — so the refusal says what was done about
    that, and the source's own `disable_all` is what did it.
    """
    marker = tmp_path / "disable_all_was_called"
    module = tmp_path / "bench_double.py"
    module.write_text(
        "from pathlib import Path\n"
        "from newt.recording import SINGLE_ARM_DESCRIPTOR, JointState, PairSource\n"
        "\n"
        "class Rig(PairSource):\n"
        "    descriptor = SINGLE_ARM_DESCRIPTOR\n"
        "    source_kind = 'TEST driving rig'\n"
        "    def drive_pair(self):\n"
        "        pass\n"
        "    def read_pair(self):\n"
        "        n = len(self.descriptor.joint_names)\n"
        "        return {self.descriptor.channels[0]: JointState(positions=[0.0] * n)}\n"
        "    def disable_all(self):\n"
        f"        Path({str(marker)!r}).write_text('yes')\n"
        "\n"
        "def make_source():\n"
        "    return Rig()\n"
    )

    proc = subprocess.run(
        [
            sys.executable, "-m", "newt", "record",
            "--source", "bench_double:make_source",
            "--json",
            "--task", "pick up the cup",
            "--dest", str(tmp_path / "episodes"),
        ],
        input=json.dumps({"cmd": "close"}) + "\n",
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(_SRC)])},
        timeout=60,
    )

    assert proc.returncode == 1, f"expected a refusal, got {proc.returncode}: {proc.stdout}"
    assert "pair source" in proc.stderr and "--json" in proc.stderr
    assert "Ctrl+H" in proc.stderr, "the refusal has to name what is missing"
    assert marker.exists(), (
        "the rig was up when this was discovered, so the refusal has to put it away"
    )
