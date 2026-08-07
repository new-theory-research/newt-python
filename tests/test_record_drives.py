"""Recording a demonstration is one action: the record tick drives the rig.

The bug these tests hold the line on, from the bench (2026-08-06, twice): a
leader/follower pair recorded through `newt record` while the follower sat
still. The same rig tracked fine under `newt teleop`, because that verb's loop
sends an action every tick — and `newt record`'s loop only ever read. A source
that has something to drive had no way to say so, so nothing was driven.

The fix is one word wide: a source may declare `drive()`, and the capture loop
calls it at the top of every tick, before the read. What driving *means* stays
in the source — the method takes nothing, returns nothing, and the library
learns only that there is a step to take. The tests here are shape tests: no
arm, no vendor driver, nothing but a fake source counting what it was asked to
do.

The load-bearing one is the first: remove the `drive()` call from
`Session._capture_loop` and it goes red while everything else stays green.
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
    DriveFailed,
    JointState,
    Session,
)

_SRC = Path(__file__).resolve().parent.parent / "src"

# Only the tests that open an episode need the extra. The tick itself does not:
# a session with a view attached runs the same loop with no writer in it, which
# is also the shape the bench was in — `--view --control`, watching, between
# takes — so the load-bearing test runs in every CI leg.
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


class _ReadOnlySource:
    """A rig that is read and never driven — every source that existed before
    this seam. It declares no `drive`, so nothing about its capture may change."""

    descriptor = SINGLE_ARM_DESCRIPTOR
    source_kind = "TEST read-only"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.disabled = False

    def read_state(self) -> dict[str, JointState | None]:
        self.calls.append("read")
        n = len(self.descriptor.joint_names)
        return {self.descriptor.channels[0]: JointState(positions=[0.1 * i for i in range(n)])}

    def disable_all(self) -> None:
        self.disabled = True


class _DrivingSource(_ReadOnlySource):
    """The composed shape: one object that moves the rig and reports what it moved.

    `drive()` stands in for whatever a kit does — read a leader, send it to a
    follower. The library never sees inside it; this test double proves that by
    doing nothing but writing down that it was called.
    """

    source_kind = "TEST driving"

    def __init__(self, fail_after: int | None = None) -> None:
        super().__init__()
        self.drives = 0
        self._fail_after = fail_after

    def drive(self) -> None:
        self.drives += 1
        self.calls.append("drive")
        if self._fail_after is not None and self.drives > self._fail_after:
            raise ConnectionError("the follower stopped answering (test double)")


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
# The one that must fail if the fix is removed
# --------------------------------------------------------------------------- #

def test_the_record_tick_drives_a_source_that_declares_a_drive_step(tmp_path):
    """A session moves the rig. Every tick, not some of them.

    This is the whole bug in one assert: a source that declares `drive()` is
    driven once per tick by the record session's own loop. Delete the drive call
    from `Session._capture_loop` and `drives` stays at 0 forever while the reads
    pile up — which is exactly what the bench saw, with a follower that never
    moved.

    The session here has a view attached and no episode, which is the state the
    bench was in for most of an hour: `--view --control`, watching, between
    takes. Recording adds a file; it is not what makes the rig move.
    """
    source = _DrivingSource()
    session = _session(source, tmp_path)
    try:
        session.attach_observer(_Watcher())
        assert _wait_for(lambda: source.drives >= 5), (
            f"the record tick never drove the source (drives={source.drives}, "
            f"reads={source.calls.count('read')}) — the follower would sit still "
            "while the leader moved, which is the bench bug"
        )
    finally:
        session.close()

    # One drive per read, in that order, with no read overtaking a drive: the
    # state a tick records is the state of a rig already told where to go.
    reads = source.calls.count("read")
    assert abs(source.drives - reads) <= 1, (
        f"{source.drives} drive(s) against {reads} read(s) — the two are one tick"
    )
    pairs = list(zip(source.calls[::2], source.calls[1::2]))
    assert all(pair == ("drive", "read") for pair in pairs[:-1]), (
        f"drive must come before the read of the same tick; saw {source.calls[:8]}"
    )


def test_a_source_with_no_drive_step_is_recorded_exactly_as_before(tmp_path):
    """A rig that only reads — the cameras-only path, the simulated stream, every
    source shipped before this — is polled the way it always was.

    Nothing is inferred and nothing is required: no drive call, no error, no
    extra attribute demanded of a source that never claimed to move anything.
    """
    source = _ReadOnlySource()
    session = _session(source, tmp_path)
    watcher = _Watcher()
    try:
        assert session.drives is False
        assert session.describe()["drives"] is False
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

    source = _DrivingSource()
    session = _session(source, tmp_path)
    try:
        session.start_episode()
        assert _wait_for(lambda: source.drives >= 10)
        path = session.end_episode(keep=True)
    finally:
        session.close()

    state_count, _dropped = session.last_episode_counts
    assert state_count > 0
    assert state_count <= source.drives, (
        f"{state_count} frame(s) written from {source.drives} driven tick(s) — a "
        "frame the rig was not driven for is a frame nobody demonstrated"
    )
    assert validate(path)["valid"]


@needs_extra
def test_a_driving_session_keeps_driving_between_takes(tmp_path):
    """The rig does not stop being driven because a take ended.

    Driving that stopped at the end of an episode would leave the follower
    standing while the hand holding the leader kept moving — and the next take
    would command it across that whole gap in a single tick. So an episode adds
    a writer; it does not start or stop the motion.
    """
    source = _DrivingSource()
    session = _session(source, tmp_path)
    try:
        session.start_episode()
        assert _wait_for(lambda: source.drives >= 3)
        session.end_episode(keep=False)

        between = source.drives
        assert _wait_for(lambda: source.drives >= between + 3), (
            "driving stopped when the take did — the rig would be commanded "
            "across the whole gap when the next take starts"
        )
    finally:
        session.close()

    stopped_at = source.drives
    time.sleep(0.1)
    assert source.drives == stopped_at, "close() must stop the driving too"


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
    source = _DrivingSource(fail_after=3)
    session = _session(source, tmp_path)
    try:
        session.start_episode()
        assert _wait_for(lambda: session.drive_failure is not None), (
            "a drive() that raised on a background thread went unnoticed — the "
            "rig would simply stop moving with no signal why"
        )
        with pytest.raises(DriveFailed) as caught:
            session.end_episode(keep=True)
    finally:
        session.close()

    message = str(caught.value)
    assert "drive() raised" in message
    assert "ConnectionError" in message, "the source's own cause has to survive"
    assert "newt rest" in message, "an operator needs the next step, not just the cause"
    assert sorted(tmp_path.glob("episode_*")) == [], "a refused episode leaves no directory"


@needs_extra
def test_a_discarded_episode_after_a_drive_failure_still_leaves_nothing(tmp_path):
    """Discarding is not a refusal — it is what the operator asked for, and it
    works the same whether or not the driving died. The refusal is for `keep`."""
    source = _DrivingSource(fail_after=2)
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
    source = _DrivingSource(fail_after=3)
    session = _session(source, tmp_path)
    try:
        session.attach_observer(_Watcher())  # a view, no episode — the bench state
        assert _wait_for(lambda: session.drive_failure is not None), (
            "drive() raised between takes and the session noticed nothing"
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
    assert sorted(tmp_path.glob("episode_*")) == []


def test_the_two_ways_driving_stops_never_say_the_same_thing(tmp_path):
    """Rule 12: mid-episode and between-takes are different situations for the
    person reading. One had an episode discarded; the other has no episode at all
    and cannot start one. A shared string would send half of them to the wrong
    place."""
    from newt.recording._seam import drive_failure_message

    detail = "ConnectionError: the follower stopped answering"
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

    source = _DrivingSource(fail_after=2)
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


# --------------------------------------------------------------------------- #
# Saying so before anything moves
# --------------------------------------------------------------------------- #

def test_the_preflight_says_whether_this_session_will_move_the_rig(tmp_path, capsys):
    """The line that was missing at the bench.

    A source that reads two arms and drives neither looks exactly like one that
    drives — right up until the follower does not follow. The preflight is where
    an operator finds out which one they have, and it says so both ways.
    """
    from newt._cli.record import _print_preflight

    for source, expected in ((_ReadOnlySource(), "no —"), (_DrivingSource(), "YES —")):
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
    """`newt record --teleop` hands the same composed rig to `newt.teleop`'s loop,
    which reads the action and sends it itself. That Session is built
    `state_pushed=True`, and it must not also drive — two commands per tick on a
    socket that takes one client is worse than none, and it is the failure that
    would arrive quietly as a follower that jitters."""
    source = _DrivingSource()
    session = _session(source, tmp_path, state_pushed=True)
    try:
        session.attach_observer(_Watcher())
        time.sleep(0.2)
    finally:
        session.close()

    assert source.drives == 0, (
        f"a pushed session drove the rig {source.drives} time(s) on its own — the "
        "caller's loop is already sending, so this is a second command per tick"
    )
    assert source.calls.count("read") == 0, "and it does not poll one either"


def test_an_agent_reading_the_contract_is_told_the_same_thing(tmp_path):
    """`drives` is in the library's own report, so every frontend says the same
    thing — a human reading a row and an agent reading a key cannot disagree."""
    driving = _session(_DrivingSource(), tmp_path)
    reading = _session(_ReadOnlySource(), tmp_path)
    try:
        assert driving.describe()["drives"] is True
        assert reading.describe()["drives"] is False
    finally:
        driving.close()
        reading.close()


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
        "from newt.recording import SINGLE_ARM_DESCRIPTOR, JointState\n"
        "\n"
        "class Rig:\n"
        "    descriptor = SINGLE_ARM_DESCRIPTOR\n"
        "    source_kind = 'TEST driving rig'\n"
        "    def drive(self):\n"
        "        pass\n"
        "    def read_state(self):\n"
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
    assert "drive()" in proc.stderr and "--json" in proc.stderr
    assert "Ctrl+H" in proc.stderr, "the refusal has to name what is missing"
    assert marker.exists(), (
        "the rig was up when this was discovered, so the refusal has to put it away"
    )
