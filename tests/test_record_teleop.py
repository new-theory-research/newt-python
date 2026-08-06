"""`newt record --teleop` — one tick drives the arm and writes the episode.

What these encode (the WHY, not just the WHAT):

- **The episode holds the action that actually drove the follower.** A
  demonstration is a sequence of (what the robot saw, what it was told to do),
  and that is what training consumes. A leader position re-read next to the tick
  instead of taken from the value handed to ``send_action`` is an action that
  never drove anything — off by microseconds most ticks and by a whole frame on
  the tick where it matters. So the fidelity test asserts on the object that was
  sent, and it asserts the leader was read exactly once per tick.
- **A read with nothing behind it is a drop.** A composed source caches the tick
  it just drove; asked again with no new tick, it says nothing rather than
  repeating itself. A duplicated frame is a lie about time, and it looks like a
  clean 30 Hz episode right up until someone trains on it.
- **The kill discards, and it drives nothing on the way out.** A panic stop is
  not a demonstration, and motion after the panic key is what the panic key
  exists to prevent — both asserted on the order of the calls the fake received.
- **The recording closes before the rig is put away.** Ctrl+C's rest move is
  motion nobody demonstrated; an episode still open through it would record the
  arms stowing themselves as part of the task.
- **Plain ``newt record`` is untouched.** The pushed path is opt-in at
  construction, and a Session that polls refuses a pushed frame rather than
  interleaving two clocks into one episode.

No hardware, no sleeping, a fake clock. Nothing here proves an arm moved.
"""
from __future__ import annotations

import importlib.util
import io
import sys

import pytest

from newt._cli.record import _EpisodeRecorder, _refuse_composed, cmd_record
from newt.recording import JointState, Session, StateDescriptor
from newt.teleop import TeleopError, run_session

_HAVE_EXTRA = (
    importlib.util.find_spec("mcap") is not None
    and importlib.util.find_spec("google.protobuf") is not None
)
needs_extra = pytest.mark.skipif(
    not _HAVE_EXTRA, reason="needs the [recording] extra (mcap/protobuf)"
)

LEADER = "fake_pair/leader"
FOLLOWER = "fake_pair/follower"
JOINTS = ["waist", "shoulder", "gripper"]


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class _Kill:
    """The kill event, duck-typed so a test can fire it mid-tick."""

    def __init__(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True

    def wait(self, timeout: float | None = None) -> bool:
        return self._set


class _Clock:
    """perf_counter that only moves when asked, and a wall clock that ticks."""

    CLOCK_REALTIME = 0

    def __init__(self) -> None:
        self.now = 0.0
        self.wall = 1_000_000_000

    def perf_counter(self) -> float:
        return self.now

    def clock_gettime_ns(self, _which) -> int:
        self.wall += 1_000_000
        return self.wall


class _Part:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self._log = log

    def halt(self) -> None:
        self._log.append(f"halt:{self.name}")

    def rest(self) -> None:
        self._log.append(f"rest:{self.name}")


class _ComposedSource:
    """A rig that drives one arm from another and records what it drove.

    Modelled on the kit's composed pair, and the two properties under test are
    its properties: ``send_action`` remembers the action it actually sent, and
    ``read_state`` reports that remembered action for the driven channel and then
    forgets it — so a second read with no tick behind it has nothing to say.
    """

    drives_and_records = True

    def __init__(self, log: list[str], *, ticks_before=None, on_tick=None) -> None:
        self.log = log
        self.descriptor = StateDescriptor(
            arms=[{"id": "fake_pair"}],
            channels=[LEADER, FOLLOWER],
            joint_names=list(JOINTS),
            state_fields=["positions"],
        )
        self._parts = [_Part("leader", log), _Part("follower", log)]
        self._sent = None
        self._on_tick = on_tick
        self._stop_after = ticks_before
        self.tick = 0
        self.sent: list[dict] = []
        self.state_reads = 0
        self.disabled = False

    # -- the teleop half ---------------------------------------------------
    def describe(self) -> str:
        return "fake composed pair (leader → follower, recorded)"

    def moving_parts(self):
        return list(self._parts)

    def read_action(self):
        self.tick += 1
        self.log.append(f"read:{self.tick}")
        if self._on_tick is not None:
            self._on_tick(self.tick)
        if self._stop_after is not None and self.tick > self._stop_after:
            raise KeyboardInterrupt
        # A fresh, unequal action every tick, so a stale repeat is visible.
        return {f"{name}.pos": float(self.tick) + i for i, name in enumerate(JOINTS)}

    def send_action(self, action) -> None:
        self.log.append(f"send:{self.tick}")
        self.sent.append(action)
        self._sent = action

    # -- the recording half ------------------------------------------------
    def read_state(self):
        self.state_reads += 1
        sent, self._sent = self._sent, None
        if sent is None:
            # No tick behind this read. Nothing is repeated: a drop is honest,
            # a duplicated frame is a lie about time.
            return {LEADER: None, FOLLOWER: None}
        return {
            LEADER: JointState(positions=[sent[f"{n}.pos"] for n in JOINTS]),
            FOLLOWER: JointState(positions=[sent[f"{n}.pos"] for n in JOINTS]),
        }

    def disable_all(self) -> None:
        self.disabled = True

    def close(self) -> None:
        self.log.append("close")


class _Recorder:
    """A TickRecorder that records what it was handed and how it was closed."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.frames: list[tuple[dict, int]] = []
        self.kept: bool | None = None

    def record_tick(self, channels, ts_ns: int) -> None:
        self.log.append(f"frame:{len(self.frames) + 1}")
        self.frames.append((channels, ts_ns))

    def finish(self, *, keep: bool) -> None:
        self.log.append(f"finish:{'keep' if keep else 'discard'}")
        self.kept = keep


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr("newt.teleop.time", c)
    return c


def _first(log: list[str], prefix: str) -> int:
    for i, entry in enumerate(log):
        if entry.startswith(prefix):
            return i
    return -1


def _last(log: list[str], prefix: str) -> int:
    hit = -1
    for i, entry in enumerate(log):
        if entry.startswith(prefix):
            hit = i
    return hit


# --------------------------------------------------------------------------- #
# Fidelity — the recorded action is the action that was sent
# --------------------------------------------------------------------------- #

def test_the_recorded_action_is_the_one_that_drove_the_follower(clock, capsys):
    """Every recorded leader frame is the action `send_action` received.

    Not a value read beside it, not a value read after it — the same one. This
    is the invariant that fails the moment someone "optimizes" the composed read
    into a second leader read, and it is what makes the episode a demonstration
    rather than two arms that happened to be near each other.
    """
    log: list[str] = []
    kill = _Kill()
    source = _ComposedSource(log, ticks_before=4)
    recorder = _Recorder(log)

    rc = run_session(source, rate_hz=30, kill=kill, recorder=recorder)

    assert rc == 0
    assert len(source.sent) == 4
    assert len(recorder.frames) == 4
    for action, (channels, _ts) in zip(source.sent, recorder.frames):
        expected = [action[f"{n}.pos"] for n in JOINTS]
        assert channels[LEADER].positions == expected


def test_the_leader_is_read_once_per_tick_and_the_state_comes_from_that_read(
    clock, capsys
):
    """One read_action per tick, one read_state per tick — no second read.

    Two reasons, and the first is fidelity: a second read of the leader beside
    the tick produces an action that never drove anything. The second is the
    single-client socket, which pays for every extra round trip in tick budget.
    """
    log: list[str] = []
    kill = _Kill()
    source = _ComposedSource(log, ticks_before=3)
    recorder = _Recorder(log)

    run_session(source, rate_hz=30, kill=kill, recorder=recorder)

    assert source.tick == 4  # 3 driven ticks, the 4th read is the Ctrl+C
    assert len(source.sent) == 3
    assert source.state_reads == 3  # exactly one per driven tick
    assert len(recorder.frames) == 3
    # Order per tick: read, send, record. A frame written before its send would
    # be a frame of a rig that had not been told anything yet.
    assert log[:6] == ["read:1", "send:1", "frame:1", "read:2", "send:2", "frame:2"]


def test_a_read_with_no_fresh_tick_behind_it_is_a_drop_not_a_repeat():
    """The anti-padding fence, asserted on the composed source itself.

    A tick that produced nothing has to say nothing. Re-emitting the last frame
    would fill the gap with a value that was true a moment ago and is being
    written down as if it were true now — which reads as a clean 30 Hz episode
    right up until someone trains on it.
    """
    source = _ComposedSource([])
    action = source.read_action()
    source.send_action(action)

    fresh = source.read_state()
    assert fresh[LEADER] is not None and fresh[FOLLOWER] is not None

    stale = source.read_state()
    assert stale == {LEADER: None, FOLLOWER: None}


# --------------------------------------------------------------------------- #
# The two endings
# --------------------------------------------------------------------------- #

def test_the_kill_drives_nothing_and_leaves_no_episode(clock, capsys):
    """Ctrl+H: nothing sent behind the kill, nothing put away, no episode.

    Asserted on order, because asserting the halt happened proves nothing if a
    send slipped in behind it — and asserting "discarded" proves nothing if the
    rig moved to its rest pose first.
    """
    log: list[str] = []
    kill = _Kill()
    source = _ComposedSource(log, on_tick=lambda t: kill.set() if t == 3 else None)
    recorder = _Recorder(log)

    rc = run_session(source, rate_hz=30, kill=kill, recorder=recorder)

    assert rc == 130
    assert recorder.kept is False
    assert _last(log, "send:") < _first(log, "halt:")
    assert _last(log, "frame:") < _first(log, "halt:")
    assert not [entry for entry in log if entry.startswith("rest:")]
    # The tick the kill landed in was read, never sent, and never recorded.
    assert "read:3" in log and "send:3" not in log
    assert len(recorder.frames) == 2


def test_the_normal_exit_closes_the_episode_before_the_rig_is_put_away(clock, capsys):
    """Ctrl+C: keep the episode, *then* stow, then de-energize.

    The rest move is motion nobody demonstrated. An episode still open through
    it would carry the arms stowing themselves as the last seconds of the task,
    and nothing in the file would say so.
    """
    log: list[str] = []
    kill = _Kill()
    source = _ComposedSource(log, ticks_before=2)
    recorder = _Recorder(log)

    rc = run_session(source, rate_hz=30, kill=kill, recorder=recorder)

    assert rc == 0
    assert recorder.kept is True
    assert _first(log, "finish:") < _first(log, "rest:") < _first(log, "halt:")
    assert log[-2:] == ["halt:leader", "halt:follower"]


def test_a_source_that_stops_answering_keeps_what_was_driven(clock, capsys):
    """A fault mid-demonstration ends the session, de-energizes, and keeps.

    Those ticks happened — an operator gave that take, and discarding it because
    the rig faulted afterwards would throw away real data. The message names the
    recording half specifically, because "the rig stopped answering" and "the rig
    drove but could not say what it did" want different next steps.
    """
    log: list[str] = []
    kill = _Kill()
    source = _ComposedSource(log)

    def _blow_up():
        raise RuntimeError("follower read timed out")

    source.read_state = _blow_up
    recorder = _Recorder(log)

    rc = run_session(source, rate_hz=30, kill=kill, recorder=recorder)

    assert rc == 1
    assert recorder.kept is True
    assert _first(log, "halt:") > 0
    err = capsys.readouterr().err
    assert "could not say what it did" in err


# --------------------------------------------------------------------------- #
# The refusals — four causes, four strings
# --------------------------------------------------------------------------- #

class _RecordOnly:
    """A recording source: reads both arms, drives neither. What Mattie ran."""

    descriptor = "descriptor"

    def read_state(self):
        return {}


class _TeleopOnly:
    """A teleop source: drives, records nothing."""

    def read_action(self):
        return {}

    def send_action(self, action) -> None:
        pass

    def moving_parts(self):
        return []


class _Undeclared(_ComposedSource):
    """Both halves present, nothing declared. A shape, not a statement."""

    drives_and_records = False


def test_each_way_a_source_cannot_do_this_gets_its_own_string():
    """Four causes, four strings — asserted pairwise distinct, not merely non-empty.

    The one that matters most is the recording-only source: it is what an
    operator reaches by doing the obvious thing, and it must say *that*, not
    "source refused".
    """
    messages = [
        _refuse_composed(_RecordOnly()),
        _refuse_composed(_TeleopOnly()),
        _refuse_composed(object()),
        _refuse_composed(_Undeclared([])),
    ]
    assert all(m is not None for m in messages)
    assert len(set(messages)) == 4
    assert "records the rig but does not drive it" in messages[0]
    assert "drives the rig but has nothing to record" in messages[1]
    assert "neither drives nor records" in messages[2]
    assert "drives_and_records" in messages[3]


def test_a_source_that_declares_both_is_accepted():
    """The declaration is the gate. Method presence only picks the message."""
    assert _refuse_composed(_ComposedSource([])) is None


@pytest.mark.parametrize(
    "args, marker",
    [
        (["--task", "t", "--teleop"], "--source is required"),
        (["--task", "t", "--teleop", "--source", "x:y", "--simulate"],
         "no simulated demonstration"),
        (["--task", "t", "--teleop", "--source", "x:y", "--json"], "no kill key"),
    ],
)
def test_the_flag_combinations_that_cannot_work_refuse_before_anything_connects(
    args, marker, monkeypatch
):
    """Each impossible combination stops at the parse, with its own reason.

    Nothing is loaded, so nothing is connected and nothing has moved — which is
    what each message says.
    """
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    rc = cmd_record(args)

    assert rc == 1
    assert marker in err.getvalue()


def test_the_help_says_the_flag_is_temporary(monkeypatch, capsys):
    """The door is spelled provisionally and says so where an operator reads it.

    Naming is not this slice's to settle; shipping a flag that looks permanent
    would settle it by accident.
    """
    rc = cmd_record(["--help"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--teleop" in out
    assert "TEMPORARY DOOR" in out and "newtrino-030" in out


# --------------------------------------------------------------------------- #
# Plain `newt record` is untouched
# --------------------------------------------------------------------------- #

def test_a_polled_session_refuses_a_pushed_frame():
    """Two clocks into one episode is refused, not interleaved.

    The polled Session is already reading the source at the state rate. A pushed
    frame on top of that would land in the same episode on a second rhythm, with
    nothing in the file saying which frame came from which.
    """
    source = _ComposedSource([])
    session = Session(source, task="t", output_dir="/tmp/newt-test-unused")

    with pytest.raises(RuntimeError) as exc:
        session.feed_state({LEADER: None})

    assert "state_pushed=True" in str(exc.value)


@needs_extra
def test_a_pushed_session_writes_exactly_the_ticks_it_was_handed(tmp_path):
    """The episode's frame count is the number of ticks that produced state.

    Not the number of ticks, and not the number of reads — a stale read is
    counted as a drop and adds nothing to the file. This is the fence again, one
    layer down, where it decides what actually lands on disk.
    """
    source = _ComposedSource([])
    session = Session(
        source, task="pick up the cup", output_dir=tmp_path, state_pushed=True
    )
    recorder = _EpisodeRecorder(session)

    for _ in range(3):
        action = source.read_action()
        source.send_action(action)
        recorder.record_tick(source.read_state(), 1_000_000_000)
    # One read with no tick behind it: all channels drop.
    recorder.record_tick(source.read_state(), 1_000_000_000)

    recorder.finish(keep=True)

    state_count, dropped = session.last_episode_counts
    assert state_count == 6  # 3 ticks x 2 channels
    assert dropped == 2  # the starved read, both channels
    assert recorder.path is not None and recorder.path.is_dir()
    session.close()


@needs_extra
def test_the_kill_path_leaves_no_directory(tmp_path):
    """A discarded episode leaves nothing behind — not an empty dir, not a temp one.

    A partial demonstration nobody chose to keep is indistinguishable on disk
    from one that was kept, unless it is not on disk.
    """
    source = _ComposedSource([])
    session = Session(source, task="t", output_dir=tmp_path, state_pushed=True)
    recorder = _EpisodeRecorder(session)
    action = source.read_action()
    source.send_action(action)
    recorder.record_tick(source.read_state(), 1_000_000_000)

    recorder.finish(keep=False)

    assert recorder.path is None
    assert list(tmp_path.glob("*/episode.json")) == []
    session.close()
