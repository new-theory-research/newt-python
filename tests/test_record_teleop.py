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
- **Declare, then build.** The declaration is read off the factory before the
  factory is called, and a factory that has not declared is never called at all.
  Construction is what connects and energizes a rig, so a verb that validates
  afterwards has already moved metal it never approved — that is not a
  hypothetical, it is what `newt rest --source recording_source:live_pair` did to
  two arms on 2026-08-05. The test asserts on whether the factory *ran*, because
  an exit code cannot tell a refusal-before from a refusal-after.

No hardware, no sleeping, a fake clock. Nothing here proves an arm moved.
"""
from __future__ import annotations

import importlib
import importlib.util
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from newt._cli.record import _EpisodeRecorder, _refuse_composed, cmd_record
from newt.recording import JointState, Session, StateDescriptor
from newt.teleop import ActionRejected, TeleopError, run_session

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

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

class _RereadingSource(_ComposedSource):
    """The optimization the fidelity test exists to catch.

    Identical in every way an exit code or a frame count can see: same channels,
    same joints, same one-frame-per-tick rhythm, no drops. The single difference
    is that `read_state` takes a fresh look at the leader instead of reporting
    the action it just handed over — a value of the right type, the right length
    and the right magnitude, describing a moment nobody commanded. That is Rule
    10's failure shape exactly, and it is the reason a fidelity test that cannot
    fail is not a test.
    """

    def read_state(self):
        self.state_reads += 1
        self._sent = None
        fresh = [float(self.tick) + 0.5 + i for i, _ in enumerate(JOINTS)]
        return {
            LEADER: JointState(positions=list(fresh)),
            FOLLOWER: JointState(positions=list(fresh)),
        }


def _drove_what_it_recorded(source, recorder) -> bool:
    """Every recorded frame equals the action `send_action` was handed for it."""
    if len(source.sent) != len(recorder.frames):
        return False
    for action, (channels, _ts) in zip(source.sent, recorder.frames):
        expected = [action[f"{n}.pos"] for n in JOINTS]
        if channels[LEADER].positions != expected:
            return False
        if channels[FOLLOWER].positions != expected:
            return False
    return True


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
    assert _drove_what_it_recorded(source, recorder)
    # And the values are actually distinct tick to tick, so the check above is
    # discriminating rather than comparing a constant against itself.
    recorded = [tuple(channels[LEADER].positions) for channels, _ in recorder.frames]
    assert len(set(recorded)) == 4


def test_a_source_that_re_reads_the_leader_fails_the_fidelity_check(clock, capsys):
    """The fidelity check can fail, and this is what failing looks like.

    A re-reading source produces the same tick count, the same frame count, the
    same exit code and zero drops — everything a summary line reports is
    identical. The only thing that separates it from a faithful one is the
    invariant above, which is why that invariant is the test and the counts are
    not.
    """
    log: list[str] = []
    source = _RereadingSource(log, ticks_before=4)
    recorder = _Recorder(log)

    rc = run_session(source, rate_hz=30, kill=_Kill(), recorder=recorder)

    assert rc == 0
    assert len(recorder.frames) == len(source.sent) == 4  # indistinguishable by count
    assert not _drove_what_it_recorded(source, recorder)


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


# --------------------------------------------------------------------------- #
# Declare, then build — the refusal that happens while the rig is still dark
# --------------------------------------------------------------------------- #

_LYING_FACTORY = """
ran = False


def make_demo():
    global ran
    ran = True
    return object()
"""

_DECLARED_FACTORY = """
from newt.teleop import drives_and_records

ran = False


@drives_and_records
def make_demo():
    global ran
    ran = True
    return object()
"""


def _rig_module(tmp_path, monkeypatch, name: str, text: str):
    """A real module on a real sys.path, whose factory records whether it ran.

    Written and imported for real rather than monkeypatched in: the invariant
    under test is exactly the difference between what `import` does and what
    `call` does, and a stubbed loader would prove the test's own arrangement
    instead of the order the verb runs in.
    """
    (tmp_path / f"{name}.py").write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


class _Tty(io.StringIO):
    """stdin that claims to be a terminal — the composed path refuses without one."""

    def isatty(self) -> bool:
        return True


def _run_composed(args, monkeypatch):
    """`newt record --teleop …` with the kill key stubbed and nothing real behind it."""
    from newt._cli.teleop import KillKey

    err = io.StringIO()
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(KillKey, "arm", lambda self: True)
    monkeypatch.setattr(KillKey, "restore", lambda self: None)

    rc = cmd_record(["--task", "pick up the cup", "--teleop", *args])
    return rc, err.getvalue()


def test_an_undeclared_factory_is_refused_without_ever_being_called(
    tmp_path, monkeypatch
):
    """The MUST-FIX, asserted on the side effect and not on the exit code.

    A factory is what connects and energizes a rig. Refusing after it has run
    leaves an operator holding two torqued arms they never approved and cannot
    move by hand — which is the bench incident this ordering exists to prevent.
    So the assertion that matters is ``ran is False``: the verb read the
    declaration off the callable and stopped there.
    """
    mod = _rig_module(tmp_path, monkeypatch, "undeclared_rig", _LYING_FACTORY)

    rc, err = _run_composed(["--source", "undeclared_rig:make_demo"], monkeypatch)

    assert rc == 2
    assert mod.ran is False, "the factory ran — the rig was built before it was allowed"
    assert "does not declare drives_and_records" in err
    assert "nothing is connected and nothing is energized" in err


def test_a_declared_factory_runs_and_the_object_is_checked_again(
    tmp_path, monkeypatch
):
    """The declaration is a gate, not a promise — and the second check catches the lie.

    Same exit code as the refusal above and a different string, because they are
    different causes with different next steps: one is a factory that never
    should have been called, the other is a factory that claimed more than it
    built. The rig is up in this one, and the message says so.
    """
    mod = _rig_module(tmp_path, monkeypatch, "declared_rig", _DECLARED_FACTORY)

    rc, err = _run_composed(["--source", "declared_rig:make_demo"], monkeypatch)

    assert rc == 2
    assert mod.ran is True, "a declared factory was gated out — the gate is too tight"
    assert "neither drives nor records" in err
    assert "does not declare drives_and_records" not in err


# --------------------------------------------------------------------------- #
# Where the composed source comes from — its own namespace, not record's
# --------------------------------------------------------------------------- #

def test_a_configured_rig_records_a_demonstration_without_naming_a_factory(
    tmp_path, monkeypatch
):
    """Two words plus a task, on a bench that declared itself once (newtrino-029).

    Asserted on which factory was reached rather than on an exit code, because
    an exit code cannot tell a resolved default from a lucky no-op. And the
    provenance line is asserted too: a source nobody typed has to say where it
    came from before anything starts.
    """
    mod = _rig_module(tmp_path, monkeypatch, "configured_rig", _LYING_FACTORY)
    config = tmp_path / "nt.toml"
    config.write_text('[sources]\ndemonstration = "configured_rig:make_demo"\n')
    monkeypatch.setenv("NT_SITE_CONFIG", str(config))

    rc, err = _run_composed([], monkeypatch)

    assert rc == 2
    assert mod.ran is False
    assert "configured_rig:make_demo" in err
    assert str(config.resolve()) in err


def test_a_rig_that_declared_only_record_is_not_volunteered_for_this(
    tmp_path, monkeypatch
):
    """`[sources].record` is not an answer to "what drives this rig while reading it".

    The recording factory reads two arms and drives neither — running it here is
    the exact thing that looked like it should work at the bench and did not.
    Reaching for it because it is the nearest declared thing would be the
    identity-fill failure with a friendly face, so the refusal names the key
    that is missing and leaves the record factory alone.
    """
    mod = _rig_module(tmp_path, monkeypatch, "record_only_rig", _LYING_FACTORY)
    config = tmp_path / "nt.toml"
    config.write_text('[sources]\nrecord = "record_only_rig:make_demo"\n')
    monkeypatch.setenv("NT_SITE_CONFIG", str(config))

    rc, err = _run_composed([], monkeypatch)

    assert rc == 2
    assert mod.ran is False
    assert "demonstration" in err
    assert "It declares: record." in err


def test_the_composed_refusal_names_the_command_typed_not_the_namespace(
    tmp_path, monkeypatch
):
    """There is no `newt demonstration`, so no refusal may tell anyone to run it.

    Reproduced at a bench on 2026-08-06: a one-character typo in a short name
    came back with "Fix the name: newt demonstration --source …", a fix
    instruction for a verb the dispatcher does not have. The namespace is right
    and stays — `[sources].demonstration` is the line the operator opens an
    editor and types — but a line they retype at a prompt has to name the
    command they typed. Both halves are asserted, because deleting the noun
    would pass the first one and lose the only thing that says where to declare.

    Driven through `cmd_record` and not through the resolver: the resolver
    defaults the command to the verb, which is correct for every other verb and
    wrong for exactly this call site, so the call site is what has to be tested.
    """
    from newt._cli import _source_spec

    monkeypatch.setattr(
        _source_spec,
        "_declared_sources",
        lambda verb: (
            [("bench_pair", "bench_kit.rig:make_demo", "a-kit")]
            if verb == "demonstration"
            else []
        ),
    )
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "absent" / "nt.toml"))

    rc, err = _run_composed(["--source", "bench_par"], monkeypatch)

    assert rc == 2
    assert "newt demonstration" not in err, f"advertised a command that does not exist:\n{err}"
    assert "Fix the name:  newt record --teleop --source bench_pair" in err
    assert "is declared for demonstration" in err


def test_the_composed_refusal_never_offers_simulate_as_the_way_out(
    tmp_path, monkeypatch
):
    """Plain `record`'s unresolved-source refusal ends with "or try --simulate".

    On this path that would be advice the very next branch refuses: there is no
    simulated demonstration, because a demonstration is a person moving a real
    arm. A fix line that does not work is worse than no fix line.
    """
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "absent" / "nt.toml"))

    rc, err = _run_composed([], monkeypatch)

    assert rc == 2
    assert "no source to run" in err
    assert "--simulate" not in err


def test_a_factory_that_cannot_be_found_says_the_rig_is_still_dark(
    tmp_path, monkeypatch
):
    """Phase one failing and phase two failing are different problems.

    An import that fails called nothing, so nothing is connected. A factory that
    raises partway through was *in the middle of connecting and energizing* when
    it did. Collapsing both into `[newt record] <exception>` leaves an operator
    with no way to tell whether there is a rig standing energized in the next
    room — which is the only part of either message they have to act on.
    """
    rc, err = _run_composed(["--source", "no_such_rig_module:make_demo"], monkeypatch)

    assert rc == 1
    assert "could not be found" in err
    assert "nothing is connected and nothing is energized" in err
    assert "newt rest" not in err  # there is nothing to put away


_RAISING_FACTORY = """
from newt.teleop import drives_and_records


@drives_and_records
def make_demo():
    raise RuntimeError("leader answered, follower did not")
"""


def test_a_factory_that_raises_mid_bring_up_says_the_rig_may_be_up(
    tmp_path, monkeypatch
):
    """The other half of the pair above, and the one that costs something.

    This factory declared, so it was allowed to run — and it raised after
    reaching one arm. The verb cannot know how far it got, so it says the one
    true thing (this is the call that energizes) and names the recovery, rather
    than reporting the exception and letting the operator assume the failure was
    clean.
    """
    _rig_module(tmp_path, monkeypatch, "raising_rig", _RAISING_FACTORY)

    rc, err = _run_composed(["--source", "raising_rig:make_demo"], monkeypatch)

    assert rc == 1
    assert "raised while building it" in err
    assert "leader answered, follower did not" in err  # the kit's own words, kept
    assert "newt rest" in err
    assert "nothing is connected" not in err


@pytest.mark.parametrize(
    "args, marker",
    [
        (["--bimanual"], "--bimanual: that flag shapes the simulated joint stream"),
        (["--drop-every", "5"], "--drop-every: that flag shapes the simulated"),
        (["--target", "3"], "exactly one episode per run"),
    ],
)
def test_a_flag_this_path_cannot_honour_is_refused_not_ignored(
    args, marker, monkeypatch
):
    """Accepting a flag and doing nothing with it is the silent-default failure.

    `--bimanual` and `--drop-every` only ever meant something to the simulated
    stream; `--target` means something real and means it per-session, on a path
    that runs one episode and exits. All three used to parse cleanly and vanish,
    so a run with `--target 10` looked like it obeyed and stopped after one — a
    green check that could not fail.
    """
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    rc = cmd_record(["--task", "t", "--teleop", "--source", "x:y", *args])

    assert rc == 1
    assert marker in err.getvalue()


def test_the_two_simulate_only_flags_are_named_together_not_swallowed(monkeypatch):
    """Both passed, both named. A refusal that mentions one leaves the other to
    be discovered on the next run, which is two trips to the bench."""
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    rc = cmd_record(
        ["--task", "t", "--teleop", "--bimanual", "--drop-every", "5", "--source", "x:y"]
    )

    assert rc == 1
    assert "--bimanual" in err.getvalue() and "--drop-every" in err.getvalue()


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


# --------------------------------------------------------------------------- #
# Five causes, five strings — the whole path, not one function's four branches
# --------------------------------------------------------------------------- #

class _RejectingSink(_ComposedSource):
    """The driven part refuses the action: built in the wrong action space."""

    def send_action(self, action) -> None:
        raise ActionRejected("this part takes joint targets; that action is a pose")


class _MuteRecorder(_ComposedSource):
    """Drove the rig fine, then could not say what it did. The clock is gone."""

    def read_state(self):
        raise OSError("state socket closed")


def test_the_five_ways_this_can_refuse_do_not_share_a_string(clock, capsys):
    """Rule 12 across the whole composed path, asserted rather than argued.

    These five reach an operator through three different layers — two are
    shape refusals from the frontend, two are the library's tick failing, and one
    is the stand-down before anything connects. All five end the same run with
    nothing recorded, so the string is the only thing that tells a reader which
    one they got, and "the source refused" would be five wrong answers.

    The fifth is the one that was broken: it is `newt teleop`'s stand-down,
    borrowed whole, and it used to greet an operator who typed `newt record
    --teleop` with teleop's name on it and teleop's command as the fix — a verb
    that drives the rig and writes nothing, which is what they were avoiding.
    """
    from newt._cli.record import _COMPOSED_SCRIPTED_FIX
    from newt._cli.teleop import _stand_down_no_tty

    texts = [
        _refuse_composed(_TeleopOnly()),
        _refuse_composed(_RecordOnly()),
    ]

    for source in (_RejectingSink([]), _MuteRecorder([])):
        capsys.readouterr()
        run_session(source, rate_hz=30, kill=_Kill(), recorder=_Recorder([]))
        texts.append(capsys.readouterr().err)

    capsys.readouterr()
    _stand_down_no_tty("newt record", "--teleop needs a TTY", _COMPOSED_SCRIPTED_FIX)
    texts.append(capsys.readouterr().err)

    assert all(t for t in texts)
    assert len(set(texts)) == 5, "two causes are sharing a string"
    # Each names its own cause, so "distinct" is not five near-identical strings
    # separated by a tick number.
    assert "drives the rig but has nothing to record" in texts[0]
    assert "records the rig but does not drive it" in texts[1]
    assert "rejected the action" in texts[2]
    assert "could not say what it did" in texts[3]
    assert "no keyboard" in texts[4]


def test_the_composed_stand_down_names_its_own_verb_not_teleops(capsys):
    """The two no-TTY stand-downs are two causes, and they must read that way.

    Same underlying condition, two commands, two fixes. The one that matters is
    the fix line: telling an operator who wanted a recorded demonstration to go
    run `newt teleop` sends them to the verb that moves the arms and writes
    nothing — advice that produces the bug report they were already filing.
    """
    from newt._cli.record import _COMPOSED_SCRIPTED_FIX
    from newt._cli.teleop import _stand_down_no_tty

    capsys.readouterr()
    assert _stand_down_no_tty() == 2
    plain = capsys.readouterr().err
    assert (
        _stand_down_no_tty("newt record", "--teleop needs a TTY", _COMPOSED_SCRIPTED_FIX)
        == 2
    )
    composed = capsys.readouterr().err

    assert plain != composed
    assert "[newt teleop]" in plain and "[newt teleop]" not in composed
    assert "newt record --teleop" in composed
    # And the composed one answers the question its own path raises: a bounded
    # run still writes an episode, and --json is not the way around this.
    assert "KEPT" in composed and "--json" in composed


# --------------------------------------------------------------------------- #
# The composability gate — a body the library has never met
# --------------------------------------------------------------------------- #

_SECOND_BODY = '''
"""A bench cell where a handwheel drives a gantry, and the pair records itself.

Nothing here subclasses anything the library ships. It is a different shape from
the pair the composed path was built against: five axes instead of three joints,
one moving part instead of two, its own channel names, and a driver that is not
an arm at all.
"""
from newt.recording import JointState, StateDescriptor
from newt.teleop import drives_and_records

AXES = ["x", "y", "z", "tilt", "jaw"]
DRIVER = "bench_cell/handwheel"
DRIVEN = "bench_cell/gantry"


class _Axis:
    name = "gantry"

    def halt(self):
        pass

    def rest(self):
        pass


class GantryDemonstration:
    drives_and_records = True

    def __init__(self):
        self.descriptor = StateDescriptor(
            arms=[{"id": "bench_cell"}],
            channels=[DRIVER, DRIVEN],
            joint_names=list(AXES),
            state_fields=["positions"],
        )
        self._sent = None
        self.tick = 0

    def describe(self):
        return "bench cell (handwheel drives gantry, recorded)"

    def moving_parts(self):
        return [_Axis()]

    def read_action(self):
        self.tick += 1
        if self.tick > 3:
            raise KeyboardInterrupt
        return {axis: float(self.tick) + i for i, axis in enumerate(AXES)}

    def send_action(self, action):
        self._sent = action

    def read_state(self):
        sent, self._sent = self._sent, None
        if sent is None:
            return {DRIVER: None, DRIVEN: None}
        row = [sent[a] for a in AXES]
        return {DRIVER: JointState(positions=list(row)),
                DRIVEN: JointState(positions=list(row))}

    def disable_all(self):
        pass

    def close(self):
        pass


@drives_and_records
def make_demo():
    return GantryDemonstration()
'''


def test_a_second_body_records_a_demonstration_with_no_library_change(
    tmp_path, monkeypatch, capsys
):
    """The gate newtrino-015 set: a second rig that declares it composes, composes.

    This body shares no code, no base class and no shape with the pair the path
    was written against — different axis count, different channel names, one
    moving part, and a driver that is a handwheel rather than an arm. It reaches
    the tick loop through the same resolve → import → declaration-gate → build
    sequence and drives through the same `run_session`. If the library had
    learned anything about the first rig, this is where that shows up: as a
    refusal, a shape assumption, or a name it does not recognize.

    Runs without the recording extra, because what it proves is the seam and not
    the file format — the end-to-end episode is the test below.
    """
    from newt._cli._source_spec import build_source, import_factory
    from newt._cli.record import _refuse_composed, _refuse_undeclared_factory

    _rig_module(tmp_path, monkeypatch, "gantry_rig", _SECOND_BODY)
    spec = "gantry_rig:make_demo"

    factory = import_factory(spec)
    assert _refuse_undeclared_factory(spec, factory) is None
    source = build_source(spec, factory)
    assert _refuse_composed(source) is None

    recorder = _Recorder([])
    rc = run_session(source, rate_hz=30, kill=_Kill(), recorder=recorder)

    assert rc == 0
    assert recorder.kept is True
    assert len(recorder.frames) == 3
    channels, _ts = recorder.frames[0]
    assert sorted(channels) == ["bench_cell/gantry", "bench_cell/handwheel"]
    assert len(channels["bench_cell/handwheel"].positions) == 5


@needs_extra
def test_the_second_bodys_episode_lands_through_the_whole_verb(
    tmp_path, monkeypatch
):
    """Same body, all the way to a directory on disk, through `cmd_record`.

    The test above proves the seam; this one proves the product — one command, a
    rig nobody in the library has heard of, and an episode written with that
    rig's own channel names in it.

    Where those names actually are matters, and this test used to look in the
    wrong place: `episode.json` carries the *arms* the descriptor declared, and
    the per-channel streams are MCAP topics in `data.mcap`. So both are checked
    at the altitude each one knows. The arm id is the weaker signal — it would
    survive a run that opened the file and wrote not one tick. The topics and
    their message counts are the claim: two channels this body invented, three
    ticks each, written by a code path that has never heard of either name.
    """
    from mcap.reader import make_reader

    _rig_module(tmp_path, monkeypatch, "gantry_rig_e2e", _SECOND_BODY)
    dest = tmp_path / "episodes"

    rc, err = _run_composed(
        ["--source", "gantry_rig_e2e:make_demo", "--dest", str(dest)], monkeypatch
    )

    assert rc == 0, err
    written = list(dest.glob("*/episode.json"))
    assert len(written) == 1

    meta = json.loads(written[0].read_text())
    assert [arm["id"] for arm in meta["robot_config"]["arms"]] == ["bench_cell"], (
        f"episode.json lost the arm this body declared: {meta['robot_config']}"
    )

    counts: dict[str, int] = {}
    with open(written[0].parent / "data.mcap", "rb") as handle:
        for _schema, channel, _message in make_reader(handle).iter_messages():
            counts[channel.topic] = counts.get(channel.topic, 0) + 1

    assert counts == {
        "robot_state/bench_cell/handwheel": 3,
        "robot_state/bench_cell/gantry": 3,
    }, f"the episode does not carry this body's channels and ticks: {counts}"


def test_the_library_holds_no_name_this_body_chose():
    """The rig's vocabulary travels as data; the SDK never learns it.

    Its axes, its channels and the thing driving it are all named in the episode
    it just wrote and must appear nowhere in the source of the library that
    wrote it. Looked for as string literals, which is what special-casing a rig
    actually looks like — `if channel == "bench_cell/gantry"` — rather than as
    substrings, so a docstring using one of these words as English is not a
    finding.

    This is the tree-wide half of the card's fence. The other half is the diff
    grep below, which is the only one that can police generic role words like
    leader and follower: those already appear in the library's own simulated
    bimanual stream, so a whole-tree ban on them would fail on code this card
    never touched.
    """
    literal = re.compile(r"""['"][^'"]*(?:handwheel|bench_cell|gantry)[^'"]*['"]""", re.I)
    hits = [
        f"{path.relative_to(_SRC)}:{i}"
        for path in sorted(_SRC.rglob("*.py"))
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
        if literal.search(line)
    ]
    assert hits == [], "the library learned a name belonging to a rig: " + "; ".join(hits)


def test_no_embodiment_string_entered_the_library_on_this_branch():
    """The card's own fence, run as a test instead of by hand.

    A diff and not a tree scan, deliberately: `leader` and `follower` are the
    roles the library's simulated bimanual stream has always used, so the
    question worth asking is not whether they appear but whether this branch
    added any. `newt-python` is public, and a rig's vendor, product name or
    factory name landing in it is the leak class — a receipt worth citing in a
    commit is not worth compiling in.
    """
    banned = re.compile(
        r"^\+.*(lerobot|trossen|widowx|yam|leader|follower|live_pair)", re.I
    )
    proc = subprocess.run(
        ["git", "diff", "origin/main...HEAD", "--", "src/"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"no origin/main to diff against: {proc.stderr.strip()}")

    hits = [line for line in proc.stdout.splitlines() if banned.match(line)]
    assert hits == [], "an embodiment's name entered src/ on this branch:\n" + "\n".join(
        hits
    )
