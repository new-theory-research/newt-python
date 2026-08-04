"""`newt teleop` — the loop, the kill, and the two endings, against a fake rig.

What these encode (the WHY, not just the WHAT):

- **Nothing may be driven after the kill fires.** That is the whole content of a
  panic key, and it is an *ordering* property: asserting the halt was called
  proves nothing if a send slipped in behind it. So every kill test asserts on
  the order of the calls the fake received, and the rest move — which is motion
  — is checked the same way.
- **The two endings differ by exactly one thing.** Ctrl+C puts the rig away and
  then de-energizes it; Ctrl+H de-energizes it where it stands. A silent
  rest-on-kill or a silent kill-on-rest is the failure these pin.
- **A halt is never invented by the verb.** A rig that has not declared how it
  stops is refused before anything is read, not de-energized by some route the
  verb made up.
- **A stop that gives up halfway is not a stop.** One part refusing must not
  cost the next part its attempt, and the refusal has to reach the operator —
  loudly, and in the exit code, because they may be walking toward the rig.
- **Two causes never share one string.** Five ways this verb refuses or fails,
  five messages, three exit codes.

The fake records the order of every call it receives. No hardware, no sleeping,
and a fake clock, so the rate arithmetic is asserted rather than measured.
"""
from __future__ import annotations

import threading

import pytest

from newt.teleop import (
    ActionRejected,
    HaltReport,
    TeleopError,
    _halt_all,
    _put_away,
    _require_source,
    run_session,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class _Kill:
    """The kill event, with every wait it was asked for recorded.

    Duck-typed rather than a threading.Event subclass: the loop's pacing is
    `wait(remaining)`, and recording what it asked for is how the tick rate gets
    asserted without a stopwatch.
    """

    def __init__(self) -> None:
        self._set = False
        self.waits: list[float] = []

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        return self._set


class _Clock:
    """A perf_counter that only moves when the test says so."""

    def __init__(self, step: float = 0.0) -> None:
        self.now = 0.0
        self.step = step

    def perf_counter(self) -> float:
        self.now += self.step
        return self.now


class _Part:
    """One declared moving part that logs what it was told to do."""

    def __init__(
        self,
        name: str,
        log: list[str],
        *,
        halt_error: Exception | None = None,
        rest_error: Exception | None = None,
        state: str | None = "idle",
        on_rest=None,
    ) -> None:
        self.name = name
        self._log = log
        self._halt_error = halt_error
        self._rest_error = rest_error
        self._state = state
        self._on_rest = on_rest

    def halt(self) -> None:
        self._log.append(f"halt:{self.name}")
        if self._halt_error is not None:
            raise self._halt_error
        self._halted = True

    def motor_state(self) -> str | None:
        return self._state

    def rest(self) -> None:
        self._log.append(f"rest:{self.name}")
        if self._on_rest is not None:
            self._on_rest()
        if self._rest_error is not None:
            raise self._rest_error


class _PartWithoutRest:
    """A part that declares a halt and nothing else — the minimum the seam takes."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self._log = log

    def halt(self) -> None:
        self._log.append(f"halt:{self.name}")


class _PartWithoutHalt:
    """A part that declares no way to stop. The verb must refuse it."""

    def __init__(self, name: str) -> None:
        self.name = name


class _Source:
    """A teleop source that records the order of every call it received."""

    def __init__(
        self,
        parts,
        log: list[str],
        *,
        read_error: Exception | None = None,
        send_error: Exception | None = None,
        fail_on_tick: int = 0,
        on_tick=None,
    ) -> None:
        self._parts = list(parts)
        self.log = log
        self._read_error = read_error
        self._send_error = send_error
        self._fail_on_tick = fail_on_tick
        self._on_tick = on_tick
        self.tick = 0
        self.sent: list[object] = []

    def describe(self) -> str:
        return "fake rig (2 parts)"

    def moving_parts(self):
        return self._parts

    def read_action(self):
        self.tick += 1
        self.log.append(f"read:{self.tick}")
        if self._on_tick is not None:
            self._on_tick(self.tick)
        if self._read_error is not None and self.tick >= self._fail_on_tick:
            raise self._read_error
        return {"tick": self.tick}

    def send_action(self, action) -> None:
        self.log.append(f"send:{action['tick']}")
        self.sent.append(action)
        if self._send_error is not None and self.tick >= self._fail_on_tick:
            raise self._send_error


class _MuteSource(_Source):
    """A source whose one-line description is a question it asks the rig.

    Plausible rather than contrived: a description that names the arms it found
    is a description that talks to them, and a rig can stop answering between
    coming up and being asked its name.
    """

    def describe(self) -> str:
        raise RuntimeError("controller has not answered yet")


class _UncountableSource(_Source):
    """A source that cannot say what it has, because that list is a query too."""

    def moving_parts(self):
        raise OSError("no route to host")


def _pair(log, **kwargs):
    return [_Part("left", log, **kwargs), _Part("right", log, **kwargs)]


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
# The pass-through
# --------------------------------------------------------------------------- #

def test_the_action_reaches_the_sink_untouched(clock, capsys):
    """v1 is a pass-through by contract. The object the source produced is the
    object the sink receives — not a copy, not a scaled version, not a filtered
    one. A transform the verb invented is a rig ending up somewhere nobody
    commanded."""
    log: list[str] = []
    kill = _Kill()
    produced: list[dict] = []

    class _Identity(_Source):
        def read_action(self):
            action = super().read_action()
            produced.append(action)
            if self.tick == 3:
                kill.set()
            return action

    source = _Identity(_pair(log), log)
    rc = run_session(source, rate_hz=30, kill=kill)

    assert rc == 130
    assert source.sent  # something was actually driven
    for made, got in zip(produced, source.sent):
        assert got is made  # identity, not equality — nothing was rebuilt on the way


def test_the_loop_paces_at_the_requested_rate(clock, capsys):
    """--rate is the contract with the sink, not a suggestion. With no time spent
    in the rig, every tick waits the full period; a loop that waited on something
    else would drive at a rate nobody asked for."""
    log: list[str] = []
    kill = _Kill()
    source = _Source(_pair(log), log, on_tick=lambda t: kill.set() if t == 4 else None)

    run_session(source, rate_hz=50, kill=kill)

    assert kill.waits  # it paced at all
    assert all(abs(w - 0.02) < 1e-9 for w in kill.waits)  # 1/50 s, every tick


def test_over_period_ticks_are_counted_rather_than_hidden(monkeypatch, capsys):
    """A tick that took longer than its period means the sink was already behind.
    Silently absorbing that is how a rig that cannot keep up looks identical to
    one that can — so it is counted and printed."""
    log: list[str] = []
    kill = _Kill()
    # 0.05s per perf_counter call, against a 1/30s period: every tick overruns.
    monkeypatch.setattr("newt.teleop.time", _Clock(step=0.05))
    source = _Source(_pair(log), log, on_tick=lambda t: kill.set() if t == 3 else None)

    run_session(source, rate_hz=30, kill=kill)

    out = capsys.readouterr().out
    assert "over-period" in out
    assert "(0 over-period)" not in out
    assert kill.waits == []  # nothing waited on an already-blown period


# --------------------------------------------------------------------------- #
# The kill — asserted on order, never on presence
# --------------------------------------------------------------------------- #

def test_no_action_is_sent_after_the_kill_fires(clock, capsys):
    """The safety content of the verb. Asserting the halt happened proves
    nothing: what matters is that nothing was driven behind it. So this asserts
    on the order of the calls the rig received."""
    log: list[str] = []
    kill = _Kill()
    # The kill lands mid-tick, between the read and the send — the worst moment,
    # and the reason the loop re-checks after the read.
    source = _Source(_pair(log), log, on_tick=lambda t: kill.set() if t == 3 else None)

    rc = run_session(source, rate_hz=30, kill=kill)

    assert rc == 130
    assert _first(log, "halt:") > 0
    assert _last(log, "send:") < _first(log, "halt:")
    assert log[-2:] == ["halt:left", "halt:right"]
    # The tick the kill landed in was read but never sent.
    assert "read:3" in log and "send:3" not in log


def test_the_kill_does_not_put_anything_away_first(clock, capsys):
    """Motion after the panic key is the thing the panic key exists to prevent.
    The rest move is motion, so the kill path must not contain one — not before
    the halt, not after it."""
    log: list[str] = []
    kill = _Kill()
    source = _Source(_pair(log), log, on_tick=lambda t: kill.set() if t == 2 else None)

    rc = run_session(source, rate_hz=30, kill=kill)

    assert rc == 130
    assert not [entry for entry in log if entry.startswith("rest:")]


def test_a_kill_during_the_rest_move_stops_the_remaining_rest_moves(clock, capsys):
    """Ctrl+H pressed while the rig is putting itself away has to be honoured
    *now*, not after the rest finishes. The remaining parts are abandoned where
    they stand — but every part is still de-energized, because the rest is what
    gets skipped, never the stop."""
    log: list[str] = []
    kill = _Kill()
    parts = [
        _Part("left", log, on_rest=kill.set),  # the kill lands during left's rest
        _Part("right", log),
    ]

    def _interrupt(tick):
        if tick == 2:
            raise KeyboardInterrupt

    source = _Source(parts, log, on_tick=_interrupt)
    rc = run_session(source, rate_hz=30, kill=kill)

    assert "rest:left" in log
    assert "rest:right" not in log  # never put away after the kill
    assert "halt:left" in log and "halt:right" in log  # both still stopped
    assert rc == 130  # the ending became a kill, and the exit code says so
    assert "emergency_stop" in capsys.readouterr().out


def test_a_kill_pressed_during_bring_up_lands_before_the_first_read(clock, capsys):
    """The kill is armed before the source is built, so it can already be set
    when the session starts. Nothing may be read or driven in that case."""
    log: list[str] = []
    kill = _Kill()
    kill.set()
    source = _Source(_pair(log), log)

    rc = run_session(source, rate_hz=30, kill=kill)

    assert rc == 130
    assert not [entry for entry in log if entry.startswith("read:")]
    assert not [entry for entry in log if entry.startswith("send:")]
    assert log == ["halt:left", "halt:right"]


# --------------------------------------------------------------------------- #
# The normal exit
# --------------------------------------------------------------------------- #

def test_ctrl_c_puts_the_rig_away_and_then_de_energizes_it(clock, capsys):
    """The two endings differ by exactly one thing. This one moves: every part
    is put away by its own declared rest, and only then de-energized — because a
    part left holding its rest pose fights the operator who picks it up."""
    log: list[str] = []
    kill = _Kill()

    def _interrupt(tick):
        if tick == 3:
            raise KeyboardInterrupt

    source = _Source(_pair(log), log, on_tick=_interrupt)
    rc = run_session(source, rate_hz=30, kill=kill)

    assert rc == 0
    assert log[-4:] == ["rest:left", "rest:right", "halt:left", "halt:right"]
    assert "interrupted" in capsys.readouterr().out


def test_a_part_with_no_rest_move_is_named_rather_than_silently_skipped(clock, capsys):
    """An embodiment with nowhere to be put is fine. An operator who believes it
    went somewhere is not — so the skip is printed."""
    log: list[str] = []
    kill = _Kill()

    def _interrupt(tick):
        if tick == 2:
            raise KeyboardInterrupt

    source = _Source([_PartWithoutRest("fixed-base", log)], log, on_tick=_interrupt)
    rc = run_session(source, rate_hz=30, kill=kill)

    assert rc == 0
    assert log[-1] == "halt:fixed-base"
    assert "declares no rest move" in capsys.readouterr().out


def test_a_failed_rest_move_still_de_energizes_every_part(clock, capsys):
    """A rest that refuses leaves the rig somewhere nobody chose. That is worth
    saying out loud — and it is never a reason to leave it energized."""
    log: list[str] = []
    kill = _Kill()
    parts = [
        _Part("left", log, rest_error=RuntimeError("the move timed out")),
        _Part("right", log),
    ]

    def _interrupt(tick):
        if tick == 2:
            raise KeyboardInterrupt

    rc = run_session(_Source(parts, log, on_tick=_interrupt), rate_hz=30, kill=kill)

    assert rc == 0
    assert "rest:right" in log  # one part's failed rest does not skip the next
    assert log[-2:] == ["halt:left", "halt:right"]
    err = capsys.readouterr().err
    assert "did not complete its rest move" in err
    assert "the move timed out" in err


# --------------------------------------------------------------------------- #
# Stopping everything, even when something refuses
# --------------------------------------------------------------------------- #

def test_one_part_refusing_to_stop_does_not_skip_the_next(clock, capsys):
    """A stop that gives up halfway is not a stop. The first refusal is reported
    and the next part is still attempted."""
    log: list[str] = []
    kill = _Kill()
    parts = [
        _Part("left", log, halt_error=RuntimeError("bus timeout")),
        _Part("right", log),
    ]
    source = _Source(parts, log, on_tick=lambda t: kill.set() if t == 2 else None)

    rc = run_session(source, rate_hz=30, kill=kill)

    assert "halt:right" in log  # attempted despite left's failure
    assert rc == 1  # non-zero, and louder than the kill's own 130
    captured = capsys.readouterr()
    assert "HALT FAILED on left" in captured.err
    assert "power it down at the wall" in captured.err.lower()
    assert "NOT CONFIRMED OFF: left" in captured.out


def test_a_refused_stop_is_never_swallowed(capsys):
    """`newt.recording` suppresses a failing torque-off on the way out of a
    read-only session. Here the torque-off *is* the safety contract, so the
    failure surfaces as a report the caller has to act on."""
    log: list[str] = []
    reports = _halt_all(
        [_Part("left", log, halt_error=OSError("no route to host")), _Part("right", log)]
    )

    assert [r.part for r in reports] == ["left", "right"]
    assert reports[0].confirmed is False
    assert "no route to host" in reports[0].detail
    assert reports[1].confirmed is True


def test_the_verb_reports_what_the_part_declares_about_itself(capsys):
    """The verb calls the declared halt and prints what the rig says afterward,
    verbatim. Interpreting that string would mean owning a vocabulary of motor
    states for rigs it has never seen."""
    log: list[str] = []
    reports = _halt_all([_Part("left", log, state="idle (back-drivable)")])

    assert reports[0].state == "idle (back-drivable)"
    assert "idle (back-drivable)" in capsys.readouterr().err


def test_a_part_whose_state_report_fails_still_counts_as_stopped(capsys):
    """The halt returning is what confirms the stop. A rig that cannot say how it
    feels afterward has still stopped, and saying 'unreported' is honest where
    inventing 'idle' would not be."""
    log: list[str] = []

    class _Mute(_Part):
        def motor_state(self):
            raise OSError("read timed out")

    reports = _halt_all([_Mute("left", log)])

    assert reports[0].confirmed is True
    assert "unreported" in reports[0].state
    assert "read timed out" in reports[0].state


# --------------------------------------------------------------------------- #
# The refusals — a rig that cannot be stopped is never started
# --------------------------------------------------------------------------- #

def test_a_part_with_no_declared_halt_is_refused_before_anything_is_read(clock, capsys):
    """The card's line: the verb calls the embodiment's declared halt and mints
    nothing. A rig that has not said how it stops is refused — not de-energized
    by some route this verb invented."""
    log: list[str] = []
    kill = _Kill()
    source = _Source([_PartWithoutHalt("left")], log)

    rc = run_session(source, rate_hz=30, kill=kill)

    assert rc == 1
    assert log == []  # nothing read, nothing driven
    err = capsys.readouterr().err
    assert "declares no halt()" in err
    assert "'left'" in err
    assert "will not invent" in err


def test_a_source_with_no_moving_parts_is_refused(clock, capsys):
    """Nothing to stop means nothing may be started."""
    log: list[str] = []
    rc = run_session(_Source([], log), rate_hz=30, kill=_Kill())

    assert rc == 1
    assert "declares no moving parts" in capsys.readouterr().err


def test_a_source_that_cannot_describe_itself_is_still_de_energized(clock, capsys):
    """Every exit leaves every part de-energized — including the exit taken while
    the rig is being *named*.

    Naming it happens after the factory connected and after the parts were
    declared, which makes it the one step where the rig is live and the session
    has not started. A failure there that walked out as a traceback would leave
    arms holding torque on the exact path this module promises against, and no
    ordering assertion elsewhere would catch it: this is about a call the loop
    never reaches."""
    log: list[str] = []

    rc = run_session(_MuteSource(_pair(log), log), rate_hz=30, kill=_Kill())

    assert rc == 1
    # Never read, never driven — and stopped anyway.
    assert log == ["halt:left", "halt:right"]
    captured = capsys.readouterr()
    assert "could not describe itself" in captured.err
    assert "controller has not answered yet" in captured.err  # the cause, not swallowed
    # The summary still runs, and does not invent a name for a rig that refused one.
    assert "rig:         unnamed" in captured.out


def test_a_source_that_cannot_list_its_parts_says_it_stopped_nothing(clock, capsys):
    """The same failure one step earlier, and it must not share the message above.

    Here the verb was never told what it could stop, so — alone among these
    refusals — it de-energized nothing and the operator is the only thing that
    can. Saying "nothing has been driven" and stopping there would read as safe."""
    log: list[str] = []

    rc = run_session(_UncountableSource(_pair(log), log), rate_hz=30, kill=_Kill())

    assert rc == 1
    assert log == []  # nothing read, nothing driven, and nothing halted either
    err = capsys.readouterr().err
    assert "could not say what its moving parts are" in err
    assert "no route to host" in err
    assert "de-energized nothing" in err
    assert "at the wall" in err


@pytest.mark.parametrize(
    "member", ["describe", "moving_parts", "read_action", "send_action"]
)
def test_a_source_missing_a_seam_member_names_that_member(member):
    """Four members, and a refusal that names the one that is missing rather than
    a TypeError from somewhere inside the loop."""
    log: list[str] = []
    source = _Source(_pair(log), log)
    setattr(source, member, None)  # the instance shadows the class's method

    with pytest.raises(TeleopError) as exc:
        _require_source(source)
    assert f"{member}()" in str(exc.value)


def test_an_unnamed_part_is_refused(clock):
    """A halt line about an unnamed part cannot tell the operator which piece of
    the rig it means."""
    part = _PartWithoutRest("", [])

    with pytest.raises(TeleopError) as exc:
        _require_source(_Source([part], []))
    assert "no name" in str(exc.value)


# --------------------------------------------------------------------------- #
# Failures inside the loop
# --------------------------------------------------------------------------- #

def test_a_failed_read_ends_the_session_rather_than_skipping_the_tick(clock, capsys):
    """Recording can drop a frame and say so — a gap in a dataset is a gap.
    Teleop cannot: the sink would sit still while whoever is driving keeps moving
    and believes it is tracking. So the first failure stops everything."""
    log: list[str] = []
    source = _Source(
        _pair(log), log, read_error=OSError("connection reset"), fail_on_tick=3
    )

    rc = run_session(source, rate_hz=30, kill=_Kill())

    assert rc == 1
    assert _last(log, "send:") == log.index("send:2")  # nothing sent after the failure
    assert log[-2:] == ["halt:left", "halt:right"]
    err = capsys.readouterr().err
    assert "stopped producing actions after 2 tick(s)" in err
    assert "connection reset" in err


def test_a_rejected_action_is_told_apart_from_a_sink_that_stopped(clock, capsys):
    """The one distinction the verb cannot make for itself. A malformed action
    and a dropped link are the same exception type from most drivers and want
    opposite fixes, so the source declares which one it saw — and the two get
    different text."""
    log: list[str] = []
    rejected = _Source(
        _pair(log), log, send_error=ActionRejected("missing key 'wrist'"), fail_on_tick=1
    )
    assert run_session(rejected, rate_hz=30, kill=_Kill()) == 1
    text_rejected = capsys.readouterr().err

    log2: list[str] = []
    dropped = _Source(
        _pair(log2), log2, send_error=OSError("broken pipe"), fail_on_tick=1
    )
    assert run_session(dropped, rate_hz=30, kill=_Kill()) == 1
    text_dropped = capsys.readouterr().err

    assert "rejected the action" in text_rejected
    assert "do not add a translation layer" in text_rejected.lower()
    assert "stopped accepting actions" in text_dropped
    assert text_rejected != text_dropped


# --------------------------------------------------------------------------- #
# The summary, the exit codes, and the strings
# --------------------------------------------------------------------------- #

def test_the_summary_names_the_rig_the_outcome_the_rates_and_the_counts(clock, capsys):
    """What the operator reads after the rig stops moving. Every field here is
    something they cannot get any other way once the process is gone."""
    log: list[str] = []
    kill = _Kill()
    source = _Source(_pair(log), log, on_tick=lambda t: kill.set() if t == 3 else None)

    run_session(source, rate_hz=30, kill=kill)
    out = capsys.readouterr().out

    assert "=== teleop session ===" in out
    assert "fake rig (2 parts)" in out          # the rig names itself
    assert "outcome:     emergency_stop" in out
    assert "30 Hz requested" in out and "Hz achieved" in out
    assert "ticks:       2 in " in out and "over-period)" in out
    assert "ALL CONFIRMED DE-ENERGIZED" in out


@pytest.mark.parametrize(
    "case,expected",
    [
        ("clean", 0),
        ("kill", 130),
        ("source_failed", 1),
        ("halt_refused", 1),
        ("no_halt_declared", 1),
    ],
)
def test_the_exit_code_table(case, expected, clock, capsys):
    """Three codes, five causes, one table. A harness reads these; two causes
    that should be told apart must never collapse into one number *and* one
    string (the strings are asserted below)."""
    log: list[str] = []
    kill = _Kill()

    def _interrupt(tick):
        if tick == 2:
            raise KeyboardInterrupt

    if case == "clean":
        source = _Source(_pair(log), log, on_tick=_interrupt)
    elif case == "kill":
        source = _Source(
            _pair(log), log, on_tick=lambda t: kill.set() if t == 2 else None
        )
    elif case == "source_failed":
        source = _Source(_pair(log), log, read_error=OSError("gone"), fail_on_tick=2)
    elif case == "halt_refused":
        parts = [_Part("left", log, halt_error=RuntimeError("bus timeout"))]
        source = _Source(parts, log, on_tick=_interrupt)
    else:
        source = _Source([_PartWithoutHalt("left")], log)

    assert run_session(source, rate_hz=30, kill=kill) == expected


def test_no_two_refusals_share_a_string(clock, capsys):
    """Rule 12, asserted rather than asserted-in-prose. Five causes reach the
    operator through this module and the frontend; a reader who sees one has to
    know which one it is."""
    from newt._cli.teleop import _stand_down_no_tty, _stand_down_unarmed

    texts: list[str] = []

    for fn in (_stand_down_no_tty, _stand_down_unarmed):
        fn()
        texts.append(capsys.readouterr().err)

    log: list[str] = []
    cases = [
        _Source([_PartWithoutHalt("left")], log),
        _Source(_pair(log), log, read_error=OSError("gone"), fail_on_tick=1),
        _Source(_pair(log), log, send_error=ActionRejected("no 'wrist'"), fail_on_tick=1),
        _Source(_pair(log), log, send_error=OSError("broken pipe"), fail_on_tick=1),
        _MuteSource(_pair(log), log),
        _UncountableSource(_pair(log), log),
    ]
    for source in cases:
        run_session(source, rate_hz=30, kill=_Kill())
        texts.append(capsys.readouterr().err)

    assert len(set(texts)) == len(texts) == 8


def test_a_halt_report_carries_why_it_was_not_confirmed():
    """The summary's 'NOT CONFIRMED OFF' has to be traceable to a cause; a bare
    boolean would leave the operator guessing what refused."""
    ok = HaltReport("left", True, "idle", None)
    bad = HaltReport("right", False, None, "OSError: no route to host")

    assert ok.confirmed and ok.state == "idle" and ok.detail is None
    assert not bad.confirmed and "no route" in bad.detail


def test_put_away_reports_whether_the_kill_landed_partway_through():
    """`_put_away` is where a normal ending can turn into a kill. It says so
    rather than leaving the caller to infer it from the event, because the
    caller has to change both the outcome and the exit code on that."""
    log: list[str] = []
    kill = _Kill()

    reports, killed = _put_away([_Part("left", log), _Part("right", log)], kill)
    assert killed is False
    assert [r.confirmed for r in reports] == [True, True]

    log2: list[str] = []
    kill2 = _Kill()
    reports2, killed2 = _put_away(
        [_Part("left", log2, on_rest=kill2.set), _Part("right", log2)], kill2
    )
    assert killed2 is True
    assert "rest:right" not in log2
    assert [r.confirmed for r in reports2] == [True, True]


def test_the_kill_event_is_a_plain_threading_event(clock, capsys):
    """The frontend owns the keyboard and this module owns the loop; the only
    thing between them is a standard Event. Anything richer would put terminal
    handling behind the seam."""
    log: list[str] = []
    kill = threading.Event()
    source = _Source(
        _pair(log), log, on_tick=lambda t: kill.set() if t == 2 else None
    )

    assert run_session(source, rate_hz=30, kill=kill) == 130
