"""The run itself: the choreography, the isolation, and the confirmation.

What these encode (the WHY, not just the WHAT):

- **The order is the design.** Every part is driven through its declared
  sequence, then *every* part is de-energized, then every part is asked what
  state it ended in. The de-energize pass is not conditional on the sequence
  going well — the arm that fell over mid-sequence is exactly the arm that must
  not be left holding — and the read-back comes last because it is a question
  about where things finished, not where they were going.
- **One part's failure never skips another part's attempt.** A step that raises
  ends that part's sequence and nothing else. The arm next to it still gets
  driven, still gets stopped, still gets asked.
- **A state nobody read is never reported as rested.** This is the promise the
  verb exists for. Three ways a part can end unconfirmed — the halt was refused,
  it declares no way to be asked, it was asked and would not answer — and all
  three exit non-zero and print the line that sends someone to the wall switch.
  A run that printed "rested" over an arm that might be holding is worse than no
  rest command at all, because a person walks up to the rig on the strength of
  it.
- **The verb reads the state; it never grades it.** What a motor state string
  means belongs to the rig. A verb that decided which strings meant "off" would
  be deciding it for hardware it has never seen, so it prints the answer verbatim
  and certifies only that it got one.
- **Four causes, four sentences, and an exit code per thing the operator would
  do differently** (Rule 12). Nothing declared, a step that failed, an arm that
  would not stop, a final state nobody could read.
"""
from __future__ import annotations

import io
import sys

import pytest

from newt._cli.rest import cmd_rest
from newt.rest import (
    ABANDONED,
    EXIT_NOTHING_DECLARED,
    EXIT_RESTED,
    EXIT_STEP_FAILED,
    EXIT_UNCONFIRMED,
    RESTED,
    STEP_FAILED,
    declared_sequence,
    read_declarations,
    require_rest_source,
    run_rest,
)


# --------------------------------------------------------------------------- #
# Fakes — one shared log, so ordering across parts is a single assertion
# --------------------------------------------------------------------------- #

class Step:
    def __init__(self, name: str, log: list[str], fail: Exception | None = None) -> None:
        self.name = name
        self._log = log
        self._fail = fail

    def run(self) -> None:
        self._log.append(f"run:{self.name}")
        if self._fail is not None:
            raise self._fail


class InterruptingStep(Step):
    """A step that raises Ctrl+C out of the middle of a move, as the keyboard would."""

    def run(self) -> None:
        self._log.append(f"run:{self.name}")
        raise KeyboardInterrupt()


class Arm:
    """A part whose every affordance can be told to misbehave.

    ``state`` is what motor_state() returns; ``no_state`` removes the method
    entirely, which is a different thing from a method that raises and has to
    stay tellable apart from it.
    """

    def __init__(
        self,
        name: str,
        log: list[str],
        steps: list[str],
        *,
        fail_on: str | None = None,
        fail_halt: Exception | None = None,
        state: object = "idle",
        fail_state: Exception | None = None,
        no_state: bool = False,
    ) -> None:
        self.name = name
        self._log = log
        self._steps = steps
        self._fail_on = fail_on
        self._fail_halt = fail_halt
        self._state = state
        self._fail_state = fail_state
        if not no_state:
            self.motor_state = self._motor_state  # type: ignore[method-assign]

    def rest_sequence(self):
        self._log.append(f"declare:{self.name}")
        return [
            Step(
                s,
                self._log,
                OSError("the arm stopped answering") if s == self._fail_on else None,
            )
            for s in self._steps
        ]

    def halt(self) -> None:
        self._log.append(f"halt:{self.name}")
        if self._fail_halt is not None:
            raise self._fail_halt

    def _motor_state(self):
        self._log.append(f"ask:{self.name}")
        if self._fail_state is not None:
            raise self._fail_state
        return self._state


class Rig:
    def __init__(self, *parts) -> None:
        self._parts = list(parts)

    def describe(self) -> str:
        return "a rig with " + ", ".join(p.name for p in self._parts)

    def moving_parts(self):
        return list(self._parts)


def _drive(rig, monkeypatch):
    """Run the verb end to end against ``rig``, returning (rc, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr("newt._cli.rest.load_source", lambda spec: rig)
    rc = cmd_rest(["--source", "fake:rig"])
    return rc, out.getvalue(), err.getvalue()


def _run(rig):
    """Call the library directly, past the CLI, for assertions on outcomes."""
    return run_rest(rig, read_declarations(require_rest_source(rig)))


# --------------------------------------------------------------------------- #
# The choreography — call order is the contract
# --------------------------------------------------------------------------- #

def test_every_part_rests_before_any_part_is_de_energized(monkeypatch):
    """Sequences first, then the stop pass, then the read-back.

    Asserted as one exact list because each boundary carries an argument. The
    steps run in the order the rig declared them, because the rig — not this
    verb — knows that going to the second pose from anywhere but the first is
    not the same move. Every halt lands after every sequence, matching the
    behavior the kit's own exit path already had. And nothing is asked what
    state it is in until it has been stopped, because a state read mid-move
    answers a question nobody asked.
    """
    log: list[str] = []
    rig = Rig(
        Arm("the left arm", log, ["above the table", "to the rest pose"]),
        Arm("the right arm", log, ["above the table", "to the rest pose"]),
    )

    rc, _, _ = _drive(rig, monkeypatch)

    assert rc == EXIT_RESTED
    assert log == [
        "declare:the left arm",
        "declare:the right arm",
        "run:above the table",
        "run:to the rest pose",
        "run:above the table",
        "run:to the rest pose",
        "halt:the left arm",
        "ask:the left arm",
        "halt:the right arm",
        "ask:the right arm",
    ]


def test_a_failed_step_stops_that_part_and_no_other(monkeypatch):
    """The isolation test, in both directions.

    Forward: the left arm's second step never runs, because the move after a
    failed move would start from a place this process cannot account for.
    Sideways: the right arm still runs its whole sequence, is still stopped, and
    is still asked. An arm that stopped answering is not a reason to leave the
    arm beside it standing.
    """
    log: list[str] = []
    rig = Rig(
        Arm("the left arm", log, ["above the table", "to the rest pose"], fail_on="above the table"),
        Arm("the right arm", log, ["above the table", "to the rest pose"]),
    )

    rc, out, err = _drive(rig, monkeypatch)

    assert rc == EXIT_STEP_FAILED
    assert log == [
        "declare:the left arm",
        "declare:the right arm",
        "run:above the table",           # the left arm's failing step
        "run:above the table",           # the right arm, unaffected
        "run:to the rest pose",
        "halt:the left arm",             # the failed arm is still de-energized
        "ask:the left arm",
        "halt:the right arm",
        "ask:the right arm",
    ]
    assert "stopped on 'above the table'" in err
    # And the summary says where each of them ended.
    assert "the left arm" in out and STEP_FAILED in out
    assert "the right arm" in out and RESTED in out


def test_an_arm_that_will_not_stop_does_not_skip_the_next_arm(monkeypatch):
    log: list[str] = []
    rig = Rig(
        Arm("the left arm", log, ["to the rest pose"], fail_halt=OSError("serial port closed")),
        Arm("the right arm", log, ["to the rest pose"]),
    )

    rc, _, err = _drive(rig, monkeypatch)

    assert rc == EXIT_UNCONFIRMED
    assert "halt:the right arm" in log and "ask:the right arm" in log
    assert "HALT UNANSWERED on the left arm" in err
    assert "power it down at the wall" in err.lower()


# --------------------------------------------------------------------------- #
# The confirmation — the promise the verb exists for
# --------------------------------------------------------------------------- #

def test_the_state_each_part_reports_is_printed_after_the_sequence(monkeypatch):
    log: list[str] = []
    rig = Rig(Arm("the left arm", log, ["to the rest pose"], state="idle"))

    rc, out, _ = _drive(rig, monkeypatch)

    assert rc == EXIT_RESTED
    assert "idle" in out
    assert log[-1] == "ask:the left arm"


def test_the_verb_prints_the_state_verbatim_and_grades_it_against_nothing(monkeypatch):
    """No verb-private definition of "off".

    A rig's motor-state vocabulary is the rig's. This verb certifies that it
    asked and got an answer, prints the answer, and leaves the meaning to the
    person reading it — which is why the summary says every part *answered*
    rather than every part is off. An allow-list of acceptable strings here
    would be this SDK deciding what "off" means for hardware it has never seen.
    """
    log: list[str] = []
    rig = Rig(Arm("the left arm", log, ["to the rest pose"], state="wherever it likes"))

    rc, out, _ = _drive(rig, monkeypatch)

    assert rc == EXIT_RESTED
    assert "wherever it likes" in out
    assert "EVERY PART ANSWERED" in out


def test_an_arm_that_cannot_be_asked_is_unconfirmed_and_never_called_rested(monkeypatch):
    """The completion criterion, stated as a test.

    The sequence ran perfectly. Nobody can say what the arm did with it. The one
    thing the run must not do is round that off to success — never exit 0, never
    the word rested, and the line that sends someone to the wall switch.
    """
    log: list[str] = []
    rig = Rig(
        Arm("the left arm", log, ["to the rest pose"], fail_state=OSError("no reply from the controller"))
    )

    rc, out, err = _drive(rig, monkeypatch)

    assert rc == EXIT_UNCONFIRMED
    assert "the left arm" in err
    assert "would not say what state it is in" in err
    assert "power it down at the wall" in err.lower()
    assert "UNCONFIRMED" in out
    assert "EVERY PART ANSWERED" not in out


def test_a_part_that_declares_no_way_to_be_asked_is_also_unconfirmed(monkeypatch):
    """Never asked and asked-but-silent are the same exit code and different sentences.

    Same physical next move, so the same code. Opposite fixes — one is an
    unwritten method and one is a rig that stopped answering — so never the same
    string.
    """
    log: list[str] = []
    rig = Rig(Arm("the left arm", log, ["to the rest pose"], no_state=True))

    rc, out, err = _drive(rig, monkeypatch)

    assert rc == EXIT_UNCONFIRMED
    assert "declares no motor_state()" in err
    assert "UNCONFIRMED" in out


def test_an_empty_state_reading_is_unconfirmed_rather_than_a_blank_success(monkeypatch):
    log: list[str] = []
    rig = Rig(Arm("the left arm", log, ["to the rest pose"], state=None))

    rc, _, err = _drive(rig, monkeypatch)

    assert rc == EXIT_UNCONFIRMED
    assert "came back empty" in err


def test_the_three_ways_to_end_unconfirmed_are_three_sentences(monkeypatch):
    """One exit code, three causes, three strings (Rule 12).

    The operator does the same thing about all three — walk up expecting it to
    be holding — which is why they share a code. The developer fixes three
    different things, which is why they never share a sentence.
    """
    said = []
    for arm in (
        Arm("arm", [], ["to the rest pose"], fail_halt=OSError("serial port closed")),
        Arm("arm", [], ["to the rest pose"], no_state=True),
        Arm("arm", [], ["to the rest pose"], fail_state=OSError("no reply")),
    ):
        out, err = io.StringIO(), io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        assert _run(Rig(arm)) == EXIT_UNCONFIRMED
        said.append(err.getvalue())

    assert len(set(said)) == 3, said


# --------------------------------------------------------------------------- #
# The exit-code table, and the four failure strings
# --------------------------------------------------------------------------- #

def test_the_exit_code_table_holds(monkeypatch):
    """Each code is a different thing the operator would do next.

    0 walk up to it · 2 nothing moved, go write a declaration · 3 something is
    where it stopped, go look · 4 assume it is holding until you have checked.
    A caller that cannot tell those apart cannot decide whether it is safe to
    approach the rig, which is the whole reason the table exists.
    """
    log: list[str] = []

    class SilentArm:
        name = "the left arm"

        def halt(self) -> None:
            log.append("halt")

    cases = [
        (EXIT_RESTED, Rig(Arm("arm", log, ["to the rest pose"]))),
        (EXIT_NOTHING_DECLARED, Rig(SilentArm())),
        (EXIT_STEP_FAILED, Rig(Arm("arm", log, ["to the rest pose"], fail_on="to the rest pose"))),
        (EXIT_UNCONFIRMED, Rig(Arm("arm", log, ["to the rest pose"], no_state=True))),
    ]
    for expected, rig in cases:
        rc, _, _ = _drive(rig, monkeypatch)
        assert rc == expected, (expected, rc)


def test_the_four_failure_causes_are_pairwise_distinct_strings(monkeypatch):
    """Nothing declared · a failed step · an arm that would not stop · an unreadable state.

    Four causes, four fixes, four sentences. Two of them sharing a string is the
    failure Rule 12 names: the reader is told something went wrong and cannot
    tell which thing, so they cannot tell whether the rig is where they left it.
    """
    log: list[str] = []

    class SilentArm:
        name = "the left arm"

        def halt(self) -> None:
            log.append("halt")

    strings = []
    for rig in (
        Rig(SilentArm()),
        Rig(Arm("the left arm", log, ["to the rest pose"], fail_on="to the rest pose")),
        Rig(Arm("the left arm", log, ["to the rest pose"], fail_halt=OSError("serial port closed"))),
        Rig(Arm("the left arm", log, ["to the rest pose"], fail_state=OSError("no reply"))),
    ):
        _, _, err = _drive(rig, monkeypatch)
        strings.append(err)

    assert len(set(strings)) == 4, strings
    # And each is recognisable on its own, not merely different from the others.
    assert "declares no rest sequence" in strings[0]
    assert "stopped on" in strings[1]
    assert "HALT UNANSWERED" in strings[2]
    assert "would not say what state it is in" in strings[3]


# --------------------------------------------------------------------------- #
# Ctrl+C — the moves are abandoned, the stop never is
# --------------------------------------------------------------------------- #

def test_ctrl_c_abandons_the_remaining_moves_and_stops_every_part(monkeypatch):
    """The interrupt takes away the moves and nothing else.

    An operator who hits Ctrl+C mid-rest wants the arms to stop going places,
    not to be left energized wherever they are. So the remaining sequences are
    dropped, every part is still de-energized and still asked, and the summary
    still says where each one ended — a run interrupted half-way is the state
    most in need of an honest report.
    """
    log: list[str] = []
    left = Arm("the left arm", log, ["above the table", "to the rest pose"])
    # Ctrl+C lands inside the left arm's first move.
    left.rest_sequence = lambda: [InterruptingStep("above the table", log)]  # type: ignore[method-assign]
    rig = Rig(left, Arm("the right arm", log, ["to the rest pose"]))

    rc, out, err = _drive(rig, monkeypatch)

    assert rc == 130
    assert "interrupted (Ctrl+C)" in err
    # Both parts stopped and both asked, even though only one ever moved.
    assert log.count("halt:the left arm") == 1
    assert log.count("halt:the right arm") == 1
    assert "run:to the rest pose" not in log
    assert ABANDONED in out


def test_an_unconfirmed_part_outranks_the_interrupt_in_the_exit_code(monkeypatch):
    """The one code that asks for something physical wins.

    Reporting "you pressed Ctrl+C" over "an arm may still be holding" would rank
    the operator's last keystroke above their safety.
    """
    log: list[str] = []
    arm = Arm("the left arm", log, ["above the table"], fail_state=OSError("no reply"))
    arm.rest_sequence = lambda: [InterruptingStep("above the table", log)]  # type: ignore[method-assign]

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    assert _run(Rig(arm)) == EXIT_UNCONFIRMED


# --------------------------------------------------------------------------- #
# The summary
# --------------------------------------------------------------------------- #

def test_the_summary_names_every_part_and_where_it_ended(monkeypatch):
    log: list[str] = []
    rig = Rig(
        Arm("the left arm", log, ["to the rest pose"], fail_on="to the rest pose"),
        Arm("the right arm", log, ["to the rest pose"], state="idle"),
    )

    rc, out, _ = _drive(rig, monkeypatch)

    assert rc == EXIT_STEP_FAILED
    assert "=== newt rest ===" in out
    assert "the left arm" in out and "the right arm" in out
    assert "a rig with the left arm, the right arm" in out
    assert "idle" in out


def test_a_rig_that_will_not_name_itself_is_still_put_away(monkeypatch):
    """describe() is a label, and this is the command you reach for after a fault.

    ``newt teleop`` refuses to start a session on a rig that will not answer,
    which is right for a verb that is about to drive it for minutes. Refusing to
    put arms away for the same reason would refuse at exactly the wrong moment.
    The label says what happened; it is never invented.
    """
    log: list[str] = []

    class Nameless(Rig):
        def describe(self) -> str:
            raise OSError("the controller is not answering")

    rc, out, _ = _drive(Nameless(Arm("the left arm", log, ["to the rest pose"])), monkeypatch)

    assert rc == EXIT_RESTED
    assert "halt:the left arm" in log
    assert "unnamed" in out and "describe()" in out


def test_declared_sequences_are_still_read_before_the_run_starts():
    """The read-everything-first guarantee survives the run being implemented."""
    log: list[str] = []
    rig = Rig(Arm("the left arm", log, ["to the rest pose"]), Arm("the right arm", log, ["to the rest pose"]))

    read_declarations(require_rest_source(rig))

    assert log == ["declare:the left arm", "declare:the right arm"]
    assert declared_sequence(rig._parts[0])[0].name == "to the rest pose"
