"""The declared rest sequence, and the refusal when there isn't one.

What these encode (the WHY, not just the WHAT):

- **An embodiment that declares no way to be put away gets a refusal, not a
  pose.** This is the whole argument of the verb. A default pose is a fact about
  hardware nobody here owns, served confidently to an operator who has no way to
  know it was invented — and a rig driven to a pose nobody chose for it is how
  something ends up somewhere nobody commanded.
- **The refusal has to be actionable.** It names the part that declared nothing,
  says what was missing, and says what declaring it would look like. A loud
  refusal that leaves the reader with no next move is only half the rule.
- **Nothing is commanded on the refusal path.** The test is not that the verb
  printed the right thing; it is that no driver call happened. An operator told
  "this rig cannot be put away" must be able to trust the rig is where they left
  it.
- **Every declaration is read before the first move.** On a two-armed rig where
  one arm declares and one does not, the refusal lands with both arms still
  standing. Half a rig put away is worse than a rig that never moved: the
  operator's mental model of where things are is wrong in a way they cannot see.
- **The two shapes of nothing are two strings.** A part with no
  ``rest_sequence()`` was never told what away means; a part whose
  ``rest_sequence()`` returns empty was asked and had nothing to say. Different
  causes, different fixes, so never the same sentence (Rule 12).
"""
from __future__ import annotations

import io
import sys

import pytest

from newt._cli.rest import cmd_rest
from newt.rest import (
    EXIT_NOTHING_DECLARED,
    EXIT_USAGE,
    NothingDeclared,
    RestError,
    declared_sequence,
    read_declarations,
    require_rest_source,
)


# --------------------------------------------------------------------------- #
# Fakes — every one of them records whether it was ever driven
# --------------------------------------------------------------------------- #

class Step:
    """A named step that records the moment it ran."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self._log = log

    def run(self) -> None:
        self._log.append(f"run:{self.name}")


class Arm:
    """A part that declares whatever the test tells it to.

    ``log`` is shared across every arm in a rig so ordering across arms is
    visible, and so "was anything commanded at all" is one assertion.
    """

    def __init__(self, name: str, log: list[str], steps: list[str] | None) -> None:
        self.name = name
        self._log = log
        self._steps = steps

    def rest_sequence(self):
        self._log.append(f"declare:{self.name}")
        return [Step(f"{self.name}/{s}", self._log) for s in (self._steps or [])]

    def halt(self) -> None:
        self._log.append(f"halt:{self.name}")


class SilentArm:
    """A part that never had rest_sequence() written on it at all."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self._log = log

    def halt(self) -> None:
        self._log.append(f"halt:{self.name}")


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


# --------------------------------------------------------------------------- #
# The refusal — the card's whole argument
# --------------------------------------------------------------------------- #

def test_a_part_that_declares_no_rest_sequence_is_refused_by_name(monkeypatch):
    log: list[str] = []
    rig = Rig(SilentArm("the left arm", log))

    rc, _, err = _drive(rig, monkeypatch)

    assert rc == EXIT_NOTHING_DECLARED
    assert "the left arm" in err
    assert "declares no rest sequence" in err


def test_the_refusal_says_what_declaring_it_would_look_like(monkeypatch):
    log: list[str] = []
    rc, _, err = _drive(Rig(SilentArm("the left arm", log)), monkeypatch)

    assert rc == EXIT_NOTHING_DECLARED
    # What was missing, whose problem it is, and the next move (Rule 12).
    assert "rest_sequence()" in err
    assert "in order" in err
    assert "name and a run()" in err
    # And it must never offer to pick one.
    assert "default" in err.lower()


def test_nothing_is_commanded_when_a_part_declares_nothing(monkeypatch):
    """The assertion that matters: no driver call, not just the right sentence."""
    log: list[str] = []
    rc, _, _ = _drive(Rig(SilentArm("the left arm", log)), monkeypatch)

    assert rc == EXIT_NOTHING_DECLARED
    assert log == [], f"the rig was touched on the refusal path: {log}"


def test_one_silent_arm_refuses_the_whole_run_before_the_other_arm_moves(monkeypatch):
    """A rig is not half put away.

    The declaring arm is listed first on purpose: a verb that checked lazily
    would have driven it through its whole sequence before discovering the
    second arm had nothing to declare, and the operator would be told "this rig
    cannot be put away" while one arm was already somewhere else.
    """
    log: list[str] = []
    rig = Rig(Arm("the left arm", log, ["to the rest pose"]), SilentArm("the right arm", log))

    rc, _, err = _drive(rig, monkeypatch)

    assert rc == EXIT_NOTHING_DECLARED
    assert "the right arm" in err
    assert not [entry for entry in log if entry.startswith(("run:", "halt:"))], log
    assert "Nothing was commanded on any part" in err


def test_an_empty_sequence_and_a_missing_one_are_different_sentences():
    log: list[str] = []
    with pytest.raises(NothingDeclared) as missing:
        declared_sequence(SilentArm("the left arm", log))
    with pytest.raises(NothingDeclared) as empty:
        declared_sequence(Arm("the left arm", log, []))

    assert str(missing.value) != str(empty.value)
    # Each has to be recognisable on its own, not just different.
    assert "declares no rest sequence" in str(missing.value)
    assert "empty rest sequence" in str(empty.value)
    assert "answered 'nothing'" in str(empty.value)


# --------------------------------------------------------------------------- #
# Reading a declaration that is there
# --------------------------------------------------------------------------- #

def test_a_declared_sequence_is_returned_in_the_order_the_rig_gave_it():
    log: list[str] = []
    steps = declared_sequence(Arm("the left arm", log, ["above the table", "to the rest pose"]))

    assert [s.name for s in steps] == [
        "the left arm/above the table",
        "the left arm/to the rest pose",
    ]
    # Declaring is not doing.
    assert log == ["declare:the left arm"]


def test_reading_declarations_touches_nothing_but_the_declaration():
    log: list[str] = []
    rig = Rig(Arm("the left arm", log, ["to the rest pose"]), Arm("the right arm", log, ["to the rest pose"]))

    plan = read_declarations(require_rest_source(rig))

    assert [part.name for part, _ in plan] == ["the left arm", "the right arm"]
    assert log == ["declare:the left arm", "declare:the right arm"]


def test_the_verb_reads_step_names_out_of_the_rig_and_imposes_none():
    """The vocabulary is the rig's, whatever it is.

    Open question the card refuses to close: vendor names carry the calibration
    collision, our names impose a taxonomy on hardware we do not own, and opaque
    names are honest and unhelpful. So the verb accepts any string and checks
    none of them — this test is the guard against a future pass quietly adding
    an allow-list.
    """
    log: list[str] = []
    steps = declared_sequence(Arm("arm", log, ["step 1", "sleep", "wherever it likes"]))
    assert [s.name for s in steps] == ["arm/step 1", "arm/sleep", "arm/wherever it likes"]


# --------------------------------------------------------------------------- #
# The seam's own refusals — a source that cannot be driven at all
# --------------------------------------------------------------------------- #

def test_a_source_that_is_not_a_rest_source_is_refused_before_anything(monkeypatch):
    class NotARig:
        pass

    rc, _, err = _drive(NotARig(), monkeypatch)
    assert rc == EXIT_USAGE
    assert "moving_parts()" in err
    assert "newt.rest.RestSource" in err


def test_a_rig_that_declares_no_moving_parts_is_refused():
    with pytest.raises(RestError) as exc:
        require_rest_source(Rig())
    assert "declares no moving parts" in str(exc.value)


def test_a_part_with_no_halt_is_refused_before_it_is_moved():
    """Moving something you cannot then de-energize is worse than not moving it.

    An arm left holding fights whoever picks it up next, so a part that can be
    driven but not stopped is refused before the first step rather than driven
    and abandoned.
    """
    log: list[str] = []

    class NoHalt:
        name = "the left arm"

        def rest_sequence(self):
            return [Step("to the rest pose", log)]

    with pytest.raises(RestError) as exc:
        require_rest_source(Rig(NoHalt()))
    assert "declares no halt()" in str(exc.value)
    assert "will not invent a way" in str(exc.value)
    assert log == []


def test_a_part_whose_declaration_raises_is_not_the_same_as_one_with_none():
    """A rig that could not answer and a rig with nothing to say want different fixes.

    The first is a link or a power problem; the second is an unwritten
    declaration. Same silence, opposite next steps, so never the same string.
    """
    log: list[str] = []

    class Faulted:
        name = "the left arm"

        def rest_sequence(self):
            raise OSError("serial port closed")

        def halt(self) -> None:
            log.append("halt")

    with pytest.raises(RestError) as raised:
        declared_sequence(Faulted())
    with pytest.raises(NothingDeclared) as missing:
        declared_sequence(SilentArm("the left arm", log))

    assert "could not say what its rest sequence is" in str(raised.value)
    assert str(raised.value) != str(missing.value)
    assert not isinstance(raised.value, NothingDeclared)
    assert log == []


def test_an_unnamed_step_is_refused_because_a_failure_could_not_be_located():
    log: list[str] = []

    class Nameless:
        name = "the left arm"

        def rest_sequence(self):
            return [Step("", log)]

        def halt(self) -> None:
            log.append("halt")

    with pytest.raises(RestError) as exc:
        declared_sequence(Nameless())
    assert "has no name" in str(exc.value)
    assert log == []
