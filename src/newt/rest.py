"""newt.rest — running an embodiment's own declared rest sequence, and reporting where it ended up.

``newt rest`` is the frontend; this is the seam and the contract. The split is
the one ``newt record`` set and ``newt teleop`` kept: the CLI parses, guards, and
renders, and everything about *when* a thing happens lives here.

**The rig declares the sequence; this module knows no poses.** A part says what
putting itself away means — which poses, in what order, over how long, through
which driver call — by declaring ``rest_sequence()``. This module executes the
declaration in the order it was given, then de-energizes, then reports what each
part says about itself. It holds no pose, no default, no vendor concept, and no
knowledge of any particular rig. Everything it knows about an embodiment, the
embodiment told it.

**A part that declares no rest sequence is refused, not given one.** There is
nothing wrong with an embodiment that has nowhere to be put. There is something
very wrong with a process that picks a pose for it and drives there — a
shaped-right default, served confidently, is how a rig ends up somewhere nobody
commanded. So the refusal names the part, says what it did not declare, and says
what declaring it would look like. It is the feature, not the gap.

**Every declaration is read before anything is commanded.** The whole run is
refused if any part declares nothing, and the refusal lands before the first
step of the first part — an operator who is told "this rig cannot be put away"
should be told it with the rig still standing where it was, not with one arm
already moved and the other refused.

**This module never decides what "off" means.** De-energizing is the part's own
declared halt, and the final state is whatever the part reports about itself,
printed verbatim and compared against nothing. A verb that decided which strings
meant "off" would be deciding it for rigs it has never seen.

**It is not calibration.** Nothing here changes what an arm believes about its
own zero. No set-zero, no factory reset, no jig. A rest sequence moves an arm to
where its rig says to leave it, and that is all it does.

Exit codes:
  0    every part ran its declared sequence and confirmed it ended de-energized
  1    a usage error, or the source refused to come up
  2    refused before anything was commanded — a part declares no rest sequence
  3    a declared step failed; that part did not finish its sequence
  4    a part went unanswered, or its final state could not be confirmed — treat
       it as still holding until you have checked it yourself
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

#: Every part ran its declared sequence and confirmed it ended de-energized.
EXIT_RESTED = 0
#: A usage error, or the source refused to come up.
EXIT_USAGE = 1
#: Refused before anything was commanded. Nothing moved.
EXIT_NOTHING_DECLARED = 2
#: A declared step failed. That part did not finish its sequence.
EXIT_STEP_FAILED = 3
#: A part went unanswered, or would not say what state it ended in. The loudest
#: outcome, and the only one where the operator has to do something physical
#: before walking up to the rig.
EXIT_UNCONFIRMED = 4


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #

@runtime_checkable
class RestStep(Protocol):
    """One step of a part's declared way of putting itself away.

    ``name`` is what the operator reads while it runs and what a failure is
    named by. The vocabulary is the rig's: this module never checks a step name
    against anything, because a taxonomy of poses would be one imposed on
    hardware nobody here owns.

    ``run()`` does the step and returns when the part has finished it. It may
    block for as long as the move takes — how long a step is, is one of the
    facts the rig owns.
    """

    name: str

    def run(self) -> None:
        ...


@runtime_checkable
class Restable(Protocol):
    """One independently rested piece of an embodiment.

    Deliberately spelled with the members ``newt.teleop.MovingPart`` already
    uses — ``name``, ``halt()``, and the optional ``motor_state()`` — so a rig
    that can be teleoperated and a rig that can be put away are one object with
    one vocabulary rather than two seams for one idea. The member this adds is
    ``rest_sequence()``.

    ``halt()`` de-energizes where the part stands and must not move first.
    ``motor_state()`` (optional) is what the part declares about itself
    afterward; it is printed verbatim and interpreted by nothing here.
    """

    name: str

    def rest_sequence(self) -> Sequence[RestStep]:
        ...

    def halt(self) -> None:
        ...


@runtime_checkable
class RestSource(Protocol):
    """What ``newt rest`` is pointed at. The only surface a new embodiment implements.

    ``describe()`` is one line naming the rig for the operator and the summary.
    ``moving_parts()`` declares the pieces that can be put away and stopped
    independently — the same declaration ``newt.teleop`` drives, so the object a
    kit already builds for one verb is the object the other verb wants.
    """

    def describe(self) -> str:
        ...

    def moving_parts(self) -> Sequence[Restable]:
        ...


class RestError(RuntimeError):
    """A run that has to stop, carrying its own operator-facing text."""


class NothingDeclared(RestError):
    """No part could be asked to rest, and nothing was commanded.

    Its own type because it is the one refusal that guarantees the rig is
    exactly where the operator left it. Every other failure happened partway
    through something.
    """


# --------------------------------------------------------------------------- #
# Reading the declaration — everything below runs before anything is commanded
# --------------------------------------------------------------------------- #

def require_rest_source(source: object) -> list[Restable]:
    """Return the declared parts, or raise the refusal naming what is missing.

    Runs immediately after the factory — the earliest moment the object exists —
    and before a single step. Whatever the factory connected is connected by
    now, which is why the refusals say so instead of implying nothing happened.
    """
    for member in ("describe", "moving_parts"):
        if not callable(getattr(source, member, None)):
            raise RestError(
                f"The source does not implement {member}(), so it is not something this "
                "verb can put away.\n"
                "Yours: --source built an object this verb cannot read a rest sequence "
                "off.\n"
                "Do now: nothing has been commanded. Whatever the factory connected is "
                "still connected, and this verb has no declared way to de-energize it.\n"
                "Then: a rest source declares describe() and moving_parts(). See "
                "newt.rest.RestSource."
            )

    try:
        parts = list(source.moving_parts())
    except Exception as exc:
        raise RestError(
            f"The source could not say what its moving parts are "
            f"({type(exc).__name__}: {exc}).\n"
            "Yours: moving_parts() raised. If it queries the rig to build that list, the "
            "rig did not answer.\n"
            "Do now: nothing has been commanded — but whatever the factory connected is "
            "still connected, and this verb was never told what it could put away or "
            "stop. Power the rig down at the wall if it may be holding.\n"
            "Then: make moving_parts() a statement about the rig rather than a query of "
            "it — declare the parts the factory already brought up."
        ) from exc

    if not parts:
        raise RestError(
            "The source declares no moving parts, so there is nothing this verb could "
            "put away.\n"
            "Yours: moving_parts() came back empty.\n"
            "Do now: nothing has been commanded.\n"
            "Then: declare every independently rested piece of the rig from "
            "moving_parts(), each with a name, a rest_sequence(), and a halt()."
        )

    for part in parts:
        name = getattr(part, "name", None)
        if not name:
            raise RestError(
                "A declared moving part has no name, so a line about where it ended up "
                "could not tell the operator which piece of the rig it means.\n"
                "Yours: one of the objects moving_parts() returned is missing `name`.\n"
                "Do now: nothing has been commanded.\n"
                "Then: give every moving part a `name` — it is what the operator reads "
                "when this verb tells them where things finished."
            )
        if not callable(getattr(part, "halt", None)):
            raise RestError(
                f"The moving part {name!r} declares no halt(), so this verb could move it "
                "and then not be able to leave it de-energized.\n"
                "Yours: the rig has not said how that part de-energizes, and this verb "
                "will not invent a way — a second definition of 'off' is one no rig "
                "agreed to.\n"
                "Do now: nothing has been commanded. Whatever the factory connected is "
                "still connected, and this process cannot de-energize it for you.\n"
                "Then: give that part a halt() that de-energizes where it stands without "
                "moving first. An arm left holding fights whoever picks it up next."
            )
    return parts


def declared_sequence(part: Restable) -> list[RestStep]:
    """Return one part's declared rest sequence, or raise the refusal naming it.

    Two shapes of nothing, two strings, because they want different fixes: a
    part with no ``rest_sequence()`` at all has never been told what away means,
    and a part whose ``rest_sequence()`` comes back empty has been asked and had
    nothing to say. Neither gets a pose from here.
    """
    name = part.name
    declare = getattr(part, "rest_sequence", None)
    if not callable(declare):
        raise NothingDeclared(
            f"{name} declares no rest sequence, so this verb has not been told what "
            "putting it away means.\n"
            "Yours: the part has no rest_sequence(), and there is no default — a pose "
            "picked here would be one nobody chose for this hardware.\n"
            "Do now: nothing has been commanded. Every part is exactly where you left "
            "it.\n"
            "Then: give the part a rest_sequence() returning the steps it should be "
            "driven through, in order, each with a name and a run(). One step is a "
            "sequence. The rig decides what they are; this verb only decides when."
        )

    try:
        steps = list(declare())
    except Exception as exc:
        raise RestError(
            f"{name} could not say what its rest sequence is "
            f"({type(exc).__name__}: {exc}).\n"
            "Yours: rest_sequence() raised. If it reads the rig to build the sequence, "
            "the rig did not answer.\n"
            "Do now: nothing has been commanded. Every part is exactly where you left "
            "it.\n"
            "Then: make rest_sequence() a statement about the rig rather than a query of "
            "it — the steps are a fact about the hardware, not a reading off it."
        ) from exc

    if not steps:
        raise NothingDeclared(
            f"{name} declares an empty rest sequence, so it has been asked what putting "
            "itself away means and answered 'nothing'.\n"
            "Yours: rest_sequence() returned no steps. An empty sequence is not a way to "
            "say 'leave it where it is' — this verb cannot tell that apart from a "
            "declaration that was never filled in.\n"
            "Do now: nothing has been commanded. Every part is exactly where you left "
            "it.\n"
            "Then: return the steps the part should be driven through, in order. If the "
            "part genuinely has nowhere to be put, say so by not declaring "
            "rest_sequence() at all."
        )

    for index, step in enumerate(steps):
        step_name = getattr(step, "name", None)
        if not step_name:
            raise RestError(
                f"Step {index + 1} of {name}'s rest sequence has no name, so a failure "
                "part-way through could not tell the operator which move it stopped on.\n"
                f"Yours: one of the steps {name}'s rest_sequence() returned is missing "
                "`name`.\n"
                "Do now: nothing has been commanded.\n"
                "Then: name every step in whatever vocabulary the rig already uses. This "
                "verb never reads the name — the operator does."
            )
        if not callable(getattr(step, "run", None)):
            raise RestError(
                f"Step {step_name!r} of {name}'s rest sequence declares no run(), so "
                "there is nothing for this verb to execute.\n"
                "Yours: the step names a move but does not carry one.\n"
                "Do now: nothing has been commanded.\n"
                "Then: give the step a run() that performs it and returns when the part "
                "has finished it. Blocking is fine — how long a step takes is the rig's "
                "business."
            )
    return steps


def read_declarations(parts: Sequence[Restable]) -> list[tuple[Restable, list[RestStep]]]:
    """Pair every part with its declared sequence, or refuse the whole run.

    Read in full before the first step of the first part, and that ordering is
    the point: on a rig where one arm declares a sequence and another does not,
    the operator is told the rig cannot be put away while both arms are still
    standing where they were — rather than after one of them has already moved.

    Failure *isolation* is a different thing and applies later: once the run
    starts, one part's step failing never skips another part's attempt.
    """
    plan: list[tuple[Restable, list[RestStep]]] = []
    undeclared: list[str] = []
    reasons: list[str] = []
    for part in parts:
        try:
            plan.append((part, declared_sequence(part)))
        except NothingDeclared as exc:
            undeclared.append(part.name)
            reasons.append(str(exc))

    if undeclared:
        raise NothingDeclared(
            "\n\n".join(reasons)
            + (
                "\n\nNothing was commanded on any part. "
                f"Refused before the first move because {', '.join(undeclared)} "
                "could not be put away, and a rig half put away is worse than a rig "
                "that never moved."
                if len(parts) > 1
                else ""
            )
        )
    return plan
