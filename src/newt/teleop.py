"""newt.teleop — the teleoperation loop, and the seam an embodiment implements.

``newt teleop`` is the frontend; this is the session. The split is the one
``newt record`` set: the CLI parses, refuses to run without a kill key, and
renders. Everything about *when* a thing happens — the tick, the order of the
ending, which exit the session is on — lives here. Everything about *what*
happens lives behind the seam, in the developer's source.

**The seam is an input, a sink, a rate, and a halt.** A source produces actions;
the same object accepts them. Whether the actions come from a second embodiment
someone is holding, a dial box, a phone, or a replayed trajectory is not
something this module can tell, and that is the point — those are instances of
an input, not the interface.

**The pass-through.** The action the source produces is handed to the sink
unchanged. No scaling, no filtering, no interpolation, no unit conversion. A
guessed transform between two action spaces is how something ends up somewhere
nobody commanded, and the sink's own planner is what bridges the gap between
ticks.

**The kill does not move anything.** Ctrl+H de-energizes every declared moving
part where it stands and exits 130. Motion after the panic key is the thing the
panic key exists to prevent — including the rest move, which is skipped the
moment the kill fires even if it has already started on another part.

**The normal exit does move.** Ctrl+C puts each part away by its own declared
rest, then de-energizes it, and exits 0. The two exits differ by exactly one
thing — whether anything was put somewhere first — and the code knows which it
is on.

**The tick can also be a recording's clock.** ``run_session(recorder=...)`` is
what ``newt record --teleop`` passes: after each action is sent, the source is
asked what state that tick produced and the answer is handed to the recorder.
One loop, because the arms take one client — and the driven channel is whatever
the source says it is, so a rig that reports the action it was just given
records the action that actually drove it rather than a second read of it.

**A halt is never invented here.** The verb calls what the embodiment declares
and reports what the embodiment declares afterward, verbatim. A source that
declares no halt is refused, loudly, before anything is driven: a second
definition of "off" living inside this module would be a definition no rig
agreed to.

**A refused halt is not swallowed.** ``newt.recording``'s session teardown
suppresses a failing torque-off, because there the torque-off is a courtesy on
the way out of a read-only session. Here it is the whole safety contract, so a
part that refuses is printed at the moment it refuses, named in the summary,
and carried out in the exit code.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

DEFAULT_RATE_HZ = 30.0


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #

@runtime_checkable
class MovingPart(Protocol):
    """One independently stoppable piece of an embodiment.

    A part is whatever the rig says can be de-energized on its own — one of a
    pair, a base, a single axis. The verb never counts them or names them; it
    stops all of them and prints what each one is called.

    ``halt()`` de-energizes where it stands and **must not move first**.
    ``motor_state()`` (optional) is what the part declares about itself
    afterward; the verb prints the string and does not interpret it, because a
    vocabulary of motor states is rig knowledge. ``rest()`` (optional) is the
    rig's own "put yourself away" — the pose, the vendor call, the sequence.
    """

    name: str

    def halt(self) -> None:
        ...


@runtime_checkable
class TeleopSource(Protocol):
    """What ``run_session`` drives. The only surface a new embodiment implements.

    ``describe()`` is one line naming the rig for the operator and the summary.
    ``moving_parts()`` declares what can be stopped. ``read_action()`` produces
    an action; ``send_action()`` accepts it unchanged. What an action *is* is
    entirely the source's business — the verb never looks inside one.
    """

    def describe(self) -> str:
        ...

    def moving_parts(self) -> Sequence[MovingPart]:
        ...

    def read_action(self) -> Any:
        ...

    def send_action(self, action: Any) -> None:
        ...


@runtime_checkable
class TickRecorder(Protocol):
    """Where a driven tick is written down, when the operator asked for that.

    The loop knows two things about a recorder and no more: every tick has a
    state to hand it, and the session ends either kept or discarded. What an
    episode is, what format it is in, and where it lands are none of this
    module's business — the frontend that passed the recorder in owns all of it,
    the same way it owns the keyboard.

    ``record_tick`` receives what ``source.read_state()`` returned for the tick
    that has just been *sent*, and the timestamp of that tick. ``finish`` runs
    once, before the rig is put away or de-energized: the rest move is motion
    nobody demonstrated, and it must not land inside the recording.
    """

    def record_tick(self, channels: Any, ts_ns: int) -> None:
        ...

    def finish(self, *, keep: bool) -> None:
        ...


#: The attribute a rig sets to say that one object both drives the rig and
#: records it, so a demonstration can be recorded from a single tick. Public and
#: importable on purpose: it is a word two codebases have to agree on, and a kit
#: author copying a string out of somebody's CLI internals has no way to know
#: when it changes.
#:
#: It is read in two places, and they answer different questions. On the
#: **factory**, before it is called — *may this rig be built at all for this?* —
#: which is the only check that can happen while nothing is connected. On the
#: **object** the factory returned, after — *did the thing it built back the
#: claim?* The first is the gate; the second catches a factory that declared
#: something it did not deliver.
DRIVES_AND_RECORDS = "drives_and_records"


def drives_and_records(factory):
    """Declare that this factory builds a rig that drives *and* records at once.

    A source factory decorated with this is saying the object it returns will
    both accept actions and report state for the same tick — the shape a
    recorded demonstration needs. It is asked for and never inferred: an object
    carrying ``send_action`` next to ``read_state`` is a shape, and a shape is
    not a statement that driving and recording the same rig at once is a thing
    this rig means to do.

    The decorator exists so the claim is readable *before the factory runs*.
    Construction is what connects and energizes hardware, so a verb that builds
    first and validates second has already moved metal it never approved::

        from newt.teleop import drives_and_records

        @drives_and_records
        def make_demo():
            # returns one object that accepts actions and reports the state the
            # tick it just drove produced
            return MyRig(bring_up(recorded=True))

    The object it returns sets the same attribute (``drives_and_records = True``)
    for the after-construction check.
    """
    setattr(factory, DRIVES_AND_RECORDS, True)
    return factory


class ActionRejected(ValueError):
    """Raise from ``send_action`` when the sink refuses the action as malformed.

    This is the one distinction the verb cannot make for itself: an action the
    sink will not accept (the input and the sink disagree about what an action
    is) and a sink that stopped answering (a link dropped, power went) are the
    same exception type from most drivers and want opposite fixes. The source
    knows which one it saw; raising this says so, and the operator gets the
    message written for that cause.
    """


class TeleopError(RuntimeError):
    """A session that has to stop, carrying its own operator-facing text."""


class _KillFired(Exception):
    """Ctrl+H, raised out of the loop. Never crosses this module's boundary."""


@dataclass(frozen=True)
class HaltReport:
    """What one part did when it was told to de-energize, and what it said after."""

    part: str
    confirmed: bool
    state: str | None = None
    detail: str | None = None


class Tally:
    """How the session went, counted as it goes.

    Mutable and passed in rather than returned, because the loop only ever exits
    by raising — a return value would be lost on exactly the paths whose counts
    matter most (the kill, the fault), and a summary reporting zero ticks after
    a real session is a lie the operator has no way to catch.
    """

    def __init__(self) -> None:
        self.ticks = 0
        #: Ticks whose read+send took longer than the period — the sink was
        #: already behind whatever is driving it.
        self.overruns = 0


# --------------------------------------------------------------------------- #
# Checking the source before anything is driven
# --------------------------------------------------------------------------- #

def _require_source(source: Any) -> list[MovingPart]:
    """Return the declared moving parts, or raise the refusal that names what is missing.

    This runs after the factory — the earliest moment the object exists — and
    before a single action is read. Whatever the factory connected is already
    connected by now, which is exactly why the refusals below say so instead of
    implying nothing happened.
    """
    for member in ("describe", "moving_parts", "read_action", "send_action"):
        if not callable(getattr(source, member, None)):
            raise TeleopError(
                f"The source does not implement {member}(), so it is not a teleop source.\n"
                "Yours: --source built an object this verb cannot drive.\n"
                "Do now: nothing has been read and nothing has been driven. Whatever the "
                "factory connected is still connected; this verb has no declared way to "
                "de-energize it.\n"
                "Then: a teleop source declares describe(), moving_parts(), read_action(), "
                "and send_action(). See newt.teleop.TeleopSource."
            )

    try:
        parts = list(source.moving_parts())
    except Exception as exc:
        raise TeleopError(
            f"The source could not say what its moving parts are "
            f"({type(exc).__name__}: {exc}).\n"
            "Yours: moving_parts() raised. If it queries the rig to build that list, the "
            "rig did not answer.\n"
            "Do now: nothing has been read and nothing has been driven — but whatever the "
            "factory connected is still connected, and this verb was never told what it "
            "could stop, so it has de-energized nothing. Power the rig down at the wall if "
            "it may be holding.\n"
            "Then: make moving_parts() a statement about the rig rather than a query of "
            "it — declare the parts the factory already brought up."
        ) from exc

    if not parts:
        raise TeleopError(
            "The source declares no moving parts, so there is nothing this verb could "
            "stop.\n"
            "Yours: moving_parts() came back empty from a source that is about to be "
            "driven.\n"
            "Do now: nothing has been read and nothing has been driven.\n"
            "Then: declare every independently de-energizable piece of the rig from "
            "moving_parts(), each with a name and a halt()."
        )

    for part in parts:
        name = getattr(part, "name", None)
        if not name:
            raise TeleopError(
                "A declared moving part has no name, so a halt line about it could not "
                "tell the operator which piece of the rig it means.\n"
                "Yours: one of the objects moving_parts() returned is missing `name`.\n"
                "Do now: nothing has been read and nothing has been driven.\n"
                "Then: give every moving part a `name` — it is what the operator reads "
                "when the kill fires."
            )
        if not callable(getattr(part, "halt", None)):
            raise TeleopError(
                f"The moving part {name!r} declares no halt(), so this session could be "
                "started and not stopped.\n"
                "Yours: the rig has not said how that part de-energizes, and this verb "
                "will not invent a way — a second definition of 'off' is one no rig "
                "agreed to.\n"
                "Do now: nothing has been read and nothing has been driven. Whatever the "
                "factory connected is still connected, and this process cannot "
                "de-energize it for you.\n"
                "Then: give that part a halt() that de-energizes where it stands without "
                "moving first."
            )
    return parts


# --------------------------------------------------------------------------- #
# Stopping — the one thing every exit has in common
# --------------------------------------------------------------------------- #

def _declared_state(part: MovingPart) -> str | None:
    """What the part says about itself, or None if it does not say.

    Printed verbatim and never compared against anything: a vocabulary of motor
    states belongs to the rig, and a verb that decided which strings meant "off"
    would be deciding it for rigs it has never seen.
    """
    reporter = getattr(part, "motor_state", None)
    if not callable(reporter):
        return None
    try:
        state = reporter()
    except Exception as exc:
        return f"unreported ({type(exc).__name__}: {exc})"
    return None if state is None else str(state)


def _halt_all(parts: Sequence[MovingPart]) -> list[HaltReport]:
    """De-energize every part, each in its own try. Returns one report per part.

    A kill that gives up halfway is the failure this exists to prevent, so one
    part's refusal never skips the next part's attempt — and every refusal is
    printed the moment it happens rather than saved for the summary, because the
    operator may be walking toward the rig while this runs.
    """
    reports: list[HaltReport] = []
    for part in parts:
        name = part.name
        try:
            part.halt()
        except Exception as exc:
            reports.append(
                HaltReport(name, False, None, f"{type(exc).__name__}: {exc}")
            )
            print(
                f"[newt teleop] HALT FAILED on {name} ({type(exc).__name__}: {exc}). "
                f"That part may STILL BE HOLDING TORQUE — power it down at the wall "
                "before approaching it.",
                file=sys.stderr,
                flush=True,
            )
            continue
        state = _declared_state(part)
        reports.append(HaltReport(name, True, state, None))
        print(
            f"[newt teleop] {name} de-energized"
            + (f" (it reports: {state})" if state else "")
            + " — back-drivable; it will settle under gravity.",
            file=sys.stderr,
            flush=True,
        )
    return reports


def _put_away(
    parts: Sequence[MovingPart], kill: threading.Event
) -> tuple[list[HaltReport], bool]:
    """The normal exit: rest each part, then de-energize all of them.

    Returns the halt reports and whether the kill fired partway through. Rest is
    a *move*, so the kill is checked before each one: a Ctrl+H pressed while the
    rig is putting itself away has to stop the remaining rest moves, not be
    honoured after them. The de-energize still runs for every part either way —
    the rest is what gets abandoned, never the stop.

    A part with no declared rest is named rather than silently skipped. There is
    nothing wrong with an embodiment that has nowhere to be put; there is
    something wrong with an operator who thinks it went somewhere.
    """
    killed = False
    for part in parts:
        if kill.is_set():
            killed = True
            print(
                "[newt teleop] kill fired during the rest move — the remaining parts "
                "are NOT being put away; every part is being de-energized where it "
                "stands.",
                file=sys.stderr,
                flush=True,
            )
            break
        rest = getattr(part, "rest", None)
        if not callable(rest):
            print(
                f"[newt teleop] {part.name} declares no rest move — de-energizing it "
                "where it is.",
                flush=True,
            )
            continue
        try:
            print(f"[newt teleop] {part.name}: putting itself away…", flush=True)
            rest()
        except Exception as exc:
            print(
                f"[newt teleop] {part.name} did not complete its rest move "
                f"({type(exc).__name__}: {exc}); de-energizing it from wherever it is. "
                "Check where it stopped before the next run.",
                file=sys.stderr,
                flush=True,
            )
    return _halt_all(parts), killed


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #

def _check_kill(kill: threading.Event) -> None:
    if kill.is_set():
        raise _KillFired()


def _loop(
    source: TeleopSource,
    rate_hz: float,
    kill: threading.Event,
    tally: Tally,
    recorder: "TickRecorder | None" = None,
) -> None:
    """Hand each action from the source to the sink, at rate_hz, until aborted.

    Only ever exits by raising: _KillFired (Ctrl+H), KeyboardInterrupt (Ctrl+C),
    or TeleopError.

    A failed read is not a skipped tick here. Recording can drop a frame and say
    so, because a gap in a dataset is a gap. Teleop cannot: the sink would sit
    still while whoever is driving keeps moving and believes it is tracking. So
    the first failure ends the session.

    With a ``recorder``, this tick is also the recording's clock. The state is
    read *after* the send and handed straight over, so what the episode holds for
    a tick is the state of a rig that has already been told where to go — and the
    source is the one that decides what that state is, including whether its
    driven channel repeats the action it was just given rather than re-reading
    it. The verb never reads twice and never looks inside either value.
    """
    period = 1.0 / rate_hz

    while True:
        # Top of tick: a Ctrl+H pressed during the last sleep lands here.
        _check_kill(kill)
        started = time.perf_counter()

        try:
            action = source.read_action()
        except Exception as exc:
            raise TeleopError(
                f"The source stopped producing actions after {tally.ticks} tick(s) "
                f"({type(exc).__name__}: {exc}).\n"
                "Yours: whatever is driving this session went unreachable mid-run — a "
                "link dropped, power went, or it faulted.\n"
                "Do now: every part is being de-energized. Nothing stale was sent; the "
                "sink holds where it was last told to go.\n"
                "Then: check the input's link and power before re-running."
            ) from exc

        # The read is usually a round trip. Re-check before the write so a Ctrl+H
        # pressed *during* it does not cost the sink one more commanded move.
        _check_kill(kill)

        try:
            source.send_action(action)
        except ActionRejected as exc:
            raise TeleopError(
                f"The sink rejected the action on tick {tally.ticks}: {exc}\n"
                "Yours: the input and the sink disagree about what an action is. This "
                "loop is a pass-through by contract, so nothing here translated it.\n"
                "Do now: every part is being de-energized.\n"
                "Then: make the input produce what the sink declares it accepts. Do not "
                "add a translation layer — a guessed mapping between two action spaces "
                "is how a rig ends up somewhere nobody commanded."
            ) from exc
        except Exception as exc:
            raise TeleopError(
                f"The sink stopped accepting actions after {tally.ticks} tick(s) "
                f"({type(exc).__name__}: {exc}).\n"
                "Yours: the driven part went unreachable or refused the move — a link "
                "dropped, power went, or it was commanded past a limit.\n"
                "Do now: every part is being de-energized. The input is fine; it is the "
                "sink that stopped.\n"
                "Then: check the driven part's link and power, and where it was standing "
                "when it stopped."
            ) from exc

        tally.ticks += 1

        if recorder is not None:
            # Stamped here, not in the writer: this is the moment the rig was
            # commanded, and a timestamp taken further down the path would date
            # the frame by how long the path took.
            ts_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
            try:
                channels = source.read_state()
            except Exception as exc:
                raise TeleopError(
                    f"The source drove the rig but could not say what it did, on tick "
                    f"{tally.ticks} ({type(exc).__name__}: {exc}).\n"
                    "Yours: read_state() raised on a source that had just accepted an "
                    "action. The rig is being driven; it is the recording half of the "
                    "same object that stopped answering.\n"
                    "Do now: every part is being de-energized, and the episode is kept "
                    "with the ticks that did record — nothing was invented for this "
                    "one.\n"
                    "Then: a composed source's read_state() reports drops as None per "
                    "channel; raising from it means something worse than a missed read, "
                    "and its message says what."
                ) from exc
            recorder.record_tick(channels, ts_ns)

        remaining = period - (time.perf_counter() - started)
        if remaining > 0:
            # Event.wait, not sleep: a Ctrl+H pressed mid-period wakes the loop
            # now rather than at the end of it.
            if kill.wait(remaining):
                _check_kill(kill)
        else:
            tally.overruns += 1


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #

def run_session(
    source: TeleopSource,
    *,
    rate_hz: float = DEFAULT_RATE_HZ,
    kill: threading.Event,
    recorder: "TickRecorder | None" = None,
) -> int:
    """Drive ``source`` at ``rate_hz`` until the kill fires or the operator ends it.

    Returns the process exit code: 0 ended cleanly, 1 the source refused or a
    part would not confirm it was de-energized, 130 the kill fired.

    ``kill`` is set by whoever owns the keyboard. This module never reads a key;
    it waits on the event, which is what lets a Ctrl+H pressed mid-period land
    immediately instead of at the next tick boundary.

    The ending happens in a ``finally``, not after the try: an exception nobody
    expected — a bug in the loop, an OOM — must still leave every part
    de-energized before its traceback surfaces.

    ``recorder``, when given, makes the tick the clock of a recording as well as
    of the rig. It is closed out first in that ending, before anything is put
    away: **the kill discards, every other exit keeps.** A kill leaves no
    episode because a panic stop is not a demonstration; a fault keeps what was
    driven up to it, because those ticks happened and throwing them away would
    lose a take the operator gave. Which one it was is said out loud by the
    recorder at the moment it happens, not inferred later from an exit code.
    """
    try:
        parts = _require_source(source)
    except TeleopError as exc:
        print(f"\n[newt teleop] {exc}", file=sys.stderr)
        return 1

    tally = Tally()
    outcome = "error"
    # Bound before the try, because the summary names the rig on every path —
    # including the one where asking the rig its name is what failed.
    rig = "unnamed (the source could not describe itself)"
    started = time.perf_counter()
    try:
        # Asked once, inside the try that owns the ending. Once, because the
        # banner and the summary must name the same rig and a description may be
        # a round trip to hardware rather than a constant. Inside, because by
        # here the factory has connected and the parts are declared: a
        # describe() that queries a rig can fail with that rig energized, and
        # every such path has to reach the halt instead of a traceback.
        try:
            rig = str(source.describe())
        except Exception as exc:
            raise TeleopError(
                f"The source could not describe itself ({type(exc).__name__}: {exc}).\n"
                "Yours: describe() raised on a source that had already declared its "
                "parts. If it reads the rig to build that line, the rig did not answer.\n"
                "Do now: nothing has been read and nothing has been driven; every "
                "declared part is being de-energized.\n"
                "Then: make describe() a statement about the rig rather than a query of "
                "it. It is a label for the operator, not a health check."
            ) from exc

        print(
            f"[newt teleop] live at {rate_hz:g} Hz — {rig}. "
            "Ctrl+C to end (it puts the rig away first), Ctrl+H to kill (it does not).",
            flush=True,
        )
        _loop(source, rate_hz, kill, tally, recorder)
    except _KillFired:
        outcome = "emergency_stop"
    except KeyboardInterrupt:
        outcome = "interrupted"
        print("\n[newt teleop] ending session (Ctrl+C).", flush=True)
    except TeleopError as exc:
        outcome = "failed"
        print(f"\n{exc}", file=sys.stderr)
    finally:
        wall_s = time.perf_counter() - started
        if recorder is not None:
            # First, and never after: putting the rig away is a move nobody
            # demonstrated, and an episode still open through it would record the
            # arms stowing themselves as part of the task.
            try:
                recorder.finish(keep=outcome != "emergency_stop")
            except Exception as exc:
                # Loud, and then on with the ending. A recording that could not be
                # closed out is bad; a rig left energized because closing it out
                # raised is worse.
                print(
                    f"[newt teleop] the recording could not be closed out "
                    f"({type(exc).__name__}: {exc}) — continuing to de-energize the rig. "
                    "Check what landed on disk before trusting it.",
                    file=sys.stderr,
                    flush=True,
                )
        if outcome == "interrupted":
            reports, killed_mid_rest = _put_away(parts, kill)
            if killed_mid_rest:
                outcome = "emergency_stop"
        else:
            reports = _halt_all(parts)

    refused = [r.part for r in reports if not r.confirmed]
    achieved = tally.ticks / wall_s if wall_s > 0 else 0.0
    print(
        f"\n=== teleop session ===\n"
        f"rig:         {rig}\n"
        f"outcome:     {outcome}\n"
        f"rate:        {rate_hz:g} Hz requested · {achieved:.1f} Hz achieved\n"
        f"ticks:       {tally.ticks} in {wall_s:.1f}s ({tally.overruns} over-period)\n"
        f"parts:       "
        + (
            "ALL CONFIRMED DE-ENERGIZED"
            if not refused
            else "NOT CONFIRMED OFF: " + ", ".join(refused)
        )
    )

    if refused:
        # Louder than the exit code, because this is the one outcome where the
        # operator has to do something physical before walking up to the rig.
        print(
            f"\n[newt teleop] {', '.join(refused)} did not confirm de-energized. Power "
            "it down at the wall before approaching it — this process cannot tell you "
            "whether it is holding.",
            file=sys.stderr,
        )
        return 1
    if outcome == "emergency_stop":
        return 130
    if outcome != "interrupted":
        return 1
    return 0
