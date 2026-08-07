"""The flow — announce, move, solve, judge, and only then write.

The order in this file is the card, and it is the only thing here that is not
negotiable:

    read every declaration  →  announce the motion  →  confirm  →  run the passes
    →  solve  →  render the verdict  →  write

Two of those arrows are load-bearing. The announcement comes before the
confirmation *and before the motion*, on every path, because suppressing a
question is a different act from suppressing a warning and the way a warning
quietly stops existing is by being bolted to a prompt somebody turned off. And
the write comes after the verdict is on the operator's screen, because the defect
this verb was filed against is a system that computes a quality number, prints it
with no scale beside it, and saves regardless — a 50-pixel solve landing as
readily as a 0.2-pixel one.

Nothing here decides what a good number is. It compares against a threshold the
rig declared and prints a word next to it.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from newt.calibration._seam import (
    EXIT_CALIBRATED,
    EXIT_PASS_FAILED,
    EXIT_SOLVE_FAILED,
    EXIT_SUSPECT,
    EXIT_USAGE,
    CalibrationError,
    CalibrationPass,
    CalibrationProcedure,
    CameraQuality,
    Verdict,
    _motion_line,
    check_cameras,
    declared_passes,
    declared_threshold,
    judge,
)

#: What ``--skip`` records when the operator gave no words of their own. It says
#: what happened and invents no motive — a plausible reason written here would be
#: this process putting words in somebody's mouth and then shipping them to every
#: episode recorded afterwards.
SKIPPED_WITHOUT_REASON = "skipped by the operator, who gave no reason"


def _say(line: str, *, error: bool = False) -> None:
    """Print as it happens, never held for a summary.

    The operator may be walking toward the rig while this runs, so a line about
    an arm that is about to move has to reach them before the run does.
    """
    print(f"[newt calibrate] {line}", file=sys.stderr if error else sys.stdout, flush=True)


def _plain(line: str = "") -> None:
    print(line, flush=True)


# --------------------------------------------------------------------------- #
# The announcement — printed on every path, before the first millimetre
# --------------------------------------------------------------------------- #

def announce_motion(rig: str, passes: Sequence[CalibrationPass]) -> None:
    """Say what is about to move, in the rig's own words, before it moves.

    Read from each pass's declared ``motion()`` rather than composed here: this
    process does not know how far an arm travels or how long it takes, and a
    sentence it made up would be a reassurance rather than a warning.
    """
    _plain()
    _say(f"THE RIG IS ABOUT TO MOVE — {rig}")
    for index, item in enumerate(passes, start=1):
        _plain(f"    pass {index} · {item.name} — {_motion_line(item)}")
    _plain(
        "    Clear the workspace. Keep hands, cables and anything you care about out "
        "of the arm's reach until this finishes."
    )
    _plain()


def confirm_motion(*, assume_yes: bool) -> bool:
    """Ask before moving, unless the caller already answered with ``--yes``.

    ``--yes`` suppresses the question and nothing else. The announcement above has
    already printed by the time this is called, on every path, which is the whole
    reason the two are separate functions rather than one block guarded by a flag.

    A non-interactive caller with no ``--yes`` is refused rather than assumed to
    have agreed. Nobody typed anything, so nobody agreed to anything, and a rig
    that starts moving on that reading is one this verb refuses to be.
    """
    if assume_yes:
        _say(
            "--yes was given, so the confirmation is skipped. The warning above is not "
            "skipped and never is."
        )
        return True

    if not sys.stdin.isatty():
        _say(
            "refused before moving: there is no terminal here to confirm at, and nobody "
            "typed anything.\n"
            "Ours: this verb will not read an absent answer as a yes when an arm is the "
            "thing that moves next.\n"
            "Do now: nothing has moved.\n"
            "Then: run it at a terminal, or pass --yes to say out loud that this is an "
            "unattended run and the workspace is clear.",
            error=True,
        )
        return False

    try:
        answer = input("[newt calibrate] Workspace clear? Type y to move, anything else to stop: ")
    except (EOFError, KeyboardInterrupt):
        _plain()
        _say("stopped before moving — no answer given. Nothing has moved.", error=True)
        return False

    if answer.strip().lower() in ("y", "yes"):
        return True
    _say("stopped before moving, on your answer. Nothing has moved.")
    return False


# --------------------------------------------------------------------------- #
# The verdict — rendered before anything is written, on every path
# --------------------------------------------------------------------------- #

def render_verdict(verdict: Verdict) -> None:
    """Print every camera's number, the threshold it was judged against, and the word.

    Never a bare number. A first-timer reading ``· 2.34px`` cannot know whether
    that is excellent or a redo, and the system this verb replaces printed exactly
    that and then saved.
    """
    width = max((len(q.camera) for q in verdict.qualities), default=0) + 1
    _plain()
    _plain("=== newt calibrate ===")
    _plain(f"threshold:   {verdict.threshold_px:g} px  (declared by this rig)")
    _plain("cameras:")
    for q in verdict.qualities:
        mark = "within" if q.reprojection_px <= verdict.threshold_px else "OUTSIDE"
        _plain(
            f"  {(q.camera + ':').ljust(width)} {q.reprojection_px:.2f} px  · {mark}"
        )
    _plain(f"verdict:     {verdict.word}")
    if verdict.within:
        _plain(
            "             every camera is inside the number this rig declared. This is a "
            "good calibration."
        )
    else:
        worst = verdict.worst
        _plain(
            f"             {worst.camera} is {worst.reprojection_px:.2f} px against a "
            f"{verdict.threshold_px:g} px threshold. The result is being saved and "
            "labelled suspect —"
        )
        _plain(
            "             it is yours to use or redo, and every episode recorded under it "
            "will carry the label."
        )
    _plain()


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #

def _describe(procedure: CalibrationProcedure) -> str:
    """One line naming the rig, or a line saying it would not give one.

    A ``describe()`` that raises is not a reason to refuse to calibrate — the rig
    would not state its own name, which is a worse label and not a worse rig. The
    label is never invented; it says what happened.
    """
    try:
        return str(procedure.describe())
    except Exception as exc:  # noqa: BLE001 — a raising describe() costs us a
        # label and nothing else; refusing the whole procedure over it would be
        # refusing for a reason that has no bearing on whether the solve works.
        return f"unnamed ({type(exc).__name__} from describe(): {exc})"


def _run_passes(passes: Sequence[CalibrationPass]) -> str | None:
    """Drive every pass in the declared order. Returns why it stopped, or None.

    A pass that raises ends the run. Unlike ``newt rest``, there is nothing to
    isolate here: the passes feed one solve, and continuing after one of them
    failed would be gathering half the observations and then solving as if they
    were all of them.
    """
    total = len(passes)
    for index, item in enumerate(passes, start=1):
        _say(f"pass {index}/{total} · {item.name} — {_motion_line(item)}")
        try:
            item.run(lambda line, name=item.name: _say(f"  {name}: {line}"))
        except CalibrationError:
            raise
        except Exception as exc:  # noqa: BLE001 — a pass is arbitrary rig-declared
            # motion and detection; whatever it raises is the operator's answer
            # about what went wrong on the bench, and this verb surfaces it rather
            # than tracing it.
            return (
                f"pass {index}/{total} {item.name!r} did not finish "
                f"({type(exc).__name__}: {exc}).\n"
                "Yours: the rig stopped part-way through a stretch of motion. It is "
                "wherever that pass left it, and the remaining passes are NOT being run — "
                "solving from half the observations would produce a number from a "
                "procedure that never happened.\n"
                "Do now: check where the arm is before reaching into the workspace. "
                "Nothing has been written; the calibration you had before is untouched.\n"
                "Then: if the board was never seen, move it into more of the cameras' "
                "views and run this again. If the arm refused the move, that is a rig "
                "problem this verb cannot solve for you."
            )
    return None


def run_calibration(
    procedure: CalibrationProcedure,
    *,
    assume_yes: bool = False,
    source_receipt: str | None = None,
) -> int:
    """Run the whole procedure and return the exit code.

    Called only after the factory has produced the object and
    :func:`newt.calibration.require_calibration_procedure` has accepted it. Every
    declaration is read at the top, before the announcement, so a rig that cannot
    be calibrated is told so while it is still standing rather than two passes in.
    """
    rig = _describe(procedure)
    provenance = f", brought up by {source_receipt}" if source_receipt else ""
    _say(f"calibrating — {rig}{provenance}")

    # Read everything first. Each of these refuses with the rig untouched, and
    # each raises a type the caller maps to its own exit code.
    threshold = declared_threshold(procedure)
    passes = declared_passes(procedure)
    cameras = check_cameras(procedure)
    _say(f"{len(cameras)} declared camera(s) connected: {', '.join(cameras)}")

    announce_motion(rig, passes)
    if not confirm_motion(assume_yes=assume_yes):
        return EXIT_USAGE

    try:
        stopped = _run_passes(passes)
    except KeyboardInterrupt:
        _plain()
        _say(
            "interrupted (Ctrl+C) mid-pass. The rig is wherever that pass left it — check "
            "before reaching in. Nothing has been written and the calibration you had "
            "before is untouched.",
            error=True,
        )
        return 130
    if stopped is not None:
        _say(stopped, error=True)
        return EXIT_PASS_FAILED

    try:
        qualities = _solved(procedure)
    except KeyboardInterrupt:
        _plain()
        _say(
            "interrupted (Ctrl+C) during the solve. The passes ran; the answer was never "
            "finished, never judged and never written.",
            error=True,
        )
        return 130
    except CalibrationError as exc:
        _say(str(exc), error=True)
        return EXIT_SOLVE_FAILED

    # Nothing above this line wrote anything, and nothing below it writes before
    # the next two statements have run in this order.
    verdict = judge(qualities, threshold)
    render_verdict(verdict)

    try:
        receipt = procedure.commit(verdict)
    except Exception as exc:  # noqa: BLE001 — commit() is the kit's own persistence
        # and can raise anything; the operator needs the number they just read to
        # not be quietly lost, so this says outright that it was.
        _say(
            f"the verdict above was reached and could not be saved "
            f"({type(exc).__name__}: {exc}).\n"
            "Yours: the rig's own commit() refused. The numbers you just read are real; "
            "nothing on disk knows them.\n"
            "Do now: the calibration you had before this run is untouched — this verb "
            "writes through the rig and the rig did not write.\n"
            "Then: check that the path this kit writes its calibration to exists and is "
            "writable, then run this again.",
            error=True,
        )
        return EXIT_SOLVE_FAILED

    _say(f"receipt: {receipt}")
    if verdict.within:
        return EXIT_CALIBRATED
    _say(
        "saved and labelled suspect — this is a result you can see the number for, not a "
        "green light. Redo it when you can."
    )
    return EXIT_SUSPECT


def _solved(procedure: CalibrationProcedure) -> Sequence[CameraQuality]:
    """Ask the rig for one number per camera, or refuse with what it said.

    Two shapes of nothing get two refusals, because they want two different next
    moves: a solver that raised has an answer about why, and a solver that
    returned nothing has quietly produced a run with no numbers to judge — which
    is the state an unjudged save comes out of.
    """
    _say("solving — this can take minutes. The rig reports below as it goes.")
    try:
        qualities = list(procedure.solve(lambda line: _say(f"  solve: {line}")))
    except CalibrationError:
        raise
    except Exception as exc:
        # The rig's solver is arbitrary code wrapping arbitrary numerics: a
        # non-convergence, a singular matrix and a missing dependency all arrive
        # here and all mean the same thing to the operator — there is no answer,
        # and nothing was written.
        raise CalibrationError(
            f"the solve did not produce an answer ({type(exc).__name__}: {exc}).\n"
            "Yours: this is the rig's own solver reporting, not this verb refusing to try. "
            "The passes ran and the observations they gathered did not resolve into "
            "camera positions.\n"
            "Do now: nothing has been written. The calibration you had before is "
            "untouched.\n"
            "Then: the usual cause is too few views of the board — run it again holding "
            "the board at more angles, and further into each camera's frame."
        ) from exc

    if not qualities:
        raise CalibrationError(
            "the solve returned no cameras, so there is nothing to judge and nothing to "
            "save.\n"
            "Yours: solve() came back empty. A run that produced no numbers is not a run "
            "that passed — this verb will not treat an empty answer as a good one.\n"
            "Do now: nothing has been written. The calibration you had before is "
            "untouched.\n"
            "Then: check that the passes actually saw the board — a procedure that "
            "gathered no observations solves to nothing."
        )
    return qualities


# --------------------------------------------------------------------------- #
# The skip — a declared decision, never an absence
# --------------------------------------------------------------------------- #

def run_skip(procedure: CalibrationProcedure, reason: str | None) -> int:
    """Write the rig's geometry-invalid declaration and say what it costs.

    This is Rule 10 in its plainest form. A person proving their cameras stream is
    not blocked by a procedure they do not need yet — and the identity geometry
    they get is a decision they made and can see, written down, travelling to
    every episode recorded under it. Never an absence, never a shaped-right
    default nobody chose.

    Nothing moves and nothing solves. The only thing this path is allowed to be
    quiet about is the arm.
    """
    rig = _describe(procedure)
    words = (reason or "").strip() or SKIPPED_WITHOUT_REASON
    try:
        written = procedure.skip(words)
    except Exception as exc:  # noqa: BLE001 — the kit's own persistence again, and
        # a skip that failed to record is worse than one that never ran: the
        # operator believes their episodes are marked and they are not.
        _say(
            f"the skip could not be recorded ({type(exc).__name__}: {exc}).\n"
            "Yours: the rig's own skip() refused, so nothing on disk says this rig's "
            "geometry is unmeasured.\n"
            "Do now: do NOT record yet. Episodes made now would claim geometry nobody "
            "measured, with nothing saying so — which is the exact failure this flag "
            "exists to prevent.\n"
            "Then: check that the path this kit writes its calibration to exists and is "
            "writable, then run this again.",
            error=True,
        )
        return EXIT_USAGE

    _plain()
    _plain("=== newt calibrate — skipped ===")
    _plain(f"rig:         {rig}")
    _plain(f"reason:      {words}")
    _plain(f"written:     {written}")
    _plain(
        "geometry:    NOT MEASURED. Every episode you record under this rig now carries a "
        "mark saying so."
    )
    _plain(
        "             Your recordings are still real — frames, timestamps, actions. Any "
        "claim about where"
    )
    _plain(
        "             something is in space is not, and now nobody downstream has to guess "
        "that."
    )
    _plain()
    _say("run `newt calibrate` when you want real geometry. Nothing about this is a fault.")
    return EXIT_CALIBRATED
