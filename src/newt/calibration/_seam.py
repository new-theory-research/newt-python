"""The contract a kit implements, and every refusal that lands before motion.

Split from ``_run`` on the line that matters here: everything in this file runs
while the rig is standing exactly where the operator left it. Nothing below
commands hardware, and every refusal in it can honestly say so.

**What ``newt`` knows about calibration, in full.** That a procedure has passes,
that passes move things, that cameras have identifiers, that a solve produces one
number per camera, and that a number is compared against a threshold somebody
declared. It does not know what the printed target is or what its squares
measure, how many passes this rig wants, what a good number is, how a camera is
addressed, or how a result is persisted. Every one of those is an embodiment fact and every one of them arrives
through the procedure the kit declared.

**The identifiers are opaque.** Camera identifiers and pass names are compared,
printed, and never parsed. No substring is special, no prefix is meaningful, no
convention is inferred — which is the direct inversion of the defect this verb
was filed against, where a rig could not calibrate unless an arm id contained the
substring ``"right"``. A rig whose only arm is ``arm0`` calibrates, and so does
one whose cameras are called ``banana`` and ``kumquat``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: Solved, and every camera's number is inside the declared threshold. Also the
#: code for a skip, which succeeded at exactly what the operator asked for — the
#: mark it writes is the honest part, not the exit code.
EXIT_CALIBRATED = 0
#: A usage error, or the source refused to come up.
EXIT_USAGE = 1
#: Refused before anything moved — the rig declares nothing this verb can run.
EXIT_NOTHING_DECLARED = 2
#: Refused before anything moved — the cameras the rig declares are not the
#: cameras this machine can see.
EXIT_CAMERAS_UNAVAILABLE = 3
#: A pass did not finish. The arm moved; the run stopped part-way through it.
EXIT_PASS_FAILED = 4
#: The solve did not answer. Nothing was judged, so nothing was written.
EXIT_SOLVE_FAILED = 5
#: Solved, judged, written — and at least one camera is outside the declared
#: threshold. The result carries a suspect label and so will every episode
#: recorded under it. Never 0: a number nobody vouches for must not exit green.
EXIT_SUSPECT = 6


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #

#: How a pass or a solve says what is happening while it happens. ``newt``
#: supplies it, prints whatever it is handed, and interprets none of it — the
#: vocabulary of "the left camera can see the board right now" is the rig's.
Notice = Callable[[str], None]


@dataclass(frozen=True)
class CameraQuality:
    """One camera's answer to "how far off is this", in pixels.

    ``camera`` is the identifier the rig uses for it, printed back verbatim.
    ``reprojection_px`` is the number the rig's own solver produced. This class
    carries no verdict: whether the number is fine is decided against a threshold
    the rig declared, one layer up, so that a solver cannot both produce a number
    and grade it.
    """

    camera: str
    reprojection_px: float


@dataclass(frozen=True)
class Verdict:
    """The judgment, composed before anything is written and printed before too.

    Exists as its own object so the ordering is structural rather than
    remembered: :func:`newt.calibration.judge` builds it, the flow prints it, and
    only then is ``commit(verdict)`` allowed to be called. A procedure that wants
    to write cannot get a ``Verdict`` without one having been rendered.
    """

    qualities: tuple[CameraQuality, ...]
    threshold_px: float
    #: The word an operator reads. WITHIN or OUTSIDE — never a bare number.
    word: str
    #: True when every camera is at or under the declared threshold.
    within: bool

    @property
    def worst(self) -> CameraQuality | None:
        return max(self.qualities, key=lambda q: q.reprojection_px, default=None)


@runtime_checkable
class CalibrationPass(Protocol):
    """One stretch of the procedure that moves the rig.

    ``name`` is what the operator reads while it runs and what a failure is named
    by; the vocabulary is the rig's and nothing here checks it against anything.

    ``motion()`` is one plain line saying what is about to move, roughly how far,
    and roughly how long. It is read *before* the pass runs and printed before the
    first millimetre — a pass that cannot say what it will do is refused rather
    than run silently, because the announcement is the requirement and the motion
    is only the thing it announces.

    ``run(notice)`` performs the pass and returns when the rig has finished it.
    Blocking is fine. Call ``notice(line)`` as often as there is something true to
    say — whether the board is being seen right now is the line a person standing
    at the rig actually needs, and a pass that says nothing for forty seconds of
    continuous motion is the flow this verb was filed to replace.
    """

    name: str

    def motion(self) -> str:
        ...

    def run(self, notice: Notice) -> None:
        ...


@runtime_checkable
class CalibrationProcedure(Protocol):
    """What ``newt calibrate`` is pointed at. The only surface a kit implements.

    One declared name is one procedure. A rig that offers camera extrinsics and
    the arm's own joint calibration declares two, and the selector then appears on
    its own — the operator types ``newt calibrate --source cameras`` only on the
    rig where the one-word form is genuinely ambiguous.

    ``describe()`` is one line naming the rig and this procedure for the operator.

    ``declared_cameras()`` and ``connected_cameras()`` are the two halves of one
    question asked at two moments: what the rig's config says it has, and what
    this machine can actually see right now. They are compared here and enumerated
    side by side in the refusal, which is the whole of the fix for an inherited
    error string that covered both "nothing is plugged in" and "the serials in
    your config are not these serials" with one sentence and no list.

    ``quality_threshold()`` is the pixel number this rig calls acceptable. It is
    declared, never defaulted — a threshold baked into ``newt`` would be an
    embodiment fact in the SDK, and a threshold silently assumed would be a number
    nobody chose deciding whether somebody's data is any good.

    ``passes()`` declares the stretches of motion, in the order they run.

    ``solve(notice)`` returns one :class:`CameraQuality` per camera. It is called
    after every pass and before any judgment; it writes nothing.

    ``commit(verdict)`` persists the result and returns the path of the receipt it
    wrote, which this verb prints. It is called only after the verdict has been
    rendered to the operator, and it is handed that verdict so the receipt can
    carry the number, the threshold and the word rather than only the answer.

    ``skip(reason)`` writes the rig's geometry-invalid declaration and returns the
    path it wrote. Nothing solves, nothing moves. The reason is the operator's
    words, or a sentence saying they gave none — never a plausible one invented
    here.
    """

    def describe(self) -> str:
        ...

    def declared_cameras(self) -> Sequence[str]:
        ...

    def connected_cameras(self) -> Sequence[str]:
        ...

    def quality_threshold(self) -> float:
        ...

    def passes(self) -> Sequence[CalibrationPass]:
        ...

    def solve(self, notice: Notice) -> Sequence[CameraQuality]:
        ...

    def commit(self, verdict: Verdict) -> str:
        ...

    def skip(self, reason: str) -> str:
        ...


class CalibrationError(RuntimeError):
    """A run that has to stop, carrying its own operator-facing text."""


class NothingDeclared(CalibrationError):
    """The rig declares nothing this verb could run, and nothing was commanded.

    Its own type because it is the refusal that guarantees the rig is exactly
    where the operator left it, and because it is the one an exit code has to be
    able to tell apart from "we tried and it went wrong".
    """


class CamerasUnavailable(CalibrationError):
    """The declared cameras are not the cameras this machine can see.

    Its own type for the same reason it has two distinct messages below: the
    operator's next move is physical for one cause and textual for the other, and
    an exit code that could not separate them would send them to the wrong one.
    """


# --------------------------------------------------------------------------- #
# Reading the declaration — everything below runs before anything is commanded
# --------------------------------------------------------------------------- #

_REQUIRED = (
    "describe",
    "declared_cameras",
    "connected_cameras",
    "quality_threshold",
    "passes",
    "solve",
    "commit",
    "skip",
)


def require_calibration_procedure(source: object) -> None:
    """Refuse anything that is not a calibration procedure, by the member it lacks.

    Runs immediately after the factory — the earliest moment the object exists.
    Whatever the factory connected is connected by now, which is why the refusal
    says so rather than implying nothing happened.
    """
    missing = [m for m in _REQUIRED if not callable(getattr(source, m, None))]
    if not missing:
        return
    raise CalibrationError(
        f"The source does not implement {', '.join(f'{m}()' for m in missing)}, so it is "
        "not a procedure this verb can run.\n"
        "Yours: --source built an object, and this verb cannot read a calibration "
        "procedure off it.\n"
        "Do now: nothing has moved. Whatever the factory connected is still connected.\n"
        "Then: a calibration procedure declares "
        f"{', '.join(f'{m}()' for m in _REQUIRED)}. See newt.calibration."
        "CalibrationProcedure — the docstring says what each one owes."
    )


def declared_threshold(procedure: CalibrationProcedure) -> float:
    """Return the rig's declared quality threshold, or refuse to invent one.

    The refusal is the point. A number defaulted here would be this SDK deciding,
    for hardware it has never seen, how far off is too far off — and it would do
    it silently, at the one moment a person is deciding whether to trust their own
    data. There is no default and there is not going to be one.
    """
    try:
        raw = procedure.quality_threshold()
    except Exception as exc:
        raise NothingDeclared(
            f"This rig could not say what reprojection error it calls acceptable "
            f"({type(exc).__name__}: {exc}).\n"
            "Yours: quality_threshold() raised. If it reads the number off a config, the "
            "read failed.\n"
            "Do now: nothing has moved.\n"
            "Then: make quality_threshold() a statement about this rig — the number is a "
            "property of these cameras and this bench, and it is declared once where the "
            "rest of the rig is declared."
        ) from exc

    if raw is None:
        raise NothingDeclared(
            "This rig declares no quality threshold, so there is no number to judge a "
            "solve against — and this verb will not pick one.\n"
            "Yours: quality_threshold() came back empty. A threshold chosen here would be "
            "the SDK deciding how good is good enough for cameras it has never seen.\n"
            "Do now: nothing has moved.\n"
            "Then: declare the acceptable mean reprojection error, in pixels, alongside "
            "the rest of this rig's declarations. If you do not know yet, say 2.0 and "
            "write down that you guessed — a guess you can see beats a default you cannot."
        )

    try:
        threshold = float(raw)
    except (TypeError, ValueError) as exc:
        raise NothingDeclared(
            f"This rig's quality threshold is {raw!r}, which is not a number of pixels.\n"
            "Yours: quality_threshold() returned something this verb cannot compare a "
            "reprojection error against.\n"
            "Do now: nothing has moved.\n"
            "Then: return the threshold as a number — mean reprojection error in pixels, "
            "per camera."
        ) from exc

    if threshold <= 0:
        raise NothingDeclared(
            f"This rig's quality threshold is {threshold} px, which no solve can ever be "
            "inside.\n"
            "Yours: a threshold at or below zero grades every camera OUTSIDE, which is a "
            "verdict word that has stopped meaning anything.\n"
            "Do now: nothing has moved.\n"
            "Then: declare a positive threshold in pixels — the number above which you "
            "would redo this calibration rather than record under it."
        )
    return threshold


def declared_passes(procedure: CalibrationProcedure) -> list[CalibrationPass]:
    """Return the declared passes with their motion lines already read, or refuse.

    Every ``motion()`` is read here, before the first pass runs, and that ordering
    is the requirement rather than a tidiness: the announcement has to be
    printable in full before anything moves, so a pass that cannot say what it
    will do is caught while the rig is still standing rather than three passes in.
    """
    try:
        passes = list(procedure.passes())
    except Exception as exc:
        raise NothingDeclared(
            f"This rig could not say what its calibration passes are "
            f"({type(exc).__name__}: {exc}).\n"
            "Yours: passes() raised. If it queries the rig to build the list, the rig did "
            "not answer.\n"
            "Do now: nothing has moved.\n"
            "Then: make passes() a statement about the procedure rather than a query of "
            "the hardware — which stretches of motion, in what order, is a fact about this "
            "rig and not a reading off it."
        ) from exc

    if not passes:
        raise NothingDeclared(
            "This rig declares an empty calibration procedure, so it has been asked what "
            "calibrating means here and answered 'nothing'.\n"
            "Yours: passes() returned no passes, and there is no default set of poses — "
            "one picked here would be motion nobody chose for this hardware.\n"
            "Do now: nothing has moved.\n"
            "Then: declare the passes this procedure runs, in order, each with a name, a "
            "motion() line, and a run()."
        )

    for index, item in enumerate(passes):
        name = getattr(item, "name", None)
        if not name:
            raise NothingDeclared(
                f"Pass {index + 1} of this procedure has no name, so a failure part-way "
                "through could not tell the operator which stretch of motion it stopped "
                "on.\n"
                "Yours: one of the objects passes() returned is missing `name`.\n"
                "Do now: nothing has moved.\n"
                "Then: name every pass in whatever vocabulary this rig already uses. This "
                "verb never reads the name — the operator does."
            )
        if not callable(getattr(item, "run", None)):
            raise NothingDeclared(
                f"The pass {name!r} declares no run(), so there is nothing for this verb "
                "to execute.\n"
                "Yours: the pass names a stretch of motion but does not carry one.\n"
                "Do now: nothing has moved.\n"
                "Then: give the pass a run(notice) that performs it and returns when the "
                "rig has finished it."
            )
        if not _motion_line(item):
            raise NothingDeclared(
                f"The pass {name!r} will not say what it is about to move, so this verb "
                "cannot announce it — and a pass it cannot announce is a pass it will not "
                "run.\n"
                "Yours: motion() is missing, raised, or came back empty. Somebody may be "
                "standing next to this arm, and the sentence telling them it is about to "
                "move is built from this line.\n"
                "Do now: nothing has moved.\n"
                "Then: return one plain line from motion() — what moves, roughly how far, "
                "roughly how long. 'both arms sweep nine poses over the table, about 45 "
                "seconds' is the whole job."
            )
    return passes


def _motion_line(item: object) -> str | None:
    """One pass's announcement line, or None when it does not have one.

    A ``motion()`` that raises and a ``motion()`` that returns nothing are the
    same fact for the caller — this verb has no sentence to print — so they get
    one refusal rather than two that ask for the same fix.
    """
    declare = getattr(item, "motion", None)
    if not callable(declare):
        return None
    try:
        line = declare()
    except Exception:  # noqa: BLE001 — a motion() that raises has failed to give
        # us a line, which is the same absence as one that returned nothing; the
        # refusal above asks for the same fix either way and says so.
        return None
    if line is None:
        return None
    text = str(line).strip()
    return text or None


def check_cameras(procedure: CalibrationProcedure) -> list[str]:
    """Compare declared cameras against seen cameras, or refuse with both lists.

    Two causes, two messages, both enumerating expected against seen — the direct
    inversion of an inherited ``"No matching cameras connected"`` that covered
    "nothing is plugged in" and "your config names different serials" with one
    string and no list. An operator reading the first should go and plug something
    in; one reading the second should go and edit a file. One sentence cannot send
    them to both.
    """
    try:
        declared = [str(c) for c in procedure.declared_cameras()]
    except Exception as exc:
        raise NothingDeclared(
            f"This rig could not say which cameras it has "
            f"({type(exc).__name__}: {exc}).\n"
            "Yours: declared_cameras() raised.\n"
            "Do now: nothing has moved.\n"
            "Then: return the identifiers this rig's config names — the same strings you "
            "would read in that file — from declared_cameras()."
        ) from exc

    if not declared:
        raise NothingDeclared(
            "This rig declares no cameras, so there is nothing whose position this "
            "procedure could solve for.\n"
            "Yours: declared_cameras() came back empty, and this verb will not go looking "
            "for cameras nobody declared.\n"
            "Do now: nothing has moved.\n"
            "Then: declare this rig's cameras where the rest of it is declared, then run "
            "this again."
        )

    try:
        seen = [str(c) for c in procedure.connected_cameras()]
    except Exception as exc:
        raise CamerasUnavailable(
            f"This rig could not be asked which cameras are actually connected "
            f"({type(exc).__name__}: {exc}).\n"
            f"Yours: connected_cameras() raised. This rig declares: {_listed(declared)}.\n"
            "Do now: nothing has moved.\n"
            "Then: this is a read of the machine rather than of the config — check that "
            "the camera runtime this kit uses is installed and can enumerate devices."
        ) from exc

    matched = [c for c in declared if c in seen]
    if matched:
        return matched

    if not seen:
        raise CamerasUnavailable(
            f"None of this rig's {len(declared)} declared cameras is connected — this "
            "machine sees no cameras at all.\n"
            f"    declared: {_listed(declared)}\n"
            "    seen:     nothing\n"
            "Yours: the config is not the problem here; the machine is enumerating zero "
            "devices, so nothing to compare against arrived.\n"
            "Do now: nothing has moved.\n"
            "Then: plug the cameras in, check the cable and the port, and run this again. "
            "If they are plugged in, the runtime this kit uses is not seeing them at all "
            "— that is a driver problem, not a calibration one."
        )

    raise CamerasUnavailable(
        f"This machine sees {len(seen)} camera(s), and not one of them is a camera this "
        "rig declares.\n"
        f"    declared: {_listed(declared)}\n"
        f"    seen:     {_listed(seen)}\n"
        "Yours: cameras are connected and answering — these two lists are simply different "
        "strings. Either the config names cameras that are not on this bench, or these "
        "cameras are not the ones it names.\n"
        "Do now: nothing has moved.\n"
        "Then: read the two lists above side by side and change whichever one is wrong. "
        "Copying the seen identifiers into this rig's declaration is usually it; swapping "
        "in the other camera is the other."
    )


def _listed(items: Sequence[str]) -> str:
    return ", ".join(items) if items else "nothing"


def judge(qualities: Sequence[CameraQuality], threshold_px: float) -> Verdict:
    """Compare every camera's number against the declared threshold.

    The one place a number becomes a word. The solver produces numbers and never
    grades them; this grades them and never produces one. That separation is why
    a bad solve cannot save itself: the thing that decides is not the thing that
    wants the answer to be good.
    """
    ordered = tuple(qualities)
    within = all(q.reprojection_px <= threshold_px for q in ordered)
    return Verdict(
        qualities=ordered,
        threshold_px=threshold_px,
        word="WITHIN" if within else "OUTSIDE",
        within=within,
    )
