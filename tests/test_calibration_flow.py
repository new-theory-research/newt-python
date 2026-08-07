"""`newt calibrate` — the flow, and the three defects it was filed to invert.

What these encode (the WHY, not just the WHAT):

- **No naming convention gates any of it.** The system this verb replaces cannot
  calibrate unless an arm id contains the substring ``"right"`` — a frontend gate
  binding to a name the config's own schema calls free-form, failing loudly and
  correctly about the wrong thing. A rig whose only arm is ``arm0`` is legal and
  must calibrate. This is meant to be a boring test; it would have caught a defect
  that reached a bench.
- **The announcement is not bolted to the prompt.** Suppressing a question and
  suppressing a warning are different acts. ``--yes`` does the first and must
  never do the second, because "the announcement quietly stopped existing in the
  agent path" is exactly how a person ends up next to an arm that moves without
  saying so.
- **Two causes never share a string.** ``"No matching cameras connected"``
  covered both "nothing is plugged in" and "your config names different serials"
  with one sentence and no list. One of those operators should go and plug
  something in; the other should go and edit a file.
- **Nothing is written before the judgment is on screen.** A number with no scale
  beside it is not a result, and a save that happens regardless is the defect at
  the centre of this card.
"""
from __future__ import annotations

import io
import re
import sys

import pytest

from newt.calibration import (
    SKIPPED_WITHOUT_REASON,
    CalibrationError,
    CameraQuality,
    CamerasUnavailable,
    NothingDeclared,
    judge,
    require_calibration_procedure,
    run_calibration,
    run_skip,
)

# --------------------------------------------------------------------------- #
# A rig, faked at the seam and nowhere deeper
# --------------------------------------------------------------------------- #


class _Pass:
    def __init__(self, name: str, motion: str = "the arm sweeps nine poses, about 45 seconds", raises=None):
        self.name = name
        self._motion = motion
        self._raises = raises
        self.ran = False

    def motion(self) -> str:
        return self._motion

    def run(self, notice) -> None:
        self.ran = True
        notice("the board is in view")
        if self._raises is not None:
            raise self._raises


class _Procedure:
    """Every member of the seam, recording the order it was asked for things."""

    def __init__(
        self,
        *,
        cameras=("cam-a", "cam-b"),
        seen=None,
        threshold=2.0,
        passes=None,
        errors=(0.4, 0.9),
        solve_raises=None,
        commit_raises=None,
        skip_raises=None,
    ):
        self._cameras = list(cameras)
        self._seen = list(cameras) if seen is None else list(seen)
        self._threshold = threshold
        self._passes = list(passes) if passes is not None else [_Pass("sweep-near"), _Pass("sweep-far")]
        self._errors = list(errors)
        self._solve_raises = solve_raises
        self._commit_raises = commit_raises
        self._skip_raises = skip_raises
        #: Every seam call, in order. The ordering assertions read this.
        self.events: list = []

    def describe(self) -> str:
        return "waldo · camera extrinsics"

    def declared_cameras(self):
        return self._cameras

    def connected_cameras(self):
        return self._seen

    def quality_threshold(self):
        return self._threshold

    def passes(self):
        return self._passes

    def solve(self, notice):
        self.events.append("solve")
        notice("bundle iteration 4 of 40")
        if self._solve_raises is not None:
            raise self._solve_raises
        return [CameraQuality(c, e) for c, e in zip(self._cameras, self._errors)]

    def commit(self, verdict):
        self.events.append(("commit", verdict.word))
        if self._commit_raises is not None:
            raise self._commit_raises
        return "/rig/calibration/run-17.json"

    def skip(self, reason):
        self.events.append(("skip", reason))
        if self._skip_raises is not None:
            raise self._skip_raises
        return "/rig/calibration/skipped.json"


def _capture(fn, monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    def _never_asked(*a, **k):
        raise AssertionError("the operator was prompted when nothing should have asked")

    monkeypatch.setattr("builtins.input", _never_asked)
    rc = fn()
    return rc, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# Test 3 — no naming convention gates the flow (M9)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "names",
    [
        ("arm0", "arm1"),
        ("follower", "leader"),
        ("right_wrist", "right_scene"),
        ("banana", "kumquat"),
        ("0", "1"),
        ("cam", "cam-2"),
    ],
    ids=["arm0", "follower", "contains-right", "nonsense", "digits", "prefix-collision"],
)
def test_any_identifier_calibrates(names, monkeypatch):
    """The `"right"`-substring defect, inverted.

    A single-arm rig legally configured as ``arm0`` could not start the studied
    flow at all, and the error told the operator to add an arm they did not have.
    Nothing here reads a name: these six rigs differ only in their identifiers and
    must be indistinguishable in outcome.
    """
    proc = _Procedure(cameras=names)
    rc, out, err = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert rc == 0, err
    assert ("commit", "WITHIN") in proc.events
    for name in names:
        assert name in out


def test_the_rig_containing_right_and_the_rig_that_does_not_are_indistinguishable(monkeypatch):
    """Stronger than the parametrize: same numbers, different words, same output.

    If a convention ever crept back in it would most likely be a special case for
    one vocabulary, which a per-rig pass rate cannot see. This diffs two runs.
    """
    a = _Procedure(cameras=("right_wrist", "right_scene"))
    b = _Procedure(cameras=("arm0", "arm1"))
    rc_a, out_a, _ = _capture(lambda: run_calibration(a, assume_yes=True), monkeypatch)
    rc_b, out_b, _ = _capture(lambda: run_calibration(b, assume_yes=True), monkeypatch)
    assert rc_a == rc_b == 0
    assert a.events == b.events

    def _normalised(text: str, first: str, second: str) -> str:
        swapped = text.replace(first, "X").replace(second, "Y")
        return re.sub(r" +", " ", swapped)  # column padding follows name length

    assert _normalised(out_a, "right_wrist", "right_scene") == _normalised(out_b, "arm0", "arm1")


def test_pass_names_are_never_parsed_either(monkeypatch):
    proc = _Procedure(passes=[_Pass("left"), _Pass("right"), _Pass("🙂")])
    rc, out, _ = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert rc == 0
    assert all(p.ran for p in proc.passes())
    assert "🙂" in out


# --------------------------------------------------------------------------- #
# Test 4 — motion is announced on every path, including the bypassed one
# --------------------------------------------------------------------------- #

def test_motion_is_announced_even_when_confirmation_is_bypassed(monkeypatch):
    """`--yes` skips the question. It does not skip the warning, ever.

    The failure mode this guards is not "the flag was implemented wrong" — it is
    the announcement being written inside the confirmation block by somebody who
    read them as one feature, so it silently stops existing on the agent path and
    on nobody's screen.
    """
    proc = _Procedure(passes=[_Pass("sweep-near", "both arms sweep nine poses, about 45 seconds")])
    rc, out, _ = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert rc == 0
    assert "THE RIG IS ABOUT TO MOVE" in out
    assert "both arms sweep nine poses, about 45 seconds" in out
    assert "Clear the workspace" in out
    # And it says outright that the question was the only thing skipped.
    assert "--yes was given" in out
    assert "never is" in out


def test_the_announcement_lands_before_the_first_millimetre(monkeypatch):
    """Ordering, asserted directly. Printed-somewhere is not the requirement."""
    proc = _Procedure(passes=[_Pass("sweep-near", "the arm sweeps, about 45 seconds")])
    _, out, _ = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    announced = out.index("THE RIG IS ABOUT TO MOVE")
    moved = out.index("the board is in view")  # printed from inside pass.run()
    assert announced < moved


def test_the_announcement_uses_the_rigs_own_words_for_every_pass(monkeypatch):
    proc = _Procedure(
        passes=[
            _Pass("near", "left arm only, 30 cm of travel, about 20 seconds"),
            _Pass("far", "both arms across the table, about a minute"),
        ]
    )
    _, out, _ = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert "left arm only, 30 cm of travel, about 20 seconds" in out
    assert "both arms across the table, about a minute" in out


def test_a_pass_that_will_not_say_what_it_moves_is_refused_before_moving(monkeypatch):
    """A pass this verb cannot announce is a pass it will not run."""
    silent = _Pass("sweep-near", motion="")
    proc = _Procedure(passes=[_Pass("ok"), silent])
    with pytest.raises(NothingDeclared) as exc:
        _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert "sweep-near" in str(exc.value)
    assert not silent.ran
    assert proc.events == []


def test_a_non_interactive_run_without_yes_refuses_rather_than_assuming_agreement(monkeypatch):
    """Nobody typed anything, so nobody agreed to anything."""
    proc = _Procedure()
    monkeypatch.setattr(sys, "stdin", io.StringIO())  # not a tty
    rc, out, err = _capture(lambda: run_calibration(proc, assume_yes=False), monkeypatch)
    assert rc == 1
    assert "THE RIG IS ABOUT TO MOVE" in out  # announced anyway — every path
    assert "nobody typed anything" in err
    assert "--yes" in err
    assert not any(p.ran for p in proc.passes())
    assert proc.events == []


# --------------------------------------------------------------------------- #
# Test 8 — two causes, two strings, both enumerating expected against seen
# --------------------------------------------------------------------------- #

def _camera_refusal(seen, monkeypatch) -> str:
    proc = _Procedure(cameras=("SN-101", "SN-102"), seen=seen)
    with pytest.raises(CamerasUnavailable) as exc:
        _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert proc.events == [], "nothing may be solved or written from a refusal"
    return str(exc.value)


def test_unplugged_cameras_and_mismatched_serials_are_two_different_strings(monkeypatch):
    """The inherited one-string defect, named here so it cannot come back quietly.

    ``"No matching cameras connected"`` was emitted for both, with no
    expected-versus-seen list. The operator reading the first has a cable problem;
    the operator reading the second has a config problem. One string cannot send
    them to two different places.
    """
    unplugged = _camera_refusal([], monkeypatch)
    mismatched = _camera_refusal(["SN-999", "SN-998"], monkeypatch)
    assert unplugged != mismatched

    # Both enumerate, side by side, because a refusal that says "no match" and
    # shows neither list makes the operator go and find both by hand.
    for message in (unplugged, mismatched):
        assert "declared:" in message
        assert "seen:" in message
        assert "SN-101" in message and "SN-102" in message

    assert "sees no cameras at all" in unplugged
    assert "nothing" in unplugged.split("seen:")[1].splitlines()[0]
    assert "SN-999" in mismatched and "SN-998" in mismatched
    # And each sends the operator somewhere different.
    assert "plug the cameras in" in unplugged.lower()
    assert "config names cameras that are not on this bench" in mismatched


def test_one_matching_camera_is_enough_to_proceed(monkeypatch):
    """A partial match is a rig you can calibrate, not a refusal.

    The refusal is for "not one of them", because that is the state where nothing
    can be measured. Refusing a two-camera rig with one camera unplugged would be
    this verb inventing a completeness rule the rig never declared.
    """
    proc = _Procedure(cameras=("SN-101", "SN-102"), seen=["SN-102"], errors=(0.5, 0.5))
    rc, out, _ = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert rc == 0
    assert "1 declared camera(s) connected: SN-102" in out


# --------------------------------------------------------------------------- #
# The threshold is declared, never defaulted (open question 3's fence)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "threshold, fragment",
    [
        (None, "declares no quality threshold"),
        ("soon", "not a number of pixels"),
        (0, "no solve can ever be inside"),
        (-1.0, "no solve can ever be inside"),
    ],
    ids=["absent", "not-a-number", "zero", "negative"],
)
def test_an_undeclared_threshold_refuses_instead_of_defaulting(threshold, fragment, monkeypatch):
    """No number is picked here for hardware this SDK has never seen.

    The temptation is a `or 2.0` on the read, which would make every rig that
    forgot to declare one silently inherit somebody else's bench.
    """
    proc = _Procedure(threshold=threshold)
    with pytest.raises(NothingDeclared) as exc:
        _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert fragment in str(exc.value)
    assert proc.events == []
    assert not any(p.ran for p in proc.passes())


def test_the_verdict_word_comes_from_the_rigs_own_threshold():
    """Same numbers, two rigs, two verdicts. The number never grades itself."""
    numbers = [CameraQuality("cam-a", 1.5)]
    assert judge(numbers, 2.0).word == "WITHIN"
    assert judge(numbers, 1.0).word == "OUTSIDE"
    assert judge(numbers, 1.5).within is True  # at the threshold is inside it


# --------------------------------------------------------------------------- #
# Nothing reaches disk unjudged
# --------------------------------------------------------------------------- #

def test_the_verdict_is_on_screen_before_commit_is_called(monkeypatch):
    """The ordering the card is about, asserted against the seam rather than a log.

    ``commit`` cannot be reached without a ``Verdict``, and the render happens
    between the two. This asserts the operator-visible half: the number and the
    word are printed before the rig is allowed to write.
    """
    printed: list[str] = []

    class _Watched(_Procedure):
        def commit(self, verdict):
            printed.append(sys.stdout.getvalue())
            return super().commit(verdict)

    proc = _Watched()
    rc, out, _ = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert rc == 0
    seen_at_commit = printed[0]
    assert "verdict:" in seen_at_commit
    assert "WITHIN" in seen_at_commit
    assert "0.40 px" in seen_at_commit
    assert "threshold:   2 px" in seen_at_commit
    assert "/rig/calibration/run-17.json" in out  # the receipt path is printed


def test_a_solve_outside_the_threshold_is_written_labelled_and_never_exits_green(monkeypatch):
    """The middle path, taken deliberately: label, do not block — but never 0.

    A hard gate strands an operator at 11pm with a threshold nobody has tuned. A
    green exit code on a number nobody vouches for is the defect this card exists
    to invert. So: it saves, it says suspect, and the exit code says so too.
    """
    proc = _Procedure(errors=(0.4, 7.9))
    rc, out, _ = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert rc == 6
    assert ("commit", "OUTSIDE") in proc.events
    assert "OUTSIDE" in out
    assert "labelled suspect" in out
    assert "7.90 px" in out


def test_a_solve_that_returns_no_cameras_is_not_treated_as_a_good_answer(monkeypatch):
    """An empty answer is the state an unjudged save comes out of."""
    proc = _Procedure(errors=())
    rc, _, err = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert rc == 5
    assert "returned no cameras" in err
    assert proc.events == ["solve"]  # nothing written


def test_a_solve_that_raises_writes_nothing(monkeypatch):
    proc = _Procedure(solve_raises=RuntimeError("singular matrix"))
    rc, _, err = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert rc == 5
    assert "singular matrix" in err
    assert "Traceback" not in err
    assert not any(isinstance(e, tuple) and e[0] == "commit" for e in proc.events)


def test_a_failed_pass_stops_the_run_before_the_solve(monkeypatch):
    second = _Pass("sweep-far", raises=RuntimeError("the arm refused the pose"))
    proc = _Procedure(passes=[_Pass("sweep-near"), second, _Pass("sweep-high")])
    rc, _, err = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert rc == 4
    assert "the arm refused the pose" in err
    assert "sweep-far" in err
    assert proc.events == []  # never solved, never written
    assert not proc.passes()[2].ran


def test_a_commit_that_refuses_says_the_numbers_are_not_on_disk(monkeypatch):
    proc = _Procedure(commit_raises=PermissionError("read-only filesystem"))
    rc, _, err = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    assert rc == 5
    assert "read-only filesystem" in err
    assert "nothing on disk knows them" in err


# --------------------------------------------------------------------------- #
# The skip — a declared decision, never an absence and never an invention
# --------------------------------------------------------------------------- #

def test_skipping_writes_the_declaration_and_moves_nothing(monkeypatch):
    proc = _Procedure()
    rc, out, _ = _capture(lambda: run_skip(proc, "just proving pixels today"), monkeypatch)
    assert rc == 0
    assert proc.events == [("skip", "just proving pixels today")]
    assert not any(p.ran for p in proc.passes())
    assert "NOT MEASURED" in out
    assert "/rig/calibration/skipped.json" in out


def test_skipping_without_a_reason_records_the_absence_rather_than_inventing_one(monkeypatch):
    """Rule 10 on the smallest surface this card has.

    A plausible motive written here would be this process putting words in
    somebody's mouth and then shipping them to every episode recorded afterwards.
    """
    proc = _Procedure()
    _capture(lambda: run_skip(proc, None), monkeypatch)
    assert proc.events == [("skip", SKIPPED_WITHOUT_REASON)]
    assert "gave no reason" in SKIPPED_WITHOUT_REASON


@pytest.mark.parametrize("reason", ["", "   "])
def test_an_empty_reason_is_the_same_as_none(reason, monkeypatch):
    proc = _Procedure()
    _capture(lambda: run_skip(proc, reason), monkeypatch)
    assert proc.events == [("skip", SKIPPED_WITHOUT_REASON)]


def test_the_skip_does_not_reprimand(monkeypatch):
    """Never-calibrated is an invitation, not a failing."""
    proc = _Procedure()
    _, out, err = _capture(lambda: run_skip(proc, None), monkeypatch)
    assert "Nothing about this is a fault" in out
    assert err == ""
    for scold in ("error", "warning", "invalid", "failed", "must"):
        assert scold not in out.lower(), f"{scold!r} in the skip path's own output"


def test_a_skip_that_could_not_be_recorded_says_do_not_record_yet(monkeypatch):
    """The one skip failure worse than not skipping: believing you are marked."""
    proc = _Procedure(skip_raises=OSError("no such directory"))
    rc, _, err = _capture(lambda: run_skip(proc, None), monkeypatch)
    assert rc == 1
    assert "do NOT record yet" in err
    assert "no such directory" in err


# --------------------------------------------------------------------------- #
# Rule 12 across the whole surface — no two causes share a string
# --------------------------------------------------------------------------- #

def _refusal(build, monkeypatch) -> str:
    proc = build()
    try:
        rc, out, err = _capture(lambda: run_calibration(proc, assume_yes=True), monkeypatch)
    except (CalibrationError, NothingDeclared, CamerasUnavailable) as exc:
        return str(exc)
    assert rc != 0
    return err


def test_no_two_causes_share_a_string(monkeypatch):
    """M7, mechanically. Eleven distinct causes, eleven distinct sentences.

    The inherited surface had ``Finalisation failed`` standing for three separate
    ``ValueError``s and one string emitted verbatim from two sites. This asserts
    the property rather than trusting a reading of the file.
    """
    causes = {
        "no-threshold": lambda: _Procedure(threshold=None),
        "bad-threshold": lambda: _Procedure(threshold="soon"),
        "impossible-threshold": lambda: _Procedure(threshold=0),
        "no-cameras-declared": lambda: _Procedure(cameras=()),
        "cameras-unplugged": lambda: _Procedure(seen=[]),
        "cameras-mismatched": lambda: _Procedure(seen=["SN-999"]),
        "no-passes": lambda: _Procedure(passes=[]),
        "pass-unnamed": lambda: _Procedure(passes=[_Pass("")]),
        "pass-silent": lambda: _Procedure(passes=[_Pass("p", motion="")]),
        "pass-failed": lambda: _Procedure(passes=[_Pass("p", raises=RuntimeError("stuck"))]),
        "solve-failed": lambda: _Procedure(solve_raises=RuntimeError("did not converge")),
        "solve-empty": lambda: _Procedure(errors=()),
        "commit-failed": lambda: _Procedure(commit_raises=OSError("read-only")),
    }
    messages = {name: _refusal(build, monkeypatch) for name, build in causes.items()}

    seen: dict[str, str] = {}
    for name, message in messages.items():
        first_line = message.strip().splitlines()[0]
        assert first_line not in seen, (
            f"{name} and {seen[first_line]} share a refusal string: {first_line!r}"
        )
        seen[first_line] = name

    # And every one of them says what to do next, which is the half of Rule 12 a
    # unique-strings check on its own would let rot.
    for name, message in messages.items():
        assert "Then:" in message, f"{name} names a cause and no next step"
        assert "Do now:" in message or "nothing has" in message.lower(), name


def test_the_source_that_is_not_a_procedure_is_refused_by_the_member_it_lacks():
    class _Half:
        def describe(self):
            return "half a rig"

        def passes(self):
            return []

    with pytest.raises(CalibrationError) as exc:
        require_calibration_procedure(_Half())
    message = str(exc.value)
    assert "declared_cameras()" in message
    assert "quality_threshold()" in message
    assert "describe" not in message.split("Then:")[0]  # it has that one
    assert "nothing has moved" in message


def test_a_full_procedure_is_accepted():
    require_calibration_procedure(_Procedure())
