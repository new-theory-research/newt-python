"""`newt calibrate` — the frontend contract, before anything is commanded.

What these encode (the WHY, not just the WHAT):

- **The verb never guesses a rig.** No ``--source`` and no declaration is a
  refusal that teaches, not a fallback to something simulated. Calibration
  measures real hardware; there is nothing to offer in its place.
- **``--source`` means the same thing for every verb.** ``record``, ``teleop``,
  ``rest`` and ``calibrate`` load a spec through one function and resolve a
  default through one ladder, so a developer who learns the contract once has
  learned it everywhere — and the four cannot drift.
- **Help is not a dry run.** ``--help`` on a verb that moves an arm must be
  reachable without the factory ever being constructed, because constructing it
  is what connects and energizes.
- **``newt`` never learns what the printed target is.** The board is the kit's
  fact. The grep at the bottom of this file is the sibling of the kit's own —
  same scope in a different repo, so a worker writing the verb trips it here
  first rather than at somebody else's install.
"""
from __future__ import annotations

import io
import pathlib
import re
import sys

import pytest

from newt._cli.calibrate import _parse, cmd_calibrate

_SRC = pathlib.Path(__file__).parent.parent / "src" / "newt"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _capture(args, monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    rc = cmd_calibrate(args)
    return rc, out.getvalue(), err.getvalue()


class _Tripwire:
    """Records whether the source factory was ever reached."""

    def __init__(self) -> None:
        self.called = False

    def __call__(self, spec):
        self.called = True
        raise AssertionError(f"the factory ran for {spec!r} — it should not have")


class _Recorder:
    """Stands in for the factory and remembers which spec reached it."""

    def __init__(self) -> None:
        self.spec = None

    def __call__(self, spec):
        self.spec = spec
        raise RuntimeError("stopped at the seam — the test wanted the spec, not a rig")


# --------------------------------------------------------------------------- #
# --help / -h
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_usage_and_never_builds_the_rig(flag, monkeypatch):
    tripwire = _Tripwire()
    monkeypatch.setattr("newt._cli.calibrate.load_source", tripwire)
    rc, out, _ = _capture([flag], monkeypatch)
    assert rc == 0
    assert "Usage: newt calibrate" in out
    assert not tripwire.called


def test_help_documents_every_exit_code_it_can_return(monkeypatch):
    """The exit-code table is the contract a script author reads.

    A caller that cannot tell "this rig declares nothing" from "the cameras are
    not the ones in your config" from "solved, and the answer is suspect" cannot
    decide what to do next — and the third one in particular must never be
    mistaken for the green path.
    """
    rc, out, _ = _capture(["--help"], monkeypatch)
    assert rc == 0
    for code in ("0", "1", "2", "3", "4", "5", "6", "130"):
        assert f"\n  {code}    " in out or f"\n  {code}  " in out


def test_help_says_the_warning_survives_the_flag_that_skips_the_question(monkeypatch):
    _, out, _ = _capture(["--help"], monkeypatch)
    assert "--yes" in out
    assert "skips the question, never the warning" in out


def test_help_offers_the_skip_without_scolding_anyone_for_taking_it(monkeypatch):
    """Never-calibrated is an invitation, not a failing."""
    _, out, _ = _capture(["--help"], monkeypatch)
    assert "--skip" in out
    assert "Skipping is" in out
    assert "supported" in out


# --------------------------------------------------------------------------- #
# Argument parsing — every refusal names what it rejected
# --------------------------------------------------------------------------- #

def test_parse_accepts_the_source_spec():
    assert _parse(["--source", "mypkg.rig:make_calibration"])["source"] == "mypkg.rig:make_calibration"


def test_parse_rejects_an_unknown_option_by_name():
    with pytest.raises(ValueError) as exc:
        _parse(["--board", "a4"])
    assert "--board" in str(exc.value)


def test_parse_rejects_a_source_flag_with_no_value():
    with pytest.raises(ValueError) as exc:
        _parse(["--source"])
    assert "--source" in str(exc.value)


def test_skip_takes_the_operators_own_words_when_they_gave_any():
    opts = _parse(["--skip", "just proving pixels"])
    assert opts["skip"] is True
    assert opts["reason"] == "just proving pixels"


def test_skip_on_its_own_carries_no_invented_reason():
    opts = _parse(["--skip"])
    assert opts["skip"] is True
    assert opts["reason"] is None


def test_a_flag_after_skip_is_read_as_a_flag_not_as_a_reason():
    """`--skip --yes` means what it looks like."""
    opts = _parse(["--skip", "--yes"])
    assert opts["reason"] is None
    assert opts["yes"] is True


# --------------------------------------------------------------------------- #
# Resolution — 029's ladder, reused rather than re-derived
# --------------------------------------------------------------------------- #

def test_an_undeclared_rig_refuses_and_never_reaches_the_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "nowhere" / "nt.toml"))
    tripwire = _Tripwire()
    monkeypatch.setattr("newt._cli.calibrate.load_source", tripwire)
    rc, _, err = _capture([], monkeypatch)
    assert rc == 1
    assert "--source was not given" in err
    assert str(tmp_path / "nowhere" / "nt.toml") in err
    assert "newt calibrate --source" in err
    assert not tripwire.called


def test_a_configured_rig_calibrates_with_no_flag(monkeypatch, tmp_path):
    config = tmp_path / "nt.toml"
    config.write_text('[sources]\ncalibrate = "declared_pkg:cameras"\n')
    monkeypatch.setenv("NT_SITE_CONFIG", str(config))
    recorder = _Recorder()
    monkeypatch.setattr("newt._cli.calibrate.load_source", recorder)

    _, _, err = _capture([], monkeypatch)
    assert recorder.spec == "declared_pkg:cameras"
    assert str(config.resolve()) in err


def test_the_flag_beats_the_file(monkeypatch, tmp_path):
    config = tmp_path / "nt.toml"
    config.write_text('[sources]\ncalibrate = "declared_pkg:cameras"\n')
    monkeypatch.setenv("NT_SITE_CONFIG", str(config))
    recorder = _Recorder()
    monkeypatch.setattr("newt._cli.calibrate.load_source", recorder)

    _capture(["--source", "typed_pkg:typed_factory"], monkeypatch)
    assert recorder.spec == "typed_pkg:typed_factory"


def test_a_short_name_resolves_in_this_verbs_own_namespace(monkeypatch):
    """The selector is the exception, never the tax.

    One declared procedure resolves in silence and the operator types two words.
    The day this rig declares the arm's own joint calibration as a second
    procedure, the selector appears on its own and nothing in `newt` changes —
    which is the reason it is load-bearing here rather than tidy: "calibration"
    already means two things on this hardware.
    """
    from newt._cli import _source_spec

    monkeypatch.setattr(
        _source_spec,
        "_declared_sources",
        lambda verb: [("cameras", "kit_calibration:cameras", "a-kit")]
        if verb == "calibrate"
        else [],
    )
    recorder = _Recorder()
    monkeypatch.setattr("newt._cli.calibrate.load_source", recorder)

    _capture(["--source", "cameras"], monkeypatch)
    assert recorder.spec == "kit_calibration:cameras"


def test_a_factory_that_refuses_is_surfaced_as_its_own_message(monkeypatch):
    def _refuse(spec):
        raise RuntimeError("no camera runtime on this machine")

    monkeypatch.setattr("newt._cli.calibrate.load_source", _refuse)
    rc, _, err = _capture(["--source", "mypkg.rig:make_calibration"], monkeypatch)
    assert rc == 1
    assert "no camera runtime on this machine" in err
    assert "Traceback" not in err


def test_ctrl_c_during_bring_up_exits_130_and_says_nothing_moved(monkeypatch):
    def _interrupt(spec):
        raise KeyboardInterrupt()

    monkeypatch.setattr("newt._cli.calibrate.load_source", _interrupt)
    rc, _, err = _capture(["--source", "mypkg.rig:make_calibration"], monkeypatch)
    assert rc == 130
    assert "nothing moved and nothing was written" in err


# --------------------------------------------------------------------------- #
# Exit codes — each refusal reaches the caller as its own number
# --------------------------------------------------------------------------- #

class _OnePass:
    name = "sweep"

    def motion(self):
        return "the arm sweeps, about 20 seconds"

    def run(self, notice):
        notice("the target is in view")


class _Stub:
    """The smallest thing that passes the seam check, tuned per test.

    Declarations are read in one order — threshold, passes, cameras — so a stub
    that leaves an earlier one empty never reaches the refusal it was written for.
    Every override here starts from a rig that would otherwise calibrate.
    """

    def __init__(self, **overrides):
        self._overrides = overrides
        self.committed = False

    def describe(self):
        return "stub rig"

    def declared_cameras(self):
        return self._overrides.get("declared", ["cam-a"])

    def connected_cameras(self):
        return self._overrides.get("seen", ["cam-a"])

    def quality_threshold(self):
        return self._overrides.get("threshold", 2.0)

    def passes(self):
        return self._overrides.get("passes", [_OnePass()])

    def solve(self, notice):
        return []

    def commit(self, verdict):
        self.committed = True
        return "/rig/receipt.json"

    def skip(self, reason):
        self.seen_reason = reason
        return "/rig/skipped.json"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"threshold": None}, 2),
        ({"passes": []}, 2),
        ({"seen": []}, 3),
        ({"seen": ["other"]}, 3),
    ],
    ids=["no-threshold", "no-passes", "cameras-unplugged", "cameras-mismatched"],
)
def test_each_pre_motion_refusal_reaches_the_caller_as_its_own_code(
    overrides, expected, monkeypatch
):
    stub = _Stub(**overrides)
    monkeypatch.setattr("newt._cli.calibrate.load_source", lambda spec: stub)
    rc, _, err = _capture(["--source", "mypkg.rig:make_calibration", "--yes"], monkeypatch)
    assert rc == expected
    assert not stub.committed
    assert "Then:" in err


def test_a_source_that_is_not_a_procedure_exits_one_and_names_the_missing_members(monkeypatch):
    class _NotAProcedure:
        def describe(self):
            return "something else entirely"

    monkeypatch.setattr("newt._cli.calibrate.load_source", lambda spec: _NotAProcedure())
    rc, _, err = _capture(["--source", "mypkg.rig:make_calibration"], monkeypatch)
    assert rc == 1
    assert "quality_threshold()" in err
    assert "newt.calibration" in err


def test_skip_reaches_the_rig_and_never_runs_a_pass(monkeypatch):
    stub = _Stub(passes=[object()])  # a pass that would explode if it were read
    monkeypatch.setattr("newt._cli.calibrate.load_source", lambda spec: stub)
    rc, out, _ = _capture(
        ["--source", "mypkg.rig:make_calibration", "--skip", "pixels only today"], monkeypatch
    )
    assert rc == 0
    assert stub.seen_reason == "pixels only today"
    assert not stub.committed
    assert "NOT MEASURED" in out


# --------------------------------------------------------------------------- #
# The verb is on the dispatcher, spelled once
# --------------------------------------------------------------------------- #

def test_the_dispatcher_knows_exactly_one_spelling_of_this_verb():
    """One name, no aliases. A second spelling is a second thing that can drift.

    ``calib`` and ``cal`` are barred by name because they are the abbreviations a
    later hand reaches for; ``extrinsics`` because it is the noun this verb
    deliberately does not make anyone type.
    """
    source = (_SRC / "_cli" / "__init__.py").read_text(encoding="utf-8")
    verbs = re.findall(r'if cmd == "([a-z][a-z0-9_-]*)"', source)
    assert verbs.count("calibrate") == 1
    for barred in ("calib", "cal", "extrinsics", "recalibrate"):
        assert barred not in verbs


def test_the_verb_is_listed_in_the_top_level_usage():
    """A verb the dispatcher answers to and the usage never mentions is a verb
    nobody finds. Nothing derives one list from the other, so this does."""
    source = (_SRC / "_cli" / "__init__.py").read_text(encoding="utf-8")
    usage = source.split("def _usage()")[1]
    assert "calibrate" in usage


# --------------------------------------------------------------------------- #
# Test 2b's sibling — `newt` never learns what the printed target is
# --------------------------------------------------------------------------- #

# Scope, stated. The vocabulary a board fact is made of — not bare numbers, since
# a `45.0` in the SDK means nothing without a name beside it and grepping for it
# would fail on a frame rate. These are the words that only appear in code that
# knows what the printed target is.
#
# The kit's copy of this test (test_calibration_board.py, newtrino-048-a cycle 2)
# greps the *installed* newt package and adds its own preset names, read from its
# one table. This copy greps this repo's `src/` and cannot know those names — a
# kit's preset roster is exactly the fact that must not be here. Both are wanted:
# the kit's catches an install, this one catches a commit, and this is the one a
# worker writing the verb trips first.
_TARGET_VOCABULARY = [
    r"charuco",
    r"\baruco\b",
    r"\bsquare_mm\b",
    r"\bmarker_mm\b",
    r"\bsquares_x\b",
    r"\bsquares_y\b",
    r"DICT_\d",
    r"\bboard_preset\b",
    r"\bpagesize",
]


def test_newt_source_contains_no_board_fact():
    """The board is declared by the kit, consumed from one source, and printed on
    its own face. A preset name or a square size in this repo is the first step
    back toward three copies of the same constants and a silent 1.42× on every
    measurement — which is the defect, not a risk of it.
    """
    patterns = [re.compile(p, re.IGNORECASE) for p in _TARGET_VOCABULARY]
    hits = []
    for path in sorted(_SRC.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for pattern in patterns:
                if pattern.search(line):
                    hits.append(f"{path.relative_to(_SRC)}:{lineno}: {line.strip()}")

    assert not hits, (
        "newt has learned an embodiment fact — the board belongs to the kit:\n"
        + "\n".join(hits)
    )
