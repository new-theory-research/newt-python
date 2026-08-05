"""`newt rest` — the frontend contract, before anything is commanded.

What these encode (the WHY, not just the WHAT):

- **The verb never guesses a rig.** No ``--source`` is a refusal, not a fallback
  to something simulated. There is nothing to put away without hardware and the
  verb offers nothing in its place.
- **``--source`` means the same thing for every verb.** ``record``, ``teleop``,
  and ``rest`` load a spec through one function, so a developer who learns the
  contract once has learned it everywhere — and the three cannot drift.
- **Help is not a dry run.** ``--help`` on a verb that moves hardware must be
  reachable without the factory ever being constructed, because constructing it
  is what connects and energizes.
- **The help text must not read as calibration.** The word "home" and its
  relatives are barred from this verb's surface: a tired operator who reads
  "home" as set-zero is one keystroke from losing an arm's zero, and that
  near-miss is the reason this verb is spelled ``rest``.
"""
from __future__ import annotations

import io
import sys

import pytest

from newt._cli.rest import _parse, cmd_rest


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _capture(args, monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    rc = cmd_rest(args)
    return rc, out.getvalue(), err.getvalue()


class _Tripwire:
    """Records whether the source factory was ever reached."""

    def __init__(self) -> None:
        self.called = False

    def __call__(self, spec):
        self.called = True
        raise AssertionError(f"the factory ran for {spec!r} — it should not have")


# --------------------------------------------------------------------------- #
# --help / -h
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_usage_and_exits_zero(flag, monkeypatch):
    tripwire = _Tripwire()
    monkeypatch.setattr("newt._cli.rest.load_source", tripwire)
    rc, out, _ = _capture([flag], monkeypatch)
    assert rc == 0
    assert "Usage: newt rest" in out
    assert not tripwire.called


def test_help_documents_the_source_flag_and_the_exit_codes(monkeypatch):
    rc, out, _ = _capture(["--help"], monkeypatch)
    assert rc == 0
    assert "--source SPEC" in out
    assert "MODULE:FACTORY" in out
    # The exit-code table is the contract a script author reads. Each code must
    # be documented, because a caller that cannot tell "nothing was declared"
    # from "an arm went unanswered" cannot decide whether to walk up to the rig.
    for code in ("0", "1", "2", "3", "4", "130"):
        assert f"\n  {code}    " in out or f"\n  {code}  " in out


def test_help_never_offers_calibration_vocabulary(monkeypatch):
    """`home`, `rehome`, `zero` are barred from this verb's surface.

    Not a style rule. The 2026-08-03 bench session had to disambiguate "move the
    arm to a rest pose" from "homing in the calibration sense" repeatedly, and a
    factory-reset script was nearly run on a working arm because one word meant
    both. The verb is spelled `rest` for this reason, and its help text has to
    hold the same line — except where it says outright that it is not that.
    """
    rc, out, _ = _capture(["--help"], monkeypatch)
    assert rc == 0
    lowered = out.lower()
    for word in ("home", "rehome", "set-zero", "factory reset"):
        assert word not in lowered, f"{word!r} in `newt rest --help`"
    # The disclaimer is the sanctioned exception and must actually be there.
    assert "not a calibration command" in lowered
    assert "its own zero" in lowered


# --------------------------------------------------------------------------- #
# Argument parsing — every refusal names what it rejected
# --------------------------------------------------------------------------- #

def test_parse_accepts_the_source_spec():
    assert _parse(["--source", "mypkg.rig:make_rig"])["source"] == "mypkg.rig:make_rig"


def test_parse_rejects_an_unknown_option_by_name():
    with pytest.raises(ValueError) as exc:
        _parse(["--rate", "30"])
    assert "--rate" in str(exc.value)


def test_parse_rejects_a_source_flag_with_no_value():
    with pytest.raises(ValueError) as exc:
        _parse(["--source"])
    assert "--source" in str(exc.value)


def test_undeclared_rig_refuses_and_never_reaches_the_factory(monkeypatch, tmp_path):
    """No flag and no declaration is still a refusal — the verb never invents a
    factory. What changed in newtrino-029 is only *where else it looks first*;
    with both places empty the answer is the same one it always was, and the
    refusal now names both."""
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "nowhere" / "nt.toml"))
    tripwire = _Tripwire()
    monkeypatch.setattr("newt._cli.rest.load_source", tripwire)
    rc, _, err = _capture([], monkeypatch)
    assert rc == 1
    assert "--source was not given" in err
    assert str(tmp_path / "nowhere" / "nt.toml") in err
    assert "newt rest --source" in err
    assert not tripwire.called


def test_a_bad_option_refuses_and_never_reaches_the_factory(monkeypatch):
    tripwire = _Tripwire()
    monkeypatch.setattr("newt._cli.rest.load_source", tripwire)
    rc, _, err = _capture(["--pose", "sleep"], monkeypatch)
    assert rc == 1
    assert "--pose" in err
    assert not tripwire.called


# --------------------------------------------------------------------------- #
# Bring-up — the factory's own refusals are surfaced, not traced
# --------------------------------------------------------------------------- #

def test_a_factory_that_refuses_is_surfaced_as_its_own_message(monkeypatch):
    def _refuse(spec):
        raise RuntimeError("no arm at /dev/ttyUSB0")

    monkeypatch.setattr("newt._cli.rest.load_source", _refuse)
    rc, _, err = _capture(["--source", "mypkg.rig:make_rig"], monkeypatch)
    assert rc == 1
    assert "no arm at /dev/ttyUSB0" in err
    assert "Traceback" not in err


def test_ctrl_c_during_bring_up_exits_130_and_says_nothing_was_put_away(monkeypatch):
    def _interrupt(spec):
        raise KeyboardInterrupt()

    monkeypatch.setattr("newt._cli.rest.load_source", _interrupt)
    rc, _, err = _capture(["--source", "mypkg.rig:make_rig"], monkeypatch)
    assert rc == 130
    assert "nothing was put away" in err


# --------------------------------------------------------------------------- #
# The verb is on the dispatcher, spelled once
# --------------------------------------------------------------------------- #

def test_the_dispatcher_knows_exactly_one_spelling_of_this_verb():
    """One name, no aliases in the dispatcher's documented surface.

    A second spelling is a second thing to learn, a second thing to document,
    and a second thing that can drift. The card that specified this verb spends
    its alias budget deliberately and exactly once; nothing else gets to.
    """
    import pathlib
    import re

    source = (
        pathlib.Path(__file__).parent.parent / "src" / "newt" / "_cli" / "__init__.py"
    ).read_text(encoding="utf-8")
    verbs = re.findall(r'if cmd == "([a-z][a-z0-9_-]*)"', source)
    assert verbs.count("rest") == 1
    for barred in ("home", "rehome", "park", "stow"):
        assert barred not in verbs
