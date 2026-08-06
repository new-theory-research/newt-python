"""`newt teleop` — the frontend contract, before anything connects.

What these encode (the WHY, not just the WHAT):

- **A session with no kill key must never reach the hardware.** The verb moves
  an embodiment. Ctrl+H is the only way to stop it mid-motion, so a terminal
  that cannot carry Ctrl+H is a refusal, not a degraded mode — and the refusal
  has to land *before* the factory runs, because the factory is what connects.
  Two different terminals fail two different ways (no TTY at all; a TTY the
  listener could not arm on), so they get two different strings: a reader who
  sees one must know which one it is (Rule 12).
- **The verb never guesses a rig.** No ``--source`` is a refusal, not a
  fallback to something simulated. There is nothing to teleoperate without
  hardware and the verb offers nothing in its place.
- **``--source`` means the same thing for every verb.** ``record`` and
  ``teleop`` load a spec through one function, so a developer who learns the
  contract once has learned it everywhere — and the two can't drift.
"""
from __future__ import annotations

import io
import sys

import pytest

from newt._cli import _source_spec
from newt._cli.teleop import (
    KillKey,
    _parse,
    _stand_down_no_tty,
    _stand_down_unarmed,
    cmd_teleop,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _capture(args, monkeypatch, *, isatty=True):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: isatty, raising=False)
    rc = cmd_teleop(args)
    return rc, out.getvalue(), err.getvalue()


class _Tripwire:
    """Records whether the source factory was ever reached."""

    def __init__(self) -> None:
        self.called = False

    def __call__(self, spec):
        self.called = True
        raise AssertionError(f"the factory ran for {spec!r} — it should not have")


# --------------------------------------------------------------------------- #
# --help / -h: usage, exit 0, no action
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_usage_and_exits_zero(flag, monkeypatch):
    tripwire = _Tripwire()
    monkeypatch.setattr("newt._cli.teleop.load_source", tripwire)
    rc, out, _ = _capture([flag], monkeypatch)
    assert rc == 0
    assert "Usage: newt teleop" in out
    assert not tripwire.called


def test_help_documents_source_and_rate(monkeypatch):
    """`--help` has to answer the two questions the verb actually takes."""
    rc, out, _ = _capture(["--help"], monkeypatch)
    assert rc == 0
    assert "--source SPEC" in out
    assert "MODULE:FACTORY" in out
    assert "--rate HZ" in out
    assert "30" in out  # the default is stated, not implied


def test_help_wins_over_every_other_argument(monkeypatch):
    """`--help` is a question, never an instruction to run something."""
    tripwire = _Tripwire()
    monkeypatch.setattr("newt._cli.teleop.load_source", tripwire)
    rc, out, _ = _capture(["--source", "mypkg.rig:make_teleop", "--help"], monkeypatch)
    assert rc == 0
    assert "Usage: newt teleop" in out
    assert not tripwire.called


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def test_parse_defaults_the_rate_and_requires_nothing_else():
    assert _parse([]) == {"source": None, "rate": 30.0}


def test_parse_reads_source_and_rate():
    opts = _parse(["--source", "mypkg.rig:make_teleop", "--rate", "50"])
    assert opts == {"source": "mypkg.rig:make_teleop", "rate": 50.0}


def test_unknown_option_names_the_option(monkeypatch):
    rc, _, err = _capture(["--speed", "30"], monkeypatch)
    assert rc == 1
    assert "'--speed'" in err
    assert "newt teleop --help" in err


def test_option_missing_its_value_names_the_option(monkeypatch):
    rc, _, err = _capture(["--source"], monkeypatch)
    assert rc == 1
    assert "--source expects a value" in err


def test_unreadable_rate_names_what_was_passed(monkeypatch):
    """A rate that isn't a number is a usage error and says so in numbers' terms."""
    rc, _, err = _capture(["--source", "m:f", "--rate", "fast"], monkeypatch)
    assert rc == 1
    assert "'fast'" in err
    assert "Hz" in err


def test_undeclared_rig_refuses_rather_than_guessing_a_rig(monkeypatch, tmp_path):
    """No `--source` and no declaration is a refusal. There is no rig to fall
    back to and the verb does not invent one — the whole point of
    MODULE:FACTORY. newtrino-029 gave the verb a second place to look, not
    permission to guess when both are empty."""
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "nowhere" / "nt.toml"))
    rc, _, err = _capture([], monkeypatch)
    assert rc == 1
    assert "--source was not given" in err
    assert str(tmp_path / "nowhere" / "nt.toml") in err


def test_non_positive_rate_stands_down_at_two(monkeypatch):
    """Zero Hz is not a slow session, it is no session — and it is caught before
    anything connects, so it shares the stand-down code, not the usage code."""
    tripwire = _Tripwire()
    monkeypatch.setattr("newt._cli.teleop.load_source", tripwire)
    rc, _, err = _capture(["--source", "m:f", "--rate", "0"], monkeypatch)
    assert rc == 2
    assert "is not a rate" in err
    assert not tripwire.called


# --------------------------------------------------------------------------- #
# The two stand-downs
# --------------------------------------------------------------------------- #

def test_no_tty_stands_down_before_the_factory_runs(monkeypatch):
    """The factory is what connects. A terminal with no Ctrl+H must be refused
    on the near side of it, or the refusal arrives after the hardware is live."""
    tripwire = _Tripwire()
    monkeypatch.setattr("newt._cli.teleop.load_source", tripwire)
    rc, _, err = _capture(
        ["--source", "mypkg.rig:make_teleop"], monkeypatch, isatty=False
    )
    assert rc == 2
    assert not tripwire.called
    assert "stdin is not a TTY" in err
    assert "timeout --signal=INT" in err  # the named escape hatch, not a shrug


def test_unarmable_listener_stands_down_before_the_factory_runs(monkeypatch):
    """A TTY that the listener cannot claim is the same danger by a different
    cause, and it is refused at the same point — nothing connected, nothing moved."""
    tripwire = _Tripwire()
    monkeypatch.setattr("newt._cli.teleop.load_source", tripwire)
    monkeypatch.setattr(KillKey, "arm", lambda self: False)
    rc, _, err = _capture(["--source", "mypkg.rig:make_teleop"], monkeypatch)
    assert rc == 2
    assert not tripwire.called
    assert "could not arm" in err
    assert "nothing has moved" in err


def test_the_two_stand_downs_do_not_share_a_string(monkeypatch):
    """Same exit code, two causes. A reader who sees one has to know which one
    it is, which means the strings cannot be equal (Rule 12)."""
    def _render(fn):
        err = io.StringIO()
        monkeypatch.setattr(sys, "stderr", err)
        rc = fn()
        return rc, err.getvalue()

    rc_tty, text_tty = _render(_stand_down_no_tty)
    rc_arm, text_arm = _render(_stand_down_unarmed)

    assert rc_tty == rc_arm == 2
    assert text_tty != text_arm
    # And they differ where it matters: each names its own cause and its own fix.
    assert "not a TTY" in text_tty and "not a TTY" not in text_arm
    assert "could not arm" in text_arm and "could not arm" not in text_tty


# --------------------------------------------------------------------------- #
# The kill key
# --------------------------------------------------------------------------- #

def test_kill_key_refuses_to_arm_without_a_tty(monkeypatch):
    """Arming reports failure rather than raising, because the caller — not the
    listener — decides what an unarmable kill key means."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert KillKey().arm() is False


def test_kill_key_restore_is_idempotent(monkeypatch):
    """`restore()` runs from a finally that can be reached twice; a second call
    must be a no-op rather than replaying a stale terminal state."""
    key = KillKey()
    key.restore()
    key.restore()
    assert key.fired.is_set() is False


# --------------------------------------------------------------------------- #
# The dispatcher and the shared --source contract
# --------------------------------------------------------------------------- #

def test_dispatcher_routes_teleop(monkeypatch):
    """`newt teleop` reaches this verb — the whole user-visible change."""
    from newt._cli import _dispatch

    seen = {}

    def _fake(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr("newt._cli.teleop.cmd_teleop", _fake)
    assert _dispatch(["teleop", "--rate", "30"]) == 0
    assert seen["args"] == ["--rate", "30"]


def test_global_help_lists_teleop(monkeypatch):
    """A verb nobody can find is not a verb."""
    from newt._cli import _usage

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    _usage()
    assert "teleop" in out.getvalue()


def test_record_and_teleop_load_a_source_through_one_function():
    """The contract a developer learns writing a recording source is the one
    they use writing a teleop source. Two copies would drift; this asserts there
    is one."""
    from newt._cli._source_spec import load_source
    from newt._cli.record import _load_source
    from newt._cli.teleop import load_source as teleop_load_source

    assert _load_source is load_source
    assert teleop_load_source is load_source


def test_source_spec_failures_each_name_the_spec():
    """Four ways a spec fails, four messages, all of them naming the spec back."""
    from newt._cli._source_spec import load_source

    with pytest.raises(ValueError) as no_colon:
        load_source("not-a-valid-spec")
    assert "not-a-valid-spec" in str(no_colon.value)
    assert "MODULE:FACTORY" in str(no_colon.value)

    with pytest.raises(RuntimeError) as no_module:
        load_source("no_such_module_xyz:make_source")
    assert "no_such_module_xyz" in str(no_module.value)

    with pytest.raises(RuntimeError) as no_attr:
        load_source("newt._cli._source_spec:no_such_factory")
    assert "no_such_factory" in str(no_attr.value)

    messages = {
        str(no_colon.value),
        str(no_module.value),
        str(no_attr.value),
    }
    assert len(messages) == 3  # no two failures share a string


def test_source_factory_failure_reaches_the_operator_untraced(monkeypatch):
    """A factory that refuses — a missing address, a driver that isn't installed —
    is the rig talking to its own developer. The verb prints what it said and
    exits 1; it does not swallow it and it does not add a traceback."""
    def _refuse(spec):
        raise RuntimeError(
            "--source 'mypkg.rig:make_teleop': factory 'make_teleop' raised while "
            "constructing the source: nt.toml has no address for the arm this drives"
        )

    monkeypatch.setattr("newt._cli.teleop.load_source", _refuse)
    monkeypatch.setattr(KillKey, "arm", lambda self: True)
    monkeypatch.setattr(KillKey, "restore", lambda self: None)
    rc, _, err = _capture(["--source", "mypkg.rig:make_teleop"], monkeypatch)
    assert rc == 1
    assert "no address" in err
    assert "Traceback" not in err


def test_ctrl_c_during_bring_up_exits_130_not_0(monkeypatch):
    """Bring-up is seconds of blocking vendor motion — long enough for an
    operator to change their mind. That Ctrl+C is an abort before the session
    started, not the documented way to finish one, so it does not share the
    clean exit's 0. Putting away whatever came up is the factory's job; it is
    the only thing holding those handles."""
    def _interrupted(spec):
        raise KeyboardInterrupt

    monkeypatch.setattr("newt._cli.teleop.load_source", _interrupted)
    monkeypatch.setattr(KillKey, "arm", lambda self: True)
    monkeypatch.setattr(KillKey, "restore", lambda self: None)
    rc, _, err = _capture(["--source", "mypkg.rig:make_teleop"], monkeypatch)
    assert rc == 130
    assert "bring-up interrupted" in err
    assert "never started" in err


def test_the_frontend_hands_the_loop_the_rate_and_the_bare_kill_event(monkeypatch):
    """The skin owns the keyboard; the session owns the loop. What crosses
    between them is the parsed rate, a plain Event, and — since newtrino-035 —
    the receipt for a source nobody typed: nothing about terminals goes behind
    the seam, and nothing about ticking stays in front of it.

    The receipt crosses as a finished phrase, not as a name and a distribution
    for the loop to assemble. Where a source came from is a CLI fact; a library
    that formats it has started to know about kits."""
    import threading

    seen = {}

    def _run_session(source, *, rate_hz, kill, source_receipt=None):
        seen["source"] = source
        seen["rate_hz"] = rate_hz
        seen["kill"] = kill
        seen["source_receipt"] = source_receipt
        return 0

    monkeypatch.setattr("newt.teleop.run_session", _run_session)
    monkeypatch.setattr("newt._cli.teleop.load_source", lambda spec: "the-source")
    monkeypatch.setattr(KillKey, "arm", lambda self: True)
    monkeypatch.setattr(KillKey, "restore", lambda self: None)

    rc, _, _ = _capture(
        ["--source", "mypkg.rig:make_teleop", "--rate", "45"], monkeypatch
    )
    assert rc == 0
    assert seen["source"] == "the-source"
    assert seen["rate_hz"] == 45.0
    assert isinstance(seen["kill"], threading.Event)
    # They typed the spec, so there is nothing to hand them a receipt for.
    assert seen["source_receipt"] is None


# --------------------------------------------------------------------------- #
# The rig declares its own source, so the verb is two words (newtrino-029)
# --------------------------------------------------------------------------- #

class _Recorder:
    """Stands in for the factory and remembers which spec reached it."""

    def __init__(self) -> None:
        self.spec = None

    def __call__(self, spec):
        self.spec = spec
        return "the-source"


def _drive(args, monkeypatch):
    """Run the verb far enough to see which factory it chose, with nothing real
    on the far side of the kill key."""
    recorder = _Recorder()
    monkeypatch.setattr("newt._cli.teleop.load_source", recorder)
    monkeypatch.setattr("newt.teleop.run_session", lambda *a, **k: 0)
    monkeypatch.setattr(KillKey, "arm", lambda self: True)
    monkeypatch.setattr(KillKey, "restore", lambda self: None)
    rc, _, err = _capture(args, monkeypatch)
    return rc, err, recorder


def test_a_configured_rig_runs_the_factory_its_own_file_names(monkeypatch, tmp_path):
    """The whole card in one assertion: `newt teleop`, no flag, on a bench that
    said once which factory builds it. Asserted on *which* factory was reached,
    because an exit code cannot tell a resolved default from a lucky no-op."""
    config = tmp_path / "nt.toml"
    config.write_text('[sources]\nteleop = "declared_pkg:declared_factory"\n')
    monkeypatch.setenv("NT_SITE_CONFIG", str(config))

    rc, err, recorder = _drive([], monkeypatch)
    assert rc == 0
    assert recorder.spec == "declared_pkg:declared_factory"
    # And it says where that came from — a default nobody typed announces itself.
    assert str(config.resolve()) in err


def test_the_flag_beats_the_file(monkeypatch, tmp_path):
    """The escape hatch answers to nobody. An operator overriding at the command
    line is the case where a config file silently winning would be worst: they
    are looking straight at the string they typed."""
    config = tmp_path / "nt.toml"
    config.write_text('[sources]\nteleop = "declared_pkg:declared_factory"\n')
    monkeypatch.setenv("NT_SITE_CONFIG", str(config))

    rc, _, recorder = _drive(["--source", "typed_pkg:typed_factory"], monkeypatch)
    assert rc == 0
    assert recorder.spec == "typed_pkg:typed_factory"


def test_a_kit_declared_source_runs_with_no_message_about_sources(
    monkeypatch, tmp_path
):
    """A rig with one possible answer gets no error, no warning, and no notice.

    This is the whole of ruling 1 (newtrino-035) in one run: a fresh kit, no
    config file, a bare `newt teleop`. Before, that printed a line explaining
    which source it had picked *and why it had proceeded without being told to*
    — an apology for not erroring, arriving before anything ran. The acceptance
    question was Mattie's: would someone walking up read this as an error, a
    warning, or a receipt? A separate line ahead of the run only ever reads as
    one of the first two.

    So stderr is empty and stdout carries nothing yet — and the fact is not
    dropped, it is handed to the loop to print inside the line it was already
    going to print. Rule 10's declared-substitution clause is satisfied by that
    hand-off, which is why the receipt is asserted here and not merely its
    absence."""
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "absent" / "nt.toml"))
    monkeypatch.setattr(
        _source_spec,
        "_declared_sources",
        lambda verb: [("live_pair", "widowx_rig:live_pair", "trossen-widowx")]
        if verb == "teleop"
        else [],
    )

    seen = {}

    def _run_session(source, *, rate_hz, kill, source_receipt=None):
        seen["source_receipt"] = source_receipt
        return 0

    recorder = _Recorder()
    monkeypatch.setattr("newt._cli.teleop.load_source", recorder)
    monkeypatch.setattr("newt.teleop.run_session", _run_session)
    monkeypatch.setattr(KillKey, "arm", lambda self: True)
    monkeypatch.setattr(KillKey, "restore", lambda self: None)

    rc, out, err = _capture([], monkeypatch)
    assert rc == 0
    assert recorder.spec == "widowx_rig:live_pair"
    assert err == ""
    assert out == ""
    assert seen["source_receipt"] == "live_pair (from the trossen-widowx kit)"
