"""`newt record` — how the verb decides what it is recording from.

This file exists because `record` is the only verb that can get precedence
wrong. `teleop` and `rest` have one selector each; `record` has two — a
``--source`` and a ``--simulate`` — and a config file that could beat either of
them would be a rig quietly recording hardware when an operator asked for a fake
joint stream, or the reverse. Until newtrino-029 the branch that governs it had
no CLI test file at all and could only be read, never run.

What these encode (the WHY, not just the WHAT):

- **Either flag beats the file, and `--simulate` beats it too.** An operator
  asking for simulation out loud is not overridden by something written down
  weeks ago. This is the one precedence rule the other two verbs cannot violate,
  and it lives in the branch shape: simulate is checked *first*, not as a
  fallback after a config lookup.
- **The rhythm is reachable with no rig at all.** `record` is the one verb that
  can offer something when nothing resolves, so its refusal says so — that is a
  real alternative, not a consolation prize, and it belongs in the refusal.
- **The frontend holds no recording behavior.** These tests build a Session and
  ask what it was built from; nothing here drives an episode.
"""
from __future__ import annotations

import io
import json
import sys

import pytest

from newt._cli import _source_spec
from newt._cli.record import _build_session, _parse, _print_preflight


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _Recorder:
    """Stands in for the factory and remembers which spec reached it."""

    def __init__(self) -> None:
        self.spec = None

    def __call__(self, spec):
        self.spec = spec
        from newt.recording import BIMANUAL_DESCRIPTOR, SimulatedSource

        # Bimanual deliberately: the bundled `--simulate` source is single-arm
        # by default, so a session built from this factory is distinguishable
        # from one built by the flag without trusting the recorder's own flag.
        return SimulatedSource(BIMANUAL_DESCRIPTOR)


def _build(args, monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr("newt._cli.record._load_source", recorder)
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    session, _receipt = _build_session(_parse(["--task", "pick up the cup", *args]))
    return session, recorder, err.getvalue()


def _build_with_receipt(args, monkeypatch):
    """Same build, keeping the half `_build` drops: what the frontend was handed
    to say about where the source came from."""
    recorder = _Recorder()
    monkeypatch.setattr("newt._cli.record._load_source", recorder)
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    session, receipt = _build_session(_parse(["--task", "pick up the cup", *args]))
    return session, receipt, err.getvalue()


def _declare_config(monkeypatch, tmp_path, value):
    config = tmp_path / "nt.toml"
    config.write_text(f'[sources]\nrecord = "{value}"\n', encoding="utf-8")
    monkeypatch.setenv("NT_SITE_CONFIG", str(config))
    return config.resolve()


# --------------------------------------------------------------------------- #
# The rig's own declaration
# --------------------------------------------------------------------------- #

def test_a_configured_rig_records_from_the_factory_its_file_names(
    monkeypatch, tmp_path
):
    """`newt record --task ...` with no source flag, on a bench that said once
    which factory builds it. Asserted on which factory was reached — an exit
    code cannot tell a resolved default from a fallback to something simulated,
    and those two are the whole difference between real data and fake data."""
    config = _declare_config(monkeypatch, tmp_path, "declared_pkg:declared_factory")

    session, recorder, err = _build([], monkeypatch)
    assert session is not None
    assert recorder.spec == "declared_pkg:declared_factory"
    assert str(config) in err  # the substitution announces itself


def test_the_source_flag_beats_the_file(monkeypatch, tmp_path):
    _declare_config(monkeypatch, tmp_path, "declared_pkg:declared_factory")

    session, recorder, _ = _build(["--source", "typed_pkg:typed_factory"], monkeypatch)
    assert session is not None
    assert recorder.spec == "typed_pkg:typed_factory"


def test_a_short_name_resolves_in_the_verbs_own_namespace(monkeypatch, tmp_path):
    """`--source simulated_pair`, not `--source recording_source:simulated_pair`.
    The verb already said "record"; the operator should not have to say it again
    in the shape of a filename."""
    monkeypatch.setattr(
        _source_spec,
        "_declared_sources",
        lambda verb: [("simulated_pair", "recording_source:simulated_pair", "a-kit")]
        if verb == "record"
        else [],
    )

    _, recorder, _ = _build(["--source", "simulated_pair"], monkeypatch)
    assert recorder.spec == "recording_source:simulated_pair"


def test_a_kit_declared_source_lands_in_the_preflight_and_nowhere_before_it(
    monkeypatch, tmp_path, capsys
):
    """Ruling 1 for `record`: the preflight row, not a line above the block.

    `record`'s startup line is a contract block, and its first row already
    answers "what am I recording from" — so the receipt belongs *in* that row.
    A notice printed above the block would be a second answer to a question the
    block was already answering, which is the shape ruling 1 removed from all
    three verbs.

    Both halves are asserted: nothing on stderr while the session is built (the
    old provenance line's stream), and the fact present in the row afterward.
    Asserting only the silence would pass for a build that dropped the
    disclosure entirely, which is Rule 10's failure wearing this ruling's
    clothes."""
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "absent" / "nt.toml"))
    monkeypatch.setattr(
        _source_spec,
        "_declared_sources",
        lambda verb: [("live_pair", "recording_source:live_pair", "trossen-widowx")]
        if verb == "record"
        else [],
    )

    session, receipt, err = _build_with_receipt([], monkeypatch)
    assert session is not None
    assert err == ""
    assert receipt == "live_pair (from the trossen-widowx kit)"

    capsys.readouterr()
    _print_preflight(session, as_json=False, source_receipt=receipt)
    rows = [ln for ln in capsys.readouterr().out.splitlines() if "source " in ln]
    assert len(rows) == 1
    assert rows[0].endswith("— live_pair (from the trossen-widowx kit)")


def test_the_json_preflight_carries_the_receipt_beside_the_contract(
    monkeypatch, tmp_path, capsys
):
    """An agent driving `--json` is owed the same fact the terminal gets.

    It goes *beside* the contract, not inside it: ``contract`` is the library's
    own preflight report, and a frontend key smuggled into it would make an
    agent's picture of the Session disagree with the Session's. Where a source
    came from is the CLI's answer, so it is the CLI's key.

    This is also the stream the old stderr notice reached — `--json` puts a
    machine-read stream on stdout and the provenance line went to stderr, so an
    agent could always see it. Folding the receipt into a human block only would
    have quietly dropped the disclosure for every non-human caller."""
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "absent" / "nt.toml"))
    monkeypatch.setattr(
        _source_spec,
        "_declared_sources",
        lambda verb: [("live_pair", "recording_source:live_pair", "trossen-widowx")]
        if verb == "record"
        else [],
    )

    session, receipt, _ = _build_with_receipt([], monkeypatch)
    capsys.readouterr()
    _print_preflight(session, as_json=True, source_receipt=receipt)
    event = json.loads(capsys.readouterr().out.strip())

    assert event["event"] == "preflight"
    assert event["source"] == "live_pair (from the trossen-widowx kit)"
    assert "source" not in event["contract"]  # the library's report, untouched


def test_a_typed_source_leaves_the_preflight_row_as_it_was(monkeypatch, capsys):
    """No receipt, no suffix. The row still says what kind of source it is —
    that is the library's answer and it never depended on this."""
    session, receipt, _ = _build_with_receipt(
        ["--source", "typed_pkg:typed_factory"], monkeypatch
    )
    assert receipt is None
    capsys.readouterr()
    _print_preflight(session, as_json=False, source_receipt=receipt)
    rows = [ln for ln in capsys.readouterr().out.splitlines() if "source " in ln]
    assert len(rows) == 1
    assert "—" not in rows[0]


# --------------------------------------------------------------------------- #
# The precedence rule only this verb can violate
# --------------------------------------------------------------------------- #

def test_simulate_beats_a_configured_source(monkeypatch, tmp_path):
    """The rule that forced the branch to be written as "simulate first".

    Written the natural way — flag, else simulate, else config — a rig with
    ``[sources].record`` declared would connect to real hardware for an operator
    who typed ``--simulate``. That is metal moving in a room where someone
    believes nothing can move, and it is the reason this test exists rather than
    a note in a docstring.
    """
    _declare_config(monkeypatch, tmp_path, "declared_pkg:declared_factory")

    session, recorder, _ = _build(["--simulate"], monkeypatch)
    assert session is not None
    assert recorder.spec is None, "a config file overrode an explicit --simulate"
    # And what it built is the bundled single-arm simulated stream, not the
    # bimanual one the declared factory would have handed back — asserted on the
    # contract the session describes, not on the flag the test double recorded.
    assert "1 channel" in session.preflight()["source_kind"]


def test_the_two_selectors_still_refuse_each_other(monkeypatch, tmp_path):
    """``--source`` and ``--simulate`` together were always mutually exclusive,
    and a config file participating in the decision does not soften that: the
    operator named two different rigs in one breath and neither is a safe
    guess."""
    _declare_config(monkeypatch, tmp_path, "declared_pkg:declared_factory")

    with pytest.raises(ValueError) as exc:
        _build(["--source", "typed_pkg:typed_factory", "--simulate"], monkeypatch)
    assert "mutually exclusive" in str(exc.value)


# --------------------------------------------------------------------------- #
# The refusal this verb alone can improve on
# --------------------------------------------------------------------------- #

def test_an_undeclared_rig_is_refused_and_offered_the_rhythm(monkeypatch, tmp_path):
    """No flag, no file, no installed kit: refuse, and then say the one true
    thing `record` can say that the other verbs cannot — the rhythm is
    exercisable with no rig at all. Naming a real alternative is not softening
    the refusal; withholding it would be the verb knowing a way forward and not
    saying it."""
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "absent" / "nt.toml"))

    session, recorder, err = _build([], monkeypatch)
    assert session is None  # cmd_record turns this into exit 2
    assert recorder.spec is None
    assert "--source was not given" in err
    assert "newt record --simulate" in err


def test_more_than_one_declared_source_asks_rather_than_picking(monkeypatch, tmp_path):
    """This kit publishes two record factories, which is why the ambiguity rule
    is not theoretical here: bare `newt record` on an unconfigured rig has a real
    choice to make and no basis for making it, so it lists both and refuses.
    `rest`, publishing one, resolves in silence — same rule, both directions."""
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "absent" / "nt.toml"))
    monkeypatch.setattr(
        _source_spec,
        "_declared_sources",
        lambda verb: [
            ("live_pair", "recording_source:live_pair", "a-kit"),
            ("simulated_pair", "recording_source:simulated_pair", "a-kit"),
        ]
        if verb == "record"
        else [],
    )

    session, recorder, err = _build([], monkeypatch)
    assert session is None
    assert recorder.spec is None
    assert "live_pair" in err and "simulated_pair" in err
    assert "more than one source is declared" in err
