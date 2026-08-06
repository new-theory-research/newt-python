"""When a kit's drivers fall out, `newt` says the one thing that puts them back.

The receipt (2026-08-05, a bench): a fresh kit, a plain `uv sync`, and the arm
drivers silently absent. What came back named a Python module and nothing else —
no owner, no next move. The command that would have fixed it was *in* a message
further down and was not seen, because of how much text sat around it.

What these encode (the WHY, not just the WHAT):

- **`newt` may not know what a robot is; it may know what a kit promises.** The
  discriminator is the exception object, never a roster of driver names. A
  hardcoded `trossen_arm`/`pyrealsense2` list in the SDK is an embodiment fact
  in the wrong repo, stale the day a kit changes hardware — so the fixtures here
  are deliberately nonsense modules with no relationship to any real driver. If
  a test only passes for something spelled like a real driver, something is
  matching on a name and the fence is down.
- **Two sites, one cause, and the second is the one that fires.** A kit that
  guards its own driver imports does it *inside* the factory, so the failure
  arrives at construction, not at import. A fix at `import_factory` alone would
  have missed the bench receipt entirely — both sites are driven here.
- **"The factory isn't here" and "the factory's imports aren't here" are two
  problems** with opposite fixes, so they may never share a string (Rule 12).
  One is a spelling or a missing install of the kit; the other is a kit that is
  installed and whose drivers fell out.
- **A message a reader cannot find the fix in has failed, however complete.**
  So the structure is asserted, not only the content: one command, alone on its
  line, blank lines around it, directly under the diagnosis.
- **The kit's own words survive.** The kit already says something true and
  specific about what it needs; the SDK's job is to put the repair command where
  it can be seen and then get out of the way, not to paraphrase or swallow it.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

from newt._cli._source_spec import KIT_SETUP, KitDependencyMissing, load_source


REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fixture kits. The missing modules are nonsense on purpose — see the docstring.
# --------------------------------------------------------------------------- #

_EAGER_KIT = """\
import quillfeather_bindings  # absent everywhere, and named after nothing


def make_source():
    return object()
"""

_LAZY_KIT = """\
def make_source():
    # A kit that guards its own driver import does it here, in the factory —
    # which is why this arrives at construction and not at import.
    try:
        import thrummingbore_sdk  # absent everywhere, and named after nothing
    except ImportError as exc:
        raise ImportError(
            "Driving the pair needs the `thrummingbore` SDK, and it is not installed.\\n"
            "Yours: it lives in this kit's optional `hardware` extra.\\n"
            f"(underlying import error: {exc})"
        ) from exc
    return object()
"""

_WORKING_KIT = """\
def make_source():
    return object()
"""


@pytest.fixture
def kits(tmp_path, monkeypatch):
    """Three importable fixture kits: eager-import, lazy-import, and fine."""
    (tmp_path / "eager_fixture_kit.py").write_text(_EAGER_KIT)
    (tmp_path / "lazy_fixture_kit.py").write_text(_LAZY_KIT)
    (tmp_path / "working_fixture_kit.py").write_text(_WORKING_KIT)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("eager_fixture_kit", "lazy_fixture_kit", "working_fixture_kit"):
        sys.modules.pop(name, None)
    return tmp_path


def _lantern(spec: str) -> str:
    """The message a spec produces, or fail if it produced no lantern at all."""
    with pytest.raises(KitDependencyMissing) as caught:
        load_source(spec)
    return str(caught.value)


# --------------------------------------------------------------------------- #
# Card test 1 — the message renders, at both sites, for neither driver by name
# --------------------------------------------------------------------------- #

def test_a_dependency_missing_at_import_time_lights_the_lantern(kits):
    """The kit's module was found; something it imports was not."""
    message = _lantern("eager_fixture_kit:make_source")

    assert KIT_SETUP in message
    assert "quillfeather_bindings" in message, (
        "the missing module must be echoed from the exception — reporting what "
        "happened, which is the half of this that is allowed"
    )
    assert "eager_fixture_kit:make_source" in message


def test_a_dependency_missing_at_construction_time_lights_the_same_lantern(kits):
    """The site that actually fires for a kit that guards its own imports.

    This is Mattie's case, mechanically: the driver import lives inside the
    factory, so `import_factory` sails through and the failure lands on the
    call that would have connected. A fix at phase one alone reads as complete
    and covers nothing she hit.
    """
    message = _lantern("lazy_fixture_kit:make_source")

    assert KIT_SETUP in message
    assert "thrummingbore" in message


def test_the_kits_own_words_are_carried_whole(kits):
    """The SDK moves the fix into view; it does not paraphrase the diagnosis.

    The kit knows what it needs and says so specifically. Wrapping that inside a
    sentence of `newt`'s own is half of what buried the recovery command at the
    bench — so every line the kit wrote is still in the message, below the
    command rather than in front of it.
    """
    message = _lantern("lazy_fixture_kit:make_source")

    for line in ("Driving the pair needs", "optional `hardware` extra", "underlying import error"):
        assert line in message, f"the kit said {line!r} and newt dropped it:\n{message}"

    command_at = next(
        i for i, line in enumerate(message.splitlines()) if line.strip() == KIT_SETUP
    )
    kit_at = next(
        i for i, line in enumerate(message.splitlines()) if "Driving the pair" in line
    )
    assert command_at < kit_at, (
        "the kit's paragraph sits above the command again — that is the exact "
        f"burial this card was filed for:\n{message}"
    )


def test_a_working_source_is_untouched(kits):
    """The lantern is a failure path and only a failure path."""
    assert load_source("working_fixture_kit:make_source") is not None


# --------------------------------------------------------------------------- #
# Card test 2 — the two causes stay apart (Rule 12, executable)
# --------------------------------------------------------------------------- #

def test_a_missing_factory_module_is_a_different_problem_with_a_different_string(kits):
    """The discriminator, from the reader's side.

    A spec naming a module nobody has is a typo or an uninstalled package: the
    kit's setup script cannot fix it and offering it would send them in a circle.
    A kit that *is* installed with its drivers absent is exactly what that script
    repairs. Same site, opposite fixes.
    """
    with pytest.raises(RuntimeError) as missing:
        load_source("no_such_kit_at_all_xyz:make_source")
    absent = str(missing.value)
    assert not isinstance(missing.value, KitDependencyMissing)

    lantern = _lantern("eager_fixture_kit:make_source")

    assert absent != lantern
    assert absent.splitlines()[0] != lantern.splitlines()[0], (
        "a shared opening line collapses two diagnoses no matter what the rest "
        "of the paragraph says"
    )
    assert KIT_SETUP not in absent, (
        "the kit's install script cannot install a kit that was never named "
        "correctly — offering it here is advice that loops"
    )
    assert "no_such_kit_at_all_xyz" in absent


def test_a_missing_submodule_is_not_read_as_a_missing_dependency(kits):
    """`import mypkg.rig` with no `mypkg` raises naming *mypkg*, not the spec.

    The discriminator has to see that as the spec's own chain being absent, or
    every mistyped package name in the fleet gets told to run a setup script it
    does not have. This is the case a plain `exc.name != module_name` check gets
    backwards.
    """
    with pytest.raises(RuntimeError) as caught:
        load_source("no_such_kit_at_all_xyz.rig:make_source")

    assert not isinstance(caught.value, KitDependencyMissing)
    assert KIT_SETUP not in str(caught.value)


def test_a_factory_that_raises_for_any_other_reason_keeps_its_own_string(kits, tmp_path,
                                                                        monkeypatch):
    """A missing address is not a missing driver, and must not be told it is."""
    (tmp_path / "refusing_fixture_kit.py").write_text(
        "def make_source():\n"
        "    raise RuntimeError('nt.toml has no address for the arm this drives')\n"
    )
    sys.modules.pop("refusing_fixture_kit", None)

    with pytest.raises(RuntimeError) as caught:
        load_source("refusing_fixture_kit:make_source")

    assert not isinstance(caught.value, KitDependencyMissing)
    assert "no address" in str(caught.value)
    assert KIT_SETUP not in str(caught.value)


# --------------------------------------------------------------------------- #
# Card test 4 — the structure law (newtrino-035, ruling 2)
# --------------------------------------------------------------------------- #

def test_the_command_is_where_a_reader_will_find_it(kits):
    """One command, alone on its line, blank lines around it, near the top.

    The law exists because the message she needed already contained the command
    that would have fixed it. Content was never the failure; placement was. This
    is the assertion that stops a later edit from folding the fix back into a
    paragraph because it read better that way.
    """
    for spec in ("eager_fixture_kit:make_source", "lazy_fixture_kit:make_source"):
        message = _lantern(spec)
        lines = message.splitlines()
        hits = [i for i, line in enumerate(lines) if line.strip() == KIT_SETUP]

        assert len(hits) == 1, f"{spec}: {len(hits)} isolated commands, not one:\n{message}"
        index = hits[0]
        assert lines[index] == f"    {KIT_SETUP}", (
            f"{spec}: the command is not alone on its line:\n{message}"
        )
        assert index == 2, (
            f"{spec}: the command is at line {index}, not directly under the "
            f"diagnosis:\n{message}"
        )
        assert not lines[1].strip() and not lines[3].strip(), (
            f"{spec}: the command is not set off by blank lines:\n{message}"
        )


# --------------------------------------------------------------------------- #
# Card test 5 — the 029 trap: no `--source` to someone who typed none
# --------------------------------------------------------------------------- #

def test_no_refusal_opens_with_a_flag_the_operator_may_not_have_typed(kits):
    """`--source 'x:y': ...` read by someone holding a terminal where they typed
    no `--source` sends them to look at a flag that is not there — 029's sharpest
    Rule 12 trap, at a site 029 did not cover. A source can arrive from the flag,
    from `nt.toml`, or from a kit's lone declaration, and these two functions are
    handed a spec with no record of which. So the flag is named in none of them
    and the spec is named in all of them: true whichever way it arrived.
    """
    messages = [_lantern("eager_fixture_kit:make_source"),
                _lantern("lazy_fixture_kit:make_source")]
    with pytest.raises(RuntimeError) as missing:
        load_source("no_such_kit_at_all_xyz:make_source")
    with pytest.raises(RuntimeError) as no_attr:
        load_source("working_fixture_kit:no_such_factory")
    with pytest.raises(ValueError) as no_colon:
        load_source("not-a-valid-spec")
    messages += [str(missing.value), str(no_attr.value), str(no_colon.value)]

    for message in messages:
        assert "--source" not in message, (
            f"a refusal from a spec of unknown origin names the flag:\n{message}"
        )
        assert "Traceback" not in message


# --------------------------------------------------------------------------- #
# Card test 3 — the fence, as a meta-test (newtrino-029 DoD 8's shape)
# --------------------------------------------------------------------------- #

# Scoped to what this card touched, the way 029 scoped its own: a repo-wide
# assertion does not hold today (`_DEFAULT_MODEL_TAG = "so101"` in robot.py is a
# real runtime default, and `newt create` legitimately handles kit names). What
# is defensible and load-bearing is that the *source-resolution surface* — the
# module that decides what to say when a rig will not load — never learns a
# driver's name.
_FENCED = ("src/newt/_cli/_source_spec.py",)

_EMBODIMENT_WORDS = re.compile(
    r"trossen|realsense|pyrealsense|widowx|lerobot|so101|aloha|franka|"
    r"dynamixel|feetech|yam|piper",
    re.IGNORECASE,
)


def test_the_source_surface_never_learns_a_drivers_name():
    """The fence, and the reason it is a test rather than a promise.

    A roster of driver module names inside `newt` is an embodiment fact in the
    wrong repo: stale the day a kit ships different hardware, and invisible to
    the kit author who would have to come here to fix it. The discriminator this
    card added is the exception object, which is why it needs no roster — and a
    later "just special-case the common one" edit is exactly what this catches.
    """
    for relative in _FENCED:
        text = (REPO_ROOT / relative).read_text()
        hits = [
            f"{relative}:{number}: {line.strip()}"
            for number, line in enumerate(text.splitlines(), 1)
            if _EMBODIMENT_WORDS.search(line)
        ]
        assert not hits, "embodiment names reached the source surface:\n" + "\n".join(hits)


def test_the_fence_test_can_actually_fail(tmp_path):
    """A grep guard that cannot fail is decoration. This falsifies it."""
    assert _EMBODIMENT_WORDS.search("    if exc.name == 'trossen_arm':")
    assert not _EMBODIMENT_WORDS.search("    if _spec_module_missing(exc, module_name):")


# --------------------------------------------------------------------------- #
# The three verbs render it, and render the same one (card § In 4)
# --------------------------------------------------------------------------- #

def _teleop(spec, monkeypatch):
    from newt._cli.teleop import KillKey, cmd_teleop

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(KillKey, "arm", lambda self: True)
    monkeypatch.setattr(KillKey, "restore", lambda self: None)
    return cmd_teleop(["--source", spec]), err.getvalue()


def _rest(spec, monkeypatch):
    from newt._cli.rest import cmd_rest

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return cmd_rest(["--source", spec]), err.getvalue()


def _record(spec, monkeypatch, tmp_path):
    from newt._cli.record import cmd_record

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    return cmd_record(
        ["--source", spec, "--task", "pick up the cup", "--dest", str(tmp_path / "out")]
    ), err.getvalue()


@pytest.mark.parametrize("spec", ["eager_fixture_kit:make_source", "lazy_fixture_kit:make_source"])
def test_every_verb_prints_the_lantern_whole_and_exits_non_zero(spec, kits, monkeypatch,
                                                                tmp_path):
    """One implementation, three verbs. Three variants of this message is the
    "two verbs that disagree" defect `_source_spec` exists to prevent — so the
    assertion is that each verb prints the *same* text, not merely a similar one.
    """
    expected = _lantern(spec)

    for name, run in (
        ("teleop", lambda: _teleop(spec, monkeypatch)),
        ("rest", lambda: _rest(spec, monkeypatch)),
        ("record", lambda: _record(spec, monkeypatch, tmp_path)),
    ):
        rc, err = run()
        assert rc != 0, f"newt {name} exited 0 on a rig that never loaded"
        assert expected in err, (
            f"newt {name} rewrote or truncated the message:\n{err}"
        )
        assert "Traceback" not in err


def test_the_composed_record_path_does_not_call_it_a_missing_factory(kits, monkeypatch,
                                                                     tmp_path):
    """`newt record --teleop` has its own refusal for "the factory could not be
    found". A kit whose drivers fell out is the opposite finding at the same
    site — the factory was found — and inheriting that string tells the operator
    to check a spelling that is correct.
    """
    from newt._cli.record import cmd_record
    from newt._cli.teleop import KillKey

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    # The composed path arms the kill key before it loads anything — correctly,
    # that is its whole safety order — so a captured-stdin test never reaches
    # the factory without this.
    monkeypatch.setattr(KillKey, "arm", lambda self: True)
    monkeypatch.setattr(KillKey, "restore", lambda self: None)
    rc = cmd_record(
        [
            "--teleop", "--source", "eager_fixture_kit:make_source",
            "--task", "pick up the cup", "--dest", str(tmp_path / "o"),
        ]
    )
    stderr = err.getvalue()

    assert rc != 0
    assert KIT_SETUP in stderr
    assert "could not be found" not in stderr, (
        f"the dependency failure inherited the missing-factory refusal:\n{stderr}"
    )


# --------------------------------------------------------------------------- #
# The contract the message cites (open question 3's narrower half)
# --------------------------------------------------------------------------- #

def test_the_path_newt_names_is_the_path_newt_runs():
    """`newt` naming `./scripts/setup` is citing our own handoff contract — but a
    contract with two copies of its own string is a claim, not a contract.

    `newt create` already owns this path: it is what the scaffolder looks for in
    a template and runs on the developer's behalf, and its display form is built
    from the same two path segments. So the refusal's command and the command the
    scaffolder actually executes have to be the same characters, or a developer
    is told to run something adjacent to the thing that exists.
    """
    from newt._cli.create import _SETUP_DISPLAY, _SETUP_REL

    assert KIT_SETUP == _SETUP_DISPLAY, (
        f"the refusal says {KIT_SETUP!r} and the scaffolder runs {_SETUP_DISPLAY!r}"
    )
    assert _SETUP_REL == ("scripts", "setup")
