"""`newt setup` — the door onto whatever setup a kit declares, and the six ways
it declines to open.

What these encode (the WHY, not just the WHAT):

- **`newt` may not know what a robot is; it may know what a kit promises.** The
  verb reads a declaration and runs what it names. Every fixture kit here is
  named after nothing — if a test only passes for something spelled like a real
  rig, something is matching on a name and the fence is down. The grep at the
  bottom is that fence made falsifiable.
- **The absence of a declaration is a refusal, never a guess.** A kit that
  promised nothing gets no fallback script path, no filename convention, no
  "helpful" default. That is the silent substitution Rule 10 bans wearing a
  convenience hat, and it is asserted as an absence because an absence is what
  a later well-meaning edit would remove.
- **Six causes, six strings** (Rule 12). No kit here; a kit in the directory
  but not in this environment; kits installed and none declaring a setup; a
  declaration that will not load; two kits declaring one; the kit's own setup
  failing. Different owners, different fixes — and the pairwise-distinctness
  test is the one that fails when someone folds two of them together.
- **The kit's failure is the kit's, by name.** `newt` adds one line saying whose
  setup exited what, and never re-wraps a message that already scrolled past.
- **The argument list is forwarded unread.** The kit's own flags reach it
  verbatim; `newt` learns nothing about what they mean. The exception is
  ``--kit`` in first position, and the test for that is that it stops being
  special anywhere else.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

from newt._cli import _source_spec, setup as setup_mod
from newt._cli.setup import EXIT_REFUSED, SETUP_GROUP, cmd_setup


REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fixture kits — importable, and named after nothing on any bench
# --------------------------------------------------------------------------- #

_RECORDING_KIT = """\
seen = []


def setup(argv):
    seen.append(list(argv))
    print("kit setup ran")
    return None
"""

_FAILING_KIT = """\
def setup(argv):
    print("the kit said what went wrong, in its own words")
    return 3
"""

_RAISING_KIT = """\
def setup(argv):
    raise RuntimeError("the kit blew up mid-install")
"""

_EXITING_KIT = """\
import sys


def setup(argv):
    sys.exit(7)
"""

_CHATTY_KIT = """\
def setup(argv):
    return "done!"
"""

_DEPENDENCY_KIT = """\
import quillfeather_bindings  # absent everywhere, and named after nothing


def setup(argv):
    return 0
"""


@pytest.fixture
def kits(tmp_path, monkeypatch):
    """Importable fixture setups, one per behaviour under test."""
    written = {
        "recording_setup_kit": _RECORDING_KIT,
        "failing_setup_kit": _FAILING_KIT,
        "raising_setup_kit": _RAISING_KIT,
        "exiting_setup_kit": _EXITING_KIT,
        "chatty_setup_kit": _CHATTY_KIT,
        "dependency_setup_kit": _DEPENDENCY_KIT,
    }
    for name, body in written.items():
        (tmp_path / f"{name}.py").write_text(body)
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def _declare(monkeypatch, *entries):
    """Hand the verb a kit registry, out loud: (name, spec, distribution).

    conftest defaults this to empty for every test in the suite, so what a test
    does not declare, no test sees — including whatever is installed on the
    machine running pytest."""
    monkeypatch.setattr(setup_mod, "_declared_setups", lambda: sorted(entries))


def _sources(monkeypatch, **by_verb):
    """Say what the *source* registry looks like — the other half of the world.

    Both reads move together for the reason `test_source_spec` moves them
    together: a verb with entries is a verb something publishes for, and a test
    that set one without the other would describe an environment that cannot
    exist."""
    monkeypatch.setattr(
        _source_spec, "_declared_sources", lambda verb: sorted(by_verb.get(verb, []))
    )
    monkeypatch.setattr(
        _source_spec,
        "_declaring_verbs",
        lambda: sorted(verb for verb, entries in by_verb.items() if entries),
    )


def _run(args, monkeypatch):
    """Run the verb, capturing both streams. Returns (rc, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    rc = cmd_setup(args)
    return rc, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# The happy path — the door opens onto the kit's own code
# --------------------------------------------------------------------------- #

def test_a_lone_declaration_resolves_and_runs(kits, monkeypatch):
    """One declared setup is not a choice; it runs."""
    _declare(monkeypatch, ("bench", "recording_setup_kit:setup", "a-kit"))

    rc, out, err = _run([], monkeypatch)

    assert rc == 0, f"a setup that returned None must exit 0; stderr={err!r}"
    assert "kit setup ran" in out


def test_the_kit_that_ran_is_named_before_it_runs(kits, monkeypatch):
    """Nobody typed this name — so the substitution is declared (Rule 10).

    A verb that runs a kit's arbitrary code against a machine without saying
    whose code it is has substituted silently, which is the failure this lane
    keeps catching. The receipt names the declaration and the kit, and says
    nothing about a rig: `newt` has nothing true to say about a rig.
    """
    _declare(monkeypatch, ("bench", "recording_setup_kit:setup", "a-kit"))

    rc, out, err = _run([], monkeypatch)

    assert "bench" in err and "a-kit" in err, (
        f"the receipt must name the declaration and the kit: {err!r}"
    )
    assert err.startswith("[newt setup]"), (
        f"the receipt is newt's line and marks itself as one: {err!r}"
    )


def test_the_receipt_stays_off_stdout(kits, monkeypatch):
    """stdout belongs to the kit's setup, which may be piping a report."""
    _declare(monkeypatch, ("bench", "recording_setup_kit:setup", "a-kit"))

    rc, out, err = _run([], monkeypatch)

    assert "[newt setup]" not in out, f"newt's receipt must not reach stdout: {out!r}"


def test_arguments_reach_the_kit_verbatim(kits, monkeypatch):
    """The kit's own flags work through this verb, unread and unreordered.

    A setup *is* a command someone typed, and kits ship setups with flags their
    users and their agents depend on (`--json` is taught to agents by name). A
    zero-argument contract would make this verb strictly weaker than the thing
    it is a door onto.
    """
    _declare(monkeypatch, ("bench", "recording_setup_kit:setup", "a-kit"))

    rc, out, err = _run(["--json", "--force", "not-a-flag"], monkeypatch)

    import recording_setup_kit

    assert rc == 0
    assert recording_setup_kit.seen == [["--json", "--force", "not-a-flag"]], (
        f"argv must arrive whole and in order: {recording_setup_kit.seen!r}"
    )


def test_newt_does_not_parse_the_kits_flags(kits, monkeypatch):
    """An unknown flag is not newt's to reject — it belongs to the kit.

    The verb has no option table for what follows it, and this is the test that
    fails if someone adds one.
    """
    _declare(monkeypatch, ("bench", "recording_setup_kit:setup", "a-kit"))

    rc, out, err = _run(["--a-flag-newt-has-never-heard-of"], monkeypatch)

    assert rc == 0, f"newt must not refuse a flag it does not own; stderr={err!r}"


# --------------------------------------------------------------------------- #
# Exit contract — the child's code, honoured on every path
# --------------------------------------------------------------------------- #

def test_a_nonzero_return_passes_through(kits, monkeypatch):
    """A failed setup fails the command, with the kit's own number."""
    _declare(monkeypatch, ("bench", "failing_setup_kit:setup", "a-kit"))

    rc, out, err = _run([], monkeypatch)

    assert rc == 3, f"the kit's exit code must survive: rc={rc}"


def test_a_failed_setup_is_attributed_to_the_kit_and_not_restated(kits, monkeypatch):
    """The kit's diagnosis already scrolled past; newt adds one line of owner.

    Re-wrapping a message the operator has just read is how a good message gets
    buried — the receipt behind the whole refusal family.
    """
    _declare(monkeypatch, ("bench", "failing_setup_kit:setup", "a-kit"))

    rc, out, err = _run([], monkeypatch)

    assert "a-kit" in err, f"the failure must name whose it is: {err!r}"
    assert "the kit said what went wrong" not in err, (
        "newt must not restate what the kit already printed"
    )
    assert len([line for line in err.splitlines() if line.strip()]) <= 2, (
        f"one receipt line and one attribution line is the whole budget: {err!r}"
    )


def test_the_kits_words_land_before_newt_talks_about_them(kits, monkeypatch):
    """"What went wrong is above" has to still be true when this is piped.

    Found on a real terminal, which is the only place it could be found: the
    kit writes to stdout and `newt` writes to stderr, and piped into a log
    stdout is block-buffered while stderr is not — so the attribution overtakes
    the diagnosis it attributes and the operator reads them backwards.
    """
    _declare(monkeypatch, ("bench", "failing_setup_kit:setup", "a-kit"))
    flushes = []

    out = io.StringIO()
    monkeypatch.setattr(out, "flush", lambda: flushes.append(len(out.getvalue())), raising=False)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    rc = cmd_setup([])

    assert rc == 3
    assert flushes, "stdout must be flushed before newt claims the kit's words are above"
    assert flushes[-1] == len(out.getvalue()), (
        "the flush has to come after the kit finished writing, not before it started"
    )


def test_sys_exit_inside_a_setup_is_honoured(kits, monkeypatch):
    """A setup that calls sys.exit() means it."""
    _declare(monkeypatch, ("bench", "exiting_setup_kit:setup", "a-kit"))

    rc, out, err = _run([], monkeypatch)

    assert rc == 7, f"sys.exit(7) must not be swallowed into a generic failure: rc={rc}"


def test_a_setup_that_raises_is_the_kits_failure_by_name(kits, monkeypatch):
    """The traceback is the kit's; the attribution says so and stops."""
    _declare(monkeypatch, ("bench", "raising_setup_kit:setup", "a-kit"))

    rc, out, err = _run([], monkeypatch)

    assert rc == 1
    assert "a-kit" in err and "RuntimeError" in err, (
        f"the raise must be attributed and named: {err!r}"
    )


def test_a_setup_that_returns_something_else_is_refused_not_guessed(kits, monkeypatch):
    """Rule 10: `newt` will not decide what a string return meant.

    "It returned something truthy, call it success" is the identity-fill
    failure in its smallest possible form.
    """
    _declare(monkeypatch, ("bench", "chatty_setup_kit:setup", "a-kit"))

    rc, out, err = _run([], monkeypatch)

    assert rc == 1, f"a contract break must not exit 0: rc={rc}"
    assert "'done!'" in err or '"done!"' in err, (
        f"say what was actually returned, so the kit author can find it: {err!r}"
    )


# --------------------------------------------------------------------------- #
# The refusals — six worlds, six strings
# --------------------------------------------------------------------------- #

def _refusal(args, monkeypatch):
    rc, out, err = _run(args, monkeypatch)
    assert rc in (EXIT_REFUSED, 1), f"a refusal must not exit 0: rc={rc}"
    return err


def test_cause_1_no_kit_anywhere_sends_you_to_create(monkeypatch):
    """Nothing is installed and nothing is declared — early in the arc."""
    _sources(monkeypatch)
    _declare(monkeypatch)

    err = _refusal([], monkeypatch)

    assert "newt create" in err
    assert sys.executable in err, (
        "which interpreter is speaking is most of the answer when nothing is installed"
    )


def test_cause_2_standing_in_the_kit_hands_back_uv_run(monkeypatch, tmp_path):
    """The bench receipt, one verb over: the kit is right there.

    Being told nothing is declared "in this environment" while standing inside
    the kit's own checkout is true and useless — the fix is four characters from
    what was already typed, and ruling 2 says the fix is the headline.
    """
    project = (tmp_path / "kit-checkout").resolve()
    monkeypatch.setattr(
        _source_spec,
        "_cwd_kit_declaring",
        lambda verb, group=None: project if group == SETUP_GROUP else None,
    )
    _sources(monkeypatch)
    _declare(monkeypatch)

    err = _refusal([], monkeypatch)

    assert "uv run newt setup" in err
    assert str(project) in err
    assert "newt create" not in err, (
        "sending someone to make a new kit while standing in one is a goose chase "
        "with the goose in the room"
    )


def test_cause_3_kits_are_here_and_none_declares_a_setup(monkeypatch):
    """The gap is the kit's, and it is named by distribution."""
    _sources(monkeypatch, teleop=[("bench_pair", "akit.rig:teleop", "a-kit")])
    _declare(monkeypatch)

    err = _refusal([], monkeypatch)

    assert "a-kit" in err, f"name the kit whose gap this is: {err!r}"
    assert SETUP_GROUP in err, "say what the kit would have to publish"


def test_cause_3_names_nothing_to_type_rather_than_inventing_a_path(monkeypatch):
    """The whole point of the card, asserted as an absence.

    There is a real temptation to be helpful here and suggest a script path.
    That is the path-convention guess this verb exists to refuse — a refusal
    with no command is honest; a refusal with a made-up one is worse than
    useless, because the operator runs it and gets a second, unrelated failure.
    """
    _sources(monkeypatch, teleop=[("bench_pair", "akit.rig:teleop", "a-kit")])
    _declare(monkeypatch)

    err = _refusal([], monkeypatch)

    assert "scripts/setup" not in err
    assert "./" not in err, f"no path convention may appear in this refusal: {err!r}"


def test_cause_4_a_declaration_that_will_not_import(monkeypatch):
    """The code a declaration names is not here — the kit's packaging, or the
    wrong environment. The import check that separates those two is the
    headline, and it comes from the machinery the source verbs already use."""
    _sources(monkeypatch)
    _declare(monkeypatch, ("bench", "no_such_fixture_module:setup", "a-kit"))

    err = _refusal([], monkeypatch)

    assert "no_such_fixture_module" in err
    assert "a-kit" in err, "say which declaration led here"
    assert "import no_such_fixture_module" in err


def test_cause_4_a_declaration_whose_module_is_here_but_incomplete(kits, monkeypatch):
    """"The module is missing" and "something under it is missing" are two
    problems with opposite fixes, and the discriminator is the machinery's, not
    a roster of driver names."""
    _sources(monkeypatch)
    _declare(monkeypatch, ("bench", "dependency_setup_kit:setup", "a-kit"))

    err = _refusal([], monkeypatch)

    assert "quillfeather_bindings" in err, (
        "the kit's own missing import is echoed, not characterised"
    )


def test_cause_4_a_declaration_pointing_at_a_missing_callable(kits, monkeypatch):
    """The module imported and does not export what was named."""
    _sources(monkeypatch)
    _declare(monkeypatch, ("bench", "recording_setup_kit:no_such_callable", "a-kit"))

    err = _refusal([], monkeypatch)

    assert "no_such_callable" in err


def test_cause_5_two_kits_declaring_is_the_operators_choice(kits, monkeypatch):
    """Picking one silently drives *a* machine, possibly not this one."""
    _sources(monkeypatch)
    _declare(
        monkeypatch,
        ("bench", "recording_setup_kit:setup", "a-kit"),
        ("rig", "failing_setup_kit:setup", "another-kit"),
    )

    err = _refusal([], monkeypatch)

    assert "--kit" in err
    assert "a-kit" in err and "another-kit" in err, (
        "both kits are named, because the kit is the basis for the choice"
    )

    import recording_setup_kit

    assert recording_setup_kit.seen == [], "nothing may run while the choice is open"


def test_the_choice_can_be_made(kits, monkeypatch):
    """--kit resolves the ambiguity the refusal named, in first position."""
    _declare(
        monkeypatch,
        ("bench", "recording_setup_kit:setup", "a-kit"),
        ("rig", "failing_setup_kit:setup", "another-kit"),
    )

    rc, out, err = _run(["--kit", "bench", "--json"], monkeypatch)

    import recording_setup_kit

    assert rc == 0, f"stderr={err!r}"
    assert recording_setup_kit.seen == [["--json"]], (
        f"--kit and its value are newt's and are not forwarded: {recording_setup_kit.seen!r}"
    )


def test_kit_is_only_special_in_first_position(kits, monkeypatch):
    """A kit's own --kit flag still reaches it from anywhere else.

    The rule is stateless on purpose. A bare selector (`newt setup widowx`)
    would mean "an argument for the kit" with one kit installed and "a choice
    for newt" with two — one string meaning two things depending on what is
    installed, which is the ambiguity this verb exists to refuse.
    """
    _declare(monkeypatch, ("bench", "recording_setup_kit:setup", "a-kit"))

    rc, out, err = _run(["--json", "--kit", "theirs"], monkeypatch)

    import recording_setup_kit

    assert rc == 0, f"stderr={err!r}"
    assert recording_setup_kit.seen == [["--json", "--kit", "theirs"]], (
        f"--kit past position one belongs to the kit: {recording_setup_kit.seen!r}"
    )


def test_an_unknown_kit_name_says_what_is_declared(kits, monkeypatch):
    """A typo in --kit is one character from right, and they can see it here."""
    _sources(monkeypatch)
    _declare(monkeypatch, ("bench", "recording_setup_kit:setup", "a-kit"))

    err = _refusal(["--kit", "bnch"], monkeypatch)

    assert "'bnch'" in err
    assert "bench" in err


def test_kit_without_a_value_is_a_usage_error(monkeypatch):
    """And it is newt's, said as newt's — a different owner from every refusal
    above, which are all about the world rather than the keystrokes."""
    rc, out, err = _run(["--kit"], monkeypatch)

    assert rc == 1
    assert "--kit expects" in err


# --------------------------------------------------------------------------- #
# Rule 12, as a property of the family rather than of any one message
# --------------------------------------------------------------------------- #

def _every_refusal(monkeypatch, tmp_path, kits_dir):
    """One rendering of each cause, from a world that actually produces it."""
    rendered = {}

    _sources(monkeypatch)
    _declare(monkeypatch)
    monkeypatch.setattr(_source_spec, "_cwd_kit_declaring", lambda verb, group=None: None)
    rendered["no kit anywhere"] = _refusal([], monkeypatch)

    project = (tmp_path / "kit-checkout").resolve()
    monkeypatch.setattr(
        _source_spec,
        "_cwd_kit_declaring",
        lambda verb, group=None: project if group == SETUP_GROUP else None,
    )
    rendered["standing in the kit"] = _refusal([], monkeypatch)
    monkeypatch.setattr(_source_spec, "_cwd_kit_declaring", lambda verb, group=None: None)

    _sources(monkeypatch, teleop=[("bench_pair", "akit.rig:teleop", "a-kit")])
    rendered["kits here, no setup"] = _refusal([], monkeypatch)

    _declare(monkeypatch, ("bench", "no_such_fixture_module:setup", "a-kit"))
    rendered["will not import"] = _refusal([], monkeypatch)

    _declare(
        monkeypatch,
        ("bench", "recording_setup_kit:setup", "a-kit"),
        ("rig", "failing_setup_kit:setup", "another-kit"),
    )
    rendered["two kits"] = _refusal([], monkeypatch)

    _declare(monkeypatch, ("bench", "failing_setup_kit:setup", "a-kit"))
    rc, _, err = _run([], monkeypatch)
    assert rc == 3
    rendered["the kit's own failure"] = err

    return rendered


def test_no_two_causes_share_a_string(kits, monkeypatch, tmp_path):
    """Rule 12's actual bar: two different problems, never one message.

    Pairwise, because the failure mode is not "a message is missing" — it is
    someone folding two neighbouring causes into one string that reads fine
    against either and sends half its readers to the wrong fix.
    """
    rendered = _every_refusal(monkeypatch, tmp_path, kits)
    causes = sorted(rendered)

    for i, left in enumerate(causes):
        for right in causes[i + 1 :]:
            assert rendered[left] != rendered[right], (
                f"{left!r} and {right!r} produce the same message:\n{rendered[left]}"
            )


def test_every_refusal_isolates_its_command_or_names_none(kits, monkeypatch, tmp_path):
    """035's structure law: the fix is the headline, or it isn't a fix.

    One command, alone on its line, with blank lines around it, above the
    explanation rather than buried at the end of it. The one exception is the
    world with nothing to type, and it is an exception on purpose — inventing a
    command there is what this card refuses.
    """
    rendered = _every_refusal(monkeypatch, tmp_path, kits)
    no_command = {"kits here, no setup", "the kit's own failure"}

    for cause, message in rendered.items():
        if cause in no_command:
            assert not re.search(r"\n\n {4}\S", message), (
                f"{cause!r} must name no command to type:\n{message}"
            )
            continue
        assert re.search(r"\n\n {4}\S[^\n]*\n\n", message), (
            f"{cause!r} does not isolate exactly one command:\n{message}"
        )


def test_a_refusal_names_the_lack_once(kits, monkeypatch, tmp_path):
    """The reprimand pattern: restating what the operator does not have, twice.

    "declares no setup. It has no setup declared." is the same absence said
    again, and it is the tone this lane keeps catching.
    """
    rendered = _every_refusal(monkeypatch, tmp_path, kits)

    for cause, message in rendered.items():
        first_line = message.splitlines()[0]
        assert first_line.lower().count("declares no") + first_line.lower().count(
            "does not declare"
        ) <= 1, f"{cause!r} says the lack more than once:\n{first_line}"


# --------------------------------------------------------------------------- #
# The help guard, and the absence that is the whole design
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_runs_nothing(flag, kits, monkeypatch):
    """The uniform guard: no action performed when -h/--help is present."""
    _declare(monkeypatch, ("bench", "recording_setup_kit:setup", "a-kit"))

    rc, out, err = _run([flag], monkeypatch)

    import recording_setup_kit

    assert rc == 0
    assert recording_setup_kit.seen == [], f"{flag} must run nothing"
    assert "setup" in out.lower()


def test_help_tells_a_kit_author_how_to_declare_one(monkeypatch):
    """The contract has to be reachable from the terminal, not only the README."""
    rc, out, err = _run(["--help"], monkeypatch)

    assert SETUP_GROUP in out
    assert "MODULE:CALLABLE" in out


def test_help_admits_the_hole_it_has(monkeypatch):
    """`newt setup --help` cannot forward a kit's own --help, and says so.

    A capability hole named in the help is a known limit; the same hole
    unmentioned is a papercut the next person rediscovers at a bench.
    """
    rc, out, err = _run(["--help"], monkeypatch)

    assert "--help" in out and "kit documents" in out


# --------------------------------------------------------------------------- #
# The fence — asserted, because a promise is not a test
# --------------------------------------------------------------------------- #

_FENCED = ("src/newt/_cli/setup.py",)

_EMBODIMENT_WORDS = re.compile(
    r"trossen|realsense|pyrealsense|widowx|lerobot|so101|aloha|franka|"
    r"dynamixel|feetech|yam|piper|udev|apt-get",
    re.IGNORECASE,
)


def test_the_setup_verb_never_learns_what_a_setup_does():
    """The invariant this card exists to protect, made falsifiable.

    A driver name, an OS package manager, a udev rule — any of them inside this
    file is an embodiment fact in the wrong repo, stale the day a kit changes
    hardware and invisible to the kit author who would have to come here to fix
    it. The verb knows a kit can declare a setup and nothing else.
    """
    for relative in _FENCED:
        text = (REPO_ROOT / relative).read_text()
        hits = [
            f"{relative}:{number}: {line.strip()}"
            for number, line in enumerate(text.splitlines(), 1)
            if _EMBODIMENT_WORDS.search(line)
        ]
        assert not hits, "embodiment names reached the setup verb:\n" + "\n".join(hits)


def test_the_verb_holds_no_path_convention():
    """The second half of the fence, and the one a "helpful" edit would breach.

    A fallback script path for a kit that declared nothing is the silent
    substitution Rule 10 bans. It would also be invisible: the verb would work
    on our kit and quietly do the wrong thing on somebody else's.
    """
    text = (REPO_ROOT / "src/newt/_cli/setup.py").read_text()
    hits = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(text.splitlines(), 1)
        if re.search(r"scripts/|\./[a-z]", line)
    ]
    assert not hits, "a path convention reached the setup verb:\n" + "\n".join(hits)


def test_the_fence_tests_can_actually_fail():
    """A grep guard that cannot fail is decoration. This falsifies both."""
    assert _EMBODIMENT_WORDS.search('    print("run apt-get install ros-humble")')
    assert not _EMBODIMENT_WORDS.search("    entry = import_factory(spec)")
    assert re.search(r"scripts/|\./[a-z]", '    fallback = "./scripts/setup"')
    assert not re.search(r"scripts/|\./[a-z]", "    return sorted(found)")
