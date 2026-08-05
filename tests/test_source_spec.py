"""`resolve_spec` — the one answer to "which factory builds this rig?".

What these encode (the WHY, not just the WHAT):

- **The ladder is one thing, so it is tested in one place.** Explicit import path
  beats short name beats the rig's declared default beats the kit's sole
  declaration beats refusal. Four separate tests can each pass while the order
  is wrong; one test that walks every rung on a fixture where each rung names a
  *different* factory is the only shape a later reorder fails loudly against.
- **A short name resolves against declarations and nothing else.** No sys.path
  scanning, no globbing for ``*_source.py``, no trying ``<name>_source:<name>``.
  A convention learned from the filesystem is an embodiment fact learned, which
  is the fence newtrino-014 put up and this verb inherits. The anti-divination
  test below plants exactly the file a guessing implementation would find.
- **The refusal teaches from a declaration, never from a roster in `newt`.** A
  list of factory names inside the SDK is stale the day a kit ships a factory,
  and it is an embodiment fact in the wrong repo. Two fixtures declaring
  different names must produce refusals naming their own.
- **It never claims to know more than it does.** With no config file and no
  installed kit, `newt` genuinely cannot say what this rig offers — so it says
  where to declare and names nothing. Inventing a plausible roster to look
  helpful is Rule 10's failure wearing a friendly face, and the test for it
  asserts an *absence*.
- **Eleven causes, eleven strings.** Rule 12 made executable: two different
  problems must never print the same sentence, because an operator who cannot
  tell them apart cannot fix either. Asserted pairwise, and asserted again on
  first lines alone — a shared opening line collapses two diagnoses no matter
  what the rest of the paragraph says.
"""
from __future__ import annotations

import pytest

from newt._cli import _source_spec
from newt._cli._source_spec import SourceNotResolved, resolve_spec


REST_EXAMPLE = "mypkg.rig:make_rig"


# --------------------------------------------------------------------------- #
# Helpers — every declaration in this file is handed over out loud
# --------------------------------------------------------------------------- #

def _declare(monkeypatch, **by_verb):
    """Hand the resolver a kit registry: verb -> [(name, spec, distribution)].

    The autouse fixture in conftest defaults this to empty for every test in the
    suite, so what a test does not declare, no test sees — including whatever
    kits happen to be installed on the machine running pytest."""
    monkeypatch.setattr(
        _source_spec,
        "_declared_sources",
        lambda verb: sorted(by_verb.get(verb, [])),
    )


def _config(monkeypatch, tmp_path, body, *, name="nt.toml"):
    """Write a rig config and point the env seam at it. Returns the resolved path
    (resolved, because that is the form the refusals print)."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("NT_SITE_CONFIG", str(path))
    return path.resolve()


def _no_config(monkeypatch, tmp_path):
    path = tmp_path / "absent" / "nt.toml"
    monkeypatch.setenv("NT_SITE_CONFIG", str(path))
    return path.resolve()


# --------------------------------------------------------------------------- #
# The ladder, as one unit (card test 5)
# --------------------------------------------------------------------------- #

def test_the_precedence_ladder_walks_down_one_rung_at_a_time(monkeypatch, tmp_path):
    """Each rung names a different factory, so every step is falsifiable.

    Five rungs, not the four the card sketched: a kit that declares exactly one
    source for a verb resolves it, which is what makes a freshly installed rig
    with no config file at all work. That rung is part of the order and has to
    be pinned in the same place as the rest of it, or a later change can reorder
    it against the config default without a single test going red.
    """
    _declare(monkeypatch, rest=[("handy", "short_pkg:short_factory", "a-kit")])
    path = _config(
        monkeypatch,
        tmp_path,
        '[sources]\nrest = "config_pkg:config_factory"\n',
    )

    # Rung 1 — an explicit import path answers to nobody.
    assert (
        resolve_spec("rest", "explicit_pkg:explicit_factory", REST_EXAMPLE)
        == "explicit_pkg:explicit_factory"
    )

    # Rung 2 — a short name the operator typed, resolved in the verb's namespace.
    assert resolve_spec("rest", "handy", REST_EXAMPLE) == "short_pkg:short_factory"

    # Rung 3 — no flag: the rig's own declared default.
    assert resolve_spec("rest", None, REST_EXAMPLE) == "config_pkg:config_factory"

    # Rung 5 — the config declares nothing for this verb, and the kit offers one.
    _config(monkeypatch, tmp_path, '[sources]\nteleop = "other_pkg:other_factory"\n')
    assert resolve_spec("rest", None, REST_EXAMPLE) == "short_pkg:short_factory"

    # Rung 7 — nothing typed, nothing declared anywhere. Refuse.
    _declare(monkeypatch)
    with pytest.raises(SourceNotResolved):
        resolve_spec("rest", None, REST_EXAMPLE)

    # And the rungs above it still hold with everything below them gone, which
    # is what "the flag answers to nobody" means when the rig is bare.
    assert resolve_spec("rest", "pkg:factory", REST_EXAMPLE) == "pkg:factory"
    assert str(path)  # the fixture path was real; nothing here read a stale one


def test_a_short_name_in_the_config_resolves_through_the_same_registry(
    monkeypatch, tmp_path
):
    """Rung 4. The operator was invited to write a short name in the file, so a
    colon-less value there is the ordinary form, not a malformed spec — the same
    lookup the flag gets, from the other declaration surface."""
    _declare(monkeypatch, rest=[("live_pair", "rest_source:live_pair", "a-kit")])
    _config(monkeypatch, tmp_path, '[sources]\nrest = "live_pair"\n')
    assert resolve_spec("rest", None, REST_EXAMPLE) == "rest_source:live_pair"


# --------------------------------------------------------------------------- #
# The hard fence (card test 6)
# --------------------------------------------------------------------------- #

def test_a_short_name_never_reaches_the_filesystem(monkeypatch, tmp_path):
    """The anti-divination test — it fails the day someone adds a helpful fallback.

    Two things are true of ``simulated_pair`` here, and a scanning implementation
    would have resolved it on either: it is genuinely declared, for a *different*
    verb, and there is genuinely a module on ``sys.path`` at exactly the name the
    kit's ``<verb>_source.py`` convention would predict, exporting exactly that
    attribute. `newt` looks up one key in what was declared for the verb it was
    asked about, finds nothing, and refuses. Learning the convention is the
    failure; finding the file by luck is the symptom.
    """
    planted = tmp_path / "simulated_pair_source.py"
    planted.write_text(
        "def simulated_pair():\n"
        "    raise AssertionError('divination — this factory should never be found')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    _declare(
        monkeypatch,
        rest=[("live_pair", "rest_source:live_pair", "a-kit")],
        record=[("simulated_pair", "recording_source:simulated_pair", "a-kit")],
    )
    _no_config(monkeypatch, tmp_path)

    with pytest.raises(SourceNotResolved) as exc:
        resolve_spec("rest", "simulated_pair", REST_EXAMPLE)

    message = str(exc.value)
    assert "simulated_pair" in message  # it names what was asked for
    assert "live_pair" in message  # and what is actually there, for rest
    assert "simulated_pair_source" not in message  # never the file it didn't read


# --------------------------------------------------------------------------- #
# The refusal teaches, and only what it can prove (card tests 7 and 8)
# --------------------------------------------------------------------------- #

def test_the_refusal_enumerates_from_the_declaration_it_was_handed(
    monkeypatch, tmp_path
):
    """Two rigs, two declarations, two different rosters in the two refusals.

    A hardcoded list of factory names inside `newt` passes a single-fixture test
    and fails this one immediately — which is the only reason to run it twice."""
    _no_config(monkeypatch, tmp_path)

    _declare(monkeypatch, rest=[("live_pair", "rest_source:live_pair", "widowx-kit")])
    with pytest.raises(SourceNotResolved) as first:
        resolve_spec("rest", "typo", REST_EXAMPLE)

    _declare(
        monkeypatch,
        rest=[
            ("bench_left", "otherkit.rig:bench_left", "another-kit"),
            ("bench_right", "otherkit.rig:bench_right", "another-kit"),
        ],
    )
    with pytest.raises(SourceNotResolved) as second:
        resolve_spec("rest", "typo", REST_EXAMPLE)

    assert "live_pair" in str(first.value)
    assert "bench_left" in str(second.value) and "bench_right" in str(second.value)
    # Neither refusal knows anything about the other rig's kit.
    assert "bench_left" not in str(first.value)
    assert "live_pair" not in str(second.value)


def test_with_nothing_declared_it_says_so_instead_of_inventing_a_roster(
    monkeypatch, tmp_path
):
    """The honest-ignorance case, asserted as an absence.

    No config file and no installed kit means `newt` cannot list what this rig
    offers. The refusal teaches *where to declare*, says plainly it found
    nothing to read, and — the half that matters — names no factory it did not
    receive from a declaration. A helpful-looking roster here is exactly the
    invented-plausible-value failure the tenets bar.
    """
    path = _no_config(monkeypatch, tmp_path)
    _declare(monkeypatch)

    with pytest.raises(SourceNotResolved) as exc:
        resolve_spec("rest", None, REST_EXAMPLE)

    message = str(exc.value)
    assert str(path) in message  # the file it looked for, at its absolute path
    assert "no installed kit declares a source for rest" in message
    assert "[sources]" in message  # where to declare it
    # Nothing is offered, because nothing is known.
    assert "live_pair" not in message
    assert "offers" not in message


def test_a_typed_name_with_nothing_installed_blames_the_install_not_the_spelling(
    monkeypatch, tmp_path
):
    """Same ignorance, the other entry point. "no source named 'live_pair'" on a
    machine where no kit is installed would send the operator to fix a spelling
    that is probably perfect. The fix is an install, and the string says so."""
    _no_config(monkeypatch, tmp_path)
    _declare(monkeypatch)

    with pytest.raises(SourceNotResolved) as exc:
        resolve_spec("rest", "live_pair", REST_EXAMPLE)

    message = str(exc.value)
    assert "neither is anything else" in message
    assert "installed in this environment" in message


# --------------------------------------------------------------------------- #
# The selector appears only when there is something to select (card test 9)
# --------------------------------------------------------------------------- #

def test_one_declared_source_resolves_and_two_ask(monkeypatch, tmp_path):
    """Asserted on behaviour, not on wording: a lone candidate is not a choice,
    and making the operator confirm it is the flag tax the whole card exists to
    remove. Two candidates and no declared default is a genuine question, and
    the answer is a refusal that lists them — never a prompt, because a recovery
    verb must not block on a menu with an arm moving, and a prompt breaks every
    non-tty caller."""
    _no_config(monkeypatch, tmp_path)

    _declare(monkeypatch, record=[("live_pair", "recording_source:live_pair", "a-kit")])
    assert resolve_spec("record", None, "mypkg.rig:make_source") == (
        "recording_source:live_pair"
    )

    _declare(
        monkeypatch,
        record=[
            ("live_pair", "recording_source:live_pair", "a-kit"),
            ("simulated_pair", "recording_source:simulated_pair", "a-kit"),
        ],
    )
    with pytest.raises(SourceNotResolved) as exc:
        resolve_spec("record", None, "mypkg.rig:make_source")
    assert "live_pair" in str(exc.value) and "simulated_pair" in str(exc.value)


def test_one_name_declared_by_two_kits_refuses_and_names_both(monkeypatch, tmp_path):
    """Two installed kits, one short name, no way to tell which arm it drives.

    Picking either is the invented-value failure with a friendly face: it would
    move metal, silently, possibly not the metal the operator meant. So it
    refuses and names both declarers with the import path that settles it."""
    _no_config(monkeypatch, tmp_path)
    _declare(
        monkeypatch,
        rest=[
            ("live_pair", "widowx_rest:live_pair", "widowx-kit"),
            ("live_pair", "yam_rest:live_pair", "yam-kit"),
        ],
    )

    with pytest.raises(SourceNotResolved) as exc:
        resolve_spec("rest", "live_pair", REST_EXAMPLE)

    message = str(exc.value)
    assert "widowx-kit" in message and "yam-kit" in message
    assert "widowx_rest:live_pair" in message  # the path that ends the ambiguity


# --------------------------------------------------------------------------- #
# The substitution is always declared (card test 10)
# --------------------------------------------------------------------------- #

def test_a_resolved_default_announces_itself_and_a_typed_one_does_not(
    monkeypatch, tmp_path, capsys
):
    """Rule 10's declared-substitution clause, both directions.

    A default that arrives silently is the failure mode this card would
    otherwise introduce — the operator has to be able to see where the thing
    driving their arm came from. When they typed it themselves there is nothing
    to declare, and a line saying so would be noise at exactly the moment the
    screen needs to be readable.

    On stderr, not stdout: ``newt record --json`` puts a machine-read stream on
    stdout and a provenance line in it would be a parse error for an agent.
    """
    path = _config(monkeypatch, tmp_path, '[sources]\nrest = "rest_source:live_pair"\n')
    _declare(monkeypatch, rest=[("live_pair", "rest_source:live_pair", "a-kit")])

    resolve_spec("rest", None, REST_EXAMPLE)
    captured = capsys.readouterr()
    assert "rest_source:live_pair" in captured.err
    assert str(path) in captured.err
    assert captured.out == ""

    resolve_spec("rest", "other_pkg:other_factory", REST_EXAMPLE)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_a_kit_supplied_default_names_the_package_it_came_from(
    monkeypatch, tmp_path, capsys
):
    """Rung 5's provenance is a different line from the config one, because it
    answers a different "where did this come from": no file was read and nothing
    was typed, so the only honest answer names the installed distribution."""
    _no_config(monkeypatch, tmp_path)
    _declare(monkeypatch, rest=[("live_pair", "rest_source:live_pair", "widowx-kit")])

    resolve_spec("rest", None, REST_EXAMPLE)
    captured = capsys.readouterr()
    assert "widowx-kit" in captured.err
    assert "rest_source:live_pair" in captured.err
    assert captured.out == ""


# --------------------------------------------------------------------------- #
# The schema split, made testable (card test 11)
# --------------------------------------------------------------------------- #

def test_sources_is_read_in_isolation_from_the_kits_own_tables(monkeypatch, tmp_path):
    """The fence in one test: `newt` takes ``[sources]`` and acknowledges nothing else.

    The hardware tables here are garbage — an arm with no port, a camera with no
    serial, a stray table nothing has ever defined. All of it is the kit's, read
    by the kit's factories, and none of it is `newt`'s to validate. If this test
    ever fails, `newt` has started parsing an embodiment's description, which is
    the coupling the whole schema split exists to prevent.
    """
    _declare(monkeypatch)
    _config(
        monkeypatch,
        tmp_path,
        "[sources]\n"
        'rest = "rest_source:live_pair"\n'
        "\n"
        "[[robot_config.arms]]\n"
        'nonsense = "not a port"\n'
        "\n"
        "[[camera_config.cameras]]\n"
        "fps = -1\n"
        "\n"
        "[something_nobody_has_ever_defined]\n"
        "x = 1\n",
    )
    assert resolve_spec("rest", None, REST_EXAMPLE) == "rest_source:live_pair"


# --------------------------------------------------------------------------- #
# Eleven causes, eleven strings (card test 4)
# --------------------------------------------------------------------------- #

def _every_refusal(monkeypatch, tmp_path) -> dict[str, str]:
    """Provoke all eleven refusals for real and collect what each one printed.

    Built by driving ``resolve_spec`` rather than by reading the strings out of
    the module, so a cause that stops being reachable takes this test down with
    it."""
    messages: dict[str, str] = {}

    def _catch(label, verb, flag):
        with pytest.raises(SourceNotResolved) as exc:
            resolve_spec(verb, flag, REST_EXAMPLE)
        messages[label] = str(exc.value)

    kit = [("live_pair", "rest_source:live_pair", "a-kit")]

    # 1 — the file is there and the TOML is broken.
    _declare(monkeypatch, rest=kit)
    _config(monkeypatch, tmp_path, "[sources\nrest = \n", name="broken.toml")
    _catch("bad-toml", "rest", None)

    # 2 — the file is there and cannot be read at all. A directory at the config
    # path is the same class of problem as a permissions bit, and it does not
    # depend on who is running pytest.
    (tmp_path / "a-directory").mkdir()
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "a-directory"))
    _catch("unreadable", "rest", None)

    # 3 — the file parsed, and [sources] is not a flat map of strings.
    _config(monkeypatch, tmp_path, "[sources]\nrest = 3\n", name="wrong-shape.toml")
    _catch("wrong-shape", "rest", None)

    # 4 — one short name, two installed kits declaring it.
    _no_config(monkeypatch, tmp_path)
    _declare(
        monkeypatch,
        rest=[
            ("live_pair", "widowx_rest:live_pair", "widowx-kit"),
            ("live_pair", "yam_rest:live_pair", "yam-kit"),
        ],
    )
    _catch("two-kits-one-name", "rest", "live_pair")

    # 5 — a short name the operator typed, against a kit that declares others.
    _declare(monkeypatch, rest=kit)
    _catch("typed-name-unknown", "rest", "live_par")

    # 6 — a short name the *file* declared, against a kit that declares others.
    _config(monkeypatch, tmp_path, '[sources]\nrest = "live_par"\n', name="typo.toml")
    _catch("config-name-unknown", "rest", None)

    # 7 — a typed short name with no declarations anywhere.
    _declare(monkeypatch)
    _no_config(monkeypatch, tmp_path)
    _catch("typed-name-nothing-installed", "rest", "live_pair")

    # 8 — a file-declared short name with no declarations anywhere.
    _config(monkeypatch, tmp_path, '[sources]\nrest = "live_pair"\n', name="orphan.toml")
    _catch("config-name-nothing-installed", "rest", None)

    # 9 — several candidates and no declared default.
    _no_config(monkeypatch, tmp_path)
    _declare(
        monkeypatch,
        rest=[
            ("bench_left", "otherkit.rig:bench_left", "another-kit"),
            ("bench_right", "otherkit.rig:bench_right", "another-kit"),
        ],
    )
    _catch("several-candidates", "rest", None)

    # 10 — nothing given, no file, nothing installed.
    _declare(monkeypatch)
    _catch("no-file-at-all", "rest", None)

    # 11 — a file that declares other verbs but not this one.
    _config(
        monkeypatch,
        tmp_path,
        '[sources]\nteleop = "teleop_source:live_pair"\n',
        name="other-verbs.toml",
    )
    _catch("file-declares-other-verbs", "rest", None)

    return messages


def test_eleven_causes_produce_eleven_distinct_strings(monkeypatch, tmp_path):
    """Rule 12, executable. "Could any other cause produce this identical
    string?" answered by asking the code instead of the author.

    This is what stops a later refactor from collapsing "your file is broken"
    into "your file declares nothing" behind one template — a change that reads
    as tidying and lands as two problems wearing one face.
    """
    messages = _every_refusal(monkeypatch, tmp_path)
    assert len(messages) == 11, "a cause stopped being reachable"

    seen: dict[str, str] = {}
    for label, message in messages.items():
        assert message not in seen, f"{label} and {seen[message]} print the same refusal"
        seen[message] = label


def test_the_eleven_refusals_differ_on_their_first_line(monkeypatch, tmp_path):
    """The diagnosis is the first line — the rest is the fix.

    An operator at a bench reads one line before deciding what kind of problem
    they have. Two causes that differ only in their fourth line have already
    sent that reader to the wrong place, even though the full strings are
    technically distinct.
    """
    messages = _every_refusal(monkeypatch, tmp_path)

    first_lines: dict[str, str] = {}
    for label, message in messages.items():
        head = message.splitlines()[0]
        assert head.startswith("newt rest:"), f"{label} does not name the verb first"
        assert head not in first_lines, (
            f"{label} and {first_lines[head]} open with the same diagnosis"
        )
        first_lines[head] = label
