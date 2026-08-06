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
- **Nineteen causes, nineteen strings.** Rule 12 made executable: two different
  problems must never print the same sentence, because an operator who cannot
  tell them apart cannot fix either. Asserted pairwise, and asserted again on
  first lines alone — a shared opening line collapses two diagnoses no matter
  what the rest of the paragraph says.
- **The namespace a source is declared under is not the command anyone types.**
  For `rest`, `record` and `teleop` they are the same word, which is why one
  argument doing both jobs went unnoticed; `newt record --teleop` resolves under
  `demonstration` and there is no such command. So the nineteen are provoked
  twice, in one tmp dir so nothing but the branding differs, and both sets go
  into the pairwise corpus. A refusal has to name the command in every line the
  reader retypes and the verb in every line about where a declaration lives —
  and one message legitimately does both.
"""
from __future__ import annotations

import sys

import pytest

from newt._cli import _source_spec
from newt._cli._source_spec import SourceNotResolved, resolve_spec
# Bound at import, so it survives conftest's autouse patch of the module
# attribute: this is the only handle in the suite on the *real* working-directory
# reader, and the four tests at the bottom of this file are the only ones that
# want it.
from newt._cli._source_spec import _cwd_kit_declaring as _the_real_cwd_reader
from newt._cli.record import _COMPOSED_COMMAND, _COMPOSED_EXAMPLE, _COMPOSED_VERB


REST_EXAMPLE = "mypkg.rig:make_rig"


# --------------------------------------------------------------------------- #
# Helpers — every declaration in this file is handed over out loud
# --------------------------------------------------------------------------- #

def _declare(monkeypatch, **by_verb):
    """Hand the resolver a kit registry: verb -> [(name, spec, distribution)].

    The autouse fixture in conftest defaults this to empty for every test in the
    suite, so what a test does not declare, no test sees — including whatever
    kits happen to be installed on the machine running pytest.

    Both reads move together, because they are two questions about one registry:
    a verb handed entries here is a verb an installed kit publishes for, so it
    shows up in ``_declaring_verbs`` too. A test that declared for ``teleop``
    and left the verb list empty would be describing an environment that cannot
    exist, and the refusals it provoked would be fiction."""
    monkeypatch.setattr(
        _source_spec,
        "_declared_sources",
        lambda verb: sorted(by_verb.get(verb, [])),
    )
    monkeypatch.setattr(
        _source_spec,
        "_declaring_verbs",
        lambda: sorted(verb for verb, entries in by_verb.items() if entries),
    )


def _stand_in_a_kit(monkeypatch, tmp_path, *verbs):
    """Put the caller inside a kit checkout that publishes ``verbs``.

    The seam, not the filesystem: what the working directory holds is real state
    on a real bench, and the tests that drive the reader itself are the four at
    the bottom of this file. Everything else says which world it is in.

    Called with no verbs, it says the other thing out loud — *you are not
    standing in a kit that publishes anything* — which is why ``_every_refusal``
    opens with it. `monkeypatch` outlives a single call to that helper, so a
    corpus that only ever set this seam would carry the last world of one run
    into the first cause of the next."""
    project = (tmp_path / "kit-checkout").resolve()
    monkeypatch.setattr(
        _source_spec,
        "_cwd_kit_declaring",
        lambda verb: project if verb in verbs else None,
    )
    return project


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
        resolve_spec("rest", "explicit_pkg:explicit_factory", REST_EXAMPLE).spec
        == "explicit_pkg:explicit_factory"
    )

    # Rung 2 — a short name the operator typed, resolved in the verb's namespace.
    assert resolve_spec("rest", "handy", REST_EXAMPLE).spec == "short_pkg:short_factory"

    # Rung 3 — no flag: the rig's own declared default.
    assert resolve_spec("rest", None, REST_EXAMPLE).spec == "config_pkg:config_factory"

    # Rung 5 — the config declares nothing for this verb, and the kit offers one.
    _config(monkeypatch, tmp_path, '[sources]\nteleop = "other_pkg:other_factory"\n')
    assert resolve_spec("rest", None, REST_EXAMPLE).spec == "short_pkg:short_factory"

    # Rung 7 — nothing typed, nothing declared anywhere. Refuse.
    _declare(monkeypatch)
    with pytest.raises(SourceNotResolved):
        resolve_spec("rest", None, REST_EXAMPLE)

    # And the rungs above it still hold with everything below them gone, which
    # is what "the flag answers to nobody" means when the rig is bare.
    assert resolve_spec("rest", "pkg:factory", REST_EXAMPLE).spec == "pkg:factory"
    assert str(path)  # the fixture path was real; nothing here read a stale one


def test_a_short_name_in_the_config_resolves_through_the_same_registry(
    monkeypatch, tmp_path
):
    """Rung 4. The operator was invited to write a short name in the file, so a
    colon-less value there is the ordinary form, not a malformed spec — the same
    lookup the flag gets, from the other declaration surface."""
    _declare(monkeypatch, rest=[("live_pair", "rest_source:live_pair", "a-kit")])
    _config(monkeypatch, tmp_path, '[sources]\nrest = "live_pair"\n')
    assert resolve_spec("rest", None, REST_EXAMPLE).spec == "rest_source:live_pair"


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
    # Updated with the newtrino-035 split. The old pin — "no installed kit
    # declares a source for rest" — was the sentence that fired in both empty
    # registries; the fact it was guarding (that the kit registry was read and
    # came back with nothing) is now stated by the string that only this world
    # can produce.
    assert "No installed kit publishes any source to newt in this environment" in message
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
    # Both pins updated with the newtrino-035 split, and for one reason: the
    # blame this test is about is now carried by a string that *only* the
    # nothing-is-installed world can print. "neither is anything else" said the
    # same thing about a kit that publishes for three other verbs, which is the
    # sentence that sent Mattie's bench looking for a virtualenv that did not
    # exist. The install is still named as the fix, not the spelling.
    assert "No installed kit publishes any source to newt in this environment" in message
    assert "Install your rig's kit into that environment" in message
    assert "live_pair" in message  # the name they typed, not corrected at them


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
    assert resolve_spec("record", None, "mypkg.rig:make_source").spec == (
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


def test_a_kit_supplied_default_says_nothing_and_hands_back_a_receipt(
    monkeypatch, tmp_path, capsys
):
    """Rung 5 is a resolve that worked, so it prints nothing at all (ruling 1).

    The disclosure is not dropped — Rule 10 still owns this rung, and it is the
    rung most in need of it: nothing was typed and no file was read. What
    changed is where it lands. A separate notice ahead of the run made a working
    rig read a message about sources before anything started, and its second
    clause explained why `newt` proceeded anyway — an apology for not erroring.
    The fact now travels back as a receipt for the verb to fold into the line it
    was already printing, which is the line the operator is definitely reading.

    Two assertions carry the ruling and they are both load-bearing: **nothing on
    either stream** (a verb that resolves cleanly is silent here), and a receipt
    that names *both* the short name and the kit — the name because it is what
    the operator would type to pin it, the kit because that is the thing they
    would uninstall, reinstall, or blame.
    """
    _no_config(monkeypatch, tmp_path)
    _declare(monkeypatch, rest=[("live_pair", "rest_source:live_pair", "widowx-kit")])

    resolved = resolve_spec("rest", None, REST_EXAMPLE)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""

    assert resolved.spec == "rest_source:live_pair"
    assert "live_pair" in resolved.receipt
    assert "widowx-kit" in resolved.receipt


def test_only_the_kit_supplied_rung_hands_back_a_receipt(monkeypatch, tmp_path):
    """The receipt is the verb's cue to say something, so it must not fire on a
    rung that already spoke or on one the operator typed themselves.

    Rung 3 named a file on stderr — the fact there is a path they can go open,
    and repeating it in the startup line would state one substitution twice.
    Rungs 1 and 2 are their own keystrokes; a receipt for what they just typed
    is the noise 029 refused. Only the rung with no file to name and no
    keystroke to credit owes the operator a line.
    """
    _declare(
        monkeypatch,
        rest=[
            ("live_pair", "rest_source:live_pair", "widowx-kit"),
            ("handy", "short_pkg:short_factory", "widowx-kit"),
        ],
    )
    _config(monkeypatch, tmp_path, '[sources]\nrest = "config_pkg:config_factory"\n')

    assert resolve_spec("rest", "pkg:factory", REST_EXAMPLE).receipt is None
    assert resolve_spec("rest", "handy", REST_EXAMPLE).receipt is None
    assert resolve_spec("rest", None, REST_EXAMPLE).receipt is None  # rung 3, the file

    _no_config(monkeypatch, tmp_path)
    _declare(monkeypatch, rest=[("live_pair", "rest_source:live_pair", "widowx-kit")])
    assert resolve_spec("rest", None, REST_EXAMPLE).receipt is not None


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
    assert resolve_spec("rest", None, REST_EXAMPLE).spec == "rest_source:live_pair"


# --------------------------------------------------------------------------- #
# Nineteen causes, nineteen strings (card test 4)
# --------------------------------------------------------------------------- #

def _every_refusal(
    monkeypatch, tmp_path, *, verb="rest", command=None, example=REST_EXAMPLE
) -> dict[str, str]:
    """Provoke all nineteen refusals for real and collect what each one printed.

    Built by driving ``resolve_spec`` rather than by reading the strings out of
    the module, so a cause that stops being reachable takes this test down with
    it.

    Eleven until newtrino-035, nineteen after. Four of the eleven fired in three
    situations that printed one sentence between them — nothing publishing to
    ``newt`` at all, kits publishing for other verbs, and the kit sitting in the
    directory the operator is standing in while a different ``newt`` answers —
    so each of those four is provoked from all three. The count moved because
    the corpus did.

    ``verb`` and ``command`` are separable here for the same reason they are
    separable in the resolver: the composed path resolves in one namespace and
    is typed as another command, and every one of these nineteen has to come out
    right on both."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    _stand_in_a_kit(monkeypatch, tmp_path)  # ...in no kit, until case 16 says so
    messages: dict[str, str] = {}

    def _catch(label, flag):
        with pytest.raises(SourceNotResolved) as exc:
            resolve_spec(verb, flag, example, command=command)
        messages[label] = str(exc.value)

    kit = [("live_pair", "rest_source:live_pair", "a-kit")]

    # 1 — the file is there and the TOML is broken.
    _declare(monkeypatch, **{verb: kit})
    _config(monkeypatch, tmp_path, f"[sources\n{verb} = \n", name="broken.toml")
    _catch("bad-toml", None)

    # 2 — the file is there and cannot be read at all. A directory at the config
    # path is the same class of problem as a permissions bit, and it does not
    # depend on who is running pytest.
    (tmp_path / "a-directory").mkdir(exist_ok=True)
    monkeypatch.setenv("NT_SITE_CONFIG", str(tmp_path / "a-directory"))
    _catch("unreadable", None)

    # 3 — the file parsed, and [sources] is not a flat map of strings.
    _config(monkeypatch, tmp_path, f"[sources]\n{verb} = 3\n", name="wrong-shape.toml")
    _catch("wrong-shape", None)

    # 4 — one short name, two installed kits declaring it.
    _no_config(monkeypatch, tmp_path)
    _declare(
        monkeypatch,
        **{verb: [
            ("live_pair", "widowx_rest:live_pair", "widowx-kit"),
            ("live_pair", "yam_rest:live_pair", "yam-kit"),
        ]},
    )
    _catch("two-kits-one-name", "live_pair")

    # 5 — a short name the operator typed, against a kit that declares others.
    _declare(monkeypatch, **{verb: kit})
    _catch("typed-name-unknown", "live_par")

    # 6 — a short name the *file* declared, against a kit that declares others.
    _config(
        monkeypatch, tmp_path, f'[sources]\n{verb} = "live_par"\n', name="typo.toml"
    )
    _catch("config-name-unknown", None)

    # 7 — a typed short name with no declarations anywhere.
    _declare(monkeypatch)
    _no_config(monkeypatch, tmp_path)
    _catch("typed-name-nothing-installed", "live_pair")

    # 8 — a file-declared short name with no declarations anywhere.
    _config(
        monkeypatch, tmp_path, f'[sources]\n{verb} = "live_pair"\n', name="orphan.toml"
    )
    _catch("config-name-nothing-installed", None)

    # 9 — several candidates and no declared default.
    _no_config(monkeypatch, tmp_path)
    _declare(
        monkeypatch,
        **{verb: [
            ("bench_left", "otherkit.rig:bench_left", "another-kit"),
            ("bench_right", "otherkit.rig:bench_right", "another-kit"),
        ]},
    )
    _catch("several-candidates", None)

    # 10 — nothing given, no file, nothing installed.
    _declare(monkeypatch)
    _catch("no-file-at-all", None)

    # 11 — a file that declares other verbs but not this one.
    _config(
        monkeypatch,
        tmp_path,
        '[sources]\nsomeotherverb = "teleop_source:live_pair"\n',
        name="other-verbs.toml",
    )
    _catch("file-declares-other-verbs", None)

    # 12-15 — the same four as 7, 8, 10 and 11, in the *other* empty-registry
    # world: kits are installed and publishing, and this verb is the gap. Same
    # symptom, different cause, different owner, and until newtrino-035 the
    # identical sentence. Each is a minimal pair with its twin above — same
    # config file, same path in the message, one variable changed — so a
    # collapse back into one string cannot hide behind a differing filename.
    sibling_kit = {
        sibling: [("live_pair", f"{sibling}_source:live_pair", "a-kit")]
        for sibling in [other for other in ("rest", "record", "teleop") if other != verb][:2]
    }

    # 12 — a typed short name; kits publish, none for this verb.
    _declare(monkeypatch, **sibling_kit)
    _no_config(monkeypatch, tmp_path)
    _catch("typed-name-verb-not-published", "live_pair")

    # 13 — a file-declared short name; kits publish, none for this verb.
    _config(
        monkeypatch, tmp_path, f'[sources]\n{verb} = "live_pair"\n', name="orphan.toml"
    )
    _catch("config-name-verb-not-published", None)

    # 14 — nothing given, no file; kits publish, none for this verb.
    _no_config(monkeypatch, tmp_path)
    _catch("no-file-verb-not-published", None)

    # 15 — a file that declares other verbs; kits publish, none for this verb.
    _config(
        monkeypatch,
        tmp_path,
        '[sources]\nsomeotherverb = "teleop_source:live_pair"\n',
        name="other-verbs.toml",
    )
    _catch("file-declares-other-verbs-verb-not-published", None)

    # 16-19 — the same four again, in the world where the kit is the directory
    # the operator is standing in and the running newt is not the project's.
    # Nothing is missing there, so the two readings above are both true and both
    # useless: they send someone to install a kit they are standing inside.
    # Minimal pairs with 7, 8, 10 and 11 — same empty registry, same config
    # file, one variable moved.
    _declare(monkeypatch)
    _stand_in_a_kit(monkeypatch, tmp_path, verb)

    # 16 — a typed short name, standing in the kit that publishes the verb.
    _no_config(monkeypatch, tmp_path)
    _catch("typed-name-kit-is-the-cwd", "live_pair")

    # 17 — a file-declared short name, standing in that kit.
    _config(
        monkeypatch, tmp_path, f'[sources]\n{verb} = "live_pair"\n', name="orphan.toml"
    )
    _catch("config-name-kit-is-the-cwd", None)

    # 18 — nothing given, no file, standing in that kit.
    _no_config(monkeypatch, tmp_path)
    _catch("no-file-kit-is-the-cwd", None)

    # 19 — a file that declares other verbs, standing in that kit.
    _config(
        monkeypatch,
        tmp_path,
        '[sources]\nsomeotherverb = "teleop_source:live_pair"\n',
        name="other-verbs.toml",
    )
    _catch("file-declares-other-verbs-kit-is-the-cwd", None)

    return messages


def _both_corpora(monkeypatch, tmp_path) -> dict[str, str]:
    """The nineteen as `newt rest`, plus the same nineteen as the composed path.

    Thirty-eight strings from one resolver, and no two of them may match. The
    composed half is not decoration: it is the half where the namespace and the
    command are different words, so it is the half a template that reuses one
    for the other collapses on."""
    corpus = {
        f"rest/{label}": message
        for label, message in _every_refusal(monkeypatch, tmp_path).items()
    }
    corpus.update(
        {
            f"composed/{label}": message
            for label, message in _every_refusal(
                monkeypatch,
                tmp_path,
                verb=_COMPOSED_VERB,
                command=_COMPOSED_COMMAND,
                example=_COMPOSED_EXAMPLE,
            ).items()
        }
    )
    return corpus


def test_nineteen_causes_produce_nineteen_distinct_strings(monkeypatch, tmp_path):
    """Rule 12, executable. "Could any other cause produce this identical
    string?" answered by asking the code instead of the author.

    This is what stops a later refactor from collapsing "your file is broken"
    into "your file declares nothing" behind one template — a change that reads
    as tidying and lands as two problems wearing one face.

    Twenty-two until newtrino-035. It moved to thirty-eight because four causes
    each became three, not because the literal was bumped to make a red test
    green: the twelve cases in ``_every_refusal`` that share a symptom and split
    on which world produced it are what carry the new sixteen, and deleting any
    of them takes this count with it.
    """
    messages = _both_corpora(monkeypatch, tmp_path)
    assert len(messages) == 38, "a cause stopped being reachable"

    seen: dict[str, str] = {}
    for label, message in messages.items():
        assert message not in seen, f"{label} and {seen[message]} print the same refusal"
        seen[message] = label


def test_the_nineteen_refusals_differ_on_their_first_line(monkeypatch, tmp_path):
    """The diagnosis is the first line — the rest is the fix.

    An operator at a bench reads one line before deciding what kind of problem
    they have. Two causes that differ only in their fourth line have already
    sent that reader to the wrong place, even though the full strings are
    technically distinct.

    The first line also has to open with the command the reader typed. A first
    line naming something they cannot run sends them to a shell to get "unknown
    verb" back, which is a second wrong diagnosis stacked on the first.
    """
    messages = _both_corpora(monkeypatch, tmp_path)

    first_lines: dict[str, str] = {}
    for label, message in messages.items():
        head = message.splitlines()[0]
        expected = "newt rest:" if label.startswith("rest/") else "newt record --teleop:"
        assert head.startswith(expected), f"{label} does not name the command first"
        assert head not in first_lines, (
            f"{label} and {first_lines[head]} open with the same diagnosis"
        )
        first_lines[head] = label


# --------------------------------------------------------------------------- #
# One empty registry was two all along (newtrino-035)
# --------------------------------------------------------------------------- #

def test_an_empty_registry_and_a_skipped_verb_are_two_different_refusals(
    monkeypatch, tmp_path
):
    """The bug this card was filed for, pinned as a minimal pair.

    Same config file, same path, same verb, same missing declaration — and two
    environments that are not the same problem. Nothing is publishing to `newt`
    here (wrong interpreter, kit not installed, install never re-synced), versus
    a kit that is installed, visible, and simply publishes nothing for this
    verb. The first is the operator's environment to fix; the second is the
    kit's gap to fix, and no amount of reinstalling touches it.

    Reproduced at a bench on 2026-08-06 in a throwaway environment with a kit
    declaring `record` and `rest` and not `teleop`: it printed the same sentence
    a machine with no kit at all printed. Rule 12 failing inside the family
    built to satisfy it.

    The interpreter path is asserted on both sides, in opposite directions. It
    is the whole answer in the first world and noise in the second, and a later
    edit that sprays it across every refusal to look thorough fails here.
    """
    _config(monkeypatch, tmp_path, "[robot_config]\nnonsense = 1\n", name="waldo.toml")

    _declare(monkeypatch)
    with pytest.raises(SourceNotResolved) as exc:
        resolve_spec("rest", None, REST_EXAMPLE)
    nothing_published = str(exc.value)

    _declare(monkeypatch, teleop=[("live_pair", "teleop_source:live_pair", "a-kit")])
    with pytest.raises(SourceNotResolved) as exc:
        resolve_spec("rest", None, REST_EXAMPLE)
    verb_not_published = str(exc.value)

    assert nothing_published.splitlines()[0] != verb_not_published.splitlines()[0], (
        "the two empty registries still open with the same diagnosis"
    )

    # World one: which interpreter is speaking is most of the answer, and the
    # zero-kits half is named beside it — a lone path sends a reader hunting for
    # another environment, and on this bench there wasn't one.
    assert sys.executable in nothing_published
    assert "No installed kit publishes any source to newt" in nothing_published
    assert "teleop" not in nothing_published

    # World two: the covered verb, read off a declaration. Never a roster of
    # sources newt made up — the anti-divination fence still holds here.
    assert "teleop" in verb_not_published
    assert sys.executable not in verb_not_published
    assert "live_pair" not in verb_not_published


# --------------------------------------------------------------------------- #
# The kit you are standing in (newtrino-035, § Direction)
# --------------------------------------------------------------------------- #

def test_standing_inside_the_kit_is_a_different_refusal_from_having_no_kit(
    monkeypatch, tmp_path
):
    """The second bench receipt, pinned as a minimal pair.

    She was inside the kit's checkout. The kit publishes the verb. The refusal
    told her no installed kit declares anything for it "in this environment" —
    true of the global tool she typed, and it reads as *go install something*
    when what she needed was four characters. Two environments were in play and
    the message could only see one.

    So the same empty registry now splits again on a fact about the directory,
    and the two refusals must not open alike: one is "nothing is publishing to
    newt anywhere I can see", the other is "it is publishing right here, and I
    am not the newt that project runs". The second hands a command; asserting
    that it *is* a command is the point of this test, because an explanation of
    environments is what shipped and what failed.
    """
    _declare(monkeypatch)
    _no_config(monkeypatch, tmp_path)

    with pytest.raises(SourceNotResolved) as exc:
        resolve_spec("rest", None, REST_EXAMPLE)
    no_kit_anywhere = str(exc.value)

    project = _stand_in_a_kit(monkeypatch, tmp_path, "rest")
    with pytest.raises(SourceNotResolved) as exc:
        resolve_spec("rest", None, REST_EXAMPLE)
    standing_in_it = str(exc.value)

    assert no_kit_anywhere.splitlines()[0] != standing_in_it.splitlines()[0], (
        "the kit in the cwd still opens with the diagnosis for having no kit"
    )

    # What it noticed: the project, by path, and that this newt is not its newt.
    assert str(project) in standing_in_it
    assert sys.executable in standing_in_it

    # The move is a command, typed as typed. Not a paragraph about environments.
    assert "uv run newt rest" in standing_in_it

    # And the advice that was wrong here is gone: they are standing in the kit,
    # so "install it, or go find the environment it is in" is a wild goose chase
    # with the goose in the room.
    assert "Install your rig's kit" in no_kit_anywhere
    assert "Install your rig's kit" not in standing_in_it


def test_the_cwd_kit_is_read_from_a_declaration_and_nothing_else(monkeypatch, tmp_path):
    """The real reader, driven against real directories — the seam's other half.

    Everything else in this file patches ``_cwd_kit_declaring`` so no test reads
    the directory pytest was launched from. That makes this the one place the
    reader itself is falsifiable, so all four of its answers are driven here:
    a project that publishes the verb, one that publishes something else, a
    directory with no readable project at all, and the case where re-running
    would change nothing.

    The fence it keeps is the same one the registry read keeps. A *declaration*
    in a file the developer wrote is a fact; a directory that merely looks like
    a kit is a guess, and guessing at file layout is how the SDK learns an
    embodiment's conventions.
    """
    project = tmp_path / "kit-checkout"
    project.mkdir()
    monkeypatch.chdir(project)

    # No project file at all — nothing to read, nothing claimed.
    assert _the_real_cwd_reader("rest") is None

    # A project that publishes something else. Present, readable, and not the
    # answer to this verb: the fork is per-verb or it sends people to `uv run`
    # for a verb that project cannot serve either.
    (project / "pyproject.toml").write_text(
        '[project]\nname = "a-kit"\n\n'
        '[project.entry-points."newt.sources.record"]\nbench_pair = "akit.rig:record"\n',
        encoding="utf-8",
    )
    assert _the_real_cwd_reader("rest") is None
    assert _the_real_cwd_reader("record") == project.resolve()

    # Broken TOML is somebody else's problem and not the one being reported.
    # This read is a courtesy on the way to a refusal about a missing source;
    # complaining about a second thing here buries the first.
    (project / "pyproject.toml").write_text("[project\nname =\n", encoding="utf-8")
    assert _the_real_cwd_reader("record") is None


def test_the_kit_in_the_cwd_says_nothing_when_you_are_already_running_its_newt(
    monkeypatch, tmp_path
):
    """`uv run newt rest` must never be the answer to `uv run newt rest`.

    The fork exists because two environments were in play. When the interpreter
    already *is* the project's, there is no second environment to send anyone
    to, and handing back the command they just typed is a loop wearing a fix's
    clothes — worse than the message it replaced, which at least pointed
    somewhere. So the reader answers None and the older, still-true reading of
    the registry gets the refusal back.
    """
    project = tmp_path / "kit-checkout"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "a-kit"\n\n'
        '[project.entry-points."newt.sources.rest"]\nbench_pair = "akit.rig:rest"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert _the_real_cwd_reader("rest") == project.resolve()

    monkeypatch.setattr(sys, "prefix", str(project / ".venv"))
    assert _the_real_cwd_reader("rest") is None


def test_no_refusal_reads_the_directory_pytest_happens_to_run_from(
    monkeypatch, tmp_path
):
    """The directory half of the machine-dependence trap, closed the same way.

    `conftest` patches the working-directory read by *function*. A refusal that
    reached the filesystem itself would notice whatever project the developer
    launched pytest from — this repo has a `pyproject.toml`, so the sentence
    would differ between a checkout, a CI runner, and a tmp dir, and the suite
    would be describing the machine instead of the code.

    Driven through conftest's fence alone, with no `_stand_in_a_kit` call: the
    corpus patches the seam at every step, so it would pass with the fence
    deleted, and a guard that cannot fail is not a guard. Verified by deleting
    the conftest line and watching this fail.
    """
    kit = tmp_path / "a-real-kit-checkout"
    kit.mkdir()
    (kit / "pyproject.toml").write_text(
        '[project]\nname = "a-kit"\n\n'
        '[project.entry-points."newt.sources.rest"]\nbench_pair = "akit.rig:rest"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(kit)

    _no_config(monkeypatch, tmp_path)
    with pytest.raises(SourceNotResolved) as exc:
        resolve_spec("rest", None, REST_EXAMPLE)
    message = str(exc.value)

    assert str(kit) not in message, "a refusal read the real working directory"
    assert "uv run" not in message


def test_no_refusal_reads_the_environment_pytest_happens_to_run_in(
    monkeypatch, tmp_path
):
    """The machine-dependence trap, closed by construction rather than by care.

    `conftest` patches the two entry-point reads by *function*, so any new code
    calling `entry_points()` directly walks straight past the fence and starts
    describing whoever's laptop is running the suite — green in CI, red at a
    bench, or the reverse. That is a worse failure than either.

    So the real read is made to explode, and the whole corpus is driven through
    it. If a refusal ever grows a third registry probe that skips the seam, this
    is the test that says so, on the commit that did it.
    """
    import importlib.metadata

    def _escaped(*args, **kwargs):
        raise AssertionError(
            "a refusal read the real environment's entry points instead of the seam"
        )

    monkeypatch.setattr(importlib.metadata, "entry_points", _escaped)

    # The bare case first, and it is the one that matters: no `_declare` call,
    # so nothing but conftest's autouse fence stands between this refusal and
    # the real registry. Written this way deliberately — the corpus below patches
    # both reads at every step, so it alone would pass with the fence deleted,
    # and a guard that cannot fail is not a guard.
    _no_config(monkeypatch, tmp_path)
    with pytest.raises(SourceNotResolved) as exc:
        resolve_spec("rest", None, REST_EXAMPLE)
    assert "publishes any source to newt" in str(exc.value)

    # Then the whole corpus, which catches a fourth probe that skips every seam.
    assert len(_both_corpora(monkeypatch, tmp_path)) == 38


# --------------------------------------------------------------------------- #
# The namespace is not the command (newtrino-030)
# --------------------------------------------------------------------------- #

def test_the_composed_path_never_tells_an_operator_to_run_a_command_that_exists_not(
    monkeypatch, tmp_path
):
    """`newt record --teleop` resolves under `demonstration`, and says so — once.

    There is no `newt demonstration`. A refusal branded with the namespace hands
    the operator a fix instruction for a verb the dispatcher does not have, so
    they run it, get "unknown verb", and are now debugging the wrong thing.
    Asserted across all nineteen causes rather than the one that was reported,
    because the branding came from a single argument doing two jobs and every
    string that argument reached had the same bug.
    """
    messages = _every_refusal(
        monkeypatch,
        tmp_path,
        verb=_COMPOSED_VERB,
        command=_COMPOSED_COMMAND,
        example=_COMPOSED_EXAMPLE,
    )
    assert len(messages) == 19

    for label, message in messages.items():
        assert "newt demonstration" not in message, (
            f"{label} advertises a command that does not exist:\n{message}"
        )
        assert "newt record --teleop" in message, (
            f"{label} never names the command the operator typed:\n{message}"
        )


def test_the_composed_refusals_still_name_the_table_the_operator_edits(
    monkeypatch, tmp_path
):
    """The other half, and the one a lazy fix breaks: `demonstration` is a real key.

    `[sources].demonstration` is the line an operator opens an editor and types,
    and `newt.sources.demonstration` is the group a kit publishes under. Deleting
    the noun to make the command-branding test pass would trade one wrong
    instruction for a message that cannot say where to declare anything. So the
    same corpus is asserted from the other side: the namespace survives, in the
    places that are about the namespace.

    One message does both, and that is the point rather than a smell — it names
    where the declaration is missing and what to retype, and those are two
    different things in two different places.
    """
    messages = _every_refusal(
        monkeypatch,
        tmp_path,
        verb=_COMPOSED_VERB,
        command=_COMPOSED_COMMAND,
        example=_COMPOSED_EXAMPLE,
    )

    # Where to declare it: the sample table carries the key, not the command.
    for label in ("no-file-at-all", "file-declares-other-verbs", "wrong-shape"):
        assert f'{_COMPOSED_VERB} = "' in messages[label], (
            f"{label} stopped naming the [sources] key to write:\n{messages[label]}"
        )

    # Whose roster this is: the namespace the lookup happened in.
    for label in ("typed-name-unknown", "several-candidates", "two-kits-one-name"):
        assert f"for {_COMPOSED_VERB}" in messages[label], (
            f"{label} stopped naming the namespace it searched:\n{messages[label]}"
        )

    both = messages["no-file-at-all"]
    assert f'{_COMPOSED_VERB} = "' in both and "newt record --teleop --source" in both


def test_a_verb_that_is_its_own_command_is_branded_exactly_as_before(
    monkeypatch, tmp_path
):
    """The regression fence on the three verbs that were never broken.

    `rest`, `record` and `teleop` are typed under the name they resolve under,
    so omitting the new argument has to be byte-identical to passing the verb as
    the command. If it ever isn't, the split leaked into the paths that had no
    ambiguity to fix.
    """
    for verb in ("rest", "record", "teleop"):
        implicit = _every_refusal(monkeypatch, tmp_path / verb, verb=verb)
        explicit = _every_refusal(
            monkeypatch, tmp_path / verb, verb=verb, command=verb
        )
        assert implicit == explicit, f"{verb} changed when the command was defaulted"
        for label, message in implicit.items():
            assert message.startswith(f"newt {verb}:"), f"{verb}/{label}: {message}"
