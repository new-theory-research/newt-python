"""MODULE:FACTORY source loading, shared by every verb that takes ``--source``.

One definition of what ``--source mypkg.rig:make_source`` means: import the
module, look up the attribute, call it with no arguments, and name the spec at
every failure point. The factory owns producing a fully formed object for
whatever rig it wraps; the CLI never guesses at embodiment shape.

Extracted verbatim from ``newt record`` when a second verb needed the same
contract. Behavior and messages are unchanged — two verbs that disagree about
what a spec means is the thing this file prevents.

It also owns *where a spec comes from when no flag was given* — ``resolve_spec``
below — for the same reason one layer up: three verbs disagreeing about where a
default comes from is the same defect. A configured rig declares its factories
once in its own site config, and the verb is then two words.

Two declaration surfaces feed it, split by the question they answer. The rig's
``nt.toml`` answers *"what does this bench use by default"* — operator-editable,
no reinstall. An installed kit's entry points answer *"what does this kit
offer"* — a code change, correctly requiring an install. Neither is a list
``newt`` carries: a roster of factory names inside the SDK is an embodiment fact
inside the SDK, and it goes stale the day a kit ships a factory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple

# The site config is the rig's own file, and this is the *only* thing ``newt``
# reads out of it: a flat map of verb name -> MODULE:FACTORY string. The
# hardware tables ([[robot_config.arms]], [[camera_config.cameras]]) belong to
# the embodiment kit and are never parsed, validated, or acknowledged here.
# That fence is what keeps ``newt`` embodiment-agnostic: it learns that a verb
# named "rest" has a factory declared for it, and nothing about what the factory
# builds.
SOURCES_TABLE = "sources"

# Matched to the resolution an embodiment kit already does for itself (its
# ``_resolve_site_config_path``): the env var if set, else this path. One config
# location, not two — a rig where ``newt`` and the kit's factories read
# different files is a debugging nightmare we would be minting deliberately.
SITE_CONFIG_ENV = "NT_SITE_CONFIG"
DEFAULT_SITE_CONFIG_PATH = "~/.config/nt/nt.toml"

# The second declaration surface: an installed kit publishes what it offers, per
# verb, as ``[project.entry-points."newt.sources.<verb>"]``. The value is the
# same MODULE:FACTORY string ``load_source`` already takes, so a short name is
# an alias a kit chose for its own code — never a pattern ``newt`` guessed at.
#
# The hard fence: a short name is resolved against this and nothing else. No
# sys.path scanning, no globbing for ``*_source.py``, no trying
# ``<name>_source:<name>``. Filesystem divination is how the SDK would learn a
# kit's file-naming convention, and a convention learned is an embodiment fact
# learned.
REGISTRY_GROUP = "newt.sources"


class Resolved(NamedTuple):
    """What the ladder answered, and what the verb still owes the operator.

    ``receipt`` is the short phrase a verb folds into the line it was already
    printing — ``<name> (from the <dist> kit)``, both halves filled in from the
    declaration that answered. It is set on exactly one rung: the one where
    nothing was typed and no file was read, so an installed kit's lone
    declaration answered on the operator's behalf. Every
    other rung is either their own keystrokes or a file that already named
    itself on stderr, and a verb handed ``None`` says nothing extra.

    A separate notice ahead of the run is what this replaces: the fact was
    always disclosed, but as an apology for proceeding rather than as a receipt
    inside the sentence the operator is actually reading."""

    spec: str
    receipt: str | None = None


class SourceNotResolved(Exception):
    """No source spec could be resolved for a verb — the four causes below.

    Carries a fully rendered message prefixed with the command the operator
    typed; the caller prints it and exits. Distinct from the errors
    ``load_source`` raises, which are about a spec we *have* and could not
    honour."""


def site_config_path() -> tuple[Path, str]:
    """The site config this rig uses, and the plain-English reason it's that one.

    Returns ``(path, provenance)``. The provenance half exists so a refusal can
    tell "I'm pointed at the wrong file" apart from "I have no file" — those are
    different problems with different fixes (Rule 12)."""
    override = os.environ.get(SITE_CONFIG_ENV) or None
    if override:
        return Path(override).expanduser().resolve(), f"from ${SITE_CONFIG_ENV}"
    return (
        Path(DEFAULT_SITE_CONFIG_PATH).expanduser().resolve(),
        f"the default location; set ${SITE_CONFIG_ENV} to point elsewhere",
    )


def _read_declared_sources(verb: str, command: str, path: Path, provenance: str) -> dict:
    """Take ``[sources]`` out of the rig's config and hand back the raw map.

    Opens the TOML, takes one table, returns strings. Everything else in the
    file is the kit's and is left untouched — see SOURCES_TABLE above.

    ``verb`` is the key looked up; ``command`` is what the operator types. See
    ``resolve_spec`` for why those are two arguments and not one."""
    import tomllib

    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        # Cause 3: the file is there and it is broken. The operator has a typo,
        # not a gap — a different problem from cause 2, so a different string.
        raise SourceNotResolved(
            f"newt {command}: your rig config at {path} ({provenance}) is not valid TOML: {exc}\n"
            f"\n"
            f"    newt {command} --source MODULE:FACTORY\n"
            f"\n"
            f"        That is the way past it right now. The file is yours, not newt's — fix "
            f"the syntax at the line named above and the flag stops being needed."
        ) from exc
    except OSError as exc:
        # Present-but-unreadable: a permissions or device problem, which is
        # neither "no file" nor "bad TOML".
        raise SourceNotResolved(
            f"newt {command}: your rig config at {path} ({provenance}) exists but could not be "
            f"read: {exc}\n"
            f"\n"
            f"    newt {command} --source MODULE:FACTORY\n"
            f"\n"
            f"        That is the way past it right now. This is a filesystem problem on this "
            f"machine, not a newt one — check the file's permissions."
        ) from exc

    declared = raw.get(SOURCES_TABLE, {})
    if not isinstance(declared, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in declared.items()
    ):
        # Also cause 3: the table parsed, but it is not the shape this reader
        # accepts. Say what shape was expected. The key in the sample table is
        # the verb, not the command — it is the line the operator edits.
        raise SourceNotResolved(
            f"newt {command}: your rig config at {path} ({provenance}) has a [{SOURCES_TABLE}] "
            f"table that is not a flat map of verb name to MODULE:FACTORY string.\n"
            f"\n"
            f"    newt {command} --source MODULE:FACTORY\n"
            f"\n"
            f"        That is the way past it right now. To fix the table itself: newt expects "
            f"exactly this shape, and reads nothing else from the file:\n"
            f"            [{SOURCES_TABLE}]\n"
            f'            {verb} = "MODULE:FACTORY"'
        )
    return declared


def _declared_sources(verb: str) -> list[tuple[str, str, str]]:
    """What the installed kits declare for this verb: (name, spec, distribution).

    Sorted by name, so refusals enumerate in a stable order. Called lazily and
    at most once per invocation — this walks the metadata of every installed
    distribution, and the explicit-long-form path must never pay for it."""
    from importlib.metadata import entry_points

    found = []
    for entry in entry_points(group=f"{REGISTRY_GROUP}.{verb}"):
        dist = getattr(entry, "dist", None)
        # A distribution we can't name is still a declaration we can honour; we
        # just don't invent a name for it.
        found.append((entry.name, entry.value, dist.name if dist is not None else "an installed package"))
    return sorted(found)


def _declaring_verbs() -> list[str]:
    """Which verbs anything installed here publishes sources for, at all.

    The companion read to ``_declared_sources``, and deliberately a second
    function rather than an argument: that one asks *"what does this verb
    have"*, this one asks *"is anything declaring to newt in this environment"*.
    Same seam discipline, for the same reason — one function the suite patches
    in one place, so no test ever reads the machine it is running on.

    Walks the group names only, never the entries: the answer is a list of verb
    names a kit chose to publish under, never a roster of factories ``newt``
    made up."""
    from importlib.metadata import entry_points

    prefix = f"{REGISTRY_GROUP}."
    return sorted(
        {
            group[len(prefix) :]
            for group in entry_points().groups
            if group.startswith(prefix) and group != prefix
        }
    )


def _cwd_kit_declaring(verb: str, group: str | None = None) -> Path | None:
    """The project you are standing in that publishes this verb, if that is why.

    The bench receipt (2026-08-06): standing inside a kit's own checkout,
    running the globally installed ``newt``, and being told no installed kit
    declares anything for the verb "in this environment". True, and useless —
    the kit was right there, and its declarations live in the *project's*
    environment, the one ``uv run`` enters and a global tool never does. The
    refusal explained the mechanism and never named the command that works,
    which was four characters away from the one already typed.

    So this reads ``pyproject.toml`` in the working directory and answers one
    question: does a project here publish ``newt.sources.<verb>``? A
    declaration, read from a file the developer wrote — never a directory
    guessed at from a file layout, which is the same fence the registry read
    keeps.

    Returns ``None`` when this interpreter is already the project's own: then
    re-running through the project changes nothing, and handing back the command
    they just typed is a loop rather than a fix.

    ``group`` is the full entry-point group to look for, and exists because the
    second declaration surface does not have a verb tail: a kit declares its
    setup under a flat ``newt.setup``, not ``newt.setup.<verb>``. Left unset it
    means ``newt.sources.<verb>``, which is every caller in this file. Whichever
    group is asked for, the answer is still read from a declaration the
    developer wrote — never from a directory that merely looks like a kit.

    One function so the suite has one seam to fence (``tests/conftest.py``),
    for the reason the two entry-point reads above have theirs — a refusal that
    reads the developer's working directory is a test that passes where it was
    written and fails where it runs."""
    import tomllib

    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return None

    if Path(sys.prefix).resolve().is_relative_to(cwd):
        return None

    try:
        with open(cwd / "pyproject.toml", "rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        # This read is a courtesy on the way to a refusal about something else.
        # A missing or broken pyproject is not the problem being reported, and
        # a second complaint stacked on the first helps nobody.
        return None

    project = raw.get("project")
    groups = project.get("entry-points") if isinstance(project, dict) else None
    if not isinstance(groups, dict):
        return None
    return cwd if groups.get(group or f"{REGISTRY_GROUP}.{verb}") else None


def _registry_state(verb: str, command: str) -> tuple[str, str, str | None]:
    """What this environment's registry actually looks like, and whose move it is.

    Until newtrino-035 the refusals below said one sentence — *"no installed kit
    declares one for <verb> either"* — in worlds that have nothing in common
    but their symptom:

    * **Nothing is publishing to** ``newt`` **here.** Wrong interpreter, kit
      never installed, install never re-synced. Which interpreter is speaking is
      then most of the answer, so it is named; and the zero-kits half is named
      beside it, because a bench where *both* environments are empty reads a
      lone interpreter path as "go find the other env" and there isn't one
      (diagnosed on this bench, 2026-08-06).
    * **Kits are publishing, and this verb is the gap.** A kit that landed three
      verbs of four, or predates the fourth. Naming the verbs that *are* covered
      turns "nobody told me anything" into a specific, checkable gap — and those
      names come from declarations, never from a list inside ``newt``.

    And the one that reads as neither, which is why it is checked first: **the
    kit is in the directory you are standing in**, publishing this verb, and
    this ``newt`` is not the one that project runs. Nothing is missing and
    nothing is stale; two environments are in play and only one of them was
    asked. That is a directory fact, not a registry fact, and it outranks both
    readings above — the environment sentence is true there and still sends the
    operator to install something they already have.

    Different causes, different owners, different fixes: one string for all
    three is what Rule 12 forbids, sitting inside the family built to satisfy
    it.

    Returns ``(noticed, move, headline)`` — what was found, what it means and
    whose it is, and the one command to put at the top of the refusal when this
    world has a better one than "name the code yourself". Only the standing-in-
    the-kit world does: there is a command that works, four characters from the
    one already typed, and ruling 2 says the fix is the headline or it isn't a
    fix. ``None`` means the refusal keeps its own ``--source`` line up there.

    Called only while building a refusal, so neither the metadata walk nor the
    directory read is on the path that resolves."""
    project = _cwd_kit_declaring(verb)
    if project is not None:
        return (
            (
                f"The project in {project} publishes {REGISTRY_GROUP}.{verb}, and this newt is "
                f"not running from it — it is running from {sys.executable}."
            ),
            "That runs the project's own newt, the one those declarations are installed in.",
            f"uv run newt {command}",
        )

    covered = [other for other in _declaring_verbs() if other != verb]
    if covered:
        return (
            f"Installed kits publish sources for {', '.join(covered)} — and none for {verb}.",
            (
                f"That gap is the kit's, not your setup's: it has to publish a "
                f"{REGISTRY_GROUP}.{verb} entry point, and be reinstalled, before {verb} has a "
                f"name here."
            ),
            None,
        )
    return (
        (
            f"No installed kit publishes any source to newt in this environment — newt is "
            f"running from {sys.executable}."
        ),
        (
            "Install your rig's kit into that environment, or run newt from the one it is "
            "already installed in."
        ),
        None,
    )


def _next_move(verb: str, command: str, example: str) -> tuple[str, str, str, str]:
    """The one command a nothing-declared refusal puts at the top, and the rest.

    Ruling 2 (newtrino-035): the fix is the headline or it isn't a fix. She hit
    an error whose recovery command was *in* the message and did not see it, so
    every refusal now isolates exactly one command — alone on its line, blank
    lines around it, before the explanation rather than at the end of it.

    Which command that is depends on the world ``_registry_state`` found. Only
    one of the three has something better than "name the code yourself": the kit
    is in the directory you are standing in, and ``uv run`` is the whole answer.
    Everywhere else the placeholder ``--source`` line is genuinely the best
    available move, because nothing was declared and there is no real name to
    hand over.

    Returns ``(noticed, move, headline, alternative)``. ``alternative`` is the
    "or name the code directly" line, and it is **empty when the headline is
    already that command** — printing one command twice in one refusal is the
    same defect this card was filed for, in the fix's clothes."""
    noticed, move, headline = _registry_state(verb, command)
    direct = f"newt {command} --source {example}"
    if headline is None:
        return noticed, move, direct, ""
    return (
        noticed,
        move,
        headline,
        f"\n        Or name the code directly, which needs no declaration:  {direct}",
    )


def _offered(entries: list[tuple[str, str, str]]) -> str:
    """Render declared names for a refusal, disambiguating only when we must.

    Bare names read fastest at a bench. A name declared twice is the one case
    where the bare list would lie about what the operator can pick, so that name
    — and only that name — carries its distribution."""
    seen: dict[str, int] = {}
    for name, _, _ in entries:
        seen[name] = seen.get(name, 0) + 1
    return ", ".join(
        name if seen[name] == 1 else f"{name} ({dist})" for name, _, dist in entries
    )


def _resolve_short(
    verb: str, command: str, name: str, example: str, origin: str | None
) -> str:
    """Turn a bare name into a spec, using only what a kit declared.

    ``origin`` is None when the operator typed the name and a rendered
    "your rig config at ... declares" clause when a *file* did. The two get
    different refusals for the same reason cause 4 exists: reading "no source
    named 'live_par'" while holding a terminal where you typed no --source
    sends you looking in the wrong place (Rule 12).

    ``verb`` names the namespace these declarations live in; ``command`` names
    the thing the operator retypes with the fix. On most verbs they are the same
    word — see ``resolve_spec`` for the path where they are not."""
    entries = _declared_sources(verb)
    matches = [entry for entry in entries if entry[0] == name]

    if len(matches) == 1:
        return matches[0][1]

    if len(matches) > 1:
        # Two kits, one name. Picking either is the identity-fill failure with a
        # friendly face — it drives *an* arm, silently, possibly not yours.
        declarers = ", ".join(f"{dist} ({spec})" for _, spec, dist in matches)
        raise SourceNotResolved(
            f"newt {command}: the name {name!r} is declared for {verb} by more than one "
            f"installed kit, and newt will not guess which one drives your rig.\n"
            f"\n"
            f"    newt {command} --source {matches[0][1]}\n"
            f"\n"
            f"        That names the code directly, which settles it — swap the spec for the "
            f"other one if it is the kit you meant.\n"
            f"        Declared by:  {declarers}\n"
            f"        Or uninstall the kit you don't mean to be running."
        )

    if entries:
        offered = _offered(entries)
        first = entries[0][0]
        if origin is None:
            # Cause 5-flag: they typed a name, and it wasn't there. The fix is
            # usually one character, and they can see it from here.
            raise SourceNotResolved(
                f"newt {command}: no source named {name!r} is declared for {verb}.\n"
                f"\n"
                f"    newt {command} --source {first}\n"
                f"\n"
                f"        This machine offers, for {verb}:  {offered}\n"
                f"        Or name the code directly, which needs no declaration:  "
                f"newt {command} --source {example}"
            )
        # Cause 5-config: a *file* named it. Say so before anything else, or the
        # operator goes looking at a command line they never typed it on.
        raise SourceNotResolved(
            f"newt {command}: {origin} = {name!r}, and no source by that name is declared "
            f"for {verb}.\n"
            f"\n"
            f"    newt {command} --source {first}\n"
            f"\n"
            f"        You did not type this — it came from that file, and the line above "
            f"overrides it this once. Fixing the value there fixes it for good.\n"
            f"        This machine offers, for {verb}:  {offered}"
        )

    # Nothing declared for this verb. Different cause, different fix: the name
    # may be perfect and the kit simply isn't installed here. Saying "no source
    # named X" alone would send them to fix a spelling that isn't wrong — and
    # ``_registry_state`` says which of the two registries this is, because
    # "nothing is installed" and "the kit skipped this verb" are not one problem.
    noticed, move, headline, alternative = _next_move(verb, command, example)
    if origin is None:
        raise SourceNotResolved(
            f"newt {command}: no source named {name!r} is declared for {verb}. {noticed}\n"
            f"\n"
            f"    {headline}\n"
            f"\n"
            f"        A bare name is an alias an installed kit publishes. {move}"
            f"{alternative}"
        )
    raise SourceNotResolved(
        f"newt {command}: {origin} = {name!r}. {noticed}\n"
        f"\n"
        f"    {headline}\n"
        f"\n"
        f"        You did not type this — it came from that file. A bare name is an "
        f"alias an installed kit publishes. {move}"
        f"{alternative}"
    )


def resolve_spec(
    verb: str, flag: str | None, example: str, *, command: str | None = None
) -> Resolved:
    """The one answer to "which factory builds this rig?", for every verb.

    The ladder, stated once and tested as a unit:

    1. ``--source`` with a colon — a raw import path, loaded directly. Neither
       the registry nor the config is consulted; the escape hatch answers to
       nobody, and it must not pay for a metadata scan it doesn't use.
    2. ``--source`` without a colon — a short name, looked up in what the
       installed kits declare *for this verb*. The verb is the namespace, which
       is what removes the doubling when a kit's module name restates the verb
       the operator already typed.
    3. ``[sources].<verb>`` with a colon — the rig's declared default.
    4. ``[sources].<verb>`` without a colon — the rig's default, by short name.
    5. No flag, no declaration, and the installed kits offer exactly one source
       for this verb — resolve it. A single candidate is not a choice, and the
       card's own rule is that ambiguity asks while a lone default resolves in
       silence. **This is the rung that makes a freshly installed kit with no
       config file at all work**, and it is broader than the card's summary
       table reads; it is deliberate, and it is stated here rather than
       discovered later.
    6. More than one candidate and nothing chosen — refuse and list them.
    7. Nothing anywhere — refuse, and teach where declarations live.

    Whenever the answer came from somewhere other than the operator's own
    keystrokes, it is disclosed (Rule 10's declared-substitution clause — a
    default that arrives silently is the failure mode this would otherwise
    introduce). *Where* it is disclosed depends on what answered. A file that
    was read names itself on stderr, here, before anything starts — the fact is
    about a path the operator can go open. Rung 5 has no file to name, and its
    disclosure travels back as ``Resolved.receipt`` for the verb to fold into
    its own startup line: a rig with one obvious answer gets no message about
    sources at all, only the line it was always going to print, carrying the
    name and the kit (newtrino-035, ruling 1). stderr, not stdout, for the ones
    that stay here, because ``newt record --json`` puts a machine-read stream on
    stdout.

    ``example`` is the verb's own ``MODULE:FACTORY`` example, used in refusals.

    ``command`` is the two jobs ``verb`` used to do, pulled apart. ``verb`` is
    the *namespace* a declaration lives in — the key in ``[sources]``, the tail
    of the ``newt.sources.<verb>`` entry-point group. ``command`` is what the
    operator types at a shell. For ``rest``, ``record`` and ``teleop`` those are
    the same word, which is why they were one argument and why nobody noticed;
    on the composed path they are not. ``newt record --teleop`` resolves in the
    ``demonstration`` namespace, and a refusal that brands itself with the
    namespace prints ``newt demonstration --source ...`` — a fix instruction for
    a command that does not exist, which is Rule 12 failing on its own terms:
    loud, and pointing at nothing.

    So the split runs through every string below. A line that says *where the
    declaration lives* keeps the verb — ``[sources].demonstration`` is the line
    the operator opens an editor and types, and renaming it in a message would
    be a different lie. A line the operator retypes at a prompt takes the
    command. One message legitimately does both: it says the name is declared
    for ``demonstration`` and to fix it with ``newt record --teleop --source``,
    because those are two different places and the message names each
    correctly."""
    command = command or verb

    if flag and ":" in flag:
        return Resolved(flag)
    if flag:
        # Rung 2. Nothing was substituted — they named it — so no provenance
        # line and no receipt; there is nothing to declare that they don't
        # already know.
        return Resolved(_resolve_short(verb, command, flag, example, origin=None))

    path, provenance = site_config_path()
    have_config = path.exists()
    declared = (
        _read_declared_sources(verb, command, path, provenance) if have_config else {}
    )
    spec = declared.get(verb)

    if spec and ":" in spec:
        print(
            f"[newt {command}] source {spec} — declared as [{SOURCES_TABLE}].{verb} in {path}",
            file=sys.stderr,
        )
        return Resolved(spec)

    if spec:
        # Rung 4. A bare name in the file is legal now, so the old "that isn't
        # MODULE:FACTORY shaped" refusal is gone from this path — a short name
        # is exactly what the operator was invited to write. What replaces it is
        # a refusal for a short name that matches nothing, carrying the same
        # "you did not type this" move.
        origin = (
            f"your rig config at {path} ({provenance}) declares [{SOURCES_TABLE}].{verb}"
        )
        resolved = _resolve_short(verb, command, spec, example, origin=origin)
        print(
            f"[newt {command}] source {resolved} — declared as [{SOURCES_TABLE}].{verb} = "
            f'"{spec}" in {path}',
            file=sys.stderr,
        )
        return Resolved(resolved)

    entries = _declared_sources(verb)

    if len(entries) == 1:
        # Rung 5. Nothing was typed and no file was read, so this is the
        # substitution most in need of declaring — and it is declared in the
        # verb's own startup line rather than in a notice ahead of it. A rig
        # with one possible answer is not a situation; printing about it before
        # anything runs made a receipt read like a warning, and the clause that
        # explained why `newt` proceeded anyway answered a question nobody
        # asked (newtrino-035, ruling 1).
        name, resolved, dist = entries[0]
        return Resolved(resolved, f"{name} (from the {dist} kit)")

    if entries:
        # Cause 6: the selector case. Both ways to choose, one for now and one
        # for good. No interactive prompt — a recovery verb must not block on a
        # menu with an arm moving, and a prompt breaks every non-tty caller.
        raise SourceNotResolved(
            f"newt {command}: more than one source is declared for {verb}, and your rig config "
            f"names no default.\n"
            f"\n"
            f"    newt {command} --source {entries[0][0]}\n"
            f"\n"
            f"        That picks one for now. Declared for {verb}:  {_offered(entries)}\n"
            f"        Or declare a default, once, in {path}:  "
            f'[{SOURCES_TABLE}]  {verb} = "{entries[0][0]}"'
        )

    noticed, move, headline, alternative = _next_move(verb, command, example)

    if not have_config:
        # Cause 1: nothing given, nothing to read, nothing declared for this
        # verb. Name every path tried — the flag that wasn't passed, the file
        # that isn't there at its resolved absolute path and why it's that path,
        # and what the kit registry here does and doesn't hold. This is the
        # refusal that cannot enumerate sources, so it names what it *can* read
        # rather than inventing a roster to look helpful.
        raise SourceNotResolved(
            f"newt {command}: no source to run. --source was not given, and there is no config "
            f"file at {path} ({provenance}). {noticed}\n"
            f"\n"
            f"    {headline}\n"
            f"\n"
            f"        {move}\n"
            f"        Or declare the factory once:\n"
            f"            [{SOURCES_TABLE}]\n"
            f'            {verb} = "{example}"\n'
            f"        in {path}, and `newt {command}` is all you ever type again."
            f"{alternative}"
        )

    # Cause 2: the file was found and read, and it declares nothing for *this*
    # verb. Naming which verbs it does declare is the difference between a
    # guess and a fix — and when it declares nothing at all, that is the whole
    # sentence. It used to be two ("declares no source for 'rest'. It has no
    # [sources] table at all."), which is the same absence twice and the tone
    # this card was filed for: the file is empty of sources, so of course it has
    # none for this verb.
    others = ", ".join(sorted(declared)) if declared else None
    found = (
        f"declares no source for {verb!r}. It declares: {others}."
        if others
        else f"has no [{SOURCES_TABLE}] table."
    )
    raise SourceNotResolved(
        f"newt {command}: your rig config at {path} ({provenance}) {found} {noticed}\n"
        f"\n"
        f"    {headline}\n"
        f"\n"
        f"        {move}\n"
        f"        Or add the line to that file:\n"
        f"            [{SOURCES_TABLE}]\n"
        f'            {verb} = "{example}"'
        f"{alternative}"
    )


class KitDependencyMissing(RuntimeError):
    """The code a spec names is here; something it imports is not.

    Its own class because the two neighbouring failures have opposite fixes and
    a verb that renders them the same way sends the operator to the wrong one.
    "The factory could not be found" is a spelling or an install of *the kit*;
    this is the kit installed and its *drivers* absent, which is a different
    command and, on the composed record path, a different account of what is
    energized."""


# The one repair command `newt` may name without learning what a robot is.
# newtrino-020 made "a kit ships an executable scripts/setup" the handoff
# contract, so citing it cites our own contract. A driver's module name would
# be an embodiment fact in the wrong repo — stale the day a kit changes
# hardware — and the fence is exactly there: **newt may not know what a robot
# is; it may know what a kit promises.**
KIT_SETUP = "./scripts/setup"


def _reported(exc: BaseException) -> str:
    """The failing code's own words, indented and kept whole.

    Kept rather than characterised, and put *below* the isolated command rather
    than in front of it. The receipt behind this whole card is a bench error
    that already carried the right command and buried it — and half of what
    buried it was `newt` wrapping a good message from a kit inside a sentence
    of its own. Rule 10 says don't swallow it; the structure law says don't let
    it stand where the fix should be."""
    return "\n".join(f"            {line}" for line in str(exc).splitlines())


def _dependency_lantern(headline: str, spec: str, exc: BaseException, noun: str = "source") -> str:
    """The one message for "the code is installed, what it needs isn't".

    One renderer, two sites (import-time and construction-time), because three
    verbs printing three variants of this is the disagreement `_source_spec`
    exists to prevent. The headline is what differs; everything a reader
    retypes is written once, here.

    Structure law (newtrino-035): the command alone on its line, blank lines
    around it, directly under the diagnosis rather than at the end of the
    paragraph."""
    return (
        f"{headline}\n"
        f"\n"
        f"    {KIT_SETUP}\n"
        f"\n"
        f"        Run that from the kit's own directory. A kit ships that script to "
        f"install what its rig needs, and a plain `uv sync` resolves only the default "
        f"dependency set — optional drivers are exactly what it leaves out.\n"
        f"        If this {noun} is your own code and not an installed kit, the import "
        f"below is yours — nothing else about the spec is wrong.\n"
        f"        The {noun}:  {spec}\n"
        f"        What it reported:\n"
        f"{_reported(exc)}"
    )


def _spec_module_missing(exc: BaseException, module_name: str) -> bool:
    """Did the spec's *own* module chain fail to import, or something under it?

    The whole discriminator, and it never learns a driver's name.
    ``ModuleNotFoundError`` carries the module that was not found. If that is
    the spec's module — or a package the spec's module lives inside, which is
    what ``import mypkg.rig`` raises when ``mypkg`` is absent — then the code
    the operator named is not here. Anything else means the named module *was*
    found, started importing, and something further down was missing: the
    signature of a kit whose drivers fell out.

    ``name`` is None only when a module body raises ``ModuleNotFoundError`` by
    hand, which is the second case, not the first."""
    name = getattr(exc, "name", None)
    if not isinstance(exc, ImportError) or name is None:
        return False
    return module_name == name or module_name.startswith(f"{name}.")


def import_factory(spec: str, noun: str = "source"):
    """Phase one: find the factory a spec names. **Never calls it.**

    ``noun`` is what the declaration is called in the refusals this raises, and
    it defaults to the word every caller but one wants. `newt setup` resolves a
    kit's setup through this same machinery — deliberately, so there is one
    dialect for "the code a declaration names is not here" — but a refusal that
    calls it a *source* sends the reader to the source ladder, which has nothing
    to do with why their setup did not run. Same message, right noun.

    Import is not connect; construction is. Splitting the two is what lets a
    verb read what a factory *declares* while the rig is still dark, and refuse
    before anything energizes — the bench receipt is a ``newt rest`` pointed at
    a read-only factory by a one-word typo, which connected and energized a rig
    on its way to a refusal that then admitted it had no way to put back what it
    had just brought up (2026-08-05).

    Verbs with nothing to validate call ``load_source`` and never see this
    split. Verbs that gate on a declaration call this, read it off the returned
    callable, and only then call ``build_source``."""
    import importlib

    if ":" not in spec:
        raise ValueError(
            f"the source {spec!r} is not MODULE:FACTORY shaped — expected e.g. "
            "'mypkg.rig:make_source'"
        )
    module_name, _, factory_name = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        if _spec_module_missing(exc, module_name):
            raise RuntimeError(
                f"no module named {exc.name!r} could be imported, so the code this "
                f"{noun} names was never found.\n"
                f"\n"
                f'    uv run python -c "import {exc.name}"\n'
                f"\n"
                f"        That runs the interpreter newt is running under, which is what "
                f"separates the two ways this happens: a spec that names a module nobody "
                f"has (a typo, or a package never installed) from one installed somewhere "
                f"this environment cannot see.\n"
                f"        The {noun}:  {spec}"
            ) from exc
        raise KitDependencyMissing(
            _dependency_lantern(
                f"the code this {noun} names is installed, but something it imports is "
                f"not — so nothing was loaded, and nothing is connected.",
                spec,
                exc,
                noun,
            )
        ) from exc
    try:
        return getattr(module, factory_name)
    except AttributeError:
        raise RuntimeError(
            f"module {module_name!r} has no attribute {factory_name!r}, so the {noun} "
            f"{spec!r} names a {'factory' if noun == 'source' else 'callable'} that module "
            f"does not export."
        ) from None


def build_source(spec: str, factory):
    """Phase two: call the factory. This is the call that touches hardware."""
    factory_name = spec.partition(":")[2]
    try:
        return factory()
    except ImportError as exc:
        # A kit that guards its own driver imports does it here, inside the
        # factory, so a missing driver arrives at construction and never at
        # import — which is why this card covers both sites and a fix at
        # `import_factory` alone would have missed the bench receipt entirely.
        # `isinstance` is reporting, not divining: an ImportError out of a
        # factory call *is* an import that failed.
        raise KitDependencyMissing(
            _dependency_lantern(
                f"the factory {factory_name!r} could not import something it needs "
                f"while bringing the rig up.",
                spec,
                exc,
            )
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"the factory {factory_name!r} raised while constructing the source "
            f"{spec!r}: {exc}"
        ) from exc


def load_source(spec: str):
    """Import a developer's source from a ``module:factory`` spec and construct it.

    The factory is called with no arguments — it owns producing a fully formed
    source (descriptor included) for whatever rig it wraps; the CLI never
    guesses at embodiment shape.

    Every failure point names the spec and what went wrong (Rule 10) — no
    silent fallback. Raises; the caller renders the message and exits, the
    loud-not-traced path."""
    return build_source(spec, import_factory(spec))
