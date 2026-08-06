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


class SourceNotResolved(Exception):
    """No source spec could be resolved for a verb — the four causes below.

    Carries a fully rendered, verb-prefixed message; the caller prints it and
    exits. Distinct from the errors ``load_source`` raises, which are about a
    spec we *have* and could not honour."""


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


def _read_declared_sources(verb: str, path: Path, provenance: str) -> dict:
    """Take ``[sources]`` out of the rig's config and hand back the raw map.

    Opens the TOML, takes one table, returns strings. Everything else in the
    file is the kit's and is left untouched — see SOURCES_TABLE above."""
    import tomllib

    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        # Cause 3: the file is there and it is broken. The operator has a typo,
        # not a gap — a different problem from cause 2, so a different string.
        raise SourceNotResolved(
            f"newt {verb}: your rig config at {path} ({provenance}) is not valid TOML: {exc}\n"
            f"        That file is yours, not newt's — fix the syntax at the line named above.\n"
            f"        To get moving right now, bypass the file: "
            f"newt {verb} --source MODULE:FACTORY"
        ) from exc
    except OSError as exc:
        # Present-but-unreadable: a permissions or device problem, which is
        # neither "no file" nor "bad TOML".
        raise SourceNotResolved(
            f"newt {verb}: your rig config at {path} ({provenance}) exists but could not be "
            f"read: {exc}\n"
            f"        That is a filesystem problem on this machine, not a newt one — check the "
            f"file's permissions.\n"
            f"        To get moving right now, bypass the file: "
            f"newt {verb} --source MODULE:FACTORY"
        ) from exc

    declared = raw.get(SOURCES_TABLE, {})
    if not isinstance(declared, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in declared.items()
    ):
        # Also cause 3: the table parsed, but it is not the shape this reader
        # accepts. Say what shape was expected.
        raise SourceNotResolved(
            f"newt {verb}: your rig config at {path} ({provenance}) has a [{SOURCES_TABLE}] "
            f"table that is not a flat map of verb name to MODULE:FACTORY string.\n"
            f"        newt expects exactly this shape, and reads nothing else from the file:\n"
            f"            [{SOURCES_TABLE}]\n"
            f'            {verb} = "MODULE:FACTORY"\n'
            f"        Fix the table in that file, or bypass it: "
            f"newt {verb} --source MODULE:FACTORY"
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


def _resolve_short(verb: str, name: str, example: str, origin: str | None) -> str:
    """Turn a bare name into a spec, using only what a kit declared.

    ``origin`` is None when the operator typed the name and a rendered
    "your rig config at ... declares" clause when a *file* did. The two get
    different refusals for the same reason cause 4 exists: reading "no source
    named 'live_par'" while holding a terminal where you typed no --source
    sends you looking in the wrong place (Rule 12)."""
    entries = _declared_sources(verb)
    matches = [entry for entry in entries if entry[0] == name]

    if len(matches) == 1:
        return matches[0][1]

    if len(matches) > 1:
        # Two kits, one name. Picking either is the identity-fill failure with a
        # friendly face — it drives *an* arm, silently, possibly not yours.
        declarers = ", ".join(f"{dist} ({spec})" for _, spec, dist in matches)
        raise SourceNotResolved(
            f"newt {verb}: the name {name!r} is declared for {verb} by more than one installed "
            f"kit, and newt will not guess which one drives your rig.\n"
            f"        Declared by:  {declarers}\n"
            f"        Name the code directly to settle it:  "
            f"newt {verb} --source {matches[0][1]}\n"
            f"        Or uninstall the kit you don't mean to be running."
        )

    if entries:
        offered = _offered(entries)
        first = entries[0][0]
        if origin is None:
            # Cause 5-flag: they typed a name, and it wasn't there. The fix is
            # usually one character, and they can see it from here.
            raise SourceNotResolved(
                f"newt {verb}: no source named {name!r} is declared for {verb}.\n"
                f"        This machine offers, for {verb}:  {offered}\n"
                f"        Fix the name:  newt {verb} --source {first}\n"
                f"        Or name the code directly, which needs no declaration:  "
                f"newt {verb} --source {example}"
            )
        # Cause 5-config: a *file* named it. Say so before anything else, or the
        # operator goes looking at a command line they never typed it on.
        raise SourceNotResolved(
            f"newt {verb}: {origin} = {name!r}, and no source by that name is declared "
            f"for {verb}.\n"
            f"        You did not type this — it came from that file. This machine offers, "
            f"for {verb}:  {offered}\n"
            f"        Fix the value there, or override it this once:  "
            f"newt {verb} --source {first}"
        )

    # Nothing declared at all. Different cause, different fix: the name may be
    # perfect and the kit simply isn't installed here. Saying "no source named
    # X" alone would send them to fix a spelling that isn't wrong.
    if origin is None:
        raise SourceNotResolved(
            f"newt {verb}: no source named {name!r} is declared for {verb} — and neither is "
            f"anything else. No installed kit declares any source for {verb} on this machine.\n"
            f"        A bare name is an alias an installed kit publishes, and newt found no "
            f"declarations at all — check that your rig's kit is installed in this environment.\n"
            f"        Or name the code directly, which needs no declaration:  "
            f"newt {verb} --source {example}"
        )
    raise SourceNotResolved(
        f"newt {verb}: {origin} = {name!r}, and no installed kit declares any source for "
        f"{verb} on this machine.\n"
        f"        You did not type this — it came from that file. A bare name is an alias an "
        f"installed kit publishes, and newt found no declarations at all — check that your "
        f"rig's kit is installed in this environment.\n"
        f"        Or name the code directly, which needs no declaration:  "
        f"newt {verb} --source {example}"
    )


def resolve_spec(verb: str, flag: str | None, example: str) -> str:
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
    keystrokes, it says so on stderr before anything starts (Rule 10's
    declared-substitution clause — a default that arrives silently is the
    failure mode this would otherwise introduce). stderr, not stdout, because
    ``newt record --json`` puts a machine-read stream on stdout.

    ``example`` is the verb's own ``MODULE:FACTORY`` example, used in refusals."""
    if flag and ":" in flag:
        return flag
    if flag:
        # Rung 2. Nothing was substituted — they named it — so no provenance
        # line; there is nothing to declare that they don't already know.
        return _resolve_short(verb, flag, example, origin=None)

    path, provenance = site_config_path()
    have_config = path.exists()
    declared = _read_declared_sources(verb, path, provenance) if have_config else {}
    spec = declared.get(verb)

    if spec and ":" in spec:
        print(
            f"[newt {verb}] source {spec} — declared as [{SOURCES_TABLE}].{verb} in {path}",
            file=sys.stderr,
        )
        return spec

    if spec:
        # Rung 4. A bare name in the file is legal now, so the old "that isn't
        # MODULE:FACTORY shaped" refusal is gone from this path — a short name
        # is exactly what the operator was invited to write. What replaces it is
        # a refusal for a short name that matches nothing, carrying the same
        # "you did not type this" move.
        origin = (
            f"your rig config at {path} ({provenance}) declares [{SOURCES_TABLE}].{verb}"
        )
        resolved = _resolve_short(verb, spec, example, origin=origin)
        print(
            f"[newt {verb}] source {resolved} — declared as [{SOURCES_TABLE}].{verb} = "
            f'"{spec}" in {path}',
            file=sys.stderr,
        )
        return resolved

    entries = _declared_sources(verb)

    if len(entries) == 1:
        # Rung 5. Say which package supplied it: nothing was typed and no file
        # was read, so this is the substitution most in need of declaring.
        name, resolved, dist = entries[0]
        print(
            f"[newt {verb}] source {resolved} — the only source {dist} declares for {verb}, "
            f"and your rig config names no default",
            file=sys.stderr,
        )
        return resolved

    if entries:
        # Cause 6: the selector case. Both ways to choose, one for now and one
        # for good. No interactive prompt — a recovery verb must not block on a
        # menu with an arm moving, and a prompt breaks every non-tty caller.
        raise SourceNotResolved(
            f"newt {verb}: more than one source is declared for {verb}, and your rig config "
            f"names no default.\n"
            f"        Declared for {verb}:  {_offered(entries)}\n"
            f"        Pick one now:  newt {verb} --source {entries[0][0]}\n"
            f"        Or declare a default, once, in {path}:  "
            f'[{SOURCES_TABLE}]  {verb} = "{entries[0][0]}"'
        )

    if not have_config:
        # Cause 1: nothing given, nothing to read, nothing installed that
        # declares anything. Name every path tried — the flag that wasn't
        # passed, the file that isn't there at its resolved absolute path and
        # why it's that path, and the kits that declare nothing. This is the
        # refusal that cannot enumerate, so it says plainly that it found
        # nothing to read rather than inventing a roster to look helpful.
        raise SourceNotResolved(
            f"newt {verb}: no source to run. --source was not given, there is no config file "
            f"at {path} ({provenance}), and no installed kit declares a source for {verb}.\n"
            f"        Nothing on this machine has been told which factory builds your rig. "
            f"Declare it once:\n"
            f"            [{SOURCES_TABLE}]\n"
            f'            {verb} = "{example}"\n'
            f"        in {path}, and `newt {verb}` is all you ever type again.\n"
            f"        Or pass it this once: newt {verb} --source {example}"
        )

    # Cause 2: the file was found and read, and it declares nothing for *this*
    # verb. Naming which verbs it does declare is the difference between a
    # guess and a fix.
    others = ", ".join(sorted(declared)) if declared else None
    has = (
        f"It declares: {others}."
        if others
        else f"It has no [{SOURCES_TABLE}] table at all."
    )
    raise SourceNotResolved(
        f"newt {verb}: your rig config at {path} ({provenance}) declares no source for "
        f"{verb!r}. {has} No installed kit declares one for {verb} either.\n"
        f"        Add the line to that file:\n"
        f"            [{SOURCES_TABLE}]\n"
        f'            {verb} = "{example}"\n'
        f"        Or pass it this once: newt {verb} --source {example}"
    )


def load_source(spec: str):
    """Import a developer's source from a ``module:factory`` spec and construct it.

    The factory is called with no arguments — it owns producing a fully formed
    source (descriptor included) for whatever rig it wraps; the CLI never
    guesses at embodiment shape.

    Every failure point names the spec and what went wrong (Rule 10) — no
    silent fallback. Raises; the caller renders the message and exits, the
    loud-not-traced path."""
    import importlib

    if ":" not in spec:
        raise ValueError(
            f"--source {spec!r} is not MODULE:FACTORY shaped — expected e.g. "
            "'mypkg.rig:make_source'"
        )
    module_name, _, factory_name = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"--source {spec!r}: failed to import module {module_name!r}: {exc}"
        ) from exc
    try:
        factory = getattr(module, factory_name)
    except AttributeError:
        raise RuntimeError(
            f"--source {spec!r}: module {module_name!r} has no attribute {factory_name!r}"
        ) from None
    try:
        return factory()
    except Exception as exc:
        raise RuntimeError(
            f"--source {spec!r}: factory {factory_name!r} raised while constructing "
            f"the source: {exc}"
        ) from exc
