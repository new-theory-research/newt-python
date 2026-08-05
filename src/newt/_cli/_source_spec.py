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


def resolve_spec(verb: str, flag: str | None, example: str) -> str:
    """The one answer to "which factory builds this rig?", for every verb.

    Precedence, stated once: an explicit ``--source`` always wins — a flag the
    operator typed is never overridden by a file. Only when no flag was given do
    we read ``[sources].<verb>`` out of the rig's own config. With neither, the
    verb refuses and names *both* places it looked.

    When the answer comes from the file, it says so on stderr before anything
    starts (Rule 10's declared-substitution clause — a default that arrives
    silently is the failure mode this would otherwise introduce). stderr, not
    stdout, because ``newt record --json`` puts a machine-read stream on stdout.

    ``example`` is the verb's own ``MODULE:FACTORY`` example, used in refusals."""
    if flag:
        return flag

    path, provenance = site_config_path()
    if not path.exists():
        # Cause 1: nothing given, and nothing to read. Name both paths tried —
        # the flag that wasn't passed and the file that isn't there, at its
        # resolved absolute path and with the reason it's that path.
        raise SourceNotResolved(
            f"newt {verb}: no source to run. --source was not given, and this rig declares "
            f"none — there is no config file at {path} ({provenance}).\n"
            f"        Your rig has not been told which factory builds it. Declare it once:\n"
            f"            [{SOURCES_TABLE}]\n"
            f'            {verb} = "{example}"\n'
            f"        in {path}, and `newt {verb}` is all you ever type again.\n"
            f"        Or pass it this once: newt {verb} --source {example}"
        )

    declared = _read_declared_sources(verb, path, provenance)
    spec = declared.get(verb)
    if not spec:
        # Cause 2: the file was found and read, and it declares nothing for
        # *this* verb. Naming which verbs it does declare is the difference
        # between a guess and a fix.
        others = ", ".join(sorted(declared)) if declared else None
        has = (
            f"It declares: {others}."
            if others
            else f"It has no [{SOURCES_TABLE}] table at all."
        )
        raise SourceNotResolved(
            f"newt {verb}: your rig config at {path} ({provenance}) declares no source for "
            f"{verb!r}. {has}\n"
            f"        Add the line to that file:\n"
            f"            [{SOURCES_TABLE}]\n"
            f'            {verb} = "{example}"\n'
            f"        Or pass it this once: newt {verb} --source {example}"
        )

    if ":" not in spec:
        # Cause 4, the sharpest Rule 12 trap on this card. ``load_source`` says
        # "--source ... is not MODULE:FACTORY shaped", which is exactly wrong to
        # read while holding a terminal where you typed no --source. A spec that
        # came from the file says so, and names the file.
        raise SourceNotResolved(
            f"newt {verb}: your rig config at {path} ({provenance}) declares "
            f"[{SOURCES_TABLE}].{verb} = {spec!r}, which is not MODULE:FACTORY shaped.\n"
            f"        You did not type this — it came from that file. Fix the value there "
            f'(e.g. "{example}").\n'
            f"        Or override it this once: newt {verb} --source {example}"
        )

    print(
        f"[newt {verb}] source {spec} — declared as [{SOURCES_TABLE}].{verb} in {path}",
        file=sys.stderr,
    )
    return spec


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
