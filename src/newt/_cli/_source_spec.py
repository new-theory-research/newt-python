"""MODULE:FACTORY source loading, shared by every verb that takes ``--source``.

One definition of what ``--source mypkg.rig:make_source`` means: import the
module, look up the attribute, call it with no arguments, and name the spec at
every failure point. The factory owns producing a fully formed object for
whatever rig it wraps; the CLI never guesses at embodiment shape.

Extracted verbatim from ``newt record`` when a second verb needed the same
contract. Behavior and messages are unchanged — two verbs that disagree about
what a spec means is the thing this file prevents.
"""
from __future__ import annotations


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
