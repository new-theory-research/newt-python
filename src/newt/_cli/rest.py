"""newt rest — put the embodiment away, and say where each part actually ended up.

Like ``newt record`` and ``newt teleop``, this module is a skin. It parses
arguments, loads the developer's source, and renders what the run returns. The
sequence, the confirmation, and the reporting live in the library — if you find
them creeping in here, they are in the wrong file.

What the verb owns and what the rig owns:
- the verb decides *when*, *in what order*, and *what must be confirmed*;
- the rig decides *what* — which poses, through which driver call, over how
  long, and what "de-energized" means for it. All of that arrives through
  ``--source MODULE:FACTORY`` and none of it is known here.

**Why this verb has no kill key while ``newt teleop`` refuses to start without
one.** Teleop is an unbounded loop driven by a human hand, and Ctrl+H is how it
is stopped. A rest sequence is a short, declared, mostly-blocking list of moves
that ends de-energized on every path. A listener armed over the top of a
blocking vendor call would be a panic key that does nothing until the call
returns, which is worse than no panic key at all. And this is the command a
person reaches for *after* something already went wrong: a recovery action that
refuses to run outside a terminal is a recovery action people work around.
Nothing here is a settled position on where motion verbs sit on the autonomy
ladder — see the card's findings.

**This is not calibration.** No set-zero, no factory reset, no jig. It moves the
rig to where the rig says to leave it.
"""
from __future__ import annotations

import sys

from newt._cli._source_spec import load_source


def _usage() -> None:
    print("Usage: newt rest [options]")
    print("")
    print("  Ask an embodiment to run its own declared rest sequence, then leave")
    print("  every part de-energized and report the state each one reports back.")
    print("  The rig declares what putting itself away means; this verb never")
    print("  picks a pose. A part that declares no rest sequence is refused, by")
    print("  name, before anything moves.")
    print("")
    print("  Reach for it after a session was killed or faulted, when there is no")
    print("  clean exit path left to ride and the arms are wherever they stopped.")
    print("")
    print("Options:")
    print("  --source SPEC   Load a developer rest source, MODULE:FACTORY")
    print("                  (e.g. mypkg.rig:make_rig). Required.")
    print("")
    print("  This moves hardware. It does not change what any arm believes about")
    print("  its own zero — it is not a calibration command.")
    print("")
    print("Exit codes:")
    print("  0    every part rested and confirmed it ended de-energized")
    print("  1    a usage error, or the source refused to come up")
    print("  2    refused before anything was commanded — no rest sequence declared")
    print("  3    a declared step failed; that part did not finish its sequence")
    print("  4    a part went unanswered or would not say what state it ended in")
    print("  130  interrupted (Ctrl+C) — the run did not finish")


def _parse(args: list[str]) -> dict:
    opts: dict = {"source": None}
    valued = {"--source": "source"}
    i = 0
    while i < len(args):
        a = args[i]
        if a in valued:
            i += 1
            if i >= len(args):
                raise ValueError(f"{a} expects a value")
            opts[valued[a]] = args[i]
        else:
            raise ValueError(f"unknown option {a!r}")
        i += 1
    return opts


def cmd_rest(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    try:
        opts = _parse(args)
    except (ValueError, IndexError) as exc:
        print(f"newt rest: {exc}", file=sys.stderr)
        print("Run 'newt rest --help' for usage.", file=sys.stderr)
        return 1

    if not opts["source"]:
        print(
            "newt rest: --source is required (the MODULE:FACTORY that builds your rig).",
            file=sys.stderr,
        )
        print("        Fix: newt rest --source mypkg.rig:make_rig", file=sys.stderr)
        return 1

    from newt.rest import (
        EXIT_NOTHING_DECLARED,
        EXIT_USAGE,
        NothingDeclared,
        RestError,
        read_declarations,
        require_rest_source,
    )

    try:
        source = load_source(opts["source"])
    except KeyboardInterrupt:
        # Bring-up is where the factory connects, and on real hardware that is
        # seconds of blocking work — long enough for an operator to change their
        # mind. Putting away whatever came up is the factory's job; it is the
        # only thing holding those handles.
        print(
            "\n[newt rest] bring-up interrupted (Ctrl+C) — nothing was put away. Check "
            "what the source reported about anything it had already connected.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        # The factory's own refusal — a missing address, a missing driver.
        # Surface it, don't trace it.
        print(f"[newt rest] {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        parts = require_rest_source(source)
        plan = read_declarations(parts)
    except NothingDeclared as exc:
        print(f"\n[newt rest] {exc}", file=sys.stderr)
        return EXIT_NOTHING_DECLARED
    except RestError as exc:
        print(f"\n[newt rest] {exc}", file=sys.stderr)
        return EXIT_USAGE

    from newt.rest import run_rest

    return run_rest(source, plan)
