"""newt calibrate — tell the software where this rig's cameras are, or say you have not.

Like ``newt record``, ``newt teleop`` and ``newt rest``, this module is a skin. It
parses arguments, resolves and loads the developer's procedure, and maps what the
run returns onto an exit code. The announcement, the passes, the judgment and the
write live in ``newt.calibration`` — if you find them creeping in here, they are
in the wrong file.

What the verb owns and what the rig owns:
- the verb decides *when* — when the motion is announced, when the operator is
  asked, when the number is judged, and when a write is allowed to happen;
- the rig decides *what* — which board, which passes, which cameras, which
  solver, what number it calls acceptable, and where a result is persisted. All
  of that arrives through the procedure the kit declared and none of it is known
  here.

**Skipping is a first-class path, not a failure.** A person proving their cameras
stream should not be blocked by a procedure they do not need yet. ``--skip``
records that decision where it can be read, and every episode recorded afterwards
carries the mark. That is the whole of it: a rig has either measured geometry
somebody vouched for, or a written statement that it does not.
"""
from __future__ import annotations

import sys

from newt._cli._source_spec import SourceNotResolved, load_source, resolve_spec


def _usage() -> None:
    print("Usage: newt calibrate [options]")
    print()
    print("  Measure where each of this rig's cameras is, so the spatial claims in")
    print("  your recordings are real ones. The rig declares the board, the passes")
    print("  and the number it calls acceptable; this verb runs them, tells you what")
    print("  is about to move before it moves, shows you every camera's error next to")
    print("  that number, and only then lets anything be written.")
    print()
    print("  Not ready to do that yet? --skip records the decision, and every episode")
    print("  you record afterwards says its geometry was never measured. Skipping is")
    print("  supported. Recording under geometry nobody measured, silently, is not.")
    print()
    print("Options:")
    print("  --source SPEC   Which calibration procedure to run — either a short name")
    print("                  your kit declares, or a full MODULE:FACTORY import path")
    print("                  (e.g. mypkg.rig:make_calibration), which needs no")
    print("                  declaration. Optional on a configured rig: with no flag")
    print("                  the verb reads [sources].calibrate from your site config")
    print("                  ($NT_SITE_CONFIG, else ~/.config/nt/nt.toml), else the one")
    print("                  calibration procedure your kit declares. The flag always")
    print("                  wins.")
    print("  --skip [WHY]    Record that this rig's geometry is not measured, and do")
    print("                  not run anything. WHY is your own words and is optional.")
    print("  --yes           Do not ask before moving. The motion is still announced —")
    print("                  this skips the question, never the warning.")
    print()
    print("  This moves hardware. The announcement of what moves, how far and for how")
    print("  long is printed before the first millimetre, on every path.")
    print()
    print("Exit codes:")
    print("  0    solved, and every camera is inside the number this rig declared —")
    print("       or the run was a --skip, which did exactly what you asked")
    print("  1    a usage error, the procedure refused to come up, or you stopped it")
    print("       before anything moved")
    print("  2    refused before anything moved — this rig declares nothing to run")
    print("  3    refused before anything moved — the cameras this rig declares are")
    print("       not the cameras this machine can see")
    print("  4    a pass did not finish; the rig moved and stopped part-way through")
    print("  5    the solve did not answer, or the answer could not be saved — either")
    print("       way nothing was judged into a config")
    print("  6    solved, judged, written, and outside the declared number — the")
    print("       result is labelled suspect and so is every episode recorded under it")
    print("  130  interrupted (Ctrl+C)")


def _parse(args: list[str]) -> dict:
    """Hand-rolled, like every other verb here.

    ``--skip`` takes an optional value, which is the one wrinkle: the operator's
    own words if they gave any, and nothing if they did not. A following token is
    read as the reason only when it is not itself a flag, so ``--skip --yes``
    means what it looks like.
    """
    opts: dict = {"source": None, "skip": False, "reason": None, "yes": False}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--source":
            i += 1
            if i >= len(args):
                raise ValueError("--source expects a value")
            opts["source"] = args[i]
        elif a == "--skip":
            opts["skip"] = True
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
                opts["reason"] = args[i]
        elif a == "--yes":
            opts["yes"] = True
        else:
            raise ValueError(f"unknown option {a!r}")
        i += 1
    return opts


def cmd_calibrate(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    try:
        opts = _parse(args)
    except (ValueError, IndexError) as exc:
        print(f"newt calibrate: {exc}", file=sys.stderr)
        print("Run 'newt calibrate --help' for usage.", file=sys.stderr)
        return 1

    # One declared procedure resolves in silence and the operator types two
    # words. The selector appears on its own the day this rig declares a second
    # one — and it will, because the arm's own joint calibration shares the word.
    try:
        spec, source_receipt = resolve_spec(
            "calibrate", opts["source"], "mypkg.rig:make_calibration"
        )
    except SourceNotResolved as exc:
        print(str(exc), file=sys.stderr)
        return 1

    from newt.calibration import (
        EXIT_CAMERAS_UNAVAILABLE,
        EXIT_NOTHING_DECLARED,
        EXIT_USAGE,
        CalibrationError,
        CamerasUnavailable,
        NothingDeclared,
        require_calibration_procedure,
        run_calibration,
        run_skip,
    )

    try:
        procedure = load_source(spec)
    except KeyboardInterrupt:
        # Bring-up is where the factory connects cameras and arms, and on real
        # hardware that is seconds of blocking work — long enough to change your
        # mind. Nothing has been asked to move by then.
        print(
            "\n[newt calibrate] bring-up interrupted (Ctrl+C) — nothing moved and nothing "
            "was written.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:  # noqa: BLE001 — the factory's own refusal — a missing
        # camera runtime, an arm that would not answer. `load_source` runs a
        # developer-supplied factory (arbitrary code, arbitrary hardware bring-up);
        # it can raise anything, and this command's job is to surface it, not
        # trace it.
        print(f"[newt calibrate] {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        require_calibration_procedure(procedure)
        if opts["skip"]:
            return run_skip(procedure, opts["reason"])
        return run_calibration(
            procedure, assume_yes=opts["yes"], source_receipt=source_receipt
        )
    except NothingDeclared as exc:
        print(f"\n[newt calibrate] {exc}", file=sys.stderr)
        return EXIT_NOTHING_DECLARED
    except CamerasUnavailable as exc:
        print(f"\n[newt calibrate] {exc}", file=sys.stderr)
        return EXIT_CAMERAS_UNAVAILABLE
    except CalibrationError as exc:
        print(f"\n[newt calibrate] {exc}", file=sys.stderr)
        return EXIT_USAGE
