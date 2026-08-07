"""newt setup — the door onto whatever setup the installed kit declares.

`newt` does exactly three things behind this door: find the declaration, run
what it names, exit with whatever it returned. It does not know that a setup
usually installs drivers, does not know what a robot is, and does not know that
a script at some conventional path is the usual answer — a kit that declares
nothing gets a refusal, never a guess. **`newt` may not know what a robot is; it
may know what a kit promises**, and a setup entry point is a promise about the
kit rather than a fact about a rig.

The mirror is deliberate. A kit already tells `newt` what code drives its arm
(``newt.sources.<verb>``); this lets it also say what code gets the machine
ready. Same declaration mechanism, same resolution machinery, same refusal
dialect — a second spelling for the same idea would be a second family of
messages saying the same things slightly differently, which is the defect
``_source_spec`` exists to prevent.

**Why the callable runs in this process rather than as a subprocess.** The card
asks for live output and the child's exit code. In-process gets both for free:
the kit's setup writes to the file descriptors `newt` was given, so streaming is
not a feature anybody implemented — it is what happens when nothing captures.
An interactive prompt gets a real tty for the same reason.

**Why the argument list is forwarded unread.** A source factory builds an object
and has nothing a command line could configure. A setup *is* a command someone
typed, and kits already ship setups with flags their users and their agents
depend on. A zero-argument contract would make this verb strictly weaker than
the thing it is a door onto. So everything after the verb is handed over as a
list of strings, and `newt` never learns what any of it means. The one word it
reads is ``--kit``, and only in first position — see ``cmd_setup``.
"""
from __future__ import annotations

import sys

from newt._cli import _source_spec
from newt._cli._source_spec import KitDependencyMissing, import_factory

# The declaration surface, flat and with no verb tail. ``newt.sources.<verb>``
# has a tail because the tail is the verb being served, and a kit that declared
# a factory for one verb has not declared one for another. There is one setup:
# a tail here would be a namespace with a single legal value, which teaches a
# kit author a rule with nothing behind it.
SETUP_GROUP = "newt.setup"

# `newt` refused and nothing of the kit's ran. Distinct from any code the kit's
# own setup exits with, which passes through untouched — a caller that needs to
# tell "newt declined" from "the setup failed" reads this, and the failure path
# also names the kit on stderr.
EXIT_REFUSED = 2

# The one word this verb reads out of the argument list, and only when it comes
# first. A bare selector (``newt setup <name>``) was the shape the design
# sketched, and it is the shape this cycle backed away from: with one kit
# installed those keystrokes are an argument for the kit, with two they are a
# choice for `newt`, and one string meaning two things depending on what is
# installed is the ambiguity Rule 10 bans. First-position-only keeps the rule
# stateless — a kit's own ``--kit`` flag still reaches it from any other slot.
KIT_FLAG = "--kit"


def _usage() -> None:
    print("Usage: newt setup [--kit NAME] [ARGS...]")
    print()
    print("  Run the setup your installed kit declares. A kit says what getting")
    print("  its rig ready means — installing drivers, finding the hardware,")
    print("  writing the rig's config — and this is the door onto it. newt runs")
    print("  what the kit named and never learns what it does.")
    print()
    print("  Everything you type is handed to the kit's setup untouched, so its")
    print("  own flags work through this verb. The exception is --kit, and only")
    print("  as the first word: that is how you pick between two installed kits")
    print("  that both declare a setup.")
    print()
    print("  On success newt prints nothing of its own — the kit's report is the")
    print("  whole output. To read the kit's own flags, run its setup the way the")
    print("  kit documents: -h/--help anywhere in this command is newt's, and")
    print("  prints this page without running anything.")
    print()
    print("Declaring a setup, if you are writing a kit:")
    print()
    print('  [project.entry-points."newt.setup"]')
    print('  bench = "mypkg.setup:setup"')
    print()
    print("  The value is MODULE:CALLABLE. newt calls it once, in its own process,")
    print("  with the arguments that followed `newt setup` as a list of strings.")
    print("  Return 0 or None when it worked, nonzero when it did not.")
    print()
    print("Exit codes:")
    print("  0    the kit's setup finished and said it worked")
    print("  1    a usage error, or the kit's setup broke its side of the contract")
    print(f"  {EXIT_REFUSED}    newt refused and nothing ran — no kit here, no declaration, a")
    print("       declaration that would not load, or more than one to choose from")
    print("  130  interrupted (Ctrl+C) — the kit's setup did not finish")
    print("  *    any other code is the kit's own, passed straight through")


def _declared_setups() -> list[tuple[str, str, str]]:
    """What the installed kits declare: (name, spec, distribution), sorted.

    The same read ``_declared_sources`` does, against this card's group. Its own
    function rather than an argument on that one, for the reason
    ``_declaring_verbs`` is its own function: one seam per question, so the
    suite fences each in one place and no test reads the machine it runs on."""
    from importlib.metadata import entry_points

    found = []
    for entry in entry_points(group=SETUP_GROUP):
        dist = getattr(entry, "dist", None)
        # A distribution we cannot name is still a declaration we can honour; we
        # just do not invent a name for it.
        found.append(
            (entry.name, entry.value, dist.name if dist is not None else "an installed package")
        )
    return sorted(found)


def _kits_publishing_sources() -> list[str]:
    """Which installed distributions publish anything to newt at all.

    The discriminator between two refusals that share a symptom and share
    nothing else: *nothing is installed here* (the user's problem, early in the
    arc) versus *kits are installed and none of them declares a setup* (the
    kit's problem, and there is nothing the user can type). Built from the same
    two reads the source refusals use, so a test that describes an environment
    describes it once."""
    return sorted(
        {
            dist
            for verb in _source_spec._declaring_verbs()
            for _, _, dist in _source_spec._declared_sources(verb)
        }
    )


def _offered(entries: list[tuple[str, str, str]]) -> str:
    """Declared names for a refusal, each carrying the kit that declared it.

    Unlike the source list, the distribution is always shown. Choosing between
    two setups is choosing which kit owns this machine, and the kit's name is
    the whole basis for that choice."""
    return ", ".join(f"{name} ({dist})" for name, _, dist in entries)


def _nothing_declared() -> str:
    """The refusal when no installed kit declares a setup — one of three worlds.

    They share a symptom and nothing else, which is exactly the shape Rule 12
    forbids collapsing. Checked in the order they outrank each other: a kit in
    the directory you are standing in is a directory fact and beats both
    registry readings, because the environment sentence is true there and still
    sends the operator to install something they already have."""
    project = _source_spec._cwd_kit_declaring("setup", SETUP_GROUP)
    if project is not None:
        return (
            f"newt setup: the project in {project} declares a setup, and this newt is not "
            f"running from it — it is running from {sys.executable}.\n"
            f"\n"
            f"    uv run newt setup\n"
            f"\n"
            f"        That runs the project's own newt, the one that declaration is installed "
            f"in. Entry points are install metadata: they exist in the environment the kit was "
            f"installed into, and a globally installed newt never enters it."
        )

    publishing = _kits_publishing_sources()
    if publishing:
        # No headline command, on purpose. There is nothing the operator can
        # type, and inventing something — a script path, a conventional
        # filename — is the silent path-convention guess this verb exists to
        # refuse, wearing a helpful hat.
        # Read aloud, because a refusal is read at a bench and not in a diff:
        # one kit installed makes "publish ... none of them" ungrammatical, and
        # a message that stumbles is a message the reader stops trusting.
        found = (
            f"{publishing[0]} is installed and publishes sources to newt, and it declares no "
            f"setup"
            if len(publishing) == 1
            else (
                f"{', '.join(publishing)} are installed and publish sources to newt, and none "
                f"of them declares a setup"
            )
        )
        return (
            f"newt setup: {found}.\n"
            f"\n"
            f"        A setup is the kit's to declare, so there is nothing you can type that "
            f"changes this — newt will not guess at a script for a kit that promised nothing. "
            f'The kit has to publish a [project.entry-points."{SETUP_GROUP}"] entry and be '
            f"installed again.\n"
            f"        Until then, whatever that kit documents as its own setup is the only "
            f"door it has."
        )

    return (
        f"newt setup: no installed kit declares a setup, and nothing publishes to newt in this "
        f"environment at all — newt is running from {sys.executable}.\n"
        f"\n"
        f"    newt create\n"
        f"\n"
        f"        A setup belongs to a kit: the kit says what getting its rig ready means, and "
        f"this verb is only the door onto it. Start a project from a starter kit, or install "
        f"the kit you already have into the environment named above."
    )


def _resolve(entries: list[tuple[str, str, str]], chosen: str | None) -> tuple[str, str, str]:
    """Which declared setup this invocation runs. Raises the refusal, or picks.

    Refuses rather than picking whenever more than one kit is in play and
    nothing chose. Running *a* setup — silently, possibly not the one that owns
    this machine — is the identity-fill failure with a friendly face."""
    if chosen is not None:
        matches = [entry for entry in entries if entry[0] == chosen]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise _Refusal(
                f"newt setup: {chosen!r} is declared as a setup by more than one installed kit, "
                f"and newt will not guess which one owns this machine.\n"
                f"\n"
                f"        Declared by:  {', '.join(dist for _, _, dist in matches)}\n"
                f"        Uninstall the kit you did not mean to be running, and the name stops "
                f"being ambiguous."
            )
        if not entries:
            raise _Refusal(_nothing_declared())
        raise _Refusal(
            f"newt setup: no installed kit declares a setup named {chosen!r}.\n"
            f"\n"
            f"    newt setup --kit {entries[0][0]}\n"
            f"\n"
            f"        Declared here:  {_offered(entries)}"
        )

    if not entries:
        raise _Refusal(_nothing_declared())

    if len(entries) == 1:
        return entries[0]

    raise _Refusal(
        f"newt setup: {len(entries)} installed kits declare a setup, and choosing which one "
        f"owns this machine is not newt's to make.\n"
        f"\n"
        f"    newt setup --kit {entries[0][0]}\n"
        f"\n"
        f"        That picks one for this run. Declared here:  {_offered(entries)}\n"
        f"        Everything else you type still goes to that kit's setup untouched."
    )


class _Refusal(Exception):
    """A rendered refusal `newt` owns, printed by ``cmd_setup`` and exited on.

    Distinct from anything the kit's setup raises: those are the kit's, and this
    verb's whole job at that point is to attribute them rather than absorb
    them."""


def cmd_setup(args: list[str]) -> int:
    # The uniform guard every verb honours — no action performed when -h or
    # --help is present anywhere. It costs this verb the ability to forward a
    # kit's own --help, which is a real hole and the smaller of two: breaking
    # the guard for one verb makes `newt <anything> --help` a command a reader
    # has to check before typing.
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    chosen = None
    forwarded = list(args)
    if forwarded and forwarded[0] == KIT_FLAG:
        if len(forwarded) < 2:
            print(
                f"newt setup: {KIT_FLAG} expects the name of a declared setup.", file=sys.stderr
            )
            print("Run 'newt setup --help' for usage.", file=sys.stderr)
            return 1
        chosen = forwarded[1]
        forwarded = forwarded[2:]

    try:
        name, spec, dist = _resolve(_declared_setups(), chosen)
    except _Refusal as refusal:
        print(str(refusal), file=sys.stderr)
        return EXIT_REFUSED

    try:
        entry = import_factory(spec, noun="setup")
    except KitDependencyMissing as exc:
        # The kit is installed and something under it is not. Its own words
        # carry the repair; this line says which declaration led here.
        print(
            f"newt setup: the setup {name!r} declared by {dist} names {spec}, and {exc}",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    except (RuntimeError, ValueError) as exc:
        print(
            f"newt setup: the setup {name!r} declared by {dist} points at {spec}, and that "
            f"could not be loaded — {exc}",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    # Rule 10's declared-substitution clause. Nobody typed this name when one
    # kit answered on its own, and running a kit's arbitrary code without
    # saying whose is the silent substitution this lane keeps catching. One
    # line, on stderr, naming the declaration and not the rig — `newt` has
    # nothing true to say about a rig.
    print(f"[newt setup] running {name} — declared by {dist}", file=sys.stderr)

    def _after_the_kit_spoke() -> None:
        """Let the kit's own output land before newt says anything about it.

        Every line below claims the kit's words are *above*. On a terminal they
        are; piped into a log, stdout is block-buffered and stderr is not, so
        the attribution overtakes the diagnosis it is attributing and the two
        arrive in the wrong order. Found on a real terminal, which is the only
        place this was ever going to be found."""
        try:
            sys.stdout.flush()
        except (OSError, ValueError):
            # A closed or broken stdout is the kit's problem to have caused and
            # nothing this line can fix; the attribution still gets printed.
            pass

    try:
        returned = entry(forwarded)
    except SystemExit as exc:
        # A setup that calls sys.exit() means it — honoured as the code it
        # named rather than swallowed into this frame.
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(str(code), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        _after_the_kit_spoke()
        print(
            f"\nnewt setup: interrupted (Ctrl+C) — {dist}'s setup did not finish, and whatever "
            f"it had already changed on this machine is still changed. Nothing was rolled back; "
            f"nothing here knows how.",
            file=sys.stderr,
        )
        return 130
    except BaseException as exc:  # noqa: BLE001 — a kit's setup is arbitrary
        # code doing arbitrary things to a machine. It can raise anything, and
        # this verb's job is to attribute it, not to characterise it.
        import traceback

        _after_the_kit_spoke()
        traceback.print_exc()
        print(
            f"newt setup: {dist}'s setup raised {type(exc).__name__} and did not finish. The "
            f"traceback above is the kit's, and the kit is where it gets fixed — newt only ran "
            f"what {name} named.",
            file=sys.stderr,
        )
        return 1

    if returned is None:
        return 0
    if isinstance(returned, int) and not isinstance(returned, bool):
        if returned != 0:
            # The kit's output has already scrolled past. One line naming whose
            # failure it was, and nothing restated — re-wrapping a message the
            # operator just read is how a good message gets buried.
            _after_the_kit_spoke()
            print(
                f"newt setup: {dist}'s setup exited {returned}. That is the kit's own exit "
                f"code, passed through — what went wrong is above, in its words.",
                file=sys.stderr,
            )
        return returned

    _after_the_kit_spoke()
    print(
        f"newt setup: {dist}'s setup returned {returned!r}, and a setup must return an exit "
        f"code or None. newt will not guess whether that meant it worked.",
        file=sys.stderr,
    )
    return 1
