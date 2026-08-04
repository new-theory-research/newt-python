"""newt create — put a starter kit on a developer's machine, and make it theirs.

A starter kit is a project you clone and own, not a library you install and use.
``git clone`` gets the bytes there but leaves a ``origin`` pointing back at us, so
the first thing a developer owns is something that still remembers where it came
from. This verb fetches the same kit at a pinned commit and stops: no remote, no
history of ours, no ongoing relationship.

**What this verb knows, and what it refuses to know.** It knows a template
*name*, and how to turn that name into something fetchable. That is the entire
allowance, and ``_template_registry`` is where it lives. It does not know that a
rig has a leader and a follower, how many cameras it wants, what they are called,
or which driver version matches its firmware. All of that already lives in the
kit, and the kit's own setup step is what asks about it. If you find yourself
reaching for a rig-shaped ``if`` in this file, the seam is in the wrong place.

**The console is the registry.** It is the only thing holding a credential that
can read a private starter kit, so it is what resolves names and serves those
kits — the developer's ``nt_`` key is the whole story, and GitHub identity never
enters it. When the console is unreachable, public templates still fetch direct
from a small offline table; private ones fail with a string that says so.

**Where the verb stops.** Every question about a rig — which arm is the leader,
what the cameras are called, which values still need measuring — is asked by the
kit's own setup step, and this verb's entire involvement is to hand it arguments
it never reads. That handoff is at the bottom of this file, and it is deliberately
the dumbest code here: an argv list, a working directory, and an exit code.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from newt._credentials import read_api_key
from newt._cli._template_registry import (
    FALLBACK_TEMPLATES,
    Template,
    find,
    names,
    parse_templates,
)

_DEFAULT_CONSOLE = "https://newtheory-console.vercel.app"

# Exit codes. Every one of these is a different thing going wrong, and each has
# its own message — a reader must never have to guess which of two causes
# produced the line they are looking at.
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_NO_SUCH_TEMPLATE = 2
EXIT_NEEDS_KEY = 3
EXIT_CONSOLE_UNREACHABLE = 4
EXIT_KEY_REJECTED = 5
EXIT_TARGET_EXISTS = 6
EXIT_FETCH_FAILED = 7
EXIT_BAD_ARCHIVE = 8
EXIT_NO_SETUP_STEP = 9
EXIT_SETUP_UNRUNNABLE = 10
EXIT_SETUP_REFUSED = 11


def _usage() -> None:
    print("Usage: newt create <template> [directory] [--setup [-- kit arguments...]]")
    print("")
    print("  Fetch a starter kit at a pinned commit and unpack it into a directory")
    print("  that is yours: no git remote pointing back at us, no history of ours,")
    print("  and the driver versions the kit pinned already stamped.")
    print("")
    print("  Run 'newt create' with no arguments to list the templates your key can")
    print("  reach. Public kits need no key; private kits are served by the console")
    print("  against your nt_ key.")
    print("")
    print("  The directory defaults to the template name, and must not already exist")
    print("  with anything in it.")
    print("")
    print("  Every kit carries its own setup step at scripts/setup — that is what asks")
    print("  about your rig, and this command never does. By default it is printed as")
    print("  the next thing to run; --setup runs it here instead.")
    print("")
    print("Options:")
    print("  --no-git   Leave the directory inert — do not run 'git init' in it")
    print("  --setup    Run the kit's own setup step after unpacking")
    print("  --json     Emit machine-readable JSON (cannot be combined with --setup)")
    print("")
    print("  Any argument this command does not recognize is handed to the kit's setup")
    print("  step untouched, and asking for that implies --setup:")
    print("      newt create <template> <dir> --some-kit-flag value --non-interactive")
    print("  The first unrecognized argument starts the handoff; everything after it")
    print("  goes to the kit. Use '--' when the kit wants an argument this command also")
    print("  owns, such as --json:")
    print("      newt create <template> <dir> --setup -- --json --non-interactive")
    print("  Run the kit's setup step with --help to see what it accepts; it differs")
    print("  per kit, and this command has no list of them.")
    print("")
    print("Environment:")
    print("  NT_API_KEY      API key override (overrides ~/.nt/credentials)")
    print("  NT_CONSOLE_URL  Console URL (default: https://newtheory-console.vercel.app)")
    print("")
    print("Exit codes:")
    print("  0    the kit was written to the directory named in the output")
    print("  1    a usage error")
    print("  2    no template by that name — the available ones are printed")
    print("  3    that template is private and no key was found")
    print("  4    the console could not be reached")
    print("  5    the console rejected your key")
    print("  6    the directory already exists and is not empty")
    print("  7    the kit's archive could not be downloaded")
    print("  8    the downloaded archive was not the shape a starter kit has")
    print("  9    the kit landed, but it declares no setup step to run")
    print("  10   the kit landed, but its setup step could not be started")
    print("  11   the kit landed, and its setup step exited non-zero")


def _console_url() -> str:
    return os.environ.get("NT_CONSOLE_URL", _DEFAULT_CONSOLE).rstrip("/")


def _resolve_key() -> str | None:
    """NT_API_KEY wins, ~/.nt/credentials is the fallback (see newt._credentials)."""
    return os.environ.get("NT_API_KEY") or read_api_key()


def _fetch_registry(console: str, api_key: str | None, *, timeout: float = 10.0) -> object:
    """GET the console's template registry. Sent with the key when there is one, so the
    response can include the private kits this developer is entitled to."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = Request(f"{console}/api/cli/templates", headers=headers, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class RegistryUnavailable(Exception):
    """The console did not answer usefully.

    ``http_status`` separates "nothing on the other end of the wire" from "something
    answered and said no" — a developer behind a captive portal and a developer
    pointed at a console that has moved are looking at different problems.
    """

    def __init__(
        self, reason: str, *, key_rejected: bool = False, http_status: int | None = None
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.key_rejected = key_rejected
        self.http_status = http_status


def load_registry(console: str, api_key: str | None) -> tuple[list[Template], str]:
    """Return ``(templates, source)`` where source is ``"console"`` or ``"fallback"``.

    Raises ``RegistryUnavailable`` when the console cannot answer. The caller decides
    what that costs — for a public template it costs nothing, because the offline
    table can still resolve it; for a private one it is the end of the road.
    """
    try:
        payload = _fetch_registry(console, api_key)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise RegistryUnavailable(
                f"{exc.code} {exc.reason}", key_rejected=True, http_status=exc.code
            ) from exc
        raise RegistryUnavailable(f"{exc.code} {exc.reason}", http_status=exc.code) from exc
    except URLError as exc:
        raise RegistryUnavailable(str(exc.reason)) from exc
    except (TimeoutError, OSError) as exc:
        raise RegistryUnavailable(str(exc)) from exc

    try:
        return parse_templates(payload), "console"
    except ValueError as exc:
        # A console that answers with the wrong shape is broken, not offline. Say which.
        raise RegistryUnavailable(f"the console's template registry is malformed — {exc}") from exc


def _print_templates(templates: list[Template] | tuple[Template, ...], source: str) -> None:
    print("")
    print("Templates:")
    for t in templates:
        suffix = "   (private — needs your nt_ key)" if t.is_private else ""
        print(f"  {t.name}{suffix}")
    if source == "fallback":
        print("")
        print("  (listed from this CLI's built-in table — the console was not reachable,")
        print("   so a kit added since this version was released would not appear here)")


def _list_templates(console: str, api_key: str | None, as_json: bool) -> int:
    try:
        templates, source = load_registry(console, api_key)
    except RegistryUnavailable as exc:
        if exc.key_rejected:
            _say_key_rejected(console, exc)
            return EXIT_KEY_REJECTED
        templates, source = list(FALLBACK_TEMPLATES), "fallback"

    if as_json:
        print(
            json.dumps(
                {
                    "registry_source": source,
                    "templates": [
                        {
                            "name": t.name,
                            "visibility": t.visibility,
                            "repo": t.repo,
                            "ref": t.ref,
                        }
                        for t in templates
                    ],
                }
            )
        )
        return EXIT_OK

    _usage()
    _print_templates(templates, source)
    return EXIT_OK


# ---------------------------------------------------------------------------
# The refusals. Five causes, five strings — no two of these may ever collapse
# into one another, and the tests assert it.
# ---------------------------------------------------------------------------

def _say_no_such_template(name: str, templates, source: str) -> None:
    print(f"newt create: no template named {name!r}.", file=sys.stderr)
    available = ", ".join(names(templates)) or "(none)"
    print(f"        Available: {available}", file=sys.stderr)
    if source == "fallback":
        print(
            "        That list came from this CLI's built-in table — the console was not "
            "reachable, so a kit added recently may be missing from it.",
            file=sys.stderr,
        )
    print(f"        Fix: newt create {names(templates)[0] if names(templates) else '<template>'}", file=sys.stderr)


def _say_needs_key(name: str) -> None:
    print(
        f"newt create: the {name!r} starter kit is private, and no NT key was found.",
        file=sys.stderr,
    )
    print(
        "        It exists — you just need a key that is entitled to it. This is not a "
        "GitHub permission; the console serves the kit against your nt_ key.",
        file=sys.stderr,
    )
    print("        Fix: run `newt login`, or set NT_API_KEY.", file=sys.stderr)


def _say_console_unreachable(name: str, console: str, exc: RegistryUnavailable) -> None:
    if exc.http_status is not None:
        # Something answered and said no. That is a different problem from silence, and
        # the fix is different too — retrying a 404 forever is not the move.
        print(
            f"newt create: the console at {console} answered {exc.reason} for its "
            "template registry.",
            file=sys.stderr,
        )
        fix = (
            "        Fix: check NT_CONSOLE_URL points at a console that serves "
            "/api/cli/templates, or report it if that is the production console."
        )
    else:
        print(
            f"newt create: cannot reach the console at {console} — {exc.reason}.",
            file=sys.stderr,
        )
        fix = (
            "        Fix: check your network and try again, or set NT_CONSOLE_URL if you "
            "are pointed at a console that has moved."
        )
    print(
        f"        {name!r} is a private kit, and the console is the only thing that can "
        "serve it, so there is no offline path for this one.",
        file=sys.stderr,
    )
    print(fix, file=sys.stderr)


def _say_key_rejected(console: str, exc: RegistryUnavailable) -> None:
    print(f"newt create: the console rejected your key ({exc.reason}).", file=sys.stderr)
    print(
        "        The key was found and sent — it was refused. Rotate it in the console, "
        "or run `newt login` again.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Acquisition — the bytes, and where they land.
#
# A tarball at a pinned commit, never ``git clone``. Cloning would put the kit on
# disk with an ``origin`` pointing back at us, which is the exact thing this verb
# exists to stop: a starter kit is a project you own, and a directory that
# remembers where it came from is a checkout of ours wearing a project's clothes.
# ---------------------------------------------------------------------------

_CODELOAD = "https://codeload.github.com"


class FetchFailed(Exception):
    """The archive did not arrive.

    ``source`` is which door was knocked on — ``"github"`` for a public kit fetched
    direct, ``"console"`` for a private one brokered for us. They are different
    problems with different fixes and they never share a message. ``code`` carries
    the console's own machine-readable reason when it gave one.
    """

    def __init__(
        self,
        reason: str,
        *,
        source: str,
        http_status: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.source = source
        self.http_status = http_status
        self.code = code


def _read_url(req: Request, timeout: float) -> bytes:
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_public_tarball(repo: str, ref: str, *, timeout: float = 120.0) -> bytes:
    """A public kit comes straight from GitHub, unauthenticated.

    This is what makes the cold path work: a machine that has never seen an NT key,
    behind nothing but a network, can still get a starter kit. Routing it through
    the console would make the console a dependency of a flow that does not need one.
    """
    url = f"{_CODELOAD}/{repo}/tar.gz/{ref}"
    try:
        return _read_url(Request(url, method="GET"), timeout)
    except HTTPError as exc:
        raise FetchFailed(
            f"{exc.code} {exc.reason}", source="github", http_status=exc.code
        ) from exc
    except URLError as exc:
        raise FetchFailed(str(exc.reason), source="github") from exc
    except (TimeoutError, OSError) as exc:
        raise FetchFailed(str(exc), source="github") from exc


def _fetch_private_tarball(
    console: str, name: str, api_key: str, *, timeout: float = 120.0
) -> bytes:
    """A private kit comes through the console, against the developer's ``nt_`` key.

    The console holds the credential that can read the repository; the developer
    holds a key we issued. GitHub never enters their story, and the repository path
    never leaves the server.
    """
    url = f"{console}/api/cli/templates/{name}/tarball"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    try:
        return _read_url(req, timeout)
    except HTTPError as exc:
        # The console names its own causes. Read the code rather than guessing from
        # the status — 401 alone cannot tell "no key" from "not your key".
        code = None
        try:
            code = json.loads(exc.read()).get("code")
        except Exception:
            code = None
        raise FetchFailed(
            f"{exc.code} {exc.reason}", source="console", http_status=exc.code, code=code
        ) from exc
    except URLError as exc:
        raise FetchFailed(str(exc.reason), source="console") from exc
    except (TimeoutError, OSError) as exc:
        raise FetchFailed(str(exc), source="console") from exc


class BadArchive(Exception):
    """What arrived was not the shape a repository archive has."""


def _strip_root(tar: tarfile.TarFile) -> str:
    """GitHub archives wrap everything in one top-level directory named after the
    repo and the commit. That name is ours, not the developer's, so it is stripped
    — the kit's own files land directly in the directory they asked for.

    A single top-level entry is also the archive-shape check: anything else did not
    come from where we think it did, and unpacking it would scatter files.
    """
    roots = {pathlib.PurePosixPath(m.name).parts[0] for m in tar.getmembers() if m.name}
    if len(roots) != 1:
        raise BadArchive(
            f"expected one top-level directory, found {len(roots)}: "
            f"{', '.join(sorted(roots)) or '(nothing)'}"
        )
    return roots.pop()


def _unpack_tarball(archive: bytes, dest: pathlib.Path) -> int:
    """Unpack into ``dest``, returning the number of files written.

    Every member is checked before anything is written: a path that climbs out of
    ``dest``, or a symlink pointing out of it, is refused for the whole archive
    rather than skipped quietly. A partial unpack that silently dropped the file
    that mattered is worse than a refusal.
    """
    try:
        tar = tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz")
    except tarfile.TarError as exc:
        raise BadArchive(f"not a readable gzip tar archive — {exc}") from exc

    with tar:
        root = _strip_root(tar)
        members = []
        for m in tar.getmembers():
            parts = pathlib.PurePosixPath(m.name).parts
            if not parts or parts[0] != root or len(parts) == 1:
                continue
            rel = pathlib.PurePosixPath(*parts[1:])
            if rel.is_absolute() or ".." in rel.parts:
                raise BadArchive(f"archive member escapes the target directory: {m.name}")
            if not (m.isfile() or m.isdir() or m.issym()):
                raise BadArchive(f"archive member is not a file, directory, or symlink: {m.name}")
            if m.issym():
                link = pathlib.PurePosixPath(m.linkname)
                if link.is_absolute() or ".." in link.parts:
                    raise BadArchive(f"archive symlink points outside the kit: {m.name}")
            m.name = str(rel)
            members.append(m)

        dest.mkdir(parents=True, exist_ok=True)
        # tarfile's "data" filter (the safe default from 3.12) is belt to the braces
        # above; it does not exist on 3.11, which this package still supports, so it
        # is applied when present rather than assumed.
        extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        try:
            tar.extractall(dest, members=members, **extra)
        except tarfile.TarError as exc:
            raise BadArchive(f"archive could not be unpacked — {exc}") from exc

    return sum(1 for m in members if m.isfile())


def _git_init(dest: pathlib.Path) -> str:
    """Make the directory a repository with NO commits and NO remote.

    The choice, stated: ``git init`` and stop. A first commit made on a developer's
    behalf would be authored in their name for content they have not read, and
    ``git log`` on their own project would open on a message we wrote. Leaving the
    repository empty means the first commit in their history is genuinely the first
    thing they decided. Doing nothing at all was the other option; it costs them a
    ``git init`` and, more to the point, ``git -C <dir> remote -v`` on a non-repo is
    an error rather than the silence that proves this directory is theirs.

    Returns what actually happened, so the caller can print it. Never raises: git
    missing is a real state on a real machine, and it does not make the kit that
    just landed any less usable.
    """
    git = shutil.which("git")
    if git is None:
        return "unavailable: git is not on PATH"
    proc = subprocess.run(
        [git, "init", "--quiet", str(dest)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return f"unavailable: git init exited {proc.returncode}" + (
            f" — {detail[0]}" if detail else ""
        )
    return "initialized"


def _resolve_target(name: str, given: str | None) -> pathlib.Path:
    return pathlib.Path(given if given else name).expanduser()


def _say_target_exists(dest: pathlib.Path) -> None:
    print(
        f"newt create: {dest} already exists and is not empty.",
        file=sys.stderr,
    )
    print(
        "        Nothing was written. Unpacking on top of it would mix a fresh kit "
        "into whatever is already there, and you would not be able to tell which "
        "file came from where.",
        file=sys.stderr,
    )
    print(
        f"        Fix: name a directory that does not exist yet, or move {dest} aside.",
        file=sys.stderr,
    )


def _say_fetch_failed(name: str, console: str, template: Template, exc: FetchFailed) -> None:
    if exc.source == "github":
        where = f"https://github.com/{template.repo}"
        if exc.http_status is not None:
            print(
                f"newt create: GitHub answered {exc.reason} for the {name!r} starter kit "
                f"at its pinned commit.",
                file=sys.stderr,
            )
            print(
                f"        {where} is public, so this is not about your credentials — the "
                "pinned commit or the repository moved, which is ours to fix.",
                file=sys.stderr,
            )
            print("        Fix: run `newt upgrade`, and report it if it persists.", file=sys.stderr)
        else:
            print(
                f"newt create: could not download the {name!r} starter kit from GitHub — "
                f"{exc.reason}.",
                file=sys.stderr,
            )
            print(
                "        The kit was found in the registry; the download itself did not "
                "complete. Nothing was written.",
                file=sys.stderr,
            )
            print("        Fix: check your network and run the same command again.", file=sys.stderr)
        return

    if exc.code == "dispensing_unconfigured":
        print(
            f"newt create: the console cannot serve the {name!r} starter kit right now — "
            "it is missing the credential that reads it.",
            file=sys.stderr,
        )
        print(
            "        Your key is fine. This is ours, not yours.",
            file=sys.stderr,
        )
        print("        Fix: report it — nothing you can change on this machine helps.", file=sys.stderr)
        return

    if exc.http_status is not None:
        print(
            f"newt create: the console at {console} answered {exc.reason} while serving the "
            f"{name!r} starter kit.",
            file=sys.stderr,
        )
        print(
            "        The registry resolved the name, so the kit exists; handing over its "
            "bytes is what failed.",
            file=sys.stderr,
        )
        print("        Fix: try again, and report it if it persists.", file=sys.stderr)
        return

    print(
        f"newt create: lost the connection to the console at {console} while downloading the "
        f"{name!r} starter kit — {exc.reason}.",
        file=sys.stderr,
    )
    print("        Nothing was written.", file=sys.stderr)
    print("        Fix: check your network and run the same command again.", file=sys.stderr)


def _say_bad_archive(name: str, exc: BadArchive) -> None:
    print(
        f"newt create: the archive for the {name!r} starter kit is not the shape a "
        f"repository archive has — {exc}.",
        file=sys.stderr,
    )
    print(
        "        Nothing was unpacked. This is ours, not yours: either the pinned "
        "commit points at something unexpected, or the download was corrupted in "
        "transit.",
        file=sys.stderr,
    )
    print("        Fix: run the same command again; report it if it repeats.", file=sys.stderr)


# ---------------------------------------------------------------------------
# The handoff — the kit's setup step, and the arguments this verb refuses to read.
#
# The kit is where rig knowledge lives, so the kit is where the asking happens.
# A template declares its setup step by shipping one: an executable ``scripts/setup``
# at its root. That is the whole declaration — no manifest, no schema, no version
# negotiation. The cost of that choice is one file: a kit whose real entry point is
# somewhere else adds a shim at that path. What it buys is that adding a starter kit
# is still a registry row and never an SDK release, and that nothing about a kit's
# setup has a second copy over here to drift.
#
# Arguments are forwarded verbatim. This verb does not know what any of them mean
# and must never learn: the moment it parses one — an arm's address, a camera's
# serial, how many of either there are — a fact about one rig has been compiled
# into a library that is supposed to work for every rig.
# ---------------------------------------------------------------------------

_SETUP_REL = ("scripts", "setup")
_SETUP_DISPLAY = "./" + "/".join(_SETUP_REL)


class SetupUnrunnable(Exception):
    """The setup step is declared but the OS would not start it."""


def _setup_step(dest: pathlib.Path) -> pathlib.Path | None:
    path = dest.joinpath(*_SETUP_REL)
    return path if path.is_file() else None


def _run_setup(dest: pathlib.Path, script: pathlib.Path, forwarded: list[str]) -> int:
    """Run the kit's setup step in the kit's directory and return its exit code.

    Standard streams are inherited, so what a developer reads is the kit's own
    report — including where it put its configuration. This verb does not choose
    that location, does not write the file, and does not paraphrase what happened.
    """
    if not os.access(script, os.X_OK):
        raise SetupUnrunnable("it is not executable")
    print("")
    print(f"Running the kit's setup step: {' '.join([_SETUP_DISPLAY, *forwarded])}")
    print("  Everything below is the kit's own output, including where it writes its")
    print("  configuration — that is the kit's choice, not this command's.")
    print("")
    sys.stdout.flush()
    try:
        proc = subprocess.run([str(script), *forwarded], cwd=str(dest))
    except OSError as exc:
        raise SetupUnrunnable(str(exc)) from exc
    return proc.returncode


def _say_no_setup_step(name: str, dest: pathlib.Path, forwarded: list[str]) -> None:
    print(
        f"newt create: the {name!r} starter kit declares no setup step to run.",
        file=sys.stderr,
    )
    print(
        f"        A kit declares one by shipping an executable {_SETUP_DISPLAY}, and "
        f"{dest} has no such file. Not every kit has one.",
        file=sys.stderr,
    )
    if forwarded:
        print(
            f"        Nothing consumed these, so they were not applied: {' '.join(forwarded)}",
            file=sys.stderr,
        )
    print(
        f"        The kit itself is on disk and intact — this is only about the step "
        f"you asked to run. Fix: read {dest}/README.md for what this kit wants next.",
        file=sys.stderr,
    )


def _say_setup_unrunnable(name: str, dest: pathlib.Path, reason: str) -> None:
    print(
        f"newt create: the {name!r} kit's setup step at {dest / pathlib.Path(*_SETUP_REL)} "
        f"could not be started — {reason}.",
        file=sys.stderr,
    )
    print(
        "        It was never launched, so nothing it does has happened. The kit "
        "itself is on disk and intact.",
        file=sys.stderr,
    )
    print(
        f"        Fix: chmod +x {dest / pathlib.Path(*_SETUP_REL)} and run it yourself, "
        "and report it — an unrunnable setup step is the kit's bug, not yours.",
        file=sys.stderr,
    )


def _say_setup_refused(name: str, dest: pathlib.Path, code: int, forwarded: list[str]) -> None:
    print(
        f"newt create: the {name!r} kit's setup step ran and exited {code}.",
        file=sys.stderr,
    )
    print(
        "        Its own output above says what it wanted; this command forwards "
        "arguments to it and does not read them, so it cannot say more than that.",
        file=sys.stderr,
    )
    print(
        f"        The kit is on disk and intact — do not run newt create again. Fix "
        f"what it named, then run it directly:",
        file=sys.stderr,
    )
    print(
        f"            cd {dest} && {' '.join([_SETUP_DISPLAY, *forwarded])}",
        file=sys.stderr,
    )


def _print_setup_next(dest: pathlib.Path, script: pathlib.Path | None) -> None:
    """The default path: say what the next step is instead of taking it.

    Running a script that arrived seconds ago, which writes configuration outside
    the directory that was asked for, is not something a "make me a directory"
    command should do without being told to.
    """
    print("")
    if script is None:
        print("  This kit ships no setup step. What it wants next is in its README:")
        print(f"      {dest}/README.md")
        return
    print("  Next — the kit's own setup step, which is the part that knows your rig:")
    print(f"      cd {dest} && {_SETUP_DISPLAY}")
    print(f"  Run it with --help first to see what it asks for. Add --setup to")
    print("  'newt create' to have it run right after unpacking instead.")


def _handoff(
    name: str, dest: pathlib.Path, run_it: bool, forwarded: list[str]
) -> int:
    script = _setup_step(dest)

    if not run_it:
        _print_setup_next(dest, script)
        return EXIT_OK

    if script is None:
        _say_no_setup_step(name, dest, forwarded)
        return EXIT_NO_SETUP_STEP

    try:
        code = _run_setup(dest, script, forwarded)
    except SetupUnrunnable as exc:
        _say_setup_unrunnable(name, dest, str(exc))
        return EXIT_SETUP_UNRUNNABLE

    if code != 0:
        _say_setup_refused(name, dest, code, forwarded)
        return EXIT_SETUP_REFUSED
    return EXIT_OK


def _acquire(
    template: Template,
    name: str,
    console: str,
    api_key: str | None,
    dest: pathlib.Path,
    *,
    do_git: bool,
    as_json: bool,
    run_setup: bool,
    forwarded: list[str],
) -> int:
    """Fetch, unpack, and say plainly what landed and where."""
    if dest.exists() and any(dest.iterdir()):
        _say_target_exists(dest)
        return EXIT_TARGET_EXISTS

    try:
        if template.is_private:
            # Guarded by the caller: a private kit with no key never gets this far.
            archive = _fetch_private_tarball(console, name, api_key or "")
        else:
            archive = _fetch_public_tarball(template.repo or "", template.ref or "")
    except FetchFailed as exc:
        # The console's own vocabulary for "who are you" outranks the transport status:
        # 401 alone cannot tell an absent key from a rejected one.
        if exc.code == "needs_key":
            _say_needs_key(name)
            return EXIT_NEEDS_KEY
        if exc.code == "key_rejected":
            print(
                f"newt create: the console rejected your key ({exc.reason}).", file=sys.stderr
            )
            print(
                "        The key was found and sent — it was refused. Rotate it in the "
                "console, or run `newt login` again.",
                file=sys.stderr,
            )
            return EXIT_KEY_REJECTED
        _say_fetch_failed(name, console, template, exc)
        return EXIT_FETCH_FAILED

    try:
        files = _unpack_tarball(archive, dest)
    except BadArchive as exc:
        _say_bad_archive(name, exc)
        return EXIT_BAD_ARCHIVE

    git_state = _git_init(dest) if do_git else "skipped: --no-git"

    if as_json:
        # --json never runs the setup step (refused up front in cmd_create), so this
        # payload describes a directory and names the command an agent runs next —
        # two clean machine-readable steps rather than one with a kit's report
        # spliced into the middle of it.
        script = _setup_step(dest)
        print(
            json.dumps(
                {
                    "template": name,
                    "repo": template.repo,
                    "ref": template.ref,
                    "directory": str(dest),
                    "files": files,
                    "git": git_state,
                    "remote": None,
                    "setup": {
                        "declared": "/".join(_SETUP_REL) if script else None,
                        "ran": False,
                        "next": f"cd {dest} && {_SETUP_DISPLAY}" if script else None,
                    },
                }
            )
        )
        return EXIT_OK

    ref_note = f" at {template.ref[:12]}" if template.ref else ""
    print(f"Created {dest} — the {name} starter kit{ref_note}, {files} files.")
    print("")
    print("  It's yours: no remote, no history of ours. The driver versions this kit")
    print("  pinned came with it, so they match the firmware it was tested against.")
    if git_state == "initialized":
        print("")
        print("  An empty git repository was initialized — no commits, no remote. Your")
        print("  first commit is yours to make:")
        print(f"      git -C {dest} add -A && git -C {dest} commit -m 'Initial commit'")
    elif git_state.startswith("unavailable"):
        print("")
        print(f"  NOT a git repository: {git_state[len('unavailable: '):]}.")
        print(f"      Run 'git init' in {dest} yourself once git is available.")
    else:
        print("")
        print(f"  NOT a git repository (--no-git). Run 'git init {dest}' if you want one.")

    return _handoff(name, dest, run_setup, forwarded)


_OWN_FLAGS = ("--json", "--no-git", "--setup", "-h", "--help")


class _Args:
    """What this verb read, and what it refused to read.

    ``forwarded`` is the second half of the command line — the kit's half. The
    split is positional: the first argument this verb does not recognize ends its
    own reading and everything from there on belongs to the kit, and ``--`` forces
    that split early for a kit that wants an argument this verb also owns. The rule
    is one sentence long on purpose; a cleverer one would need this verb to know
    which forwarded flags take values, which is knowing about rigs.
    """

    def __init__(self) -> None:
        self.help = False
        self.as_json = False
        self.do_git = True
        self.setup = False
        self.positional: list[str] = []
        self.forwarded: list[str] = []


def _parse(args: list[str]) -> _Args:
    parsed = _Args()
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            parsed.forwarded = args[i + 1 :]
            break
        if a in _OWN_FLAGS:
            if a in ("-h", "--help"):
                parsed.help = True
            elif a == "--json":
                parsed.as_json = True
            elif a == "--no-git":
                parsed.do_git = False
            else:
                parsed.setup = True
            i += 1
            continue
        if a.startswith("-"):
            parsed.forwarded = args[i:]
            break
        parsed.positional.append(a)
        i += 1
    return parsed


def _say_too_many_positionals(positional: list[str]) -> None:
    print(
        f"newt create: takes a template and at most one directory; got "
        f"{len(positional)} arguments ({', '.join(positional)}).",
        file=sys.stderr,
    )
    print(
        "        Nothing was written. Arguments for the kit's setup step start at the "
        "first flag, so a bare word after the directory belongs to nothing.",
        file=sys.stderr,
    )
    print(
        f"        Fix: newt create {positional[0]} {positional[1]} "
        f"-- {' '.join(positional[2:])}   (if those were meant for the kit)",
        file=sys.stderr,
    )


def _say_json_cannot_run_setup() -> None:
    print(
        "newt create: --json cannot be combined with running the kit's setup step.",
        file=sys.stderr,
    )
    print(
        "        --json is this command's report on stdout, and the kit's setup step "
        "writes its own report to the same place; splicing them would give you "
        "neither.",
        file=sys.stderr,
    )
    print(
        "        Nothing was written. Fix: run them as two steps — 'newt create "
        "<template> <dir> --json' names the setup command in its output, then run "
        "that command yourself.",
        file=sys.stderr,
    )


def _say_no_template_named(forwarded: list[str]) -> None:
    print(
        f"newt create: no template named — the arguments given "
        f"({' '.join(forwarded)}) are for a kit's setup step, and no kit was asked for.",
        file=sys.stderr,
    )
    print(
        "        Nothing was written. Fix: put the template name first — "
        "newt create <template> [directory] "
        f"{' '.join(forwarded)}",
        file=sys.stderr,
    )
    print("        Run 'newt create' with no arguments to see the templates.", file=sys.stderr)


def cmd_create(args: list[str]) -> int:
    parsed = _parse(args)
    if parsed.help:
        _usage()
        return EXIT_OK

    as_json = parsed.as_json
    do_git = parsed.do_git
    positional = parsed.positional
    forwarded = parsed.forwarded
    # Handing the kit arguments is asking for it to run. A flag that was accepted
    # and then quietly dropped because --setup was missing is the silent kind of
    # failure this verb is not allowed to have.
    run_setup = parsed.setup or bool(forwarded)

    console = _console_url()
    api_key = _resolve_key()

    if not positional:
        if forwarded:
            _say_no_template_named(forwarded)
            return EXIT_USAGE
        if parsed.setup:
            print(
                "newt create: --setup runs a kit's setup step, and no kit was named.",
                file=sys.stderr,
            )
            print(
                "        Nothing was written. Fix: newt create <template> [directory] "
                "--setup, or drop --setup to list the templates.",
                file=sys.stderr,
            )
            return EXIT_USAGE
        return _list_templates(console, api_key, as_json)

    if len(positional) > 2:
        _say_too_many_positionals(positional)
        return EXIT_USAGE

    if as_json and run_setup:
        _say_json_cannot_run_setup()
        return EXIT_USAGE

    name = positional[0]
    dest = _resolve_target(name, positional[1] if len(positional) > 1 else None)

    try:
        templates, source = load_registry(console, api_key)
    except RegistryUnavailable as exc:
        if exc.key_rejected:
            _say_key_rejected(console, exc)
            return EXIT_KEY_REJECTED
        # The console is down. Public kits still have somewhere to come from; private
        # ones do not, and saying "no such template" here would be a lie.
        fallback = find(FALLBACK_TEMPLATES, name)
        if fallback is None:
            _say_no_such_template(name, FALLBACK_TEMPLATES, "fallback")
            return EXIT_NO_SUCH_TEMPLATE
        if fallback.is_private:
            _say_console_unreachable(name, console, exc)
            return EXIT_CONSOLE_UNREACHABLE
        templates, source = list(FALLBACK_TEMPLATES), "fallback"

    template = find(templates, name)
    if template is None:
        _say_no_such_template(name, templates, source)
        return EXIT_NO_SUCH_TEMPLATE

    if template.is_private and not api_key:
        # No point asking the console to broker a private kit for someone who has not
        # said who they are — and "you need a key" is a better sentence than whatever
        # a 401 would come back with.
        _say_needs_key(name)
        return EXIT_NEEDS_KEY

    if not template.is_private and not (template.repo and template.ref):
        # A public row with nowhere to fetch from is a broken registry, not a missing
        # template — and a developer must not be told to go find a key for it.
        print(
            f"newt create: the registry lists {name!r} as public but gives no repository "
            "and pinned commit to fetch it from.",
            file=sys.stderr,
        )
        print("        That is ours, not yours — nothing was written.", file=sys.stderr)
        print("        Fix: report it; `newt upgrade` may already carry the correction.", file=sys.stderr)
        return EXIT_FETCH_FAILED

    return _acquire(
        template,
        name,
        console,
        api_key,
        dest,
        do_git=do_git,
        as_json=as_json,
        run_setup=run_setup,
        forwarded=forwarded,
    )
