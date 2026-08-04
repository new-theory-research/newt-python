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
"""
from __future__ import annotations

import json
import os
import sys
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
EXIT_NOT_ACQUIRED = 70


def _usage() -> None:
    print("Usage: newt create <template> [directory]")
    print("")
    print("  Fetch a starter kit at a pinned commit and unpack it into a directory")
    print("  that is yours: no git remote pointing back at us, no history of ours,")
    print("  and the driver versions the kit pinned already stamped.")
    print("")
    print("  Run 'newt create' with no arguments to list the templates your key can")
    print("  reach. Public kits need no key; private kits are served by the console")
    print("  against your nt_ key.")
    print("")
    print("Options:")
    print("  --json   Emit machine-readable JSON")
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


def _console_url() -> str:
    return os.environ.get("NT_CONSOLE_URL", _DEFAULT_CONSOLE).rstrip("/")


def _resolve_key() -> str | None:
    """NT_API_KEY wins, ~/.nt/credentials is the fallback (see newt._credentials)."""
    return os.environ.get("NT_API_KEY") or read_api_key()


def _fetch_registry(console: str, api_key: str | None, *, timeout: float = 10.0) -> object:
    """GET the console's template registry. Sent with the key when there is one, so the
    response can include the private kits this developer is entitled to."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = Request(f"{console}/api/templates", headers=headers, method="GET")
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
            "/api/templates, or report it if that is the production console."
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


def cmd_create(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return EXIT_OK

    as_json = "--json" in args
    positional = [a for a in args if not a.startswith("-")]

    console = _console_url()
    api_key = _resolve_key()

    if not positional:
        return _list_templates(console, api_key, as_json)

    name = positional[0]

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

    if template.is_private and not template.repo:
        # The console answered and did not hand over a fetchable location for this kit:
        # either no key was sent, or the key that was sent is not entitled to it.
        _say_needs_key(name)
        return EXIT_NEEDS_KEY

    print(
        f"newt create: resolved {name!r} to {template.repo}@{(template.ref or '')[:12]}, "
        "and stopped — this build does not fetch and unpack yet.",
        file=sys.stderr,
    )
    return EXIT_NOT_ACQUIRED
