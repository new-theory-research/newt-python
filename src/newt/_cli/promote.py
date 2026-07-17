"""newt promote — keep a fine-tune's checkpoint band and serve it, from the terminal.

    newt promote <job-handle> --band <token>          register the band as a model
    newt promote <job-handle> --band <token> --json    machine-readable body passthrough

The morning-after verb. A run you trained overnight has checkpoint bands, each
evaluated; the payoff is *pick one and serve it*. This is the CLI twin of the
console's promote button (console-002): the SAME route, the SAME contract, one line.
The registered model is born ``pending``; the normal admission chain takes it live.
The output names the model and says plainly what happens next — a safety check is
running, and ``newt models`` is where you watch it.

The load-bearing honesty (issue #28): every server refusal prints its reason
VERBATIM. A band whose eval hasn't landed, a checkpoint whose location the training
pipeline hasn't reported yet (nt-runway#25), a run already promoted (here's its
existing model) — each 409's plain ``detail`` reaches the developer word-for-word,
never collapsed into a generic "promote failed."

``--band`` is passed VERBATIM. The route matches it against the stored eval-snapshot
band (a zero-padded step like ``010000``); the CLI does NOT guess a pad width — a
mismatch surfaces the route's honest 409 rather than a client-side reshape (the
band-normalization contract is an open question, flagged in the card, not guessed
here).

Featherweight, same as the rest of the CLI: stdlib ``urllib`` only, an ``nt_`` key on
the ``Authorization: Bearer`` header. The console's promote route learned to accept a
Bearer key beside its browser session (this card's portal half) so this verb can reach
it at all.
"""
from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

# Reuse the shipped finetune.py helpers verbatim (Rule 2, Rule 6) — the key resolver,
# the flag scanner, the console-URL reader, and the terminal-color helper are the one
# canonical copy; promote does not fork them.
from newt._cli.finetune import (
    _c,
    _console_url,
    _GREEN,
    _opt_value,
    _resolve_key,
)


def _usage() -> None:
    print("Usage: newt promote <job-handle> --band <token> [--json]")
    print("")
    print("  Keep a fine-tune's checkpoint band and serve it — the CLI twin of the")
    print("  console's promote button. The band is registered as a model born")
    print("  `pending`; the normal admission chain takes it live. Watch it with")
    print("  `newt models`.")
    print("")
    print("Arguments:")
    print("  <job-handle>     The run whose checkpoint you want to serve. List your")
    print("                   runs with `newt finetune --list`.")
    print("")
    print("Options:")
    print("  --band <token>   Which checkpoint band to serve — the evaluated step,")
    print("                   passed verbatim (e.g. 010000). Required.")
    print("  --json           Emit the route's JSON response body on stdout")
    print("                   (the registered model, or the server's refusal detail).")
    print("")
    print("Environment:")
    print("  NT_API_KEY      API key override (overrides ~/.nt/credentials).")
    print("  NT_CONSOLE_URL  Console URL (default: https://newtheory-console.vercel.app)")


def _first_positional(args: list[str]) -> str | None:
    """The first non-flag argument (the job handle). Skips flags and the value that
    follows a space-form value flag like ``--band X`` — so ``newt promote --band 010000
    fc-abc`` still finds ``fc-abc`` as the handle, and ``--band=010000`` (which carries
    its own value) never swallows the positional after it."""
    skip_next = False
    value_flags = {"--band"}
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a.startswith("-"):
            if a in value_flags:
                skip_next = True
            continue
        return a
    return None


# ---------------------------------------------------------------------------
# The one HTTP round-trip — split out so tests exercise the render matrix without a
# network. Mirrors finetune.py's `_launch`: stdlib urllib, Bearer key, JSON body.
# A 201 returns the parsed body; any 4xx/5xx raises HTTPError (handled by the caller).
# ---------------------------------------------------------------------------
def _promote(
    console: str,
    api_key: str,
    job_handle: str,
    band: str,
    *,
    timeout: float = 30.0,
) -> dict:
    url = f"{console}/api/finetune/runs/{quote(job_handle, safe='')}/promote"
    req = Request(
        url,
        data=json.dumps({"band": band}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _error_body(exc: HTTPError) -> dict | None:
    """The route's JSON error body as a dict, or None when there isn't one.

    Read ONCE — both the verbatim ``detail`` (the #28 human surface) and the full body
    (the ``--json`` passthrough) come from this single read, so an agent gets
    ``detail``/``uid`` structured while a human gets the plain reason. (finetune.py's
    ``_http_error_detail`` reads-and-consumes the stream and returns only ``detail``; it
    can't also yield the whole body, so promote reads the body itself here — same
    verbatim-surfacing behavior, one read.)"""
    try:
        raw = exc.read()
    except Exception:  # noqa: BLE001 - a body we can't read is just "no detail"
        return None
    if not raw:
        return None
    try:
        body = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return body if isinstance(body, dict) else None


def _explain_promote_http_error(
    exc: HTTPError, handle: str, band: str, *, as_json: bool
) -> int:
    """Render a promote refusal. The #28 seam: every 409 (and 400/503 when the server
    names a reason) prints the route's ``detail`` VERBATIM plus the right next step;
    ``--json`` passes the whole error body through on stdout so an agent gets the
    structured ``detail``/``uid``. Human/instructional prose always goes to stderr."""
    body = _error_body(exc)
    detail = body.get("detail") if isinstance(body, dict) else None
    error = body.get("error") if isinstance(body, dict) else None

    # Structured passthrough for agents — the route's own body, unrewritten.
    if as_json and body is not None:
        print(json.dumps(body))

    code = exc.code

    if code == 409:
        # The whole point of the verb: the server's plain reason, word-for-word.
        if detail:
            print(f"newt promote: {detail}", file=sys.stderr)
        else:
            # No detail body — name the error code honestly rather than swallow it.
            print(
                f"newt promote: the console refused this promote ({error or 'conflict'}).",
                file=sys.stderr,
            )
        if error == "already_promoted":
            uid = body.get("uid") if isinstance(body, dict) else None
            if uid:
                print(f"  Existing model: {uid}", file=sys.stderr)
            print("  See it with:  newt models", file=sys.stderr)
        else:
            print("  Check the run's bands with:  newt finetune --list", file=sys.stderr)
        return 1

    if code == 404:
        # No cross-team oracle (#28): an unknown handle and a not-yours handle are the
        # SAME 404 — never confirm someone else's run exists.
        print(
            f"newt promote: no run with handle {handle!r} in your catalog.",
            file=sys.stderr,
        )
        print("        Fix: newt finetune --list", file=sys.stderr)
        return 1

    if code == 400:
        if detail:
            print(f"newt promote: {detail}", file=sys.stderr)
        else:
            print(f"newt promote: {band!r} isn't a valid band token.", file=sys.stderr)
        print("        Fix: newt promote <job-handle> --band <n>", file=sys.stderr)
        return 1

    if code == 401:
        print(
            "newt promote: authentication failed — your key was missing or rejected.",
            file=sys.stderr,
        )
        print("  Run `newt login` to authenticate, or set NT_API_KEY.", file=sys.stderr)
        return 1

    if code == 503:
        # A real server-side write failure — surface it, never paper over it (Rule 10).
        if detail:
            print(f"newt promote: {detail}", file=sys.stderr)
        else:
            print(
                f"newt promote: the console couldn't register this model right now ({code}).",
                file=sys.stderr,
            )
        print("  This is server-side — retry shortly.", file=sys.stderr)
        return 1

    # An unexpected code — surface the raw envelope honestly, never guess a meaning.
    print(f"newt promote: promote failed ({code}): {exc.reason}", file=sys.stderr)
    if detail:
        print(f"  {detail}", file=sys.stderr)
    return 1


def cmd_promote(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    as_json = "--json" in args
    # Instructional/progress output goes to stderr in --json mode so stdout carries
    # nothing but the route's JSON body (composable with $(...) / jq).
    out = sys.stderr if as_json else sys.stdout

    handle = _first_positional(args)
    band = _opt_value(args, "--band")

    if not handle:
        print(
            "newt promote: a job handle is required — which run's checkpoint to serve.",
            file=sys.stderr,
        )
        print("        Fix: newt promote <job-handle> --band <n>", file=sys.stderr)
        return 1
    if not band:
        print(
            "newt promote: --band <token> is required — which checkpoint band to serve.",
            file=sys.stderr,
        )
        print("        Fix: newt promote <job-handle> --band <n>", file=sys.stderr)
        return 1

    api_key = _resolve_key()
    if not api_key:
        print(
            "newt: no API key found.\n"
            "  Run `newt login` to authenticate, or set NT_API_KEY.",
            file=sys.stderr,
        )
        return 1

    console = _console_url()

    try:
        body = _promote(console, api_key, handle, band)
    except HTTPError as exc:
        return _explain_promote_http_error(exc, handle, band, as_json=as_json)
    except URLError as exc:
        print(f"newt promote: cannot reach {console}: {exc.reason}", file=sys.stderr)
        print("  Set NT_CONSOLE_URL if you're running a local console.", file=sys.stderr)
        return 1

    # 201 — the band is registered, born `pending`. `--json` passes the route body
    # through verbatim on stdout; the human lines go to `out` (stderr in --json mode).
    if as_json:
        print(json.dumps(body))

    tag = body.get("tag") if isinstance(body, dict) else None
    status = body.get("status") if isinstance(body, dict) else None

    if not tag:
        # A 201 must carry the server-minted tag; without it we can't name the model.
        # Surface the raw body rather than fabricate a name (Rule 10).
        print(
            f"newt promote: promote succeeded but the response carried no model tag: {body!r}",
            file=sys.stderr,
        )
        return 1

    print(_c(_GREEN, "Promoted — your checkpoint is registered as a model."), file=out)
    print(f"  model:   {tag}", file=out)
    print(f"  status:  {status or 'pending'}", file=out)
    print("", file=out)
    print(
        "  Safety check running — usually live in a few minutes; watch with `newt models`.",
        file=out,
    )
    return 0
