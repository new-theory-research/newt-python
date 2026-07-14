"""newt finetune — launch a training run on NT's GPUs with your key, and watch it.

    newt finetune --dataset <name>       launch a run, then poll it to completion
    newt finetune --handle <job>         re-attach to a run already launched
    newt finetune --handle <job> --status   print the run's state ONCE and exit
    newt finetune --dataset <name> --json   machine-readable handle + terminal state

`--dataset` POSTs to the NT console's ``/api/finetune`` endpoint, authenticated by
your ``nt_`` key. The console launches the Modal training job **server-side, under
NT's own Modal credentials**, and hands back a job handle — no Modal credential ever
reaches this client (training spec §3.1). The CLI then polls ``/api/finetune/status``
against that handle until the run reaches a terminal state:

    succeeded → the model tag + a pointer to its report card
    failed    → the pipeline gate that failed, NAMED (Rule 10)

Ctrl-C never loses the run: the handle is printed at launch, and
``newt finetune --handle <job>`` re-attaches to it.

Featherweight on purpose, same as the rest of the CLI: stdlib ``urllib`` only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from newt._credentials import read_api_key

# ANSI colors — same semantic roles as the sibling verbs.
_RESET = "\033[0m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_GRAY = "\033[90m"

_DEFAULT_CONSOLE = "https://newtheory-console.vercel.app"

# Poll cadence. Training runs are long (hours); the interval is the console/Modal
# poll rate, the deadline a backstop just past the pipeline's own ~11h timeout so the
# CLI never hangs forever on a wedged run.
_POLL_INTERVAL_S = 15.0
_MAX_WAIT_S = 12 * 3600
_HEARTBEAT_EVERY_S = 60.0

# Terminal states the poll loop stops on — must match training/finetune_pipeline.py's
# STATUS_* constants (the Modal side that mints them) and lib/finetune-status.ts.
_TERMINAL = ("succeeded", "failed")


def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


def _console_url() -> str:
    return os.environ.get("NT_CONSOLE_URL", _DEFAULT_CONSOLE).rstrip("/")


def _resolve_key() -> str | None:
    return os.environ.get("NT_API_KEY") or read_api_key()


def _opt_value(args: list[str], name: str) -> str | None:
    """Value for ``--name X`` or ``--name=X``. Returns None if the flag is absent or
    its value is missing / looks like another flag (so ``--dataset --json`` is caught
    as a missing value, not a dataset literally named ``--json``)."""
    for i, a in enumerate(args):
        if a == name:
            nxt = args[i + 1] if i + 1 < len(args) else None
            return nxt if (nxt and not nxt.startswith("-")) else None
        if a.startswith(name + "="):
            return a[len(name) + 1 :] or None
    return None


def _usage() -> None:
    print("Usage: newt finetune (--dataset <name> | --handle <job>) [--status] [--json]")
    print("")
    print("  Launch a training run on NT's GPUs with your key, then watch it to")
    print("  completion. The launch happens server-side under NT's Modal credentials —")
    print("  no Modal credential ever touches this client.")
    print("")
    print("Options:")
    print("  --dataset <name>  The staged dataset to fine-tune on. Launches a new run.")
    print("  --handle <job>    Re-attach to a run already launched (poll only).")
    print("  --status          With --handle: print the run's current state once and")
    print("                    exit — a one-shot check, no blocking watch.")
    print("  --json            Emit machine-readable JSON (handle + terminal status/tag).")
    print("")
    print("Environment:")
    print("  NT_API_KEY      API key override (overrides ~/.nt/credentials).")
    print("  NT_CONSOLE_URL  Console URL (default: https://newtheory-console.vercel.app)")


# ---------------------------------------------------------------------------
# HTTP round-trips (one each) — split out so tests exercise the orchestration and
# rendering without a network.
# ---------------------------------------------------------------------------
def _launch(console: str, api_key: str, dataset: str, *, timeout: float = 30.0) -> dict:
    req = Request(
        f"{console}/api/finetune",
        data=json.dumps({"dataset": dataset}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _poll_status(console: str, api_key: str, job_handle: str, *, timeout: float = 30.0) -> dict:
    url = f"{console}/api/finetune/status?job_handle={quote(job_handle, safe='')}"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Terminal rendering — pure: (status dict) -> (human text, exit code).
# ---------------------------------------------------------------------------
def _render_terminal(status: dict) -> tuple[str, int]:
    state = status.get("status")
    if state == "succeeded":
        tag = status.get("tag")
        report_card = status.get("report_card")
        lines = [_c(_GREEN, "Fine-tune succeeded.")]
        # tag/report-card are surfaced from the run, never fabricated — a run that
        # hasn't produced one yet says so plainly rather than printing a fake handle.
        lines.append(f"  your model:   {tag if tag else '(pending — not yet assigned)'}")
        lines.append(f"  report card:  {report_card if report_card else '(pending)'}")
        # The tag is only useful if the developer knows how to point at it. Print the
        # exact SDK call — model= is the escape hatch, and their own fine-tune is the
        # one legitimate reason to reach for it.
        if tag:
            lines.append("")
            lines.append("  Drive it from Python:")
            lines.append("      from newt import Robot")
            lines.append(f'      robot = Robot(model="{tag}")')
        return "\n".join(lines), 0

    if state == "failed":
        gate = status.get("gate")
        head = f"Fine-tune failed at gate: {gate}" if gate else "Fine-tune failed."
        return _c(_RED, head), 1

    # Anything else reaching here is a contract violation — surface it, don't guess.
    return _c(_RED, f"Fine-tune ended in an unexpected state: {status!r}"), 1


def _terminal_json(job_handle: str, status: dict) -> str:
    return json.dumps(
        {
            "job_handle": job_handle,
            "status": status.get("status"),
            "gate": status.get("gate"),
            "tag": status.get("tag"),
            "report_card": status.get("report_card"),
        }
    )


def cmd_finetune(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    as_json = "--json" in args
    as_status = "--status" in args
    dataset = _opt_value(args, "--dataset")
    handle = _opt_value(args, "--handle")

    # Instructional/progress output goes to stderr in --json mode so stdout carries
    # nothing but the final JSON object (composable with $(...) / jq).
    out = sys.stderr if as_json else sys.stdout

    if dataset and handle:
        print("newt finetune: pass --dataset OR --handle, not both.", file=sys.stderr)
        return 1
    if as_status and not handle:
        print("newt finetune: --status needs --handle <job> — the run to check on.", file=sys.stderr)
        print("        Fix: newt finetune --handle <job> --status", file=sys.stderr)
        return 1
    if not dataset and not handle:
        print("newt finetune: --dataset <name> is required (or --handle <job> to re-attach).", file=sys.stderr)
        print("        Fix: newt finetune --dataset my-task", file=sys.stderr)
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

    # --- one-shot status check (no watch) -----------------------------------
    if as_status:
        return _status_once(console, api_key, handle, as_json=as_json)

    # --- launch (unless re-attaching to an existing handle) -----------------
    job_handle = handle
    if dataset:
        try:
            launched = _launch(console, api_key, dataset)
        except HTTPError as exc:
            _explain_launch_http_error(exc, dataset)
            return 1
        except URLError as exc:
            print(f"newt finetune: cannot reach {console}: {exc.reason}", file=sys.stderr)
            print("  Set NT_CONSOLE_URL if you're running a local console.", file=sys.stderr)
            return 1

        job_handle = launched.get("job_handle") if isinstance(launched, dict) else None
        if not job_handle:
            print(f"newt finetune: launch response carried no job handle: {launched!r}", file=sys.stderr)
            return 1

        print(f"Launched fine-tune on dataset {dataset!r}.", file=out)
        print(f"  job handle:   {_c(_GRAY, job_handle)}", file=out)
        print(f"  watch page:   {console}/runs/{job_handle}", file=out)
        print(f"  check later:  newt finetune --handle {job_handle} --status", file=out)
        print(f"  re-attach:    newt finetune --handle {job_handle}", file=out)
        print("", file=out)

    print(f"Watching {job_handle} … (Ctrl-C to stop watching; the run keeps going)", file=out, flush=True)

    # --- poll to a terminal state -------------------------------------------
    try:
        status = _watch(console, api_key, job_handle, out=out)
    except KeyboardInterrupt:
        print(
            f"\nStopped watching. The run is still going — re-attach with:\n"
            f"  newt finetune --handle {job_handle}",
            file=sys.stderr,
        )
        return 130
    if status is None:
        return 1  # error already surfaced by _watch

    if as_json:
        print(_terminal_json(job_handle, status))
        return 0 if status.get("status") == "succeeded" else 1

    text, code = _render_terminal(status)
    print(text)
    return code


def _watch(console: str, api_key: str, job_handle: str, *, out) -> dict | None:
    """Poll until the run is terminal; return the terminal status dict, or None after
    surfacing a loud error. Never returns a fabricated 'succeeded'."""
    deadline = time.monotonic() + _MAX_WAIT_S
    last_heartbeat = time.monotonic()
    while time.monotonic() < deadline:
        try:
            status = _poll_status(console, api_key, job_handle)
        except HTTPError as exc:
            if exc.code == 404:
                print(
                    f"newt finetune: no such job {job_handle!r} for your key "
                    f"({exc.code}). Check the handle, or that you launched it.",
                    file=sys.stderr,
                )
            else:
                print(f"newt finetune: status check failed ({exc.code}): {exc.reason}", file=sys.stderr)
            return None
        except URLError as exc:
            print(f"newt finetune: network error while polling: {exc.reason}", file=sys.stderr)
            return None

        state = status.get("status") if isinstance(status, dict) else None
        if state in _TERMINAL:
            return status

        now = time.monotonic()
        if now - last_heartbeat >= _HEARTBEAT_EVERY_S:
            print(f"  … still {state or 'running'}", file=out, flush=True)
            last_heartbeat = now
        time.sleep(_POLL_INTERVAL_S)

    print(
        f"newt finetune: gave up watching {job_handle!r} after "
        f"{int(_MAX_WAIT_S // 3600)}h — the run may still be going. Re-attach with "
        f"`newt finetune --handle {job_handle}`.",
        file=sys.stderr,
    )
    return None


def _status_once(console: str, api_key: str, job_handle: str, *, as_json: bool) -> int:
    """Fetch the run's current state exactly once, print it, and return.

    Unlike the watch loop this never sleeps or re-polls — it's the `--status` check a
    developer runs to peek at a long run without babysitting it. Exit 0 means "I fetched
    the state" (even mid-run or after a failure); nonzero means the fetch itself failed
    (unknown handle, network) — so scripts branch on the printed status, not the code."""
    try:
        status = _poll_status(console, api_key, job_handle)
    except HTTPError as exc:
        if exc.code == 404:
            print(
                f"newt finetune: no such job {job_handle!r} for your key ({exc.code}). "
                "Check the handle, or that you launched it.",
                file=sys.stderr,
            )
        else:
            print(f"newt finetune: status check failed ({exc.code}): {exc.reason}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"newt finetune: network error while checking status: {exc.reason}", file=sys.stderr)
        return 1

    if as_json:
        print(_terminal_json(job_handle, status))
        return 0

    state = status.get("status") if isinstance(status, dict) else None
    if state in _TERMINAL:
        # Terminal states get the full render (tag + Robot snippet on success, named
        # gate on failure) — but the fetch succeeded, so exit 0 regardless.
        text, _ = _render_terminal(status)
        print(text)
    else:
        print(f"{job_handle}: {state or 'unknown'} — not done yet.")
        print(f"  Watch it live:  newt finetune --handle {job_handle}")
    return 0


def _explain_launch_http_error(exc: HTTPError, dataset: str) -> None:
    if exc.code == 401:
        print("newt finetune: authentication failed — your key was rejected.", file=sys.stderr)
        print("  Rotate your key in the console, or run `newt login` again.", file=sys.stderr)
    elif exc.code == 400:
        print(f"newt finetune: the console rejected dataset {dataset!r} (bad request).", file=sys.stderr)
    elif exc.code == 503:
        print("newt finetune: the training launch is unavailable right now (503).", file=sys.stderr)
        print("  This is server-side (Modal launch not configured/reachable) — not your key.", file=sys.stderr)
    else:
        print(f"newt finetune: launch failed ({exc.code}): {exc.reason}", file=sys.stderr)
