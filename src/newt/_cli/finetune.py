"""newt finetune — launch a training run on NT's GPUs with your key, and watch it.

    newt finetune --dataset ./my-folder  upload a local dataset folder, then launch
    newt finetune --dataset <name>       launch on an already-staged dataset name
    newt finetune --dataset <name> --steps 20000   set total training steps
    newt finetune --dataset <name> --name kitchen-grasp  name the model this run makes
    newt finetune --dataset <name> --fresh   ignore any checkpoint, retrain from scratch
    newt finetune --handle <job>         re-attach to a run already launched
    newt finetune --handle <job> --status   print the run's state ONCE and exit
    newt finetune --dataset <name> --json   machine-readable handle + terminal state
    newt finetune --list                 your recent runs (handle, dataset, state, when)

``--dataset`` takes either a **path** or a **name**, and the CLI prints which it
chose (never a silent branch). A path (a ``/`` in the argument, or an existing local
directory) is validated locally, uploaded to your NT namespace, and launched against
the staged folder name. A bare name skips the upload and launches straight against
the already-staged dataset — the original behavior, unchanged.

`--dataset` POSTs to the NT console's ``/api/finetune`` endpoint, authenticated by
your ``nt_`` key. The console launches the Modal training job **server-side, under
NT's own Modal credentials**, and hands back a job handle — no Modal credential ever
reaches this client (training spec §3.1). The CLI then polls ``/api/finetune/status``
against that handle until the run reaches a terminal state:

    succeeded → the model tag + a pointer to its report card
    failed    → what happened in plain words, the raw gate + any server detail on the
                technical line, the run page repeated, and the concrete next steps

When a run fails at a gate that runs AFTER ``train`` (frame-check / registry+reload /
serve), the CLI adds a survival line: training completed, the checkpoint is safe, and
nothing needs retraining — the failure is in post-processing. This is derived purely
from the gate order the client already knows (``_PIPELINE_GATES``), never new server
state; a failure at intake or train prints no such line (nothing survived).

Ctrl-C never loses the run: the handle is printed at launch, and
``newt finetune --handle <job>`` re-attaches to it. ``newt finetune --list`` prints
your recent runs so a handle is never lost to a scrollback — its ``state`` column is
the last status the console recorded; for live state, poll a handle with ``--status``.

Featherweight on purpose, same as the rest of the CLI: stdlib ``urllib`` only.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from newt._credentials import read_api_key

# A staged dataset name: the safe-segment alphabet the console's sign route
# accepts (apps/console/lib/uploads-request.ts::isSafeSegment). A folder whose
# basename falls outside it can't be a namespace segment, so the upload is
# refused with a rename hint rather than a downstream 400.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")

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

# The training pipeline's gate SEQUENCE, mirrored client-side. Source of truth is the
# server pipeline's own ordered gate list (PIPELINE_GATES). We keep only the ORDER, and
# only to answer one question with zero new server state — did the failing gate come
# AFTER `train`? If it did, training finished and the checkpoint is safe on the volume
# (the survival line below). `registry+reload` is deliberately ONE gate (it writes the
# registry entry AND reloads it), not `registry`. A gate name we can't place in this
# sequence is surfaced honestly, never guessed "after train" — a wrong guess would tell
# someone their checkpoint is safe when it isn't (Rule 10).
_PIPELINE_GATES = ("intake", "train", "frame-check", "registry+reload", "serve")

# Plain-language translation of each pipeline gate — what the developer was waiting on
# when the run stopped. The HEADLINE always uses this, never the bare internal token: a
# developer has never seen `registry+reload`, and Mattie's live receipt (2026-07-17) was
# the terminal line `Fine-tune failed at gate: train` — no why, no next step, internal
# `gate` vocabulary. The raw token stays visible on the technical line below. A gate NOT
# in this map (an unknown/renamed one) falls back to a generic headline and shows its raw
# token only on the technical line — never guessed into a plausible phrase (Rule 10).
# `eval` is mapped defensively for a checkpoint-eval gate; the entry only ever fires if
# the wire actually names it, so it invents nothing.
_GATE_PLAIN = {
    "intake": "while checking your dataset",
    "train": "during training",
    "frame-check": "while checking the trained model's frames",
    "registry+reload": "while registering your model",
    "serve": "while bringing your model online",
    "eval": "during checkpoint evaluation",
}

# Where a developer goes when the terminal output isn't enough. The run page (repeated at
# failure) is the primary place; the docs are the secondary pointer.
_HELP_DOCS = "https://newtheory-docs.vercel.app/docs/getting-started"


def _failure_detail(status: dict) -> str | None:
    """The server's own human-readable failure text, if the wire carried any.

    The pipeline persists the cause under the gate (``_status_from_exception`` sets
    ``error``); if/when the console status route passes it through (as ``detail`` or
    verbatim ``error``), this surfaces it on the technical line. Read both keys so a real
    cause reaches the developer the moment either lands on the wire. Returns None when
    neither is present — absence renders the honest 'this is all the run reported' line,
    never a guessed cause (Rule 10)."""
    for key in ("detail", "error"):
        val = status.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _survival_block(gate: str | None) -> str | None:
    """The 'your checkpoint is safe' block for a terminal failure — or None.

    Returns the survival message ONLY when `gate` names a gate that runs strictly
    AFTER `train` in the known sequence: training completed, so the checkpoint is safe
    and nothing needs retraining. Returns None for a failure at intake/train (nothing
    survived — no false comfort, Rule 10) and for a gate name not in `_PIPELINE_GATES`
    (we can't place it, so we say nothing rather than guess). Derived purely from gate
    order — no server round-trip."""
    if gate not in _PIPELINE_GATES:
        return None
    if _PIPELINE_GATES.index(gate) <= _PIPELINE_GATES.index("train"):
        return None
    return _c(
        _GREEN,
        "training completed — your checkpoint is safe; the failure is in\n"
        f"post-processing (gate: {gate}); nothing needs retraining",
    )


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


def _steps_raw(args: list[str]) -> str | None:
    """Raw string value for ``--steps`` (space or ``=`` form), or None when the flag is
    absent. Unlike ``_opt_value`` this does NOT reject a leading ``-``: a negative like
    ``-5`` must reach validation to be rejected with a named error, never silently read
    as a missing value. A present-but-empty flag returns ``""`` (also a named error)."""
    for i, a in enumerate(args):
        if a == "--steps":
            return args[i + 1] if i + 1 < len(args) else ""
        if a.startswith("--steps="):
            return a[len("--steps=") :]
    return None


def _validate_steps(raw: str | None) -> tuple[int | None, str | None]:
    """``(steps, error)`` for the ``--steps`` value. ``raw is None`` (flag absent) →
    ``(None, None)``: no override, the server picks its default. Otherwise the value
    must be a positive whole number — ``0``, negatives, and non-numeric are rejected
    with a NAMED error and no coercion (Rule 10). The server enforces the min/max
    bounds; this is only the shape check that keeps a garbage value off the wire."""
    if raw is None:
        return None, None
    text = raw.strip()
    try:
        value = int(text)
    except ValueError:
        return None, f"--steps must be a whole number of training steps, got {raw!r}."
    if value <= 0:
        return None, f"--steps must be a positive number of training steps, got {value}."
    return value, None


# ft-024: --name is an optional model name, validated client-side to the SAME slug rule
# the console enforces (lowercase alphanumeric + hyphens, length 3..40) so a bad name
# never reaches the wire.
_NAME_SLUG = re.compile(r"^[a-z0-9-]+$")
_NAME_MIN = 3
_NAME_MAX = 40


def _name_raw(args: list[str]) -> str | None:
    """Raw string value for ``--name`` (space or ``=`` form), or None when the flag is
    absent. A present-but-empty flag (``--name`` with no value, ``--name=``, or a value
    that looks like another flag) returns ``""`` so validation surfaces a named error —
    never silently treated as "no name" (Rule 10)."""
    for i, a in enumerate(args):
        if a == "--name":
            nxt = args[i + 1] if i + 1 < len(args) else None
            return nxt if (nxt is not None and not nxt.startswith("-")) else ""
        if a.startswith("--name="):
            return a[len("--name=") :]
    return None


def _validate_name(raw: str | None) -> tuple[str | None, str | None]:
    """``(name, error)`` for the ``--name`` value. ``raw is None`` (flag absent) →
    ``(None, None)``: no name, the model is named after the dataset (the default). A
    present value must be a slug — lowercase letters, digits, and hyphens, length 3..40.
    Anything else is a NAMED error with no coercion (spec §2, Rule 10): a name with
    spaces or uppercase is refused, never silently reshaped into something that works.
    The server re-validates the same rule; this keeps a bad value off the wire."""
    if raw is None:
        return None, None
    text = raw.strip()
    if len(text) < _NAME_MIN or len(text) > _NAME_MAX:
        return None, (
            f"--name must be between {_NAME_MIN} and {_NAME_MAX} characters, "
            f"got {len(text)}: {raw!r}."
        )
    if not _NAME_SLUG.match(text):
        return None, (
            "--name must be a slug — lowercase letters, digits, and hyphens only "
            f"(no spaces or uppercase): {raw!r}."
        )
    return text, None


def _usage() -> None:
    print("Usage: newt finetune (--dataset <path|name> | --handle <job> | --list) [--status] [--json]")
    print("")
    print("  Launch a training run on NT's GPUs with your key, then watch it to")
    print("  completion. The launch happens server-side under NT's Modal credentials —")
    print("  no Modal credential ever touches this client.")
    print("")
    print("Options:")
    print("  --dataset <path|name>  What to fine-tune on. Pass a local folder (e.g.")
    print("                         ./my-export) to upload it and launch on it, or a")
    print("                         staged dataset name to launch on it directly. The")
    print("                         CLI prints which it detected before it acts.")
    print("  --steps <int>          Total training steps for this run (with --dataset).")
    print("                         Omit to use the server default.")
    print("  --name <slug>          Name for the model this run produces (with --dataset).")
    print("                         Omit to name it after the dataset.")
    print("  --fresh                Ignore any existing checkpoint and retrain from scratch")
    print("                         (with --dataset). Omit to resume a completed run's")
    print("                         post-processing.")
    print("  --handle <job>         Re-attach to a run already launched (poll only).")
    print("  --status               With --handle: print the run's current state once")
    print("                         and exit — a one-shot check, no blocking watch.")
    print("  --list                 List your recent runs (handle, dataset, state,")
    print("                         created). The state column is the LAST recorded")
    print("                         status, not a live poll — for live state, check a")
    print("                         handle with --status.")
    print("  --json                 Emit machine-readable JSON (handle + terminal state;")
    print("                         with --list, the raw runs array).")
    print("")
    print("Environment:")
    print("  NT_API_KEY      API key override (overrides ~/.nt/credentials).")
    print("  NT_CONSOLE_URL  Console URL (default: https://newtheory-console.vercel.app)")


# ---------------------------------------------------------------------------
# HTTP round-trips (one each) — split out so tests exercise the orchestration and
# rendering without a network.
# ---------------------------------------------------------------------------
def _launch(
    console: str,
    api_key: str,
    dataset: str,
    *,
    steps: int | None = None,
    name: str | None = None,
    fresh: bool = False,
    timeout: float = 30.0,
) -> dict:
    # `steps` rides only when the developer set it (--steps); absent, the field is
    # omitted entirely so the server applies its own default — never a client-fabricated
    # step count (Rule 10). `name` (ft-024, --name) and `fresh` (--fresh) ride the same
    # way: each present in the body ONLY when the developer set it, so an un-flagged
    # launch is byte-for-byte the request it was before these flags existed.
    payload: dict = {"dataset": dataset}
    if steps is not None:
        payload["steps"] = steps
    if name is not None:
        payload["name"] = name
    if fresh:
        payload["fresh"] = True
    req = Request(
        f"{console}/api/finetune",
        data=json.dumps(payload).encode(),
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


def _list_jobs(console: str, api_key: str, *, timeout: float = 30.0) -> list[dict]:
    """The caller's recent runs, newest first, from GET /api/finetune/jobs — owner-
    scoped server-side by the `nt_` key (the route returns only this key's owner's
    rows). Returns the raw list; a response missing the `jobs` array is a contract
    violation, surfaced loud rather than treated as 'no runs' (Rule 10)."""
    req = Request(f"{console}/api/finetune/jobs", headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    jobs = body.get("jobs") if isinstance(body, dict) else None
    if not isinstance(jobs, list):
        raise ValueError(f"list response carried no jobs array: {body!r}")
    return jobs


# ---------------------------------------------------------------------------
# Terminal rendering — pure: (status dict) -> (human text, exit code).
# ---------------------------------------------------------------------------
def _render_terminal(
    status: dict,
    *,
    console: str | None = None,
    job_handle: str | None = None,
    dataset: str | None = None,
) -> tuple[str, int]:
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
        detail = _failure_detail(status)

        # Plain-words headline — what the developer was waiting on when the run stopped,
        # NEVER the bare internal gate token (Rule 10; Mattie's receipt: the old headline
        # `Fine-tune failed at gate: train` said nothing a developer could act on). An
        # unknown/renamed gate falls back to the generic line — the raw token surfaces
        # only on the technical line below, never guessed into a phrase.
        phrase = _GATE_PLAIN.get(gate) if isinstance(gate, str) else None
        head = (
            f"Fine-tune didn't finish — it stopped {phrase}."
            if phrase
            else "Fine-tune didn't finish."
        )
        lines = [_c(_RED, head)]

        # If the failing gate ran after `train`, say plainly what SURVIVED: the
        # checkpoint. This is the one wire point for the survival line, so every
        # terminal render path (watch, re-attach, --status — all route through here)
        # prints it exactly once.
        survived = _survival_block(gate)
        if survived:
            lines.append(survived)

        # The technical line — the raw wire token (and the server's own words if it sent
        # any), kept UNDER the plain headline for the developer who wants the exact gate.
        if isinstance(gate, str) and gate:
            tech = f"failed at gate {gate!r}"
        else:
            tech = "the run reported a failure with no gate named"
        if detail:
            tech += f" — {detail}"
        lines.append(_c(_GRAY, f"  technical:   {tech}"))

        # Repeat the run's page (already printed at launch) — the primary place to see
        # more than the CLI was handed.
        if console and job_handle:
            lines.append(_c(_GRAY, f"  run page:    {console}/runs/{job_handle}"))

        # Honesty (Rule 10): when the wire carried nothing beyond the gate name, say so —
        # never invent a cause. The run page may carry more than reached the CLI. Today
        # the wire IS bare here: the pipeline persists the cause but the console status
        # route drops it before this client (portal#94) — this line is honest until it lands.
        if not detail:
            tail = " — its page above may have more." if (console and job_handle) else "."
            lines.append(_c(_GRAY, f"  This is all the run reported{tail}"))

        # Concrete next steps: re-run this exact dataset (only when we know it — never a
        # fabricated name, Rule 10), and where to go for help.
        lines.append("")
        if dataset:
            lines.append(f"  try again:   newt finetune --dataset {dataset}")
        lines.append(f"  get help:    {_HELP_DOCS}")

        return "\n".join(lines), 1

    # Anything else reaching here is a contract violation — surface it, don't guess.
    return _c(_RED, f"Fine-tune ended in an unexpected state: {status!r}"), 1


def _terminal_json(
    job_handle: str,
    status: dict,
    *,
    steps: int | None = None,
    name: str | None = None,
) -> str:
    # `steps` is the effective total-steps for this run: the value the developer set
    # with --steps, or null when they didn't (the server default is in force). Null is
    # honest here — the CLI never invents the server's default to fill it (Rule 10).
    # `name` (ft-024) is the effective model name: the --name the developer set, or null
    # when they didn't (the model is named after the dataset) — null, never a fabricated
    # name.
    return json.dumps(
        {
            "job_handle": job_handle,
            "status": status.get("status"),
            "gate": status.get("gate"),
            "tag": status.get("tag"),
            "report_card": status.get("report_card"),
            "steps": steps,
            "name": name,
            # Live telemetry passthrough (may be null): the run's step/loss/throughput/
            # ETA while training, sourced from the console's `progress` block. Additive —
            # existing keys are unchanged, so a consumer that ignores it sees no diff.
            "progress": status.get("progress") if isinstance(status, dict) else None,
        }
    )


# ---------------------------------------------------------------------------
# Live-progress rendering — pure: (status dict) -> one compact line, or None.
# ---------------------------------------------------------------------------
def _fmt_duration(seconds) -> str | None:
    """A compact human duration ('3h 12m' / '28m' / '45s'), or None if unusable. Never
    raises on a garbage value — live telemetry is best-effort, so a bad ETA just drops."""
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return None
    if total < 0:
        return None
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{secs}s"


def _render_progress(status: dict) -> str | None:
    """A compact, single-line live-progress string from a status dict's `progress`
    block, or None when there's no usable live telemetry yet.

    Pure and best-effort: every field is optional (the trainer reports whatever lerobot
    logged), so a run with only a step, or only a loss, still renders what it has and
    silently drops the rest — never fabricates a missing scalar (Rule 10). Returns None
    when `progress` is absent/empty so the caller can fall back to its honest
    'no telemetry yet' line."""
    progress = status.get("progress") if isinstance(status, dict) else None
    if not isinstance(progress, dict):
        return None

    parts: list[str] = []

    gate = progress.get("active_gate") or progress.get("gate")
    if gate:
        parts.append(f"gate: {gate}")

    step = progress.get("step")
    total = progress.get("total_steps")
    if isinstance(step, (int, float)) and not isinstance(step, bool):
        if isinstance(total, (int, float)) and not isinstance(total, bool) and total > 0:
            pct = int(step / total * 100)
            parts.append(f"step {int(step):,} / {int(total):,} ({pct}%)")
        else:
            parts.append(f"step {int(step):,}")

    loss = progress.get("loss")
    if isinstance(loss, (int, float)) and not isinstance(loss, bool):
        parts.append(f"loss {loss:.3f}")

    sps = progress.get("samples_per_s")
    if isinstance(sps, (int, float)) and not isinstance(sps, bool):
        parts.append(f"{int(sps)} smp/s")

    mem = progress.get("gpu_mem_gb")
    if isinstance(mem, (int, float)) and not isinstance(mem, bool):
        parts.append(f"{mem:.1f} GiB")

    eta = _fmt_duration(progress.get("eta_s"))
    if eta:
        parts.append(f"~{eta} left")

    if not parts:
        return None
    return "   ".join(parts)


# ---------------------------------------------------------------------------
# --list rendering — pure: (jobs list) -> human text. Newest-first ordering and
# owner-scoping are the route's job; this just tabulates what it returns.
# ---------------------------------------------------------------------------
def _render_jobs_table(jobs: list[dict]) -> str:
    """A readable table of the caller's runs: handle, dataset, state, created. The
    `state` column prints the console's LAST-RECORDED status verbatim — it is not a
    live poll (terminal states are set elsewhere; today many rows read `launched`).
    The caption points at `--status` for live state so the column is never mistaken
    for a fresh check (honest staleness, not invented live state — Rule 10)."""
    headers = ("HANDLE", "DATASET", "STATE", "CREATED")
    rows = [
        (
            str(j.get("job_handle") or "—"),
            str(j.get("dataset") or "—"),
            str(j.get("status") or "—"),
            str(j.get("created_at") or "—"),
        )
        for j in jobs
    ]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(4)
    ]

    def _line(cells, color=None):
        parts = [cells[i].ljust(widths[i]) for i in range(4)]
        text = "  ".join(parts).rstrip()
        return _c(color, text) if color else text

    out_lines = [_line(headers, _GRAY)]
    out_lines.extend(_line(r) for r in rows)
    out_lines.append("")
    out_lines.append(
        _c(_GRAY, "state = last recorded status; for live state: newt finetune --handle <job> --status")
    )
    return "\n".join(out_lines)


def _cmd_list(console: str, api_key: str, *, as_json: bool, out) -> int:
    """`newt finetune --list` — print the caller's recent runs. `--json` emits the raw
    list on a clean stdout; an owner with no runs gets a clear, friendly line (not an
    error)."""
    try:
        jobs = _list_jobs(console, api_key)
    except HTTPError as exc:
        if exc.code == 401:
            print("newt finetune: authentication failed — your key was rejected.", file=sys.stderr)
            print("  Rotate your key in the console, or run `newt login` again.", file=sys.stderr)
        else:
            print(f"newt finetune: could not list runs ({exc.code}): {exc.reason}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"newt finetune: cannot reach {console}: {exc.reason}", file=sys.stderr)
        print("  Set NT_CONSOLE_URL if you're running a local console.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"newt finetune: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(jobs))
        return 0

    if not jobs:
        print("No fine-tune runs yet.", file=out)
        print("  Launch one with:  newt finetune --dataset <path|name>", file=out)
        return 0

    print(_render_jobs_table(jobs), file=out)
    return 0


# ---------------------------------------------------------------------------
# Path-vs-name detection + folder upload. The name branch is the
# original behavior verbatim; the path branch validates locally, uploads, and
# hands the SAME launch code the staged folder name.
# ---------------------------------------------------------------------------
def _looks_like_path(arg: str) -> bool:
    """Explicit, documented rule: an argument is a PATH if it contains a path
    separator, OR it resolves to an existing local directory. Otherwise it's a
    staged dataset NAME. Precedence for the ambiguous case (a bare token that also
    names a local directory): the directory wins — so the developer's folder is
    uploaded, and the CLI says so — never a silent guess."""
    if "/" in arg or os.sep in arg or (os.altsep and os.altsep in arg):
        return True
    return Path(arg).is_dir()


def _staged_name_for(export_dir: Path) -> str:
    """The name the folder is staged under, launched by, and tracked as — one name
    threaded through. It's the folder's basename, which must be a safe namespace
    segment (the console's sign route rejects anything else). Raises with a rename
    hint if it isn't."""
    name = export_dir.resolve().name
    if not name or name in (".", "..") or len(name) > 128 or not _SAFE_SEGMENT.match(name):
        raise ValueError(
            f"the folder name {name!r} can't be a dataset name — use letters, "
            "digits, '.', '_' or '-' (max 128 chars). Rename the folder and retry."
        )
    return name


def _format_size(nbytes: int) -> str:
    mb = nbytes / (1024 * 1024)
    if mb >= 10:
        return f"{mb:.0f} MB"
    if mb >= 1:
        return f"{mb:.1f} MB"
    return f"{nbytes / 1024:.0f} KB"


def _dir_size(export_dir: Path) -> int:
    return sum(p.stat().st_size for p in export_dir.rglob("*") if p.is_file())


def _make_progress(out, total_bytes: int):
    """A progress callback for NTCloudSink.upload_directory that redraws an upload
    meter in place on a TTY, and stays quiet otherwise (so --json stdout and piped
    logs aren't spammed with carriage returns)."""
    is_tty = hasattr(out, "isatty") and out.isatty()

    def _progress(done_files: int, total_files: int, done_bytes: int, total: int) -> None:
        if not is_tty:
            return
        pct = (done_bytes / total * 100) if total else 100.0
        print(
            f"\r  … {done_files}/{total_files} files "
            f"({_format_size(done_bytes)} / {_format_size(total)}) {pct:.0f}%",
            end="",
            file=out,
            flush=True,
        )
        if done_files == total_files:
            print("", file=out)

    return _progress


def _resolve_dataset_arg(dataset_arg: str, console: str, api_key: str, out) -> str | None:
    """Turn ``--dataset``'s argument into the staged dataset name to launch against,
    printing the detected branch. Name ⇒ returned unchanged (no upload). Path ⇒
    validated locally, uploaded, and the staged folder name returned. Returns None
    after surfacing a loud error (bad path, malformed export, failed upload) — the
    caller then exits without launching."""
    if not _looks_like_path(dataset_arg):
        print(
            f"Using staged dataset {dataset_arg!r} "
            "(no local folder — launching against the staged name).",
            file=out,
        )
        return dataset_arg

    export_dir = Path(dataset_arg)
    if not export_dir.is_dir():
        print(
            f"newt finetune: {dataset_arg!r} looks like a path but isn't an "
            "existing directory.",
            file=sys.stderr,
        )
        print(
            "        Fix: point --dataset at your exported dataset folder, or pass "
            "a staged dataset name.",
            file=sys.stderr,
        )
        return None

    try:
        staged = _staged_name_for(export_dir)
    except ValueError as exc:
        print(f"newt finetune: {exc}", file=sys.stderr)
        return None

    # Import lazily so the name path (and every non-upload path) never pays the
    # recording import — and so `import newt._cli.finetune` stays featherweight.
    from newt.recording._cloud_sink import NTCloudSink, validate_lerobot_export

    # Validate on the laptop, BEFORE a byte moves: a malformed export fails here,
    # not after the whole upload and a server-side intake rejection.
    try:
        validate_lerobot_export(export_dir)
    except RuntimeError as exc:
        print(f"newt finetune: {exc}", file=sys.stderr)
        return None

    total_bytes = _dir_size(export_dir)
    print(
        f"Detected a local folder — uploading {_format_size(total_bytes)}… "
        f"staged as {staged}",
        file=out,
        flush=True,
    )

    try:
        sink = NTCloudSink(staged, api_key=api_key, console_url=console)
        # Already validated above — don't re-read the export just to re-check it.
        sink.upload_directory(export_dir, validate=False, progress=_make_progress(out, total_bytes))
    except RuntimeError as exc:
        print(f"\nnewt finetune: upload failed — {exc}", file=sys.stderr)
        return None

    print(f"  uploaded {_format_size(total_bytes)}, staged as {staged}", file=out)
    return staged


def cmd_finetune(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    as_json = "--json" in args
    as_status = "--status" in args
    as_list = "--list" in args
    dataset = _opt_value(args, "--dataset")
    handle = _opt_value(args, "--handle")

    # --steps is a launch-only, client-validated override for total training steps.
    # Validated here BEFORE any network so a bad value never reaches the console.
    steps, steps_err = _validate_steps(_steps_raw(args))
    steps_present = _steps_raw(args) is not None

    # --name (ft-024) is a launch-only, client-validated model name. Validated here
    # BEFORE any network, same as --steps. --fresh is a launch-only boolean.
    name, name_err = _validate_name(_name_raw(args))
    name_present = _name_raw(args) is not None
    fresh = "--fresh" in args

    # Instructional/progress output goes to stderr in --json mode so stdout carries
    # nothing but the final JSON object (composable with $(...) / jq).
    out = sys.stderr if as_json else sys.stdout

    if as_list and (dataset or handle):
        print("newt finetune: --list shows your runs — don't combine it with --dataset/--handle.", file=sys.stderr)
        print("        Fix: newt finetune --list", file=sys.stderr)
        return 1
    if dataset and handle:
        print("newt finetune: pass --dataset OR --handle, not both.", file=sys.stderr)
        return 1
    if steps_err:
        print(f"newt finetune: {steps_err}", file=sys.stderr)
        print("        Fix: newt finetune --dataset <name> --steps 20000", file=sys.stderr)
        return 1
    if steps_present and not dataset:
        # --steps sets the step count for a NEW launch; on --handle/--list there's
        # nothing to apply it to. Refuse loudly rather than silently ignore it (Rule 10).
        print("newt finetune: --steps only applies when launching with --dataset.", file=sys.stderr)
        print("        Fix: newt finetune --dataset <name> --steps 20000", file=sys.stderr)
        return 1
    if name_err:
        print(f"newt finetune: {name_err}", file=sys.stderr)
        print("        Fix: newt finetune --dataset <name> --name kitchen-grasp", file=sys.stderr)
        return 1
    if name_present and not dataset:
        # --name names a NEW launch's model; on --handle/--list there's nothing to name.
        # Refuse loudly rather than silently ignore it (Rule 10).
        print("newt finetune: --name only applies when launching with --dataset.", file=sys.stderr)
        print("        Fix: newt finetune --dataset <name> --name kitchen-grasp", file=sys.stderr)
        return 1
    if fresh and not dataset:
        # --fresh forces a retrain of a NEW launch; on --handle/--list there's nothing to
        # retrain. Refuse loudly rather than silently ignore it (Rule 10, spec §2).
        print("newt finetune: --fresh only applies when launching with --dataset.", file=sys.stderr)
        print("        Fix: newt finetune --dataset <name> --fresh", file=sys.stderr)
        return 1
    if as_status and not handle:
        print("newt finetune: --status needs --handle <job> — the run to check on.", file=sys.stderr)
        print("        Fix: newt finetune --handle <job> --status", file=sys.stderr)
        return 1
    if not as_list and not dataset and not handle:
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

    # --- list the caller's recent runs (no launch, no watch) ----------------
    if as_list:
        return _cmd_list(console, api_key, as_json=as_json, out=out)

    # --- one-shot status check (no watch) -----------------------------------
    if as_status:
        return _status_once(console, api_key, handle, as_json=as_json)

    # --- launch (unless re-attaching to an existing handle) -----------------
    job_handle = handle
    if dataset:
        # Path-vs-name detection. A path validates + uploads here and
        # resolves to its staged name; a name passes straight through unchanged.
        # Either way, `dataset` below is the staged name the launch runs against.
        dataset = _resolve_dataset_arg(dataset, console, api_key, out)
        if dataset is None:
            return 1  # bad path / malformed export / failed upload — already surfaced

        try:
            launched = _launch(console, api_key, dataset, steps=steps, name=name, fresh=fresh)
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
        if steps is not None:
            print(f"  steps:        {steps}", file=out)
        if name is not None:
            print(f"  name:         {name}", file=out)
        if fresh:
            print("  fresh:        forcing a full retrain (ignoring any existing checkpoint)", file=out)
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
        # steps/name are the effective overrides this launch carried (None on a --handle
        # re-attach, where the CLI didn't set them).
        print(_terminal_json(job_handle, status, steps=steps, name=name))
        return 0 if status.get("status") == "succeeded" else 1

    # `dataset` is the staged name this launch ran against, or None on a --handle
    # re-attach (nothing to re-run with — the try-again line is then omitted, never
    # a fabricated dataset name). `console`/`job_handle` repeat the run page at failure.
    text, code = _render_terminal(
        status, console=console, job_handle=job_handle, dataset=dataset
    )
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
            # Prefer live telemetry (step/loss/ETA) when the run is reporting it; fall
            # back to the honest bare-state line before the first progress lands.
            live = _render_progress(status) if isinstance(status, dict) else None
            if live:
                print(f"  {live}", file=out, flush=True)
            else:
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
        # Terminal states get the full render (tag + Robot snippet on success, plain-words
        # failure + run page + next steps on failure) — but the fetch succeeded, so exit 0
        # regardless. No dataset here (a --status check doesn't carry it), so the try-again
        # line is omitted rather than fabricated (Rule 10); the run page still points on.
        text, _ = _render_terminal(status, console=console, job_handle=job_handle)
        print(text)
    else:
        # A live run: show step/loss/ETA when the run is reporting it, else the honest
        # 'not done yet' line (never a fabricated number before the first report).
        live = _render_progress(status)
        if live:
            print(f"{job_handle}: {state or 'running'} — {live}")
        else:
            print(f"{job_handle}: {state or 'unknown'} — not done yet.")
        print(f"  Watch it live:  newt finetune --handle {job_handle}")
    return 0


def _http_error_detail(exc: HTTPError) -> str | None:
    """The server's ``detail`` string from a JSON error body, or None when there isn't
    one. The console names the EXACT reason a request was refused in ``detail`` — the
    out-of-bounds steps message (`steps must be between 2000 and 100000`), a bad ``--name``
    slug, or a ``--name`` collision (`a model named "…" already exists … NOT launched`).
    Surfacing that verbatim is how a bounds/name error actually reaches the developer
    instead of a generic 'bad request' (ft-020 papercut: the CLI was swallowing the
    server's detail). Best-effort: a non-JSON or bodyless error just yields None and the
    caller falls back to its generic line."""
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
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, str) and detail else None


def _explain_launch_http_error(exc: HTTPError, dataset: str) -> None:
    detail = _http_error_detail(exc)
    if exc.code == 401:
        print("newt finetune: authentication failed — your key was rejected.", file=sys.stderr)
        print("  Rotate your key in the console, or run `newt login` again.", file=sys.stderr)
    elif exc.code == 400:
        # Surface the server's own reason verbatim when it names one (bad --steps bounds,
        # bad --name slug) — otherwise fall back to the generic dataset line.
        if detail:
            print(f"newt finetune: the console rejected the request — {detail}", file=sys.stderr)
        else:
            print(f"newt finetune: the console rejected dataset {dataset!r} (bad request).", file=sys.stderr)
    elif exc.code == 409:
        # ft-024: a --name collision (or any conflict). The server names the tag and that
        # the run was NOT launched (fail before spend); surface it verbatim.
        if detail:
            print(f"newt finetune: {detail}", file=sys.stderr)
        else:
            print("newt finetune: that name is already taken — the run was not launched.", file=sys.stderr)
    elif exc.code == 503:
        print("newt finetune: the training launch is unavailable right now (503).", file=sys.stderr)
        print("  This is server-side (Modal launch not configured/reachable) — not your key.", file=sys.stderr)
    else:
        print(f"newt finetune: launch failed ({exc.code}): {exc.reason}", file=sys.stderr)
