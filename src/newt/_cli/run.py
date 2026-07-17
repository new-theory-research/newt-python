"""newt run — run one real inference against your model and print the result.

    newt run <tag>                              run the contract-matched snapshot against <tag>
    newt run <tag> --snapshot pour_coffee_beans use a different bundled observation
    newt run <tag> --prompt "stack the cups"    override the snapshot's recorded prompt
    newt run <tag> --json                        machine-readable mirror

The hero verb: take a model tag, authenticate, load a real bundled observation, call
the model ONCE against prod, and print what came back — the resolved model, the
round-trip latency, and the action-chunk shape.

Snapshot selection is contract-aware. With no `--snapshot`, `newt run` reads the resolved
model's declared contract (`robot.contract`) and picks the bundled snapshot whose state
dimension and camera set MATCH — an 8-axis model gets an 8-axis snapshot, a 6-axis
SO-101 model gets a 6-axis one. If no bundled snapshot matches the contract, that's an
honest error naming what's available for which shapes — never a silent coercion of a
wrong-shaped frame (Rule 10). `--snapshot` overrides the choice explicitly; a mismatched
explicit choice fails client-side with the same contract error the SDK raises for any bad
obs (console-011's pre-flight).

v1 is hardware-free by design: no robot is connected and nothing moves. This is a live
inference against your model, said plainly in the output so no one mistakes it for a
robot demo. `--embodiment` and the streaming loop against real hardware are a separate
future phase, not built here.

Built entirely from parts that already ship: `Robot(model=tag)` resolves the tag
through the registry inside its own constructor, `infer(obs)` is a self-contained
one-shot that opens the WS, sends one frame, and returns a labeled chunk plus latency
without touching any hardware callback, and `newt.snapshots` bundles real recorded
observations that carry their own prompt.
"""
from __future__ import annotations

import json
import os
import sys
import threading

from newt._credentials import read_api_key

# The fallback snapshot when a model's contract declares nothing to match on (a base
# whose registry entry carries no state_shape / cameras — there's no shape to key off, so
# the historical default stands). A model WITH a contract gets a contract-matched snapshot
# instead (see _select_snapshot). cup_stacking is the docs' own example: a developer types
# `newt run <tag>` and gets a real cup-stacking observation without knowing a snapshot
# name exists.
_DEFAULT_SNAPSHOT = "cup_stacking"

# issue #38 — honest cold-start feedback. Two independent seams:
#   1. While a call is in flight past this many seconds, print ONE stderr line
#      saying the model may be waking up — never silence, never a fabricated ETA.
#   2. When the call returns, if the whole call (total_ms) dwarfed the final
#      attempt's own latency, or a retry happened, the latency line renders the
#      split honestly instead of labeling the whole wait "latency".
_WAKING_THRESHOLD_S = 5.0
_WAKING_LINE = "model is waking up — first call after idle can take a few minutes"
_SPLIT_GAP_MS = 5000.0

# ANSI colors — same semantic roles as the sibling verbs.
_RESET = "\033[0m"
_GREEN = "\033[92m"   # pop-green: resolved model headline
_MINT = "\033[96m"    # dim mint: latency
_GRAY = "\033[90m"    # warm gray: axes / framing (muted facts)


def _c(code: str, text: str) -> str:
    """Wrap text in an ANSI escape when stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


def _usage() -> None:
    print("Usage: newt run <tag> [options]")
    print("")
    print("  Run one real inference against your model and print what came back.")
    print("  Loads a bundled observation, calls the model once against prod, and prints")
    print("  the resolved model, the round-trip latency, and the action-chunk shape.")
    print("")
    print("  No robot is connected and nothing moves — this is a live inference against")
    print("  your model, not a robot demo.")
    print("")
    print("Arguments:")
    print("  <tag>   Model tag or UID to run (required)")
    print("")
    print("Options:")
    print("  --snapshot <name>  Bundled observation to send (default: matched to the")
    print("                     model's contract — 8-axis rig, 6-axis SO-101)")
    print("  --prompt <text>    Override the snapshot's recorded prompt")
    print("  --json             Emit machine-readable JSON")
    print("")
    print("Environment:")
    print("  NT_API_KEY        API key override (overrides ~/.nt/credentials).")
    print("  NT_BOOTSTRAP_URL  Override registry discovery base URL.")
    print("  NT_INFERENCE_URL  Override inference endpoint directly (skips discovery).")


# ---------------------------------------------------------------------------
# Argument scan — hand-rolled, matching every sibling verb's manual style (Rule 9)
# ---------------------------------------------------------------------------

# Flags that consume the following token as their value. Named so the positional
# tag scan skips a value like `--snapshot cup_stacking` instead of reading it as the tag.
_VALUE_FLAGS = ("--snapshot", "--prompt")


def _opt_value(args: list[str], name: str) -> str | None:
    """Value for ``--name X`` or ``--name=X``. Returns None if the flag is absent or its
    value is missing / looks like another flag (so ``--snapshot --json`` is a missing
    value, not a snapshot literally named ``--json``). Mirrors finetune.py::_opt_value."""
    for i, a in enumerate(args):
        if a == name:
            nxt = args[i + 1] if i + 1 < len(args) else None
            return nxt if (nxt and not nxt.startswith("-")) else None
        if a.startswith(name + "="):
            return a[len(name) + 1 :] or None
    return None


def _positional_tag(args: list[str]) -> str | None:
    """First bareword that is not a flag or a value consumed by a value-flag.

    ``newt run so101 --snapshot red_cube --json`` → ``so101``.
    """
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in _VALUE_FLAGS:
            skip_next = True   # its value token is consumed, never the tag
            continue
        if a.startswith("-"):  # --json, --snapshot=x, -h, etc.
            continue
        return a
    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _shape_str(chunk) -> str:
    """`(H, D)` for the action chunk — the shape summary, never the raw vector dump."""
    shape = getattr(chunk, "shape", ())
    return "(" + ", ".join(str(d) for d in shape) + ")"


def _latency_line(resp) -> str:
    """The latency line — split honestly when the whole call dwarfed the final
    attempt's own latency, or a retry happened (issue #38).

    `resp.latency_ms` is the final attempt's send+recv only; `resp.total_ms` is
    real wall-clock for the WHOLE call (connect, any cold-start retry, every
    failed verifier-retry attempt and its backoff). Both are facts the client
    measured — never a guessed duration. On the ordinary warm path the two are
    equal and this renders byte-identical to the plain `Xms` it always did.
    """
    total_ms = getattr(resp, "total_ms", resp.latency_ms)
    retries = getattr(resp, "retries", 0)
    gap_ms = total_ms - resp.latency_ms
    if retries <= 0 and gap_ms <= _SPLIT_GAP_MS:
        return f"{resp.latency_ms:.0f}ms"
    note = "first call woke the model" if gap_ms > _SPLIT_GAP_MS else "retrying"
    total_s = int(total_ms // 1000)
    return f"{resp.latency_ms:.0f}ms ({note}: {total_s}s total)"


def _call_with_waking_notice(fn, *, enabled: bool, threshold: float = _WAKING_THRESHOLD_S):
    """Call fn(); if it hasn't returned within `threshold` seconds, print the
    #38 waking line to stderr once (issue #38's silence fix — say something the
    moment a call runs long, not only after it finally returns).

    `enabled` gates the whole mechanism at the call site: the caller passes
    False for --json or when stderr isn't a TTY, so the line is TTY-only, never
    machine-readable output, and — because the timer is one-shot — never
    repeated.
    """
    if not enabled:
        return fn()
    timer = threading.Timer(threshold, lambda: print(_WAKING_LINE, file=sys.stderr))
    timer.daemon = True
    timer.start()
    try:
        return fn()
    finally:
        timer.cancel()


def _render_human(tag: str, resp, snapshot: str) -> None:
    """The resolved model, latency, action-chunk shape summary, and the honest framing."""
    model = resp.model or tag
    print(_c(_GREEN, model))
    print(f"  latency   {_c(_MINT, _latency_line(resp))}")
    print(f"  action    {_shape_str(resp.action_chunk)}  {_c(_GRAY, ' '.join(resp.axes))}")
    print(f"  snapshot  {_c(_GRAY, snapshot)}")
    print("")
    # The non-negotiable honesty seam: a live inference is NOT a live robot. Wording is
    # the worker's canonical framing (no docs `run` line existed to track — flagged for
    # the follow-on docs arc to ratify).
    print(_c(_GRAY, "No robot is connected and nothing moved — this was a live inference"))
    print(_c(_GRAY, "against your model."))


def _surface_model_status(exc) -> None:
    """Surface a pending/dead OWN-model status VERBATIM — never swallowed (Rule 10).

    Where the server detail carries `model_status` (a developer's own model that is
    pending or dead, not yet servable), that field is the whole reason to run your own
    tag — so it rides out on its own line, verbatim, in addition to the server-authored
    `.message` already printed above. The full #28 error taxonomy is that issue's work;
    this verb must not bury detail the wire already carries.
    """
    ctx = getattr(exc, "context", None) or {}
    status = ctx.get("model_status")
    if status is not None:
        print(f"  model status: {status}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Contract-aware snapshot selection
# ---------------------------------------------------------------------------

def _select_snapshot(snapshots, contract) -> str | None:
    """Pick the bundled snapshot matching `contract`, or None if nothing matches.

    The match is on SHAPE, the thing that actually has to line up: a snapshot qualifies
    when the contract's state_shape and cameras (whichever the contract declares) equal
    the snapshot's. Snapshots are scanned in registry order, so cup_stacking wins the
    8-axis tie over pour_coffee_beans — preserving the historical 8-axis default.

    Two Rule-10 seams:
      - A contract that declares NOTHING to match on (state_shape and cameras both
        absent — a base whose entry carries only action_shape, or the empty
        registry of the NT_INFERENCE_URL override) has no shape to key off, so we fall
        back to _DEFAULT_SNAPSHOT rather than guess. The default is a real 8-axis frame;
        if it's wrong for this model, the server's contract check catches it — we don't
        fabricate a match.
      - When the contract DOES declare a shape and no snapshot matches it, we return None
        (the caller prints an honest no-match error) — never coerce a wrong-shaped frame.
    """
    if contract is None or (contract.state_shape is None and contract.cameras is None):
        return _DEFAULT_SNAPSHOT

    for name in snapshots.available():
        desc = snapshots.describe(name)
        if contract.state_shape is not None and desc["state_shape"] != contract.state_shape:
            continue
        if contract.cameras is not None and desc["cameras"] != tuple(contract.cameras):
            continue
        return name
    return None


def _print_no_match_error(snapshots, tag: str, contract) -> None:
    """Print the honest no-bundled-snapshot-matches-this-contract error to stderr.

    Names the contract we couldn't match AND every bundled snapshot's shape, so the
    developer sees exactly why (and can pass an explicit --snapshot or record their own
    frame). Never coerces a wrong-shaped snapshot onto the wire (Rule 10)."""
    print(
        f"newt: no bundled snapshot matches the contract for {tag!r} "
        f"(state_shape={contract.state_shape}, cameras={contract.cameras}).",
        file=sys.stderr,
    )
    print("  Bundled snapshots and their shapes:", file=sys.stderr)
    for name in snapshots.available():
        desc = snapshots.describe(name)
        print(
            f"    {name}: state_shape={desc['state_shape']}, cameras={desc['cameras']}",
            file=sys.stderr,
        )
    print(
        "  Pass --snapshot <name> to send one explicitly, or record a matching frame "
        "with `newt record`.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Error rendering — one house-shaped dispatch, shared by construction and infer
# ---------------------------------------------------------------------------

def _render_error(exc) -> int:
    """Render a NewTheoryError as the house 2-line stderr shape and return exit code 1.

    Shared by both boundaries that can raise it — Robot construction (auth / model
    resolution / registry) and infer() (contract / server / verifier / protocol) — so a
    given error prints identically no matter which call surfaced it."""
    import newt

    if isinstance(exc, newt.AuthError):
        print(f"newt: authentication failed — {exc.message}", file=sys.stderr)
        print("  Run `newt login` to authenticate, or set NT_API_KEY.", file=sys.stderr)
    elif isinstance(exc, newt.ModelNotFoundError):
        print(f"newt: model not found — {exc.message}", file=sys.stderr)
    elif isinstance(exc, newt.BaseNotDeployableError):
        print(f"newt: model not deployable — {exc.message}", file=sys.stderr)
    elif isinstance(exc, newt.RegistryUnavailable):
        print(f"newt: registry unreachable — {exc.message}", file=sys.stderr)
    elif isinstance(exc, newt.ContractMismatchError):
        print(f"newt: contract mismatch — {exc.message}", file=sys.stderr)
        _surface_model_status(exc)
    elif isinstance(exc, newt.ServerError):
        print(f"newt: server error — {exc.message}", file=sys.stderr)
        _surface_model_status(exc)
    elif isinstance(exc, newt.VerifierError):
        print(f"newt: verifier unavailable — {exc.message}", file=sys.stderr)
    elif isinstance(exc, newt.ProtocolError):
        print(f"newt: protocol error — {exc.message}", file=sys.stderr)
    else:
        # Any other NewTheoryError — surface its message rather than swallow it (Rule 10).
        print(f"newt: {exc.message}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cmd_run(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    as_json = "--json" in args

    tag = _positional_tag(args)
    if not tag:
        print("newt: no model tag given", file=sys.stderr)
        print(
            "  Usage: newt run <tag> [--snapshot <name>] [--prompt <text>] [--json]",
            file=sys.stderr,
        )
        return 1

    api_key = os.environ.get("NT_API_KEY") or read_api_key()
    if not api_key:
        print(
            "newt: no API key found — run `newt login` to authenticate, or set NT_API_KEY.",
            file=sys.stderr,
        )
        return 1

    explicit_snapshot = _opt_value(args, "--snapshot")
    prompt_override = _opt_value(args, "--prompt")

    from newt import snapshots

    import newt

    # Construct the Robot FIRST — it fetches the registry and resolves the tag's contract,
    # which contract-aware snapshot selection needs. Construction is also where auth /
    # model-resolution / registry errors surface.
    try:
        robot = newt.Robot(api_key=api_key, model=tag)
    except newt.NewTheoryError as exc:
        return _render_error(exc)

    # Choose the snapshot. An explicit --snapshot wins as-is (a mismatch with the model's
    # contract is caught client-side by the SDK's own pre-flight on infer(), console-011).
    # Otherwise pick the snapshot whose shape matches the resolved contract; if the
    # contract declares a shape nothing bundled matches, fail honestly (Rule 10) —
    # never coerce a wrong-shaped frame onto the wire.
    if explicit_snapshot is not None:
        snapshot = explicit_snapshot
    else:
        contract = getattr(robot, "contract", None)
        snapshot = _select_snapshot(snapshots, contract)
        if snapshot is None:
            _print_no_match_error(snapshots, tag, contract)
            return 1

    # Load the chosen snapshot. An unknown explicit name is a helpful, network-free error
    # listing what's available, never a raw KeyError traceback.
    try:
        obs = snapshots.load(snapshot)
    except KeyError:
        print(f"newt: unknown snapshot {snapshot!r}", file=sys.stderr)
        print(
            f"  Available snapshots: {', '.join(snapshots.available())}.",
            file=sys.stderr,
        )
        return 1

    if prompt_override is not None:
        obs["prompt"] = prompt_override

    try:
        # The waking-line timer is TTY-only and never fires under --json (issue
        # #38): a script parsing --json must never see an extra stderr line
        # mid-call.
        waking_enabled = (not as_json) and sys.stderr.isatty()
        resp = _call_with_waking_notice(
            lambda: robot.infer(obs), enabled=waking_enabled, threshold=_WAKING_THRESHOLD_S
        )
    except newt.NewTheoryError as exc:
        return _render_error(exc)

    if as_json:
        print(
            json.dumps(
                {
                    "tag": tag,
                    "model": resp.model or tag,
                    "snapshot": snapshot,
                    "prompt": obs.get("prompt"),
                    "latency_ms": resp.latency_ms,
                    "total_ms": getattr(resp, "total_ms", resp.latency_ms),
                    "retries": getattr(resp, "retries", 0),
                    "action_chunk": {
                        "shape": list(getattr(resp.action_chunk, "shape", ())),
                        "axes": resp.axes,
                    },
                }
            )
        )
        return 0

    _render_human(tag, resp, snapshot)
    return 0
