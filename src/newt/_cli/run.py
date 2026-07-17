"""newt run — run one real inference against your model and print the result.

    newt run <tag>                              run the default snapshot against <tag>
    newt run <tag> --snapshot pour_coffee_beans use a different bundled observation
    newt run <tag> --prompt "stack the cups"    override the snapshot's recorded prompt
    newt run <tag> --json                        machine-readable mirror

The hero verb: take a model tag, authenticate, load a real bundled observation, call
the model ONCE against prod, and print what came back — the resolved model, the
round-trip latency, and the action-chunk shape.

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

# The docs' own example. A developer types `newt run <tag>` and gets a real cup-stacking
# observation without having to know a snapshot name exists.
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
    print("  --snapshot <name>  Bundled observation to send (default: cup_stacking)")
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

    ``newt run nt0-fp3-pour --snapshot pour_coffee_beans --json`` → ``nt0-fp3-pour``.
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

    snapshot = _opt_value(args, "--snapshot") or _DEFAULT_SNAPSHOT
    prompt_override = _opt_value(args, "--prompt")

    from newt import snapshots

    # Validate the snapshot BEFORE we touch the wire — an unknown name is a helpful,
    # network-free error listing what's available, never a raw KeyError traceback.
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

    import newt

    try:
        robot = newt.Robot(api_key=api_key, model=tag)
        # The waking-line timer is TTY-only and never fires under --json (issue
        # #38): a script parsing --json must never see an extra stderr line
        # mid-call.
        waking_enabled = (not as_json) and sys.stderr.isatty()
        resp = _call_with_waking_notice(
            lambda: robot.infer(obs), enabled=waking_enabled, threshold=_WAKING_THRESHOLD_S
        )
    except newt.AuthError as exc:
        print(f"newt: authentication failed — {exc.message}", file=sys.stderr)
        print("  Run `newt login` to authenticate, or set NT_API_KEY.", file=sys.stderr)
        return 1
    except newt.ModelNotFoundError as exc:
        print(f"newt: model not found — {exc.message}", file=sys.stderr)
        return 1
    except newt.BaseNotDeployableError as exc:
        print(f"newt: model not deployable — {exc.message}", file=sys.stderr)
        return 1
    except newt.RegistryUnavailable as exc:
        print(f"newt: registry unreachable — {exc.message}", file=sys.stderr)
        return 1
    except newt.ContractMismatchError as exc:
        print(f"newt: contract mismatch — {exc.message}", file=sys.stderr)
        _surface_model_status(exc)
        return 1
    except newt.ServerError as exc:
        print(f"newt: server error — {exc.message}", file=sys.stderr)
        _surface_model_status(exc)
        return 1
    except newt.VerifierError as exc:
        print(f"newt: verifier unavailable — {exc.message}", file=sys.stderr)
        return 1
    except newt.ProtocolError as exc:
        print(f"newt: protocol error — {exc.message}", file=sys.stderr)
        return 1

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
