"""newt record — the keyboard frontend on newt.recording.Session.

This module holds ZERO recording behavior. It is a skin: it reads keystrokes (or
line-delimited JSON commands), calls ``Session.start_episode()`` /
``end_episode(keep=)`` / ``status()`` / ``close()``, and renders what they return.
Every decision about episodes, format, atomicity, timing, and the kill lives in
``newt.recording.Session`` — if you find that logic creeping in here, it is in the
wrong file (the layering invariant).

The rhythm (the Phase-0 bench grammar):
- preflight prints the contract Session describes and refuses with the exact fix
  if the destination is not writable (frontend courtesy — the library only
  describes; the refusal is this skin's call);
- SPACE starts / stops an episode; at the stop, ENTER keeps, D discards, R redoes;
- a live frame counter + a moving joint readout print during capture;
- Ctrl+H kills: Session.kill() torques off via the source and leaves no partial
  dir, then the process exits 130;
- a kept-count toward ``--target``;
- ``--json`` drives the same Session from line-delimited stdin commands and emits
  line-delimited events — for an agent. Agents are a door, not load-bearing.
- non-TTY without ``--json`` stands down loudly: there is no keyboard to read.
"""
from __future__ import annotations

import json
import select
import sys
import termios
import time
import tty
from pathlib import Path


def _usage() -> None:
    print("Usage: newt record [options]")
    print("")
    print("  Record NT v0.0.3 episodes from an embodiment. SPACE starts/stops an")
    print("  episode; ENTER keeps it, D discards, R redoes. Ctrl+H is the kill.")
    print("")
    print("Options:")
    print("  --task TEXT     Language task prompt recorded in every episode (required).")
    print("  --dest DIR      Episode output directory (default: ./episodes).")
    print("  --simulate      Record from a fake joint stream, no hardware.")
    print("  --source SPEC   Load a developer RecordingSource, MODULE:FACTORY")
    print("                  (e.g. mypkg.rig:make_source). Mutually exclusive")
    print("                  with --simulate.")
    print("  --bimanual      (simulate) Drive a 2-arm leader/follower stream.")
    print("  --target N      Stop after N kept episodes.")
    print("  --hz N          State sample rate (default: 30).")
    print("  --author TEXT   Declared author written to episode.json provenance.")
    print("  --license TEXT  Declared license written to episode.json provenance.")
    print("  --drop-every N  (simulate) Inject a dropped read every N ticks.")
    print("  --json          Agent mode: line-delimited JSON events + stdin commands.")
    print("")
    print("  Recording needs the extra:  pip install \"newt[recording]\"")


def _parse(args: list[str]) -> dict:
    opts = {
        "task": None,
        "dest": "episodes",
        "simulate": False,
        "source": None,
        "bimanual": False,
        "target": None,
        "hz": 30,
        "author": None,
        "license": None,
        "drop_every": 0,
        "json": False,
    }
    flags = {"--simulate": "simulate", "--bimanual": "bimanual", "--json": "json"}
    # option -> (key, converter)
    valued = {
        "--task": ("task", str),
        "--dest": ("dest", str),
        "--source": ("source", str),
        "--target": ("target", int),
        "--hz": ("hz", int),
        "--author": ("author", str),
        "--license": ("license", str),
        "--drop-every": ("drop_every", int),
    }
    i = 0
    while i < len(args):
        a = args[i]
        if a in flags:
            opts[flags[a]] = True
        elif a in valued:
            key, conv = valued[a]
            i += 1
            if i >= len(args):
                raise ValueError(f"{a} expects a value")
            opts[key] = conv(args[i])
        else:
            raise ValueError(f"unknown option {a!r}")
        i += 1
    return opts


def _load_source(spec: str):
    """Import a developer's ``RecordingSource`` from a ``module:factory`` spec
    and construct it. The factory is called with no arguments — it owns
    producing a fully formed source (descriptor included) for whatever rig it
    wraps; the CLI never guesses at embodiment shape.

    Every failure point names the spec and what went wrong (Rule 10) — no
    silent fallback to simulate. Raises; the caller (``cmd_record``) renders
    the message and exits, the same loud-not-traced path used for the missing-
    extra lantern today."""
    import importlib

    if ":" not in spec:
        raise ValueError(
            f"--source {spec!r} is not MODULE:FACTORY shaped — expected e.g. "
            "'mypkg.rig:make_source'"
        )
    module_name, _, factory_name = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"--source {spec!r}: failed to import module {module_name!r}: {exc}"
        ) from exc
    try:
        factory = getattr(module, factory_name)
    except AttributeError:
        raise RuntimeError(
            f"--source {spec!r}: module {module_name!r} has no attribute {factory_name!r}"
        ) from None
    try:
        return factory()
    except Exception as exc:
        raise RuntimeError(
            f"--source {spec!r}: factory {factory_name!r} raised while constructing "
            f"the source: {exc}"
        ) from exc


def _build_session(opts: dict):
    """Build the Session the frontend drives. ``--source`` loads a developer-
    supplied RecordingSource; ``--simulate`` wires the bundled SimulatedSource
    (unchanged, byte-identical); neither refuses loudly rather than guessing a
    rig. All of it is library construction — no behavior."""
    from newt.recording import (
        BIMANUAL_DESCRIPTOR,
        SINGLE_ARM_DESCRIPTOR,
        Session,
        SimulatedSource,
    )

    if opts["source"] and opts["simulate"]:
        raise ValueError("--source and --simulate are mutually exclusive — pick one.")

    if opts["source"]:
        source = _load_source(opts["source"])
    elif opts["simulate"]:
        descriptor = BIMANUAL_DESCRIPTOR if opts["bimanual"] else SINGLE_ARM_DESCRIPTOR
        source = SimulatedSource(descriptor, drop_every=opts["drop_every"])
    else:
        # The hardware path needs a --source; the CLI refuses loudly rather
        # than guessing a rig.
        print(
            "[newt record] No embodiment wired for live capture from the CLI yet.",
            file=sys.stderr,
        )
        print(
            "        Fix: run with --simulate to exercise the rhythm, or point\n"
            "        --source at your own MODULE:FACTORY (e.g. mypkg.rig:make_source).",
            file=sys.stderr,
        )
        return None

    return Session(
        source,
        task=opts["task"],
        output_dir=opts["dest"],
        state_hz=opts["hz"],
        author=opts["author"],
        license=opts["license"],
        target=opts["target"],
    )


# --------------------------------------------------------------------------- #
# Preflight (frontend courtesy: print the contract, refuse on non-writable dest)
# --------------------------------------------------------------------------- #

def _print_preflight(session, as_json: bool) -> bool:
    """Print the contract the Session describes. Returns False (refuse) only when
    the destination is not writable — the one refusal this skin owns. The library
    never blocks; this is the frontend deciding."""
    report = session.preflight()
    if as_json:
        _emit({"event": "preflight", "contract": report})
    else:
        print("=" * 64)
        print("newt record — preflight contract")
        print("=" * 64)
        print(f"  source        : {report['source_kind']}")
        print(f"  state dims    : {len(report['joint_names'])} joints {report['joint_names']}")
        print(f"  state channels: {', '.join(report['channels'])}")
        print(f"  state rate    : {report['state_hz']} Hz")
        print(f"  destination   : {report['destination']}")
        print(f"  format        : {report['format']}")
        print(f"  task          : {report['task']}")
        if report["cameras"]:
            cams = ", ".join(f"{c['id']}@{c['width']}x{c['height']}/{c['fps']}fps" for c in report["cameras"])
            print(f"  cameras       : {len(report['cameras'])} RGB — {cams}")
        else:
            print("  cameras       : none — state-only capture")
            if report.get("camera_stub_reason"):
                print(f"                  {report['camera_stub_reason']}")
        prov = report["provenance"]
        print(f"  provenance    : author={prov['author']} license={prov['license']} (unverified)")
        if report.get("target"):
            print(f"  target        : {report['target']} kept episode(s)")
        print("=" * 64)

    if not report.get("destination_writable", True):
        msg = f"destination {report['destination']} is not writable."
        fix = "Pick a writable --dest you own, then run newt record again."
        if as_json:
            _emit({"event": "refused", "reason": msg, "fix": fix})
        else:
            print(f"\n[newt record] REFUSING TO RECORD — {msg}", flush=True)
            print(f"        Fix: {fix}", flush=True)
        return False
    return True


# --------------------------------------------------------------------------- #
# Interactive (keyboard) frontend
# --------------------------------------------------------------------------- #

_SPACE = " "
_ENTER = ("\r", "\n")
_CTRL_H = "\x08"


def _run_interactive(session, opts: dict) -> int:
    if not _print_preflight(session, as_json=False):
        session.close()
        return 2

    saved = _enter_cbreak()
    print("\n[newt record] Ctrl+H kills (torque-off, no partial episode).", flush=True)
    target = opts["target"]
    try:
        while True:
            print("\n[session] SPACE to start an episode (Ctrl+C to end the session) …", flush=True)
            if not _wait_for_space(session):
                return 130  # Ctrl+H during idle
            ep_id = session.start_episode()
            print(f"[rec] episode {ep_id} — recording (SPACE to stop) …", flush=True)

            killed = _record_until_stop(session)
            if killed:
                session.kill()
                print("\n[rec] KILLED — torque off, episode discarded (no dir).", flush=True)
                return 130

            st = session.status()
            print(f"\n[rec] stopped — {st.state_count} state frames, {st.dropped_state} dropped.", flush=True)
            report = session.dropped_report()
            if report:
                print(f"[rec] {report}", flush=True)

            verdict = _get_verdict()
            while verdict == "redo":
                session.end_episode(keep=False)
                print("[verdict] REDO — discarded; recording again.", flush=True)
                ep_id = session.start_episode()
                print(f"[rec] episode {ep_id} — recording (SPACE to stop) …", flush=True)
                if _record_until_stop(session):
                    session.kill()
                    print("\n[rec] KILLED — torque off, episode discarded (no dir).", flush=True)
                    return 130
                verdict = _get_verdict()

            if verdict == "keep":
                path = session.end_episode(keep=True)
                print(f"[verdict] KEPT — {path}", flush=True)
            else:
                session.end_episode(keep=False)
                print("[verdict] DISCARDED — no directory written.", flush=True)

            kept = session.status().kept
            if target is not None and kept >= target:
                print(f"\n[session] target reached — {kept}/{target} kept.", flush=True)
                break
    except KeyboardInterrupt:
        print("\n[session] ended by operator.", flush=True)
    finally:
        _restore_cbreak(saved)
        session.close()

    print(f"\n[session] done — {session.status().kept} episode(s) kept under {Path(opts['dest']).resolve()}.", flush=True)
    return 0


def _enter_cbreak():
    if not sys.stdin.isatty():
        return None
    try:
        saved = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return saved
    except (termios.error, OSError):
        return None


def _restore_cbreak(saved) -> None:
    if saved is None:
        return
    try:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
    except (termios.error, ValueError, OSError):
        pass


def _read_key(timeout: float) -> str | None:
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except (ValueError, OSError):
        return None
    if not ready:
        return None
    try:
        return sys.stdin.read(1)
    except (ValueError, OSError):
        return None


def _wait_for_space(session) -> bool:
    """Block until SPACE (start). Returns False if Ctrl+H (kill) is pressed."""
    while True:
        key = _read_key(0.1)
        if key == _CTRL_H:
            return False
        if key == _SPACE:
            return True


def _record_until_stop(session) -> bool:
    """Print the live readout while recording; stop on SPACE. Returns True if
    Ctrl+H (kill) was pressed. The Session is already capturing on its own thread;
    this loop only renders status and watches the keyboard — no capture logic."""
    last = 0.0
    while True:
        key = _read_key(0.05)
        if key == _CTRL_H:
            return True
        if key == _SPACE:
            return False
        now = time.monotonic()
        if now - last >= 0.1:  # ~10 Hz readout refresh
            _live_indicator(session.status())
            last = now


def _live_indicator(st) -> None:
    if st.last_positions:
        first = next(iter(st.last_positions.values()))
        pos = " ".join(f"{p:+.2f}" for p in first)
    else:
        pos = "(waiting for first read)"
    sys.stdout.write(f"\r[rec] frames={st.state_count:5d}  dropped={st.dropped_state:3d}  pos[{pos}]")
    sys.stdout.flush()


def _get_verdict() -> str:
    """ENTER=keep, D=discard, R=redo. cbreak single-key read."""
    print("\n[verdict] ENTER=keep  D=discard  R=redo", flush=True)
    while True:
        key = _read_key(0.5)
        if key is None:
            continue
        if key in _ENTER:
            return "keep"
        low = key.lower()
        if low == "d":
            return "discard"
        if low == "r":
            return "redo"


# --------------------------------------------------------------------------- #
# JSON (agent) frontend — same Session, line-delimited commands + events
# --------------------------------------------------------------------------- #

def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _run_json(session, opts: dict) -> int:
    """Drive the same Session from line-delimited JSON on stdin. Each line is a
    command: {"cmd": "start"} | {"cmd": "stop", "keep": true|false} |
    {"cmd": "status"} | {"cmd": "close"}. Every action emits a JSON event line.
    A door for agents — it drives the library, holds no behavior of its own."""
    if not _print_preflight(session, as_json=True):
        session.close()
        return 2

    target = opts["target"]
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                _emit({"event": "error", "detail": f"not JSON: {line!r}"})
                continue

            action = cmd.get("cmd")
            if action == "start":
                ep_id = session.start_episode()
                _emit({"event": "started", "episode_id": ep_id})
            elif action == "stop":
                keep = bool(cmd.get("keep", True))
                path = session.end_episode(keep=keep)
                state_count, dropped_state = session.last_episode_counts
                st = session.status()
                _emit({
                    "event": "stopped",
                    "kept": keep,
                    "path": str(path) if path else None,
                    "state_count": state_count,
                    "dropped_state": dropped_state,
                    "kept_total": st.kept,
                })
                if target is not None and st.kept >= target:
                    _emit({"event": "target_reached", "kept": st.kept, "target": target})
                    break
            elif action == "status":
                st = session.status()
                _emit({
                    "event": "status",
                    "recording": st.recording,
                    "episode_id": st.episode_id,
                    "state_count": st.state_count,
                    "dropped_state": st.dropped_state,
                    "kept": st.kept,
                    "target": st.target,
                })
            elif action == "close":
                break
            else:
                _emit({"event": "error", "detail": f"unknown cmd {action!r}"})
    finally:
        session.close()
        _emit({"event": "closed", "kept": session.status().kept})
    return 0


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #

def cmd_record(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    try:
        opts = _parse(args)
    except (ValueError, IndexError) as exc:
        print(f"newt record: {exc}", file=sys.stderr)
        print("Run 'newt record --help' for usage.", file=sys.stderr)
        return 1

    if not opts["task"]:
        print("newt record: --task is required (the language prompt recorded per episode).", file=sys.stderr)
        print("        Fix: newt record --task \"pick up the cup\" --simulate", file=sys.stderr)
        return 1

    # Non-TTY without --json: there is no keyboard to read. Stand down loudly.
    if not opts["json"] and not sys.stdin.isatty():
        print(
            "[newt record] stdin is not a TTY and --json was not set — no keyboard to drive the rhythm.",
            file=sys.stderr,
        )
        print(
            "        Fix: run newt record in a real terminal, or use --json to drive it\n"
            "        from an agent with line-delimited stdin commands.",
            file=sys.stderr,
        )
        return 2

    try:
        session = _build_session(opts)
    except Exception as exc:
        # Lantern (missing extra) or construction failure — surface it, don't trace.
        print(f"[newt record] {exc}", file=sys.stderr)
        return 1
    if session is None:
        return 2

    if opts["json"]:
        return _run_json(session, opts)
    return _run_interactive(session, opts)
