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
- a camera bridge that died mid-episode makes the keep a refusal: the Session
  raises, nothing is committed, and the process exits 3 — a distinct code from
  the bad-invocation 1 and the frontend's own refusal 2;
- a kept-count toward ``--target``;
- ``--json`` drives the same Session from line-delimited stdin commands and emits
  line-delimited events — for an agent. Agents are a door, not load-bearing.
- non-TTY without ``--json`` stands down loudly: there is no keyboard to read.

``--teleop`` is a second rhythm on the same Session and a temporary door: one
embodiment drives another and the same tick writes the episode, so one command
records a demonstration — someone moving a rig while the rig writes down what
they did. It borrows the teleop grammar whole — the kill key
armed before anything connects, Ctrl+H de-energizes where the arms stand,
Ctrl+C puts them away — and adds one decision of its own: the kill discards the
episode, every other ending keeps it, and which one happened is printed rather
than left to the exit code. The flag is spelled provisionally; newtrino-030
names it.
"""
from __future__ import annotations

import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

from newt._cli._source_spec import SourceNotResolved, load_source, resolve_spec


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
    print("  --source SPEC   Which RecordingSource to run — either a short name")
    print("                  your kit declares, or a full MODULE:FACTORY import")
    print("                  path (e.g. mypkg.rig:make_source), which needs no")
    print("                  declaration. Mutually exclusive with --simulate.")
    print("                  Optional on a configured rig: with neither flag the")
    print("                  verb reads [sources].record from your site config")
    print("                  ($NT_SITE_CONFIG, else ~/.config/nt/nt.toml), else the")
    print("                  one your kit declares. Either flag wins over the file.")
    print("  --bimanual      (simulate) Drive a 2-arm leader/follower stream.")
    print("  --target N      Stop after N kept episodes.")
    print("  --hz N          State sample rate (default: 30).")
    print("  --author TEXT   Declared author written to episode.json provenance.")
    print("  --license TEXT  Declared license written to episode.json provenance.")
    print("  --drop-every N  (simulate) Inject a dropped read every N ticks.")
    print("  --json          Agent mode: line-delimited JSON events + stdin commands.")
    print("  --teleop        TEMPORARY DOOR — record a demonstration: one embodiment")
    print("                  drives another and the same tick writes the episode.")
    print("                  Needs a --source that declares it does both.")
    print("                  One episode per run; Ctrl+C ends it and keeps it,")
    print("                  Ctrl+H kills (de-energize where it stands, no episode).")
    print("                  This flag is a bench door pending the naming ruling in")
    print("                  newtrino-030 — expect it to be spelled differently.")
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
        "teleop": False,
    }
    flags = {
        "--simulate": "simulate",
        "--bimanual": "bimanual",
        "--json": "json",
        "--teleop": "teleop",
    }
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


# The MODULE:FACTORY contract lives in _source_spec — one definition, two verbs.
# This alias stays because it is the name the frontend and its tests call.
_load_source = load_source


def _build_session(opts: dict):
    """Build the Session the frontend drives. ``--source`` loads a developer-
    supplied RecordingSource; ``--simulate`` wires the bundled SimulatedSource
    (unchanged, byte-identical); with neither flag the rig's own ``[sources]``
    declaration answers, and a rig that declares nothing is refused loudly
    rather than guessed at. All of it is library construction — no behavior.

    Precedence: either flag beats the file. ``--simulate`` in particular beats a
    configured ``[sources].record`` — an operator asking for simulation out loud
    is not overridden by a config file."""
    from newt.recording import (
        BIMANUAL_DESCRIPTOR,
        SINGLE_ARM_DESCRIPTOR,
        Session,
        SimulatedSource,
    )

    if opts["source"] and opts["simulate"]:
        raise ValueError("--source and --simulate are mutually exclusive — pick one.")

    if opts["simulate"]:
        descriptor = BIMANUAL_DESCRIPTOR if opts["bimanual"] else SINGLE_ARM_DESCRIPTOR
        source = SimulatedSource(descriptor, drop_every=opts["drop_every"])
    else:
        try:
            spec = resolve_spec("record", opts["source"], "mypkg.rig:make_source")
        except SourceNotResolved as exc:
            print(str(exc), file=sys.stderr)
            # The one thing record can offer that teleop and rest cannot: there
            # is a rhythm to exercise here without any rig at all.
            print(
                "        Or exercise the rhythm with no rig at all: newt record --simulate",
                file=sys.stderr,
            )
            return None
        source = _load_source(spec)

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
    from newt.recording import CameraCaptureFailed

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
            if st.frame_counts:
                cams = ", ".join(
                    f"{cam_id}: {n} frames, {st.dropped_frames.get(cam_id, 0)} dropped"
                    for cam_id, n in sorted(st.frame_counts.items())
                )
                print(f"[rec] cameras — {cams}.", flush=True)
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
                try:
                    path = session.end_episode(keep=True)
                except CameraCaptureFailed as exc:
                    print(f"\n[newt record] EPISODE REFUSED — {exc}", file=sys.stderr, flush=True)
                    return 3
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
    line = f"\r[rec] frames={st.state_count:5d}  dropped={st.dropped_state:3d}  pos[{pos}]"
    if st.frame_counts:
        cams = " ".join(f"{cam_id}={n}" for cam_id, n in sorted(st.frame_counts.items()))
        line += f"  cam[{cams}]"
        camdrop = sum(st.dropped_frames.values())
        if camdrop:
            line += f" camdrop={camdrop}"
    sys.stdout.write(line)
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
    from newt.recording import CameraCaptureFailed

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
                try:
                    path = session.end_episode(keep=keep)
                except CameraCaptureFailed as exc:
                    _emit({"event": "refused", "cause": exc.cause, "reason": str(exc)})
                    return 3
                state_count, dropped_state = session.last_episode_counts
                frame_counts, dropped_frames = session.last_episode_frames
                st = session.status()
                _emit({
                    "event": "stopped",
                    "kept": keep,
                    "path": str(path) if path else None,
                    "state_count": state_count,
                    "dropped_state": dropped_state,
                    "frame_counts": frame_counts,
                    "dropped_frames": dropped_frames,
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
                    "frame_counts": st.frame_counts,
                    "dropped_frames": st.dropped_frames,
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
# --teleop: the demonstration door (temporary — newtrino-030 names it)
#
# The rest of this file is a skin on Session. This section is a skin on two
# libraries at once: newt.teleop owns the tick, the kill and the two endings;
# newt.recording.Session owns the episode. Neither learns about the other — the
# tick hands its state to a recorder, and the recorder below is the thirty lines
# that turn "a tick happened" into "an episode has one more frame in it".
#
# One process, because the arms take one client. One episode per run, because
# the keyboard is already spoken for: Ctrl+H is the kill and Ctrl+C is the end,
# and a SPACE rhythm on top of a live teleop loop is a second grammar this door
# is too temporary to introduce.
# --------------------------------------------------------------------------- #

#: What a rig sets to say it does both. Asked for, never inferred: an object
#: carrying `send_action` next to `read_state` is a shape, and a shape is not a
#: statement that driving and recording the same rig at once is a thing this rig
#: means to do. The name is provisional with the flag.
_DECLARATION = "drives_and_records"


def _refuse_composed(source) -> str | None:
    """The refusal for a source that cannot do this, or None if it can.

    Four causes, four strings. The presence checks below choose *which* message
    an operator gets; none of them grants the capability — that is the
    declaration's job, and the last branch is the only one that says yes.
    """
    drives = all(
        callable(getattr(source, member, None))
        for member in ("read_action", "send_action", "moving_parts")
    )
    records = callable(getattr(source, "read_state", None)) and getattr(
        source, "descriptor", None
    ) is not None

    if records and not drives:
        return (
            "This source records the rig but does not drive it, so nothing would move "
            "and the episode would be a rig sitting still.\n"
            "Yours: the --source you passed is a recording source — it reads the rig and "
            "drives none of it. Running it under --teleop is the one thing that looks "
            "like it should work and does not.\n"
            "Do now: nothing has been recorded. Whatever the factory connected is "
            "connected.\n"
            "Then: point --source at the factory that builds the pair for driving AND "
            f"recording (it declares {_DECLARATION}), or drop --teleop and record the "
            "arms as something else moves them."
        )
    if drives and not records:
        return (
            "This source drives the rig but has nothing to record from it, so there "
            "would be a demonstration and no episode of it.\n"
            "Yours: the --source you passed is a teleop source — it has read_action() "
            "and send_action() but no descriptor/read_state() for the episode to be "
            "written from.\n"
            "Do now: nothing has been recorded. Whatever the factory connected is "
            "connected.\n"
            "Then: use the factory that builds the composed pair, or run `newt teleop "
            "--source ...` if driving without recording is what you meant."
        )
    if not drives and not records:
        return (
            "This source neither drives nor records — it is not a rig this verb can "
            "use at all.\n"
            "Yours: the object --source built has neither read_action()/send_action() "
            "nor descriptor/read_state().\n"
            "Do now: nothing has been recorded. Whatever the factory connected is "
            "connected.\n"
            "Then: check that MODULE:FACTORY is the one you meant — a factory that "
            "returns a config, a driver handle or None lands exactly here."
        )
    if getattr(source, _DECLARATION, False) is not True:
        return (
            f"This source can drive and can record, but it does not declare "
            f"{_DECLARATION}, so this verb will not decide for it that doing both at "
            "once is safe on this rig.\n"
            "Yours: the object has the methods for both halves. That is a shape, not a "
            "statement — a part built to be read can be sent actions it will refuse, "
            "and finding that out mid-demonstration is the failure this refusal "
            "exists to prevent.\n"
            "Do now: nothing has been recorded. Whatever the factory connected is "
            "connected.\n"
            f"Then: set `{_DECLARATION} = True` on the object your factory returns, "
            "once the part it drives is built the way driving needs."
        )
    return None


class _EpisodeRecorder:
    """One episode, fed by the teleop tick and closed out at the ending.

    The frontend's translation layer and nothing else: `newt.teleop` calls two
    methods, this calls three Session methods, and no decision about episodes
    lives in between. It prints, because saying what happened to the recording
    at the moment it happens is a frontend's job — and because "kept" and
    "discarded" must never be something an operator has to infer from an exit
    code.
    """

    def __init__(self, session) -> None:
        self._session = session
        self.episode_id = session.start_episode()
        self.path: "Path | None" = None
        self.refused = False

    def record_tick(self, channels, ts_ns: int) -> None:
        self._session.feed_state(channels, ts_ns)

    def finish(self, *, keep: bool) -> None:
        from newt.recording import CameraCaptureFailed

        if not keep:
            self._session.end_episode(keep=False)
            print(
                f"\n[newt record] episode {self.episode_id} DISCARDED — the kill fired, "
                "and a panic stop is not a demonstration. No directory was written.",
                flush=True,
            )
            return
        try:
            self.path = self._session.end_episode(keep=True)
        except CameraCaptureFailed as exc:
            self.refused = True
            print(
                f"\n[newt record] episode {self.episode_id} REFUSED — {exc}",
                file=sys.stderr,
                flush=True,
            )
            return
        state_count, dropped = self._session.last_episode_counts
        frames, dropped_frames = self._session.last_episode_frames
        line = (
            f"\n[newt record] episode {self.episode_id} KEPT — {self.path}\n"
            f"              {state_count} state frames, {dropped} dropped"
        )
        if frames:
            line += "; " + ", ".join(
                f"camera {cam_id}: {n} frames, {dropped_frames.get(cam_id, 0)} dropped"
                for cam_id, n in sorted(frames.items())
            )
        print(line, flush=True)


def _run_teleop(opts: dict) -> int:
    """`newt record --teleop` — drive one embodiment from another, and write it down.

    The order is the safety argument, and it is `newt teleop`'s: the kill key is
    armed before the factory runs, because the factory is what connects and
    energizes, and a session that cannot be killed must never reach it. What is
    new is only where the episode opens — after the preflight the operator reads,
    and before the first tick.
    """
    from newt._cli.teleop import KillKey, _stand_down_no_tty, _stand_down_unarmed
    from newt.teleop import run_session

    if not sys.stdin.isatty():
        return _stand_down_no_tty()

    kill_key = KillKey()
    if not kill_key.arm():
        return _stand_down_unarmed()

    session = None
    try:
        try:
            source = _load_source(opts["source"])
        except KeyboardInterrupt:
            print(
                "\n[newt record] bring-up interrupted (Ctrl+C) — nothing was recorded "
                "and the session never started. Check what the source reported about "
                "anything it had already connected.",
                file=sys.stderr,
            )
            return 130
        except Exception as exc:
            print(f"[newt record] {exc}", file=sys.stderr)
            return 1

        refusal = _refuse_composed(source)
        if refusal is not None:
            print(f"\n[newt record] REFUSING TO RECORD — {refusal}", file=sys.stderr)
            return 2

        from newt.recording import Session

        session = Session(
            source,
            task=opts["task"],
            output_dir=opts["dest"],
            # One clock. The tick that drives the rig is the tick that writes
            # the frame, so the Session must not also be polling: --hz is
            # that one number, and the rig's own planner rate is its business.
            state_pushed=True,
            state_hz=opts["hz"],
            author=opts["author"],
            license=opts["license"],
        )
        if not _print_preflight(session, as_json=False):
            return 2

        recorder = _EpisodeRecorder(session)
        print(
            f"\n[newt record] recording episode {recorder.episode_id} while you drive. "
            "Ctrl+C ends the session and KEEPS it; Ctrl+H kills — de-energize where "
            "the arms stand, no episode.",
            flush=True,
        )
        rc = run_session(
            source, rate_hz=float(opts["hz"]), kill=kill_key.fired, recorder=recorder
        )
        if recorder.refused and rc == 0:
            # The rig ended cleanly; the episode did not. Same code plain record
            # uses for a recording it would not commit.
            return 3
        return rc
    finally:
        kill_key.restore()
        if session is not None:
            session.close()


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

    if opts["teleop"]:
        if not opts["source"]:
            print(
                "newt record --teleop: --source is required — it is what knows which "
                "embodiment drives which, and how to build the driven one so that it "
                "can be driven.",
                file=sys.stderr,
            )
            print(
                "        Fix: newt record --teleop --source mypkg.rig:make_demo "
                "--task \"pick up the cup\"",
                file=sys.stderr,
            )
            return 1
        if opts["simulate"]:
            print(
                "[newt record] --teleop and --simulate: there is no simulated "
                "demonstration. A recorded demonstration is a person moving a real "
                "arm; with no rig there is nothing to move and nothing to write down.",
                file=sys.stderr,
            )
            print(
                "        Fix: drop --teleop to prove the episode format against the "
                "simulated stream, or run --teleop on the bench with both arms up.",
                file=sys.stderr,
            )
            return 1
        if opts["json"]:
            print(
                "[newt record] --teleop and --json: this loop moves hardware, so it "
                "refuses to run without a keyboard, and --json is what an agent uses "
                "*instead* of one. There is no kill key on that path.",
                file=sys.stderr,
            )
            print(
                "        Fix: run --teleop in a real terminal. Whether an agent should "
                "ever drive a demonstration — and what its kill would be — is "
                "unanswered, and this verb will not answer it by accident.",
                file=sys.stderr,
            )
            return 1
        return _run_teleop(opts)

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
