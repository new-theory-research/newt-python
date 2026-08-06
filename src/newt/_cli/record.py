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

**Declare, then build.** The factory's declaration is read before the factory is
called, so a rig that is not going to be allowed to do this is refused while it
is still dark. Construction is what connects and energizes; a verb that
validates after it has built has already moved metal it never approved.
"""
from __future__ import annotations

import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

from newt._cli._source_spec import (
    KitDependencyMissing,
    SourceNotResolved,
    build_source,
    import_factory,
    load_source,
    resolve_spec,
)
from newt.teleop import DRIVES_AND_RECORDS


def _usage() -> None:
    print("Usage: newt record [options]")
    print()
    print("  Record NT v0.0.3 episodes from an embodiment. SPACE starts/stops an")
    print("  episode; ENTER keeps it, D discards, R redoes. Ctrl+H is the kill.")
    print()
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
    print("  --view          Serve this session as a page on this machine and print")
    print("                  the URL. The robot, your cameras and your joint traces,")
    print("                  live from the moment the session starts — recording adds")
    print("                  a file, it does not change the picture. The robot is")
    print("                  drawn only if your kit declares a description; without")
    print("                  one you get the cameras, the traces, and a line saying")
    print("                  why there is no body. Needs: pip install \"newt[view]\"")
    print("  --view-port N   Port for that page (default: 9099). The stream it reads")
    print("                  takes the next port up.")
    print("  --control       Let the page start and stop takes, not just watch them.")
    print("                  Adds four routes to the server --view already runs:")
    print("                  POST /api/episode/start, POST /api/episode/stop,")
    print("                  GET /api/session, GET /api/episodes. Without this flag")
    print("                  they answer 404 and say why, so a plain --view stays a")
    print("                  page that can only look. Off by default deliberately:")
    print("                  the server binds every interface, so this is the")
    print("                  difference between anyone on the network watching your")
    print("                  session and anyone on the network recording into it.")
    print("  --page-dir DIR  Serve an already-built page at / instead of the built-in")
    print("                  one, which moves to /view so your page can embed it.")
    print("                  Nothing is built here — point this at a build's output")
    print("                  directory, the one with index.html in it.")
    print("  --teleop        TEMPORARY DOOR — record a demonstration: one embodiment")
    print("                  drives another and the same tick writes the episode.")
    print("                  The factory must declare it does both, and that is read")
    print("                  before anything connects. Resolves its source from")
    print("                  [sources].demonstration, not [sources].record — driving")
    print("                  and reading are different factories on the same rig.")
    print("                  One episode per run; Ctrl+C ends it and keeps it,")
    print("                  Ctrl+H kills (de-energize where it stands, no episode).")
    print("                  Takes neither --target (one episode, no count to stop")
    print("                  at) nor the simulate-only flags; passing one is refused")
    print("                  rather than ignored.")
    print("                  This flag is a bench door pending the naming ruling in")
    print("                  newtrino-030 — expect it to be spelled differently, and")
    print("                  the config key to move with it.")
    print()
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
        "view": False,
        "view_port": None,
        "control": False,
        "page_dir": None,
    }
    flags = {
        "--simulate": "simulate",
        "--bimanual": "bimanual",
        "--json": "json",
        "--teleop": "teleop",
        "--view": "view",
        "--control": "control",
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
        "--view-port": ("view_port", int),
        "--page-dir": ("page_dir", str),
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

    Returns ``(session, source_receipt)``. The receipt is the phrase naming
    where the source came from when an installed kit's lone declaration answered
    on the operator's behalf, and it is a return value rather than a print
    because the place it belongs is the preflight the frontend is about to
    show — not a line ahead of it. It is ``None`` on every path the operator
    chose out loud, ``--simulate`` included. ``(None, None)`` is the refusal.

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

    source_receipt = None
    if opts["simulate"]:
        descriptor = BIMANUAL_DESCRIPTOR if opts["bimanual"] else SINGLE_ARM_DESCRIPTOR
        source = SimulatedSource(descriptor, drop_every=opts["drop_every"])
    else:
        try:
            spec, source_receipt = resolve_spec(
                "record", opts["source"], "mypkg.rig:make_source"
            )
        except SourceNotResolved as exc:
            print(str(exc), file=sys.stderr)
            # The one thing record can offer that teleop and rest cannot: there
            # is a rhythm to exercise here without any rig at all.
            print(
                "        Or exercise the rhythm with no rig at all: newt record --simulate",
                file=sys.stderr,
            )
            return None, None
        source = _load_source(spec)

    return (
        Session(
            source,
            task=opts["task"],
            output_dir=opts["dest"],
            state_hz=opts["hz"],
            author=opts["author"],
            license=opts["license"],
            target=opts["target"],
        ),
        source_receipt,
    )


# --------------------------------------------------------------------------- #
# Preflight (frontend courtesy: print the contract, refuse on non-writable dest)
# --------------------------------------------------------------------------- #

def _print_preflight(session, as_json: bool, source_receipt: str | None = None) -> bool:
    """Print the contract the Session describes. Returns False (refuse) only when
    the destination is not writable — the one refusal this skin owns. The library
    never blocks; this is the frontend deciding.

    ``source_receipt`` names where the source came from when nobody typed it.
    The ``source`` row is where it goes: this block is the first thing `record`
    prints and that row is already the answer to "what am I recording from" —
    an extra line above the block to say the same thing is the notice ruling 1
    removed. On the JSON stream it is a key beside the contract rather than
    inside it, because the contract is the library's report and this fact is the
    frontend's."""
    report = session.preflight()
    if as_json:
        _emit({"event": "preflight", "contract": report, "source": source_receipt})
    else:
        print("=" * 64)
        print("newt record — preflight contract")
        print("=" * 64)
        kind = report["source_kind"]
        print(f"  source        : {kind}{f' — {source_receipt}' if source_receipt else ''}")
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


def _run_interactive(session, opts: dict, source_receipt: str | None = None) -> int:
    from newt.recording import CameraCaptureFailed

    if not _print_preflight(session, as_json=False, source_receipt=source_receipt):
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


def _run_json(session, opts: dict, source_receipt: str | None = None) -> int:
    """Drive the same Session from line-delimited JSON on stdin. Each line is a
    command: {"cmd": "start"} | {"cmd": "stop", "keep": true|false} |
    {"cmd": "status"} | {"cmd": "close"}. Every action emits a JSON event line.
    A door for agents — it drives the library, holds no behavior of its own."""
    from newt.recording import CameraCaptureFailed

    if not _print_preflight(session, as_json=True, source_receipt=source_receipt):
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

#: What a rig sets to say it does both — the public word, from `newt.teleop`, so
#: a kit author has something to import instead of a string copied out of a CLI
#: module. Asked for, never inferred: an object carrying `send_action` next to
#: `read_state` is a shape, and a shape is not a statement that driving and
#: recording the same rig at once is a thing this rig means to do.
_DECLARATION = DRIVES_AND_RECORDS

#: The verb namespace the composed path resolves its source in — separate from
#: `record`, because the factory that reads a rig and the factory that drives it
#: while reading it are different code, and a bench that has declared one has
#: not declared the other. Provisional with the flag: `--teleop` and
#: `[sources].demonstration` are the same naming question and they move together.
_COMPOSED_VERB = "demonstration"

#: What the composed path shows an operator who has nothing declared. Not
#: `make_source` — the factory this path needs builds a different rig.
_COMPOSED_EXAMPLE = "mypkg.rig:make_demo"

#: The command the operator actually typed, which on this path is not the verb
#: the source resolves under. Every other verb answers to its own name, so the
#: resolver defaults the two to the same word; here they diverge, and a refusal
#: that told this operator to retype `newt demonstration --source ...` would be
#: naming a command that does not exist.
_COMPOSED_COMMAND = "record --teleop"

#: The two no-keyboard stand-downs are `newt teleop`'s, borrowed whole. What is
#: NOT borrowable is the fix: `newt teleop` is not the command this operator
#: typed, and running it would drive the rig and write nothing — the thing they
#: were trying to avoid. So this path brings its own, and the one line teleop's
#: version cannot say is what a bounded run does to the episode.
_COMPOSED_SCRIPTED_FIX = (
    "        Fix: run newt record --teleop in a real terminal. To drive it from a\n"
    "        script anyway, bound the session before it starts:\n"
    '          timeout --signal=INT 30 newt record --teleop --task "pick up the cup"\n'
    "        SIGINT lands where Ctrl+C lands: the episode is KEPT as it stands, and\n"
    "        then the rig is put away and de-energized. It is an exit, not a kill — it\n"
    "        moves, and it writes down however much of a demonstration the timeout\n"
    "        allowed.\n"
    "        Not a fix: --json. That is what an agent uses INSTEAD of a keyboard, and\n"
    "        this path refuses it for the same reason it is refusing you."
)

_COMPOSED_UNARMED_FIX = (
    "        Fix: run newt record --teleop from an ordinary interactive shell — not\n"
    "        under a wrapper that owns the terminal (tmux send-keys, an editor's run\n"
    "        pane, a CI runner). Nothing has been connected, nothing has moved, and no\n"
    "        episode was opened."
)

#: Said out loud because the bench hit the other version of it: `newt record`
#: applied tension to two arms it was only reading, so the operator could not
#: move the follower by hand and nothing had told them why. This path energizes
#: too, and the difference is that here it is the point — a verb that drives has
#: to hold the parts it is about to command. Stating it is what makes it a
#: decision rather than an inherited side effect, and it goes after the preflight
#: because by then the factory has already brought the rig up: the operator is
#: standing in front of two live arms as they read it.
_COMPOSED_ENGAGEMENT = (
    "\n[newt record] both arms are up and energized. This verb drives, so it owns "
    "that: the follower is held because it is about to be commanded, and the "
    "leader is held by its own servos and is still the arm you move by hand.\n"
    "              If the leader will not move by hand, stop — that is a rig that "
    "cannot be demonstrated on, and no amount of pulling makes it one."
)


def _refuse_undeclared_factory(spec: str, factory) -> str | None:
    """The gate, read off the factory *before it is called*, or None to proceed.

    Every other refusal on this path costs the operator two connected arms: the
    factory is what brings the rig up and energizes it, so a verb that builds
    first and validates second has already moved metal it never approved. This
    one is free — it happens while the rig is still dark, so "nothing moved" is
    a fact rather than a hope. (Receipt, 2026-08-05: `newt rest` pointed at a
    read-only factory by a one-word typo, which connected and energized a rig on
    its way to a refusal that could not put back what it had just brought up.)

    Import is what happened; construction is what did not. A module-level
    `@drives_and_records` marker is readable from the first and gates the
    second.
    """
    if getattr(factory, _DECLARATION, False) is True:
        return None
    return (
        f"The factory {spec} does not declare {_DECLARATION}, and this verb will not "
        "connect a rig to find out whether it meant to.\n"
        "Yours: the factory was found and NOT called. Recording a demonstration means "
        "one object drives the rig and reports what it drove on the same tick, and a "
        "factory that has not said it builds that is either the wrong one or a rig "
        "that has not been taught this yet.\n"
        "Do now: nothing is connected and nothing is energized — this was checked "
        "before the factory ran, which is why there is nothing to put away.\n"
        f"Then: decorate the factory that builds the driving-and-recording pair with "
        f"`@newt.teleop.drives_and_records` (and set `{_DECLARATION} = True` on the "
        f"object it returns), or point this at that factory if it already exists — "
        f"`newt record --teleop --source {_COMPOSED_EXAMPLE}`."
    )


def _refuse_composed(source) -> str | None:
    """The second check, on the object the factory returned — or None if it holds.

    Four causes, four strings. The presence checks below choose *which* message
    an operator gets; none of them grants the capability — that is the
    declaration's job, and the last branch is the only one that says yes.

    Reaching any of these means the factory carried the declaration and what it
    built does not back it, so every string here says that: the rig is up,
    because a declared factory is allowed to run. Belt and braces to the gate in
    `_refuse_undeclared_factory`, and the only thing that can catch a factory
    that claims more than it builds.
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
            f"Yours: the factory declared {_DECLARATION} and built a recording source — "
            "it reads the rig and drives none of it. That combination is a bug in the "
            "kit, not a wrong flag: the declaration is what let this factory run at "
            "all.\n"
            "Do now: nothing has been recorded, and the rig this factory brought up is "
            "up. Put it away with `newt rest`.\n"
            f"Then: the declared factory has to return the pair built for driving AND "
            "recording — or the declaration belongs on the other factory in that module."
        )
    if drives and not records:
        return (
            "This source drives the rig but has nothing to record from it, so there "
            "would be a demonstration and no episode of it.\n"
            f"Yours: the factory declared {_DECLARATION} and built a teleop source — it "
            "has read_action() and send_action() but no descriptor/read_state() for the "
            "episode to be written from.\n"
            "Do now: nothing has been recorded, and the rig this factory brought up is "
            "up. Put it away with `newt rest`.\n"
            "Then: the declared factory has to return the composed pair — or drop the "
            "declaration and run `newt teleop --source ...`, which is what this object "
            "is actually for."
        )
    if not drives and not records:
        return (
            "This source neither drives nor records — it is not a rig this verb can "
            "use at all.\n"
            f"Yours: the factory declared {_DECLARATION} and returned an object with "
            "neither read_action()/send_action() nor descriptor/read_state(). A factory "
            "that returns a config, a driver handle or None lands exactly here.\n"
            "Do now: nothing has been recorded. Whether anything connected is up to "
            "what that factory did before it returned — check its output above.\n"
            "Then: the declaration is on a factory that does not build a source. Move "
            "it to the one that does."
        )
    if getattr(source, _DECLARATION, False) is not True:
        return (
            f"This source can drive and can record, but the object does not declare "
            f"{_DECLARATION} — only the factory did, and this verb takes the object's "
            "word over the factory's.\n"
            "Yours: the object has the methods for both halves. That is a shape, not a "
            "statement — a part built to be read can be sent actions it will refuse, "
            "and finding that out mid-demonstration is the failure this refusal "
            "exists to prevent.\n"
            "Do now: nothing has been recorded, and the rig this factory brought up is "
            "up. Put it away with `newt rest`.\n"
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
        self.path: Path | None = None
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
    energizes, and a session that cannot be killed must never reach it.

    Two things this path adds to that order. The episode opens after the
    preflight the operator reads and before the first tick. And the factory's
    declaration is read *between* finding it and calling it — declare, then
    build — so a rig that was never going to be allowed to do this is refused
    while it is still dark.
    """
    from newt._cli.teleop import KillKey, _stand_down_no_tty, _stand_down_unarmed
    from newt.teleop import run_session

    if not sys.stdin.isatty():
        return _stand_down_no_tty(
            "newt record", "--teleop needs a TTY", _COMPOSED_SCRIPTED_FIX
        )

    kill_key = KillKey()
    if not kill_key.arm():
        return _stand_down_unarmed(
            "newt record", "this demonstration", _COMPOSED_UNARMED_FIX
        )

    session = None
    view = None
    try:
        try:
            spec, source_receipt = resolve_spec(
                _COMPOSED_VERB,
                opts["source"],
                _COMPOSED_EXAMPLE,
                command=_COMPOSED_COMMAND,
            )
        except SourceNotResolved as exc:
            # No `--simulate` postscript here: this is the one path that forbids
            # it, and offering it as the way out would be advice that refuses.
            print(str(exc), file=sys.stderr)
            return 2

        try:
            factory = import_factory(spec)
        except KitDependencyMissing as exc:
            # Not the refusal below: that one says the factory could not be
            # found, and this factory was found — its own imports weren't. Same
            # site, opposite fix, so they may not share a string (Rule 12).
            print(
                f"\n[newt record] REFUSING TO RECORD — {exc}\n"
                "        Nothing was called, so nothing is connected and nothing is "
                "energized. There is nothing to put away.",
                file=sys.stderr,
            )
            return 1
        except Exception as exc:  # noqa: BLE001 — see comment below: nothing was
            # called yet, so any failure here is a resolve-time problem the operator
            # needs to see as text, not as a traceback.
            # Phase one failed: the name did not resolve to a callable. Nothing
            # was called, so the honest thing to say is that the rig is still
            # dark — which is also what tells the operator this is a spelling
            # problem and not a hardware one.
            print(
                f"\n[newt record] REFUSING TO RECORD — the factory this source names "
                f"could not be found.\n"
                f"Yours: {exc}\n"
                "Do now: nothing was called, so nothing is connected and nothing is "
                "energized. There is nothing to put away.\n"
                "Then: fix the spec — check the module is installed in this "
                "environment and that the factory is spelled the way the module "
                "exports it.",
                file=sys.stderr,
            )
            return 1

        refusal = _refuse_undeclared_factory(spec, factory)
        if refusal is not None:
            print(f"\n[newt record] REFUSING TO RECORD — {refusal}", file=sys.stderr)
            return 2

        try:
            source = build_source(spec, factory)
        except KeyboardInterrupt:
            print(
                "\n[newt record] bring-up interrupted (Ctrl+C) — nothing was recorded "
                "and the session never started. Check what the source reported about "
                "anything it had already connected.",
                file=sys.stderr,
            )
            return 130
        except KitDependencyMissing as exc:
            # The import failed inside the call that connects, so unlike the
            # phase-one twin above this one cannot claim the rig is dark: a kit
            # that acquires its driver partway through bring-up has already
            # reached whatever it reached. Say what is known, not what is
            # comfortable.
            print(
                f"\n[newt record] REFUSING TO RECORD — {exc}\n"
                "        This is the call that connects, so if the factory had brought "
                "anything up before the import failed, `newt rest` puts it away.",
                file=sys.stderr,
            )
            return 1
        except Exception as exc:  # noqa: BLE001 — the rig's own factory can raise
            # anything while bringing arbitrary hardware up; this call is what
            # connects and energizes, so the failure must be reported (with the
            # put-away instruction below), never let propagate as a bare traceback.
            # Phase two failed: the factory was found, it had declared itself,
            # and it raised partway through bringing the rig up. The state of
            # the rig is the thing that separates this from the refusal above —
            # this is the call that connects and energizes, so whatever it had
            # reached before it raised is reached, and saying "nothing is
            # connected" here would be a comfortable lie.
            print(
                f"\n[newt record] REFUSING TO RECORD — the rig's own factory raised "
                f"while building it.\n"
                f"Yours: {exc}\n"
                "Do now: this is the call that connects and energizes, so anything it "
                "had brought up before it raised is up. Put it away with `newt rest` "
                "before walking up to it.\n"
                "Then: the message above is the kit's, not this verb's — that factory "
                "is where the fix is.",
                file=sys.stderr,
            )
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
        # The receipt lands once on this path too, and it lands here rather than
        # in the teleop line below: the preflight is what the operator reads
        # before two arms come up, and `record --teleop` prints both.
        if not _print_preflight(session, as_json=False, source_receipt=source_receipt):
            return 2

        if opts["view"]:
            try:
                view = _open_view(session, opts)
            except Exception as exc:  # noqa: BLE001 — same reason as the plain path:
                # the refusal already names its cause, owner and next step. But this
                # path has energized arms behind it, so the refusal has to say so.
                print(
                    f"[newt record] the live view did not start.\n{exc}\n"
                    "        The arms are up: this failed after the factory built the "
                    "rig. `newt rest` puts them away.",
                    file=sys.stderr,
                )
                return 1

        print(_COMPOSED_ENGAGEMENT, flush=True)

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
        if view is not None:
            view.close()
        if session is not None:
            session.close()


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #

#: Said once, in one place, on both frontends. A human reads it as a line under
#: the URLs; an agent reads it as a field. Two spellings of this sentence would
#: eventually become two different claims about the same rig.
_NO_ROBOT_NOTE = (
    "no robot drawn — this rig's source declares no description. "
    "Cameras and joint traces are live."
)


def _open_view(session, opts: dict, *, as_json: bool = False):
    """Start the live view, attach it to the session, say where to open it.

    A frontend courtesy in exactly the shape the rest of this file is one: the
    library decides what a view is and what it publishes, and this reports the
    address. To a person that is two lines at most — the URL for the browser on
    this machine, and, when this machine can name itself on a network, the one a
    laptop beside it can use. No second address is printed when it would only mean
    "this desk", because a link that is true for one reader and false for another
    is worse than one line.

    To an agent it is one ``view`` event on the same stdout stream every other
    event uses. ``--json`` promises line-delimited JSON and nothing else, so three
    bare ``[view] …`` lines ahead of the first record would break the parse of the
    caller that took that promise literally — which is every caller worth having.
    """
    from newt.live import DEFAULT_PORT, LiveView, SessionControl

    port = opts["view_port"] or DEFAULT_PORT
    control = None
    if opts["control"]:
        # No sink factory: the Session's own LocalSink already put the episode where
        # it belongs, and a second delivery to the same place would be one delivery
        # too many. A caller with a store to push to builds this same object with a
        # `sink_for` and gets the push states; `newt record` is not that caller.
        control = SessionControl(session, dataset_root=opts["dest"])
    view = LiveView(
        descriptor=session.descriptor,
        task=opts["task"],
        source_kind=session.describe()["source_kind"],
        camera_ids=session.camera_ids,
        declaration=session.view_declaration,
        port=port,
        grpc_port=port + 1,
        control=control,
        page_dir=opts["page_dir"],
    )
    view.start()
    session.attach_observer(view)
    local, shareable = view.urls()
    robot_drawn = session.view_declaration is not None
    if as_json:
        _emit({
            "event": "view",
            "url": local,
            "network_url": shareable,
            "robot_drawn": robot_drawn,
            "control": bool(control),
            "page_dir": opts["page_dir"],
            "note": None if robot_drawn else _NO_ROBOT_NOTE,
        })
    else:
        print()
        print(f"[view] {local}")
        if shareable:
            print(f"[view] {shareable}  (from another machine on this network)")
        if not robot_drawn:
            print(f"[view] {_NO_ROBOT_NOTE}")
        if control:
            # Said out loud, every time. The flag was typed once and the session runs
            # for hours; whoever walks up to this terminal should be able to read off
            # it that the page on the network can record, not just watch.
            print(
                "[view] this page can start and stop takes — anyone who can open it "
                "can record into this session"
            )
        if opts["page_dir"]:
            print(f"[view] serving {opts['page_dir']} at /; the live view is at /view")
    return view


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

    # Both of these only exist as behaviour of the server --view runs. Passing one
    # without it would otherwise be accepted and do nothing, which is the silent
    # no-op Rule 10 is about — and the two get different sentences because a person
    # who asked for control and a person who built a page have different next steps.
    if opts["control"] and not opts["view"]:
        print(
            "[newt record] --control without --view: there is no page to control "
            "this session from. --control adds routes to the server --view runs, "
            "and without --view that server is never started.",
            file=sys.stderr,
        )
        print("        Fix: add --view.", file=sys.stderr)
        return 1
    if opts["page_dir"] and not opts["view"]:
        print(
            "[newt record] --page-dir without --view: nothing would serve that "
            "directory. --page-dir replaces the page the --view server hands out, "
            "and without --view there is no server and no page.",
            file=sys.stderr,
        )
        print(
            "        Fix: add --view. Add --control too if that page is meant to "
            "start and stop takes rather than watch them.",
            file=sys.stderr,
        )
        return 1

    if opts["teleop"]:
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
        # Three flags this path cannot honour. Accepting them and doing nothing
        # is the failure Rule 10 names: the operator asked for something, the
        # verb said nothing, and the run looks like it obeyed. Two causes, not
        # three — a flag that only ever meant something to the simulated stream
        # is a different mistake from a flag that means something real and means
        # it per-session rather than per-run.
        simulate_only = [
            flag
            for flag, key in (("--bimanual", "bimanual"), ("--drop-every", "drop_every"))
            if opts[key]
        ]
        if simulate_only:
            print(
                f"[newt record] --teleop and {' and '.join(simulate_only)}: "
                f"{'those flags shape' if len(simulate_only) > 1 else 'that flag shapes'} "
                "the simulated joint stream, and a demonstration is a real rig being "
                "moved by a real hand. There is no stream here to shape.",
                file=sys.stderr,
            )
            print(
                "        Fix: drop "
                + " and ".join(simulate_only)
                + ". What the rig is and how many arms it has comes from the factory "
                "your kit declares, not from a flag.",
                file=sys.stderr,
            )
            return 1
        if opts["target"] is not None:
            print(
                f"[newt record] --teleop and --target {opts['target']}: the composed "
                "path records exactly one episode per run — Ctrl+C ends the session and "
                "keeps it — so there is no count for it to stop at.",
                file=sys.stderr,
            )
            print(
                "        Fix: drop --target and run the command once per demonstration. "
                "The rig is put away between runs, which is the ending you want between "
                "takes anyway.",
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
        session, source_receipt = _build_session(opts)
    except Exception as exc:  # noqa: BLE001 — Lantern (missing extra) or a
        # construction failure — surface it, don't trace.
        print(f"[newt record] {exc}", file=sys.stderr)
        return 1
    if session is None:
        return 2

    view = None
    if opts["view"]:
        try:
            view = _open_view(session, opts, as_json=opts["json"])
        except Exception as exc:  # noqa: BLE001 — every LiveViewUnavailable already
            # names its cause, its owner and the next step; a traceback on top of
            # one of those is noise. A port refusal reads the same way.
            print(f"[newt record] the live view did not start.\n{exc}", file=sys.stderr)
            session.close()
            return 1

    try:
        if opts["json"]:
            return _run_json(session, opts, source_receipt)
        return _run_interactive(session, opts, source_receipt)
    finally:
        if view is not None:
            view.close()
