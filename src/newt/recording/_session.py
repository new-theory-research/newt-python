"""``newt.recording.Session`` — the layering invariant lives here.

ALL recording behavior is in this class: the capture loop, episode lifecycle,
dropped-frame counting, atomicity, the kill path, the descriptive preflight.
Frontends (``newt record``, an agent driving ``--json``, a future TUI) hold ZERO
behavior — they translate input to ``start_episode()`` / ``end_episode(keep=)`` /
``status()`` / ``close()`` and render what those return. If logic about episodes,
format, atomicity, or timing appears in a frontend, it is in the wrong place.

The capture loop runs on a background thread between ``start_episode()`` and
``end_episode()``, polling the RecordingSource at the state rate and writing every
read into the in-flight episode. A frontend that wants a live readout calls
``status()`` on its own cadence; it never touches the loop or the writer.

One Session does not poll: ``state_pushed=True`` says the caller's own loop is
the clock and hands each tick over through ``feed_state()``. That exists for the
loop that is already driving the hardware it would otherwise be polled for —
single-client arms cannot answer two clocks — and it changes who calls, never
what is written.

Generalized from rebot-bench's ``record_session.py``: the rhythm, the dropped-
frame report, the kill-leaves-no-dir guarantee, and the simulate-via-fake-source
pattern are all preserved — lifted out of the keyboard script into the library so
every frontend gets them for free.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from newt.recording._seam import (
    CameraCaptureFailed,
    RecordingSource,
    StateDescriptor,
    camera_failure_message,
)
from newt.recording._sink import LocalSink, Sink

# Community-native rate: Molmo extraction is 30fps/30Hz; this is the spec
# state_frequency default. Override per Session for a different rig.
DEFAULT_STATE_HZ = 30


@dataclass
class SessionStatus:
    """A read-only snapshot a frontend renders. No behavior — pure data."""

    recording: bool
    episode_id: str | None
    state_count: int
    dropped_state: int
    kept: int
    target: int | None
    last_positions: dict[str, list[float]] | None
    closed: bool
    # Per-camera counts for the in-flight episode. Empty on a state-only session,
    # which is what a frontend renders against — the camera lines appear only for
    # a rig that actually declared cameras.
    frame_counts: dict[str, int] = field(default_factory=dict)
    dropped_frames: dict[str, int] = field(default_factory=dict)


class Session:
    """One recording session against one embodiment.

    Construct with a RecordingSource-shaped object (``read_state`` + ``descriptor``,
    optional ``disable_all`` / ``close``), a task prompt, and an output directory.
    Then drive the rhythm: ``start_episode()`` opens a fresh episode and spins up
    the capture loop; ``end_episode(keep=True)`` commits it atomically (or
    ``keep=False`` discards it, leaving no directory). ``status()`` returns a
    snapshot at any time; ``close()`` tears down (torque-off via the source's
    ``disable_all``) and is safe to call from a kill path.
    """

    def __init__(
        self,
        source: RecordingSource,
        task: str,
        output_dir: str | Path,
        *,
        cameras: list | None = None,
        state_hz: int = DEFAULT_STATE_HZ,
        author: str | None = None,
        license: str | None = None,
        camera_stub_reason: str | None = None,
        target: int | None = None,
        sink: Sink | None = None,
        state_pushed: bool = False,
    ) -> None:
        if not hasattr(source, "read_state") or not hasattr(source, "descriptor"):
            raise TypeError(
                "Session(source=...) needs a RecordingSource: an object with a "
                "`descriptor` and a `read_state()` method. See newt.recording.RecordingSource."
            )
        self._source = source
        self._task = task
        self._dest = Path(output_dir)
        self._state_hz = state_hz
        self._period_s = 1.0 / state_hz
        self._target = target
        self._camera_stub_reason = camera_stub_reason
        self._sink = sink if sink is not None else LocalSink(self._dest)
        # Who owns the state clock. False (the default, and every existing
        # caller) is this Session: start_episode spins the capture thread and it
        # polls the source at the state rate. True means the caller's own loop is
        # the clock and pushes each tick through feed_state — the case where the
        # same tick that drives an arm is the one that has to record it, and two
        # clocks on single-client hardware are not an option.
        self._state_pushed = state_pushed

        # Camera specs (CameraSpec instances) are accepted but state-only capture
        # is the default; the writer imports lazily, so we hold raw specs here.
        # A source that opened cameras declares them itself — adopting them here
        # (rather than in a frontend) is what keeps `newt record` a skin with no
        # camera flag and no camera identity.
        self._cameras = list(cameras) if cameras else list(getattr(source, "cameras", None) or [])
        # Whether the source polls itself (a rig bridge) or the caller pushes with
        # feed_frame. Only a source bridge gets a camera thread.
        self._reads_frames = callable(getattr(source, "read_frames", None))
        self._check_camera_bridge(source, declared_by_caller=bool(cameras))
        self._check_cameras_are_declarable()

        # Provenance — declared, never verified (local-first capture).
        from newt.recording._writer import DEFAULT_AUTHOR, DEFAULT_LICENSE

        self._author = author or DEFAULT_AUTHOR
        self._license = license or DEFAULT_LICENSE

        self._writer = None  # set during an episode
        self._loop_thread: threading.Thread | None = None
        self._camera_thread: threading.Thread | None = None
        self._camera_failure: tuple[str, Exception] | None = None
        self._stop_loop = threading.Event()
        self._lock = threading.Lock()
        self._kept = 0
        self._last_positions: dict[str, list[float]] | None = None
        self._last_state_count = 0
        self._last_dropped_state = 0
        self._last_frame_counts: dict[str, int] = {}
        self._last_dropped_frames: dict[str, int] = {}
        self._closed = False

        # Observers: the second sink. Every read the capture loops take is offered
        # to each observer as well as to the episode writer — one read of the
        # hardware, two destinations. An observer never gates capture and never
        # changes what is written; see attach_observer().
        self._observers: list = []
        self._observer_failures: list[tuple[str, Exception]] = []
        # True once an observer is attached: the loops then run for the whole
        # session rather than only between start_episode and end_episode, because
        # a live view that goes dark whenever nobody is recording is not a preview.
        self._streaming = False

    def _check_camera_bridge(self, source, *, declared_by_caller: bool) -> None:
        """Cameras and the read that fills them are one contract — refuse half of it.

        Both halves missing is the common case (state-only capture) and is fine.
        Either half alone would record something no one asked for: declared cameras
        with no reader means empty video files and an episode.json naming cameras
        that never delivered; a reader with nothing declared means frames pulled
        off hardware and thrown away while the episode claims to be state-only.

        The contract is about the SOURCE's two halves, so it is checked only when
        the source is the one declaring cameras. A caller that passes ``cameras=``
        explicitly owns its own bridge and pushes through ``feed_frame`` — the
        pattern that method's docstring has always described — and asking that
        caller for a ``read_frames()`` would demand a second, racing bridge.
        """
        has_reader = self._reads_frames
        if self._cameras and not has_reader and not declared_by_caller:
            raise TypeError(
                f"This source declares {len(self._cameras)} camera(s) "
                f"({', '.join(str(c.id) for c in self._cameras)}) but has no "
                "read_frames() to deliver their frames.\n"
                "Yours: a source that opens cameras owns reading them — the library "
                "never opens or polls a camera itself.\n"
                "Fix: add `read_frames(self) -> dict[camera id, frame | None]` to the "
                "source, or stop declaring `cameras` and record state only."
            )
        if has_reader and not self._cameras:
            raise TypeError(
                "This source has read_frames() but declares no cameras, so every "
                "frame it reads would be discarded and the episode would claim to "
                "be state-only.\n"
                "Yours: `cameras` is what the episode declares and what the writer "
                "opens encoders for; read_frames() alone declares nothing.\n"
                "Fix: expose `cameras` as a list of newt.recording.CameraSpec — one "
                "per camera the source actually opened — or drop read_frames()."
            )

    def _check_cameras_are_declarable(self) -> None:
        """Refuse a camera set the episode cannot honestly describe.

        ``episode.json``'s ``camera_config`` carries ONE width, height and fps for
        the whole set (v0.0.3), so a rig whose cameras disagree would be described
        by whichever one came first and described wrongly for the rest. That is the
        identity-extrinsics failure in a different costume: a shaped-right number
        standing in for one nobody measured. Refuse at construction — before an
        operator spends three minutes recording something the file will misdescribe.

        This does not decide the format question (per-camera dimensions); it
        refuses to answer it by accident.
        """
        if len(self._cameras) < 2:
            return
        shapes = {(c.width, c.height, c.fps) for c in self._cameras}
        if len(shapes) == 1:
            return
        declared = ", ".join(f"{c.id}={c.width}x{c.height}@{c.fps}fps" for c in self._cameras)
        raise TypeError(camera_failure_message("resolution_not_declarable", declared))

    @property
    def last_episode_counts(self) -> tuple[int, int]:
        """(state_count, dropped_state) of the most recently ended episode. A
        frontend reports these after end_episode(), since the live status() drops
        to zero once nothing is recording."""
        return self._last_state_count, self._last_dropped_state

    @property
    def last_episode_frames(self) -> tuple[dict[str, int], dict[str, int]]:
        """(frames written, frames dropped) per camera id for the most recently
        ended episode. Both empty for a state-only session — a frontend renders
        the camera lines only when there were cameras."""
        return dict(self._last_frame_counts), dict(self._last_dropped_frames)

    # --- the descriptive preflight (reverse-contract courtesy) --------------

    @property
    def descriptor(self) -> StateDescriptor:
        return self._source.descriptor

    @property
    def view_declaration(self):
        """The source's ``ViewDeclaration``, or None if it declares no robot.

        Read-only and pass-through on purpose. A live view needs a description
        file, an entity prefix and a joint mapping; all three are the kit's to
        state and none of them is newt's to infer, so this hands over what the
        source said and nothing else. None means the kit declared nothing, which
        a view renders as an empty pane with a sentence — never a stand-in body.
        """
        return getattr(self._source, "view_declaration", None)

    @property
    def camera_ids(self) -> list[str]:
        """The ids of the cameras this session records, in declared order."""
        return [str(c.id) for c in self._cameras]

    def describe(self) -> dict:
        """Describe the exact contract this session will record — the reverse
        contract expressed descriptively. The library DESCRIBES; a frontend
        decides whether to refuse. This never blocks and never exits.

        Returns the superset capture shape: source, joints, channels, rate,
        destination, format, cameras, provenance. A frontend prints it; the
        Session has no opinion about whether the user proceeds.
        """
        d = self.descriptor
        return {
            "source_kind": getattr(self._source, "source_kind", type(self._source).__name__),
            "joint_names": list(d.joint_names),
            "channels": [f"robot_state/{c}" for c in d.channels],
            "state_fields": list(d.state_fields),
            "state_hz": self._state_hz,
            "destination": str(self._dest.resolve()),
            "format": "NT episode v0.0.3 (the only format this library writes)",
            "cameras": [
                {"id": c.id, "width": c.width, "height": c.height, "fps": c.fps}
                for c in self._cameras
            ],
            "camera_stub_reason": self._camera_stub_reason if not self._cameras else None,
            "task": self._task,
            "provenance": {"author": self._author, "license": self._license, "verified": False},
            "target": self._target,
        }

    def preflight(self) -> dict:
        """Alias for :meth:`describe`. The library is descriptive; this returns
        the contract a frontend may print and (optionally) refuse on. It performs
        a writability probe of the destination and reports it as a field — it does
        NOT exit or raise on a non-writable dest; that refusal is the frontend's
        call."""
        report = self.describe()
        report["destination_writable"] = self._probe_writable()
        return report

    def _probe_writable(self) -> bool:
        try:
            self._dest.mkdir(parents=True, exist_ok=True)
            probe = self._dest / ".newt_write_probe"
            probe.write_text("ok")
            probe.unlink()
            return True
        except OSError:
            return False

    # --- observers: one read, two destinations ------------------------------

    def attach_observer(self, observer) -> None:
        """Add a second destination for everything the capture loops read.

        The episode file and a live view are two sinks on one source, and this is
        the split that makes them one: the loops read the hardware once, hand each
        read to the writer when an episode is in flight, and hand the same read to
        every observer always. Recording therefore cannot interrupt or degrade a
        view — starting an episode adds a writer, it does not add a read.

        An observer is offered ``on_state(channels, ts_ns)`` and
        ``on_frames(frames, ts_ns)``; both are optional, and both receive exactly
        what the source returned, including the ``None`` entries that mark a
        dropped read. An observer that raises is dropped from the session and its
        failure is reported through ``observer_failures()`` — it never takes the
        capture loop, and therefore an episode, down with it.

        Attaching one starts the capture loops for the rest of the session, so a
        view is live before the first episode and stays live between episodes.
        """
        if self._closed:
            raise RuntimeError("Session is closed; construct a new Session to record again.")
        with self._lock:
            self._observers.append(observer)
        self._streaming = True
        self._start_loops()

    @property
    def camera_failure(self) -> tuple[str, Exception] | None:
        """(cause, exception) if the camera bridge stopped answering, else None.

        The twin of ``observer_failures()`` for the first sink rather than the
        second. ``end_episode(keep=True)`` already refuses on this, loudly; this
        property is for a frontend that wants to say so *before* somebody spends
        another three minutes recording into a session whose cameras died. It
        reports only what the capture loop actually observed — there is no probe
        here, and a camera nobody has read from yet is neither healthy nor broken.
        """
        return self._camera_failure

    def observer_failures(self) -> list[tuple[str, Exception]]:
        """(observer name, exception) for every observer dropped mid-session.

        Read by a frontend so a view that died says so out loud instead of just
        going still. Never empty-by-design: an empty list means every attached
        observer is still receiving reads.
        """
        with self._lock:
            return list(self._observer_failures)

    def _notify(self, method: str, *args) -> None:
        """Offer one read to every observer, outside the writer lock.

        Outside the lock on purpose: an observer is a second sink, and a slow one
        must not hold up the read that the episode file depends on.
        """
        for observer in list(self._observers):
            hook = getattr(observer, method, None)
            if hook is None:
                continue
            try:
                hook(*args)
            except Exception as exc:  # noqa: BLE001 — an observer is the second
                # sink, never the first. Whatever it does wrong, the episode has
                # to keep being written; dropping the observer and recording why
                # is how that failure stays visible instead of becoming silence.
                with self._lock:
                    if observer in self._observers:
                        self._observers.remove(observer)
                    self._observer_failures.append((type(observer).__name__, exc))

    def _start_loops(self) -> None:
        """Start whichever capture threads are not already running. Idempotent —
        called by both ``start_episode`` and ``attach_observer``, and either may
        come first."""
        self._stop_loop.clear()
        if not self._state_pushed and (
            self._loop_thread is None or not self._loop_thread.is_alive()
        ):
            self._loop_thread = threading.Thread(
                target=self._capture_loop, name="newt-record-capture", daemon=True
            )
            self._loop_thread.start()
        if (
            self._cameras
            and self._reads_frames
            and (self._camera_thread is None or not self._camera_thread.is_alive())
        ):
            # Its own thread on purpose: a camera read blocks for as long as the
            # hardware takes, and the state loop's rhythm is not this card's to
            # move. Started only when the source declared cameras, so the
            # state-only path runs exactly the threads it ran before — and never
            # for a caller-owned bridge, which pushes through feed_frame itself.
            self._camera_thread = threading.Thread(
                target=self._camera_loop, name="newt-record-cameras", daemon=True
            )
            self._camera_thread.start()

    # --- episode lifecycle --------------------------------------------------

    def start_episode(
        self,
        *,
        task: str | None = None,
        dest: str | Path | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        """Open a fresh episode and start the capture loop. Returns the episode id.

        The loop polls the source at the state rate on a background thread, writing
        every read into the in-flight episode and counting every dropped channel.

        **What a take may change, and what it may not.** The three keyword
        arguments override, for this episode only, the three things that describe
        what a take *is about* and *where it lands*: its language task, its output
        directory, its tags. Everything else — the source, the cameras it opened,
        the descriptor, the state rate — is the session's contract with the
        hardware and is deliberately not overridable here. Changing those would
        mean re-opening hardware mid-session, which drops the reads a live view is
        drawing from; changing these three does not touch the rig at all.

        Passing nothing keeps the session's own values, so every existing caller
        behaves exactly as before. An empty ``dest`` directory is created on first
        use by the writer, which is why a page can name a dataset that does not
        exist yet without a separate create step.
        """
        if self._closed:
            raise RuntimeError("Session is closed; construct a new Session to record again.")
        if self._writer is not None:
            raise RuntimeError("An episode is already recording; end it before starting another.")

        from newt.recording._writer import DEFAULT_TAGS, EpisodeWriter

        episode_dest = Path(dest) if dest is not None else self._dest
        self._writer = EpisodeWriter(
            dest_root=episode_dest,
            task_name=task if task is not None else self._task,
            state_frequency=self._state_hz,
            cameras=self._cameras,
            descriptor=self.descriptor,
            camera_stub_reason=self._camera_stub_reason,
            author=self._author,
            license=self._license,
            tags=tuple(tags) if tags is not None else DEFAULT_TAGS,
        )
        self._camera_failure = None
        self._start_loops()
        self._notify("on_episode", "started", self._writer.episode_id)
        return self._writer.episode_id

    def retag_episode(self, tags: list[str] | tuple[str, ...]) -> None:
        """Replace the in-flight episode's tags before it commits.

        A person knows what a take was after watching it, not before starting it,
        so the tags a take ends with are the ones worth writing. They go into
        ``episode.json`` with the rest of the episode config — one file, one truth
        — rather than into a sidecar that a later reader has to know to look for.

        Refuses when nothing is recording: a tag with no episode to belong to would
        otherwise be accepted and silently dropped.
        """
        with self._lock:
            writer = self._writer
            if writer is None:
                raise RuntimeError(
                    "No episode is recording, so there is nothing to tag. Tags are "
                    "written into the in-flight episode's own file; call this "
                    "between start_episode() and end_episode()."
                )
            writer.tags = tuple(tags)

    def _capture_loop(self) -> None:
        """Poll the source at the state rate and write each read. The only place
        the state capture rhythm lives, and cameras do not enter it: a source that
        declared cameras is read by ``_camera_loop`` on its own thread, and a
        frontend that owns its own bridge pushes through ``feed_frame``. Either
        way the state rate is the state rate."""
        while not self._stop_loop.is_set():
            ts_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
            channels = self._source.read_state()
            shown: dict[str, list[float]] = {}
            with self._lock:
                if self._writer is None and not self._streaming:
                    return
                for key, state in channels.items():
                    if state is None:
                        if self._writer is not None:
                            self._writer.note_dropped_state()
                        continue
                    if self._writer is not None:
                        self._writer.write_state(key, state, ts_ns)
                    shown[key] = list(state.positions)
                if shown:
                    self._last_positions = shown
            # The second sink, outside the lock and after the first: the episode
            # file has already taken this read, so nothing an observer does can
            # change or delay what was written.
            self._notify("on_state", channels, ts_ns)
            time.sleep(self._period_s)

    def _camera_loop(self) -> None:
        """Pull one frame per declared camera from the source's bridge and write it.

        Runs only when the source declared cameras. The library still opens no
        camera — it asks the source for the next frame exactly as the state loop
        asks it for the next joint reading. A camera that returns nothing this read
        is a counted drop; nothing is duplicated, interpolated, or zero-filled to
        keep a count even.

        A read that raises is the camera having stopped answering mid-episode: the
        loop stops and records the failure, and ``end_episode(keep=True)`` refuses
        rather than committing an episode whose video quietly ends early.
        """
        period = 1.0 / max(1, max(int(getattr(c, "fps", 0) or 0) for c in self._cameras))
        cam_ids = [c.id for c in self._cameras]
        next_at = time.monotonic()
        while not self._stop_loop.is_set():
            try:
                frames = self._source.read_frames()
            except Exception as exc:  # noqa: BLE001 — this is a background thread;
                # an uncaught exception here would kill the thread silently (Python
                # doesn't propagate thread exceptions to the caller) and frames would
                # just stop arriving with no signal why. Recording the cause and
                # stopping cleanly is how the camera's failure becomes visible at all.
                self._fail_cameras("stopped_answering", exc)
                return
            # Timestamped after the read returns: that is when the frame arrived,
            # and a blocking read would otherwise stamp it early by its own duration.
            ts_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
            try:
                with self._lock:
                    if self._writer is None and not self._streaming:
                        return
                    for cam_id in cam_ids:
                        frame = frames.get(cam_id)
                        if frame is None:
                            if self._writer is not None:
                                self._writer.note_dropped_frame(cam_id)
                            continue
                        if self._writer is not None:
                            self._writer.write_frame(cam_id, frame, ts_ns)
            except Exception as exc:  # noqa: BLE001 — same background-thread reason
                # as the read above: an encoder write failure must be recorded and
                # stop the loop, not die silently on a thread nothing is watching.
                self._fail_cameras("encoder_refused", exc)
                return
            # The second sink, outside the lock and after the first — see the note
            # on the same call in _capture_loop.
            self._notify("on_frames", frames, ts_ns)
            next_at += period
            time.sleep(max(0.0, next_at - time.monotonic()))

    def _fail_cameras(self, cause: str, exc: Exception) -> None:
        """Record why the camera bridge stopped and stop capturing. The episode is
        refused at ``end_episode``, not here — this runs on the camera thread, and
        a thread that tears down the in-flight episode underneath the state loop is
        a race, not a refusal."""
        self._camera_failure = (cause, exc)
        self._stop_loop.set()

    def feed_state(self, channels: dict, ts_ns: int | None = None) -> None:
        """Push one tick's state into the in-flight episode. The state twin of
        ``feed_frame``, and the same reason: the loop that already holds the
        hardware is the one that reads it.

        ``channels`` is what ``read_state()`` returns — one entry per declared
        channel, ``None`` where that channel produced nothing this tick. A
        ``None`` is a counted drop, exactly as it is on the polled path: nothing
        is repeated, interpolated or zero-filled to keep the count even.

        ``ts_ns`` is the tick's own timestamp, passed in rather than taken here,
        because the honest answer to *when was this state true* is when the tick
        that produced it ran — not when it reached the writer.

        Only a Session constructed with ``state_pushed=True`` accepts this. On a
        polled Session the capture thread is already reading the source, and a
        second writer on a second clock would interleave two rhythms into one
        episode with nothing in the file saying so.
        """
        if not self._state_pushed:
            raise RuntimeError(
                "feed_state() on a Session that polls its source itself.\n"
                "Yours: this Session was constructed without state_pushed=True, so its "
                "capture thread is already reading the source at the state rate; these "
                "pushed frames would be a second clock writing into the same episode.\n"
                "Fix: construct the Session with state_pushed=True when your own loop "
                "owns the tick, or stop calling feed_state and let the Session poll."
            )
        ts = ts_ns if ts_ns is not None else time.clock_gettime_ns(time.CLOCK_REALTIME)
        with self._lock:
            shown: dict[str, list[float]] = {}
            for key, state in channels.items():
                if state is None:
                    if self._writer is not None:
                        self._writer.note_dropped_state()
                    continue
                if self._writer is not None:
                    self._writer.write_state(key, state, ts)
                shown[key] = list(state.positions)
            if shown:
                self._last_positions = shown
        # The second sink, on the pushed path too: the tick that drove the rig and
        # wrote the frame is the one a view draws, which is what keeps the picture
        # and the file the same event rather than two readings of it.
        self._notify("on_state", channels, ts)

    def feed_frame(self, cam_id: str, frame, ts_ns: int | None = None) -> None:
        """Push one camera frame into the in-flight episode. A frontend that owns
        a camera bridge calls this; the library never opens a camera itself (no
        hardware IO baked into the loop). No-op when no episode is recording."""
        ts = ts_ns if ts_ns is not None else time.clock_gettime_ns(time.CLOCK_REALTIME)
        with self._lock:
            if self._writer is not None:
                self._writer.write_frame(cam_id, frame, ts)
        self._notify("on_frames", {cam_id: frame}, ts)

    def _join_loops(self) -> None:
        """Wait for both capture threads to leave the writer alone. Called before
        the writer is finalized or abandoned — a frame written after the encoder's
        pipe is closed is a broken pipe, not a frame."""
        for thread in (self._loop_thread, self._camera_thread):
            if thread is not None:
                thread.join(timeout=5.0)
        self._camera_thread = None

    def end_episode(self, keep: bool) -> Path | None:
        """Stop the capture loop and either keep or discard the episode.

        ``keep=True`` commits atomically (episode.json last, temp dir renamed into
        place) and returns the final directory. ``keep=False`` removes the temp dir
        whole, leaving no directory, and returns ``None``. Either way the in-flight
        episode is finished — start a new one to record again.
        """
        if self._writer is None:
            raise RuntimeError("No episode is recording; call start_episode() first.")

        if not self._streaming:
            self._stop_loop.set()
            self._join_loops()
        # While a view is attached the loops keep running across the episode
        # boundary — the preview does not blink because a take ended. Clearing the
        # writer under the lock is the whole barrier that needs: both loops take
        # the same lock before they touch it, so no read can land in a writer that
        # is already being finalized.

        with self._lock:
            writer = self._writer
            self._writer = None
            # Remember the just-finished episode's counts so a frontend can report
            # them after the in-flight writer is cleared (status() goes to zero once
            # nothing is recording).
            self._last_state_count = writer.state_count
            self._last_dropped_state = writer.dropped_state
            self._last_frame_counts = writer.frame_counts
            self._last_dropped_frames = writer.dropped_frames

        self._notify("on_episode", "stopped", writer.episode_id)

        if not keep:
            writer.abandon()
            return None

        if self._camera_failure is not None:
            # The video ends before the episode does. Committing it would hand
            # someone a file whose joints run three minutes and whose camera stops
            # at forty seconds, with nothing in the episode saying so.
            cause, exc = self._camera_failure
            self._camera_failure = None
            writer.abandon()
            raise CameraCaptureFailed(
                cause, camera_failure_message(cause, f"{type(exc).__name__}: {exc}")
            )

        duration_s = writer.state_count * self._period_s
        path = writer.keep(duration_s)
        self._sink.deliver(path)
        self._kept += 1
        return path

    # --- status + teardown --------------------------------------------------

    def status(self) -> SessionStatus:
        """A snapshot a frontend renders: recording flag, current episode id, the
        running state/dropped counts, kept-toward-target, and the last positions
        read (for the live readout). Pure data, no side effects."""
        with self._lock:
            w = self._writer
            return SessionStatus(
                recording=w is not None,
                episode_id=w.episode_id if w is not None else None,
                state_count=w.state_count if w is not None else 0,
                dropped_state=w.dropped_state if w is not None else 0,
                kept=self._kept,
                target=self._target,
                last_positions=self._last_positions,
                closed=self._closed,
                frame_counts=w.frame_counts if w is not None else {},
                dropped_frames=w.dropped_frames if w is not None else {},
            )

    def dropped_report(self) -> str | None:
        """A human line summarizing dropped reads for the in-flight episode, or
        None when nothing is recording or nothing dropped. A frontend prints it;
        the report itself is computed here so every frontend says the same thing.

        Camera drops are reported per camera and never rolled into the state
        total: a rig that lost a tenth of one camera's frames and none of its
        joints has a specific problem, and one blended percentage hides it."""
        with self._lock:
            w = self._writer
            if w is None:
                return None
            parts: list[str] = []
            total = w.state_count + w.dropped_state
            if w.dropped_state and total:
                pct = 100.0 * w.dropped_state / total
                parts.append(f"{w.dropped_state}/{total} state reads dropped ({pct:.1f}%)")
            dropped_frames = w.dropped_frames
            frame_counts = w.frame_counts
            for cam_id, dropped in sorted(dropped_frames.items()):
                reads = frame_counts.get(cam_id, 0) + dropped
                pct = 100.0 * dropped / reads if reads else 0.0
                parts.append(f"camera {cam_id}: {dropped}/{reads} frames dropped ({pct:.1f}%)")
            if not parts:
                return None
            return f"dropped-frame report — episode {w.episode_id}: " + "; ".join(parts) + "."

    def kill(self) -> None:
        """Emergency teardown: abandon any in-flight episode (no partial dir),
        torque off via the source, and close. The behavior behind a frontend's
        kill key — exit code is the frontend's call (130 by convention)."""
        if self._writer is not None:
            self._stop_loop.set()
            self._join_loops()
            with self._lock:
                writer = self._writer
                self._writer = None
            writer.abandon()
        self.close()

    def close(self) -> None:
        """Stop any loop, torque off via the source's ``disable_all`` (if present),
        and release the source. Idempotent and safe from a kill path. Does NOT
        commit an in-flight episode — call end_episode(keep=True) first to keep it,
        or it is discarded as a partial."""
        if self._closed:
            return
        self._streaming = False
        self._stop_loop.set()
        if self._writer is not None:
            with self._lock:
                writer = self._writer
                self._writer = None
            writer.abandon()
        disable_all = getattr(self._source, "disable_all", None)
        if callable(disable_all):
            try:
                disable_all()
            except Exception:  # noqa: BLE001, S110 — close() is "safe from a kill
                # path" per the docstring: torque-off must be attempted regardless of
                # what disable_all() does, and a failure here must not skip the
                # close() release below — a rig left both energized AND unreleased
                # is worse than one that raised quietly on the way down.
                pass
        close = getattr(self._source, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001, S110 — same kill-path guarantee: this
                # is the last release step, and close() must always mark itself
                # closed (_closed = True below) even if the source's own close()
                # misbehaves.
                pass
        self._closed = True
