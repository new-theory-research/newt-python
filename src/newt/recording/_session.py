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

Generalized from rebot-bench's ``record_session.py``: the rhythm, the dropped-
frame report, the kill-leaves-no-dir guarantee, and the simulate-via-fake-source
pattern are all preserved — lifted out of the keyboard script into the library so
every frontend gets them for free.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from newt.recording._seam import RecordingSource, StateDescriptor
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
        output_dir: "str | Path",
        *,
        cameras: "list | None" = None,
        state_hz: int = DEFAULT_STATE_HZ,
        author: str | None = None,
        license: str | None = None,
        camera_stub_reason: str | None = None,
        target: int | None = None,
        sink: Sink | None = None,
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

        # Camera specs (CameraSpec instances) are accepted but state-only capture
        # is the default; the writer imports lazily, so we hold raw specs here.
        self._cameras = list(cameras) if cameras else []

        # Provenance — declared, never verified (local-first capture).
        from newt.recording._writer import DEFAULT_AUTHOR, DEFAULT_LICENSE

        self._author = author or DEFAULT_AUTHOR
        self._license = license or DEFAULT_LICENSE

        self._writer = None  # set during an episode
        self._loop_thread: threading.Thread | None = None
        self._stop_loop = threading.Event()
        self._lock = threading.Lock()
        self._kept = 0
        self._last_positions: dict[str, list[float]] | None = None
        self._last_state_count = 0
        self._last_dropped_state = 0
        self._closed = False

    @property
    def last_episode_counts(self) -> tuple[int, int]:
        """(state_count, dropped_state) of the most recently ended episode. A
        frontend reports these after end_episode(), since the live status() drops
        to zero once nothing is recording."""
        return self._last_state_count, self._last_dropped_state

    # --- the descriptive preflight (reverse-contract courtesy) --------------

    @property
    def descriptor(self) -> StateDescriptor:
        return self._source.descriptor

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

    # --- episode lifecycle --------------------------------------------------

    def start_episode(self) -> str:
        """Open a fresh episode and start the capture loop. Returns the episode id.

        The loop polls the source at the state rate on a background thread, writing
        every read into the in-flight episode and counting every dropped channel.
        """
        if self._closed:
            raise RuntimeError("Session is closed; construct a new Session to record again.")
        if self._writer is not None:
            raise RuntimeError("An episode is already recording; end it before starting another.")

        from newt.recording._writer import EpisodeWriter

        self._writer = EpisodeWriter(
            dest_root=self._dest,
            task_name=self._task,
            state_frequency=self._state_hz,
            cameras=self._cameras,
            descriptor=self.descriptor,
            camera_stub_reason=self._camera_stub_reason,
            author=self._author,
            license=self._license,
        )
        self._stop_loop.clear()
        self._loop_thread = threading.Thread(
            target=self._capture_loop, name="newt-record-capture", daemon=True
        )
        self._loop_thread.start()
        return self._writer.episode_id

    def _capture_loop(self) -> None:
        """Poll the source at the state rate and write each read. The only place
        the capture rhythm lives. Cameras: opened captures (if any) are read by
        the frontend's camera bridge via ``feed_frame`` — state capture is here,
        camera frame pulls are pushed in to keep hardware-IO out of the library's
        loop thread when no cameras are wired (the common case)."""
        while not self._stop_loop.is_set():
            ts_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
            channels = self._source.read_state()
            shown: dict[str, list[float]] = {}
            with self._lock:
                if self._writer is None:
                    return
                for key, state in channels.items():
                    if state is None:
                        self._writer.note_dropped_state()
                        continue
                    self._writer.write_state(key, state, ts_ns)
                    shown[key] = list(state.positions)
                if shown:
                    self._last_positions = shown
            time.sleep(self._period_s)

    def feed_frame(self, cam_id: str, frame, ts_ns: int | None = None) -> None:
        """Push one camera frame into the in-flight episode. A frontend that owns
        a camera bridge calls this; the library never opens a camera itself (no
        hardware IO baked into the loop). No-op when no episode is recording."""
        with self._lock:
            if self._writer is None:
                return
            self._writer.write_frame(
                cam_id, frame, ts_ns if ts_ns is not None else time.clock_gettime_ns(time.CLOCK_REALTIME)
            )

    def end_episode(self, keep: bool) -> "Path | None":
        """Stop the capture loop and either keep or discard the episode.

        ``keep=True`` commits atomically (episode.json last, temp dir renamed into
        place) and returns the final directory. ``keep=False`` removes the temp dir
        whole, leaving no directory, and returns ``None``. Either way the in-flight
        episode is finished — start a new one to record again.
        """
        if self._writer is None:
            raise RuntimeError("No episode is recording; call start_episode() first.")

        self._stop_loop.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)

        with self._lock:
            writer = self._writer
            self._writer = None
            # Remember the just-finished episode's counts so a frontend can report
            # them after the in-flight writer is cleared (status() goes to zero once
            # nothing is recording).
            self._last_state_count = writer.state_count
            self._last_dropped_state = writer.dropped_state

        if not keep:
            writer.abandon()
            return None

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
            )

    def dropped_report(self) -> "str | None":
        """A human line summarizing dropped reads for the in-flight episode, or
        None when nothing is recording or nothing dropped. A frontend prints it;
        the report itself is computed here so every frontend says the same thing."""
        with self._lock:
            w = self._writer
            if w is None:
                return None
            total = w.state_count + w.dropped_state
            if w.dropped_state == 0 or total == 0:
                return None
            pct = 100.0 * w.dropped_state / total
            return (
                f"dropped-frame report — episode {w.episode_id}: "
                f"{w.dropped_state}/{total} reads dropped ({pct:.1f}%)."
            )

    def kill(self) -> None:
        """Emergency teardown: abandon any in-flight episode (no partial dir),
        torque off via the source, and close. The behavior behind a frontend's
        kill key — exit code is the frontend's call (130 by convention)."""
        if self._writer is not None:
            self._stop_loop.set()
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5.0)
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
            except Exception:
                pass
        close = getattr(self._source, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        self._closed = True
