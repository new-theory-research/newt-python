"""Writes one NT v0.0.3 episode and nothing else — the single-format law.

Ported from rebot-bench ``episode_writer.py``. There is exactly one output
layout: the NT episode directory with ``data.mcap``, optional
``cameras/<id>/color.mp4``, and an ``episode.json`` written last via ``rename(2)``.
No LeRobot path, no format options, no plugins. Converters (Molmo / LeRobot-bound)
are downstream tools that read NT episodes; they do not live here.

Atomicity is the spine. The whole episode is built inside a temp directory; on
keep, ``episode.json`` is written last and the temp dir is renamed into place. On
discard or a mid-episode kill, the temp dir is removed whole, so a partial episode
never appears under the destination.

The MCAP container is the single source of truth for timing. Camera frames get a
synchronized marker message per frame on ``camera/<id>/color``; the writer enforces
the frame-count invariant (one MCAP marker per encoded video frame) before commit.

``mcap`` and ``numpy`` ship with the ``recording`` extra and are imported lazily
through the lantern guard — this module is never reached by a bare ``import newt``.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

from newt.recording._lantern import require

# Provenance declared but not verifiable in local-first capture (no account, no
# network in the capture path). Present in every episode.json so downstream
# tooling sees the claim and its unverified status.
DEFAULT_AUTHOR = "newt-recording/alpha"
DEFAULT_LICENSE = "UNVERIFIED-ALPHA"

# The one and only format this writer emits.
FORMAT_VERSION = "0.0.3"

# What ``episode_config.tags`` says when the caller states nothing. It marks the
# format generation, not the take — a caller that has something to say about this
# particular episode passes ``tags=`` and replaces it.
DEFAULT_TAGS = ("alpha",)

# Color video encode profile, spec v0.0.3 "Camera data".
_FFMPEG_COLOR_ARGS = [
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",
    "-bf", "0",            # no B-frames
    "-g", "30",            # 1-second keyframe interval at 30fps
    "-x264-params", "keyint=30:min-keyint=30",
]


@dataclass
class CameraSpec:
    """A configured RGB camera. Color-only is first-class in v0.0.3; depth slots
    in later by adding depth_intrinsics + a depth path without reworking this
    writer."""

    id: str
    width: int
    height: int
    fps: int = 30


@dataclass
class EpisodeWriter:
    """Accumulates one episode in a temp dir, commits atomically on keep."""

    dest_root: Path
    task_name: str
    state_frequency: int
    cameras: list[CameraSpec]
    descriptor: object  # recording._seam.StateDescriptor; drives channels + arms
    camera_stub_reason: str | None = None  # set LOUDLY when cameras are unavailable
    author: str = DEFAULT_AUTHOR
    license: str = DEFAULT_LICENSE
    # What this take was, in the caller's words. Written through to
    # ``episode_config.tags`` verbatim — this writer defines no vocabulary and
    # rejects no string, because a taxonomy is a product decision and the file
    # format is not the place to hold one.
    tags: tuple[str, ...] = DEFAULT_TAGS

    episode_id: str = field(init=False)
    _tmp: Path = field(init=False)
    _start_ns: int = field(init=False)
    _state_ns: list[int] = field(init=False, default_factory=list)
    _frame_writers: dict[str, subprocess.Popen] = field(init=False, default_factory=dict)
    _frame_counts: dict[str, int] = field(init=False, default_factory=dict)
    _state_count: int = field(init=False, default=0)
    _dropped_state: int = field(init=False, default=0)
    _dropped_frames: dict[str, int] = field(init=False, default_factory=dict)
    _camera_marker_ns: dict[str, list[int]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        # Lazy + guarded: mcap and numpy are recording-extra deps. Reaching this
        # constructor without the extra lights the lantern naming newt[recording].
        mcap_writer = require("mcap.writer", "mcap")
        self._np = require("numpy", "numpy")
        from newt.recording._proto import file_descriptor_set

        self.episode_id = secrets.token_hex(4)  # 8 hex chars
        self.dest_root.mkdir(parents=True, exist_ok=True)
        self._tmp = Path(
            tempfile.mkdtemp(prefix=f".episode_{self.episode_id}_", dir=self.dest_root)
        )
        self._start_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)

        # Not a `with`: the handle outlives this constructor, closed explicitly in
        # _finalize_streams (commit) or abandon (discard) — whichever the episode hits.
        self._mcap_file = open(self._tmp / "data.mcap", "wb")  # noqa: SIM115
        self._mcap = mcap_writer.Writer(self._mcap_file)
        self._mcap.start()
        self._schema_id = self._mcap.register_schema(
            name="robot.RobotState",
            encoding="protobuf",
            data=file_descriptor_set(),
        )
        # One MCAP channel per descriptor channel suffix. A single kinesthetic arm
        # registers exactly `robot_state/<id>`; a leader/follower pair registers
        # `robot_state/<arm>/leader` and `.../follower` per v0.0.3.
        self._state_channels: dict[str, int] = {}
        for suffix in self.descriptor.channels:
            self._state_channels[suffix] = self._mcap.register_channel(
                topic=f"robot_state/{suffix}",
                message_encoding="protobuf",
                schema_id=self._schema_id,
            )
        self._channel_seq: dict[str, int] = {s: 0 for s in self.descriptor.channels}

        # Per-camera color channel + a raw-frame ffmpeg pipe. Empty schema: the
        # marker payload is empty; the message exists only to timestamp a frame.
        self._cam_channel: dict[str, int] = {}
        for cam in self.cameras:
            self._camera_marker_ns[cam.id] = []
            self._frame_counts[cam.id] = 0
            empty_schema = self._mcap.register_schema(
                name="camera.FrameMarker", encoding="", data=b""
            )
            self._cam_channel[cam.id] = self._mcap.register_channel(
                topic=f"camera/{cam.id}/color",
                message_encoding="",
                schema_id=empty_schema,
            )
            cam_dir = self._tmp / "cameras" / cam.id
            cam_dir.mkdir(parents=True, exist_ok=True)
            self._frame_writers[cam.id] = self._spawn_encoder(cam, cam_dir / "color.mp4")

    def _spawn_encoder(self, cam: CameraSpec, out_path: Path) -> subprocess.Popen:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{cam.width}x{cam.height}", "-r", str(cam.fps),
            "-i", "-",
            *_FFMPEG_COLOR_ARGS,
            str(out_path),
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # --- per-tick capture ---------------------------------------------------

    def write_state(self, channel_key: str, state, ts_ns: int) -> None:
        """Append one robot-state message on ``robot_state/<channel_key>``.
        ``state`` is a recording._seam.JointState. sequence is per-channel;
        state_count counts total messages across channels."""
        from newt.recording._proto import robot_state_class

        msg = robot_state_class()(
            joint_positions=state.positions,
            joint_velocities=state.velocities,
            joint_efforts=state.efforts,
            rotor_temperatures=state.rotor_temperatures,
            driver_temperatures=state.driver_temperatures,
        )
        self._mcap.add_message(
            channel_id=self._state_channels[channel_key],
            log_time=ts_ns,
            publish_time=ts_ns,
            sequence=self._channel_seq[channel_key],
            data=msg.SerializeToString(),
        )
        self._channel_seq[channel_key] += 1
        self._state_ns.append(ts_ns)
        self._state_count += 1

    def note_dropped_state(self) -> None:
        """A read returned None for a channel. Count it; never swallow it. The
        per-episode report surfaces the total."""
        self._dropped_state += 1

    def note_dropped_frame(self, cam_id: str) -> None:
        """A camera read returned no frame for ``cam_id``. Count it; never swallow
        it, and never write a substitute frame to keep the count even — a gap in
        the video is the truth, and the markers say where it is."""
        self._dropped_frames[cam_id] = self._dropped_frames.get(cam_id, 0) + 1

    def write_frame(self, cam_id: str, frame, ts_ns: int) -> None:
        """Append one color frame: pipe pixels to ffmpeg and drop a synchronized
        marker on the MCAP timeline. sequence == frame index."""
        idx = self._frame_counts[cam_id]
        self._frame_writers[cam_id].stdin.write(
            self._np.ascontiguousarray(frame).tobytes()
        )
        self._mcap.add_message(
            channel_id=self._cam_channel[cam_id],
            log_time=ts_ns,
            publish_time=ts_ns,
            sequence=idx,
            data=b"",
        )
        self._camera_marker_ns[cam_id].append(ts_ns)
        self._frame_counts[cam_id] = idx + 1

    @property
    def dropped_state(self) -> int:
        return self._dropped_state

    @property
    def dropped_frames(self) -> dict[str, int]:
        """Dropped camera reads this episode, per camera id. Cameras that dropped
        nothing are absent rather than zero — an empty dict is a clean episode."""
        return dict(self._dropped_frames)

    @property
    def frame_counts(self) -> dict[str, int]:
        """Frames written this episode, per camera id — the number the frame-count
        invariant will check against the encoded file at commit."""
        return dict(self._frame_counts)

    @property
    def state_count(self) -> int:
        return self._state_count

    # --- commit / abandon ---------------------------------------------------

    def _finalize_streams(self) -> dict[str, int]:
        """Close ffmpeg pipes; return the encoded frame count per camera read
        back from the file via ffprobe — the count the validator will check."""
        for proc in self._frame_writers.values():
            proc.stdin.close()
            proc.wait()
        self._mcap.finish()
        self._mcap_file.close()

        encoded: dict[str, int] = {}
        for cam in self.cameras:
            path = self._tmp / "cameras" / cam.id / "color.mp4"
            encoded[cam.id] = _ffprobe_frame_count(path)
        return encoded

    def keep(self, duration_s: float) -> Path:
        """Commit the episode. Enforce the frame-count invariant, write
        episode.json last via rename(2), then rename the temp dir into place.
        Returns the final episode directory."""
        encoded = self._finalize_streams()

        for cam in self.cameras:
            markers = len(self._camera_marker_ns[cam.id])
            if markers != encoded[cam.id]:
                self.abandon()
                raise RuntimeError(
                    f"frame-count invariant violated for camera '{cam.id}': "
                    f"{markers} MCAP markers != {encoded[cam.id]} encoded frames. "
                    "Episode discarded; nothing written."
                )

        sdk_version, sdk_version_source = _sdk_version_info()
        sdk_commit, sdk_commit_dirty, sdk_commit_source = _git_info()

        episode = {
            "episode_config": {
                "task_name": self.task_name,
                "tags": list(self.tags),
                "duration": round(duration_s, 3),
            },
            "robot_config": {
                "arms": [dict(a) for a in self.descriptor.arms],
                "state_frequency": self.state_frequency,
                # control_frequency omitted: kinesthetic capture has no control loop.
            },
            "camera_config": self._camera_config(),
            "provenance": {
                "author": self.author,
                "license": self.license,
                "verified": False,
            },
            # Writer identity: what software wrote this file, separate from
            # `provenance` (who authored the episode). Each value is either
            # genuinely determined (source says how) or None with an honest
            # "undetermined: <reason>" source — never a plausible-looking guess.
            "writer": {
                "sdk_version": sdk_version,
                "sdk_version_source": sdk_version_source,
                "sdk_commit": sdk_commit,
                "sdk_commit_dirty": sdk_commit_dirty,
                "sdk_commit_source": sdk_commit_source,
            },
            "recording_started_at": _rfc3339_ns(self._start_ns),
            "format_version": FORMAT_VERSION,
        }
        if self.camera_stub_reason:
            episode["camera_config"]["stub_reason"] = self.camera_stub_reason

        self._atomic_write_json(self._tmp / "episode.json", episode)

        final = self.dest_root / f"episode_{self.episode_id}"
        os.rename(self._tmp, final)
        return final

    def abandon(self) -> None:
        """Discard the episode: tear down encoders and remove the temp dir whole.
        No partial directory is ever left under the destination."""
        for proc in self._frame_writers.values():
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 — abandon() must tear down every
                # encoder regardless of how any one of them misbehaves; a stuck or
                # already-dead ffmpeg process that won't close/wait cleanly still
                # gets force-killed rather than leaving abandon() hung or partial.
                proc.kill()
        try:
            self._mcap.finish()
            self._mcap_file.close()
        except Exception:  # noqa: BLE001, S110 — same discard guarantee: the
            # overriding goal per the docstring is "no partial directory is ever
            # left," so an mcap finish/close failure must not skip the rmtree below.
            pass
        shutil.rmtree(self._tmp, ignore_errors=True)

    # --- helpers ------------------------------------------------------------

    def _camera_config(self) -> dict:
        if not self.cameras:
            return {"cameras": []}
        first = self.cameras[0]
        return {
            "width": first.width,
            "height": first.height,
            "fps": first.fps,
            "cameras": [
                {
                    "id": cam.id,
                    "frame": "world",
                    # No color_intrinsics: uncalibrated cameras declare none rather
                    # than inventing them. color-only (no depth_intrinsics) per v0.0.3.
                }
                for cam in self.cameras
            ],
        }

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)


def _sdk_version_info() -> tuple[str | None, str]:
    """This install's ``newt`` version and how it was determined. Prefers
    ``importlib.metadata`` (works for any install method that registers package
    metadata — pip, uv, git-ref, wheel); falls back to the hardcoded
    ``newt.__version__`` constant only when metadata can't be read (editable
    installs, some dev layouts), and records that fallback as a fallback rather
    than presenting it as equally reliable — ``__version__`` is known to drift
    from the installed version (Rule 10: state the mechanism, not just the
    value)."""
    try:
        return _pkg_version("newt"), "importlib.metadata"
    except PackageNotFoundError:
        from newt import __version__

        return __version__, "fallback: newt.__version__ (importlib.metadata unavailable; may be stale)"


def _git_info() -> tuple[str | None, bool | None, str]:
    """Git commit + dirty flag for the running newt-python checkout, sourced by
    invoking git against this file's own directory. commit/dirty are ``None``
    with an honest reason in the returned source string when git metadata isn't
    determinable — no git binary, a non-git checkout (wheel/sdist install), or a
    git invocation that errors are all real possible outcomes, not bugs to
    paper over."""
    repo_dir = Path(__file__).resolve().parent
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, None, f"undetermined: git rev-parse failed to run ({exc.__class__.__name__})"
    if commit_result.returncode != 0:
        return None, None, "undetermined: not a git checkout"
    commit = commit_result.stdout.strip()

    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return commit, None, (
            f"git rev-parse HEAD (dirty flag undetermined: git status failed to run, "
            f"{exc.__class__.__name__})"
        )
    if status_result.returncode != 0:
        return commit, None, "git rev-parse HEAD (dirty flag undetermined: git status exited non-zero)"

    dirty = bool(status_result.stdout.strip())
    return commit, dirty, "git rev-parse HEAD + git status --porcelain"


def _ffprobe_frame_count(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    text = out.stdout.strip()
    return int(text) if text.isdigit() else 0


def _rfc3339_ns(ns: int) -> str:
    """Nanosecond-precision RFC 3339 UTC timestamp, matching the spec sample."""
    secs, rem_ns = divmod(ns, 1_000_000_000)
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(secs))
    return f"{base}.{rem_ns:09d}+00:00"
