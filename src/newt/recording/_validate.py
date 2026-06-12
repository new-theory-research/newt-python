"""Validator for NT v0.0.3 episode directories.

Ported from rebot-bench ``validate.py``. A standalone check, not a dependency on
any nt-platform validator (read as reference only). It answers one question: is
this directory a valid, complete NT episode a consumer can trust?

Checks, in order:
  1. episode.json present and parses — the atomic completeness marker. Its
     absence means the episode is partial and must be skipped.
  2. format_version is the one format this library writes (single-format law).
  3. data.mcap present and readable.
  4. A robot_state channel is present with more than zero messages.
  5. State timestamps are monotonic non-decreasing.
  6. For each camera, the MCAP color-marker count equals the encoded video frame
     count from ffprobe (the frame-count invariant).

``mcap`` ships with the ``recording`` extra and is imported lazily through the
lantern guard.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from newt.recording._lantern import require
from newt.recording._writer import FORMAT_VERSION


def validate(episode_dir: Path) -> dict:
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    meta_path = episode_dir / "episode.json"
    if not meta_path.exists():
        record("episode_json_present", False, "episode.json missing — episode is partial, skip it")
        return _verdict(episode_dir, checks)
    try:
        meta = json.loads(meta_path.read_text())
        record("episode_json_present", True, "present and parses")
    except (json.JSONDecodeError, OSError) as exc:
        record("episode_json_present", False, f"episode.json does not parse: {exc}")
        return _verdict(episode_dir, checks)

    version = meta.get("format_version")
    record(
        "format_version",
        version == FORMAT_VERSION,
        f"format_version is {version!r}"
        + ("" if version == FORMAT_VERSION else f" — expected {FORMAT_VERSION!r} (single-format law)"),
    )

    mcap_path = episode_dir / "data.mcap"
    if not mcap_path.exists():
        record("data_mcap_present", False, "data.mcap missing")
        return _verdict(episode_dir, checks)
    record("data_mcap_present", True, "present")

    try:
        state_topics, state_count, monotonic, camera_marker_counts = _scan_mcap(mcap_path)
    except Exception as exc:
        record("data_mcap_readable", False, f"data.mcap is not readable: {exc}")
        return _verdict(episode_dir, checks)
    record("data_mcap_readable", True, "readable")

    if state_count > 0 and state_topics:
        record("robot_state_channel", True, f"{state_topics[0]} with {state_count} messages")
    else:
        record("robot_state_channel", False, "no robot_state channel with messages found")

    record(
        "timestamps_monotonic",
        monotonic,
        "state timestamps non-decreasing" if monotonic else "state timestamps go backwards",
    )

    cameras = meta.get("camera_config", {}).get("cameras", [])
    if not cameras:
        record("frame_count_invariant", True, "no cameras — invariant not applicable")
    else:
        for cam in cameras:
            cam_id = cam["id"]
            markers = camera_marker_counts.get(cam_id, 0)
            video = episode_dir / "cameras" / cam_id / "color.mp4"
            if not video.exists():
                record(f"frame_count[{cam_id}]", False, f"color.mp4 missing for camera '{cam_id}'")
                continue
            frames = _ffprobe_frame_count(video)
            ok = markers == frames
            record(
                f"frame_count[{cam_id}]",
                ok,
                f"{markers} MCAP markers vs {frames} video frames" + ("" if ok else " — MISMATCH"),
            )

    return _verdict(episode_dir, checks)


def _scan_mcap(path: Path):
    reader_mod = require("mcap.reader", "mcap")
    state_topics: list[str] = []
    state_count = 0
    last_state_ns = -1
    monotonic = True
    camera_markers: dict[str, int] = {}

    with open(path, "rb") as f:
        reader = reader_mod.make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if topic.startswith("robot_state/"):
                state_count += 1
                if topic not in state_topics:
                    state_topics.append(topic)
                if message.log_time < last_state_ns:
                    monotonic = False
                last_state_ns = message.log_time
            elif topic.startswith("camera/") and topic.endswith("/color"):
                cam_id = topic.split("/")[1]
                camera_markers[cam_id] = camera_markers.get(cam_id, 0) + 1
    return state_topics, state_count, monotonic, camera_markers


def _ffprobe_frame_count(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", str(path),
        ],
        capture_output=True, text=True,
    )
    text = out.stdout.strip()
    return int(text) if text.isdigit() else 0


def _verdict(episode_dir: Path, checks: list[dict]) -> dict:
    passed = all(c["ok"] for c in checks)
    return {"episode": str(episode_dir), "valid": passed, "checks": checks}
