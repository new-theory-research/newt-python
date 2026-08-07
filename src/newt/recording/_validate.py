"""Validator for NT v0.0.3 episode directories.

Ported from rebot-bench ``validate.py``. A standalone check, not a dependency on
any nt-platform validator (read as reference only). It answers one question: is
this directory a valid, complete NT episode a consumer can trust?

Checks, in order:
  1. episode.json present and parses — the atomic completeness marker. Its
     absence means the episode is partial and must be skipped.
  2. format_version is the one format this library writes (single-format law).
  3. data.mcap present and structurally readable.
  4. Every protobuf channel decodes at least one message with its declared schema.
  5. A robot_state channel is present with more than zero messages.
  6. State timestamps are monotonic non-decreasing.
  7. For each camera, the MCAP color-marker count equals the encoded video frame
     count from ffprobe (the frame-count invariant).

``mcap`` ships with the ``recording`` extra and is imported lazily through the
lantern guard.
"""
from __future__ import annotations

import json
import math
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

    if not isinstance(meta, dict):
        raise TypeError(
            "episode.json contains "
            f"a JSON {type(meta).__name__}, not an object; the recording writer "
            "must write an object and rewrite this episode"
        )

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
        scan = _scan_mcap(mcap_path)
        state_topics, state_count, monotonic, camera_marker_counts, decode_failures = scan
    except Exception as exc:  # noqa: BLE001 — this function's whole job is running
        # a battery of checks and recording pass/fail per check (see the pattern
        # above and below); "data_mcap_readable" IS the check for "does this parse
        # at all," so whatever _scan_mcap raises becomes that check's False, not a
        # crash of the validator itself.
        record("data_mcap_readable", False, f"data.mcap is not readable: {exc}")
        return _verdict(episode_dir, checks)
    record("data_mcap_readable", True, "container records are structurally readable")

    record(
        "protobuf_channels_decodable",
        not decode_failures,
        "one message decoded on every protobuf channel"
        if not decode_failures
        else "; ".join(decode_failures),
    )

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
    for cam in cameras:
        cam_id = cam.get("id", "<missing-id>")
        ok, detail = _geometry_provenance_check(cam)
        record(f"geometry_provenance[{cam_id}]", ok, detail)
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


def _geometry_provenance_check(camera: dict) -> tuple[bool, str]:
    provenance = camera.get("geometry_provenance")
    if provenance is None:
        return True, "no geometry claim — episode predates explicit geometry provenance"
    if not isinstance(provenance, dict):
        return False, _geometry_repair("geometry_provenance must be an object")

    kind = provenance.get("kind")
    extrinsics = camera.get("extrinsics")
    if kind == "invalid":
        if extrinsics is not None:
            return False, _geometry_repair("invalid geometry also carries extrinsics")
        if not _nonempty(provenance.get("reason")):
            return False, _geometry_repair("invalid geometry has no non-empty reason")
        return True, f"explicitly invalid: {provenance['reason']}"
    if kind not in {"measured", "declared"}:
        return False, _geometry_repair(f"unknown geometry provenance kind {kind!r}")
    if not _is_extrinsics(extrinsics):
        return False, _geometry_repair(f"{kind} geometry has no finite 4x4 extrinsics matrix")
    if kind == "measured":
        receipt = provenance.get("receipt")
        if not _portable_receipt(receipt):
            return False, _geometry_repair(
                "measured geometry has no portable episode-relative receipt"
            )
        return True, f"measured geometry; calibration receipt: {receipt}"
    source = provenance.get("source")
    if not _nonempty(source):
        return False, _geometry_repair("declared geometry has no non-empty source")
    return True, f"declared geometry; source: {source}"


def _geometry_repair(cause: str) -> str:
    return (
        f"{cause}; episode metadata is inconsistent — the recording owner must repair "
        "the camera's extrinsics and geometry_provenance together"
    )


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _portable_receipt(value: object) -> bool:
    if not _nonempty(value):
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _is_extrinsics(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(row, list)
            and len(row) == 4
            and all(
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(number)
                for number in row
            )
            for row in value
        )
    )


def _scan_mcap(path: Path):
    reader_mod = require("mcap.reader", "mcap")
    state_topics: list[str] = []
    state_count = 0
    last_state_ns = -1
    monotonic = True
    camera_markers: dict[str, int] = {}
    decoded_channels: set[int] = set()
    decode_failures: list[str] = []

    with open(path, "rb") as f:
        reader = reader_mod.make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if (
                schema is not None
                and schema.encoding == "protobuf"
                and channel.id not in decoded_channels
            ):
                decoded_channels.add(channel.id)
                failure = _protobuf_decode_failure(schema, channel, message.data)
                if failure is not None:
                    decode_failures.append(failure)
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
    return state_topics, state_count, monotonic, camera_markers, decode_failures


def _protobuf_decode_failure(schema, channel, payload: bytes) -> str | None:
    """Return a cause-specific failure for the channel's first message."""
    descriptor_pb2 = require("google.protobuf.descriptor_pb2", "protobuf")
    descriptor_pool = require("google.protobuf.descriptor_pool", "protobuf")
    protobuf_message = require("google.protobuf.message", "protobuf")
    message_factory = require("google.protobuf.message_factory", "protobuf")

    leading_text = schema.data[:80].decode("utf-8", errors="replace").lstrip()
    if leading_text.startswith("syntax"):
        found = leading_text.splitlines()[0]
        return (
            f"channel {channel.topic!r}: protobuf schema payload is .proto source text "
            f"({found!r}), not a serialized FileDescriptorSet; the recording "
            "writer must "
            "store descriptor-set bytes and rewrite this episode"
        )

    descriptor_set = descriptor_pb2.FileDescriptorSet()
    try:
        descriptor_set.ParseFromString(schema.data)
    except protobuf_message.DecodeError as exc:
        return (
            f"channel {channel.topic!r}: protobuf schema payload is not a serialized "
            f"FileDescriptorSet ({exc}); the recording writer must store "
            "descriptor-set bytes and rewrite this episode"
        )

    try:
        pool = descriptor_pool.DescriptorPool()
        for file_descriptor in descriptor_set.file:
            pool.Add(file_descriptor)
        message_descriptor = pool.FindMessageTypeByName(schema.name)
        message_type = message_factory.GetMessageClass(message_descriptor)
    except (KeyError, TypeError, ValueError) as exc:
        return (
            f"channel {channel.topic!r}: FileDescriptorSet cannot rebuild "
            f"protobuf message {schema.name!r} ({exc}); the recording writer "
            "must emit a complete, internally consistent descriptor set"
        )

    try:
        message_type.FromString(payload)
    except protobuf_message.DecodeError as exc:
        return (
            f"channel {channel.topic!r}: schema rebuilt, but its first message "
            f"does not decode as {schema.name!r} ({exc}); the recording writer "
            "must encode messages with the schema registered on this channel"
        )
    return None


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


def _verdict(episode_dir: Path, checks: list[dict]) -> dict:
    passed = all(c["ok"] for c in checks)
    return {"episode": str(episode_dir), "valid": passed, "checks": checks}
