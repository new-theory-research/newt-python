"""Goldens for the recording sink seam — `Sink`, `LocalSink`, `Session(sink=)`.

The protocol itself is stdlib-only (see `_sink.py`), so the `LocalSink`-only tests
here run in core-only CI. Tests that drive a full `Session` through the capture
loop need the `recording` extra (mcap/protobuf) to construct real episodes, same
as the rest of `test_recording.py`.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

_HAVE_EXTRA = (
    importlib.util.find_spec("mcap") is not None
    and importlib.util.find_spec("google.protobuf") is not None
)
needs_extra = pytest.mark.skipif(not _HAVE_EXTRA, reason="needs the [recording] extra (mcap/protobuf)")


class _RecordingSink:
    """A test-double sink that observes every delivery — nothing else."""

    def __init__(self) -> None:
        self.delivered: list[Path] = []

    def deliver(self, episode_dir: Path) -> None:
        self.delivered.append(Path(episode_dir))


class _RaisingSink:
    """A sink that always fails delivery — proves a failure surfaces loudly
    (Rule 10) instead of being swallowed or counted as a successful keep."""

    def deliver(self, episode_dir: Path) -> None:
        raise RuntimeError("simulated delivery failure")


def test_local_sink_satisfies_sink_protocol(tmp_path):
    """`LocalSink` is a real `Sink` — the protocol is `runtime_checkable` so a
    future cloud sink (or a test double) can be checked the same way."""
    from newt.recording import LocalSink, Sink

    assert isinstance(LocalSink(tmp_path), Sink)
    assert isinstance(_RecordingSink(), Sink)


def test_local_sink_deliver_raises_on_missing_directory(tmp_path):
    """A `LocalSink` asked to deliver a path that doesn't exist raises loudly
    (Rule 10) rather than pretending the hand-off succeeded."""
    from newt.recording import LocalSink

    sink = LocalSink(tmp_path)
    missing = tmp_path / "episode_doesnotexist"
    with pytest.raises(RuntimeError):
        sink.deliver(missing)


def test_local_sink_deliver_raises_when_path_outside_output_dir(tmp_path):
    """A `LocalSink` only vouches for paths under its own `output_dir` — handed
    a real directory that lives elsewhere, it raises instead of silently
    accepting a mismatched hand-off."""
    from newt.recording import LocalSink

    sink = LocalSink(tmp_path / "sink_root")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(RuntimeError):
        sink.deliver(elsewhere)


@needs_extra
def test_sink_delivers_on_keep_and_not_on_discard(tmp_path):
    """The protocol-seam guarantee: a test-double sink observes delivery for
    every kept episode, in order, and none for a discarded one.

    Drives keep, discard, keep — exactly two deliveries, matching the two
    directories that actually land on disk.
    """
    from newt.recording import Session, SimulatedSource, SINGLE_ARM_DESCRIPTOR

    sink = _RecordingSink()
    session = Session(
        SimulatedSource(SINGLE_ARM_DESCRIPTOR),
        task="pick up the cup",
        output_dir=tmp_path,
        sink=sink,
    )
    kept_paths = []
    for verdict in ("keep", "discard", "keep"):
        session.start_episode()
        time.sleep(0.15)
        path = session.end_episode(keep=(verdict == "keep"))
        if path is not None:
            kept_paths.append(path)
    session.close()

    assert session.status().kept == 2
    assert sink.delivered == kept_paths, (
        f"sink should observe exactly the kept paths in order; "
        f"got {sink.delivered}, expected {kept_paths}"
    )


@needs_extra
def test_sink_failure_is_loud_and_not_counted_as_kept(tmp_path):
    """A sink that fails to deliver a kept episode must raise out of
    `end_episode`, not fail silently — and the episode must not be counted as
    kept, since delivery is what makes "kept" true end to end (Rule 10)."""
    from newt.recording import Session, SimulatedSource, SINGLE_ARM_DESCRIPTOR

    session = Session(
        SimulatedSource(SINGLE_ARM_DESCRIPTOR),
        task="pick up the cup",
        output_dir=tmp_path,
        sink=_RaisingSink(),
    )
    session.start_episode()
    time.sleep(0.15)
    with pytest.raises(RuntimeError, match="simulated delivery failure"):
        session.end_episode(keep=True)

    assert session.status().kept == 0, (
        "a failed delivery must not be counted as kept — the increment happens "
        "only after deliver() succeeds"
    )
    session.close()


@needs_extra
def test_omitting_sink_defaults_to_local_sink_and_still_validates(tmp_path):
    """Not passing `sink=` at all must behave exactly as before this brief: the
    episode lands on disk under `output_dir` and passes the validator — the
    default `LocalSink` is a no-op hand-off, not a second write path."""
    from newt.recording import Session, SimulatedSource, SINGLE_ARM_DESCRIPTOR, validate

    session = Session(
        SimulatedSource(SINGLE_ARM_DESCRIPTOR),
        task="pick up the cup",
        output_dir=tmp_path,
    )
    session.start_episode()
    time.sleep(0.15)
    path = session.end_episode(keep=True)
    session.close()

    assert path is not None
    result = validate(path)
    assert result["valid"], f"episode {path} failed validation: {result['checks']}"
