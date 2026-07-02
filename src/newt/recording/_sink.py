"""The recording sink — where a committed episode goes after the writer commits it.

The atomic-commit and frame-count invariants (``episode.json``-rename-last, MCAP
marker count == encoded frame count) live entirely in ``EpisodeWriter.keep()``
(``_writer.py``) and do not move here. A ``Sink`` receives an episode directory
only after the writer has already committed it; ``deliver`` is a hand-off point,
never a second write path.

Featherweight on purpose, same as ``_seam.py``: stdlib only, safe to import (and
construct a default sink) without the ``recording`` extra installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Sink(Protocol):
    """The delivery seam. What a ``Session`` hands a committed episode to, and the
    only surface a new destination (e.g. a future cloud sink) implements.

    ``deliver`` is called once per kept episode, after ``EpisodeWriter.keep()``
    has already committed it. The sink receives the finished episode directory
    and is responsible for getting it to its destination — it must not
    re-implement or relocate the commit invariants (those stay recorder-side,
    per ``recording.md`` §2). A sink that cannot deliver a kept episode must
    raise; a silent drop of a kept episode is never acceptable (Rule 10).
    """

    def deliver(self, episode_dir: Path) -> None:
        ...


@dataclass
class LocalSink:
    """Today's local-disk destination, behind the ``Sink`` protocol.

    ``EpisodeWriter.keep()`` already commits the episode directly into
    ``output_dir`` — this sink does not move, copy, or rename anything.
    ``deliver`` is a verification of place, not a second write path: the
    episode is already where it belongs by the time this is called, so a
    correct call is a no-op. It raises loudly (Rule 10) if the delivered path
    is missing or isn't actually under this sink's own ``output_dir`` — a
    kept episode failing delivery must never be a silent drop.
    """

    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    def deliver(self, episode_dir: Path) -> None:
        episode_dir = Path(episode_dir)
        if not episode_dir.exists():
            raise RuntimeError(
                f"LocalSink.deliver: episode directory does not exist: {episode_dir}"
            )
        root = self.output_dir.resolve()
        resolved = episode_dir.resolve()
        if root not in resolved.parents:
            raise RuntimeError(
                f"LocalSink.deliver: {resolved} is not under this sink's "
                f"output_dir {root}"
            )
