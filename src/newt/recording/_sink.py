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
