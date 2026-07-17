"""newt.Embodiment — the protocol any hardware driver must satisfy.

An embodiment is any object with two methods: read_state() and execute(chunk).
That's the contract between the user's hardware and the model session. The
scaffolder generates a class implementing this protocol; you can also write
your own for any hardware — no inheritance, no registration required.

Protocol design: portal/wiki/specs/embodiment-noun.md
"""
from __future__ import annotations

from typing import TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

try:
    from typing import Protocol
except ImportError:  # Python 3.7 fallback (typing_extensions)
    from typing_extensions import Protocol  # type: ignore[assignment]


@runtime_checkable
class Embodiment(Protocol):
    """Protocol for any hardware driver passed to Robot(embodiment=...).

    Any object with these two methods is an embodiment. The scaffolder
    (create-newt-robot) generates a class that satisfies this protocol.
    You can also write your own for any hardware — no inheritance, no
    registration. Just implement both methods.

    Methods
    -------
    read_state()
        Return an observation dict for the current sensor snapshot.
        Same shape read_state= expects: optional keys "state" (float32
        ndarray, shape model-dependent — see robot.contract.state_shape),
        "images" (dict of camera arrays keyed by robot.contract.cameras),
        "prompt" (str). Missing fields are firehose-coerced server-side —
        partial dicts (even {}) are fine.

    execute(action_chunk)
        Apply one action chunk to the hardware. Receives an ndarray of
        shape (action_horizon, action_dim) — model-dependent; see
        robot.contract.action_shape — the same chunk execute= receives in
        Robot.run(). Called once per inference cycle in non-stream mode;
        never called in stream mode.
    """

    def read_state(self) -> dict:
        """Return the current observation dict."""
        ...

    def execute(self, action_chunk: np.ndarray) -> None:
        """Apply one action chunk to the hardware."""
        ...
