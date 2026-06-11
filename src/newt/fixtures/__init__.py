"""newt.fixtures — deprecated alias for newt.snapshots.

This module is kept for one release. Use ``newt.snapshots`` instead.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "newt.fixtures is deprecated and will be removed in a future release. "
    "Use newt.snapshots instead.",
    DeprecationWarning,
    stacklevel=2,
)

from newt.snapshots import available, load  # noqa: E402, F401

__all__ = ["load", "available"]
