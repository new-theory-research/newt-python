"""newt.live — the session on a screen while it happens.

Install: ``pip install "newt[view]"``.

One class and one exception:

- ``LiveView``            — a second sink on a Session's reads. Publishes the
                             robot, the cameras and the joint traces to a Rerun
                             stream on this machine, and serves the compact page
                             that displays them. Attach it with
                             ``Session.attach_observer(view)``.
- ``LiveViewUnavailable`` — every way the view cannot be built, each instance
                             naming which input was missing and whose it was.

``import newt`` does not import this, and nothing here is on the recording path:
a rig with no view installed records exactly the episodes it recorded before.
"""
from __future__ import annotations

from newt.live._view import (
    DEFAULT_GRPC_PORT,
    DEFAULT_PORT,
    LiveView,
    LiveViewUnavailable,
)

__all__ = [
    "DEFAULT_GRPC_PORT",
    "DEFAULT_PORT",
    "LiveView",
    "LiveViewUnavailable",
]
