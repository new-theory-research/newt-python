"""newt.live — the session on a screen while it happens.

Install: ``pip install "newt[view]"``.

Looking:

- ``LiveView``            — a second sink on a Session's reads. Publishes the
                             robot, the cameras and the joint traces to a Rerun
                             stream on this machine, and serves the compact page
                             that displays them. Attach it with
                             ``Session.attach_observer(view)``.
- ``LiveViewUnavailable`` — every way the view cannot be built, each instance
                             naming which input was missing and whose it was.

Driving:

- ``SessionControl``      — start a take, stop a take, read the session's state
                             and the list of takes it has recorded. The contract
                             a collection page consumes, and the whole of it: four
                             operations, nothing that knows what a robot is.
- ``ControlRefused``      — a control request that cannot be honoured, carrying an
                             HTTP status, a stable reason slug, and the sentence to
                             show. No two causes share a slug or a sentence.
- ``Take``                — one attempt at recording as the page sees it: its
                             episode id, dataset, task, tags, and where its
                             delivery to the session's sink has got to.

``import newt`` does not import this, and nothing here is on the recording path:
a rig with no view installed records exactly the episodes it recorded before.
"""
from __future__ import annotations

from newt.live._control import ControlRefused, SessionControl, Take
from newt.live._view import (
    DEFAULT_GRPC_PORT,
    DEFAULT_PORT,
    LiveView,
    LiveViewUnavailable,
)

__all__ = [
    "DEFAULT_GRPC_PORT",
    "DEFAULT_PORT",
    "ControlRefused",
    "LiveView",
    "LiveViewUnavailable",
    "SessionControl",
    "Take",
]
