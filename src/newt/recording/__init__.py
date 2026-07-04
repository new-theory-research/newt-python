"""newt.recording — the Session library and the NT v0.0.3 episode format.

Install: ``pip install "newt[recording]"``.

The public surface:
- ``Session``          — all recording behavior; the layering moat. Frontends
                          (``newt record``, an agent on ``--json``) hold none of it.
- ``RecordingSource``  — the seam protocol an embodiment implements (``read_state``
                          + ``descriptor``, optional ``disable_all`` / ``close``).
- ``StateDescriptor``  — the static shape of a state stream (arms, channels, joints).
- ``JointState``       — one synchronized joint snapshot.
- ``SimulatedSource``  — a hardware-free source for ``--simulate`` and tests.
- ``Sink``             — the seam protocol a destination implements (``deliver``);
                          what a Session hands a committed episode directory to.
- ``LocalSink``        — today's local-disk destination behind ``Sink``; the
                          writer already commits into ``output_dir``, so this
                          sink only verifies place, never moves or copies.
- ``NTCloudSink``      — uploads a committed episode to the developer's NT
                          cloud namespace via a server-minted signed URL per
                          file; never holds the GCS write credential.
- ``CameraSpec``       — a configured RGB camera (lazy; needs the extra).
- ``validate``         — validate an NT v0.0.3 episode directory (lazy; needs the extra).

``import newt`` stays featherweight: it does NOT import this package, and importing
``newt.recording`` itself pulls in only stdlib (the seam). The heavy deps — ``mcap``,
``opencv-python``, ``protobuf`` — are imported lazily the moment you actually record
or validate, and a lantern names ``pip install "newt[recording]"`` if the extra is
missing.
"""
from __future__ import annotations

# Featherweight seam — stdlib only, safe to import without the extra.
from newt.recording._lantern import RecordingExtraMissing
from newt.recording._seam import (
    BIMANUAL_DESCRIPTOR,
    SINGLE_ARM_DESCRIPTOR,
    JointState,
    RecordingSource,
    SimulatedSource,
    StateDescriptor,
)
from newt.recording._cloud_sink import NTCloudSink
from newt.recording._session import DEFAULT_STATE_HZ, Session, SessionStatus
from newt.recording._sink import LocalSink, Sink

__all__ = [
    "BIMANUAL_DESCRIPTOR",
    "CameraSpec",
    "DEFAULT_STATE_HZ",
    "JointState",
    "LocalSink",
    "NTCloudSink",
    "RecordingExtraMissing",
    "RecordingSource",
    "SINGLE_ARM_DESCRIPTOR",
    "Sink",
    "SimulatedSource",
    "Session",
    "SessionStatus",
    "StateDescriptor",
    "validate",
]


def __getattr__(name: str):
    """Lazy access to the heavy surface: ``CameraSpec`` and ``validate`` pull in
    recording-extra deps on first touch (with the lantern if the extra is absent),
    so naming them in ``__all__`` never forces ``mcap`` at import time."""
    if name == "CameraSpec":
        from newt.recording._writer import CameraSpec

        return CameraSpec
    if name == "validate":
        from newt.recording._validate import validate

        return validate
    raise AttributeError(f"module 'newt.recording' has no attribute {name!r}")
