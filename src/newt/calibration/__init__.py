"""newt.calibration — telling the software where a rig's cameras are, honestly.

``newt calibrate`` is the frontend; this is the seam and the contract. The split
is the one ``newt record``, ``newt teleop`` and ``newt rest`` already set: the CLI
parses, guards and renders, and everything about *when* a thing happens lives
here.

**The kit declares the procedure; this module knows no hardware.** A rig says
what calibrating means for it — which passes, which board, which cameras, which
solver, what number it calls acceptable — by implementing
:class:`CalibrationProcedure`. This module executes that declaration, announces
the motion before it starts, judges the answer against the rig's own threshold,
shows the judgment, and only then lets the rig write. It holds no board, no pose
set, no pass count, no threshold and no knowledge of any particular rig.
Everything it knows about an embodiment, the embodiment told it.

**Nothing reaches disk before the verdict reaches the operator.** The write is
downstream of a judgment somebody can see, on every path, always. That ordering
is the reason this verb exists: the system it replaces computes a reprojection
error, renders it as a bare ``· 2.34px``, and saves regardless — a 50-pixel solve
landing as readily as a 0.2-pixel one.

**Skipping is a supported path and it is loud.** ``newt calibrate --skip`` writes
the rig's geometry-invalid declaration, and that declaration travels to every
episode recorded afterwards. Identity geometry is either a decision the operator
made and can see, or an error. Never an absence, never a shaped-right default
nobody chose.

**Nothing here reads a name.** Camera identifiers and pass names are compared,
printed, and never parsed — no substring is special and no convention is
inferred. A rig whose only arm is ``arm0`` calibrates.

Exit codes:
  0    solved and every camera is inside the rig's declared threshold — or the
       run was a ``--skip``, which succeeded at exactly what was asked of it
  1    a usage error, the source refused to come up, or the operator stopped it
       before anything moved
  2    refused before anything moved — the rig declares nothing to run
  3    refused before anything moved — the declared cameras are not the cameras
       this machine can see
  4    a pass did not finish; the rig moved and the run stopped part-way through
  5    the solve did not answer, or the answer could not be saved — either way
       nothing was judged into a config
  6    solved, judged, written, and outside the declared threshold — the result
       carries a suspect label and so will every episode recorded under it
  130  interrupted (Ctrl+C)
"""
from __future__ import annotations

from newt.calibration._run import (
    SKIPPED_WITHOUT_REASON,
    announce_motion,
    confirm_motion,
    render_verdict,
    run_calibration,
    run_skip,
)
from newt.calibration._seam import (
    EXIT_CALIBRATED,
    EXIT_CAMERAS_UNAVAILABLE,
    EXIT_NOTHING_DECLARED,
    EXIT_PASS_FAILED,
    EXIT_SOLVE_FAILED,
    EXIT_SUSPECT,
    EXIT_USAGE,
    CalibrationError,
    CalibrationPass,
    CalibrationProcedure,
    CameraQuality,
    CamerasUnavailable,
    NothingDeclared,
    Notice,
    Verdict,
    check_cameras,
    declared_passes,
    declared_threshold,
    judge,
    require_calibration_procedure,
)

__all__ = [
    "EXIT_CALIBRATED",
    "EXIT_CAMERAS_UNAVAILABLE",
    "EXIT_NOTHING_DECLARED",
    "EXIT_PASS_FAILED",
    "EXIT_SOLVE_FAILED",
    "EXIT_SUSPECT",
    "EXIT_USAGE",
    "SKIPPED_WITHOUT_REASON",
    "CalibrationError",
    "CalibrationPass",
    "CalibrationProcedure",
    "CameraQuality",
    "CamerasUnavailable",
    "NothingDeclared",
    "Notice",
    "Verdict",
    "announce_motion",
    "check_cameras",
    "confirm_motion",
    "declared_passes",
    "declared_threshold",
    "judge",
    "render_verdict",
    "require_calibration_procedure",
    "run_calibration",
    "run_skip",
]
