"""Purity goldens — `import newt` stays featherweight, recording stays guarded.

These run in core-only CI (no `recording` extra installed). They pin the
brief-252 promise: a developer who only runs inference never pays for mcap, cv2,
or protobuf, and a developer who reaches for recording without the extra gets a
lantern that names the fix — not a bare ImportError.

If one of these breaks, the featherweight guarantee has regressed, not an
implementation detail.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


def test_bare_import_newt_pulls_no_recording_deps():
    """A developer runs `import newt` and none of mcap/cv2/protobuf load.

    We import newt in a FRESH interpreter (so nothing a prior test imported
    pollutes sys.modules) and assert the heavy recording deps are absent. This is
    the brief-252 tripwire: recording deps must not leak into the core import path.
    """
    code = (
        "import sys; import newt; "
        "leaked = [m for m in ('mcap', 'cv2', 'google.protobuf') if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, f"`import newt` failed: {out.stderr}"
    leaked = out.stdout.strip()
    assert leaked == "", (
        f"`import newt` leaked recording deps into sys.modules: {leaked!r}. "
        "Recording deps (mcap/cv2/protobuf) must stay lazy behind newt[recording]."
    )


def test_importing_recording_package_stays_featherweight():
    """`import newt.recording` itself pulls in only stdlib (the seam) — no mcap/cv2.

    Touching the recording package to construct a descriptor or a simulated source
    must not force the heavy deps. They load only when you actually record or
    validate.
    """
    code = (
        "import sys; import newt.recording; "
        "from newt.recording import SimulatedSource, SINGLE_ARM_DESCRIPTOR; "
        "SimulatedSource(SINGLE_ARM_DESCRIPTOR).read_state(); "
        "leaked = [m for m in ('mcap', 'cv2', 'google.protobuf') if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, f"`import newt.recording` failed: {out.stderr}"
    leaked = out.stdout.strip()
    assert leaked == "", (
        f"`import newt.recording` + seam use leaked heavy deps: {leaked!r}. "
        "Only stdlib may load until a Session actually records or validates."
    )


def _mcap_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("mcap") is not None


@pytest.mark.skipif(
    _mcap_installed(),
    reason="extra IS installed — the lantern only fires when mcap is absent",
)
def test_recording_without_extra_lights_the_lantern():
    """Without the extra, recording raises a lantern that names `newt[recording]`.

    A developer who calls into the writer path without installing the extra must
    see the install command, not a stack trace ending in `ModuleNotFoundError:
    No module named 'mcap'`.
    """
    from newt.recording import Session, SimulatedSource, SINGLE_ARM_DESCRIPTOR
    from newt.recording._lantern import RecordingExtraMissing

    session = Session(
        SimulatedSource(SINGLE_ARM_DESCRIPTOR),
        task="t",
        output_dir="/tmp/newt_record_lantern_test",
    )
    with pytest.raises(RecordingExtraMissing) as exc:
        session.start_episode()  # constructs the EpisodeWriter -> needs mcap
    msg = str(exc.value)
    assert 'pip install "newt[recording]"' in msg, (
        f"lantern must name the install command; got: {msg!r}"
    )
    assert "mcap" in msg, "lantern should name the missing dep"
