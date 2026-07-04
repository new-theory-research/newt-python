"""A test-double RecordingSource for `newt record --source` CLI tests.

Stands in for a developer's own hardware module — imported by MODULE:FACTORY
spec, never by name, exactly as the CLI loader would import a real rig. Built
on the featherweight seam (`newt.recording._seam`) so no `[recording]` extra
is required to import this file.
"""
from __future__ import annotations

from newt.recording import SINGLE_ARM_DESCRIPTOR, SimulatedSource


def make_source():
    """Factory the CLI's --source loader calls with no arguments."""
    return SimulatedSource(SINGLE_ARM_DESCRIPTOR)


def make_raising_source():
    """A factory that always raises during construction — proves the CLI's
    --source loader surfaces the failure loudly instead of falling back to
    simulate."""
    raise RuntimeError("fake hardware initialization failure (test double)")
