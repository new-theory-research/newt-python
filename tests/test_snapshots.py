"""Tests for newt.snapshots and the newt.fixtures deprecation alias.

These tests verify:
- newt.snapshots.load() / available() work correctly (unit, no live endpoint)
- newt.fixtures still works for one release but emits a DeprecationWarning
- The warning text names newt.snapshots (so old-code authors know where to go)
- The alias returns data byte-identical to the real module
"""
from __future__ import annotations

import importlib
import sys
import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# newt.snapshots
# ---------------------------------------------------------------------------


def test_snapshots_available_returns_known_names() -> None:
    """available() lists the two bundled snapshots."""
    import newt.snapshots as snapshots

    names = snapshots.available()
    assert "cup_stacking" in names
    assert "pour_coffee_beans" in names


def test_snapshots_load_returns_expected_structure() -> None:
    """load() returns an obs dict with the three required keys in the right shapes."""
    import newt.snapshots as snapshots

    obs = snapshots.load("cup_stacking")

    assert set(obs.keys()) >= {"state", "images", "prompt"}

    # state: (8,) float32
    assert isinstance(obs["state"], np.ndarray)
    assert obs["state"].dtype == np.float32
    assert obs["state"].shape == (8,)

    # images: three cameras, each (3, 240, 320) uint8
    assert isinstance(obs["images"], dict)
    for cam_name, frame in obs["images"].items():
        assert isinstance(frame, np.ndarray), f"images[{cam_name!r}] must be ndarray"
        assert frame.dtype == np.uint8, f"images[{cam_name!r}] must be uint8"
        assert frame.shape == (3, 240, 320), f"images[{cam_name!r}] shape must be (3,240,320)"

    # prompt: non-empty string
    assert isinstance(obs["prompt"], str)
    assert obs["prompt"]


def test_snapshots_load_cup_stacking_prompt_override() -> None:
    """cup_stacking's prompt is overridden to the clean flagship base task description."""
    import newt.snapshots as snapshots

    obs = snapshots.load("cup_stacking")
    assert obs["prompt"] == "Stack one cup into another cup."


def test_snapshots_load_red_cube_is_so101_shaped() -> None:
    """red_cube is the 6-axis SO-101 frame that unblocks `newt run <so101-fine-tune>`.

    It must carry the exact SO-101 contract shape: a (6,) state, and top/side cameras at
    (3, 224, 224) — the shape the live red-cube-bowl fine-tune declares and the starter
    embodiment sends. This is the whole reason the snapshot exists; a wrong shape here is
    the E2E hard-stop it fixes, back again.
    """
    import newt.snapshots as snapshots

    obs = snapshots.load("red_cube")

    assert obs["state"].dtype == np.float32
    assert obs["state"].shape == (6,), "SO-101 state is 6-dim (5 arm joints + gripper)"

    assert set(obs["images"].keys()) == {"top", "side"}, "SO-101 cameras are top/side"
    for cam in ("top", "side"):
        frame = obs["images"][cam]
        assert frame.dtype == np.uint8
        assert frame.shape == (3, 224, 224), f"images[{cam!r}] must be the (3,224,224) SO-101 shape"

    # The real recorded task prompt rides verbatim (not overridden) — the exact string the
    # red-cube-bowl model trained on, so infer() returns a non-degenerate chunk.
    assert isinstance(obs["prompt"], str) and "red cube" in obs["prompt"].lower()


def test_snapshots_describe_red_cube_signature() -> None:
    """describe() reports red_cube's selection signature — (6,) state, top/side cameras —
    read from the bundled data, which is what contract-aware selection keys off."""
    import newt.snapshots as snapshots

    desc = snapshots.describe("red_cube")
    assert desc["state_shape"] == (6,)
    assert desc["cameras"] == ("top", "side")


def test_snapshots_load_all_available() -> None:
    """load() works for every name returned by available()."""
    import newt.snapshots as snapshots

    for name in snapshots.available():
        obs = snapshots.load(name)
        assert "state" in obs, f"snapshot {name!r} missing 'state'"
        assert "images" in obs, f"snapshot {name!r} missing 'images'"
        assert "prompt" in obs, f"snapshot {name!r} missing 'prompt'"


def test_snapshots_load_unknown_name_raises_key_error() -> None:
    """load() raises KeyError for an unknown snapshot name."""
    import newt.snapshots as snapshots

    with pytest.raises(KeyError, match="not_a_real_snapshot"):
        snapshots.load("not_a_real_snapshot")

    # error message should mention available names
    with pytest.raises(KeyError) as exc_info:
        snapshots.load("not_a_real_snapshot")
    assert "cup_stacking" in str(exc_info.value)


# ---------------------------------------------------------------------------
# newt.fixtures deprecation alias
# ---------------------------------------------------------------------------


def _reload_fixtures_module():
    """Force a fresh import of newt.fixtures so the module-level warning fires."""
    # Remove from cache so the next import re-executes the module body.
    sys.modules.pop("newt.fixtures", None)
    return importlib.import_module("newt.fixtures")


def test_fixtures_import_emits_deprecation_warning() -> None:
    """Importing newt.fixtures emits a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _reload_fixtures_module()

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_warnings, "Expected a DeprecationWarning when importing newt.fixtures"


def test_fixtures_deprecation_warning_names_snapshots() -> None:
    """The DeprecationWarning message tells callers to use newt.snapshots."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _reload_fixtures_module()

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_warnings, "Expected a DeprecationWarning"
    msg = str(deprecation_warnings[0].message)
    assert "newt.snapshots" in msg, (
        f"DeprecationWarning must name 'newt.snapshots' so callers know where to go; got: {msg!r}"
    )


def test_fixtures_load_works_and_returns_identical_data() -> None:
    """newt.fixtures.load() works and returns byte-identical data to newt.snapshots.load()."""
    import newt.snapshots as snapshots

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fixtures = _reload_fixtures_module()

    snap_obs = snapshots.load("cup_stacking")
    fix_obs = fixtures.load("cup_stacking")

    np.testing.assert_array_equal(snap_obs["state"], fix_obs["state"])
    for cam in snap_obs["images"]:
        np.testing.assert_array_equal(snap_obs["images"][cam], fix_obs["images"][cam])
    assert snap_obs["prompt"] == fix_obs["prompt"]


def test_fixtures_available_returns_same_names() -> None:
    """newt.fixtures.available() returns the same list as newt.snapshots.available()."""
    import newt.snapshots as snapshots

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fixtures = _reload_fixtures_module()

    assert fixtures.available() == snapshots.available()
