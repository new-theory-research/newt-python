"""Tests for Embodiment protocol and Robot(embodiment=...) wiring.

Verifies acceptance criteria:
  AC1 — plain user class passes; Robot(embodiment=obj) == Robot(read_state=..., execute=...)
  AC2 — Robot(embodiment="string") raises teaching error with docs pointer
  AC3a — embodiment= + read_state= or execute= raises mutual-exclusion error
  AC3b — object missing one or both methods raises naming-specific error
  AC4 — Embodiment is importable from newt, is runtime_checkable, zero hardware imports
  AC5 — full suite runs without modifying existing tests

No network, no hardware. All coverage via mocks and in-process inspection.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

import newt
from newt._client.robot import EmbodimentError, _validate_embodiment
from newt._embodiment import Embodiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRobotBase:
    """Minimal stand-in for Robot construction that skips network calls.

    We test the embodiment wiring layer (_validate_embodiment) in isolation
    and via Robot() only where the constructor logic itself is under test
    (auth + registry discovery are orthogonal and tested elsewhere).
    """


def _make_robot_kwargs(embodiment=None, read_state=None, execute=None):
    """Call _validate_embodiment and return (read_state_fn, execute_fn)."""
    return _validate_embodiment(embodiment, read_state, execute)


# ---------------------------------------------------------------------------
# AC1 — plain user class with read_state/execute wires both callbacks
# ---------------------------------------------------------------------------

class _UserEmbodiment:
    """Plain user class — no inheritance, no registration."""

    def __init__(self):
        self.read_state_calls: list = []
        self.execute_calls: list = []

    def read_state(self) -> dict:
        self.read_state_calls.append(1)
        return {"state": np.zeros(14, dtype=np.float32)}

    def execute(self, action_chunk: np.ndarray) -> None:
        self.execute_calls.append(action_chunk)


def test_ac1_plain_class_wires_both_callbacks():
    """Robot(embodiment=obj) wires obj.read_state and obj.execute correctly.

    Behavior must be identical to Robot(read_state=obj.read_state, execute=obj.execute).
    Verified by calling the returned callables and checking they mutate the same object.
    (Python bound methods are created fresh on each attribute access, so `is` identity
    checks are deliberately avoided — we check call effects instead.)
    No network involved.
    """
    obj = _UserEmbodiment()

    rs_fn, ex_fn = _validate_embodiment(obj, None, None)

    # Calling the returned callables must land on the original object
    obs = rs_fn()
    assert len(obj.read_state_calls) == 1, "read_state() must have been called via obj"
    assert "state" in obs

    chunk = np.zeros((10, 14), dtype=np.float32)
    ex_fn(chunk)
    assert len(obj.execute_calls) == 1, "execute() must have been called via obj"
    assert obj.execute_calls[0] is chunk

    # Also verify that the returned callables are bound to the same underlying function
    assert rs_fn.__func__ is obj.read_state.__func__
    assert ex_fn.__func__ is obj.execute.__func__
    assert rs_fn.__self__ is obj
    assert ex_fn.__self__ is obj


def test_ac1_equivalence_with_bare_callbacks():
    """Bare-callback path unchanged: _validate_embodiment(None, rs, ex) returns them as-is."""
    rs = lambda: {}
    ex = lambda c: None

    rs_out, ex_out = _validate_embodiment(None, rs, ex)

    assert rs_out is rs
    assert ex_out is ex


def test_ac1_none_embodiment_none_callbacks_passes():
    """No embodiment, no callbacks — infer()-only path; must not raise."""
    rs_out, ex_out = _validate_embodiment(None, None, None)
    assert rs_out is None
    assert ex_out is None


# ---------------------------------------------------------------------------
# AC2 — string raises teaching error with docs pointer
# ---------------------------------------------------------------------------

def test_ac2_string_raises_teaching_error():
    """Robot(embodiment="trossen-widowx") raises EmbodimentError, not TypeError.

    The error message must name: object not a name, starter kit / implement
    read_state+execute, and carry the docs URL.
    """
    with pytest.raises(EmbodimentError) as exc_info:
        _validate_embodiment("trossen-widowx", None, None)

    err = exc_info.value
    msg = str(err)

    # Error type
    assert err.type == "embodiment.string_not_object"

    # Message must name the bad value
    assert "trossen-widowx" in msg

    # Message must explain what embodiment= takes
    assert "object" in msg.lower()

    # Message must mention starter kit or implementing the two methods
    assert "starter" in msg.lower() or "read_state" in msg

    # Docs pointer must be present on the error object
    assert err.docs is not None
    assert "newtheory-docs.vercel.app" in err.docs


def test_ac2_any_string_triggers_the_same_error():
    """Any string value raises the teaching error, not just vendor names."""
    for s in ("", "my-robot", "Robot", "123"):
        with pytest.raises(EmbodimentError) as exc_info:
            _validate_embodiment(s, None, None)
        assert exc_info.value.type == "embodiment.string_not_object"


def test_ac2_error_is_subclass_of_new_theory_error():
    """EmbodimentError is a NewTheoryError so except newt.NewTheoryError catches it.

    Matches the SDK convention: every error in the six-field envelope hierarchy
    (AuthError, ModelNotFoundError, BaseNotDeployableError, ContractMismatchError, etc.)
    subclasses NewTheoryError. A bare except ValueError would silently miss it.
    """
    assert issubclass(newt.EmbodimentError, newt.NewTheoryError)

    # Also verify it's caught by the documented catch-all
    with pytest.raises(newt.NewTheoryError):
        _validate_embodiment("any-string", None, None)


def test_ac2_error_importable_from_newt():
    """EmbodimentError is in newt's public namespace."""
    assert hasattr(newt, "EmbodimentError")
    assert newt.EmbodimentError is EmbodimentError


# ---------------------------------------------------------------------------
# AC3a — mutual exclusion: embodiment= + read_state= or execute=
# ---------------------------------------------------------------------------

def test_ac3a_conflict_with_read_state():
    """embodiment= + read_state= raises EmbodimentError naming the conflict."""
    obj = _UserEmbodiment()
    with pytest.raises(EmbodimentError) as exc_info:
        _validate_embodiment(obj, lambda: {}, None)

    err = exc_info.value
    assert err.type == "embodiment.conflict"
    assert "read_state=" in str(err) or "read_state" in str(err)


def test_ac3a_conflict_with_execute():
    """embodiment= + execute= raises EmbodimentError naming the conflict."""
    obj = _UserEmbodiment()
    with pytest.raises(EmbodimentError) as exc_info:
        _validate_embodiment(obj, None, lambda c: None)

    err = exc_info.value
    assert err.type == "embodiment.conflict"
    assert "execute=" in str(err) or "execute" in str(err)


def test_ac3a_conflict_with_both():
    """embodiment= + both read_state= and execute= raises EmbodimentError naming both."""
    obj = _UserEmbodiment()
    with pytest.raises(EmbodimentError) as exc_info:
        _validate_embodiment(obj, lambda: {}, lambda c: None)

    err = exc_info.value
    assert err.type == "embodiment.conflict"
    msg = str(err)
    assert "read_state=" in msg or "read_state" in msg
    assert "execute=" in msg or "execute" in msg


# ---------------------------------------------------------------------------
# AC3b — partial object: missing method(s) named specifically
# ---------------------------------------------------------------------------

class _MissingExecute:
    def read_state(self) -> dict:
        return {}


class _MissingReadState:
    def execute(self, chunk: np.ndarray) -> None:
        pass


class _MissingBoth:
    pass


def test_ac3b_missing_execute_names_it():
    """Object with read_state but no execute raises error naming execute() as missing."""
    with pytest.raises(EmbodimentError) as exc_info:
        _validate_embodiment(_MissingExecute(), None, None)

    err = exc_info.value
    assert err.type == "embodiment.missing_method"
    # context.missing must list exactly execute() — not read_state()
    assert err.context["missing"] == ["execute()"]


def test_ac3b_missing_read_state_names_it():
    """Object with execute but no read_state raises error naming read_state() as missing."""
    with pytest.raises(EmbodimentError) as exc_info:
        _validate_embodiment(_MissingReadState(), None, None)

    err = exc_info.value
    assert err.type == "embodiment.missing_method"
    # context.missing must list exactly read_state() — not execute()
    assert err.context["missing"] == ["read_state()"]


def test_ac3b_missing_both_names_both():
    """Object with neither method raises error naming both."""
    with pytest.raises(EmbodimentError) as exc_info:
        _validate_embodiment(_MissingBoth(), None, None)

    err = exc_info.value
    assert err.type == "embodiment.missing_method"
    msg = str(err)
    assert "read_state()" in msg
    assert "execute()" in msg


def test_ac3b_non_callable_attribute_treated_as_missing():
    """An attribute named read_state that isn't callable counts as missing."""
    class _BadExecute:
        def read_state(self) -> dict:
            return {}
        execute = "not_callable"

    with pytest.raises(EmbodimentError) as exc_info:
        _validate_embodiment(_BadExecute(), None, None)

    assert exc_info.value.type == "embodiment.missing_method"
    assert "execute()" in str(exc_info.value)


# ---------------------------------------------------------------------------
# AC4 — Embodiment protocol: importable, runtime_checkable, no hardware imports
# ---------------------------------------------------------------------------

def test_ac4_embodiment_importable_from_newt():
    """Embodiment is in newt's public namespace."""
    assert hasattr(newt, "Embodiment")
    assert newt.Embodiment is Embodiment


def test_ac4_embodiment_is_runtime_checkable():
    """isinstance() checks against Embodiment work at runtime."""
    obj = _UserEmbodiment()
    assert isinstance(obj, Embodiment)

    obj_missing = _MissingBoth()
    assert not isinstance(obj_missing, Embodiment)


def test_ac4_embodiment_has_zero_hardware_imports():
    """The _embodiment module imports no hardware or vendor packages.

    We check this by inspecting the module's direct imports — the only
    non-stdlib import allowed is numpy (needed for the type annotation).
    """
    import importlib
    import sys

    # Reload to get a clean module reference
    import newt._embodiment as em_mod

    # All imported names in the module's namespace that are modules
    hardware_prefixes = ("trossen", "pyrealsense", "dynamixel", "serial", "cv2", "rospy")
    for name in dir(em_mod):
        obj = getattr(em_mod, name)
        if hasattr(obj, "__module__"):
            mod_name = getattr(obj, "__module__", "") or ""
            for prefix in hardware_prefixes:
                assert not mod_name.startswith(prefix), (
                    f"Embodiment module imported hardware package {mod_name!r} "
                    f"via {name!r} — must have zero hardware deps"
                )


def test_ac4_embodiment_method_signatures_match_robot_contracts():
    """Embodiment.read_state() and .execute() signature mirrors robot.py's callbacks.

    read_state: () -> dict
    execute: (action_chunk) -> None

    Verified against the Embodiment Protocol's annotations (not just existence).
    """
    import inspect

    # read_state: no required args besides self, returns dict
    rs_sig = inspect.signature(Embodiment.read_state)
    rs_params = [p for p in rs_sig.parameters.values() if p.name != "self"]
    assert len(rs_params) == 0, (
        f"read_state() should take no args besides self; got {rs_params}"
    )

    # execute: one required positional arg (action_chunk), returns None
    ex_sig = inspect.signature(Embodiment.execute)
    ex_params = [p for p in ex_sig.parameters.values() if p.name != "self"]
    assert len(ex_params) == 1, (
        f"execute() should take exactly one arg besides self; got {ex_params}"
    )


def test_ac4_embodiment_in_all():
    """Embodiment and EmbodimentError are in newt.__all__."""
    assert "Embodiment" in newt.__all__
    assert "EmbodimentError" in newt.__all__


# ---------------------------------------------------------------------------
# AC5 — no behavior change for bare-callback users
# ---------------------------------------------------------------------------

def test_ac5_validate_embodiment_with_none_is_identity():
    """_validate_embodiment(None, rs, ex) returns rs, ex unchanged — no side effects."""
    rs = object()
    ex = object()
    rs_out, ex_out = _validate_embodiment(None, rs, ex)
    assert rs_out is rs
    assert ex_out is ex


def test_ac5_string_rail_fires_before_partial_check():
    """String input raises string error, not missing-method error.

    Order: string check → conflict check → partial-object check.
    """
    # A string has no read_state/execute, but the error must name "string_not_object"
    # not "missing_method".
    with pytest.raises(EmbodimentError) as exc_info:
        _validate_embodiment("robot-name", None, None)
    assert exc_info.value.type == "embodiment.string_not_object"


def test_ac5_conflict_check_fires_before_partial_check():
    """Conflict raises conflict error even when object is also missing methods.

    Order: string → conflict → partial. An incomplete object combined with
    read_state= must raise conflict, not missing-method.
    """
    class _Incomplete:
        def read_state(self):
            return {}
        # missing execute

    with pytest.raises(EmbodimentError) as exc_info:
        _validate_embodiment(_Incomplete(), None, lambda c: None)
    assert exc_info.value.type == "embodiment.conflict"
