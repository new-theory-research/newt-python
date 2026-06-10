"""Offline unit tests for Robot.__repr__ / __str__ (brief-234 receipt).

Why these matter:
  The public docs' milestone-1 finish line shows an exact string:
      nt0-fp3 · contract received · (50,8) · 8 labeled axes
  Before this brief, print(Robot()) emitted the default Python object repr —
  making the documented receipt a lie. These tests pin the string so the docs
  can't drift silently.
"""
from __future__ import annotations

import newt

_NT0_AXES = [
    "shoulder_pan", "shoulder_lift", "elbow", "forearm_roll",
    "wrist_angle", "wrist_rotate", "gripper", "extra",
]

_FULL_REGISTRY = [
    {
        "uid": "ft_base_nt0fp3",
        "tags": ["nt0-fp3"],
        "type": "fine_tune",
        "base": "base_nt0fp3",
        "endpoint": "wss://example/stream",
        "contract": {
            "action_shape": [50, 8],
            "action_axes": _NT0_AXES,
        },
    },
]


def _make_robot(registry, model=None):
    """Construct a Robot bypassing __init__ and set internals directly."""
    robot = newt.Robot.__new__(newt.Robot)
    robot._registry = registry
    robot._model = model
    return robot


# ---------------------------------------------------------------------------
# AC 1 — exact documented string when contract + axes are present
# ---------------------------------------------------------------------------

def test_repr_full_contract_matches_docs_receipt():
    """print(Robot()) must produce the exact string the milestone-1 docs show.

    This is the receipt the docs promise. If it breaks, the getting-started
    guide silently lies to every new developer.
    """
    robot = _make_robot(_FULL_REGISTRY)
    expected = "nt0-fp3 · contract received · (50,8) · 8 labeled axes"
    assert repr(robot) == expected
    assert str(robot) == expected  # AC 3: __str__ == __repr__


def test_repr_uses_tag_over_uid():
    """Display name is the first tag (nt0-fp3), not the raw UID (ft_base_nt0fp3).

    The docs example uses the human-friendly tag. A raw UID in the repr would
    confuse developers who typed the tag in their code.
    """
    robot = _make_robot(_FULL_REGISTRY)
    assert "nt0-fp3" in repr(robot)
    assert "ft_base_nt0fp3" not in repr(robot)


def test_repr_explicit_model_arg():
    """Robot(model='nt0-fp3') resolves the same display string as Robot().

    model=None falls back to _DEFAULT_MODEL_UID which resolves to the same entry.
    """
    robot_default = _make_robot(_FULL_REGISTRY, model=None)
    robot_explicit = _make_robot(_FULL_REGISTRY, model="nt0-fp3")
    assert repr(robot_default) == repr(robot_explicit)


# ---------------------------------------------------------------------------
# AC 2 — degraded / partial construction never raises, makes zero network calls
# ---------------------------------------------------------------------------

def test_repr_empty_registry_degrades_gracefully():
    """NT_INFERENCE_URL override leaves _registry=[]. repr must not raise.

    A developer using the env-override affordance (CI/smoke testing) still
    gets a useful string, not a crash.
    """
    robot = _make_robot(registry=[], model=None)
    result = repr(robot)
    # Must not raise; must contain some model identifier, not 'contract received'
    assert "contract received" not in result
    assert isinstance(result, str) and result


def test_repr_contract_missing_axes_degrades():
    """Model entry exists but contract has no action_axes → 'contract pending'.

    Avoids crashing when the server returns a partial contract or the
    action_axes field hasn't been injected yet.
    """
    registry_no_axes = [
        {
            "uid": "ft_base_nt0fp3",
            "tags": ["nt0-fp3"],
            "endpoint": "wss://example/stream",
            "contract": {"action_shape": [50, 8]},  # axes missing
        }
    ]
    robot = _make_robot(registry_no_axes)
    result = repr(robot)
    assert "contract received" not in result
    assert "nt0-fp3" in result  # still shows the model name
    assert isinstance(result, str)


def test_repr_contract_missing_shape_degrades():
    """Contract has axes but no action_shape → 'contract pending'."""
    registry_no_shape = [
        {
            "uid": "ft_base_nt0fp3",
            "tags": ["nt0-fp3"],
            "endpoint": "wss://example/stream",
            "contract": {"action_axes": _NT0_AXES},  # shape missing
        }
    ]
    robot = _make_robot(registry_no_shape)
    result = repr(robot)
    assert "contract received" not in result
    assert isinstance(result, str)


def test_repr_no_registry_attr_does_not_raise():
    """Worst-case partial construction: _registry not set. repr catches Exception."""
    robot = object.__new__(newt.Robot)
    # Do not set _registry or _model at all.
    result = repr(robot)
    assert isinstance(result, str) and result  # something, not a raise


# ---------------------------------------------------------------------------
# AC 3 — __str__ and __repr__ are the same callable
# ---------------------------------------------------------------------------

def test_str_is_repr():
    """__str__ must be the same as __repr__ — print() and repr() give the same string."""
    robot = _make_robot(_FULL_REGISTRY)
    assert str(robot) == repr(robot)
    assert type(robot).__str__ is type(robot).__repr__
