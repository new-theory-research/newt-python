"""Offline unit tests for Robot.__repr__ / __str__ (brief-234 receipt).

Why these matter:
  The public docs' milestone-1 finish line shows an exact string:
      so101 · contract received · (30,6) · 6 labeled axes
  Before this brief, print(Robot()) emitted the default Python object repr —
  making the documented receipt a lie. These tests pin the string so the docs
  can't drift silently. They also pin the default: model=None must resolve
  through _DEFAULT_MODEL_UID to the so101 base entry.
"""
from __future__ import annotations

import newt
from newt._client import robot as _robot_mod
from newt._client.robot import _DEFAULT_MODEL_UID

_SO101_AXES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

_FULL_REGISTRY = [
    {
        "uid": _DEFAULT_MODEL_UID,
        "tags": ["so101"],
        "type": "fine_tune",
        "endpoint": "wss://example/stream",
        "contract": {
            "action_shape": [30, 6],
            "action_axes": _SO101_AXES,
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
    expected = "so101 · contract received · (30,6) · 6 labeled axes"
    assert repr(robot) == expected
    assert str(robot) == expected  # AC 3: __str__ == __repr__


def test_repr_uses_tag_over_uid():
    """Display name is the first tag (so101), not the raw UID.

    The docs example uses the human-friendly tag. A raw UID in the repr would
    confuse developers who typed the tag in their code.
    """
    robot = _make_robot(_FULL_REGISTRY)
    assert "so101" in repr(robot)
    assert _DEFAULT_MODEL_UID not in repr(robot)


def test_repr_explicit_model_arg():
    """Robot(model='so101') resolves the same display string as Robot().

    model=None falls back to _DEFAULT_MODEL_UID which resolves to the same entry.
    """
    robot_default = _make_robot(_FULL_REGISTRY, model=None)
    robot_explicit = _make_robot(_FULL_REGISTRY, model="so101")
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
            "uid": _DEFAULT_MODEL_UID,
            "tags": ["so101"],
            "endpoint": "wss://example/stream",
            "contract": {"action_shape": [30, 6]},  # axes missing
        }
    ]
    robot = _make_robot(registry_no_axes)
    result = repr(robot)
    assert "contract received" not in result
    assert "so101" in result  # still shows the model name
    assert isinstance(result, str)


def test_repr_contract_missing_shape_degrades():
    """Contract has axes but no action_shape → 'contract pending'."""
    registry_no_shape = [
        {
            "uid": _DEFAULT_MODEL_UID,
            "tags": ["so101"],
            "endpoint": "wss://example/stream",
            "contract": {"action_axes": _SO101_AXES},  # shape missing
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
# Default resolution — a BARE Robot() (model=None) must resolve through
# _DEFAULT_MODEL_UID to the so101 base entry via the real construction path.
# Network-free: the /v1/models fetch is monkeypatched, no WS is opened at
# construction. This is the receipt that the shipped default is so101, not a
# withdrawn model — if _DEFAULT_MODEL_UID stops matching the registry, a bare
# Robot() is broken for every developer, and this test catches it.
# ---------------------------------------------------------------------------

def test_bare_robot_resolves_default_to_so101(monkeypatch):
    """Robot() with no model resolves _DEFAULT_MODEL_UID → the so101 base entry."""
    captured = {}

    def fake_fetch(bootstrap_url, api_key):
        captured["called"] = True
        return _FULL_REGISTRY

    monkeypatch.delenv("NT_INFERENCE_URL", raising=False)
    monkeypatch.setattr(_robot_mod, "_fetch_registry", fake_fetch)

    robot = newt.Robot(api_key="nt_test")  # model=None → default

    assert captured.get("called") is True  # went through real discovery, no network
    assert robot._url == "wss://example/stream"  # resolved the so101 entry's endpoint
    assert _DEFAULT_MODEL_UID == "ft_6341c5_d13da9"  # the shipped default is the so101 base
    assert repr(robot) == "so101 · contract received · (30,6) · 6 labeled axes"


# ---------------------------------------------------------------------------
# AC 3 — __str__ and __repr__ are the same callable
# ---------------------------------------------------------------------------

def test_str_is_repr():
    """__str__ must be the same as __repr__ — print() and repr() give the same string."""
    robot = _make_robot(_FULL_REGISTRY)
    assert str(robot) == repr(robot)
    assert type(robot).__str__ is type(robot).__repr__
