"""Offline unit tests for robot.contract + the local obs pre-flight (console-011).

Why these matter — each maps to a promise this brief makes:
  - robot.contract exposes the model's real declared shape the SDK already
    fetched (state_shape, cameras, image_shape, action_shape, action_axes) so a
    developer stops reaching into robot._registry. It must NEVER fabricate a
    contract the payload doesn't carry (Rule 10): a contract-less model resolves
    to None, not a shaped-right default.
  - The obs pre-flight fails a wrong-shaped observation INSTANTLY, client-side,
    BEFORE any WS connection — turning a ~50s cold-start round-trip into a
    microsecond keyboard-time error, with expected-vs-got named. A valid obs is
    untouched (byte-identical frame); a partial obs the server firehose-coerces
    is NOT failed.
  - The four riders: RegistryUnavailable catchable via except newt.NewTheoryError
    (MRO + construction both hold); run()'s no-callbacks error is a NewTheoryError;
    newt.fixtures resolves on attribute access with the DeprecationWarning on use,
    not on `import newt`; py.typed ships.

No live server: registries are monkeypatched onto Robot via __new__ (mirrors
test_robot_repr.py / test_inference_response.py).
"""
from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest

import newt
from newt._client.robot import (
    ContractMismatchError,
    ModelContract,
    NewTheoryError,
    RegistryUnavailable,
    _DEFAULT_MODEL_UID,
    _build_obs_frame,
    _pack,
    _resolve_contract,
    _validate_obs_against_contract,
)

# --- Grounding: shapes confirmed against real receipts ----------------------
# state_shape/cameras — SO101 starter run.py:322-324 parses these off _registry.
# image_shape [3,378,378], action_shape [30,6] — so101 serve contract, documented
# in newt-starter-so101/embodiment.py:107,119-124.
# action_shape/action_axes — parsed by the SDK itself (robot.py _resolve_action_axes).
_SO101_AXES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

_SO101_REGISTRY = [
    {
        "uid": "ft_so101",
        "tags": ["so101", "so-101"],
        "type": "fine_tune",
        "endpoint": "wss://example/stream",
        "contract": {
            "state_shape": [6],
            "cameras": ["top", "side"],
            "image_shape": [3, 378, 378],
            "action_shape": [30, 6],
            "action_axes": _SO101_AXES,
            # A field this view doesn't name — must survive on .raw, not be dropped.
            "future_field": "carried-through",
        },
    },
]


def _make_robot(registry, model="so101", read_state=None, execute=None):
    """Construct a Robot bypassing __init__ (no network); set only what's used."""
    robot = newt.Robot.__new__(newt.Robot)
    robot._registry = registry
    robot._model = model
    robot._api_key = "nt_test"
    robot._connect_timeout = 120.0
    robot._read_state = read_state
    robot._execute = execute
    robot._degradation_warned = False
    robot._cold_start_retry_consumed = False
    return robot


def _valid_obs():
    return {
        "state": np.zeros((6,), dtype=np.float32),
        "images": {
            "top": np.zeros((3, 378, 378), dtype=np.uint8),
            "side": np.zeros((3, 378, 378), dtype=np.uint8),
        },
    }


def _blow_up_on_connect(*_args, **_kwargs):
    """Stand-in _ws_connect that fails loudly if the pre-flight let a bad obs through.

    The whole feature is 'fail BEFORE the wire' — if control ever reaches here on a
    bad obs, the pre-flight didn't fire first and the test must fail.
    """
    raise AssertionError(
        "_ws_connect was reached — the pre-flight must reject a bad obs "
        "BEFORE any WS connection (that is the ~50s the feature saves)."
    )


# ---------------------------------------------------------------------------
# robot.contract accessor — the real payload shape, resolve-or-None
# ---------------------------------------------------------------------------

def test_contract_exposes_the_grounded_fields():
    """robot.contract returns the five confirmed fields, normalized to tuples."""
    robot = _make_robot(_SO101_REGISTRY, model="so101")
    c = robot.contract
    assert isinstance(c, ModelContract)
    assert c.state_shape == (6,)
    assert c.cameras == ("top", "side")
    assert c.image_shape == (3, 378, 378)
    assert c.action_shape == (30, 6)
    assert c.action_axes == tuple(_SO101_AXES)


def test_contract_raw_carries_unnamed_fields():
    """Any field the payload carries that this view doesn't name survives on .raw.

    Guarantees the accessor never has to invent or drop a field — forward-compat
    payload keys stay reachable.
    """
    robot = _make_robot(_SO101_REGISTRY, model="so101")
    assert robot.contract.raw["future_field"] == "carried-through"
    # .raw is read-only — a developer can't mutate the SDK's cached contract.
    with pytest.raises(TypeError):
        robot.contract.raw["future_field"] = "mutated"


def test_contract_resolves_by_tag_and_default_uid():
    """Resolution matches _resolve_action_axes: by tag, and model=None → default UID."""
    assert _resolve_contract(_SO101_REGISTRY, "so-101").state_shape == (6,)
    # Default UID path: model=None resolves through _DEFAULT_MODEL_UID to the
    # so101 base entry (the shipped default), not by guessing.
    default_registry = [{"uid": _DEFAULT_MODEL_UID, "tags": ["so101"],
                         "endpoint": "wss://x", "contract": {"state_shape": [6]}}]
    assert _resolve_contract(default_registry, None).state_shape == (6,)


def test_contract_is_none_when_no_contract_never_fabricated():
    """A model whose registry entry has no contract → None, NEVER a shaped-right default.

    This is the single most important honesty check (Rule 10): validating against
    an invented contract is worse than not validating.
    """
    no_contract = [{"uid": "ft_bare", "tags": ["bare"], "endpoint": "wss://x"}]
    robot = _make_robot(no_contract, model="bare")
    assert robot.contract is None
    assert _resolve_contract(no_contract, "bare") is None


def test_contract_is_none_for_env_override_empty_registry():
    """NT_INFERENCE_URL path leaves _registry=[] → contract is None, not a crash."""
    robot = _make_robot([], model=None)
    assert robot.contract is None
    assert _resolve_contract([], "so101") is None


def test_contract_is_none_for_unknown_model():
    """A model absent from the registry → None (never raises, never guesses)."""
    robot = _make_robot(_SO101_REGISTRY, model="does-not-exist")
    assert robot.contract is None


# ---------------------------------------------------------------------------
# Obs pre-flight — instant client-side fail, expected-vs-got, BEFORE the wire
# ---------------------------------------------------------------------------

def test_wrong_state_dim_fails_before_ws_connect():
    """A (8,) state to a model wanting (6,) fails instantly, before any WS connect."""
    robot = _make_robot(_SO101_REGISTRY, model="so101")
    robot._ws_connect = _blow_up_on_connect  # must NOT be reached
    bad = _valid_obs()
    bad["state"] = np.zeros((8,), dtype=np.float32)
    with pytest.raises(ContractMismatchError) as exc:
        robot.infer(bad)
    msg = str(exc.value)
    assert "(8,)" in msg and "(6,)" in msg  # got vs expected both named
    assert exc.value.type == "contract_mismatch.state_shape"
    assert exc.value.context["expected_shape"] == [6]
    assert exc.value.context["got_shape"] == [8]


def test_wrong_camera_key_fails_before_ws_connect():
    """A camera key the contract doesn't name (a typo) fails instantly, before the wire."""
    robot = _make_robot(_SO101_REGISTRY, model="so101")
    robot._ws_connect = _blow_up_on_connect
    bad = _valid_obs()
    bad["images"] = {"wrist": np.zeros((3, 378, 378), dtype=np.uint8)}
    with pytest.raises(ContractMismatchError) as exc:
        robot.infer(bad)
    msg = str(exc.value)
    assert "wrist" in msg and "top" in msg and "side" in msg
    assert exc.value.type == "contract_mismatch.camera_unknown"


def test_wrong_image_shape_fails_before_ws_connect():
    """A (3,240,320) image to a model wanting (3,378,378) fails instantly, before the wire."""
    robot = _make_robot(_SO101_REGISTRY, model="so101")
    robot._ws_connect = _blow_up_on_connect
    bad = _valid_obs()
    bad["images"]["top"] = np.zeros((3, 240, 320), dtype=np.uint8)
    with pytest.raises(ContractMismatchError) as exc:
        robot.infer(bad)
    msg = str(exc.value)
    assert "(3, 240, 320)" in msg and "(3, 378, 378)" in msg
    assert exc.value.type == "contract_mismatch.image_shape"


def test_run_preflights_first_obs_before_ws_connect():
    """run() validates the first obs BEFORE opening the WS — no ~50s round-trip.

    read_state hands back a bad obs; _ws_connect blows up if reached. A correct
    pre-flight raises ContractMismatchError first, so the blow-up never fires.
    """
    bad = _valid_obs()
    bad["state"] = np.zeros((8,), dtype=np.float32)
    robot = _make_robot(
        _SO101_REGISTRY, model="so101",
        read_state=lambda: bad, execute=lambda chunk: None,
    )
    robot._ws_connect = _blow_up_on_connect
    with pytest.raises(ContractMismatchError):
        robot.run("pick up the cube")


# ---------------------------------------------------------------------------
# Pure win: a valid obs is untouched; a partial obs is not failed
# ---------------------------------------------------------------------------

def test_valid_obs_passes_and_frame_is_byte_identical():
    """A correct obs passes the pre-flight AND its wire frame is byte-for-byte unchanged.

    The pure-win guarantee: the pre-flight must not mutate, reorder, or re-encode.
    """
    robot = _make_robot(_SO101_REGISTRY, model="so101")
    obs = _valid_obs()
    state_ref = obs["state"]
    before = _pack(_build_obs_frame(obs, "prompt", 30.0, "so101"))
    robot._validate_obs(obs)  # must not raise, must not touch obs
    after = _pack(_build_obs_frame(obs, "prompt", 30.0, "so101"))
    assert before == after
    assert obs["state"] is state_ref  # same object — no copy, no coercion
    assert set(obs["images"]) == {"top", "side"}


def test_partial_obs_missing_camera_is_not_failed():
    """A partial obs (one of two cameras) passes — the server zero-fills the rest.

    Demanding a field the server firehose-coerces would be a regression, not a
    validation.
    """
    robot = _make_robot(_SO101_REGISTRY, model="so101")
    partial = {"state": np.zeros((6,), dtype=np.float32),
               "images": {"top": np.zeros((3, 378, 378), dtype=np.uint8)}}
    robot._validate_obs(partial)  # no raise


def test_partial_obs_absent_state_and_empty_obs_are_not_failed():
    """Absent state, and even {}, pass — only PROVIDED fields are checked."""
    robot = _make_robot(_SO101_REGISTRY, model="so101")
    robot._validate_obs({"images": {"top": np.zeros((3, 378, 378), dtype=np.uint8)}})
    robot._validate_obs({})  # fully partial — server fills everything


def test_no_contract_makes_preflight_a_noop():
    """No contract available → the pre-flight can't check anything and never fails.

    A missing contract means 'I can't check this here', not 'this is invalid'.
    """
    # Wrong-looking obs, but contract is None → no-op (not a fabricated check).
    weird = {"state": np.zeros((99,), dtype=np.float32),
             "images": {"nonsense": np.zeros((1, 2, 3), dtype=np.uint8)}}
    _validate_obs_against_contract(weird, None, "so101")  # no raise
    robot = _make_robot([], model=None)  # env-override empty registry
    robot._validate_obs(weird)  # no raise


# ---------------------------------------------------------------------------
# Exception hygiene: RegistryUnavailable under NewTheoryError
# ---------------------------------------------------------------------------

def test_registry_unavailable_is_newtheory_error():
    """except newt.NewTheoryError catches the most-likely first-call failure."""
    assert issubclass(RegistryUnavailable, NewTheoryError)
    assert issubclass(newt.RegistryUnavailable, newt.NewTheoryError)


def test_registry_unavailable_constructs_with_its_own_signature():
    """Reparenting must not break RegistryUnavailable(bootstrap_url, reason) construction.

    The custom (bootstrap_url, reason, docs) signature builds the six-field
    envelope — MRO passing but construction broken would be a silent regression.
    """
    exc = RegistryUnavailable("http://x", "connection refused")
    assert isinstance(exc, NewTheoryError)
    assert exc.code == 503
    assert exc.type == "registry.unavailable"
    assert "http://x" in exc.message and "connection refused" in exc.message
    assert exc.context == {"bootstrap_url": "http://x", "reason": "connection refused"}
    assert exc.trace_id == ""
    assert str(exc) == exc.message
    # docs defaults to None; passes through when given.
    assert RegistryUnavailable("http://x", "why", docs="http://d").docs == "http://d"


def test_run_no_callbacks_raises_newtheory_error():
    """run() without read_state/execute raises a NewTheoryError, not a bare TypeError.

    Consistent with EmbodimentError (the sibling misconfig error) so a developer
    catching newt.NewTheoryError catches this too.
    """
    robot = _make_robot(_SO101_REGISTRY, model="so101", read_state=None, execute=None)
    with pytest.raises(NewTheoryError) as exc:
        robot.run("do a thing")
    assert "infer(" in str(exc.value)  # message still points at the one-shot path
    assert exc.value.type == "embodiment.missing_callbacks"


# ---------------------------------------------------------------------------
# newt.fixtures resolution (PEP 562): warns on USE, not on `import newt`
# ---------------------------------------------------------------------------

def test_newt_fixtures_resolves_on_attribute_access_with_warning():
    """newt.fixtures resolves via attribute access and warns on use."""
    # A prior `from newt import fixtures` anywhere in the session binds
    # newt.fixtures as a real attribute, which would bypass __getattr__. Clear
    # both the sys.modules entry AND the bound attribute so the PEP 562 path
    # (and the on-use warning) actually exercises.
    sys.modules.pop("newt.fixtures", None)
    if "fixtures" in vars(newt):
        del newt.fixtures
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fixtures = newt.fixtures  # PEP 562 __getattr__
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert "newt.snapshots" in str(caught[0].message)
    assert callable(fixtures.available) and callable(fixtures.load)


def test_import_newt_does_not_warn():
    """`import newt` must NOT emit the fixtures DeprecationWarning (fires on use only)."""
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import newt"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# py.typed marker ships with the package
# ---------------------------------------------------------------------------

def test_py_typed_marker_present_in_package():
    """PEP 561 marker is in the installed package (wheel-ship verified in closeout)."""
    import pathlib
    marker = pathlib.Path(newt.__file__).parent / "py.typed"
    assert marker.exists()
