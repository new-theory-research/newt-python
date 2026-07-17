"""Contract-aware snapshot selection for `newt run`.

`newt run <tag>` with no --snapshot must send the model a frame OF THE RIGHT SHAPE:
an 8-axis flagship model gets an 8-axis snapshot, a 6-axis SO-101 model gets a 6-axis one.
Sending a cup_stacking (8-axis) frame to an SO-101 model is the exact hard-stop this
selection fixes — before it, every SO-101 fine-tune run failed on shape.

These tests encode WHY selection matters:
- A contract with a declared shape must be MATCHED, not ignored (the SO-101 bug).
- A contract with nothing to match on (a flagship base carrying only action_shape) must
  fall back to the historical default, not error.
- A shape nothing bundled matches must fail HONESTLY, naming the available shapes —
  never coerce a wrong-shaped frame onto the wire (Rule 10).
- An explicit --snapshot still overrides, and a mismatched explicit choice still fails
  with console-011's client-side contract error.
"""
from __future__ import annotations

import io
import sys
from types import MappingProxyType

import numpy as np

import newt
import newt.snapshots as snapshots
from newt._client.robot import ContractMismatchError, InferenceResponse, ModelContract
from newt._cli.run import _select_snapshot, cmd_run


# --- Contract fixtures — the two real families, plus an unmatchable one ------

def _contract(state_shape=None, cameras=None, image_shape=None) -> ModelContract:
    """A ModelContract with just the fields selection reads. raw is required by the
    frozen dataclass; selection never touches it."""
    return ModelContract(
        state_shape=state_shape,
        cameras=cameras,
        image_shape=image_shape,
        action_shape=None,
        action_axes=None,
        raw=MappingProxyType({}),
    )


# flagship 8-axis contract, cameras exactly cup_stacking's — the positively-matched flagship case.
_FIXTURE_CONTRACT = _contract(
    state_shape=(8,),
    cameras=("right-wrist-camera", "surrounding1", "surrounding2"),
)
# The LIVE red-cube-bowl contract as the SDK actually sees it: state (6,), image
# (3,224,224), cameras=None. The real serve contract carries `cameras` as a
# {required, expected} DICT, which ModelContract._from_dict leaves unparsed (None) — so
# selection keys off state_shape alone, and it must land on red_cube. (This is exactly why
# selection can't hard-require a camera-set match: the flagship's own live contract
# doesn't expose one.)
_SO101_CONTRACT = _contract(state_shape=(6,), image_shape=(3, 224, 224))
# A hypothetical SO-101 contract that DOES parse cameras (a flat top/side list) — red_cube
# must match it on both state and cameras.
_SO101_CONTRACT_WITH_CAMERAS = _contract(
    state_shape=(6,), cameras=("top", "side"), image_shape=(3, 224, 224)
)
# flagship BASE with only action_shape declared — no state_shape / cameras to key off.
_NO_SIGNAL_CONTRACT = _contract()
# A shape nothing bundled matches (e.g. a 14-axis bimanual).
_UNMATCHABLE_CONTRACT = _contract(state_shape=(14,))


# --- _select_snapshot unit --------------------------------------------------

def test_select_no_signal_falls_back_to_default():
    """A contract declaring neither state_shape nor cameras (flagship base) → cup_stacking,
    the historical default — we don't guess a shape we can't see."""
    assert _select_snapshot(snapshots, _NO_SIGNAL_CONTRACT) == "cup_stacking"


def test_select_none_contract_falls_back_to_default():
    """A None contract (NT_INFERENCE_URL override, empty registry) → cup_stacking."""
    assert _select_snapshot(snapshots, None) == "cup_stacking"


def test_select_fixture_contract_matches_cup_stacking():
    """An 8-axis flagship contract is positively MATCHED to cup_stacking (state + cameras),
    and wins the 8-axis tie over pour_coffee_beans via registry order."""
    assert _select_snapshot(snapshots, _FIXTURE_CONTRACT) == "cup_stacking"


def test_select_so101_contract_matches_red_cube():
    """THE fix: the live 6-axis red-cube-bowl contract (cameras unparsed → None) selects
    red_cube, not the 8-axis cup_stacking that used to be sent to every model. This is the
    E2E hard-stop the whole arc closes."""
    assert _select_snapshot(snapshots, _SO101_CONTRACT) == "red_cube"


def test_select_so101_contract_with_parsed_cameras_matches_red_cube():
    """When an SO-101 contract DOES expose a flat top/side camera list, red_cube matches on
    both state and cameras — selection isn't fooled into cup_stacking by the camera check."""
    assert _select_snapshot(snapshots, _SO101_CONTRACT_WITH_CAMERAS) == "red_cube"


def test_select_unmatchable_contract_returns_none():
    """A 14-axis contract matches no bundled snapshot → None (caller errors honestly),
    never a coerced wrong-shaped frame."""
    assert _select_snapshot(snapshots, _UNMATCHABLE_CONTRACT) is None


# --- cmd_run integration — a fake Robot carrying a contract -----------------

_AXES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]


def _fake_response(model="fixture-base-pour"):
    chunk = np.zeros((50, 8), dtype=np.float32)
    return InferenceResponse(action_chunk=chunk, axes=list(_AXES), latency_ms=12.0, model=model)


class _FakeRobotWithContract:
    """Stand-in Robot exposing a .contract, recording the obs infer() received."""

    def __init__(self, *, contract, infer_return=None, infer_raises=None):
        self.contract = contract
        self._infer_return = infer_return
        self._infer_raises = infer_raises
        self.infer_obs = None

    def infer(self, obs, prompt=None):
        self.infer_obs = obs
        if self._infer_raises is not None:
            raise self._infer_raises
        return self._infer_return


def _run_with_contract(args, monkeypatch, *, contract, infer_return=None, infer_raises=None):
    out, err = io.StringIO(), io.StringIO()
    robot = _FakeRobotWithContract(
        contract=contract, infer_return=infer_return, infer_raises=infer_raises
    )
    monkeypatch.setattr(newt, "Robot", lambda *a, **kw: robot)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    code = cmd_run(args)
    return code, out.getvalue(), err.getvalue(), robot


def test_fixture_model_defaults_to_cup_stacking(monkeypatch):
    """A flagship model whose contract has no shape to match → cup_stacking rides the obs."""
    code, out, err, robot = _run_with_contract(
        ["fixture-base-pour"], monkeypatch,
        contract=_NO_SIGNAL_CONTRACT, infer_return=_fake_response(),
    )
    assert code == 0, f"stderr={err!r}"
    assert robot.infer_obs["prompt"] == "Stack one cup into another cup.", (
        "cup_stacking's canonical prompt must ride the obs for a flagship model"
    )
    assert "cup_stacking" in out


def test_so101_model_selects_red_cube(monkeypatch):
    """End-to-end: `newt run <so101-fine-tune>` with the live red-cube-bowl contract sends
    red_cube — 6-axis state, top/side cameras — and its real recorded prompt rides the obs.
    Before this arc it sent cup_stacking (8-axis) and hard-stopped on shape."""
    code, out, err, robot = _run_with_contract(
        ["ft-64cad948c4e9f09c-red-cube-bowl"], monkeypatch,
        contract=_SO101_CONTRACT, infer_return=_fake_response(),
    )
    assert code == 0, f"stderr={err!r}"
    assert robot.infer_obs["state"].shape == (6,), "a 6-axis state must reach the wire"
    assert set(robot.infer_obs["images"].keys()) == {"top", "side"}
    assert "red cube" in robot.infer_obs["prompt"].lower(), "the real recorded prompt rides"
    assert "red_cube" in out


def test_no_match_contract_errors_naming_available_shapes(monkeypatch):
    """A contract nothing bundled matches → honest error naming the contract shape AND
    every bundled snapshot's shape, exit 1, no wire call (Rule 10)."""
    code, out, err, robot = _run_with_contract(
        ["some-14-axis-model"], monkeypatch,
        contract=_UNMATCHABLE_CONTRACT, infer_return=_fake_response(),
    )
    assert code == 1
    assert err.startswith("newt: no bundled snapshot matches"), f"house shape: {err!r}"
    assert "(14,)" in err, "the unmatched contract shape must be named"
    # Every bundled snapshot's shape is listed so the developer sees why.
    for name in snapshots.available():
        assert name in err, f"{name} must be listed with its shape"
    assert "state_shape=(8,)" in err, "the bundled 8-axis shape must be named"
    assert robot.infer_obs is None, "no obs may reach the wire on a no-match (Rule 10)"


def test_explicit_snapshot_overrides_selection(monkeypatch):
    """--snapshot wins over contract-aware selection: an explicit pour_coffee_beans is
    sent even to a flagship model whose default would be cup_stacking."""
    code, out, err, robot = _run_with_contract(
        ["fixture-base-pour", "--snapshot", "pour_coffee_beans"], monkeypatch,
        contract=_NO_SIGNAL_CONTRACT, infer_return=_fake_response(),
    )
    assert code == 0, f"stderr={err!r}"
    assert "surrounding1" in robot.infer_obs["images"], "the explicit snapshot's frames ride"
    assert "pour_coffee_beans" in out


def test_explicit_snapshot_mismatch_still_fails_with_contract_error(monkeypatch):
    """An explicit --snapshot whose shape contradicts the model's contract still fails
    client-side with console-011's teaching error (the SDK's own pre-flight raises it on
    infer()). Selection never silences that — explicit means explicit."""
    exc = ContractMismatchError(
        code=4422,
        type="contract_mismatch.state_shape",
        message="Observation 'state' has shape (8,), but model 'so101-ft' expects state_shape (6,).",
        context={"model": "so101-ft", "expected_shape": [6], "got_shape": [8]},
    )
    code, out, err, robot = _run_with_contract(
        ["so101-ft", "--snapshot", "cup_stacking"], monkeypatch,
        contract=_SO101_CONTRACT, infer_raises=exc,
    )
    assert code == 1
    assert err.startswith("newt: contract mismatch"), f"house shape: {err!r}"
    assert "(6,)" in err and "(8,)" in err, "expected-vs-got must survive to the developer"
