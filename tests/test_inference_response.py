"""Offline unit tests for Robot.infer()'s response surface (brief-228).

These don't touch the wire — they pin the developer-visible guarantees of
InferenceResponse and the registry label resolution that infer() depends on.
The live request/response path is covered by scripts/smoke_infer.py (needs creds).

Why these matter:
  - The "evaluate the API" audience reads the repr to learn what came back. If the
    repr drops labels, loses the shape, or guesses dim names when real ones exist,
    that audience is misled. Each assert below maps to one such regression.
  - The default-UID path: model=None must resolve through _DEFAULT_MODEL_UID to
    the so101 base entry's labels, not by guessing.
"""
from __future__ import annotations

import numpy as np

import newt
from newt._client.robot import (
    InferenceResponse,
    _DEFAULT_MODEL_UID,
    _resolve_action_axes,
)

_SO101_AXES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# Shape of the /v1/models JSON the SDK caches in self._registry: str keys, fine-tunes
# already carry the resolved (inherited) contract because the server injects it.
_FAKE_REGISTRY = [
    {
        "uid": _DEFAULT_MODEL_UID,
        "tags": ["so101"],
        "type": "base",
        "base": None,
        "endpoint": "wss://example/stream",
        "contract": {"action_shape": [30, 6], "action_axes": _SO101_AXES},
    },
    {
        "uid": "ft_4hn40z6a",
        "tags": ["clean_table"],
        "type": "fine_tune",
        "base": _DEFAULT_MODEL_UID,
        "contract": {"action_shape": [30, 6], "action_axes": _SO101_AXES},
    },
    {  # deployable model whose contract omits labels — must fall back, not crash
        "uid": "ft_nolabels",
        "tags": ["mystery"],
        "type": "base",
        "base": None,
        "endpoint": "wss://example/stream",
        "contract": {"action_shape": [30, 6]},
    },
]


def test_inference_response_exported():
    """import newt; newt.InferenceResponse must be reachable for the eval audience."""
    assert hasattr(newt, "InferenceResponse")
    assert newt.InferenceResponse is InferenceResponse


def test_repr_shows_shape_labels_and_latency():
    """The repr is the product: shape + semantic labels + latency on one line.

    This is the exact string the brief promises a developer evaluating the API sees.
    """
    resp = InferenceResponse(
        np.zeros((30, 6), dtype=np.float32), _SO101_AXES, 261.0, model="so101"
    )
    assert repr(resp) == (
        "action_chunk (30, 6): shoulder_pan, shoulder_lift, elbow_flex, "
        "wrist_flex, wrist_roll, gripper | latency 261ms"
    )
    assert str(resp) == repr(resp)


def test_action_chunk_is_canonical_no_regression():
    """.action_chunk returns the exact raw ndarray run()'s execute() would receive."""
    chunk = np.arange(12, dtype=np.float32).reshape(2, 6)
    resp = InferenceResponse(chunk, _SO101_AXES, 12.0)
    assert resp.action_chunk is chunk
    assert resp.latency_ms == 12.0
    assert resp.model is None


def test_registry_resolves_labels_for_base_tag_and_finetune():
    """Labels resolve by default-UID, by tag, and for inherited fine-tune contracts."""
    assert _resolve_action_axes(_FAKE_REGISTRY, None) == _SO101_AXES  # default UID
    assert _resolve_action_axes(_FAKE_REGISTRY, "so101") == _SO101_AXES  # base tag
    assert _resolve_action_axes(_FAKE_REGISTRY, "clean_table") == _SO101_AXES  # ft tag


def test_registry_returns_none_when_no_labels():
    """No labels -> None (caller falls back to dim_N), never a wrong guess or a crash."""
    assert _resolve_action_axes(_FAKE_REGISTRY, "nope") is None  # unknown model
    assert _resolve_action_axes([], "so101") is None  # empty registry (env override)
    assert _resolve_action_axes(_FAKE_REGISTRY, "mystery") is None  # contract w/o axes


def test_dim_n_fallback_is_index_named_not_mislabeled():
    """When the registry has no labels, fall back to dim_0..dim_N sized to the chunk.

    The fallback must never borrow another model's labels — index names make it
    obvious to the developer that semantic labels weren't available.
    """
    robot = newt.Robot.__new__(newt.Robot)  # skip __init__/network
    robot._registry = []
    robot._model = "mystery"
    axes = robot._action_axes_for(np.zeros((30, 6)))
    assert axes == [f"dim_{i}" for i in range(6)]
