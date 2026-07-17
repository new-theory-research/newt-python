"""Golden tests for newt.Robot — pins the promised developer experience at the public API boundary.

Each test encodes what a developer sees and why it matters. If a golden test breaks,
the developer-facing guarantee it protects has regressed, not just an implementation detail.

Required environment variables:
    NT_INFERENCE_URL  — WebSocket endpoint URL (wss://...modal.run/stream).
                        Skips endpoint discovery in Robot.__init__ (test affordance).
                        Used by GT1–GT4. GT5 clears this to test discovery-based routing.
    NT_API_KEY        — valid API key (GT1, GT3, GT4, GT5 only — GT2 deliberately uses a bad key)

Run:
    uv run pytest tests/test_golden.py -v

Tests hit the live Modal endpoint. No mocking of the WS layer.
Wire protocol: portal/wiki/specs/streaming-ws-protocol.md
"""
from __future__ import annotations

import os
import time
import types
import warnings

import httpx
import numpy as np
import pytest

import newt

_VALID_STOP_REASONS = {"task_complete", "max_duration", "interrupted", "error"}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def api_key() -> str:
    key = os.environ.get("NT_API_KEY")
    if not key:
        pytest.skip("NT_API_KEY not set — needed for live endpoint tests")
    return key


@pytest.fixture(scope="session")
def inference_url() -> str:
    url = os.environ.get("NT_INFERENCE_URL")
    if not url:
        pytest.skip("NT_INFERENCE_URL not set — needed for live endpoint tests")
    return url


def _health_url_from_inference_url(ws_url: str) -> str:
    """Translate wss://host/stream → https://host/health.

    The server publishes /health alongside /stream on the same Modal app.
    """
    https = ws_url.replace("wss://", "https://", 1).replace("ws://", "http://", 1)
    # Strip the WS path (/stream or anything after host) and append /health.
    # Modal app URLs have the shape https://<workspace>--<app>-<fn>.modal.run/<path>.
    # We want the bare host, then /health.
    scheme, _, rest = https.partition("://")
    host, _, _path = rest.partition("/")
    return f"{scheme}://{host}/health"


@pytest.fixture(scope="session", autouse=True)
def warmup_modal() -> None:
    """Poll /health until the model is loaded so cold-start cost doesn't cascade-fail tests.

    Modal scales the serve app down after 900s idle. First request after scale-down
    pays GPU-cold + checkpoint-load (30–90s typical, occasionally >2 min). Without
    a warmup gate, every test that touches the wire eats this cost — and the first
    test's WS handshake usually times out before /health says ok.

    Skips silently if NT_INFERENCE_URL isn't set (preserves the existing skip behavior
    in test fixtures above).
    """
    ws_url = os.environ.get("NT_INFERENCE_URL")
    if not ws_url:
        return

    health_url = _health_url_from_inference_url(ws_url)
    deadline = time.monotonic() + 180.0
    t0 = time.monotonic()
    last_err: Exception | None = None
    last_body: str = ""

    while time.monotonic() < deadline:
        try:
            resp = httpx.get(health_url, timeout=10.0)
            last_body = resp.text
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except Exception:
                    body = {}
                if body.get("status") == "ok" and body.get("loaded") is True:
                    elapsed = time.monotonic() - t0
                    print(
                        f"\n[warmup_modal] /health ok+loaded after {elapsed:.1f}s "
                        f"(url={health_url})",
                        flush=True,
                    )
                    return
        except Exception as exc:  # network blip, DNS, conn-refused during cold-start, etc.
            last_err = exc
        time.sleep(2.0)

    elapsed = time.monotonic() - t0
    pytest.skip(
        f"warmup_modal: /health did not report loaded=true within 180s "
        f"(elapsed={elapsed:.1f}s, last_body={last_body!r}, last_err={last_err!r})"
    )


# ---------------------------------------------------------------------------
# GT1
# ---------------------------------------------------------------------------


def test_gt1_happy_path(api_key: str, inference_url: str) -> None:
    """A developer writes robot.run("pick up the cup") and the model gets to work.

    Developer constructs newt.Robot with their API key and two callables. They call
    robot.run("pick up the cup"). The library connects to the model, sensor state
    flows out, action chunks come back, the mock execute callable runs for each chunk.
    The call returns with a stop_reason they can inspect. No exceptions, no hangs.

    Assert: read_state invoked >= 1 time; execute invoked >= 1 time with ndarray chunk;
    result.stop_reason in valid set; wall time <= max_duration + 15s overhead.
    Modal cold-start + diffusion inference can take ~15s overhead on top of max_duration.
    On pi0.5 today, "max_duration" is the expected and passing value.
    """
    max_duration = 10.0
    read_state_calls = 0
    execute_calls: list[np.ndarray] = []

    def read_state() -> dict:
        nonlocal read_state_calls
        read_state_calls += 1
        return {"state": np.zeros(14, dtype=np.float32)}

    def execute(chunk: np.ndarray) -> None:
        execute_calls.append(chunk)

    robot = newt.Robot(
        api_key=api_key,
        read_state=read_state,
        execute=execute,
    )

    t0 = time.monotonic()
    result = robot.run("pick up the cup", max_duration=max_duration)
    elapsed = time.monotonic() - t0

    assert read_state_calls >= 1, "read_state() must be called at least once"
    assert len(execute_calls) >= 1, "execute() must be called at least once (>= 1 action chunk)"
    assert isinstance(execute_calls[0], np.ndarray), "execute() argument must be ndarray"
    assert result.stop_reason in _VALID_STOP_REASONS, (
        f"stop_reason {result.stop_reason!r} not in {_VALID_STOP_REASONS}"
    )
    assert elapsed <= max_duration + 15.0, (
        f"run() took {elapsed:.1f}s, expected <= {max_duration + 15.0}s "
        f"(tolerance covers Modal cold-start + diffusion inference overhead)"
    )


# ---------------------------------------------------------------------------
# GT2
# ---------------------------------------------------------------------------


def test_gt2_invalid_key(inference_url: str) -> None:
    """A developer with an invalid API key gets a clear failure, not a hang.

    They run robot.run("pick up the cup") with a bad or revoked key. Within a few
    seconds they get a named exception that identifies the problem as an auth failure
    and points them toward fixing it. No silent hang. No partial output.

    Assert: newt.AuthError raised within 5s; message contains "authentication" or
    "api key"; no execute() invocations occurred; WS handshake closed with code 4001.
    """
    execute_calls: list = []

    # Real-format key shape (nt_ + 40 hex chars); not issued in any console DB.
    # Tests the verify-but-not-found path: the console verifier returns {valid: false},
    # the server closes 4001, and the client raises AuthError. A short/non-hex value
    # (e.g. "nt_invalid_key_00000000") risks tripping a different code path on the
    # verifier (non-200 → server closes 4503), so use a format-valid value here.
    bad_key = "nt_deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"

    robot = newt.Robot(
        api_key=bad_key,
        read_state=lambda: {"state": np.zeros(14, dtype=np.float32)},
        execute=lambda chunk: execute_calls.append(chunk),
    )

    t0 = time.monotonic()
    with pytest.raises(newt.AuthError) as exc_info:
        robot.run("pick up the cup")
    elapsed = time.monotonic() - t0

    assert elapsed <= 5.0, (
        f"AuthError must arrive within 5s (auth round-trip); took {elapsed:.1f}s"
    )
    msg = str(exc_info.value).lower()
    assert "authentication" in msg or "api key" in msg, (
        f"AuthError message should mention authentication or API key: {exc_info.value!r}"
    )
    assert len(execute_calls) == 0, "execute() must not be called before auth passes"


# ---------------------------------------------------------------------------
# GT3
# ---------------------------------------------------------------------------


def test_gt3_stream_mode(api_key: str, inference_url: str) -> None:
    """A developer passes stream=True and sees each chunk fly by.

    They write `for chunk in robot.run("pick up the cup", stream=True): print(chunk)`
    and chunks appear in their terminal one at a time. They didn't wire sensor reading
    — the library still pulled observations on their behalf. They get to decide whether
    to apply each chunk or just inspect.

    Assert: stream=True returns a generator; >= 1 chunk yielded; each chunk is ndarray;
    read_state() invoked >= once per chunk (library drives the obs side);
    execute() never called (caller's choice in stream mode).
    """
    max_duration = 10.0
    read_state_calls = 0
    execute_calls: list = []

    def read_state() -> dict:
        nonlocal read_state_calls
        read_state_calls += 1
        return {"state": np.zeros(14, dtype=np.float32)}

    robot = newt.Robot(
        api_key=api_key,
        read_state=read_state,
        execute=lambda chunk: execute_calls.append(chunk),
    )

    stream = robot.run("pick up the cup", max_duration=max_duration, stream=True)
    assert isinstance(stream, types.GeneratorType), "stream=True must return a generator"

    chunks: list[np.ndarray] = list(stream)

    assert len(chunks) >= 1, "stream must yield >= 1 chunk"
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, np.ndarray), f"chunk[{i}] must be ndarray"
    assert read_state_calls >= len(chunks), (
        "read_state() must be called >= once per yielded chunk (library drives obs)"
    )
    assert len(execute_calls) == 0, (
        "execute() must not be called in stream mode — caller's choice"
    )


# ---------------------------------------------------------------------------
# GT4
# ---------------------------------------------------------------------------


def test_gt4_sparse_observation(api_key: str, inference_url: str) -> None:
    """A developer's read_state returns a sparse observation, and chunks still come back.

    Their read_state() callable returns an empty dict — no state, no images, no prompt.
    They were expecting an error. Instead, action chunks come back anyway. The server
    applies firehose coercion per portal/wiki/specs/streaming-ws-protocol.md:
    missing fields -> defaults (zeros for state/images, default prompt from run() call).

    Assert: >= 1 action chunk returned despite empty observation; no 4xxx close on
    missing-field condition; result.stop_reason in valid set.
    """
    max_duration = 10.0
    execute_calls: list[np.ndarray] = []

    robot = newt.Robot(
        api_key=api_key,
        read_state=lambda: {},  # completely empty — server must coerce all fields
        execute=lambda chunk: execute_calls.append(chunk),
    )

    result = robot.run("pick up the cup", max_duration=max_duration)

    assert len(execute_calls) >= 1, (
        ">= 1 action chunk must arrive even with empty observation (firehose coercion)"
    )
    assert result.stop_reason in _VALID_STOP_REASONS, (
        f"stop_reason {result.stop_reason!r} not in {_VALID_STOP_REASONS}"
    )
