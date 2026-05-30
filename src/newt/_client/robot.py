"""newt.Robot — developer-facing client handle for New Theory inference.

Wire protocol: portal/wiki/specs/streaming-ws-protocol.md
"""
from __future__ import annotations

import functools
import os
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable

import msgpack
import numpy as np
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect


# ---------------------------------------------------------------------------
# msgpack-numpy codec — wire-compatible with serve_openpi.py
# Source: imitation_learning/src/infra/inference/msgpack_numpy.py
# ---------------------------------------------------------------------------

def _pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_pack = functools.partial(msgpack.packb, default=_pack_array)
_unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class AuthError(Exception):
    """API key is missing, malformed, or rejected by the server (WS close 4001)."""


@dataclass
class RunResult:
    """Returned by Robot.run() when stream=False."""
    stop_reason: str


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Maps model name → WS endpoint URL for multi-server routing (one server per
# model family). Fine-tune variants (UIDs or tags like "nt0-fp3-pour-coffee-beans")
# go to the SAME server as their base model; their identifier is forwarded in
# the first obs frame's `model` field and resolved server-side.
# Default (model=None): connects to the "nt0-fp3" server; server uses its own
# default checkpoint when no model field is present in the obs frame.

_MODEL_ENDPOINTS: dict[str, str] = {
    "pi05-aloha": "wss://newtheory--ntdeva-openpi-serve-serve.modal.run/stream",
    "nt0-fp3":    "wss://newtheory--ntdeva-nt0-fp3-serve-serve.modal.run/stream",
}


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

class Robot:
    """Client handle for a New Theory inference endpoint.

    Args:
        api_key:    NT API key (nt_xxx); sent as Bearer in WS handshake.
        read_state: callable returning an observation dict. Optional keys:
                    "state" (float32 ndarray (14,)), "images" (dict of camera
                    arrays), "prompt" (str). Missing fields are firehose-coerced
                    by the server — partial dicts are fine.
        execute:    callable receiving an action chunk ndarray (action_horizon, 14).
                    Called once per inference cycle in default (non-stream) mode.
                    Never called in stream mode.
        model:      Model identifier (UID or tag); forwarded as-is in the first
                    obs frame. The server resolves it to a checkpoint. None
                    (default) omits the field and lets the server use its own
                    default checkpoint. Power-user escape hatch — most
                    developers should leave this unset.
        connect_timeout: Seconds to wait for the WS handshake. Defaults to
                    120s to tolerate Modal cold-start (scale-down → GPU +
                    checkpoint-load can take 30–90s).

    Default usage:
        robot = newt.Robot(
            api_key=os.environ["NT_API_KEY"],
            read_state=lambda: {"state": arm.get_joints()},
            execute=lambda chunk: arm.move_to(chunk[0]),
        )
        result = robot.run("pick up the cup", max_duration=30)
        # result.stop_reason == "max_duration" on pi0.5 today

    Stream usage (caller applies chunks, library drives obs):
        for chunk in robot.run("pick up the cup", stream=True):
            arm.move_to(chunk[0])
    """

    def __init__(
        self,
        api_key: str,
        read_state: Callable[[], dict],
        execute: Callable[[np.ndarray], None],
        model: str | None = None,
        connect_timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._read_state = read_state
        self._execute = execute
        self._connect_timeout = connect_timeout
        # Forwarded as-is in the first obs frame; None = server uses its default.
        # SDK never interprets UID vs tag — the server resolves the identifier.
        self._model = model

        # Test-affordance: NT_INFERENCE_URL overrides registry lookup so
        # golden tests + smoke can be repointed without touching the registry.
        env_url = os.environ.get("NT_INFERENCE_URL")
        if env_url:
            self._url = env_url
        else:
            # Fine-tune variants (UIDs/tags) are not in _MODEL_ENDPOINTS;
            # they connect to the same server as their base via NT_INFERENCE_URL.
            # Without NT_INFERENCE_URL, only top-level model family names work.
            _endpoint_key = model if model is not None else "nt0-fp3"
            if _endpoint_key not in _MODEL_ENDPOINTS:
                raise ValueError(
                    f"Unknown model {model!r}; known: {list(_MODEL_ENDPOINTS)}. "
                    "Set NT_INFERENCE_URL to connect to a custom endpoint."
                )
            self._url = _MODEL_ENDPOINTS[_endpoint_key]

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        max_duration: float = 30.0,
        stream: bool = False,
    ) -> RunResult | Generator[np.ndarray, None, None]:
        """Run inference.

        Args:
            prompt:       Language instruction for the model.
            max_duration: Wall-clock time limit in seconds.
            stream:       If True, return a generator yielding action chunks.
                          Library still calls read_state() per chunk; execute()
                          is never called. If False (default), drive the loop
                          internally and return RunResult.

        Raises:
            AuthError: API key rejected by server (WS close 4001).
        """
        if stream:
            return self._stream(prompt, max_duration)
        return self._run_blocking(prompt, max_duration)

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _ws_connect(self):
        """Open authenticated WS connection. Raises AuthError on HTTP-level rejection.

        open_timeout defaults to 120s (set on Robot via `connect_timeout=`) so
        cold-start cost (Modal scale-down → GPU + checkpoint-load, 30–90s typical)
        doesn't surface as a handshake timeout.
        """
        try:
            return connect(
                self._url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                open_timeout=self._connect_timeout,
            )
        except InvalidStatus as exc:
            status = getattr(exc, "response", None)
            code = getattr(status, "status_code", 0) if status else 0
            raise AuthError(
                f"Authentication failed during WS upgrade (HTTP {code}). "
                "Check your API key and rotate it in the NT console if needed."
            ) from exc

    def _run_blocking(self, prompt: str, max_duration: float) -> RunResult:
        import time as _time
        ws = self._ws_connect()
        stop_reason = "error"
        try:
            first = True
            frame_no = 0
            while True:
                obs = self._read_state()
                frame = _build_obs_frame(
                    obs, prompt,
                    max_duration if first else None,
                    self._model if first else None,
                )
                first = False
                frame_no += 1

                payload = _pack(frame)
                print(
                    f"[newt debug] frame {frame_no}: sending {len(payload)} bytes",
                    flush=True,
                )
                _send_t0 = _time.time()
                try:
                    ws.send(payload)
                    print(
                        f"[newt debug] frame {frame_no}: send returned in "
                        f"{(_time.time()-_send_t0)*1000:.1f}ms",
                        flush=True,
                    )
                except ConnectionClosed:
                    print(
                        f"[newt debug] frame {frame_no}: send raised "
                        f"ConnectionClosed after {(_time.time()-_send_t0)*1000:.1f}ms",
                        flush=True,
                    )
                    pass  # server may have initiated close; drain recv for terminal

                try:
                    raw = ws.recv()
                except ConnectionClosed as exc:
                    rcvd = getattr(exc, "rcvd", None)
                    print(
                        f"[newt] WS closed by server: "
                        f"code={getattr(rcvd, 'code', None)} "
                        f"reason={getattr(rcvd, 'reason', None)!r}",
                        flush=True,
                    )
                    _check_auth_error(exc)
                    break  # connection closed (non-auth)

                parsed = _unpack(raw)
                ftype = _str_field(parsed, "type")

                if ftype == "action":
                    chunk = parsed.get("chunk")
                    self._execute(chunk)
                elif ftype == "terminal":
                    stop_reason = _str_field(parsed, "stop_reason") or "error"
                    break
        finally:
            try:
                ws.close()
            except Exception:
                pass
        return RunResult(stop_reason=stop_reason)

    def _stream(
        self, prompt: str, max_duration: float
    ) -> Generator[np.ndarray, None, None]:
        ws = self._ws_connect()
        try:
            first = True
            while True:
                obs = self._read_state()
                frame = _build_obs_frame(
                    obs, prompt,
                    max_duration if first else None,
                    self._model if first else None,
                )
                first = False

                try:
                    ws.send(_pack(frame))
                except ConnectionClosed:
                    return  # server closed; no more chunks

                try:
                    raw = ws.recv()
                except ConnectionClosed as exc:
                    _check_auth_error(exc)
                    return

                parsed = _unpack(raw)
                ftype = _str_field(parsed, "type")

                if ftype == "action":
                    yield parsed.get("chunk")
                elif ftype == "terminal":
                    return
        finally:
            try:
                ws.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_models(api_key: str, base_url: str | None = None) -> list[dict]:
    """Fetch the list of available models from the NT inference server.

    Args:
        api_key:  NT API key (nt_xxx).
        base_url: Override HTTP base URL (e.g. http://localhost:8000). Defaults to
                  deriving from NT_INFERENCE_URL env var or the default nt0-fp3 server.

    Returns:
        List of model dicts with uid, tags, type, and base fields.

    Raises:
        AuthError: API key rejected by the server.
    """
    import json
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    if base_url is None:
        env_url = os.environ.get("NT_INFERENCE_URL")
        ws_url = env_url or _MODEL_ENDPOINTS["nt0-fp3"]
        # wss://host/path → https://host  (strip protocol scheme + trailing path)
        base_url = ws_url.replace("wss://", "https://").replace("ws://", "http://").rsplit("/", 1)[0]

    url = base_url.rstrip("/") + "/v1/models"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:
        if exc.code == 401:
            raise AuthError(
                "Authentication failed: API key rejected by /v1/models. "
                "Rotate your key in the NT console."
            ) from exc
        raise


def _build_obs_frame(
    obs: dict, prompt: str, max_duration: float | None, model: str | None = None
) -> dict:
    frame = {k: v for k, v in obs.items()}
    frame["type"] = "obs"
    if not frame.get("prompt"):
        frame["prompt"] = prompt
    if max_duration is not None:
        frame["max_duration"] = max_duration
    if model is not None:
        frame["model"] = model
    return frame


def _str_field(frame: dict, key: str) -> str:
    val = frame.get(key)
    if isinstance(val, bytes):
        return val.decode()
    return val or ""


def _check_auth_error(exc: ConnectionClosed) -> None:
    """Raise AuthError if close code is 4001 (auth failure)."""
    rcvd = getattr(exc, "rcvd", None)
    if rcvd and getattr(rcvd, "code", None) == 4001:
        raise AuthError(
            "Authentication failed: API key is invalid or revoked. "
            "Rotate your key in the NT console."
        ) from exc
