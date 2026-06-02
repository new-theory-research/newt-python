"""newt.Robot — developer-facing client handle for New Theory inference.

Wire protocol: portal/wiki/specs/streaming-ws-protocol.md
"""
from __future__ import annotations

import functools
import os
import warnings
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

class NewTheoryError(Exception):
    """Base class for all New Theory server-emitted errors.

    Mirrors the canonical six-field error envelope (per
    portal/wiki/operating-docs/error-style.md). Every WS close-code error
    inherits from this class; branch on `exc.type` for specific failures.

    Attributes:
        code:     Numeric WS close code (or HTTP status for client-side errors).
        type:     Two-level dotted identifier (e.g. "auth.invalid_key").
        message:  Human-readable prose: what's wrong, expected vs. got, next action.
        context:  Machine-readable specifics (varies by error type).
        docs:     Stable URL into the docs site (may be None if not populated).
        trace_id: Server-generated cross-reference into server logs.
    """

    def __init__(
        self,
        code: int,
        type: str,
        message: str,
        context: dict,
        docs: str | None = None,
        trace_id: str = "",
    ) -> None:
        self.code = code
        self.type = type
        self.message = message
        self.context = context
        self.docs = docs
        self.trace_id = trace_id
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


class AuthError(NewTheoryError):
    """API key is missing, malformed, or rejected (WS close 4001 / HTTP 401).

    Group-by-domain: every `auth.*` type raises this class; branch on
    `exc.type` for specific failures (e.g. "auth.invalid_key").
    """


class DegradationWarning(UserWarning):
    """Server reports that one or more expected cameras were absent from the obs frame.

    The connection succeeds and actions are returned; missing cameras were zero-filled
    by the server. Actions may be degraded relative to the model's trained distribution.
    Emitted at most once per Robot.run() call via warnings.warn.
    """


class ProtocolError(NewTheoryError):
    """Obs frame could not be parsed or has an unrecognized type (WS close 4400).

    Group-by-domain: every `protocol.*` type raises this class; branch on
    `exc.type` for specific failures (`protocol.malformed_msgpack`,
    `protocol.missing_type`, `protocol.unknown_type`).
    """


class ServerError(NewTheoryError):
    """Server-side error during inference or in the WS handler (WS close 4500).

    Group-by-domain: every `server.*` type raises this class; branch on
    `exc.type` for specific failures (`server.inference_error`,
    `server.internal`).
    """


class VerifierError(NewTheoryError):
    """Console verifier infrastructure failure at handshake (WS close 4503).

    Group-by-domain: every `verifier.*` type raises this class; branch on
    `exc.type` for specific failures (e.g. `verifier.unavailable`).
    """


class ContractMismatchError(NewTheoryError):
    """Obs frame shapes don't match the resolved model's declared contract (WS close 4422).

    Mirrors the canonical six-field error envelope (per
    portal/wiki/operating-docs/error-style.md): every field on the envelope
    is a field on the exception. Group-by-domain — every `contract_mismatch.*`
    type raises this class; branch on `exc.type` for specific failures
    (`contract_mismatch.state_shape`, `contract_mismatch.camera_missing`, etc.).

    Attributes:
        code:     Numeric WS close code (4422).
        type:     Two-level dotted identifier (e.g. "contract_mismatch.state_shape").
        message:  Human-readable prose: what's wrong, expected vs. got, next action.
        context:  Machine-readable specifics (model, expected_shape, got_shape, etc.).
        docs:     Stable URL into the docs site (may be None if not populated).
        trace_id: Server-generated cross-reference into server logs.
    """


class ModelNotFoundError(NewTheoryError):
    """Requested model UID or tag not found in the registry (client-side, before WS connection).

    Mirrors the canonical six-field error envelope (per
    portal/wiki/operating-docs/error-style.md). Raised during Robot construction
    when discovery's response doesn't contain the requested model identifier.

    Attributes:
        code:     4404 (client-side; not a WS close code — fires before connection).
        type:     "model_not_found.unknown_identifier"
        message:  What was requested, what's available, how to fix.
        context:  Machine-readable: requested model, known UIDs and tags.
        docs:     Stable URL (may be None).
        trace_id: Empty string for client-side errors (no server trace to reference).
    """

    def __init__(
        self,
        model: str | None,
        known: list,
        docs: str | None = None,
    ) -> None:
        model_str = repr(model)
        message = (
            f"Model {model_str} not found in registry. "
            f"Known models: {known}. "
            "Check spelling or list with newt.list_models()."
        )
        super().__init__(
            code=4404,
            type="model_not_found.unknown_identifier",
            message=message,
            context={"requested": model, "known_models": known},
            docs=docs,
            trace_id="",
        )


class RegistryUnavailable(Exception):
    """Registry fetch failed before WS connection could be established.

    Mirrors the canonical six-field error envelope (per
    portal/wiki/operating-docs/error-style.md). Raised when GET /v1/models fails
    (network error, 5xx, or malformed JSON). Single attempt; no retry. Developer
    retries by re-instantiating Robot.

    Attributes:
        code:     503 (HTTP-convention; client-side before WS connection).
        type:     "registry.unavailable"
        message:  What was tried, why it failed, how to fix.
        context:  Machine-readable: bootstrap_url, reason string.
        docs:     Stable URL (may be None).
        trace_id: Empty string for client-side errors.
    """

    def __init__(
        self,
        bootstrap_url: str,
        reason: str,
        docs: str | None = None,
    ) -> None:
        self.code = 503
        self.type = "registry.unavailable"
        self.message = (
            f"Could not reach NT inference registry at {bootstrap_url}: "
            f"{reason}. Check NT_BOOTSTRAP_URL / NT_INFERENCE_URL, or retry."
        )
        self.context = {"bootstrap_url": bootstrap_url, "reason": reason}
        self.docs = docs
        self.trace_id = ""
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


@dataclass
class RunResult:
    """Returned by Robot.run() when stream=False."""
    stop_reason: str


# ---------------------------------------------------------------------------
# Bootstrap URL + registry discovery
# ---------------------------------------------------------------------------
# The SDK fetches GET <bootstrap_url>/v1/models once per Robot construction
# to discover which WS endpoint serves the requested model. Bootstrap URL
# resolution order (highest to lowest precedence):
#   1. NT_BOOTSTRAP_URL env var — explicit override
#   2. Derived from NT_INFERENCE_URL env var — strip WS scheme + path to HTTPS host
#   3. _DEFAULT_BOOTSTRAP_URL constant — production NT0-FP3 (registry holder today)
# NT_INFERENCE_URL takes full precedence at the Robot level: if set, discovery
# is skipped and the env URL is used directly (test/smoke affordance).

_DEFAULT_BOOTSTRAP_URL = "https://newtheory--ntdeva-nt0-fp3-serve-serve.modal.run"
_DEFAULT_MODEL_UID = "ft_base_nt0fp3"


def _resolve_bootstrap_url() -> str:
    """HTTPS base URL for registry discovery, per the resolution order above."""
    if url := os.environ.get("NT_BOOTSTRAP_URL"):
        return url
    if ws_url := os.environ.get("NT_INFERENCE_URL"):
        https = ws_url.replace("wss://", "https://", 1).replace("ws://", "http://", 1)
        return https.rsplit("/", 1)[0]
    return _DEFAULT_BOOTSTRAP_URL


def _fetch_registry(bootstrap_url: str, api_key: str) -> list:
    """GET <bootstrap_url>/v1/models. Single attempt; raises on failure.

    Raises:
        AuthError:          401 from the registry endpoint.
        RegistryUnavailable: Network error, 5xx, or non-JSON response.
    """
    import json
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    url = bootstrap_url.rstrip("/") + "/v1/models"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:
        if exc.code == 401:
            raise AuthError(
                code=4001,
                type="auth.invalid_key",
                message=(
                    "Authentication failed: API key rejected by registry /v1/models. "
                    "Rotate your key in the NT console."
                ),
                context={},
            ) from exc
        raise RegistryUnavailable(bootstrap_url, f"HTTP {exc.code}") from exc
    except URLError as exc:
        raise RegistryUnavailable(bootstrap_url, str(exc.reason)) from exc
    except Exception as exc:
        raise RegistryUnavailable(bootstrap_url, str(exc)) from exc


def _resolve_model_endpoint(registry: list, model: str | None, bootstrap_url: str) -> str:
    """Resolve a model UID or tag to its WS endpoint URL from the registry response.

    model=None resolves to _DEFAULT_MODEL_UID. Raises ModelNotFoundError when the
    identifier doesn't match any entry in the registry.
    """
    target = model if model is not None else _DEFAULT_MODEL_UID
    for entry in registry:
        uid = entry.get("uid") or ""
        tags = entry.get("tags") or []
        if target == uid or target in tags:
            endpoint = entry.get("endpoint")
            if endpoint:
                return endpoint
    known: list = []
    for entry in registry:
        uid = entry.get("uid")
        if uid:
            known.append(uid)
        for tag in (entry.get("tags") or []):
            known.append(tag)
    raise ModelNotFoundError(model, known)


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
        model:      Model identifier (UID or tag). The SDK resolves it via
                    endpoint discovery (GET /v1/models on construction) and also
                    forwards it in the first obs frame so the server resolves the
                    checkpoint. None (default) resolves to the default base model.
                    Power-user escape hatch — most developers should leave this unset.
        connect_timeout: Seconds to wait for the WS handshake. Defaults to
                    120s to tolerate Modal cold-start (scale-down → GPU +
                    checkpoint-load can take 30–90s).

    NT_INFERENCE_URL override: if set, endpoint discovery is skipped and this
                    URL is used directly. Takes highest precedence. Smoke and
                    golden tests use this to repoint at a specific server without
                    touching the registry.

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

        # Reset to False at the start of each run() call (see _run_blocking/_stream).
        self._degradation_warned: bool = False

        # Test-affordance: NT_INFERENCE_URL takes highest precedence — if set,
        # discovery is skipped and this URL is used directly. Smoke + golden
        # tests use this to repoint at specific servers without touching the registry.
        env_url = os.environ.get("NT_INFERENCE_URL")
        if env_url:
            self._url = env_url
            self._registry: list = []
        else:
            # Discovery: fetch the registry from the bootstrap URL and resolve the
            # requested model to its WS endpoint. Single network call per construction.
            bootstrap_url = _resolve_bootstrap_url()
            self._registry = _fetch_registry(bootstrap_url, api_key)
            self._url = _resolve_model_endpoint(self._registry, model, bootstrap_url)

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
            AuthError:             API key rejected by server (WS close 4001).
            ContractMismatchError: Obs frame shapes don't match the resolved
                                   model's declared contract (WS close 4422).
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
            http_code = getattr(status, "status_code", 0) if status else 0
            raise AuthError(
                code=4001,
                type="auth.invalid_key",
                message=(
                    f"Authentication failed during WS upgrade (HTTP {http_code}). "
                    "Check your API key and rotate it in the NT console if needed."
                ),
                context={},
            ) from exc

    def _run_blocking(self, prompt: str, max_duration: float) -> RunResult:
        import time as _time
        self._degradation_warned = False
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
                    _check_contract_mismatch_close(exc, self._model)
                    break  # connection closed (non-auth, non-contract-mismatch)

                parsed = _unpack(raw)
                _check_contract_mismatch_frame(parsed, self._model)
                ftype = _str_field(parsed, "type")

                if ftype == "action":
                    chunk = parsed.get("chunk")
                    if not self._degradation_warned:
                        self._degradation_warned = True
                        _maybe_warn_degradation(parsed, self._model)
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
        self._degradation_warned = False
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
                    _check_contract_mismatch_close(exc, self._model)
                    return

                parsed = _unpack(raw)
                _check_contract_mismatch_frame(parsed, self._model)
                ftype = _str_field(parsed, "type")

                if ftype == "action":
                    if not self._degradation_warned:
                        self._degradation_warned = True
                        _maybe_warn_degradation(parsed, self._model)
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
        base_url = _resolve_bootstrap_url()

    url = base_url.rstrip("/") + "/v1/models"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:
        if exc.code == 401:
            raise AuthError(
                code=4001,
                type="auth.invalid_key",
                message=(
                    "Authentication failed: API key rejected by /v1/models. "
                    "Rotate your key in the NT console."
                ),
                context={},
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
    """Raise AuthError if close code is 4001 (auth failure).

    Fallback for when the binary envelope frame was not received before the close.
    C6 adds a frame-level handler (_check_envelope_frame) that parses the full
    six-field envelope; this function fires for the bare-close case.
    """
    rcvd = getattr(exc, "rcvd", None)
    if rcvd and getattr(rcvd, "code", None) == 4001:
        raise AuthError(
            code=4001,
            type="auth.invalid_key",
            message=(
                "Authentication failed: API key is invalid or revoked. "
                "Rotate your key in the NT console."
            ),
            context={},
        ) from exc


def _decode_key(d: dict, key: str, default=None):
    """Read `key` from a msgpack-decoded dict, accepting both str and bytes keys."""
    if key in d:
        return d[key]
    bkey = key.encode("utf-8")
    if bkey in d:
        val = d[bkey]
        # Decode top-level string values from bytes for envelope fields.
        if isinstance(val, bytes):
            try:
                return val.decode("utf-8")
            except UnicodeDecodeError:
                return val
        return val
    return default


def _check_contract_mismatch_frame(parsed: dict, model: str | None = None) -> None:
    """Raise ContractMismatchError if the parsed frame is a 4422 close envelope.

    The server sends a msgpack binary message before the WS close frame
    carrying the canonical six-field error envelope (code, type, message,
    context, docs, trace_id — per portal/wiki/operating-docs/error-style.md).
    This function catches that message (keyed on code == 4422) and raises
    with every field on the envelope mapped onto the exception. Handles both
    str and bytes map keys from msgpack decoding.
    """
    code = _decode_key(parsed, "code")
    if code != 4422:
        return
    raise ContractMismatchError(
        code=code,
        type=_decode_key(parsed, "type", "contract_mismatch.unknown"),
        message=_decode_key(parsed, "message", "Contract mismatch."),
        context=_decode_key(parsed, "context", {}) or {},
        docs=_decode_key(parsed, "docs"),
        trace_id=_decode_key(parsed, "trace_id", "") or "",
    )


def _maybe_warn_degradation(parsed: dict, model: str | None) -> None:
    """Emit DegradationWarning if the first action frame carries missing_expected_cameras.

    Called once per run() on the first action frame. Does nothing when the
    warnings field is absent or empty (the happy path has no overhead).
    """
    warnings_field = _decode_key(parsed, "warnings")
    if not isinstance(warnings_field, dict):
        return
    missing = (
        warnings_field.get("missing_expected_cameras")
        or warnings_field.get(b"missing_expected_cameras")
    )
    if not missing:
        return
    missing_strs = [c.decode() if isinstance(c, bytes) else c for c in missing]
    model_str = model or "nt0-fp3"
    warnings.warn(
        DegradationWarning(
            f"Model {model_str!r} expected cameras not all present. "
            f"Missing: {missing_strs}. "
            "Missing cameras zero-filled; actions may be degraded."
        ),
        stacklevel=3,
    )


def _check_contract_mismatch_close(exc: ConnectionClosed, model: str | None = None) -> None:
    """Raise ContractMismatchError if close code is 4422.

    Fallback for the case where the binary envelope frame was not received
    (e.g. the connection dropped between send_bytes and close). Raises with
    empty context + a default message — the developer still gets a typed
    exception, not stop_reason="error".
    """
    rcvd = getattr(exc, "rcvd", None)
    if rcvd and getattr(rcvd, "code", None) == 4422:
        model_str = f" for model={model}" if model else ""
        raise ContractMismatchError(
            code=4422,
            type="contract_mismatch.unknown",
            message=(
                f"Contract mismatch{model_str}: obs frame shapes don't match "
                "the resolved model's declared contract. Server did not deliver "
                "an envelope; check server logs for details."
            ),
            context={"model": model} if model else {},
            docs=None,
            trace_id="",
        ) from exc
