"""newt.Robot — developer-facing client handle for New Theory inference.

Wire protocol: portal/wiki/specs/streaming-ws-protocol.md
"""
from __future__ import annotations

import functools
import os
import time
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

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


class ColdStartRetry(UserWarning):
    """First WS connection timed out; SDK retried with an extended timeout (180s).

    Modal containers serving large NT0-FP3 checkpoints (~5GB) take 60–90s to warm
    up from cold. On the FIRST connection attempt of a Robot instance's lifecycle,
    a TimeoutError triggers one automatic retry with connect_timeout=180. Emitted
    exactly once per Robot instance via warnings.warn. Subsequent connections (warm
    container) don't retry.
    """


class EnvOverrideWarning(UserWarning):
    """NT_INFERENCE_URL is set, bypassing dynamic /v1/models discovery for ALL models.

    Emitted at most once per Robot instance at construction time. This override is
    intentional for smoke and golden tests; in normal usage it means per-model
    cross-app routing is disabled and every Robot() will hit the same endpoint
    regardless of which model was requested.
    """


class VerifierTransientRetry(UserWarning):
    """Key verifier temporarily unavailable; SDK retrying automatically.

    The NT key-verification service occasionally takes a few seconds to become
    available after a cold start. When this happens the server closes the WS
    connection with close code 4503 before the first obs frame is sent. The SDK
    retries with bounded backoff (≤45s total) so the first documented call works
    without a hand-written retry loop. Emitted on the first retry via
    warnings.warn; subsequent retries are silent. A definitively-invalid key
    (AuthError) is never retried.
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

    def __str__(self) -> str:
        lines = [
            f"ServerError(code={self.code}, type={self.type}, trace_id={self.trace_id})"
        ]
        for k, v in (self.context or {}).items():
            lines.append(f"  context.{k}={v}")
        lines.append(f"  {self.message}  (Check type and context before retrying.)")
        return "\n".join(lines)


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


class BaseNotDeployableError(NewTheoryError):
    """Requested model is a base (lineage anchor) — not directly deployable (client-side, before WS connection).

    Raised when the requested model identifier resolves to a registry entry that
    has no endpoint field — the entry is a lineage anchor, not a deployable fine-tune.
    SDK raises before opening the WS connection. context.fine_tunes carries the
    dynamically-resolved deployable fine-tunes for this family.

    Attributes:
        code:     4424 (client-side; not a WS close code — fires before connection).
        type:     "model.base_not_deployable"
        message:  Human-readable: what was requested, which fine-tunes are available.
        context:  {"model": model_id, "fine_tunes": [list of deployable tags]}.
        docs:     Stable URL (may be None).
        trace_id: Empty string for client-side errors (no server trace).
    """


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


class InferenceResponse:
    """Returned by Robot.infer() — one labeled action chunk from a single request.

    The streaming Robot.run()/execute loop delivers bare ndarray chunks. infer()
    is the single-request primitive for the "evaluate the API" audience: it wraps
    the same raw chunk with the semantic axis labels and latency so a developer can
    see what they got back without consulting the wire spec.

    Attributes:
        action_chunk: The raw action chunk ndarray, shape (action_horizon, action_dim).
                      Canonical accessor — identical to what execute() receives in run().
        axes:         Semantic name per action dim (len == action_dim), e.g.
                      ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]. Falls back to
                      ["dim_0", ..., "dim_N"] when the registry carries no labels for
                      the resolved model.
        latency_ms:   Wall-clock round-trip for the single request, in milliseconds.
        model:        Resolved model identifier (UID or tag), or None when not known
                      (e.g. NT_INFERENCE_URL override skips registry discovery).
    """

    def __init__(
        self,
        action_chunk: np.ndarray,
        axes: list[str],
        latency_ms: float,
        model: str | None = None,
    ) -> None:
        self.action_chunk = action_chunk
        self.axes = axes
        self.latency_ms = latency_ms
        self.model = model

    def __repr__(self) -> str:
        shape = getattr(self.action_chunk, "shape", ())
        shape_str = ", ".join(str(d) for d in shape)
        axes_str = ", ".join(self.axes)
        return (
            f"action_chunk ({shape_str}): {axes_str} | "
            f"latency {self.latency_ms:.0f}ms"
        )

    __str__ = __repr__


# ---------------------------------------------------------------------------
# Bootstrap URL + registry discovery
# ---------------------------------------------------------------------------
# The SDK fetches GET <bootstrap_url>/v1/models once per Robot construction
# to discover which WS endpoint serves the requested model. Bootstrap URL
# resolution order (highest to lowest precedence):
#   1. NT_BOOTSTRAP_URL env var — explicit override
#   2. Derived from NT_INFERENCE_URL env var — strip WS scheme + path to HTTPS host
#   3. _DEFAULT_BOOTSTRAP_URL constant — always-on Railway registry (no cold start)
# NT_INFERENCE_URL takes full precedence at the Robot level: if set, discovery
# is skipped and the env URL is used directly (test/smoke affordance).

_DEFAULT_BOOTSTRAP_URL = "https://nt-registry-production.up.railway.app"
_DEFAULT_MODEL_UID = "ft_base_nt0fp3"


def _resolve_bootstrap_url() -> str:
    """HTTPS base URL for registry discovery, per the resolution order above."""
    if url := os.environ.get("NT_BOOTSTRAP_URL"):
        return url
    if ws_url := os.environ.get("NT_INFERENCE_URL"):
        https = ws_url.replace("wss://", "https://", 1).replace("ws://", "http://", 1)
        return https.rsplit("/", 1)[0]
    return _DEFAULT_BOOTSTRAP_URL


def _key_format_hint(api_key: str) -> str:
    """One-line nudge when a key is the wrong format (brief-229).

    The `ak_…` key from Clerk's console-profile panel is the landmine that cost a
    beta tester ~30 min: it's not an NT inference key. Name it specifically when the
    presented key doesn't start with `nt_`; otherwise give the generic next action.
    """
    if not api_key.startswith("nt_"):
        return (
            " NT keys start with nt_; the key you sent does not"
            + (f" (it starts with {api_key.split('_', 1)[0]}_)" if "_" in api_key else "")
            + ". An ak_ key comes from the wrong flow — create one in the console "
            "Create-key flow."
        )
    return " Rotate your key in the NT console if it was revoked."


def _http_error_detail(exc) -> str | None:
    """Best-effort: pull FastAPI's {"detail": ...} body off an HTTPError, or None.

    The server's 401 for a rejected key carries the key-format hint in `detail`
    (brief-229). Surfacing it keeps the SDK message in lockstep with the server
    without the SDK hardcoding a second copy. Never raises — diagnostics only.
    """
    import json

    try:
        body = exc.read()
    except Exception:
        return None
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except Exception:
        return None
    detail = parsed.get("detail") if isinstance(parsed, dict) else None
    return detail if isinstance(detail, str) and detail else None


def _fetch_registry(bootstrap_url: str, api_key: str) -> list:
    """GET <bootstrap_url>/v1/models. Single attempt; raises on failure.

    Raises:
        AuthError:          401 from the registry endpoint (bad/wrong-format/rejected
                            key). NOT RegistryUnavailable — a rejected key is a
                            client error, not an outage (brief-229).
        RegistryUnavailable: Network error, 5xx, or non-JSON response — genuine
                            registry/verifier outages only.

    Note on VerifierError: the registry path cannot raise VerifierError. The verifier
    runs inside the registry server; when it's unavailable the server returns HTTP 5xx
    (before the JWT is validated), which becomes RegistryUnavailable. VerifierError
    (WS close 4503) is only emitted on the WS inference path after the TCP+HTTP
    upgrade, when the server checks the key mid-handshake.
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
            # Prefer the server's detail (it carries the key-format hint); fall back
            # to a locally-derived hint so a bad key never reads as a registry outage.
            detail = _http_error_detail(exc)
            message = (
                f"Authentication failed: {detail}"
                if detail
                else "Authentication failed: API key rejected by registry /v1/models."
                + _key_format_hint(api_key)
            )
            raise AuthError(
                code=4001,
                type="auth.invalid_key",
                message=message,
                context={"key_prefix": api_key[:8]},
            ) from exc
        raise RegistryUnavailable(bootstrap_url, f"HTTP {exc.code}") from exc
    except URLError as exc:
        raise RegistryUnavailable(bootstrap_url, str(exc.reason)) from exc
    except Exception as exc:
        raise RegistryUnavailable(bootstrap_url, str(exc)) from exc


def _find_deployable_fine_tunes(base_uid: str, registry: list) -> list[str]:
    """Return the first tag (or UID) of each registry entry that has base_uid as its
    direct base and has an endpoint (i.e. is deployable)."""
    result = []
    for entry in registry:
        if entry.get("base") == base_uid and entry.get("endpoint"):
            tags = entry.get("tags") or []
            uid = entry.get("uid") or ""
            result.append(tags[0] if tags else uid)
    return result


def _resolve_model_endpoint(registry: list, model: str | None, bootstrap_url: str) -> str:
    """Resolve a model UID or tag to its WS endpoint URL from the registry response.

    model=None resolves to _DEFAULT_MODEL_UID. Raises ModelNotFoundError when the
    identifier doesn't match any entry in the registry. Raises BaseNotDeployableError
    when the entry exists but has no endpoint (lineage anchor, not deployable).
    """
    target = model if model is not None else _DEFAULT_MODEL_UID
    matched_entry = None
    for entry in registry:
        uid = entry.get("uid") or ""
        tags = entry.get("tags") or []
        if target == uid or target in tags:
            matched_entry = entry
            break

    if matched_entry is not None:
        endpoint = matched_entry.get("endpoint")
        if endpoint:
            return endpoint
        # Entry exists but no endpoint — lineage anchor, not deployable.
        entry_uid = matched_entry.get("uid") or target
        deployable_tags = _find_deployable_fine_tunes(entry_uid, registry)
        if deployable_tags:
            message = (
                f"Model {target!r} is a base model — not directly deployable. "
                f"Use one of its fine-tunes: {', '.join(deployable_tags)}."
            )
        else:
            message = (
                f"Model {target!r} is a base model — not directly deployable. "
                "No deployable fine-tunes registered for this family yet."
            )
        raise BaseNotDeployableError(
            code=4424,
            type="model.base_not_deployable",
            message=message,
            context={"model": target, "fine_tunes": deployable_tags},
            docs="https://newtheory-docs.vercel.app/docs/api/errors#model-base-not-deployable",
            trace_id="",
        )

    known: list = []
    for entry in registry:
        uid = entry.get("uid")
        if uid:
            known.append(uid)
        for tag in (entry.get("tags") or []):
            known.append(tag)
    raise ModelNotFoundError(model, known)


def _resolve_action_axes(registry: list, model: str | None) -> list[str] | None:
    """Return the resolved model's contract.action_axes from the registry, or None.

    The /v1/models response already injects the inherited contract onto fine-tune
    entries (server resolves the base: chain), so a flat per-entry lookup is enough.
    Returns None when the model isn't found, has no contract, or the contract carries
    no action_axes — callers fall back to dim_N labels. Never raises; the labels are
    cosmetic and must not break the inference path.
    """
    target = model if model is not None else _DEFAULT_MODEL_UID
    for entry in registry:
        uid = entry.get("uid") or ""
        tags = entry.get("tags") or []
        if target == uid or target in tags:
            contract = entry.get("contract") or {}
            axes = contract.get("action_axes")
            if isinstance(axes, list) and axes:
                return [str(a) for a in axes]
            return None
    return None


# ---------------------------------------------------------------------------
# Verifier-unavailable retry
# ---------------------------------------------------------------------------
# The NT key-verification service can be momentarily unavailable when a cold
# system first receives traffic (server emits WS close 4503 with type
# "verifier.unavailable"). The error message prescribes its own fix ("retry in
# a few seconds") — the SDK implements that prescription so the documented
# first-call snippet works verbatim.
#
# Policy:
#   - Only VerifierError with type "verifier.unavailable" is retried (transient).
#   - AuthError (definitively-bad key) is NEVER retried — zero added latency.
#   - Max 4 retries, backoff 3–8s per attempt, total budget ≤45s.
#   - warnings.warn(VerifierTransientRetry) on the FIRST retry only (mirrors
#     ColdStartRetry's pattern).

_VERIFIER_MAX_RETRIES = 4
_VERIFIER_BACKOFF_SECONDS = (3.0, 5.0, 7.0, 8.0)  # one per retry slot

_T = TypeVar("_T")


def _with_verifier_retry(fn: Callable[[], _T]) -> _T:
    """Call fn(); retry transparently on transient VerifierError (verifier.unavailable).

    fn must be a zero-argument callable. Each attempt opens a fresh connection so
    fn must be side-effect-free with respect to state outside itself (i.e. suitable
    to re-run from scratch). AuthError passes through immediately — zero retries.

    Emits VerifierTransientRetry on the first retry.
    """
    last_exc: VerifierError | None = None
    for attempt in range(_VERIFIER_MAX_RETRIES + 1):  # attempt 0 is the initial try
        try:
            return fn()
        except VerifierError as exc:
            if exc.type != "verifier.unavailable":
                # Non-transient verifier error — not eligible for retry.
                raise
            last_exc = exc
            if attempt == _VERIFIER_MAX_RETRIES:
                break  # budget exhausted; fall through to re-raise
            delay = _VERIFIER_BACKOFF_SECONDS[attempt]
            if attempt == 0:
                warnings.warn(
                    VerifierTransientRetry(
                        f"Key verifier temporarily unavailable; retrying in {delay:.0f}s "
                        f"(attempt {attempt + 1}/{_VERIFIER_MAX_RETRIES}). "
                        "Subsequent calls hit the warm verifier."
                    ),
                    stacklevel=3,
                )
            time.sleep(delay)
    # All retries exhausted — re-raise with the original message intact.
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Embodiment validation helpers
# ---------------------------------------------------------------------------
# Three rails, all teaching errors (not bare TypeErrors):
#   1. String → developer thought embodiment= takes a name string.
#   2. Mutual exclusion → embodiment= combined with read_state= or execute=.
#   3. Partial object → object exists but is missing one or both methods.

_EMBODIMENT_DOCS = "https://nt-docs-eight.vercel.app/docs/api/errors#embodiment"


class EmbodimentError(NewTheoryError):
    """Raised when the value passed to Robot(embodiment=...) is invalid.

    Three sub-cases, each with a teaching message:
      - embodiment.string_not_object: a string was passed (name-based API hallucination).
      - embodiment.conflict:          embodiment= combined with read_state= or execute=.
      - embodiment.missing_method:    object is missing read_state or execute (or both).

    Attributes:
        code:     4422 (HTTP-convention; client-side construction error, no WS involved).
        type:     "embodiment.<sub_case>"
        message:  Human-readable prose explaining what's wrong and how to fix it.
        context:  Machine-readable specifics.
        docs:     Stable URL to the embodiment errors reference.
        trace_id: Empty string (client-side; no server trace).
    """

    def __init__(self, type: str, message: str, context: dict) -> None:
        super().__init__(
            code=4422,
            type=type,
            message=message,
            context=context,
            docs=_EMBODIMENT_DOCS,
            trace_id="",
        )

    def __str__(self) -> str:
        return self.message


def _validate_embodiment(embodiment: Any, read_state: Any, execute: Any) -> tuple:
    """Validate embodiment= and return (read_state_fn, execute_fn) to wire.

    Raises EmbodimentError on any of the three bad-input cases.
    Returns the two callables to assign to self._read_state / self._execute.
    When embodiment is None, the bare read_state/execute kwargs are returned
    unchanged (None or callable — Robot's existing optional handling applies).
    """
    if embodiment is None:
        return read_state, execute

    # Rail 1: string → teaching error
    if isinstance(embodiment, str):
        raise EmbodimentError(
            type="embodiment.string_not_object",
            message=(
                f"Robot(embodiment=) takes your embodiment object, not a name string "
                f"(got {embodiment!r}). "
                "Generate one with a starter kit, or implement read_state() and "
                "execute() on any class. "
                f"See {_EMBODIMENT_DOCS}"
            ),
            context={"got": embodiment},
        )

    # Rail 2: mutual exclusion
    conflicts = []
    if read_state is not None:
        conflicts.append("read_state=")
    if execute is not None:
        conflicts.append("execute=")
    if conflicts:
        raise EmbodimentError(
            type="embodiment.conflict",
            message=(
                f"Robot() received both embodiment= and {', '.join(conflicts)}. "
                "Pick one path: pass embodiment= (an object with read_state() and "
                "execute()), or pass read_state= and execute= as separate callbacks. "
                "The two paths are equivalent; embodiment= is convenience shorthand."
            ),
            context={"conflict_kwargs": conflicts},
        )

    # Rail 3: partial object — check both methods, name all missing ones
    missing = []
    if not callable(getattr(embodiment, "read_state", None)):
        missing.append("read_state()")
    if not callable(getattr(embodiment, "execute", None)):
        missing.append("execute()")
    if missing:
        raise EmbodimentError(
            type="embodiment.missing_method",
            message=(
                f"The object passed as embodiment= is missing: {', '.join(missing)}. "
                "An embodiment must implement both read_state() -> dict and "
                "execute(action_chunk) -> None. "
                f"See {_EMBODIMENT_DOCS}"
            ),
            context={
                "missing": missing,
                "got_type": type(embodiment).__name__,
            },
        )

    # Valid embodiment — extract the two methods as callables
    return embodiment.read_state, embodiment.execute


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

class Robot:
    """Client handle for a New Theory inference endpoint.

    Args:
        api_key:     NT API key (nt_xxx); sent as Bearer in WS handshake.
                     Optional — if not provided, the SDK resolves credentials in
                     order: api_key arg → ~/.nt/credentials → NT_API_KEY env var.
                     Raises AuthError if none of the three sources yield a key.
        embodiment:  An object implementing the Embodiment protocol (read_state()
                     and execute()). Convenience shorthand for passing read_state=
                     and execute= separately. Mutually exclusive with read_state= /
                     execute=. Any class with those two methods qualifies — no
                     inheritance or registration required. Passing a string raises
                     EmbodimentError with a pointer to the setup guide.
        read_state:  callable returning an observation dict. Optional keys:
                     "state" (float32 ndarray (14,)), "images" (dict of camera
                     arrays), "prompt" (str). Missing fields are firehose-coerced
                     by the server — partial dicts are fine.
        execute:     callable receiving an action chunk ndarray (action_horizon, 14).
                     Called once per inference cycle in default (non-stream) mode.
                     Never called in stream mode.
        model:       Model identifier (UID or tag). The SDK resolves it via
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

    Embodiment object (preferred for objects you instantiate):
        from embodiment import TrossenWidowX

        robot = newt.Robot(
            embodiment=TrossenWidowX.from_config(),
        )

    Bare callbacks (closures, lambdas — remain forever):
        robot = newt.Robot(
            read_state=lambda: {"state": arm.get_joints()},
            execute=lambda chunk: arm.move_to(chunk[0]),
        )

    Stream usage (caller applies chunks, library drives obs):
        for chunk in robot.run("pick up the cup", stream=True):
            arm.move_to(chunk[0])
    """

    def __init__(
        self,
        api_key: str | None = None,
        embodiment: Any = None,
        read_state: Callable[[], dict] | None = None,
        execute: Callable[[np.ndarray], None] | None = None,
        model: str | None = None,
        connect_timeout: float = 120.0,
    ) -> None:
        # embodiment= is convenience shorthand for read_state= + execute= on one object.
        # Validate and unpack first; read_state/execute are optional (infer() path).
        read_state, execute = _validate_embodiment(embodiment, read_state, execute)

        # read_state/execute are optional: the one-shot infer() path never uses
        # them, so an API-evaluator can construct Robot(api_key=...) and call
        # infer() with no hardware callbacks. run() requires both and guards for it.

        # Credential resolution: api_key arg → ~/.nt/credentials → NT_API_KEY env var
        if api_key is None:
            from newt._credentials import read_api_key
            api_key = read_api_key() or os.environ.get("NT_API_KEY")
        if not api_key:
            raise AuthError(
                code=401,
                type="auth.no_credentials",
                message=(
                    "No API key found. Run `newt login` to authenticate, "
                    "or set the NT_API_KEY environment variable."
                ),
                context={},
            )

        self._api_key = api_key
        self._read_state = read_state
        self._execute = execute
        self._connect_timeout = connect_timeout
        # Forwarded as-is in the first obs frame; None = server uses its default.
        # SDK never interprets UID vs tag — the server resolves the identifier.
        self._model = model

        # Reset to False at the start of each run() call (see _run_blocking/_stream).
        self._degradation_warned: bool = False

        # Consumed after the first _ws_connect() attempt (success or retry).
        # Never reset — cold-start retry fires at most once per Robot instance.
        self._cold_start_retry_consumed: bool = False

        # Test-affordance: NT_INFERENCE_URL takes highest precedence — if set,
        # discovery is skipped and this URL is used directly. Smoke + golden
        # tests use this to repoint at specific servers without touching the registry.
        env_url = os.environ.get("NT_INFERENCE_URL")
        if env_url:
            warnings.warn(
                EnvOverrideWarning(
                    f"NT_INFERENCE_URL is set to {env_url!r} — this overrides dynamic "
                    "/v1/models discovery for ALL models. "
                    "Unset it to use per-model cross-app routing."
                ),
                stacklevel=2,
            )
            self._url = env_url
            self._registry: list = []
        else:
            # Discovery: fetch the registry from the bootstrap URL and resolve the
            # requested model to its WS endpoint. Single network call per construction.
            bootstrap_url = _resolve_bootstrap_url()
            self._registry = _fetch_registry(bootstrap_url, api_key)
            self._url = _resolve_model_endpoint(self._registry, model, bootstrap_url)

    def __repr__(self) -> str:
        try:
            target = self._model if self._model is not None else _DEFAULT_MODEL_UID
            # Resolve display name: prefer first tag, fall back to UID/target.
            display = target
            for entry in self._registry:
                uid = entry.get("uid") or ""
                tags = entry.get("tags") or []
                if target == uid or target in tags:
                    display = tags[0] if tags else uid or target
                    contract = entry.get("contract") or {}
                    shape = contract.get("action_shape")
                    axes = contract.get("action_axes")
                    if shape and len(shape) == 2 and axes:
                        return (
                            f"{display} · contract received · "
                            f"({shape[0]},{shape[1]}) · {len(axes)} labeled axes"
                        )
                    break
            return f"{display} · contract pending"
        except Exception:
            return "<Robot>"

    __str__ = __repr__

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
            ProtocolError:         Frame could not be parsed or has unrecognized type
                                   (WS close 4400).
            ModelNotFoundError:    Requested model not found server-side (WS close 4404).
            ContractMismatchError: Obs frame shapes don't match the resolved
                                   model's declared contract (WS close 4422).
            ServerError:           Inference failure or internal server error
                                   (WS close 4500).
            VerifierError:         Console verifier infrastructure failure
                                   (WS close 4503).
        """
        if self._read_state is None or self._execute is None:
            raise TypeError(
                "Robot.run() requires read_state and execute callbacks. "
                "Construct Robot(api_key, read_state, execute) to drive a robot, "
                "or use Robot(api_key).infer(obs) for one-shot inference without hardware."
            )
        if stream:
            return self._stream(prompt, max_duration)
        return self._run_blocking(prompt, max_duration)

    def infer(self, obs: dict, prompt: str | None = None) -> InferenceResponse:
        """One-shot inference: send a single observation, get one labeled action chunk.

        The single-request counterpart to run()'s streaming loop. Same wire and
        transport (per tenet T1 — invariant client): opens the same authenticated WS,
        sends one obs frame, receives one action chunk, returns it wrapped with the
        model's semantic axis labels and the round-trip latency. Does NOT touch the
        read_state/execute callbacks — infer() is self-contained.

        Args:
            obs:    Observation dict, same shape read_state() returns. Optional keys:
                    "state", "images"/"depth_maps"/"intrinsics"/"extrinsics", "prompt".
                    Missing fields are firehose-coerced server-side — partial dicts
                    (even {}) are fine. If obs carries "prompt", it's used as-is.
            prompt: Language instruction. Used only when obs has no "prompt" of its own
                    (run()'s frame builder prefers the obs prompt). Defaults to "" so the
                    server applies its default prompt when neither is supplied.

        Returns:
            InferenceResponse — .action_chunk (raw ndarray), .axes (semantic labels),
            .latency_ms, .model.

        Raises:
            AuthError, ProtocolError, ModelNotFoundError, ContractMismatchError,
            ServerError, VerifierError — same typed errors as run(), surfaced from
            the shared close-code routing.

        Transient verifier unavailability (VerifierError with type
        "verifier.unavailable") is retried automatically with bounded backoff
        (≤45s total, ≤4 retries). AuthError always propagates immediately.
        """
        import time as _time

        def _attempt() -> InferenceResponse:
            ws = self._ws_connect()
            try:
                frame = _build_obs_frame(obs, prompt or "", None, self._model)
                t0 = _time.perf_counter()
                try:
                    ws.send(_pack(frame))
                except ConnectionClosed as exc:
                    # Server may have closed before recv; drain the close for a typed error.
                    _check_close_error(exc, self._model)
                    raise

                try:
                    raw = ws.recv()
                except ConnectionClosed as exc:
                    _check_close_error(exc, self._model)
                    raise
                latency_ms = (_time.perf_counter() - t0) * 1000.0

                parsed = _unpack(raw)
                _check_error_envelope_frame(parsed)
                ftype = _str_field(parsed, "type")

                if ftype != "action":
                    # terminal-first or any non-action frame: nothing to wrap. Fail loud
                    # rather than return an empty/mislabeled chunk.
                    raise ServerError(
                        code=4500,
                        type="server.no_action",
                        message=(
                            f"infer() expected an action frame, got type={ftype!r}. "
                            "The server returned no action chunk for this single request."
                        ),
                        context={"frame_type": ftype, "model": self._model},
                    )

                _maybe_warn_degradation(parsed, self._model)
                chunk = parsed.get("chunk")
                axes = self._action_axes_for(chunk)
                return InferenceResponse(
                    action_chunk=chunk,
                    axes=axes,
                    latency_ms=latency_ms,
                    model=self._model,
                )
            finally:
                try:
                    ws.close()
                except Exception:
                    pass

        return _with_verifier_retry(_attempt)

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _action_axes_for(self, chunk: Any) -> list[str]:
        """Semantic axis labels for the resolved model, or dim_N fallback.

        Pulls contract.action_axes from the registry fetched at construction. Falls
        back to dim_0..dim_N (sized to the chunk's last dim) with a debug-level log
        when the registry has no labels — e.g. NT_INFERENCE_URL override (empty
        registry) or a model whose contract omits action_axes. Never mislabels: the
        fallback is index-named, not guessed.
        """
        axes = _resolve_action_axes(self._registry, self._model)
        if axes:
            return axes
        import logging

        shape = getattr(chunk, "shape", None)
        n = shape[-1] if shape else 0
        logging.getLogger("newt").debug(
            "No action_axes in registry for model=%r; falling back to dim_0..dim_%d",
            self._model,
            max(n - 1, 0),
        )
        return [f"dim_{i}" for i in range(n)]

    def _ws_connect(self):
        """Open authenticated WS connection. Raises AuthError on HTTP-level rejection.

        open_timeout defaults to 120s (set on Robot via `connect_timeout=`) so
        cold-start cost (Modal scale-down → GPU + checkpoint-load, 30–90s typical)
        doesn't surface as a handshake timeout.

        On the FIRST call of this Robot instance's lifecycle, a TimeoutError triggers
        one automatic retry with open_timeout=180 and emits a ColdStartRetry warning.
        Subsequent calls (warm container, reconnects) do not retry. If the 180s retry
        also times out, the original TimeoutError is re-raised unmasked.
        """
        try:
            ws = connect(
                self._url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                open_timeout=self._connect_timeout,
            )
            self._cold_start_retry_consumed = True
            return ws
        except TimeoutError as original_exc:
            if self._cold_start_retry_consumed:
                raise
            self._cold_start_retry_consumed = True
            model_str = self._model or "nt0-fp3"
            warnings.warn(
                ColdStartRetry(
                    f"Cold-start retry for model={model_str!r} "
                    "(warming up the container, may take 60-90s). "
                    "Subsequent calls hit the warm container."
                ),
                stacklevel=3,
            )
            try:
                return connect(
                    self._url,
                    additional_headers={"Authorization": f"Bearer {self._api_key}"},
                    open_timeout=180,
                )
            except Exception:
                raise original_exc
        except InvalidStatus as exc:
            self._cold_start_retry_consumed = True
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
        return _with_verifier_retry(lambda: self._run_blocking_once(prompt, max_duration))

    def _run_blocking_once(self, prompt: str, max_duration: float) -> RunResult:
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
                    _check_close_error(exc, self._model)
                    break  # connection closed cleanly (no known error code)

                parsed = _unpack(raw)
                _check_error_envelope_frame(parsed)
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
        # Verifier retry wrapper for the stream path. VerifierError fires on the
        # first WS operation (before any chunks are yielded), so it's safe to retry
        # the whole connection from scratch. After the first successful chunk, any
        # VerifierError would be a mid-stream event (not a cold-verifier race) and
        # propagates normally.
        last_exc: VerifierError | None = None
        warned = False
        for attempt in range(_VERIFIER_MAX_RETRIES + 1):
            try:
                yield from self._stream_once(prompt, max_duration)
                return
            except VerifierError as exc:
                if exc.type != "verifier.unavailable" or attempt == _VERIFIER_MAX_RETRIES:
                    raise
                last_exc = exc
                delay = _VERIFIER_BACKOFF_SECONDS[attempt]
                if not warned:
                    warned = True
                    warnings.warn(
                        VerifierTransientRetry(
                            f"Key verifier temporarily unavailable; retrying in {delay:.0f}s "
                            f"(attempt {attempt + 1}/{_VERIFIER_MAX_RETRIES}). "
                            "Subsequent calls hit the warm verifier."
                        ),
                        stacklevel=3,
                    )
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _stream_once(
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
                    _check_close_error(exc, self._model)
                    return

                parsed = _unpack(raw)
                _check_error_envelope_frame(parsed)
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
    """Fetch the list of available models from the always-on NT registry.

    Args:
        api_key:  NT API key (nt_xxx).
        base_url: Override HTTP base URL (e.g. http://localhost:8000). Defaults to
                  the NT registry (Railway); overridden by NT_BOOTSTRAP_URL env var.

    Returns:
        List of model dicts with uid, tags, type, and base fields.

    Raises:
        AuthError: API key rejected by the registry.
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
            detail = _http_error_detail(exc)
            message = (
                f"Authentication failed: {detail}"
                if detail
                else "Authentication failed: API key rejected by /v1/models."
                + _key_format_hint(api_key)
            )
            raise AuthError(
                code=4001,
                type="auth.invalid_key",
                message=message,
                context={"key_prefix": api_key[:8]},
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


# ---------------------------------------------------------------------------
# Close-code routing
# ---------------------------------------------------------------------------

# Hardcoded per C1 decision — type-checked, predictable, no install-time YAML
# parsing. Catalog-driven migration is deferred.
_CLOSE_CODE_TO_EXCEPTION: dict[int, type[NewTheoryError]] = {
    4001: AuthError,
    4400: ProtocolError,
    4404: ModelNotFoundError,
    4422: ContractMismatchError,
    4500: ServerError,
    4503: VerifierError,
}

# Default type per close code — used for bare-close fallback when no envelope
# frame was received before the close.
_DEFAULT_TYPE_FOR_CODE: dict[int, str] = {
    4001: "auth.invalid_key",
    4400: "protocol.unknown_type",
    4404: "model_not_found.unknown_identifier",
    4422: "contract_mismatch.unknown",
    4500: "server.internal",
    4503: "verifier.unavailable",
}


def _check_error_envelope_frame(parsed: dict) -> None:
    """Raise the typed exception if `parsed` is a server error envelope frame.

    The server sends a msgpack binary message immediately before the WS close
    frame, carrying the canonical six-field error envelope (code, type, message,
    context, docs, trace_id — per portal/wiki/operating-docs/error-style.md).
    Covers all close codes: 4001, 4400, 4404, 4422, 4500, 4503.

    Uses NewTheoryError.__init__ directly to bypass per-subclass custom
    constructors (e.g. ModelNotFoundError's client-side convenience init).
    """
    code = _decode_key(parsed, "code")
    exc_class = _CLOSE_CODE_TO_EXCEPTION.get(code)
    if exc_class is None:
        return
    type_ = _decode_key(parsed, "type") or _DEFAULT_TYPE_FOR_CODE.get(code, "unknown.error")
    message = _decode_key(parsed, "message") or f"Server error (code {code})."
    context = _decode_key(parsed, "context") or {}
    docs = _decode_key(parsed, "docs")
    trace_id = _decode_key(parsed, "trace_id") or ""

    inst = exc_class.__new__(exc_class)
    NewTheoryError.__init__(inst, code=code, type=type_, message=message,
                            context=context, docs=docs, trace_id=trace_id)
    raise inst


def _check_close_error(exc: ConnectionClosed, model: str | None = None) -> None:
    """Raise the typed exception for any known 4xxx WS close code.

    Fallback for when no envelope frame was received before the close
    (e.g. connection dropped between send_bytes and close). Raises with
    minimal context — developer still gets a typed exception, not
    stop_reason="error".
    """
    rcvd = getattr(exc, "rcvd", None)
    if not rcvd:
        return
    code = getattr(rcvd, "code", None)
    exc_class = _CLOSE_CODE_TO_EXCEPTION.get(code)
    if exc_class is None:
        return

    reason = getattr(rcvd, "reason", "") or ""
    type_ = _DEFAULT_TYPE_FOR_CODE.get(code, "unknown.error")
    message = reason or f"Server closed connection with code {code}."
    context: dict = {"model": model} if model else {}

    inst = exc_class.__new__(exc_class)
    NewTheoryError.__init__(inst, code=code, type=type_, message=message,
                            context=context, docs=None, trace_id="")
    raise inst from exc
