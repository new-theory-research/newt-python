"""newt — the New Theory developer-facing Python library.

Importable as `import newt`. Distribution name on PyPI: `newt`.

Public surface:
- `newt.Robot`                — top-level robot handle
- `newt.Embodiment`           — typing.Protocol for hardware drivers passed to
                                 Robot(embodiment=...); any object with read_state()
                                 and execute() satisfies it — no inheritance required
- `newt.NewTheoryError`       — base class for all server-emitted errors; six-field
                                 envelope (code, type, message, context, docs, trace_id)
- `newt.AuthError`            — raised when API key is rejected (WS close 4001 / HTTP 401)
- `newt.EndpointUnavailableError` — raised when the WS upgrade is rejected at the HTTP
                                 layer for a non-auth reason (404 = endpoint not deployed /
                                 stale registry URL; 5xx = service down). Never an API-key issue.
- `newt.EmbodimentError`      — raised when Robot(embodiment=...) receives an invalid
                                 value: a string name, a conflict with read_state=/execute=,
                                 or an object missing one or both required methods
- `newt.ProtocolError`        — raised when obs frame is malformed or has unknown type
                                 (WS close 4400)
- `newt.BaseNotDeployableError` — raised when the requested model is a base (lineage anchor)
                                 with no endpoint; fires client-side before WS connection
                                 with context.fine_tunes listing deployable alternatives
- `newt.ModelNotFoundError`   — raised when the requested model UID or tag isn't found
                                 (client-side before WS connection, or WS close 4404)
- `newt.ContractMismatchError` — raised when obs frame shapes don't match the resolved
                                 model's declared contract (WS close 4422)
- `newt.ServerError`          — raised on server-side inference or internal error
                                 (WS close 4500)
- `newt.VerifierError`        — raised when the console verifier is unavailable at
                                 handshake (WS close 4503)
- `newt.RegistryUnavailable`  — raised when the registry fetch itself fails
                                 (network error, 5xx, or malformed JSON)
- `newt.ColdStartRetry`       — warnings.warn'd once per Robot instance on first-connect
                                 timeout; SDK retried with 180s timeout (Modal cold-start)
- `newt.DegradationWarning`   — warnings.warn'd once per run() when expected cameras
                                 are absent; actions succeed but may be degraded
- `newt.EnvOverrideWarning`   — warnings.warn'd once per Robot instance when
                                 NT_INFERENCE_URL is set, bypassing /v1/models discovery
- `newt.VerifierTransientRetry` — warnings.warn'd once per call on first retry when the
                                 key verifier is temporarily unavailable (WS close 4503,
                                 type "verifier.unavailable"); SDK retries automatically
                                 with bounded backoff (≤45s)
- `newt.RunResult`            — returned by Robot.run() (non-stream mode)
- `newt.InferenceResponse`    — returned by Robot.infer() (one-shot); wraps the raw
                                 action chunk with semantic axis labels + latency
- `newt.list_models`          — fetch available models from the inference server
- `newt.snapshots`            — real recorded observations (`snapshots.load("cup_stacking")`)
                                 for trying inference without hardware
- `newt.fixtures`             — deprecated alias for `newt.snapshots`; emits DeprecationWarning

Internal:
- `newt._client` — edge client, invariant per tenet T1
"""

from newt._client.robot import (
    AuthError,
    BaseNotDeployableError,
    ColdStartRetry,
    ContractMismatchError,
    DegradationWarning,
    EmbodimentError,
    EndpointUnavailableError,
    EnvOverrideWarning,
    InferenceResponse,
    ModelNotFoundError,
    NewTheoryError,
    ProtocolError,
    RegistryUnavailable,
    Robot,
    RunResult,
    ServerError,
    VerifierError,
    VerifierTransientRetry,
    list_models,
)
from newt._embodiment import Embodiment
from newt import snapshots

__all__ = [
    "AuthError",
    "BaseNotDeployableError",
    "ColdStartRetry",
    "ContractMismatchError",
    "DegradationWarning",
    "Embodiment",
    "EmbodimentError",
    "EndpointUnavailableError",
    "EnvOverrideWarning",
    "InferenceResponse",
    "ModelNotFoundError",
    "NewTheoryError",
    "ProtocolError",
    "RegistryUnavailable",
    "Robot",
    "RunResult",
    "ServerError",
    "VerifierError",
    "VerifierTransientRetry",
    "snapshots",
    "list_models",
]
