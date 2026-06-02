"""newt — the New Theory developer-facing Python library.

Importable as `import newt`. Distribution name on PyPI: `newt`.

Public surface:
- `newt.Robot`                — top-level robot handle
- `newt.NewTheoryError`       — base class for all server-emitted errors; six-field
                                 envelope (code, type, message, context, docs, trace_id)
- `newt.AuthError`            — raised when API key is rejected (WS close 4001 / HTTP 401)
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
- `newt.DegradationWarning`   — warnings.warn'd once per run() when expected cameras
                                 are absent; actions succeed but may be degraded
- `newt.RunResult`            — returned by Robot.run() (non-stream mode)
- `newt.list_models`          — fetch available models from the inference server
- `newt.trossen.WidowX_250`   — vendor-namespaced robot class (planned)

Internal:
- `newt._client` — edge client, invariant per tenet T1
- `newt._translation` — action-format translation layer
"""

from newt._client.robot import (
    AuthError,
    BaseNotDeployableError,
    ContractMismatchError,
    DegradationWarning,
    ModelNotFoundError,
    NewTheoryError,
    ProtocolError,
    RegistryUnavailable,
    Robot,
    RunResult,
    ServerError,
    VerifierError,
    list_models,
)

__all__ = [
    "AuthError",
    "BaseNotDeployableError",
    "ContractMismatchError",
    "DegradationWarning",
    "ModelNotFoundError",
    "NewTheoryError",
    "ProtocolError",
    "RegistryUnavailable",
    "Robot",
    "RunResult",
    "ServerError",
    "VerifierError",
    "list_models",
]
