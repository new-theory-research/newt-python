"""newt — the New Theory developer-facing Python library.

Importable as `import newt`. Distribution name on PyPI: `newt`.

Public surface:
- `newt.Robot`                — top-level robot handle
- `newt.AuthError`            — raised when API key is rejected by the server
- `newt.ContractMismatchError` — raised when obs frame shapes don't match the
                                 resolved model's declared contract (WS close 4422)
- `newt.ModelNotFoundError`   — raised when the requested model UID or tag isn't
                                 in the registry (client-side, before WS connection)
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
    ContractMismatchError,
    DegradationWarning,
    ModelNotFoundError,
    RegistryUnavailable,
    Robot,
    RunResult,
    list_models,
)

__all__ = [
    "AuthError",
    "ContractMismatchError",
    "DegradationWarning",
    "ModelNotFoundError",
    "RegistryUnavailable",
    "Robot",
    "RunResult",
    "list_models",
]
