"""newt — the New Theory developer-facing Python library.

Importable as `import newt`. Distribution name on PyPI: `newt`.

Public surface:
- `newt.Robot`     — top-level robot handle
- `newt.AuthError` — raised when API key is rejected by the server
- `newt.RunResult` — returned by Robot.run() (non-stream mode)
- `newt.trossen.WidowX_250` — vendor-namespaced robot class (planned)

Internal:
- `newt._client` — edge client, invariant per tenet T1
- `newt._translation` — action-format translation layer
"""

from newt._client.robot import AuthError, Robot, RunResult

__all__ = ["AuthError", "Robot", "RunResult"]
