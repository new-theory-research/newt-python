"""newt status — key provenance, overrides, and registry connectivity."""
from __future__ import annotations

import json
import os
import sys
import time

from newt._credentials import CREDENTIALS_PATH, read_api_key

_RESET = "\033[0m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_AMBER = "\033[93m"
_GRAY = "\033[90m"


def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


def _resolve_key() -> tuple[str | None, str]:
    """Return (api_key, source) — source is 'env_var' | 'credentials_file' | 'none'."""
    env_key = os.environ.get("NT_API_KEY")
    if env_key:
        return env_key, "env_var"
    file_key = read_api_key()
    if file_key:
        return file_key, "credentials_file"
    return None, "none"


def _effective_bootstrap_url() -> str:
    """Effective registry bootstrap URL — mirrors _resolve_bootstrap_url in robot.py."""
    if url := os.environ.get("NT_BOOTSTRAP_URL"):
        return url
    if ws_url := os.environ.get("NT_INFERENCE_URL"):
        https = ws_url.replace("wss://", "https://", 1).replace("ws://", "http://", 1)
        return https.rsplit("/", 1)[0]
    return "https://nt-registry-production.up.railway.app"


def cmd_status(args: list[str]) -> int:
    as_json = "--json" in args

    api_key, key_source = _resolve_key()
    bootstrap_override = os.environ.get("NT_BOOTSTRAP_URL")
    inference_override = os.environ.get("NT_INFERENCE_URL")
    effective_url = _effective_bootstrap_url()

    registry_reachable: bool = False
    latency_ms: int | None = None
    model_count: int | None = None
    error_kind: str | None = None   # 'no_key' | 'bad_key' | 'registry_down' | 'error'
    error_detail: str | None = None

    if not api_key:
        error_kind = "no_key"
        error_detail = "No API key. Run `newt login` to authenticate, or set NT_API_KEY."
    else:
        import newt

        t0 = time.monotonic()
        try:
            models = newt.list_models(api_key)
            latency_ms = round((time.monotonic() - t0) * 1000)
            registry_reachable = True
            model_count = len(models)
        except newt.AuthError as exc:
            latency_ms = round((time.monotonic() - t0) * 1000)
            registry_reachable = True   # registry responded (401 is a response)
            error_kind = "bad_key"
            error_detail = str(exc)
        except newt.RegistryUnavailable as exc:
            latency_ms = round((time.monotonic() - t0) * 1000)
            registry_reachable = False
            error_kind = "registry_down"
            error_detail = str(exc)
        except Exception as exc:
            latency_ms = round((time.monotonic() - t0) * 1000)
            registry_reachable = False
            error_kind = "error"
            error_detail = str(exc)

    if as_json:
        data: dict = {
            "key_source": key_source,
            "overrides": {
                "NT_BOOTSTRAP_URL": bootstrap_override,
                "NT_INFERENCE_URL": inference_override,
            },
            "registry_reachable": registry_reachable,
            "latency_ms": latency_ms,
        }
        print(json.dumps(data))
        return 0 if error_kind is None else 1

    _render(
        api_key=api_key,
        key_source=key_source,
        bootstrap_override=bootstrap_override,
        inference_override=inference_override,
        effective_url=effective_url,
        registry_reachable=registry_reachable,
        latency_ms=latency_ms,
        model_count=model_count,
        error_kind=error_kind,
        error_detail=error_detail,
    )
    return 0 if error_kind is None else 1


def _render(
    *,
    api_key: str | None,
    key_source: str,
    bootstrap_override: str | None,
    inference_override: str | None,
    effective_url: str,
    registry_reachable: bool,
    latency_ms: int | None,
    model_count: int | None,
    error_kind: str | None,
    error_detail: str | None,
) -> None:
    if key_source == "env_var":
        src = "NT_API_KEY (environment)"
    elif key_source == "credentials_file":
        src = f"credentials file  ({CREDENTIALS_PATH})"
    else:
        src = "none"

    print(f"Key source:   {src}")
    if api_key:
        prefix = api_key[:12] + "…" if len(api_key) > 12 else api_key
        print(f"Key:          {_c(_GRAY, prefix)}")

    if bootstrap_override or inference_override:
        print()
        print(_c(_AMBER, "Overrides active:"))
        if bootstrap_override:
            print(f"  NT_BOOTSTRAP_URL  = {bootstrap_override}")
        if inference_override:
            print(f"  NT_INFERENCE_URL  = {inference_override}")

    print(f"Bootstrap:    {effective_url}")
    print()

    if error_kind == "no_key":
        print(_c(_RED, "Registry:     skipped (no key)"))
        print(f"              {error_detail}")
    elif error_kind == "bad_key":
        print(_c(_RED, f"Registry:     reachable ({latency_ms}ms) — key rejected"))
        print(f"              {error_detail}")
        print("              Rotate your key in the console, or run `newt login` again.")
    elif error_kind in ("registry_down", "error"):
        print(_c(_RED, f"Registry:     unreachable ({latency_ms}ms)"))
        print(f"              {error_detail}")
    else:
        ms = f"{latency_ms}ms" if latency_ms is not None else "?"
        n = f"  •  {model_count} model{'s' if model_count != 1 else ''}" if model_count is not None else ""
        print(_c(_GREEN, f"Registry:     reachable ({ms}){n}"))
