"""newt logout — delete local credentials (inverse of newt login).

Semantics match gh/vercel convention:
- Deletes ~/.nt/credentials ONLY; never revokes the key server-side.
- Remote revocation belongs to the console keys page.
- Idempotent: no credentials file → exit 0 with 'already logged out' message.
- NT_API_KEY in environment: named loudly after logout because we can't unset a
  parent shell's environment variable.

Directory handling: remove the credentials file; remove ~/.nt/ only if it is
now empty. This mirrors write_api_key, which creates the dir when writing —
cleanup is symmetrical.
"""
from __future__ import annotations

import json
import os
import sys

from newt._credentials import CREDENTIALS_DIR, CREDENTIALS_PATH

_DEFAULT_CONSOLE = "https://console.newtheory.ai"


def _console_url() -> str:
    return os.environ.get("NT_CONSOLE_URL", _DEFAULT_CONSOLE).rstrip("/")


def cmd_logout(args: list[str]) -> int:
    as_json = "--json" in args

    credentials_existed = CREDENTIALS_PATH.exists()
    env_key_set = bool(os.environ.get("NT_API_KEY"))

    if credentials_existed:
        CREDENTIALS_PATH.unlink()
        # Remove the directory only if it is now empty — leave it alone if
        # other files (e.g. config, future keys) exist alongside credentials.
        try:
            CREDENTIALS_DIR.rmdir()
        except OSError:
            # Directory non-empty or removal failed — leave it; file is gone.
            pass
        action = "removed"
    else:
        action = "already_logged_out"

    if as_json:
        print(json.dumps({
            "action": action,
            "credentials_path": str(CREDENTIALS_PATH),
            "env_var_warning": env_key_set,
        }))
        return 0

    if credentials_existed:
        print(f"Logged out.")
        print(f"  Removed:  {CREDENTIALS_PATH}")
        print(f"  The key itself remains valid — revoke it at {_console_url()}/settings/keys")
    else:
        print("Already logged out — no credentials file found.")

    if env_key_set:
        print()
        print(
            "Warning: NT_API_KEY is still set in your environment and will be used for auth."
            " Unset it if you intended a full sign-out."
        )

    return 0
