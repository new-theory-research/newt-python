"""newt login — browser-pairing flow (Stripe pattern).

1. POST /api/cli/auth/start  → browser_url + user_code + poll_url
2. Print URL + code; attempt webbrowser.open (silent on SSH/headless)
3. Poll /api/cli/auth/poll every 2s until confirmed, expired, or TTL deadline
4. Write key to ~/.nt/credentials (chmod 600), print loud completion message

TTL handling: the start response is checked for `expires_in` (seconds) or
`expires_at` (Unix timestamp). If neither is present, we fall back to the
server-side TTL constant of 10 minutes. As of the initial implementation the
server does NOT return an expiry field, so the 10-minute fallback is always
used in practice. When the server starts returning one, the client will use it
automatically.

--print flag: runs the identical pairing flow but routes all instructional output
to stderr and writes the bare key to stdout; ~/.nt/credentials is never touched.
Compose with: KEY=$(newt login --print)
"""
from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
import webbrowser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from newt._credentials import write_api_key

_DEFAULT_CONSOLE = "https://console-production-91bb.up.railway.app"
_POLL_INTERVAL_S = 2.0
_MAX_WAIT_S = 10 * 60  # 10 minutes, matching server TTL
_REEMIT_INTERVAL_S = 30.0  # re-print URL+code every ~30s during polling

# Single expiry message used for all expiration paths (deadline reached OR
# server signals expired). Naming NT_API_KEY here is intentional: headless
# environments (CI, agents, bare SSH) cannot complete the browser flow and
# must use a key directly.
_EXPIRY_MSG = (
    "Pairing expired — run `newt login` again. "
    "In non-interactive environments (CI, agents), set NT_API_KEY instead."
)


def _console_url() -> str:
    return os.environ.get("NT_CONSOLE_URL", _DEFAULT_CONSOLE).rstrip("/")


def _device_name() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return platform.node() or "unknown"


def _usage() -> None:
    print("Usage: newt login [options]")
    print("")
    print("  Opens a browser pairing flow and writes your API key to ~/.nt/credentials.")
    print("  In headless environments, set NT_API_KEY instead.")
    print("")
    print("Options:")
    print("  --print  Print the key to stdout; do not write credentials.")
    print("           Compose with: KEY=$(newt login --print)")
    print("")
    print("Environment:")
    print("  NT_API_KEY     Set this instead of running login (CI, agents, SSH).")
    print("  NT_CONSOLE_URL Override the console URL (default: https://console-production-91bb.up.railway.app)")


def cmd_login(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    print_only = "--print" in args

    # All instructional output goes to stderr when --print is set so that
    # stdout carries nothing but the bare key (composable with $(...)).
    out = sys.stderr if print_only else sys.stdout

    console = _console_url()
    device = _device_name()

    print("Starting authentication…", file=out)

    # Step 1: create a pairing record on the console
    body = json.dumps({"device_name": device}).encode()
    req = Request(
        f"{console}/api/cli/auth/start",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except HTTPError as exc:
        print(f"\nError from console ({exc.code}): {exc.reason}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"\nCannot reach {console}: {exc.reason}", file=sys.stderr)
        print("Set NT_CONSOLE_URL if you're running a local console.", file=sys.stderr)
        return 1

    browser_url: str = data["browser_url"]
    user_code: str = data["user_code"]
    poll_url: str = data["poll_url"]

    # Derive the poll deadline from the server's expiry hint when present,
    # falling back to the _MAX_WAIT_S constant.  The server currently does
    # not return either field, so the fallback always applies in practice.
    now = time.monotonic()
    if "expires_in" in data:
        deadline = now + float(data["expires_in"])
    elif "expires_at" in data:
        # expires_at is a Unix wall-clock timestamp; convert to monotonic.
        wall_remaining = float(data["expires_at"]) - time.time()
        deadline = now + max(wall_remaining, 0.0)
    else:
        deadline = now + _MAX_WAIT_S

    print(f"\n  Open this URL to authenticate:\n\n    {browser_url}\n", file=out)
    print(f"  Confirm this code matches what you see in your browser:\n", file=out)
    print(f"      {user_code}\n", file=out)

    # Step 2: attempt browser open (silent failure for SSH/headless rigs)
    try:
        opened = webbrowser.open(browser_url)
    except Exception:
        opened = False

    if opened:
        print(f"  Browser opened. If nothing appeared, paste this URL manually:\n    {browser_url}", file=out)
    else:
        print(f"  No browser detected — open {browser_url} on any device.", file=out)
        print(
            "  Scripting or agent? Use `newt login --print` or set NT_API_KEY instead.",
            file=out,
        )

    print(f"\nWaiting for you to confirm at {browser_url} ...", file=out, flush=True)

    # Step 3: poll until confirmed, expired, or deadline
    _reemit_counter = 0  # counts poll iterations; re-emit URL+code every _REEMIT_INTERVAL_S
    _reemit_every = max(1, int(_REEMIT_INTERVAL_S / _POLL_INTERVAL_S))
    while time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
        _reemit_counter += 1
        if _reemit_counter % _reemit_every == 0:
            print(
                f"  Still waiting — confirm at {browser_url}  (code: {user_code})",
                file=out,
                flush=True,
            )

        try:
            with urlopen(Request(poll_url), timeout=15) as resp:
                poll = json.loads(resp.read())
        except HTTPError as exc:
            if exc.code in (410, 404):
                # Server signals the pairing is gone — no point continuing.
                print(f"\n\n{_EXPIRY_MSG}", file=sys.stderr)
                return 1
            print(f"\n\nUnexpected error polling ({exc.code}): {exc.reason}", file=sys.stderr)
            return 1
        except URLError as exc:
            print(f"\n\nNetwork error while polling: {exc.reason}", file=sys.stderr)
            return 1

        status = poll.get("status")
        if status == "pending":
            continue

        if status == "expired":
            # Poll body explicitly signals expiry before our local deadline.
            print(f"\n\n{_EXPIRY_MSG}", file=sys.stderr)
            return 1

        if status == "confirmed":
            key = poll.get("key")
            if not key:
                # Key was already burned (duplicate poll race) — shouldn't happen
                # in normal flow, but treat as success since the login already ran.
                print(
                    "\n\nLogin was confirmed but the key was already delivered. "
                    "Check ~/.nt/credentials.",
                    file=sys.stderr,
                )
                return 1

            if print_only:
                # Composability contract: bare key on stdout, nothing else.
                print(key)
                print(
                    "\n\nKey not saved — export NT_API_KEY or store it yourself.",
                    file=sys.stderr,
                )
            else:
                write_api_key(key)
                prefix = key[:12] if len(key) > 12 else key
                print(f"\n\nLogged in successfully.")
                print(f"  Key written to:  ~/.nt/credentials  (mode 0600)")
                print(f"  Key prefix:      {prefix}…")
                print(f"  Device:          {device}")
                print(f"\nThe SDK will read this key automatically. No NT_API_KEY needed.")
                print(f"To revoke this key, visit the console key management page.")
                print(f"\nUsing Claude Code? Run `newt skill install` to equip it with the onboarding guide.")
            return 0

        # Unknown status — escalate rather than silently retry
        print(f"\n\nUnexpected poll response: {poll}", file=sys.stderr)
        return 1

    print(f"\n\n{_EXPIRY_MSG}", file=sys.stderr)
    return 1
