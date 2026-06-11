"""~/.nt/credentials read/write utilities.

The credentials file format is a single line:
    api_key = nt_<hex>

**Credential resolution precedence (canonical statement):**
NT_API_KEY environment variable wins; ~/.nt/credentials file is the fallback.
In code: ``os.environ.get("NT_API_KEY") or read_api_key()``.
Every resolution site in the SDK and CLI must follow this order and cite here
rather than restating the rule.

`newt login` writes the credentials file after pairing with the console.
"""
from __future__ import annotations

import stat
from pathlib import Path

CREDENTIALS_DIR = Path.home() / ".nt"
CREDENTIALS_PATH = CREDENTIALS_DIR / "credentials"


def read_api_key() -> str | None:
    """Read the API key from ~/.nt/credentials. Returns None if absent or unreadable."""
    if not CREDENTIALS_PATH.exists():
        return None
    try:
        for line in CREDENTIALS_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("api_key"):
                _, _, value = line.partition("=")
                key = value.strip()
                if key:
                    return key
            elif line.startswith("nt_"):
                return line
    except OSError:
        return None
    return None


def write_api_key(key: str) -> None:
    """Write the API key to ~/.nt/credentials with chmod 600."""
    CREDENTIALS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    CREDENTIALS_PATH.write_text(f"api_key = {key}\n")
    CREDENTIALS_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
