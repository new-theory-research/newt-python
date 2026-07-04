"""Credential-hygiene gate for the ``objectCreator`` SA (Rule 10 / brief
capture-002's one unforgivable failure mode): the ``nt-episodes-writer`` key
must never be embedded, downloaded, or cached anywhere in the shipped SDK
tree (``src/``).

This is a static scan, not a mock-based unit test — it must fail loudly if a
future change accidentally bakes real key material into the tree, without
false-alarming on the SA name being *mentioned* in prose (e.g. the
``_cloud_sink.py`` module docstring explains the write path by name).
"""
from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

# Hallmarks of actual GCP service-account key material — not just the SA
# name. These never legitimately appear in application source.
_KEY_MATERIAL_MARKERS = (
    "BEGIN PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    '"private_key"',
    '"type": "service_account"',
    '"type":"service_account"',
)

# Lines allowed to reference this env var name: only a lookup, never a literal
# value assigned to it (which would mean a key-file path or contents baked in).
_ALLOWED_ENV_VAR_LOOKUPS = ("os.environ", "os.getenv")


def _iter_source_files():
    for path in sorted(_SRC.rglob("*")):
        if path.is_file():
            yield path


def test_no_sa_key_material_anywhere_in_sdk_tree():
    hits = []
    for path in _iter_source_files():
        text = path.read_text(errors="ignore")
        for marker in _KEY_MATERIAL_MARKERS:
            if marker in text:
                hits.append(f"{path}: contains {marker!r}")
    assert not hits, "SA key material found in SDK tree:\n" + "\n".join(hits)


def test_google_application_credentials_only_referenced_as_env_var_lookup():
    """The env var *name* is fine to reference (e.g. in docs telling a deployer
    to set it); a literal value assigned to it would mean a credential path or
    the key contents got baked into the tree."""
    violations = []
    for path in _iter_source_files():
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
            if "GOOGLE_APPLICATION_CREDENTIALS" not in line:
                continue
            if any(marker in line for marker in _ALLOWED_ENV_VAR_LOOKUPS):
                continue
            if "GOOGLE_APPLICATION_CREDENTIALS" in line and "=" not in line.split(
                "GOOGLE_APPLICATION_CREDENTIALS", 1
            )[1].split("#")[0]:
                # Mentioned but not assigned (e.g. prose/docstring) — fine.
                continue
            violations.append(f"{path}:{lineno}: {line.strip()!r}")
    assert not violations, (
        "GOOGLE_APPLICATION_CREDENTIALS assigned a literal value in source "
        "(should only ever be an env var lookup):\n" + "\n".join(violations)
    )
