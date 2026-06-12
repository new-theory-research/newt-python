"""CI golden — CLI reference parity check.

Two-direction check:
  (a) Every verb documented in cli.mdx exists in the dispatcher.
  (b) Every verb the dispatcher handles appears in cli.mdx.

A failing test means one of:
  - A verb was added to cli.mdx that the dispatcher doesn't handle (invented surface)
  - A verb was added to the dispatcher without being documented (coverage gap)

Run:
    uv run pytest tests/test_cli_reference.py -v

To point at a portal repo in a non-standard location:
    PORTAL_DOCS_ROOT=/path/to/portal/apps/docs uv run pytest tests/test_cli_reference.py -v
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess

import pytest


# ---------------------------------------------------------------------------
# Dispatcher verbs — parsed from source
# ---------------------------------------------------------------------------

def _dispatcher_verbs() -> set[str]:
    """Parse top-level verb names from the CLI dispatcher __init__.py.

    Looks for: if cmd == "<verb>":
    Fails hard if the dispatcher source can't be read — that itself is a signal.
    """
    init_path = pathlib.Path(__file__).parent.parent / "src" / "newt" / "_cli" / "__init__.py"
    source = init_path.read_text(encoding="utf-8")
    verbs = re.findall(r'if cmd == "([a-z][a-z0-9_-]*)"', source)
    assert verbs, "dispatcher __init__.py yielded no verbs — pattern may be stale"
    return set(verbs)


# ---------------------------------------------------------------------------
# Documented verbs — parsed from cli.mdx
# ---------------------------------------------------------------------------

def _find_portal_root() -> pathlib.Path | None:
    """Find the portal repo root, resolving through git worktrees.

    From a worktree, --git-common-dir points to the main .git dir, whose parent
    is the real repo root. From there ../portal is the sibling portal repo.
    """
    test_dir = pathlib.Path(__file__).parent
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=test_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    git_dir = pathlib.Path(result.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = (test_dir / git_dir).resolve()
    # git_dir = <repo>/.git  →  .parent = <repo>  →  .parent.parent = repos root
    repos_root = git_dir.parent.parent
    portal = repos_root / "portal"
    return portal if portal.exists() else None


def _find_cli_mdx() -> pathlib.Path | None:
    """Find apps/docs/content/docs/reference/cli.mdx in the portal repo."""
    env_root = os.environ.get("PORTAL_DOCS_ROOT")
    if env_root:
        candidate = pathlib.Path(env_root) / "content" / "docs" / "reference" / "cli.mdx"
        if candidate.exists():
            return candidate

    portal_root = _find_portal_root()
    if portal_root:
        candidate = (
            portal_root / "apps" / "docs" / "content" / "docs" / "reference" / "cli.mdx"
        )
        if candidate.exists():
            return candidate

    return None


def _documented_verbs(mdx_path: pathlib.Path) -> set[str]:
    """Parse top-level command names from cli.mdx h3 headings.

    Matches: ### `newt <verb>`
    Excludes subcommands (h4: #### `newt skill install`) and non-command h3s.
    """
    text = mdx_path.read_text(encoding="utf-8")
    # Backtick + newt + one word = top-level verb; two words = subcommand (h4, excluded)
    verbs = re.findall(r"^### `newt ([a-z][a-z0-9_-]*)`", text, re.MULTILINE)
    return set(verbs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cli_mdx() -> pathlib.Path:
    path = _find_cli_mdx()
    if path is None:
        pytest.skip(
            "cli.mdx not found — set PORTAL_DOCS_ROOT=/path/to/portal/apps/docs to locate it"
        )
    return path


# ---------------------------------------------------------------------------
# Two-direction parity tests
# ---------------------------------------------------------------------------

def test_dispatcher_verbs_all_documented(cli_mdx: pathlib.Path) -> None:
    """Every verb the dispatcher handles must appear in cli.mdx.

    If a verb is added to the dispatcher without updating the reference page,
    this test fails: "dispatcher verb 'X' missing from docs".
    """
    dispatcher = _dispatcher_verbs()
    documented = _documented_verbs(cli_mdx)

    missing_from_docs = dispatcher - documented
    assert not missing_from_docs, (
        "dispatcher verb(s) missing from docs: "
        + ", ".join(f"'{v}'" for v in sorted(missing_from_docs))
        + f"\n  dispatcher has: {sorted(dispatcher)}"
        + f"\n  docs have:      {sorted(documented)}"
    )


def test_documented_verbs_all_in_dispatcher(cli_mdx: pathlib.Path) -> None:
    """Every verb documented in cli.mdx must be handled by the dispatcher.

    If a verb is added to cli.mdx that the dispatcher doesn't handle,
    this test fails: "documented verb 'X' not found in dispatcher".
    """
    dispatcher = _dispatcher_verbs()
    documented = _documented_verbs(cli_mdx)

    invented = documented - dispatcher
    assert not invented, (
        "documented verb(s) not found in dispatcher: "
        + ", ".join(f"'{v}'" for v in sorted(invented))
        + f"\n  docs have:      {sorted(documented)}"
        + f"\n  dispatcher has: {sorted(dispatcher)}"
    )
