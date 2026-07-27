#!/usr/bin/env python3
"""Leak gate — this repo is public, so nothing internal may live in it.

Scans every tracked file for two classes of thing that must never appear here:

  1. Structural patterns — concrete internal infrastructure identifiers (a specific
     serving hostname, a private repo reference). Written as plaintext regexes because
     they describe a *shape*, not a secret.

  2. Denylisted terms — internal project/model jargon. Stored as sha256 digests in
     scripts/leak_terms.sha256, never as plaintext. A public file listing our internal
     terms would be a map of what to grep our history for. This is obscurity, not
     secrecy: short tokens are brute-forceable given the digests. The point is that the
     public repo does not *advertise* the terms.

The plaintext term list and the regeneration command live in the private portal repo at
wiki/operating-docs/leak-gate-terms.txt.

What this gate deliberately does NOT flag: vendor names on their own. The SDK already
talks to a Railway-hosted API in public code, so "we use Modal / GCS / Railway" is not
the secret — *which app, which bucket, which workspace* is. A bare `modal.run` mention
passes; `<workspace>--<app>.modal.run` does not.

Usage:
    python scripts/leak_gate.py              # scan tracked files, exit 1 on any finding
    python scripts/leak_gate.py --self-test  # prove the gate can still fail

Escape hatch: put `leak-gate-allow` in a comment on the offending line, and say why in
the PR. Use it for lines that assert we *don't* leak, not to wave through real residue.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOW_PRAGMA = "leak-gate-allow"
TERMS_FILE = Path(__file__).with_name("leak_terms.sha256")

# Repos under our GitHub org that are genuinely public. Anything else the org owns is
# private, and naming it here tells outsiders what to go knock on. Adding to this set is
# a deliberate act: confirm the repo is public first.
PUBLIC_REPOS = {"newt-python", "lerobot"}

# Each rule owns its own message — two different causes must never print the same string.
STRUCTURAL_RULES = [
    (
        "modal-app-host",
        re.compile(r"[A-Za-z0-9-]+--[A-Za-z0-9-]+\.modal\.run"),
        "a concrete internal serving hostname (workspace + app name)",
    ),
    (
        "internal-workspace-host",
        re.compile(r"\b[A-Za-z0-9-]+\.workers\.dev\b"),
        "an NT-only internal workspace host",
    ),
]

_ORG_REF = re.compile(r"new-theory-research/([A-Za-z0-9_.-]+)")
_TOKEN = re.compile(r"[a-z0-9]+")


def load_terms() -> set[str]:
    if not TERMS_FILE.exists():
        sys.exit(f"leak gate: missing {TERMS_FILE} — the gate cannot run without its denylist")
    digests = {
        line.strip()
        for line in TERMS_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    if not digests:
        sys.exit(f"leak gate: {TERMS_FILE} has no digests — refusing to run a gate that cannot fail")
    return digests


def hashed(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def denied_tokens(line: str, digests: set[str]) -> list[str]:
    """Tokens on this line whose digest is denylisted.

    Adjacent tokens are also checked joined, so a separator launders nothing: if `foobar`
    is denied, then `foo-bar`, `foo_bar` and `Foo Bar` all trip too.
    """
    tokens = _TOKEN.findall(line.lower())
    candidates = tokens + [a + b for a, b in itertools.pairwise(tokens)]
    return [t for t in candidates if hashed(t) in digests]


def scan_line(path: str, lineno: int, line: str, digests: set[str]) -> list[tuple[str, str, str]]:
    if ALLOW_PRAGMA in line:
        return []
    findings = []
    for rule_id, pattern, why in STRUCTURAL_RULES:
        for match in pattern.findall(line):
            findings.append((f"{path}:{lineno}", rule_id, f"{match!r} is {why}"))
    for match in _ORG_REF.findall(line):
        repo = match.removesuffix(".git").rstrip(".,)")
        if repo not in PUBLIC_REPOS:
            findings.append(
                (
                    f"{path}:{lineno}",
                    "private-repo-ref",
                    f"new-theory-research/{repo} is not on the public-repo allowlist",
                )
            )
    for token in denied_tokens(line, digests):
        findings.append(
            (f"{path}:{lineno}", "internal-term", f"{token!r} is on the internal-term denylist")
        )
    return findings


def scan_paths(paths: list[Path], root: Path, digests: set[str]) -> list[tuple[str, str, str]]:
    findings = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable — nothing greppable in it
        rel = str(path.relative_to(root))
        for lineno, line in enumerate(text.splitlines(), start=1):
            findings.extend(scan_line(rel, lineno, line, digests))
    return findings


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    return [root / name for name in out.split("\0") if name and (root / name).is_file()]


def report(findings: list[tuple[str, str, str]]) -> None:
    print(f"LEAK GATE FAILED — {len(findings)} finding(s) in a PUBLIC repo\n")
    for location, rule_id, detail in findings:
        print(f"  {location}")
        print(f"    rule:  {rule_id}")
        print(f"    found: {detail}")
    print(
        "\nWhose problem: ours, in this change. This repo is world-readable, and everything"
        "\nabove names internal infrastructure or internal jargon."
        "\n\nWhat to do next:"
        "\n  - If the artifact only runs with internal creds, it belongs in newt-python-internal"
        "\n    (or nt-runway if it drives internal serving) — not here."
        "\n  - If it is example/test text, replace the identifier with a placeholder."
        f"\n  - If it is genuinely a false positive, add `{ALLOW_PRAGMA}` in a comment on that"
        "\n    line and justify it in the PR."
    )


def self_test() -> int:
    """Prove the gate can still fail. A gate nobody can trip is not a gate."""
    digests = load_terms()
    # Every fixture below is assembled at runtime, so no literal in this file matches a
    # rule this file scans for. The gate scans its own source like any other tracked
    # file — no self-exemption, because a scanner with a blind spot at its own path is
    # exactly where residue would hide.
    canary = "leakgatecanary" + "0" * 3
    modal, workers, org = "modal" + ".run", "workers" + ".dev", "new-theory-" + "research"
    cases = {
        "modal-app-host": f"url = 'wss://someworkspace--someapp-serve.{modal}/stream'",
        "internal-workspace-host": f"docs at https://someplace.newtheory.{workers}/x",
        "private-repo-ref": f"pip install git+ssh://git@github.com/{org}/some-private-repo.git",
        "internal-term": f"# {canary} appears in a comment with nothing incriminating near it",
    }
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rule_id, line in cases.items():
            probe = root / f"{rule_id}.txt"
            probe.write_text(line + "\n")
            hits = {f[1] for f in scan_paths([probe], root, digests)}
            if rule_id not in hits:
                failures.append(f"rule {rule_id!r} did not fire on its own fixture: {line!r}")
            probe.unlink()

        allowed = root / "allowed.txt"
        allowed.write_text(f"assert 'x--y.{modal}' not in url  # {ALLOW_PRAGMA}\n")
        if scan_paths([allowed], root, digests):
            failures.append(f"the {ALLOW_PRAGMA} pragma did not suppress a finding")

        clean = root / "clean.txt"
        clean.write_text(f"a wss://host/stream url and github.com/{org}/newt-python\n")
        if scan_paths([clean], root, digests):
            failures.append("clean fixture produced a finding — the gate is over-firing")

    if failures:
        print("LEAK GATE SELF-TEST FAILED — the gate is no longer able to catch what it claims:\n")
        for line in failures:
            print(f"  - {line}")
        print(
            "\nWhose problem: ours — someone weakened a rule or emptied the denylist."
            "\nNext: restore the rule, or if the change was deliberate, update the fixtures"
            "\nin self_test() in the same commit so the gate stays honest."
        )
        return 1
    print(f"SELF-TEST PASS: {len(cases)} rules fire, the pragma suppresses, clean text is clean")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    findings = scan_paths(tracked_files(root), root, load_terms())
    if findings:
        report(findings)
        return 1
    print("LEAK GATE PASS: no internal identifiers or denylisted terms in tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
