#!/usr/bin/env python3
"""Metadata leak gate — branch names and pull-request titles are public surfaces too.

scripts/leak_gate.py reads what is *inside* tracked files. It never looked at the
repository's own metadata, and metadata is every bit as world-readable: anyone can list
this repo's branches and open pull requests without cloning a line of it. Eight branches
named after an internal work lane accumulated here, one per landed card, before anyone
noticed — deleted 2026-08-06. This check closes that surface.

Two denylists, on purpose:

  1. scripts/leak_terms.sha256 — the content list, shared with leak_gate.py. A branch
     named after internal infrastructure is as public as one named after an internal
     lane, so metadata is checked against this list too.

  2. scripts/leak_metadata_terms.sha256 — terms that may fire on metadata ONLY. They name
     internal work lanes, and tracked comments on main cite those lanes' card IDs as
     provenance for why a behaviour is the way it is. That is legitimate, and there is a
     lot of it, so folding these digests into the content list would fail the content
     gate on main immediately. Different surface, different denylist.

Why the digests are committed here rather than fetched at runtime from the private list:
a fetch needs a secret, and secrets are withheld from workflow runs on pull requests from
forks. The check would skip in exactly the case where an outside contributor might
introduce something — and a gate that silently skips is worse than no gate.

The plaintext terms and the regeneration command live in the private portal repo at
wiki/operating-docs/leak-gate-metadata-terms.txt.

Usage:
    python scripts/leak_gate_metadata.py              # scan live branches + open PRs
    python scripts/leak_gate_metadata.py --self-test  # prove the check can still fail
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

# scripts/ is a directory of scripts, not a package — put it on the path so the sibling
# gate's matcher can be imported instead of copied. Two copies of a tokenizer drift, and
# the drift would be silent in exactly the direction that matters.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from leak_gate import denied_tokens, hashed, load_terms, tracked_files  # noqa: E402

CONTENT_TERMS_FILE = Path(__file__).with_name("leak_terms.sha256")
METADATA_TERMS_FILE = Path(__file__).with_name("leak_metadata_terms.sha256")

GH_PR_FIELDS = "number,title,url"

# Each rule owns its own message — two different causes must never print the same string.
BRANCH_RULE = "branch-name"
PR_TITLE_RULE = "pr-title"


def load_digests() -> set[str]:
    """Content terms ∪ metadata terms — what metadata is checked against.

    The union runs one way only. leak_gate.py loads the content list alone, because the
    metadata terms appear legitimately in tracked comments and would fail it on main.
    """
    return load_terms(CONTENT_TERMS_FILE) | load_terms(METADATA_TERMS_FILE)


def repo_root() -> Path:
    return Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )


def remote_branches() -> list[str]:
    """Every branch name published on origin, as a stranger browsing the repo sees them."""
    proc = subprocess.run(
        ["git", "ls-remote", "--heads", "origin"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        sys.exit(
            "metadata leak gate: could not enumerate remote branches — "
            f"`git ls-remote --heads origin` exited {proc.returncode}.\n"
            f"  git said: {proc.stderr.strip() or '(nothing on stderr)'}\n"
            "Whose problem: ours — the runner or the local checkout, not the author of the\n"
            "change under review. This exits non-zero rather than passing on an empty list,\n"
            "because an unread surface is how the branch names got here in the first place.\n"
            "Next: confirm an `origin` remote exists and is reachable (actions/checkout\n"
            "leaves one configured), then re-run."
        )
    names = []
    for line in proc.stdout.splitlines():
        _, _, ref = line.partition("\t")
        if ref.startswith("refs/heads/"):
            names.append(ref.removeprefix("refs/heads/"))
    return names


def open_pull_requests() -> list[dict]:
    """Open pull requests on this repo, titles included.

    Three ways this can go wrong and three different sentences, because "gh is missing"
    and "gh ran and refused" want different fixes from different people.
    """
    if shutil.which("gh") is None:
        sys.exit(
            "metadata leak gate: `gh` is not on PATH, so open pull-request titles could not\n"
            "be read. Whose problem: ours — the environment running this check, not the\n"
            "change under review. The check refuses to report on branches alone and call\n"
            "that a pass.\n"
            "Next: install the GitHub CLI (it is preinstalled on GitHub-hosted runners), or\n"
            "run this from an environment that has it."
        )
    proc = subprocess.run(
        ["gh", "pr", "list", "--state", "open", "--json", GH_PR_FIELDS],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(
            "metadata leak gate: the open-pull-request query failed — "
            f"`gh pr list` exited {proc.returncode}.\n"
            f"  gh said: {proc.stderr.strip() or '(nothing on stderr)'}\n"
            "Whose problem: ours — most often a token without `pull-requests: read`, or no\n"
            "GH_TOKEN in the environment at all.\n"
            "Next: set GH_TOKEN and grant the job `permissions: pull-requests: read`, then\n"
            "re-run. Do not drop this step to get green — pull-request titles are published\n"
            "the moment the pull request opens."
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        sys.exit(
            "metadata leak gate: `gh pr list` succeeded but its output was not the JSON this\n"
            f"check knows how to read ({exc}).\n"
            "Whose problem: ours — a gh version whose `--json` shape moved under us.\n"
            f"Next: run `gh pr list --state open --json {GH_PR_FIELDS}` by hand, compare the\n"
            "shape, and update this parser in the same commit."
        )


# Findings name the offending branch or pull request outright — its number, its title, its
# URL. By the time this check sees any of it, the name is already published on a
# world-readable repo, so printing it in CI output discloses nothing that `git ls-remote`
# would not. Do not "fix" this into a redacted form later: a finding nobody can act on is a
# finding nobody acts on.
def scan_branch(name: str, digests: set[str]) -> list[tuple[str, str, str]]:
    return [
        (f"branch origin/{name}", BRANCH_RULE, f"the branch name contains {token!r}")
        for token in denied_tokens(name, digests)
    ]


def scan_pull_request(pull: dict, digests: set[str]) -> list[tuple[str, str, str]]:
    title = pull.get("title", "")
    return [
        (
            f"pull request #{pull.get('number')} — {pull.get('url')}",
            PR_TITLE_RULE,
            f"the title {title!r} contains {token!r}",
        )
        for token in denied_tokens(title, digests)
    ]


def report(findings: list[tuple[str, str, str]]) -> None:
    print(f"METADATA LEAK GATE FAILED — {len(findings)} finding(s) on a PUBLIC repo's metadata\n")
    for location, rule_id, detail in findings:
        print(f"  {location}")
        print(f"    rule:  {rule_id}")
        print(f"    found: {detail}")
    print(
        "\nWhose problem: ours. Nothing above is inside a file — these are branch names and"
        "\npull-request titles, which GitHub publishes to anyone who asks, and each one names"
        "\nan internal lane or a piece of internal infrastructure."
        "\n\nWhat to do next:"
        "\n  - A branch that has landed: delete it — `git push origin --delete <branch>`."
        "\n  - A branch still in use: rename it — `git push origin <old>:<new>` then"
        "\n    `git push origin --delete <old>`, and re-point any open pull request."
        "\n  - A pull request: retitle it — `gh pr edit <number> --title '<new title>'`."
        "\n\nThe convention this enforces: branch names and pull-request titles on this repo"
        "\ncarry no internal lane names. Use the bare card number or a plain description of"
        "\nthe change, and let the private card link out to the pull request instead."
    )


def _committed_term_in_tree(digests: set[str]) -> str | None:
    """A really-denied token, discovered from the tree rather than written down here.

    Nothing in this file may name a denied term — writing one here, split or whole, would
    publish the very thing the digest file exists to keep unpublished. So the strongest
    available proof that a *committed* digest still matches real text is to go and find
    one: the metadata terms name lanes whose card IDs are cited in tracked comments on
    main. Usually one is there; not always. When the tree stops carrying one this returns
    None, the self-test says so out loud, and the synthetic fixture still proves the
    mechanism. The returned token is never printed.
    """
    for path in tracked_files(repo_root()):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line in text.splitlines():
            found = denied_tokens(line, digests)
            if found:
                return found[0]
    return None


def self_test() -> int:
    """Prove the check can still fail. A gate nobody can trip is not a gate."""
    digests = load_digests()  # exits if either committed list is missing or empty
    failures = []

    # The fixture token is synthetic and assembled at runtime, for two reasons. leak_gate.py
    # scans this file like any other tracked file — and, more to the point, a real denied
    # term spelled out here in any form would defeat the whole arrangement.
    head, tail = "internallane", "fixture000"
    fixture = head + tail
    probe = digests | {hashed(fixture)}

    # The shape a lane branch actually had: <lane>-<card number>-<slug>.
    branch = f"{fixture}-041-tighten-the-retry-path"
    if BRANCH_RULE not in {f[1] for f in scan_branch(branch, probe)}:
        failures.append(f"rule {BRANCH_RULE!r} did not fire on its own fixture: {branch!r}")

    title = f"{fixture}-041: tighten the retry path"
    pull = {"number": 41, "title": title, "url": "https://example.invalid/pull/41"}
    if PR_TITLE_RULE not in {f[1] for f in scan_pull_request(pull, probe)}:
        failures.append(f"rule {PR_TITLE_RULE!r} did not fire on its own fixture: {title!r}")

    # Separators launder nothing: the adjacent-token join is what makes a hyphenated or
    # spaced lane name reduce to the same token as the joined one.
    hyphenated = f"{head}-{tail}-041-tighten-the-retry-path"
    if not scan_branch(hyphenated, probe):
        failures.append(f"a separator hid a denied term in a branch name: {hyphenated!r}")

    # Against the real committed lists, ordinary work must stay quiet — including this
    # change's own branch and title, which is the check dogfooding its own convention.
    clean_branch = "leak-gate-metadata-pass"
    clean_pull = {
        "number": 78,
        "title": "add a metadata pass to the leak gate",
        "url": "https://example.invalid/pull/78",
    }
    if scan_branch(clean_branch, digests) or scan_pull_request(clean_pull, digests):
        failures.append("a clean branch name or title produced a finding — the check over-fires")

    live = _committed_term_in_tree(digests)
    if live is not None:
        if BRANCH_RULE not in {f[1] for f in scan_branch(f"{live}-041-a-landed-card", digests)}:
            failures.append(
                "a term from the committed denylist did not fire on a branch named after it"
            )

    if failures:
        print("METADATA LEAK GATE SELF-TEST FAILED — the check can no longer catch what it claims:\n")
        for line in failures:
            print(f"  - {line}")
        print(
            "\nWhose problem: ours — someone weakened the matcher or emptied a denylist."
            "\nNext: restore it, or if the change was deliberate, update the fixtures in"
            "\nself_test() in the same commit so the check stays honest."
        )
        return 1
    proof = (
        "a committed denylist term also fires (found in the tree, not named here)"
        if live is not None
        else "NOTE: no committed term is present in the tree today, so only the synthetic "
        "fixture was exercised"
    )
    print(f"SELF-TEST PASS: both rules fire, separators do not launder, clean names stay clean; {proof}")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    digests = load_digests()
    branches = remote_branches()
    pulls = open_pull_requests()
    findings = []
    for name in branches:
        findings.extend(scan_branch(name, digests))
    for pull in pulls:
        findings.extend(scan_pull_request(pull, digests))
    if findings:
        report(findings)
        return 1
    print(
        f"METADATA LEAK GATE PASS: {len(branches)} remote branch(es) and {len(pulls)} open "
        "pull request(s), no internal names among them"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
