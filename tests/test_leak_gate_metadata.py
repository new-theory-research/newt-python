"""The metadata gate's matcher, pinned on branch-name and pull-request-title shapes.

scripts/leak_gate_metadata.py carries its own ``--self-test``, and that is what CI runs
ahead of the real check. This is the unit-level half, and it exists for the two properties
that decide whether the check is worth having at all:

  - punctuation launders nothing. The branch that started this (eight of them, one per
    landed card) was ``<lane>-<number>-<slug>``; the same lane could as easily be written
    with an underscore, or spaced out in a pull-request title. All of those must trip.
  - ordinary work stays quiet. A check that fires on ``fix-models-list`` gets switched off
    within a week, and then it protects nothing.

Every token here is synthetic and its digest is injected by the test. No real denylisted
term may be written into this repo — that is the thing the digest files exist to prevent.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_gate():
    """Load the script by path. ``scripts/`` is not an importable package, and adding it
    to the dependency-scanner's world to make a plain ``import`` work would be a bigger
    change than the test is worth."""
    spec = importlib.util.spec_from_file_location(
        "leak_gate_metadata", _SCRIPTS / "leak_gate_metadata.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()

_TERM = "internallane" + "fixture001"
_DIGESTS = {gate.hashed(_TERM)}


@pytest.mark.parametrize(
    "branch",
    [
        f"{_TERM}-041-tighten-the-retry-path",  # the shape the real branches had
        "internallane-fixture001-041-tighten-the-retry-path",  # hyphen between the halves
        "internallane_fixture001_041",  # underscore
        f"prefix/{_TERM}",  # namespaced under a slash
        _TERM.upper(),  # shouted
    ],
)
def test_denied_term_is_caught_however_the_branch_punctuates_it(branch):
    findings = gate.scan_branch(branch, _DIGESTS)
    assert findings, f"{branch!r} slipped through — a separator laundered the term"
    assert findings[0][1] == gate.BRANCH_RULE


@pytest.mark.parametrize(
    "branch",
    [
        "main",
        "fix-models-list",  # `newt models` is a shipped verb; over-firing here kills the check
        "leak-gate-metadata-pass",
        "director-iter-record-teleop-bench",
        "internal-lane-notes",  # the halves apart, not adjacent — not the term
    ],
)
def test_ordinary_branch_names_are_left_alone(branch):
    assert gate.scan_branch(branch, _DIGESTS) == []


def test_pull_request_title_is_scanned_and_the_finding_names_the_pull_request():
    """The title is the public part, and the finding has to be actionable: whoever reads
    CI needs the number and the URL to retitle it. Nothing is disclosed by printing them —
    an open pull request is already published."""
    pull = {
        "number": 41,
        "title": f"{_TERM} 041: tighten the retry path",
        "url": "https://example.invalid/pull/41",
    }
    findings = gate.scan_pull_request(pull, _DIGESTS)
    assert len(findings) == 1
    location, rule_id, detail = findings[0]
    assert rule_id == gate.PR_TITLE_RULE
    assert "41" in location and "https://example.invalid/pull/41" in location
    assert _TERM in detail


def test_metadata_is_checked_against_the_content_denylist_too():
    """A branch named after internal infrastructure is as public as one named after an
    internal lane, so the union — not the metadata list alone — is what metadata is
    matched against. If this ever narrows to one file, the other surface goes unguarded."""
    union = gate.load_digests()
    assert gate.load_terms(gate.CONTENT_TERMS_FILE) <= union
    assert gate.load_terms(gate.METADATA_TERMS_FILE) <= union


def test_the_two_denylists_are_kept_apart():
    """The metadata terms are held in their own file because they must not fire on file
    contents — tracked comments cite those lanes' card IDs on main today, and the content
    gate would fail on them. Copying one list into the other is the way that distinction
    gets lost, so it is asserted rather than remembered."""
    content = gate.load_terms(gate.CONTENT_TERMS_FILE)
    metadata = gate.load_terms(gate.METADATA_TERMS_FILE)
    assert not (content & metadata), (
        "a digest appears in both denylists — either a metadata term was pasted into the "
        "content list (which will fail the content gate on main) or the reverse"
    )


def test_the_gate_module_names_no_denied_term():
    """The check's own source is public. If a real term were ever spelled out in it — in a
    fixture, a docstring, an example — the digest file would be protecting nothing."""
    union = gate.load_digests()
    source = (_SCRIPTS / "leak_gate_metadata.py").read_text()
    hits = [t for line in source.splitlines() for t in gate.denied_tokens(line, union)]
    assert not hits, "the metadata gate's own source names a denied term"
