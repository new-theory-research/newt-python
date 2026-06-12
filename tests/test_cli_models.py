"""Offline unit tests for `newt models` CLI command.

Tests cover the developer-visible contract WITHOUT touching the network.
Each test encodes why the behavior matters, not just what it does.
"""
from __future__ import annotations

import io
import json
import sys

import pytest

import newt
from newt._client.robot import AuthError, RegistryUnavailable
from newt._cli.models import cmd_models, _render_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: list[str], monkeypatch, models_return=None, side_effect=None):
    """Run cmd_models with a mocked newt.list_models, capturing stdout/stderr."""
    captured_out = io.StringIO()
    captured_err = io.StringIO()

    if side_effect is not None:
        def fake_list_models(*a, **kw):
            raise side_effect
    else:
        def fake_list_models(*a, **kw):
            return models_return if models_return is not None else []

    monkeypatch.setattr(newt, "list_models", fake_list_models)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")
    # Ensure stdout/stderr are captured
    monkeypatch.setattr(sys, "stdout", captured_out)
    monkeypatch.setattr(sys, "stderr", captured_err)

    exit_code = cmd_models(args)
    return exit_code, captured_out.getvalue(), captured_err.getvalue()


# Real 7-model registry payload (structure mirrors the live API, which mirrors
# model_registry.yaml). Post brief-253a: customer-facing `nt0` tags lead; the
# `nt0-fp3` legacy aliases are retained AFTER them so stale installs keep
# resolving (T1) but never surface in human output.
_REAL_7_MODELS = [
    {
        "uid": "ft_base_nt0fp3",
        "tags": ["nt0", "nt0-fp3"],
        "type": "base",
        "base": None,
        "contract": {
            "action_axes": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]
        },
    },
    {
        "uid": "ft_4hn40z6a",
        "tags": ["nt0-pour-coffee-beans", "nt0-fp3-pour-coffee-beans", "pour-coffee-beans"],
        "type": "fine_tune",
        "base": "ft_base_nt0fp3",
        "contract": {"action_axes": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]},
    },
    {
        "uid": "ft_6d0cfd51_e63fbb",
        "tags": ["nt0-clean-table", "nt0-fp3-clean-table"],
        "type": "fine_tune",
        "base": "ft_base_nt0fp3",
        "contract": {"action_axes": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]},
    },
    {
        "uid": "ft_6d0cfd51_c9d8ba",
        "tags": ["nt0-pour-coffee-beans-wm", "nt0-fp3-pour-coffee-beans-wm"],
        "type": "fine_tune",
        "base": "ft_base_nt0fp3",
        "contract": {"action_axes": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]},
    },
    {
        "uid": "ft_base_molmoact2",
        "tags": ["molmoact2"],
        "type": "base",
        "base": None,
        "contract": None,
    },
    {
        "uid": "ft_molmoact2_yam",
        "tags": ["molmoact2-yam", "molmoact2-bimanual-yam"],
        "type": "fine_tune",
        "base": "ft_base_molmoact2",
        "contract": {"state_shape": [14]},
    },
    {
        "uid": "ft_base_pi05_aloha",
        "tags": ["pi05_aloha"],
        "type": "base",
        "base": None,
        "contract": {"state_shape": [14]},
    },
]


# ---------------------------------------------------------------------------
# Grouped layout: A-option design
# ---------------------------------------------------------------------------

def test_grouped_layout_three_families():
    """The real 7-model registry renders as three family groups.

    Three distinct base models → three groups, separated by blank lines.
    A developer scanning the output sees at a glance which fine-tunes belong
    to which base — something the old flat dump made impossible.
    """
    output = _render_models(_REAL_7_MODELS)
    groups = [g.strip() for g in output.split("\n\n") if g.strip()]
    assert len(groups) == 3, (
        f"expected 3 family groups separated by blank lines, got {len(groups)}: {groups!r}"
    )


def test_grouped_layout_base_names_lead():
    """Base display names lead each family group — nt0 / molmoact2 / pi05_aloha.

    The human name (primary tag) should appear first on the base line, not the
    opaque uid. A developer picking a model should find it by name. Post-253a the
    customer-facing name is `nt0`, never the `nt0-fp3` legacy alias.
    """
    output = _render_models(_REAL_7_MODELS)
    lines = output.splitlines()
    # Base lines are the non-indented ones
    base_lines = [l for l in lines if l and not l.startswith("    ")]
    names = [l.split()[0] for l in base_lines]
    assert names == ["nt0", "molmoact2", "pi05_aloha"], (
        f"base display names must lead each group in order: {names!r}"
    )


def test_grouped_layout_fine_tunes_indented():
    """Fine-tunes are indented 4 spaces under their base — not at top level.

    Indentation communicates hierarchy without tree glyphs. The developer
    immediately reads: these tasks run on this base.
    """
    output = _render_models(_REAL_7_MODELS)
    lines = output.splitlines()
    ft_lines = [l for l in lines if l.startswith("    ")]
    assert len(ft_lines) == 4, (
        f"expected 4 indented fine-tune lines, got {len(ft_lines)}: {ft_lines!r}"
    )


def test_axes_appear_once_on_nt0_base_only():
    """Action axes appear exactly once: on the nt0 base line.

    Fine-tunes inherit axes — repeating them on every fine-tune line was noise.
    molmoact2 and pi05_aloha have no action_axes so they get none. This test
    exists because the old flat renderer printed inherited axes on every row.
    """
    output = _render_models(_REAL_7_MODELS)
    lines = output.splitlines()

    # Count lines that contain "axes"
    axes_lines = [l for l in lines if "axes" in l]
    assert len(axes_lines) == 1, (
        f"axes fragment must appear exactly once, got {len(axes_lines)}: {axes_lines!r}"
    )

    # That one line must be the nt0 base line (display name leads it)
    assert axes_lines[0].split()[0] == "nt0", (
        f"axes line must be the nt0 base, got: {axes_lines[0]!r}"
    )

    # All axes present on that line
    for ax in ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]:
        assert ax in axes_lines[0], f"axis {ax!r} missing from base line: {axes_lines[0]!r}"


def test_axes_absent_on_contract_less_bases():
    """molmoact2 (null contract) and pi05_aloha (no action_axes) get no axes fragment.

    A base with no action_axes contract is not missing data — axes just aren't
    part of its schema. The renderer must not emit an empty or confusing axes field.
    """
    output = _render_models(_REAL_7_MODELS)
    groups = [g.strip() for g in output.split("\n\n") if g.strip()]
    molmo_group = next(g for g in groups if "molmoact2" in g.splitlines()[0])
    pi_group = next(g for g in groups if "pi05_aloha" in g.splitlines()[0])

    assert "axes" not in molmo_group, (
        f"molmoact2 has no action_axes — must not emit axes fragment: {molmo_group!r}"
    )
    assert "axes" not in pi_group, (
        f"pi05_aloha has no action_axes — must not emit axes fragment: {pi_group!r}"
    )


def test_name_derivation_task_tags():
    """Fine-tune task names derive from the PRIMARY tag, prefix stripped — never an alias.

    Each label is tags[0] with the base prefix removed:
    pour-coffee-beans — nt0-pour-coffee-beans → strip nt0-.
    clean-table — nt0-clean-table → strip nt0- (the legacy alias nt0-fp3-clean-table
        sits at tags[1] and must NOT win; that was the fp3-leak this brief kills).
    pour-coffee-beans-wm — nt0-pour-coffee-beans-wm → strip nt0-.
    yam — molmoact2-yam → strip molmoact2- (primary tag, not the longer alias).
    """
    output = _render_models(_REAL_7_MODELS)
    lines = output.splitlines()
    ft_lines = [l.strip() for l in lines if l.startswith("    ")]
    task_names = [l.split()[0] for l in ft_lines]

    assert "pour-coffee-beans" in task_names, f"task names: {task_names!r}"
    assert "clean-table" in task_names, f"task names: {task_names!r}"
    assert "pour-coffee-beans-wm" in task_names, f"task names: {task_names!r}"
    assert "yam" in task_names, f"task names: {task_names!r}"


def test_fine_tune_uids_visible():
    """Fine-tune uids appear on their lines (secondary, after task name).

    Developers reference models by uid in code. The name is scannable; the uid
    is the actual handle.
    """
    output = _render_models(_REAL_7_MODELS)
    for uid in ["ft_4hn40z6a", "ft_6d0cfd51_e63fbb", "ft_6d0cfd51_c9d8ba", "ft_molmoact2_yam"]:
        assert uid in output, f"fine-tune uid {uid!r} must appear in output"


def test_uid_alignment_within_family():
    """Fine-tune uids are column-aligned within a family.

    Same leading whitespace (4 spaces) + task name padded to column width.
    This makes the uid column visually scannable.
    """
    output = _render_models(_REAL_7_MODELS)
    lines = output.splitlines()
    # Get the nt0-fp3 family's fine-tune lines
    nt0_fts = []
    in_nt0 = False
    for line in lines:
        if line and not line.startswith("    "):
            in_nt0 = line.split()[0] == "nt0"
        elif in_nt0 and line.startswith("    "):
            nt0_fts.append(line)

    # Find the position where uid starts on each line (after padding)
    uid_positions = []
    for line in nt0_fts:
        # uid starts after the padded task name + 2-space separator
        parts = line.split("  ")
        # first part is "    taskname"
        uid_start = len(parts[0]) + 2  # 2 spaces separator
        uid_positions.append(uid_start)

    assert len(set(uid_positions)) == 1, (
        f"uid column must be aligned across family fine-tunes, positions: {uid_positions!r}"
    )


def test_no_box_drawing_or_tree_glyphs():
    """No tree glyphs, box-drawing characters, or decorative unicode in output.

    Indentation only — clean, pipeable, grep-friendly.
    """
    output = _render_models(_REAL_7_MODELS)
    forbidden = ["├", "└", "─", "│", "┌", "┐", "┘", "└", "╭", "╰", "►", "▷", "→"]
    for char in forbidden:
        assert char not in output, f"found forbidden glyph {char!r} in output"


def test_no_line_exceeds_100_chars():
    """No output line exceeds 100 characters on the real dataset.

    The old flat dump wrapped badly on small terminals because it repeated long
    axes lists on every fine-tune row. The new layout must stay under 100 chars.
    """
    output = _render_models(_REAL_7_MODELS)
    for line in output.splitlines():
        assert len(line) <= 100, f"line exceeds 100 chars ({len(line)}): {line!r}"


# ---------------------------------------------------------------------------
# Orphan fine-tunes
# ---------------------------------------------------------------------------

def test_orphan_fine_tune_renders_without_crashing():
    """A fine-tune whose base isn't in the payload renders without crashing.

    An orphan appears at top level, clearly labeled with its base reference.
    This guards against partial registry payloads or future splits where a
    base is temporarily absent.
    """
    models = [
        {
            "uid": "ft_orphan_abc123",
            "tags": ["orphan-task"],
            "type": "fine_tune",
            "base": "ft_base_missing",
            "contract": None,
        }
    ]
    # Must not raise
    output = _render_models(models)
    assert "ft_orphan_abc123" in output, "orphan uid must appear"
    assert "ft_base_missing" in output, "orphan base reference must appear"


def test_orphan_renders_at_top_level_before_families():
    """Orphan fine-tunes render before family groups, at top level (no indent)."""
    models = [
        {
            "uid": "ft_orphan_abc123",
            "tags": ["orphan-task"],
            "type": "fine_tune",
            "base": "ft_base_missing",
            "contract": None,
        },
        {
            "uid": "ft_base_nt0fp3",
            "tags": ["nt0", "nt0-fp3"],
            "type": "base",
            "base": None,
            "contract": {"action_axes": ["x", "y", "z"]},
        },
    ]
    output = _render_models(models)
    lines = output.splitlines()
    non_empty = [l for l in lines if l.strip()]
    # Orphan must come before any family
    orphan_idx = next(i for i, l in enumerate(non_empty) if "ft_orphan_abc123" in l)
    base_idx = next(i for i, l in enumerate(non_empty) if l.split()[0] == "nt0")
    assert orphan_idx < base_idx, "orphan must appear before family groups"
    # Orphan line is not indented
    orphan_line = non_empty[orphan_idx]
    assert not orphan_line.startswith("    "), f"orphan must not be indented: {orphan_line!r}"


# ---------------------------------------------------------------------------
# Golden: developer who just logged in types `newt models` and sees what
# their key can drive — catalog renders compactly, exit 0.
# ---------------------------------------------------------------------------

def test_models_renders_catalog(monkeypatch):
    """A developer who just logged in types `newt models` and sees the catalog.

    The command must exit 0 and print each model's UID on its line.
    The developer can scan the list to find the model they want.
    """
    exit_code, out, err = _run([], monkeypatch, models_return=_REAL_7_MODELS)

    assert exit_code == 0, f"expected exit 0, got {exit_code}; stderr={err!r}"
    assert "ft_base_nt0fp3" in out
    assert "ft_base_molmoact2" in out
    assert err == ""


def test_models_renders_axes_from_contract(monkeypatch):
    """A developer scanning the catalog sees action axes on the base model line.

    Axes appear exactly once (on the base line), not repeated on fine-tunes.
    Bases without action_axes (molmoact2, pi05_aloha) have no axes fragment.
    This test exists because the old renderer once read a nonexistent top-level
    'axes' key and the column silently never printed.
    """
    exit_code, out, _ = _run([], monkeypatch, models_return=_REAL_7_MODELS)

    assert exit_code == 0
    lines = out.splitlines()
    nt0_base_line = next(l for l in lines if l.split() and l.split()[0] == "nt0")

    assert "axes" in nt0_base_line, f"axes must render on the nt0 base line: {nt0_base_line!r}"
    assert "gripper" in nt0_base_line, f"axis labels must appear: {nt0_base_line!r}"

    # Fine-tune lines must NOT repeat axes
    ft_lines = [l for l in lines if l.startswith("    ")]
    for line in ft_lines:
        assert "axes" not in line, f"fine-tune must not repeat axes: {line!r}"

    # molmoact2 and pi05_aloha base lines have no axes
    molmo_line = next(l for l in lines if "molmoact2" in l and not l.startswith("    "))
    pi_line = next(l for l in lines if "pi05_aloha" in l and not l.startswith("    "))
    assert "axes" not in molmo_line
    assert "axes" not in pi_line


# ---------------------------------------------------------------------------
# Golden: `--json` emits a valid JSON array — byte-identical to raw API response
# ---------------------------------------------------------------------------

def test_models_json_flag_emits_valid_array(monkeypatch):
    """`--json` emits a JSON array so agents and scripts can parse the catalog.

    Every model dict must be present. The output must parse as a list.
    """
    exit_code, out, err = _run(["--json"], monkeypatch, models_return=_REAL_7_MODELS)

    assert exit_code == 0, f"expected exit 0; stderr={err!r}"
    parsed = json.loads(out)
    assert isinstance(parsed, list), f"expected JSON array, got {type(parsed)}"
    assert len(parsed) == len(_REAL_7_MODELS)
    uids = [m["uid"] for m in parsed]
    assert "ft_base_nt0fp3" in uids
    assert "ft_base_molmoact2" in uids


def test_models_json_unchanged_by_renderer(monkeypatch):
    """--json output is byte-identical to json.dumps(models).

    The human renderer must not touch the JSON path. Scripts that pipe
    `newt models --json` to jq cannot tolerate any mutation.
    """
    exit_code, out, _ = _run(["--json"], monkeypatch, models_return=_REAL_7_MODELS)

    assert exit_code == 0
    # Must parse to the original list with no modification
    parsed = json.loads(out)
    assert parsed == _REAL_7_MODELS, "json output must be the raw models list, unmodified"


# ---------------------------------------------------------------------------
# Golden (brief-253a): no human-facing CLI text contains "fp3"
# ---------------------------------------------------------------------------

def test_golden_no_fp3_in_human_facing_output():
    """A developer reading `newt models` never SEES "fp3" — the rename's whole point.

    "FP3" is a training-run designation that leaked into the product name. The
    legacy `nt0-fp3-*` alias tags stay in the registry so stale installs keep
    resolving (T1), but they must never surface to a human. UIDs are exempt by
    design — `ft_base_nt0fp3` is an immutable handle, renamed separately under
    the design-gated brief-253c — so we mask the uids out before scanning, which
    is exactly the carve-out the brief declares ("JSON/UID surfaces exempt").

    This golden is the verification gate for the renderer fix: with the old
    longest-stripped-remainder logic it printed "fp3-clean-table" and failed here.
    """
    output = _render_models(_REAL_7_MODELS)

    # Mask uids — the one place "fp3" is allowed to appear (immutable handle).
    scrubbed = output
    for uid in (m["uid"] for m in _REAL_7_MODELS):
        scrubbed = scrubbed.replace(uid, "")

    assert "fp3" not in scrubbed.lower(), (
        f"fp3 leaked into human-facing model names (uids masked out): {scrubbed!r}"
    )


def test_golden_no_fp3_via_cmd_models_render_path(monkeypatch):
    """The same guarantee through the real `newt models` command, not just the renderer.

    Exercises cmd_models end-to-end (non-json path) so the gate covers what a
    developer actually runs, not only the internal helper. Uids masked per the
    same exemption.
    """
    exit_code, out, err = _run([], monkeypatch, models_return=_REAL_7_MODELS)
    assert exit_code == 0, f"expected exit 0, got {exit_code}; stderr={err!r}"

    scrubbed = out
    for uid in (m["uid"] for m in _REAL_7_MODELS):
        scrubbed = scrubbed.replace(uid, "")

    assert "fp3" not in scrubbed.lower(), (
        f"fp3 leaked into `newt models` output (uids masked out): {scrubbed!r}"
    )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_models_bad_key_exits_nonzero_and_blames_key(monkeypatch):
    """A developer with a revoked key sees an auth failure, not a registry outage.

    This is the affordance guard: the command must say the KEY is bad, not
    report that the registry is down. Otherwise the developer chases phantom
    infra problems.
    """
    exc = AuthError(
        code=4001,
        type="auth.invalid_key",
        message="Authentication failed: API key rejected by registry /v1/models. Rotate your key.",
        context={"key_prefix": "nt_testke"},
    )
    exit_code, out, err = _run([], monkeypatch, side_effect=exc)

    assert exit_code != 0, "bad key must produce non-zero exit"
    assert "authentication" in err.lower() or "key" in err.lower(), (
        f"error must mention authentication or key, got: {err!r}"
    )
    # Must NOT say the registry is down when the key is the problem
    assert "registry" not in err.lower() or "key" in err.lower(), (
        f"must not only blame registry when key is bad: {err!r}"
    )


def test_models_registry_unavailable_exits_nonzero(monkeypatch):
    """When the registry is genuinely unreachable, the command fails with context."""
    exc = RegistryUnavailable(
        bootstrap_url="https://nt-registry-production.up.railway.app",
        reason="connection refused",
    )
    exit_code, out, err = _run([], monkeypatch, side_effect=exc)

    assert exit_code != 0
    assert "registry" in err.lower() or "unreachable" in err.lower()


def test_models_no_key_exits_nonzero_and_tells_user_what_to_do(monkeypatch, tmp_path):
    """Without a key, the developer gets a clear 'run newt login' message."""
    import io as _io
    import sys as _sys

    # Remove NT_API_KEY from env and point credentials to an empty tmp dir
    monkeypatch.delenv("NT_API_KEY", raising=False)

    # Patch read_api_key to return None (no credentials file)
    import newt._cli.models as models_mod
    monkeypatch.setattr(models_mod, "read_api_key", lambda: None)

    captured_err = _io.StringIO()
    monkeypatch.setattr(_sys, "stderr", captured_err)
    monkeypatch.setattr(_sys, "stdout", _io.StringIO())

    exit_code = cmd_models([])

    assert exit_code != 0
    err = captured_err.getvalue()
    assert "login" in err.lower() or "NT_API_KEY" in err, (
        f"must tell the developer how to fix: {err!r}"
    )


def test_models_empty_catalog(monkeypatch):
    """An empty catalog prints a clean message rather than silently outputting nothing."""
    exit_code, out, err = _run([], monkeypatch, models_return=[])

    assert exit_code == 0
    assert out.strip()  # some output, even if empty


def test_models_json_empty_catalog_is_valid_array(monkeypatch):
    """--json with an empty catalog still emits a valid JSON array, not nothing."""
    exit_code, out, err = _run(["--json"], monkeypatch, models_return=[])

    assert exit_code == 0
    parsed = json.loads(out)
    assert parsed == []
