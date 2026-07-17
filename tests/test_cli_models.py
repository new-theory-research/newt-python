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


# Real 7-model registry payload (structure matches live API)
_REAL_7_MODELS = [
    {
        "uid": "ft_base_nt0fp3",
        "tags": ["nt0-fp3"],
        "type": "base",
        "base": None,
        "contract": {
            "action_axes": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]
        },
    },
    {
        "uid": "ft_4hn40z6a",
        "tags": ["nt0-fp3-pour-coffee-beans", "pour-coffee-beans"],
        "type": "fine_tune",
        "base": "ft_base_nt0fp3",
        "contract": {"action_axes": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]},
    },
    {
        "uid": "ft_6d0cfd51_e63fbb",
        "tags": ["nt0-fp3-clean-table"],
        "type": "fine_tune",
        "base": "ft_base_nt0fp3",
        "contract": {"action_axes": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]},
    },
    {
        "uid": "ft_6d0cfd51_c9d8ba",
        "tags": ["nt0-fp3-pour-coffee-beans-wm"],
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
    """Base display names lead each family group — nt0-fp3 / molmoact2 / pi05_aloha.

    The human name (primary tag) should appear first on the base line, not the
    opaque uid. A developer picking a model should find it by name.
    """
    output = _render_models(_REAL_7_MODELS)
    lines = output.splitlines()
    # Base lines are the non-indented ones
    base_lines = [l for l in lines if l and not l.startswith("    ")]
    names = [l.split()[0] for l in base_lines]
    assert names == ["nt0-fp3", "molmoact2", "pi05_aloha"], (
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


def test_axes_appear_once_on_nt0fp3_base_only():
    """Action axes appear exactly once: on the nt0-fp3 base line.

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

    # That one line must be the nt0-fp3 base line
    assert "nt0-fp3" in axes_lines[0], (
        f"axes line must be the nt0-fp3 base, got: {axes_lines[0]!r}"
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
    """Fine-tune task names derive correctly from tags, not raw uids.

    pour-coffee-beans — shortest non-prefixed tag wins.
    clean-table — single base-prefixed tag, strip prefix.
    pour-coffee-beans-wm — strip prefix, only candidate.
    bimanual-yam — all tags share molmoact2- prefix; strip + take longest.
    """
    output = _render_models(_REAL_7_MODELS)
    lines = output.splitlines()
    ft_lines = [l.strip() for l in lines if l.startswith("    ")]
    task_names = [l.split()[0] for l in ft_lines]

    assert "pour-coffee-beans" in task_names, f"task names: {task_names!r}"
    assert "clean-table" in task_names, f"task names: {task_names!r}"
    assert "pour-coffee-beans-wm" in task_names, f"task names: {task_names!r}"
    assert "bimanual-yam" in task_names, f"task names: {task_names!r}"


def test_uids_absent_by_default():
    """No raw uid appears anywhere in the default (human) listing.

    Mattie's 2026-07-17 friction report: uids on every line dominated visually
    while serving no human action browsing the catalog. The tag/display name
    is the identity a developer scans by; uids are dropped by default and
    restored only with --ids (see test_ids_flag_restores_uids).
    """
    output = _render_models(_REAL_7_MODELS)
    for uid in [
        "ft_base_nt0fp3",
        "ft_4hn40z6a",
        "ft_6d0cfd51_e63fbb",
        "ft_6d0cfd51_c9d8ba",
        "ft_base_molmoact2",
        "ft_molmoact2_yam",
        "ft_base_pi05_aloha",
    ]:
        assert uid not in output, f"uid {uid!r} must NOT appear in default output: {output!r}"


def test_ids_flag_restores_uids():
    """--ids (show_ids=True) restores every model's uid alongside its name.

    Scripts and agents that still want the raw handle get it back on request;
    the default stays clean for humans scanning the catalog.
    """
    output = _render_models(_REAL_7_MODELS, show_ids=True)
    for uid in [
        "ft_base_nt0fp3",
        "ft_4hn40z6a",
        "ft_6d0cfd51_e63fbb",
        "ft_6d0cfd51_c9d8ba",
        "ft_base_molmoact2",
        "ft_molmoact2_yam",
        "ft_base_pi05_aloha",
    ]:
        assert uid in output, f"--ids must restore uid {uid!r}: {output!r}"


def test_uid_alignment_within_family_with_ids():
    """With --ids, fine-tune uids are column-aligned within a family.

    Same leading whitespace (4 spaces) + task name padded to column width.
    This makes the uid column visually scannable when uids are shown at all.
    """
    output = _render_models(_REAL_7_MODELS, show_ids=True)
    lines = output.splitlines()
    # Get the nt0-fp3 family's fine-tune lines
    nt0_fts = []
    in_nt0 = False
    for line in lines:
        if line and not line.startswith("    "):
            in_nt0 = "nt0-fp3" in line
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
    base is temporarily absent. The orphan's own uid is a raw handle (hidden
    by default, like everywhere else); the `[base: ...]` reference is kept
    regardless — it's the diagnostic explaining WHY this fine-tune is orphaned,
    not the model's own identity handle.
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
    assert "orphan-task" in output, "orphan task name must appear"
    assert "ft_base_missing" in output, "orphan base reference must appear"
    assert "ft_orphan_abc123" not in output, "orphan's own uid must be hidden by default"

    output_ids = _render_models(models, show_ids=True)
    assert "ft_orphan_abc123" in output_ids, "--ids must restore the orphan's own uid"


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
            "tags": ["nt0-fp3"],
            "type": "base",
            "base": None,
            "contract": {"action_axes": ["x", "y", "z"]},
        },
    ]
    output = _render_models(models)
    lines = output.splitlines()
    non_empty = [l for l in lines if l.strip()]
    # Orphan must come before any family
    orphan_idx = next(i for i, l in enumerate(non_empty) if "orphan-task" in l)
    base_idx = next(i for i, l in enumerate(non_empty) if "nt0-fp3" in l)
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

    The command must exit 0 and print each base's display name (tag) on its
    line — the human identity, not the raw uid. The developer can scan the
    list to find the model they want.
    """
    exit_code, out, err = _run([], monkeypatch, models_return=_REAL_7_MODELS)

    assert exit_code == 0, f"expected exit 0, got {exit_code}; stderr={err!r}"
    assert "nt0-fp3" in out
    assert "molmoact2" in out
    assert err == ""


def test_models_renders_catalog_with_ids(monkeypatch):
    """--ids restores the base uids in the full CLI path (not just the renderer)."""
    exit_code, out, err = _run(["--ids"], monkeypatch, models_return=_REAL_7_MODELS)

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
    nt0_base_line = next(l for l in lines if "nt0-fp3" in l and not l.startswith("    "))

    assert "axes" in nt0_base_line, f"axes must render on the nt0-fp3 base line: {nt0_base_line!r}"
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


def test_models_json_unaffected_by_ids_flag(monkeypatch):
    """--json combined with --ids still emits the raw, unmodified array.

    --ids is a human-output concern only; agents get the full payload either way.
    """
    exit_code, out, _ = _run(["--json", "--ids"], monkeypatch, models_return=_REAL_7_MODELS)

    assert exit_code == 0
    assert json.loads(out) == _REAL_7_MODELS


# ---------------------------------------------------------------------------
# Identity line — which credential produced this listing (masked, never raw)
#
# Mattie's 2026-07-17 friction report: with multiple accounts/keys, nothing
# showed WHICH account a listing was for. The Key line answers that from data
# already local at resolve time (the key string is in hand; the source — env
# var vs credentials file — is known the same way `newt status` knows it).
# No new network call, no wire change.
# ---------------------------------------------------------------------------

def test_identity_line_shows_masked_key_and_env_source(monkeypatch):
    """Default human output leads with a Key line: masked key + env source."""
    exit_code, out, err = _run([], monkeypatch, models_return=_REAL_7_MODELS)

    assert exit_code == 0, f"stderr={err!r}"
    assert "Key nt_" in out
    assert "(environment)" in out


def test_identity_line_shows_credentials_file_source(monkeypatch):
    """When the key resolves from ~/.nt/credentials, the source label says so."""
    import io as _io

    import newt._cli.models as models_mod
    from newt._cli.models import cmd_models

    monkeypatch.delenv("NT_API_KEY", raising=False)
    monkeypatch.setattr(models_mod, "read_api_key", lambda: "nt_ab12cd34ef56gh78840eae1f")
    monkeypatch.setattr(newt, "list_models", lambda *a, **kw: _REAL_7_MODELS)

    captured_out = _io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_out)
    monkeypatch.setattr(sys, "stderr", _io.StringIO())

    exit_code = cmd_models([])
    out = captured_out.getvalue()

    assert exit_code == 0
    assert "(credentials file)" in out
    assert "840eae1f" in out


def test_identity_line_never_leaks_full_key(monkeypatch):
    """The full key never appears in human output — masked prefix only.

    Rule 10: a masked hint is fine; the full secret landing in a terminal,
    shell history, screen-share, or CI log is not.
    """
    import io as _io

    import newt._cli.models as models_mod
    from newt._cli.models import cmd_models

    full_key = "nt_ab12cd34ef56gh78840eae1f"
    monkeypatch.delenv("NT_API_KEY", raising=False)
    monkeypatch.setattr(models_mod, "read_api_key", lambda: full_key)
    monkeypatch.setattr(newt, "list_models", lambda *a, **kw: _REAL_7_MODELS)

    captured_out = _io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_out)
    monkeypatch.setattr(sys, "stderr", _io.StringIO())

    cmd_models([])
    out = captured_out.getvalue()

    assert full_key not in out, f"full key must never appear in output: {out!r}"
    assert "840eae1f" in out, "masked suffix (last 8 chars) must still appear"
    assert "•" in out, "masking bullets must appear"


def test_mask_key_matches_console_convention():
    """_mask_key mirrors the console's own masking: nt_ + 8 bullets + last 8 chars.

    apps/console/db/schema.ts: `prefix` = last 8 chars of the plaintext key,
    rendered as `nt_••••••••{prefix}`. Matching it means a developer recognizes
    the SAME key across `newt models` and the console's Keys page.
    """
    from newt._cli.models import _mask_key

    assert _mask_key("nt_ab12cd34ef56gh78840eae1f") == "nt_••••••••840eae1f"


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
