"""Offline unit tests for `newt upgrade` — the verb + the passive once-a-day notice.

No network: `_fetch_latest` and `subprocess.run` are monkeypatched, and the timestamp
cache is repointed at a tmp dir. Each test encodes WHY the behavior matters (console-014):

  - `newt upgrade --print` prints the EXACT documented command and never runs it;
  - the run path invokes `uv tool upgrade newt` ONLY when the install is a confirmed
    uv-tool, and print-don't-runs otherwise (never guess-executes a package manager);
  - the passive notice fires only when the versions actually differ, is silent on every
    failure, honors the once-a-day cache, never touches stdout / is skipped under --json,
    honors NEWT_NO_UPDATE_CHECK=1, and never slows a command (a hard ~1s timeout cap).
"""
from __future__ import annotations

import io
import json
import sys
import types
from datetime import date

import pytest

import newt._cli.upgrade as up
from newt._cli import main


# ---------------------------------------------------------------------------
# cmd_upgrade — the verb
# ---------------------------------------------------------------------------

def _run_verb(args, monkeypatch, *, latest=None, is_uv_tool=True, run_rc=0, run_raises=None):
    """Run cmd_upgrade with the endpoint + install-detection + subprocess mocked.

    `latest` is the dict `_fetch_latest` returns (None ⇒ offline, use the local default).
    Returns (rc, out, err, run_calls) where run_calls records every subprocess.run argv.
    """
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(up, "_fetch_latest", lambda *a, **k: latest)
    monkeypatch.setattr(up, "_is_uv_tool_install", lambda: is_uv_tool)

    run_calls = []

    def _fake_run(argv, *a, **k):
        run_calls.append(argv)
        if run_raises is not None:
            raise run_raises
        return types.SimpleNamespace(returncode=run_rc)

    monkeypatch.setattr(up.subprocess, "run", _fake_run)
    rc = up.cmd_upgrade(args)
    return rc, out.getvalue(), err.getvalue(), run_calls


def test_print_emits_exact_command_and_does_not_run(monkeypatch):
    """`newt upgrade --print` prints EXACTLY `uv tool upgrade newt` and runs nothing —
    the composable, offline-safe way to see the command."""
    rc, out, err, run_calls = _run_verb(["--print"], monkeypatch, latest=None)
    assert rc == 0
    assert out.strip() == "uv tool upgrade newt"
    assert run_calls == [], "--print must never invoke a subprocess"


def test_print_does_not_run_even_on_a_uv_tool_install(monkeypatch):
    """--print is print-don't-run regardless of install detection."""
    _, _, _, run_calls = _run_verb(["--print"], monkeypatch, latest=None, is_uv_tool=True)
    assert run_calls == []


def test_run_invokes_the_documented_command_on_a_confirmed_uv_tool(monkeypatch):
    """On a confirmed uv-tool install, `newt upgrade` runs `uv tool upgrade newt` and
    returns the subprocess exit code."""
    rc, out, _, run_calls = _run_verb([], monkeypatch, latest=None, is_uv_tool=True, run_rc=0)
    assert rc == 0
    assert run_calls == [["uv", "tool", "upgrade", "newt"]]
    assert "uv tool upgrade newt" in out


def test_run_returns_the_subprocess_exit_code(monkeypatch):
    """A failed upgrade command surfaces its non-zero exit code (Rule 10 — not swallowed)."""
    rc, _, _, run_calls = _run_verb([], monkeypatch, latest=None, is_uv_tool=True, run_rc=7)
    assert rc == 7
    assert run_calls == [["uv", "tool", "upgrade", "newt"]]


def test_ambiguous_install_prints_and_does_not_run(monkeypatch):
    """When the install method can't be confirmed, print-don't-run with a one-line why —
    never guess-execute a package manager against the wrong environment (Rule 10)."""
    rc, out, err, run_calls = _run_verb([], monkeypatch, latest=None, is_uv_tool=False)
    assert rc == 0
    assert run_calls == [], "an unconfirmed install must NOT run a package manager"
    assert out.strip() == "uv tool upgrade newt"
    assert "couldn't confirm" in err.lower()


def test_run_prefers_the_endpoint_command_single_source(monkeypatch):
    """The verb prefers the endpoint's `upgrade` string when reachable — the command
    lives in one place, so a future change ships without a client release."""
    latest = {"latest": "9.9.9", "upgrade": "uv tool upgrade newt --reinstall"}
    rc, _, _, run_calls = _run_verb([], monkeypatch, latest=latest, is_uv_tool=True)
    assert rc == 0
    assert run_calls == [["uv", "tool", "upgrade", "newt", "--reinstall"]]


def test_missing_uv_binary_fails_loud_not_silent(monkeypatch):
    """If `uv` isn't on PATH, say so and exit non-zero — never a silent success."""
    rc, _, err, _ = _run_verb(
        [], monkeypatch, latest=None, is_uv_tool=True, run_raises=FileNotFoundError()
    )
    assert rc == 1
    assert "uv" in err.lower()


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_prints_usage_and_runs_nothing(flag, monkeypatch):
    rc, out, _, run_calls = _run_verb([flag], monkeypatch)
    assert rc == 0
    assert "upgrade" in out.lower()
    assert run_calls == []


# ---------------------------------------------------------------------------
# run_update_check — the passive once-a-day notice
# ---------------------------------------------------------------------------

def _check(monkeypatch, tmp_path, *, installed="0.0.1", fetch=None, args=None, cache_pre=None, opt_out=False):
    """Drive run_update_check with the version fetch + installed version + cache mocked.

    Returns (out, err, fetch_calls). `fetch` is the dict `_fetch_latest` returns (or None
    ⇒ endpoint failure). `cache_pre` seeds the on-disk cache. `opt_out` sets
    NEWT_NO_UPDATE_CHECK=1. The cache dir is repointed at tmp_path so nothing touches the
    real ~/.nt/.
    """
    monkeypatch.setattr(up._credentials, "CREDENTIALS_DIR", tmp_path)
    monkeypatch.setattr(up, "_installed_version", lambda: installed)
    if opt_out:
        monkeypatch.setenv("NEWT_NO_UPDATE_CHECK", "1")
    else:
        monkeypatch.delenv("NEWT_NO_UPDATE_CHECK", raising=False)

    fetch_calls = []

    def _fake_fetch(console, *, timeout):
        fetch_calls.append({"console": console, "timeout": timeout})
        return fetch

    monkeypatch.setattr(up, "_fetch_latest", _fake_fetch)

    if cache_pre is not None:
        (tmp_path / "update-check.json").write_text(json.dumps(cache_pre))

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    up.run_update_check(args or [])
    return out.getvalue(), err.getvalue(), fetch_calls


_STALE = {"latest": "0.0.2", "upgrade": "uv tool upgrade newt"}


def test_notice_appears_on_stderr_when_stale(monkeypatch, tmp_path):
    """A newer `latest` than the installed version prints exactly one stderr line."""
    out, err, calls = _check(monkeypatch, tmp_path, installed="0.0.1", fetch=_STALE)
    assert "newt 0.0.2 available — run 'newt upgrade'" in err
    assert out == "", "the notice must NEVER touch stdout"
    assert len(calls) == 1


def test_notice_absent_when_current(monkeypatch, tmp_path):
    """When installed == latest, the notice does not fire (Finding 2 — only on a real
    difference)."""
    out, err, _ = _check(monkeypatch, tmp_path, installed="0.0.2", fetch=_STALE)
    assert err == ""
    assert out == ""


def test_notice_absent_and_silent_when_endpoint_dead(monkeypatch, tmp_path):
    """A dead/slow/bad-JSON endpoint (`_fetch_latest` → None) prints nothing and raises
    nothing — a dead endpoint costs nothing (Rule 10/11)."""
    out, err, calls = _check(monkeypatch, tmp_path, installed="0.0.1", fetch=None)
    assert err == ""
    assert out == ""
    assert len(calls) == 1  # it tried once


def test_notice_absent_within_the_daily_window(monkeypatch, tmp_path):
    """A cache entry for today short-circuits the check — no fetch, no notice."""
    today = date.today().isoformat()
    out, err, calls = _check(
        monkeypatch, tmp_path, installed="0.0.1", fetch=_STALE,
        cache_pre={"last_check": today, "latest": "0.0.2"},
    )
    assert calls == [], "within the daily window there must be NO fetch"
    assert err == ""


def test_check_skipped_entirely_under_json(monkeypatch, tmp_path):
    """Under --json the check is skipped — no fetch, and stdout stays byte-clean for
    agents parsing it."""
    out, err, calls = _check(monkeypatch, tmp_path, installed="0.0.1", fetch=_STALE, args=["--json"])
    assert calls == [], "--json must skip the check entirely"
    assert out == ""
    assert err == ""


def test_opt_out_env_disables_the_check(monkeypatch, tmp_path):
    """NEWT_NO_UPDATE_CHECK=1 fully disables the check — no fetch, no notice."""
    out, err, calls = _check(monkeypatch, tmp_path, installed="0.0.1", fetch=_STALE, opt_out=True)
    assert calls == []
    assert err == ""
    assert out == ""


def test_check_is_capped_at_one_second(monkeypatch, tmp_path):
    """The passive GET is hard-capped at ~1s so a hung endpoint never delays a command
    (Rule 11)."""
    _, _, calls = _check(monkeypatch, tmp_path, installed="0.0.1", fetch=_STALE)
    assert len(calls) == 1
    assert calls[0]["timeout"] <= 1.0


def test_default_check_timeout_is_one_second():
    assert up._CHECK_TIMEOUT_S <= 1.0


def test_absent_when_installed_version_unreadable(monkeypatch, tmp_path):
    """If importlib.metadata can't report our version, fail silent — never fabricate a
    version to compare (Rule 10)."""
    out, err, calls = _check(monkeypatch, tmp_path, installed=None, fetch=_STALE)
    assert calls == [], "no fetch when we can't read our own version"
    assert err == ""
    assert out == ""


def test_stale_run_records_the_daily_cache(monkeypatch, tmp_path):
    """A completed check writes today's date to the cache so a second same-day run is a
    no-op (the once/day guarantee has teeth)."""
    _check(monkeypatch, tmp_path, installed="0.0.1", fetch=_STALE)
    cache = json.loads((tmp_path / "update-check.json").read_text())
    assert cache["last_check"] == date.today().isoformat()


# ---------------------------------------------------------------------------
# main() wiring — post-output, success-only, never for `upgrade`
# ---------------------------------------------------------------------------

def _run_main(argv, monkeypatch, *, dispatch_rc):
    """Run main() with _dispatch stubbed to a fixed rc and run_update_check spied on.
    Returns (exit_code, check_called_args). Because main() imports run_update_check
    fresh each call, patching the module attribute intercepts it."""
    import newt._cli as cli

    monkeypatch.setattr(cli, "_dispatch", lambda args: dispatch_rc)
    called = []
    monkeypatch.setattr(up, "run_update_check", lambda args: called.append(args))
    monkeypatch.setattr(sys, "argv", ["newt"] + argv)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code, called


def test_main_runs_check_after_a_successful_command(monkeypatch):
    code, called = _run_main(["models"], monkeypatch, dispatch_rc=0)
    assert code == 0
    assert called == [["models"]], "the check runs post-output after a success"


def test_main_skips_check_after_a_failed_command(monkeypatch):
    code, called = _run_main(["models"], monkeypatch, dispatch_rc=1)
    assert code == 1
    assert called == [], "a failed command must not trigger the notice"


def test_main_never_runs_check_for_upgrade_itself(monkeypatch):
    code, called = _run_main(["upgrade"], monkeypatch, dispatch_rc=0)
    assert code == 0
    assert called == [], "`newt upgrade` just changed the version — no self-nudge"
