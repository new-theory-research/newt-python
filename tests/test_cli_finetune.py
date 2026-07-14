"""Offline unit tests for `newt finetune` — the launch-and-watch verb.

No network: the two HTTP round-trips (`_launch`, `_poll_status`) and `time.sleep`
are monkeypatched. Each test encodes why the behavior matters:
  - the launch prints a re-attachable handle, then polls to a terminal state;
  - a terminal FAILURE names the failing pipeline gate (Rule 10) and exits non-zero;
  - a success surfaces the tag + report-card pointer, never a fabricated one;
  - `--json` emits handle + terminal status/tag on a clean stdout;
  - no Modal credential is reachable from the client (the whole point of the design).
"""
from __future__ import annotations

import io
import json
import sys
from urllib.error import HTTPError

import pytest

import newt._cli.finetune as ft
from newt._cli.finetune import cmd_finetune, _render_terminal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args, monkeypatch, *, launch=None, statuses=None, key="nt_testkey"):
    """Run cmd_finetune with mocked _launch / _poll_status, capturing stdout+stderr.

    `launch` is the dict _launch returns (default: a handle). `statuses` is a list of
    status dicts _poll_status returns in order (the last one is usually terminal).
    """
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(ft.time, "sleep", lambda *_a, **_k: None)

    if key is None:
        monkeypatch.delenv("NT_API_KEY", raising=False)
        monkeypatch.setattr(ft, "read_api_key", lambda: None)
    else:
        monkeypatch.setenv("NT_API_KEY", key)

    if launch is not None:
        monkeypatch.setattr(ft, "_launch", lambda *a, **k: launch)

    if statuses is not None:
        seq = list(statuses)
        def _poll(*a, **k):
            return seq.pop(0) if len(seq) > 1 else seq[0]
        monkeypatch.setattr(ft, "_poll_status", _poll)

    rc = cmd_finetune(args)
    return rc, out.getvalue(), err.getvalue()


_HANDLE = {"job_handle": "fc-abc123", "dataset": "my-task", "status": "launched"}
_SUCCEEDED = {
    "status": "succeeded",
    "gate": None,
    "tag": "ft-placeholder-9f2a",
    "report_card": "gs://ntdeva-reports/fc-abc123/card.json",
}
_FAILED_AT_TRAIN = {"status": "failed", "gate": "train", "tag": None, "report_card": None}


# ---------------------------------------------------------------------------
# Golden: launch → poll → succeeded
# ---------------------------------------------------------------------------

def test_launch_prints_handle_then_reports_success(monkeypatch):
    """A developer runs `newt finetune --dataset X`, sees the handle, and — once the
    run lands — the model tag and report card. Exit 0."""
    rc, out, err = _run(
        ["--dataset", "my-task"], monkeypatch, launch=_HANDLE, statuses=[_SUCCEEDED]
    )
    assert rc == 0, f"expected exit 0; stderr={err!r}"
    assert "fc-abc123" in out, "the job handle must be printed so it's re-attachable"
    assert "ft-placeholder-9f2a" in out, "the resulting tag must surface on success"
    assert "card.json" in out, "the report-card pointer must surface on success"


def test_success_after_running_polls(monkeypatch):
    """A run that reports `running` before `succeeded` keeps polling to the terminal
    state — it doesn't stop at the first non-terminal poll."""
    running = {"status": "running", "gate": None, "tag": None, "report_card": None}
    rc, out, _ = _run(
        ["--dataset", "d"], monkeypatch, launch=_HANDLE, statuses=[running, _SUCCEEDED]
    )
    assert rc == 0
    assert "ft-placeholder-9f2a" in out


# ---------------------------------------------------------------------------
# Terminal FAILURE names the gate (Rule 10)
# ---------------------------------------------------------------------------

def test_failure_names_the_gate_and_exits_nonzero(monkeypatch):
    """A pipeline gate failure must NAME the gate and exit non-zero — a developer must
    know it was the `train` gate, not a generic 'something failed'."""
    rc, out, err = _run(
        ["--dataset", "d"], monkeypatch, launch=_HANDLE, statuses=[_FAILED_AT_TRAIN]
    )
    assert rc == 1, "a failed run must exit non-zero"
    assert "train" in out, f"the failing gate must be named: {out!r}"


def test_render_terminal_failure_names_gate():
    text, code = _render_terminal(_FAILED_AT_TRAIN)
    assert code == 1
    assert "train" in text


def test_render_terminal_success_pending_tag_is_honest():
    """A finished run that produced no tag says '(pending)', never a fabricated tag."""
    text, code = _render_terminal(
        {"status": "succeeded", "gate": None, "tag": None, "report_card": None}
    )
    assert code == 0
    assert "pending" in text.lower()


# ---------------------------------------------------------------------------
# --json
# ---------------------------------------------------------------------------

def test_json_emits_handle_and_terminal_state(monkeypatch):
    """`--json` emits a single JSON object on stdout: handle + terminal status/tag."""
    rc, out, err = _run(
        ["--dataset", "my-task", "--json"], monkeypatch, launch=_HANDLE, statuses=[_SUCCEEDED]
    )
    assert rc == 0
    parsed = json.loads(out)  # stdout must be clean JSON — no human lines leaking in
    assert parsed["job_handle"] == "fc-abc123"
    assert parsed["status"] == "succeeded"
    assert parsed["tag"] == "ft-placeholder-9f2a"


def test_json_failure_exits_nonzero_and_names_gate(monkeypatch):
    rc, out, _ = _run(
        ["--dataset", "d", "--json"], monkeypatch, launch=_HANDLE, statuses=[_FAILED_AT_TRAIN]
    )
    assert rc == 1
    parsed = json.loads(out)
    assert parsed["status"] == "failed"
    assert parsed["gate"] == "train"


# ---------------------------------------------------------------------------
# --handle re-attach (no launch)
# ---------------------------------------------------------------------------

def test_handle_reattaches_without_launching(monkeypatch):
    """`newt finetune --handle <job>` polls an existing run and must NOT launch a new
    one — otherwise re-attaching would double-spend GPU."""
    launched = {"called": False}
    def _boom(*a, **k):
        launched["called"] = True
        raise AssertionError("--handle must not launch a new run")
    monkeypatch.setattr(ft, "_launch", _boom)

    rc, out, _ = _run(
        ["--handle", "fc-xyz"], monkeypatch, statuses=[_SUCCEEDED]
    )
    assert rc == 0
    assert launched["called"] is False


# ---------------------------------------------------------------------------
# Success surfaces the Robot snippet (the tag is useless without the call)
# ---------------------------------------------------------------------------

def test_success_prints_robot_snippet(monkeypatch):
    """On success the exact `Robot(model="<tag>")` call is printed — the tag alone
    doesn't tell a developer how to point at their new model."""
    rc, out, _ = _run(
        ["--dataset", "d"], monkeypatch, launch=_HANDLE, statuses=[_SUCCEEDED]
    )
    assert rc == 0
    assert 'Robot(model="ft-placeholder-9f2a")' in out


def test_pending_tag_prints_no_snippet():
    """A finished run with no tag yet must not print a Robot() call around a fake tag."""
    text, code = _render_terminal(
        {"status": "succeeded", "gate": None, "tag": None, "report_card": None}
    )
    assert code == 0
    assert "Robot(model=" not in text


# ---------------------------------------------------------------------------
# Launch breadcrumbs — watch page + one-shot --status hint
# ---------------------------------------------------------------------------

def test_launch_prints_watch_page_and_status_hint(monkeypatch):
    """Launch must print the console watch page (/runs/<handle>) and the copy-paste
    one-shot check — the two ways a developer re-finds a run after closing the CLI."""
    rc, out, _ = _run(
        ["--dataset", "my-task"], monkeypatch, launch=_HANDLE, statuses=[_SUCCEEDED]
    )
    assert rc == 0
    assert "/runs/fc-abc123" in out, "the console watch-page URL must be printed at launch"
    assert "--handle fc-abc123 --status" in out, "the one-shot re-check must be printed"


# ---------------------------------------------------------------------------
# --status one-shot: fetch once, print, exit — no blocking poll
# ---------------------------------------------------------------------------

def test_status_one_shot_succeeded_exits_zero_polls_once(monkeypatch):
    """`--handle X --status` fetches state exactly once and exits 0, surfacing the tag
    and Robot snippet — no watch loop."""
    calls = {"n": 0}
    def _poll(*a, **k):
        calls["n"] += 1
        return _SUCCEEDED
    monkeypatch.setattr(ft, "_poll_status", _poll)
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")

    rc = cmd_finetune(["--handle", "fc-abc123", "--status"])
    assert rc == 0
    assert calls["n"] == 1, "one-shot status must poll exactly once, never loop"
    assert 'Robot(model="ft-placeholder-9f2a")' in out.getvalue()


def test_status_one_shot_running_exits_zero(monkeypatch):
    """A run still in progress: --status reports the state and exits 0 (the fetch
    succeeded) — it does not block waiting for a terminal state."""
    running = {"status": "running", "gate": None, "tag": None, "report_card": None}
    calls = {"n": 0}
    def _poll(*a, **k):
        calls["n"] += 1
        return running
    monkeypatch.setattr(ft, "_poll_status", _poll)
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")

    rc = cmd_finetune(["--handle", "fc-abc123", "--status"])
    assert rc == 0
    assert calls["n"] == 1
    assert "running" in out.getvalue()


def test_status_one_shot_json_is_clean(monkeypatch):
    """`--status --json` emits a single JSON object with the current state — clean
    stdout, scriptable."""
    running = {"status": "running", "gate": None, "tag": None, "report_card": None}
    monkeypatch.setattr(ft, "_poll_status", lambda *a, **k: running)
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")

    rc = cmd_finetune(["--handle", "fc-abc123", "--status", "--json"])
    assert rc == 0
    parsed = json.loads(out.getvalue())
    assert parsed["job_handle"] == "fc-abc123"
    assert parsed["status"] == "running"


def test_status_requires_handle(monkeypatch):
    """`--status` without `--handle` is an error — there's no run to check."""
    rc, out, err = _run(["--dataset", "d", "--status"], monkeypatch)
    assert rc == 1
    assert "handle" in err.lower()


def test_status_one_shot_404_exits_nonzero(monkeypatch):
    """An unknown handle fails the fetch itself — exit non-zero, name the handle."""
    def _not_found(*a, **k):
        raise HTTPError("url", 404, "Not Found", {}, None)
    monkeypatch.setattr(ft, "_poll_status", _not_found)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")

    rc = cmd_finetune(["--handle", "fc-missing", "--status"])
    assert rc == 1
    assert "fc-missing" in err.getvalue()


# ---------------------------------------------------------------------------
# Argument + auth guards
# ---------------------------------------------------------------------------

def test_no_dataset_or_handle_errors(monkeypatch):
    rc, out, err = _run([], monkeypatch)
    assert rc == 1
    assert "dataset" in err.lower()


def test_dataset_and_handle_are_mutually_exclusive(monkeypatch):
    rc, out, err = _run(["--dataset", "d", "--handle", "fc-x"], monkeypatch)
    assert rc == 1
    assert "both" in err.lower() or "or" in err.lower()


def test_no_key_tells_user_what_to_do(monkeypatch):
    rc, out, err = _run(["--dataset", "d"], monkeypatch, key=None)
    assert rc == 1
    assert "login" in err.lower() or "NT_API_KEY" in err


def test_dataset_flag_with_no_value_is_caught(monkeypatch):
    """`--dataset` with no value (or followed by another flag) is a missing dataset,
    not a dataset literally named '--json'."""
    rc, out, err = _run(["--dataset", "--json"], monkeypatch)
    assert rc == 1
    assert "dataset" in err.lower()


# ---------------------------------------------------------------------------
# Error paths from the console
# ---------------------------------------------------------------------------

def test_launch_401_blames_the_key(monkeypatch):
    def _bad_key(*a, **k):
        raise HTTPError("url", 401, "Unauthorized", {}, None)
    monkeypatch.setattr(ft, "_launch", _bad_key)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")

    rc = cmd_finetune(["--dataset", "d"])
    assert rc == 1
    assert "key" in err.getvalue().lower() or "auth" in err.getvalue().lower()


def test_status_404_reports_unknown_handle(monkeypatch):
    def _not_found(*a, **k):
        raise HTTPError("url", 404, "Not Found", {}, None)
    monkeypatch.setattr(ft, "_poll_status", _not_found)
    monkeypatch.setattr(ft.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)

    rc = cmd_finetune(["--handle", "fc-missing"])
    assert rc == 1
    assert "fc-missing" in err.getvalue()


# ---------------------------------------------------------------------------
# Help guard — exits 0, does nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_guard_exits_zero_no_action(flag, monkeypatch):
    launched = {"called": False}
    monkeypatch.setattr(ft, "_launch", lambda *a, **k: launched.__setitem__("called", True))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    rc = cmd_finetune([flag])
    assert rc == 0
    assert launched["called"] is False
    assert "finetune" in out.getvalue()


# ---------------------------------------------------------------------------
# The unforgivable failure: a Modal credential reachable from the client
# ---------------------------------------------------------------------------

def test_finetune_module_imports_no_modal_and_holds_no_credential():
    """The CLI must never import Modal or carry a Modal credential — a developer's key
    buys a job handle, never the ability to talk to Modal (training spec §3.1)."""
    import pathlib

    source = pathlib.Path(ft.__file__).read_text()
    assert "import modal" not in source, "the CLI must not import the Modal SDK"
    lowered = source.lower()
    for forbidden in ("modal-key", "modal-secret", "modal_token", "modal_secret"):
        assert forbidden not in lowered, f"no Modal credential may appear in the client: {forbidden!r}"
