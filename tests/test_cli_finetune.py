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
import os
import sys
from urllib.error import HTTPError

import pytest

import newt._cli.finetune as ft
from newt._cli.finetune import (
    cmd_finetune,
    _render_terminal,
    _survival_block,
    _render_jobs_table,
)


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
# --steps: an optional, client-validated total-training-steps override (ft-020)
# ---------------------------------------------------------------------------

def test_validate_steps_pure_shape():
    """The client-side shape check: absent → no override; a positive int passes; 0,
    negatives, and non-numeric are rejected with a NAMED error (never coerced)."""
    assert ft._validate_steps(None) == (None, None)
    assert ft._validate_steps("20000") == (20000, None)
    assert ft._validate_steps("  15000 ") == (15000, None)  # whitespace tolerated

    v, e = ft._validate_steps("0")
    assert v is None and "--steps" in e and "positive" in e.lower()
    v, e = ft._validate_steps("-5")
    assert v is None and "--steps" in e and "positive" in e.lower()
    v, e = ft._validate_steps("abc")
    assert v is None and "--steps" in e and "whole number" in e.lower()
    v, e = ft._validate_steps("3.5")
    assert v is None and "--steps" in e  # a float is not a whole number
    v, e = ft._validate_steps("")  # --steps with no value
    assert v is None and "--steps" in e


def test_steps_forwards_to_launch(monkeypatch):
    """`--dataset X --steps 20000` reaches the console launch as steps=20000."""
    captured = {}

    def fake_launch(console, api_key, dataset, *, steps=None, **k):
        captured["steps"] = steps
        return _HANDLE

    monkeypatch.setattr(ft, "_launch", fake_launch)
    rc, out, err = _run(["--dataset", "d", "--steps", "20000"], monkeypatch, statuses=[_SUCCEEDED])
    assert rc == 0
    assert captured["steps"] == 20000
    assert "steps:        20000" in out  # echoed to the human on launch


def test_steps_equals_form_forwards_to_launch(monkeypatch):
    captured = {}

    def fake_launch(console, api_key, dataset, *, steps=None, **k):
        captured["steps"] = steps
        return _HANDLE

    monkeypatch.setattr(ft, "_launch", fake_launch)
    rc, out, err = _run(["--dataset", "d", "--steps=15000"], monkeypatch, statuses=[_SUCCEEDED])
    assert rc == 0
    assert captured["steps"] == 15000


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "3.5"])
def test_steps_bad_value_rejected_before_any_network(bad, monkeypatch):
    """A garbage --steps value is rejected with a named error and NO launch — the bad
    value never reaches the wire (Rule 10)."""
    calls = []
    monkeypatch.setattr(ft, "_launch", lambda *a, **k: calls.append(1) or _HANDLE)
    rc, out, err = _run(["--dataset", "d", "--steps", bad], monkeypatch)
    assert rc == 1
    assert "--steps" in err
    assert calls == []  # never launched


def test_steps_requires_dataset(monkeypatch):
    """`--steps` sets a NEW launch's step count; on --handle re-attach there's nothing
    to apply it to, so it's refused loudly rather than silently ignored."""
    calls = []
    monkeypatch.setattr(ft, "_launch", lambda *a, **k: calls.append(1) or _HANDLE)
    monkeypatch.setattr(ft, "_poll_status", lambda *a, **k: _SUCCEEDED)
    rc, out, err = _run(["--handle", "fc-x", "--steps", "5"], monkeypatch)
    assert rc == 1
    assert "--steps" in err and "dataset" in err.lower()
    assert calls == []


def test_json_carries_effective_steps(monkeypatch):
    """`--json` output includes the effective steps the launch carried."""
    rc, out, err = _run(
        ["--dataset", "d", "--steps", "20000", "--json"],
        monkeypatch,
        launch=_HANDLE,
        statuses=[_SUCCEEDED],
    )
    assert rc == 0
    obj = json.loads(out)
    assert obj["steps"] == 20000


def test_json_steps_null_when_not_overridden(monkeypatch):
    """No --steps → the JSON `steps` is null (server default in force), never a
    CLI-invented default."""
    rc, out, err = _run(
        ["--dataset", "d", "--json"], monkeypatch, launch=_HANDLE, statuses=[_SUCCEEDED]
    )
    assert rc == 0
    obj = json.loads(out)
    assert obj["steps"] is None


def test_launch_payload_omits_steps_when_absent_includes_when_set(monkeypatch):
    """The launch POST body carries `steps` ONLY when the developer set it — absent, the
    field is omitted so the server applies its own default."""
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"job_handle": "fc-x"}'

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(ft, "urlopen", fake_urlopen)

    ft._launch("http://console", "nt_k", "my-task")
    assert "steps" not in captured["body"]

    ft._launch("http://console", "nt_k", "my-task", steps=1234)
    assert captured["body"]["steps"] == 1234


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
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")  # reach the poll, not the key guard

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


# ---------------------------------------------------------------------------
# Path-vs-name detection: a folder uploads-then-launches; a bare
# name is byte-identical to before. The upload leg drives the real NTCloudSink
# against a fake console (batch sign + PUT); _launch/_poll_status stay mocked.
# ---------------------------------------------------------------------------

_NAMESPACE = "0123456789abcdef"
_BUCKET = "nt-episodes"


class _FakeResp:
    def __init__(self, body: bytes = b"") -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _make_export(tmp_path, name="my_export", *, valid=True):
    import json as _json

    export = tmp_path / name
    (export / "meta").mkdir(parents=True)
    (export / "data").mkdir(parents=True)
    if valid:
        (export / "meta" / "info.json").write_text(
            _json.dumps(
                {
                    "codebase_version": "v3.0",
                    "features": {
                        "action": {"dtype": "float32", "shape": [6]},
                        "observation.state": {"dtype": "float32", "shape": [6]},
                    },
                }
            )
        )
    (export / "data" / "episode_000000.parquet").write_bytes(b"fake-parquet-bytes")
    return export


def _install_fake_console(monkeypatch, events, dataset):
    """Fake the console's batch/single sign + GCS PUT, appending 'sign'/'put' to
    `events` in call order so a test can assert the upload happened BEFORE launch."""
    import json as _json

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        if url.endswith("/api/uploads/sign"):
            events.append("sign")
            body = _json.loads(req.data)
            if "paths" in body:
                urls = [
                    {
                        "path": p,
                        "url": f"https://storage.googleapis.com/{_BUCKET}/{_NAMESPACE}/{dataset}/{p}?sig=fake",
                        "objectPath": f"gs://{_BUCKET}/{_NAMESPACE}/{dataset}/{p}",
                        "expiresAt": "2026-07-16T23:00:00.000Z",
                    }
                    for p in body["paths"]
                ]
                return _FakeResp(
                    _json.dumps(
                        {"namespace": _NAMESPACE, "dataset": dataset, "count": len(urls), "urls": urls}
                    ).encode()
                )
            p = body["path"]
            return _FakeResp(
                _json.dumps(
                    {
                        "url": f"https://storage.googleapis.com/{_BUCKET}/{_NAMESPACE}/{dataset}/{p}?sig=fake",
                        "objectPath": f"gs://{_BUCKET}/{_NAMESPACE}/{dataset}/{p}",
                        "expiresAt": "2026-07-16T23:00:00.000Z",
                    }
                ).encode()
            )
        events.append("put")
        return _FakeResp(b"")

    monkeypatch.setattr("newt.recording._cloud_sink.urlopen", fake_urlopen)


def test_path_uploads_then_launches_and_prints_staged_name(monkeypatch, tmp_path):
    """`newt finetune --dataset ./folder` validates, uploads, and launches — printing
    the branch decision, the size, and the staged name; the launch runs against that
    staged name."""
    events: list[str] = []
    export = _make_export(tmp_path, "my_export")
    _install_fake_console(monkeypatch, events, "my_export")

    launched_with = {}

    def fake_launch(console, api_key, dataset, **k):
        events.append("launch")
        launched_with["dataset"] = dataset
        return {"job_handle": "fc-abc123"}

    monkeypatch.setattr(ft, "_launch", fake_launch)
    monkeypatch.setattr(ft, "_poll_status", lambda *a, **k: _SUCCEEDED)
    monkeypatch.setattr(ft.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("NT_API_KEY", "nt_" + "a" * 40)
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    rc = cmd_finetune(["--dataset", str(export)])
    assert rc == 0, f"stderr={err.getvalue()!r}"

    text = out.getvalue()
    assert "uploading" in text.lower(), "the upload size must be printed"
    assert "staged as my_export" in text, "the staged name must be printed"
    assert launched_with["dataset"] == "my_export", "launch must run against the staged name"
    # Upload (sign + put) must all happen BEFORE the launch.
    assert "launch" in events and "sign" in events and "put" in events
    assert events.index("launch") == len(events) - 1, f"launch must be last: {events}"
    assert all(e in ("sign", "put") for e in events[: events.index("launch")])


def test_malformed_folder_fails_before_any_upload_or_launch(monkeypatch, tmp_path):
    """A folder with no meta/info.json fails BEFORE a byte moves: no sign, no PUT, no
    launch — the pre-transfer gate. The message names the fixable problem."""
    events: list[str] = []
    export = _make_export(tmp_path, "broken_export", valid=False)
    _install_fake_console(monkeypatch, events, "broken_export")
    monkeypatch.setattr(ft, "_launch", lambda *a, **k: events.append("launch") or {"job_handle": "x"})
    monkeypatch.setattr(ft, "_poll_status", lambda *a, **k: _SUCCEEDED)
    monkeypatch.setenv("NT_API_KEY", "nt_" + "a" * 40)
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    rc = cmd_finetune(["--dataset", str(export)])
    assert rc == 1
    assert events == [], f"nothing should sign, upload, or launch: {events}"
    assert "meta/info.json" in err.getvalue(), "the message must name the fixable problem"


def test_bare_name_is_byte_identical_no_upload(monkeypatch, tmp_path):
    """A bare NAME launches exactly as before — the launch call is unchanged and NO
    upload is attempted. Regression guard for the path added by this card."""
    events: list[str] = []
    # If the cloud sink's urlopen is ever touched for a name, that's an upload — fail.
    _install_fake_console(monkeypatch, events, "svla_so101_pickplace")

    launched_with = {}

    def fake_launch(console, api_key, dataset, **k):
        events.append("launch")
        launched_with["dataset"] = dataset
        return _HANDLE

    monkeypatch.setattr(ft, "_launch", fake_launch)
    monkeypatch.setattr(ft, "_poll_status", lambda *a, **k: _SUCCEEDED)
    monkeypatch.setattr(ft.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("NT_API_KEY", "nt_" + "a" * 40)
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    rc = cmd_finetune(["--dataset", "svla_so101_pickplace"])
    assert rc == 0, f"stderr={err.getvalue()!r}"
    assert launched_with["dataset"] == "svla_so101_pickplace", "the bare name must reach _launch verbatim"
    assert "sign" not in events and "put" not in events, f"a name must not upload: {events}"


def test_path_with_separator_but_missing_dir_errors_not_name(monkeypatch, tmp_path):
    """An argument with a path separator that doesn't resolve to a directory is a bad
    path — NOT silently treated as a name (Rule 10). No launch happens."""
    events: list[str] = []
    monkeypatch.setattr(ft, "_launch", lambda *a, **k: events.append("launch") or _HANDLE)
    monkeypatch.setattr(ft, "_poll_status", lambda *a, **k: _SUCCEEDED)
    monkeypatch.setenv("NT_API_KEY", "nt_" + "a" * 40)
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    rc = cmd_finetune(["--dataset", str(tmp_path / "does" / "not" / "exist")])
    assert rc == 1
    assert events == [], "a bad path must not launch"
    assert "isn't an existing directory" in err.getvalue()


# ---------------------------------------------------------------------------
# Failure warmth — the survival line. A run that fails at a gate AFTER `train`
# says the checkpoint is safe; a failure at intake/train must NOT (no false
# comfort, Rule 10). Asserted on all three terminal render paths: watch,
# re-attach, --status. This is the exact "lost night" the card fixes.
# ---------------------------------------------------------------------------

_POST_TRAIN_GATES = ("frame-check", "registry+reload", "serve")
_NO_SURVIVE_GATES = ("intake", "train")


def _failed_at(gate):
    return {"status": "failed", "gate": gate, "tag": None, "report_card": None}


def _run_failed(path, failed, monkeypatch):
    """Drive a terminal failure through one of the three render paths."""
    if path == "watch":
        return _run(["--dataset", "d"], monkeypatch, launch=_HANDLE, statuses=[failed])
    if path == "reattach":
        return _run(["--handle", "fc-x"], monkeypatch, statuses=[failed])
    if path == "status":
        return _run(["--handle", "fc-x", "--status"], monkeypatch, statuses=[failed])
    raise AssertionError(f"unknown render path {path!r}")


@pytest.mark.parametrize("path", ["watch", "reattach", "status"])
@pytest.mark.parametrize("gate", _POST_TRAIN_GATES)
def test_survival_line_prints_for_post_train_gates(path, gate, monkeypatch):
    """A failure at frame-check / registry+reload / serve means training finished —
    the CLI must say the checkpoint is safe and nothing needs retraining, on EVERY
    terminal render path. The gate is still named."""
    _rc, out, _err = _run_failed(path, _failed_at(gate), monkeypatch)
    assert "your checkpoint is safe" in out, f"[{path}/{gate}] survival line missing: {out!r}"
    assert "nothing needs retraining" in out, f"[{path}/{gate}] retrain reassurance missing"
    assert "post-processing" in out, f"[{path}/{gate}] must locate the failure in post-processing"
    assert gate in out, f"[{path}/{gate}] the failing gate must still be named"


@pytest.mark.parametrize("path", ["watch", "reattach", "status"])
@pytest.mark.parametrize("gate", _NO_SURVIVE_GATES)
def test_survival_line_absent_for_intake_and_train(path, gate, monkeypatch):
    """A failure at intake or train means the checkpoint did NOT complete — the
    survival line must NOT print (Rule 10, no false comfort). This is the mutation
    guard: hard-code the 'safe' branch on and these assertions fail. The gate is
    still named so the developer knows what broke."""
    _rc, out, _err = _run_failed(path, _failed_at(gate), monkeypatch)
    assert "your checkpoint is safe" not in out, f"[{path}/{gate}] false comfort leaked: {out!r}"
    assert "nothing needs retraining" not in out, f"[{path}/{gate}] false retrain reassurance leaked"
    assert gate in out, f"[{path}/{gate}] the failing gate must still be named"


def test_survival_block_is_derived_only_from_gate_order():
    """The helper is a pure function of gate order — after-train gates get the block,
    intake/train get None, and a gate we can't PLACE gets None (surfaced by saying
    nothing, never a guessed 'safe')."""
    for g in _POST_TRAIN_GATES:
        assert _survival_block(g) is not None, g
    for g in _NO_SURVIVE_GATES:
        assert _survival_block(g) is None, g
    assert _survival_block(None) is None
    assert _survival_block("some-unknown-gate") is None


def test_render_terminal_post_train_failure_says_safe():
    text, code = _render_terminal(_failed_at("registry+reload"))
    assert code == 1
    assert "your checkpoint is safe" in text
    assert "registry+reload" in text


def test_render_terminal_train_failure_no_false_comfort():
    text, code = _render_terminal(_failed_at("train"))
    assert code == 1
    assert "checkpoint is safe" not in text


# ---------------------------------------------------------------------------
# `newt finetune --list` — the caller's recent runs. Owner-scoping is enforced
# server-side (the route's bun tests); here we prove the CLI faithfully renders
# only what its own key's row-set returns, emits the raw list under --json, and
# gives an empty owner a friendly line, not an error.
# ---------------------------------------------------------------------------

_KEY_A = "nt_" + "a" * 40
_KEY_C = "nt_" + "c" * 40


def test_list_renders_only_this_keys_rows(monkeypatch):
    """Two keys, disjoint owner row-sets: each key's --list shows ONLY its owner's
    runs (the CLI renders exactly what the owner-scoped route returned for that key)."""
    rows_by_key = {
        _KEY_A: [{"job_handle": "fc-a1", "dataset": "da", "status": "launched", "created_at": "2026-07-14T10:00:00Z"}],
        _KEY_C: [{"job_handle": "fc-c1", "dataset": "dc", "status": "launched", "created_at": "2026-07-14T09:00:00Z"}],
    }
    monkeypatch.setattr(ft, "_list_jobs", lambda console, api_key, **k: rows_by_key[api_key])

    _rc_a, out_a, _ = _run(["--list"], monkeypatch, key=_KEY_A)
    assert "fc-a1" in out_a and "fc-c1" not in out_a, out_a

    _rc_c, out_c, _ = _run(["--list"], monkeypatch, key=_KEY_C)
    assert "fc-c1" in out_c and "fc-a1" not in out_c, out_c


def test_list_empty_prints_friendly_launch_line(monkeypatch):
    """An owner with no runs gets a clear, friendly line naming the launch command —
    not an error, not a bare empty table."""
    monkeypatch.setattr(ft, "_list_jobs", lambda *a, **k: [])
    rc, out, _ = _run(["--list"], monkeypatch)
    assert rc == 0
    assert "no fine-tune runs" in out.lower()
    assert "--dataset" in out, "the empty-state line must name how to launch one"


def test_list_json_emits_raw_array(monkeypatch):
    """`--list --json` emits the raw runs array on a clean stdout — scriptable."""
    rows = [{"job_handle": "fc-a1", "dataset": "da", "status": "launched", "created_at": "t"}]
    monkeypatch.setattr(ft, "_list_jobs", lambda *a, **k: rows)
    rc, out, _ = _run(["--list", "--json"], monkeypatch)
    assert rc == 0
    assert json.loads(out) == rows


def test_render_jobs_table_columns_and_honest_state_caption():
    """The table carries handle/dataset/state/created, and its caption surfaces the
    staleness HONESTLY — the state column is the last recorded status, and live state
    comes from --status (no invented live state)."""
    rows = [{"job_handle": "fc-a1", "dataset": "da", "status": "launched", "created_at": "2026-07-14T10:00:00Z"}]
    text = _render_jobs_table(rows)
    for col in ("HANDLE", "DATASET", "STATE", "CREATED"):
        assert col in text, f"missing column {col}"
    assert "fc-a1" in text and "launched" in text
    assert "last recorded" in text.lower(), "the state column's staleness must be captioned"
    assert "--status" in text, "the caption must point at --status for live state"


def test_list_rejects_combination_with_dataset(monkeypatch):
    """`--list` is a standalone view — combining it with --dataset/--handle is a loud
    error, never a silent branch."""
    rc, _out, err = _run(["--list", "--dataset", "d"], monkeypatch)
    assert rc == 1
    assert "list" in err.lower()


def test_list_no_key_tells_user_what_to_do(monkeypatch):
    rc, _out, err = _run(["--list"], monkeypatch, key=None)
    assert rc == 1
    assert "login" in err.lower() or "NT_API_KEY" in err


def test_list_401_blames_the_key(monkeypatch):
    def _rejected(*a, **k):
        raise HTTPError("url", 401, "Unauthorized", {}, None)
    monkeypatch.setattr(ft, "_list_jobs", _rejected)
    rc, _out, err = _run(["--list"], monkeypatch)
    assert rc == 1
    assert "key" in err.lower() or "auth" in err.lower()


# ---------------------------------------------------------------------------
# Live round-trip — credential-gated, skips loudly without NEWT_E2E_KEY (never
# mocked green). Uploads a tiny real export to the real console and launches.
# ---------------------------------------------------------------------------

def test_live_path_upload_round_trip(tmp_path, capsys):
    key = os.environ.get("NEWT_E2E_KEY")
    if not key:
        pytest.skip("NEWT_E2E_KEY not set — skipping the live upload+launch round-trip")

    os.environ["NT_API_KEY"] = key
    export = _make_export(tmp_path, "newt-e2e-export")
    rc = cmd_finetune(["--dataset", str(export), "--json"])
    captured = capsys.readouterr()
    # stdout is the launch JSON (handle + terminal state); stderr carries the
    # human upload/staging lines. A staged name must have been printed.
    assert "staged as newt-e2e-export" in captured.err, captured.err
    parsed = json.loads(captured.out)
    assert parsed.get("job_handle"), f"live launch returned no handle: {captured.out!r}"
    assert rc in (0, 1)  # 0 succeeded / 1 failed-at-gate — both are real terminal states
