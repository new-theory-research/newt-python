"""Offline unit tests for `newt promote` — keep a checkpoint band and serve it.

No network: the one HTTP round-trip (`_promote`) is monkeypatched. Each test encodes
why the behavior matters:
  - a 201 names the new model and says plainly what happens next (safety check running);
  - EVERY 409 prints the server's `detail` VERBATIM (issue #28) — the single most
    important behavior on this verb: a swallowed reason is the failure it exists to
    prevent;
  - `already_promoted` also names the existing model's uid;
  - a 404 is the no-oracle copy pointing at `newt finetune --list`;
  - `--json` passes the route body through on a clean stdout (201 body AND error body);
  - a missing key renders the no-key block and returns 1 — never a keyless call;
  - a missing handle / missing --band is a house arg error.
"""
from __future__ import annotations

import io
import json
import sys
from urllib.error import HTTPError

import pytest

import newt._cli.promote as pr
from newt._cli.promote import cmd_promote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args, monkeypatch, *, response=None, error=None, key="nt_testkey"):
    """Run cmd_promote with `_promote` mocked, capturing stdout+stderr.

    `response` is the dict `_promote` returns (the 201 body). `error` is an HTTPError to
    raise instead (an error-path response). `key=None` forces the no-key path.
    Returns (exit_code, stdout, stderr).
    """
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    if key is None:
        monkeypatch.setattr(pr, "_resolve_key", lambda: None)
    else:
        monkeypatch.setattr(pr, "_resolve_key", lambda: key)

    called = {"n": 0}

    def _fake_promote(*a, **k):
        called["n"] += 1
        if error is not None:
            raise error
        return response

    monkeypatch.setattr(pr, "_promote", _fake_promote)

    rc = cmd_promote(args)
    return rc, out.getvalue(), err.getvalue(), called["n"]


def _http_error(code, body):
    """An HTTPError whose body is a JSON string with `error`/`detail`/`uid` — the shape
    the promote route returns on a refusal (a readable BytesIO, read exactly once)."""
    return HTTPError("url", code, "err", {}, io.BytesIO(json.dumps(body).encode()))


_OK = {
    "uid": "ft_team_launch1",
    "tag": "ft-team-so101-pickplace",
    "checkpoint_uri": "modal://vol/checkpoints/010000/pretrained_model/",
    "status": "pending",
}


# ---------------------------------------------------------------------------
# 201 — the model is registered, born pending; the output names it + what's next
# ---------------------------------------------------------------------------

def test_201_names_model_and_says_whats_next(monkeypatch):
    """A developer promotes a band and sees the new model's name, its `pending` status,
    and the plain what-happens-next line — not a bare success."""
    rc, out, err, n = _run(["fc-abc", "--band", "010000"], monkeypatch, response=_OK)
    assert rc == 0, f"stderr={err!r}"
    assert n == 1, "the route must be called once"
    assert "ft-team-so101-pickplace" in out, "the new model's tag must be named"
    assert "pending" in out, "the model's status must be shown"
    assert "safety check running" in out.lower(), "the what-happens-next copy must print"
    assert "newt models" in out, "it must point at `newt models` to watch admission"


def test_201_json_passes_body_through_clean_stdout(monkeypatch):
    """`--json` emits the route's 201 body verbatim on a clean stdout; the human lines
    go to stderr so stdout is scriptable."""
    rc, out, err, _ = _run(["fc-abc", "--band", "010000", "--json"], monkeypatch, response=_OK)
    assert rc == 0
    assert json.loads(out) == _OK, "stdout must be exactly the route body"
    assert "safety check running" in err.lower(), "human copy goes to stderr under --json"


def test_201_missing_tag_is_surfaced_not_fabricated(monkeypatch):
    """A 201 that somehow carries no tag is a contract violation — surfaced loud, never
    a fabricated model name (Rule 10)."""
    rc, _out, err, _ = _run(
        ["fc-abc", "--band", "010000"], monkeypatch, response={"uid": "x", "status": "pending"}
    )
    assert rc == 1
    assert "no model tag" in err


# ---------------------------------------------------------------------------
# Every 409 prints the server's `detail` VERBATIM (issue #28) — the load-bearing check
# ---------------------------------------------------------------------------

def test_409_band_not_promotable_detail_verbatim(monkeypatch):
    detail = "This checkpoint's eval hasn't completed, so it can't be served yet. Promote a band whose eval status is 'complete'."
    rc, _out, err, _ = _run(
        ["fc-abc", "--band", "002000"],
        monkeypatch,
        error=_http_error(409, {"error": "band_not_promotable", "detail": detail}),
    )
    assert rc == 1
    assert detail in err, "the server's plain reason must print word-for-word"


def test_409_checkpoint_uri_missing_detail_verbatim(monkeypatch):
    """The honest pre-nt-runway#25 live outcome — the checkpoint's location wasn't
    reported yet. Its detail must reach the developer verbatim, nt-runway#25 pointer and
    all."""
    detail = (
        "This checkpoint's location wasn't reported by the training pipeline, so it "
        "can't be served yet. Re-run this fine-tune once the pipeline reports checkpoint "
        "locations (nt-runway#25), and this band will be promotable."
    )
    rc, _out, err, _ = _run(
        ["fc-abc", "--band", "010000"],
        monkeypatch,
        error=_http_error(409, {"error": "checkpoint_uri_missing", "detail": detail}),
    )
    assert rc == 1
    assert detail in err
    assert "nt-runway#25" in err, "the producer-half pointer must survive verbatim"


def test_409_already_promoted_prints_detail_and_existing_uid(monkeypatch):
    """`already_promoted` prints its detail verbatim AND names the existing model's uid
    so the developer lands on it, and points at `newt models`."""
    detail = "This run already has a registered model — a run serves one model. Open it below."
    rc, _out, err, _ = _run(
        ["fc-abc", "--band", "010000"],
        monkeypatch,
        error=_http_error(
            409, {"error": "already_promoted", "detail": detail, "uid": "ft_existing_9f2a"}
        ),
    )
    assert rc == 1
    assert detail in err
    assert "ft_existing_9f2a" in err, "the existing model's uid must be surfaced"
    assert "newt models" in err


def test_409_json_passes_error_body_through(monkeypatch):
    """`--json` on a 409 passes the whole error body through on stdout — an agent gets
    `error`/`detail`/`uid` structured, not just a human line."""
    body = {"error": "already_promoted", "detail": "…", "uid": "ft_existing_9f2a"}
    rc, out, _err, _ = _run(
        ["fc-abc", "--band", "010000", "--json"], monkeypatch, error=_http_error(409, body)
    )
    assert rc == 1
    assert json.loads(out) == body, "stdout must carry the route's structured error body"


# ---------------------------------------------------------------------------
# 404 — the no-oracle copy (never confirm someone else's run)
# ---------------------------------------------------------------------------

def test_404_is_no_oracle_and_points_at_list(monkeypatch):
    rc, _out, err, _ = _run(
        ["fc-nope", "--band", "010000"],
        monkeypatch,
        error=_http_error(404, {"error": "not_found"}),
    )
    assert rc == 1
    assert "fc-nope" in err, "the handle must be named in the no-oracle copy"
    assert "newt finetune --list" in err, "it must point at how to find your runs"


# ---------------------------------------------------------------------------
# 400 — bad band token
# ---------------------------------------------------------------------------

def test_400_reports_band_error(monkeypatch):
    rc, _out, err, _ = _run(
        ["fc-abc", "--band", "01/00"],
        monkeypatch,
        error=_http_error(400, {"error": "bad_request"}),
    )
    assert rc == 1
    assert "band" in err.lower()


def test_400_surfaces_detail_when_present(monkeypatch):
    detail = "the band token must be alphanumeric"
    rc, _out, err, _ = _run(
        ["fc-abc", "--band", "01/00"],
        monkeypatch,
        error=_http_error(400, {"error": "bad_request", "detail": detail}),
    )
    assert rc == 1
    assert detail in err


# ---------------------------------------------------------------------------
# 401 — key was rejected → the login path
# ---------------------------------------------------------------------------

def test_401_points_at_login(monkeypatch):
    rc, _out, err, _ = _run(
        ["fc-abc", "--band", "010000"],
        monkeypatch,
        error=_http_error(401, {"error": "unauthorized"}),
    )
    assert rc == 1
    assert "login" in err.lower() or "NT_API_KEY" in err


# ---------------------------------------------------------------------------
# Argument + auth guards
# ---------------------------------------------------------------------------

def test_missing_key_renders_no_key_block_and_never_calls_route(monkeypatch):
    """No key → the no-key block (`run newt login`) and return 1 — never a keyless call
    to the route (Rule 10, the missing-key silent fall-through the validator flags)."""
    rc, _out, err, n = _run(["fc-abc", "--band", "010000"], monkeypatch, response=_OK, key=None)
    assert rc == 1
    assert "login" in err.lower() or "NT_API_KEY" in err
    assert n == 0, "the route must NOT be called without a key"


def test_missing_band_is_house_arg_error(monkeypatch):
    rc, _out, err, n = _run(["fc-abc"], monkeypatch, response=_OK)
    assert rc == 1
    assert "--band" in err
    assert n == 0, "no network call on a missing --band"


def test_missing_handle_is_house_arg_error(monkeypatch):
    rc, _out, err, n = _run(["--band", "010000"], monkeypatch, response=_OK)
    assert rc == 1
    assert "handle" in err.lower()
    assert n == 0, "no network call on a missing handle"


def test_handle_found_after_band_flag(monkeypatch):
    """`newt promote --band 010000 fc-abc` still finds fc-abc as the handle — the
    space-form --band value doesn't swallow the positional after it."""
    rc, out, err, n = _run(["--band", "010000", "fc-abc"], monkeypatch, response=_OK)
    assert rc == 0, f"stderr={err!r}"
    assert n == 1


# ---------------------------------------------------------------------------
# No argparse / no new HTTP library — the flat-branch, stdlib-only conformance (Rule 9)
# ---------------------------------------------------------------------------

def test_promote_module_uses_stdlib_only():
    import pathlib

    source = pathlib.Path(pr.__file__).read_text()
    for forbidden in ("import argparse", "import requests", "import httpx"):
        assert forbidden not in source, f"promote must not use {forbidden!r} (Rule 9)"


# ---------------------------------------------------------------------------
# Help guard — exits 0, does nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_guard_exits_zero_no_call(flag, monkeypatch):
    rc, out, _err, n = _run([flag], monkeypatch, response=_OK)
    assert rc == 0
    assert n == 0, "the help guard must not call the route"
    assert "promote" in out, "usage must print on --help"
