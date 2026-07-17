"""Offline unit tests for the `newt run` CLI command.

Tests exercise the developer-visible contract WITHOUT touching the network: the
`Robot`/`infer` boundary is monkeypatched, snapshots load from the real bundled
assets. Each test encodes WHY the behavior matters, not just what it does — the
honesty seams (framing line present, `model_status` surfaced verbatim, no-key →
login hint, unknown snapshot → helpful list) are the load-bearing ones.
"""
from __future__ import annotations

import io
import json
import sys

import numpy as np
import pytest

import newt
from newt._client.robot import (
    AuthError,
    BaseNotDeployableError,
    ContractMismatchError,
    InferenceResponse,
    ModelNotFoundError,
    ProtocolError,
    RegistryUnavailable,
    ServerError,
    VerifierError,
)
from newt._cli.run import cmd_run


_AXES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]


def _fake_response(model="fixture-base-pour", latency_ms=42.0):
    """A labeled action chunk shaped like a real infer() return (50, 8)."""
    chunk = np.zeros((50, 8), dtype=np.float32)
    return InferenceResponse(action_chunk=chunk, axes=list(_AXES), latency_ms=latency_ms, model=model)


class _FakeRobot:
    """Stand-in for newt.Robot — records the model it was built with, returns a canned
    response from infer(), or raises a preset exception. Never touches the network."""

    def __init__(self, *, api_key=None, model=None, infer_return=None, infer_raises=None, **kw):
        self.model = model
        self._infer_return = infer_return
        self._infer_raises = infer_raises
        self.infer_obs = None

    def infer(self, obs, prompt=None):
        self.infer_obs = obs
        if self._infer_raises is not None:
            raise self._infer_raises
        return self._infer_return


def _run(args, monkeypatch, *, infer_return=None, infer_raises=None, construct_raises=None, key="nt_testkey"):
    """Invoke cmd_run with a mocked Robot boundary, capturing stdout/stderr."""
    out, err = io.StringIO(), io.StringIO()
    captured = {}

    def fake_robot_cls(*a, **kw):
        captured["model"] = kw.get("model")
        if construct_raises is not None:
            raise construct_raises
        return _FakeRobot(infer_return=infer_return, infer_raises=infer_raises, **kw)

    monkeypatch.setattr(newt, "Robot", fake_robot_cls)
    if key is not None:
        monkeypatch.setenv("NT_API_KEY", key)
    else:
        monkeypatch.delenv("NT_API_KEY", raising=False)
        import newt._cli.run as run_mod
        monkeypatch.setattr(run_mod, "read_api_key", lambda: None)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    code = cmd_run(args)
    return code, out.getvalue(), err.getvalue(), captured


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_prints_model_latency_shape_and_framing(monkeypatch):
    """A developer types `newt run <tag>` and sees their model answer for real.

    The output must carry the four load-bearing facts: the resolved model, the
    round-trip latency, the action-chunk SHAPE (not the raw vector), and the honest
    framing line — because a live inference is not a live robot and the verb must
    never let that ambiguity stand.
    """
    resp = _fake_response(model="fixture-base-pour", latency_ms=37.0)
    code, out, err, _ = _run(["fixture-base-pour"], monkeypatch, infer_return=resp)

    assert code == 0, f"expected exit 0; stderr={err!r}"
    assert err == ""
    assert "fixture-base-pour" in out, "resolved model must appear"
    assert "37ms" in out, "latency must appear"
    assert "(50, 8)" in out, "action-chunk shape summary must appear"
    assert "gripper" in out, "axis labels must appear"
    # Honest framing — no robot, nothing moved
    assert "No robot is connected" in out and "nothing moved" in out, (
        f"framing line must state no robot connected / nothing moved: {out!r}"
    )
    # Never dump the raw action vector in human output
    assert "0.0" not in out and "[[" not in out, "human output must not dump the raw vector"


def test_happy_path_passes_tag_to_robot(monkeypatch):
    """The positional <tag> is what constructs Robot(model=tag) — the SDK does the
    registry fetch + tag→endpoint resolve inside its own constructor."""
    resp = _fake_response()
    code, _, _, captured = _run(["my-cool-tag"], monkeypatch, infer_return=resp)
    assert code == 0
    assert captured["model"] == "my-cool-tag", "Robot must be built with the given tag"


def test_default_snapshot_is_cup_stacking(monkeypatch):
    """With no --snapshot, the docs' own example (cup_stacking) is loaded and its
    recorded prompt rides in the obs."""
    resp = _fake_response()
    captured_obs = {}

    def fake_robot_cls(*a, **kw):
        r = _FakeRobot(infer_return=resp, **kw)
        orig = r.infer

        def infer(obs, prompt=None):
            captured_obs["obs"] = obs
            return orig(obs, prompt)

        r.infer = infer
        return r

    monkeypatch.setattr(newt, "Robot", fake_robot_cls)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    code = cmd_run(["some-tag"])
    assert code == 0, f"stderr={err.getvalue()!r}"
    assert captured_obs["obs"]["prompt"] == "Stack one cup into another cup.", (
        "cup_stacking's canonical prompt must ride in the obs"
    )


# ---------------------------------------------------------------------------
# --snapshot selection + --prompt override
# ---------------------------------------------------------------------------

def test_snapshot_selection_loads_pour_coffee_beans(monkeypatch):
    """`--snapshot pour_coffee_beans` loads that observation, not the default."""
    resp = _fake_response()
    captured_obs = {}

    def fake_robot_cls(*a, **kw):
        r = _FakeRobot(infer_return=resp, **kw)

        def infer(obs, prompt=None):
            captured_obs["obs"] = obs
            return resp

        r.infer = infer
        return r

    monkeypatch.setattr(newt, "Robot", fake_robot_cls)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    code = cmd_run(["some-tag", "--snapshot", "pour_coffee_beans"])
    assert code == 0, f"stderr={err.getvalue()!r}"
    # pour_coffee_beans keeps its recorded prompt (no override in the snapshot map)
    assert captured_obs["obs"]["prompt"], "pour_coffee_beans must carry its recorded prompt"
    assert "surrounding1" in captured_obs["obs"]["images"], "obs must carry the camera frames"


def test_prompt_override_replaces_snapshot_prompt(monkeypatch):
    """`--prompt` overrides the snapshot's recorded prompt when passed."""
    resp = _fake_response()
    captured_obs = {}

    def fake_robot_cls(*a, **kw):
        r = _FakeRobot(infer_return=resp, **kw)

        def infer(obs, prompt=None):
            captured_obs["obs"] = obs
            return resp

        r.infer = infer
        return r

    monkeypatch.setattr(newt, "Robot", fake_robot_cls)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    code = cmd_run(["some-tag", "--prompt", "put the block in the bowl"])
    assert code == 0, f"stderr={err.getvalue()!r}"
    assert captured_obs["obs"]["prompt"] == "put the block in the bowl", (
        "--prompt must override the snapshot's recorded prompt"
    )


# ---------------------------------------------------------------------------
# --json
# ---------------------------------------------------------------------------

def test_json_mirror_is_structured_and_prose_free(monkeypatch):
    """`--json` emits the machine-readable mirror — tag, model, latency, shape/axes —
    and NO framing prose (that belongs only in the human output)."""
    resp = _fake_response(model="fixture-base-pour", latency_ms=42.0)
    code, out, err, _ = _run(["fixture-base-pour", "--json"], monkeypatch, infer_return=resp)

    assert code == 0, f"stderr={err!r}"
    data = json.loads(out)
    assert data["tag"] == "fixture-base-pour"
    assert data["model"] == "fixture-base-pour"
    assert data["latency_ms"] == 42.0
    assert data["action_chunk"]["shape"] == [50, 8]
    assert data["action_chunk"]["axes"] == _AXES
    assert data["snapshot"] == "cup_stacking"
    # No framing prose in the machine mirror
    assert "No robot is connected" not in out, "--json must not carry framing prose"


# ---------------------------------------------------------------------------
# Missing tag / missing key
# ---------------------------------------------------------------------------

def test_missing_tag_prints_house_error(monkeypatch):
    """`newt run` with no tag → the house 2-line error naming the shape, not a crash."""
    code, out, err, _ = _run([], monkeypatch)
    assert code == 1
    assert err.startswith("newt: "), f"must be the house error shape: {err!r}"
    assert "tag" in err.lower()
    assert "newt run <tag>" in err, "must name the command shape"


def test_missing_key_tells_user_to_login(monkeypatch):
    """No key → the no-key block naming `newt login`, return 1, no retry loop (Rule 10)."""
    code, out, err, _ = _run(["some-tag"], monkeypatch, key=None)
    assert code == 1
    assert "login" in err.lower() or "NT_API_KEY" in err, (
        f"must tell the developer how to authenticate: {err!r}"
    )


# ---------------------------------------------------------------------------
# Unknown snapshot
# ---------------------------------------------------------------------------

def test_unknown_snapshot_lists_available_names(monkeypatch):
    """`--snapshot nope` → a helpful 2-line error listing the available names, never a
    raw KeyError traceback."""
    code, out, err, _ = _run(["some-tag", "--snapshot", "nope"], monkeypatch)
    assert code == 1
    assert err.startswith("newt: "), f"house error shape: {err!r}"
    assert "nope" in err
    assert "cup_stacking" in err and "pour_coffee_beans" in err, (
        f"must list the available snapshot names: {err!r}"
    )
    assert "Traceback" not in err, "must not be a raw traceback"


# ---------------------------------------------------------------------------
# Error render matrix — house shape `newt: <problem> — <hint>` to stderr, exit 1
# ---------------------------------------------------------------------------

def _auth_error():
    return AuthError(
        code=4001,
        type="auth.invalid_key",
        message="API key rejected by the inference server.",
        context={},
    )


def test_auth_error_renders_login_hint(monkeypatch):
    """AuthError → house error + the `newt login` hint (the single fix for a bad key)."""
    code, out, err, _ = _run(["some-tag"], monkeypatch, construct_raises=_auth_error())
    assert code == 1
    assert err.startswith("newt: "), f"house shape: {err!r}"
    assert "API key rejected by the inference server." in err, "server message surfaced"
    assert "newt login" in err, "must hint `newt login`"


def test_model_not_found_renders_message(monkeypatch):
    """ModelNotFoundError → house error carrying the SDK's known-models message."""
    exc = ModelNotFoundError(model="typo-tag", known=["fixture-base", "fixture-base-pour"])
    code, out, err, _ = _run(["typo-tag"], monkeypatch, construct_raises=exc)
    assert code == 1
    assert err.startswith("newt: ")
    assert "not found" in err.lower()
    assert "typo-tag" in err, "the bad tag must be named"


def test_base_not_deployable_renders_message(monkeypatch):
    """BaseNotDeployableError → house error; a base tag isn't directly runnable."""
    exc = BaseNotDeployableError(
        code=4424,
        type="model.base_not_deployable",
        message="'fixture-base' is a base — run one of its fine-tunes: fixture-base-pour.",
        context={"model": "fixture-base", "fine_tunes": ["fixture-base-pour"]},
    )
    code, out, err, _ = _run(["fixture-base"], monkeypatch, construct_raises=exc)
    assert code == 1
    assert err.startswith("newt: ")
    assert "base" in err.lower()
    assert "fixture-base-pour" in err, "the deployable fine-tunes must survive to the user"


def test_registry_unavailable_renders_message(monkeypatch):
    """RegistryUnavailable → house error; the registry itself is unreachable."""
    exc = RegistryUnavailable(
        bootstrap_url="https://nt-registry-production.up.railway.app",
        reason="connection refused",
    )
    code, out, err, _ = _run(["some-tag"], monkeypatch, construct_raises=exc)
    assert code == 1
    assert err.startswith("newt: ")
    assert "registry" in err.lower() or "unreachable" in err.lower()


def test_contract_mismatch_renders_message(monkeypatch):
    """ContractMismatchError → house error carrying the expected-vs-got detail."""
    exc = ContractMismatchError(
        code=4422,
        type="contract_mismatch.state_shape",
        message="state shape mismatch: model expects (14,), obs carried (8,).",
        context={"model": "pi05_aloha"},
    )
    code, out, err, _ = _run(["pi05_aloha"], monkeypatch, infer_raises=exc)
    assert code == 1
    assert err.startswith("newt: ")
    assert "contract" in err.lower() or "mismatch" in err.lower()
    assert "(14,)" in err, "the expected shape must survive verbatim"


def test_server_error_surfaces_model_status_verbatim(monkeypatch):
    """The single most important honesty check (Rule 10): a ServerError whose detail
    carries `model_status` — a developer's OWN model that is pending/dead — surfaces
    that status VERBATIM, never collapsed into a generic 'server error' line. The whole
    point of running your own tag is to learn WHY it isn't answering."""
    exc = ServerError(
        code=4500,
        type="server.model_not_ready",
        message="Model is not currently servable.",
        context={"model_status": "pending: checkpoint still loading (est. 90s)"},
    )
    code, out, err, _ = _run(["my-own-ft"], monkeypatch, infer_raises=exc)
    assert code == 1
    assert err.startswith("newt: ")
    # The server-authored message survives...
    assert "Model is not currently servable." in err
    # ...AND the pending status rides out verbatim, not swallowed
    assert "pending: checkpoint still loading (est. 90s)" in err, (
        f"model_status must be surfaced verbatim: {err!r}"
    )


def test_verifier_error_renders_message(monkeypatch):
    """VerifierError → house error; the key verifier was unavailable at handshake."""
    exc = VerifierError(
        code=4503,
        type="verifier.unavailable",
        message="Key verifier temporarily unavailable — retry shortly.",
        context={},
    )
    code, out, err, _ = _run(["some-tag"], monkeypatch, infer_raises=exc)
    assert code == 1
    assert err.startswith("newt: ")
    assert "verifier" in err.lower()


def test_protocol_error_renders_message(monkeypatch):
    """ProtocolError → house error; the obs frame couldn't be parsed."""
    exc = ProtocolError(
        code=4400,
        type="protocol.malformed_msgpack",
        message="Obs frame could not be parsed.",
        context={},
    )
    code, out, err, _ = _run(["some-tag"], monkeypatch, infer_raises=exc)
    assert code == 1
    assert err.startswith("newt: ")
    assert "protocol" in err.lower()


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

def test_help_prints_usage_and_exits_zero(monkeypatch):
    """`newt run --help` prints usage including the honest framing, exit 0."""
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    code = cmd_run(["--help"])
    assert code == 0
    text = out.getvalue()
    assert "Usage: newt run <tag>" in text
    assert "No robot is connected" in text, "usage must carry the honest framing too"


# ---------------------------------------------------------------------------
# Issue #38 — honest cold-start latency: split rendering, --json additive fields,
# the waking-line stderr notice, and warm-path byte-stability.
#
# The receipt: a cold fine-tune's first call printed `latency 83609ms` as if the
# whole container-wake wait were steady-state inference speed; a warm call
# answered in 1863ms. The fix never fabricates a duration — it renders the two
# REAL numbers the SDK measured (the final attempt's own latency_ms, and total_ms
# for the whole call) and says only what the client actually knows (a retry
# happened / the call ran long), never a guessed ETA.
# ---------------------------------------------------------------------------


def test_cold_start_renders_honest_split_not_bare_latency(monkeypatch):
    """A call whose total_ms dwarfs its final-attempt latency_ms must render the
    split — never print the inflated number bare as if it were `newt run`'s
    steady-state speed.

    If this regresses: a developer's first call against a cold fine-tune sees
    `latency 83609ms` with no context and concludes their model is catastrophically
    slow, when 99% of calls (once warm) look nothing like that.
    """
    chunk = np.zeros((50, 8), dtype=np.float32)
    resp = InferenceResponse(
        action_chunk=chunk, axes=list(_AXES), latency_ms=1863.0, model="so101-red-cube-bowl",
        total_ms=83609.0, retries=1,
    )
    code, out, err, _ = _run(["so101-red-cube-bowl"], monkeypatch, infer_return=resp)

    assert code == 0, f"stderr={err!r}"
    assert "1863ms" in out, "the real final-attempt latency must still be named"
    assert "83s total" in out, "the real total elapsed time must be named"
    assert "woke" in out.lower(), "the honest cause (cold start) must be named in plain language"
    # No fabricated ETA: a specific "will take Nmin" claim is never printed — only
    # what the client measured (elapsed) or is retrying, per issue #38's discipline.
    assert "60-90" not in out and "30-90" not in out


def test_verifier_retry_with_small_gap_still_renders_honest_note(monkeypatch):
    """A retry that resolved fast (small total-vs-latency gap) still must say a
    retry happened — the render trigger is 'gap dwarfs latency OR retries > 0',
    not gap size alone, so a fast-resolving retry isn't silently hidden."""
    chunk = np.zeros((50, 8), dtype=np.float32)
    resp = InferenceResponse(
        action_chunk=chunk, axes=list(_AXES), latency_ms=180.0, model="fixture-base-pour",
        total_ms=3400.0, retries=1,
    )
    code, out, err, _ = _run(["fixture-base-pour"], monkeypatch, infer_return=resp)

    assert code == 0
    assert "180ms" in out
    assert "retr" in out.lower(), "a retry that happened must be named even with a small gap"


def test_warm_path_render_is_byte_unchanged(monkeypatch):
    """The ordinary warm path — no retries, total_ms == latency_ms — must render
    EXACTLY the pre-#38 single-line latency, byte for byte. The split machinery
    must never fire on a call with nothing to explain."""
    resp = _fake_response(model="fixture-base-pour", latency_ms=37.0)
    code, out, err, _ = _run(["fixture-base-pour"], monkeypatch, infer_return=resp)

    assert code == 0
    assert "  latency   37ms\n" in out
    assert "total" not in out
    assert "woke" not in out.lower()
    assert "retr" not in out.lower()


def test_json_gains_total_ms_and_retries_additively(monkeypatch):
    """--json's existing fields are untouched; total_ms/retries are additive."""
    chunk = np.zeros((50, 8), dtype=np.float32)
    resp = InferenceResponse(
        action_chunk=chunk, axes=list(_AXES), latency_ms=1863.0, model="so101-red-cube-bowl",
        total_ms=83609.0, retries=1,
    )
    code, out, err, _ = _run(["so101-red-cube-bowl", "--json"], monkeypatch, infer_return=resp)

    assert code == 0, f"stderr={err!r}"
    data = json.loads(out)
    # Existing fields, unchanged shape/values.
    assert data["latency_ms"] == 1863.0
    assert data["model"] == "so101-red-cube-bowl"
    assert data["action_chunk"]["shape"] == [50, 8]
    # New, additive fields.
    assert data["total_ms"] == 83609.0
    assert data["retries"] == 1


def test_json_warm_path_total_ms_defaults_to_latency_ms(monkeypatch):
    """A warm response built without total_ms/retries (the common case) must still
    carry the new --json fields, defaulting sanely rather than raising."""
    resp = _fake_response(model="fixture-base-pour", latency_ms=42.0)
    code, out, err, _ = _run(["fixture-base-pour", "--json"], monkeypatch, infer_return=resp)

    assert code == 0
    data = json.loads(out)
    assert data["total_ms"] == 42.0
    assert data["retries"] == 0


# ---------------------------------------------------------------------------
# The waking-line stderr notice — TTY-only, never --json, never repeated
# ---------------------------------------------------------------------------


class _SlowFakeRobot(_FakeRobot):
    """Like _FakeRobot, but infer() sleeps briefly before returning — simulates a
    call running past the waking-line threshold without a real multi-second wait."""

    def __init__(self, *, delay_s=0.05, **kw):
        super().__init__(**kw)
        self._delay_s = delay_s

    def infer(self, obs, prompt=None):
        import time as _time
        _time.sleep(self._delay_s)
        return super().infer(obs, prompt)


def _run_slow(args, monkeypatch, *, infer_return, threshold=0.01, delay_s=0.05, key="nt_testkey"):
    """Like _run(), but with a slow FakeRobot and a tiny waking-line threshold so
    the timer fires within the test's own delay, without a real multi-second sleep."""
    out, err = io.StringIO(), io.StringIO()

    def fake_robot_cls(*a, **kw):
        return _SlowFakeRobot(delay_s=delay_s, infer_return=infer_return, **kw)

    monkeypatch.setattr(newt, "Robot", fake_robot_cls)
    monkeypatch.setenv("NT_API_KEY", key)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    import newt._cli.run as run_mod
    monkeypatch.setattr(run_mod, "_WAKING_THRESHOLD_S", threshold)

    code = cmd_run(args)
    return code, out.getvalue(), err.getvalue()


def test_waking_line_prints_on_stderr_when_tty_and_slow(monkeypatch):
    """A slow call, human output, stderr as a TTY → the #38 waking line prints to
    stderr exactly once, and stdout stays clean of it.

    If this regresses: a developer waiting on a genuinely cold container sees
    total silence for the whole wait, exactly the friction issue #38 reports.
    """
    resp = _fake_response()
    out_io, err_io = io.StringIO(), io.StringIO()
    monkeypatch.setattr(err_io, "isatty", lambda: True)

    def fake_robot_cls(*a, **kw):
        return _SlowFakeRobot(delay_s=0.05, infer_return=resp, **kw)

    monkeypatch.setattr(newt, "Robot", fake_robot_cls)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")
    monkeypatch.setattr(sys, "stdout", out_io)
    monkeypatch.setattr(sys, "stderr", err_io)
    import newt._cli.run as run_mod
    monkeypatch.setattr(run_mod, "_WAKING_THRESHOLD_S", 0.01)

    code = cmd_run(["some-tag"])

    assert code == 0
    assert err_io.getvalue().count("model is waking up") == 1, (
        f"waking line must print exactly once: {err_io.getvalue()!r}"
    )
    assert "waking" not in out_io.getvalue(), "the waking line must never land on stdout"


def test_waking_line_absent_when_stderr_not_a_tty(monkeypatch):
    """Same slow call, but stderr is NOT a TTY (the default io.StringIO, and every
    CI/redirected-output run) → no waking line. Piping `newt run` output must not
    inject an extra line a script doesn't expect."""
    resp = _fake_response()
    code, out, err = _run_slow(["some-tag"], monkeypatch, infer_return=resp, threshold=0.01, delay_s=0.05)
    assert code == 0
    assert "waking" not in err, f"non-TTY stderr must never see the waking line: {err!r}"


def test_waking_line_absent_under_json_even_if_tty(monkeypatch):
    """--json must never print the waking line, even on a slow call with stderr
    as a TTY — a script parsing --json output must never see extra stderr noise
    mid-call that it didn't ask for."""
    resp = _fake_response()
    out_io, err_io = io.StringIO(), io.StringIO()
    monkeypatch.setattr(err_io, "isatty", lambda: True)

    def fake_robot_cls(*a, **kw):
        return _SlowFakeRobot(delay_s=0.05, infer_return=resp, **kw)

    monkeypatch.setattr(newt, "Robot", fake_robot_cls)
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")
    monkeypatch.setattr(sys, "stdout", out_io)
    monkeypatch.setattr(sys, "stderr", err_io)
    import newt._cli.run as run_mod
    monkeypatch.setattr(run_mod, "_WAKING_THRESHOLD_S", 0.01)

    code = cmd_run(["some-tag", "--json"])

    assert code == 0
    assert "waking" not in err_io.getvalue(), "--json must never see the waking line"
    assert json.loads(out_io.getvalue())["tag"] == "some-tag"


def test_waking_line_absent_on_fast_call(monkeypatch):
    """A fast (non-slow) call must never print the waking line, even with a tiny
    threshold and stderr as a TTY — the timer must be cancelled once fn() returns,
    not left to fire after the fact."""
    resp = _fake_response()
    out_io, err_io = io.StringIO(), io.StringIO()
    monkeypatch.setattr(err_io, "isatty", lambda: True)
    monkeypatch.setattr(newt, "Robot", lambda *a, **kw: _FakeRobot(infer_return=resp, **kw))
    monkeypatch.setenv("NT_API_KEY", "nt_testkey")
    monkeypatch.setattr(sys, "stdout", out_io)
    monkeypatch.setattr(sys, "stderr", err_io)
    import newt._cli.run as run_mod
    # A real threshold (5s) — the fast fake call returns immediately, well under it.
    monkeypatch.setattr(run_mod, "_WAKING_THRESHOLD_S", 5.0)

    code = cmd_run(["some-tag"])

    assert code == 0
    assert "waking" not in err_io.getvalue()
