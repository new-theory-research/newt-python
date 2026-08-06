"""Offline unit tests for `newt episodes pull` and `newt episodes push`.

No network: pull's two HTTP round-trips (`_fetch_manifest`, `_download_url`) are
monkeypatched; push's console pre-check (`_list_dataset_objects`) is monkeypatched and the
whole GCS transport under the real `NTCloudSink` is faked at
`newt.recording._cloud_sink.urlopen`, so a push test sees exactly what would have crossed the
wire. Files are written into `tmp_path`. Each test encodes why the behavior matters.

pull:
  - the manifest is fetched authed, then every file is downloaded into --dest, recreating the
    dataset's relative layout;
  - progress is reported by FILES COMPLETED, never an invented percentage (Rule 10);
  - a rerun is resumable — a file already present at the manifest's size is SKIPPED, not
    re-downloaded;
  - `--json` emits a machine-readable result (files, bytes) on a clean stdout;
  - a missing key prints the `newt login` hint; a 404 names the dataset.

push (the write-once verb — a refusal that arrives late costs a name that can never be
reused, portal#98):
  - upload ORDER survives the CLI end to end: episode.json last within an episode,
    manifest.json last of everything, episodes never interleaved;
  - every refusal lands BEFORE the first sign call and the first PUT — asserted by counting
    them, not by reading the message;
  - the six refusal causes never share one string (Rule 12), and neither do the three shapes
    of "nothing to push";
  - a PUT that fails mid-push reports how many episodes landed and never exits 0 (Rule 10);
  - `--json` keeps stdout pure and the preflight report on stderr.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import newt._cli.episodes as ep
from newt._cli.episodes import cmd_episodes, _should_skip


def _manifest(files):
    """A download manifest: files is a list of (path, body_bytes) → entries with sizes."""
    return {
        "namespace": "ns0",
        "dataset": "my-ds",
        "count": len(files),
        "urls": [
            {"path": p, "url": f"https://storage.googleapis.com/nt-episodes/{p}?sig=x",
             "expiresAt": "2026-01-01T00:30:00.000Z", "size": len(b)}
            for p, b in files
        ],
    }


def _run(args, monkeypatch, *, manifest=None, bodies=None, key="nt_testkey",
         manifest_error=None):
    """Run cmd_episodes(args) with mocked HTTP, capturing stdout+stderr.

    `manifest` is the dict `_fetch_manifest` returns; `manifest_error` an exception it raises.
    `bodies` maps a relative path → the bytes `_download_url` returns for its url. Records the
    set of paths actually downloaded so resume tests can assert a skip.
    """
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    if key is None:
        monkeypatch.delenv("NT_API_KEY", raising=False)
        monkeypatch.setattr(ep, "read_api_key", lambda: None)
    else:
        monkeypatch.setenv("NT_API_KEY", key)

    if manifest_error is not None:
        def _fetch(*a, **k):
            raise manifest_error
        monkeypatch.setattr(ep, "_fetch_manifest", _fetch)
    elif manifest is not None:
        monkeypatch.setattr(ep, "_fetch_manifest", lambda *a, **k: manifest)

    downloaded_paths: list[str] = []
    if bodies is not None:
        def _dl(url, **k):
            # Recover the path from the fake signed URL's tail.
            path = url.split("nt-episodes/", 1)[1].split("?", 1)[0]
            downloaded_paths.append(path)
            return bodies[path]
        monkeypatch.setattr(ep, "_download_url", _dl)

    rc = cmd_episodes(args)
    return rc, out.getvalue(), err.getvalue(), downloaded_paths


# ---------------------------------------------------------------------------
# Golden: fetch manifest → download every file into --dest
# ---------------------------------------------------------------------------

def test_pull_downloads_all_files_into_dest(monkeypatch, tmp_path):
    files = [("meta/info.json", b'{"k":1}'), ("data/chunk-000/f-000.parquet", b"PARQUETDATA")]
    dest = tmp_path / "out"
    rc, out, err, dl = _run(
        ["pull", "my-ds", "--dest", str(dest)],
        monkeypatch,
        manifest=_manifest(files),
        bodies={p: b for p, b in files},
    )
    assert rc == 0, f"stderr={err!r}"
    # Every file landed under --dest, recreating the relative layout.
    assert (dest / "meta/info.json").read_bytes() == b'{"k":1}'
    assert (dest / "data/chunk-000/f-000.parquet").read_bytes() == b"PARQUETDATA"
    assert sorted(dl) == ["data/chunk-000/f-000.parquet", "meta/info.json"]


def test_pull_default_dest_is_dataset_name(monkeypatch, tmp_path):
    files = [("meta/info.json", b"x")]
    monkeypatch.chdir(tmp_path)
    rc, out, err, dl = _run(
        ["pull", "my-ds"], monkeypatch, manifest=_manifest(files), bodies={"meta/info.json": b"x"}
    )
    assert rc == 0, f"stderr={err!r}"
    # Default dest is ./<dataset>.
    assert (tmp_path / "my-ds" / "meta/info.json").read_bytes() == b"x"


# ---------------------------------------------------------------------------
# Progress: files-completed, NEVER an invented percentage
# ---------------------------------------------------------------------------

def test_progress_is_files_completed_never_a_percentage(monkeypatch, tmp_path):
    files = [("a.txt", b"aa"), ("b.txt", b"bbbb"), ("c.txt", b"c")]
    dest = tmp_path / "out"
    rc, out, err, dl = _run(
        ["pull", "my-ds", "--dest", str(dest)],
        monkeypatch,
        manifest=_manifest(files),
        bodies={p: b for p, b in files},
    )
    assert rc == 0
    # Files-completed counters appear; NO invented percentage anywhere.
    assert "1/3" in out and "2/3" in out and "3/3" in out
    assert "%" not in out, "progress must never print a percentage it can't source"


# ---------------------------------------------------------------------------
# Resume: a rerun skips a file already present at the manifest's size
# ---------------------------------------------------------------------------

def test_rerun_skips_files_already_present_at_matching_size(monkeypatch, tmp_path):
    files = [("meta/info.json", b'{"k":1}'), ("big.bin", b"0123456789")]
    dest = tmp_path / "out"
    # Pre-create ONE file at the exact manifest size (a completed prior pull of it).
    (dest / "meta").mkdir(parents=True)
    (dest / "meta/info.json").write_bytes(b'{"k":1}')  # 7 bytes, matches

    rc, out, err, dl = _run(
        ["pull", "my-ds", "--dest", str(dest)],
        monkeypatch,
        manifest=_manifest(files),
        bodies={p: b for p, b in files},
    )
    assert rc == 0, f"stderr={err!r}"
    # The already-present, size-matching file was NOT re-downloaded; the other WAS.
    assert dl == ["big.bin"], f"expected only big.bin downloaded, got {dl}"
    assert "already present" in out
    assert (dest / "big.bin").read_bytes() == b"0123456789"


def test_size_mismatch_is_redownloaded_not_skipped(monkeypatch, tmp_path):
    files = [("f.bin", b"1234567890")]  # size 10
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "f.bin").write_bytes(b"OLD")  # size 3, mismatched → must re-download
    rc, out, err, dl = _run(
        ["pull", "my-ds", "--dest", str(dest)],
        monkeypatch, manifest=_manifest(files), bodies={"f.bin": b"1234567890"}
    )
    assert rc == 0
    assert dl == ["f.bin"], "a size mismatch must re-download, never assume complete"
    assert (dest / "f.bin").read_bytes() == b"1234567890"


def test_should_skip_never_matches_on_unknown_size(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    # A null/unknown size never fabricates a match (Rule 10).
    assert _should_skip(p, None) is False
    assert _should_skip(p, 3) is True
    assert _should_skip(p, 4) is False
    assert _should_skip(tmp_path / "missing", 3) is False


# ---------------------------------------------------------------------------
# --json shape
# ---------------------------------------------------------------------------

def test_json_emits_machine_readable_result_on_clean_stdout(monkeypatch, tmp_path):
    files = [("a", b"aa"), ("b", b"bbb")]
    dest = tmp_path / "out"
    # Pre-place "a" at matching size so the result shows one skip.
    dest.mkdir()
    (dest / "a").write_bytes(b"aa")
    rc, out, err, dl = _run(
        ["pull", "my-ds", "--dest", str(dest), "--json"],
        monkeypatch, manifest=_manifest(files), bodies={"a": b"aa", "b": b"bbb"}
    )
    assert rc == 0, f"stderr={err!r}"
    payload = json.loads(out)  # stdout is pure JSON (progress went to stderr)
    assert payload["dataset"] == "my-ds"
    assert payload["total_files"] == 2
    assert payload["downloaded"] == 1
    assert payload["skipped"] == 1
    assert payload["bytes"] == 3  # only "b" (3 bytes) was downloaded
    assert "%" not in out


# ---------------------------------------------------------------------------
# Errors: missing key, 404, malformed manifest
# ---------------------------------------------------------------------------

def test_missing_key_prints_login_hint(monkeypatch, tmp_path):
    rc, out, err, dl = _run(
        ["pull", "my-ds", "--dest", str(tmp_path)], monkeypatch, key=None,
        manifest=_manifest([("a", b"a")]),
    )
    assert rc == 1
    assert "newt login" in err
    assert dl == [], "no download should be attempted without a key"


def test_404_names_the_dataset(monkeypatch, tmp_path):
    from urllib.error import HTTPError
    err_404 = HTTPError("http://x", 404, "Not Found", {}, None)
    rc, out, err, dl = _run(
        ["pull", "ghost", "--dest", str(tmp_path)], monkeypatch, manifest_error=err_404,
    )
    assert rc == 1
    assert "ghost" in err and "404" in err


def test_malformed_manifest_fails_loud(monkeypatch, tmp_path):
    rc, out, err, dl = _run(
        ["pull", "my-ds", "--dest", str(tmp_path)], monkeypatch,
        manifest={"namespace": "ns0", "dataset": "my-ds"},  # no urls array
    )
    assert rc == 1
    assert "malformed manifest" in err


def test_pull_requires_a_dataset_name(monkeypatch, tmp_path):
    rc, out, err, dl = _run(["pull"], monkeypatch, key="nt_testkey")
    assert rc == 1
    assert "dataset name is required" in err


# ---------------------------------------------------------------------------
# push — helpers. The console listing is faked; the sink underneath is the REAL
# NTCloudSink with only its transport replaced, so these tests see the actual
# sign/PUT sequence the CLI drives.
# ---------------------------------------------------------------------------

_NAMESPACE = "0123456789abcdef"
_BUCKET = "nt-episodes"


class _FakeHTTPResp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResp":
        return self

    def __exit__(self, *_exc) -> None:
        return False


class _Transport:
    """The whole GCS side of a push, recorded.

    `sign_paths` is every path handed to /api/uploads/sign in request order; `put_paths` is
    every object that actually landed, in landing order. A refusal test asserts BOTH are
    empty — the claim isn't "the message says nothing was uploaded", it's that nothing was.
    """

    def __init__(self, dataset: str, *, put_error=None, fail_put_matching=None) -> None:
        self.dataset = dataset
        self.sign_paths: list[str] = []
        self.put_paths: list[str] = []
        self._put_error = put_error
        self._fail_put_matching = fail_put_matching

    def urlopen(self, req, timeout=None):
        if req.full_url.endswith("/api/uploads/sign"):
            path = json.loads(req.data)["path"]
            self.sign_paths.append(path)
            return _FakeHTTPResp(json.dumps({
                "url": f"https://storage.googleapis.com/{_BUCKET}/{_NAMESPACE}"
                       f"/{self.dataset}/{path}?sig=fake",
                "objectPath": f"gs://{_BUCKET}/{_NAMESPACE}/{self.dataset}/{path}",
                # Far-future so the client's own expiry guard never fires in these tests.
                "expiresAt": "2099-01-01T00:00:00.000Z",
            }).encode())
        rel = req.full_url.split(f"/{self.dataset}/", 1)[1].split("?", 1)[0]
        if self._put_error is not None and (
            self._fail_put_matching is None or self._fail_put_matching in rel
        ):
            raise self._put_error
        self.put_paths.append(rel)
        return _FakeHTTPResp(b"")


def _make_episode(source: Path, name: str, *, task: str | None = "pick up the cup",
                  committed: bool = True) -> Path:
    """One episode directory shaped like `newt record` leaves it: two data files plus the
    episode.json that says it was committed and which task it carries. `committed=False`
    omits episode.json — what an interrupted recording leaves behind."""
    episode_dir = source / name
    (episode_dir / "cameras" / "front").mkdir(parents=True)
    (episode_dir / "data.mcap").write_bytes(b"fake-mcap-bytes")
    (episode_dir / "cameras" / "front" / "color.mp4").write_bytes(b"fake-mp4-bytes")
    if committed:
        config: dict = {"duration": 1.0}
        if task is not None:
            config["task_name"] = task
        (episode_dir / "episode.json").write_text(
            json.dumps({"episode_config": config, "format_version": "0.0.3"})
        )
    return episode_dir


def _run_push(args, monkeypatch, *, key="nt_testkey", objects=(), listing=None,
              listing_error=None, put_error=None, fail_put_matching=None):
    """Run cmd_episodes(["push", ...]) with the console listing and the GCS transport faked.

    `objects` is what /api/uploads/list reports already living under the caller's namespace —
    push's write-once pre-check. `listing` overrides the whole response (malformed-shape
    tests); `listing_error` makes the pre-check raise. `put_error` is raised by the signed-URL
    PUT, for every file or only those whose remote path contains `fail_put_matching`.

    Returns (rc, stdout, stderr, transport) — the transport is how a test proves a refusal
    beat the network rather than merely claiming to.
    """
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    if key is None:
        monkeypatch.delenv("NT_API_KEY", raising=False)
        monkeypatch.setattr(ep, "read_api_key", lambda: None)
    else:
        monkeypatch.setenv("NT_API_KEY", key)

    dataset = args[args.index("--dataset") + 1] if "--dataset" in args else "unset"

    if listing_error is not None:
        def _list(*a, **k):
            raise listing_error
        monkeypatch.setattr(ep, "_list_dataset_objects", _list)
    else:
        response = listing if listing is not None else {
            "namespace": _NAMESPACE, "dataset": dataset, "objects": list(objects),
        }
        monkeypatch.setattr(ep, "_list_dataset_objects", lambda *a, **k: response)

    transport = _Transport(dataset, put_error=put_error, fail_put_matching=fail_put_matching)
    monkeypatch.setattr("newt.recording._cloud_sink.urlopen", transport.urlopen)
    # A URLError PUT retries with backoff — never sleep for real in a test.
    monkeypatch.setattr(time, "sleep", lambda *_a: None)

    rc = cmd_episodes(["push"] + args)
    return rc, out.getvalue(), err.getvalue(), transport


def _object(path: str) -> dict:
    """One entry of the /api/uploads/list response."""
    return {"path": path, "size": 12}


# ---------------------------------------------------------------------------
# push: ordering survives the CLI end to end
# ---------------------------------------------------------------------------

def test_push_keeps_episode_json_last_per_episode_and_manifest_last_of_all(monkeypatch, tmp_path):
    """The completeness sentinels only mean anything if they land LAST.

    episode.json last within an episode means a half-uploaded episode never reads as whole;
    manifest.json last of everything means a half-uploaded dataset never reads as complete.
    Both are order guarantees, and the CLI is what actually sequences them — so assert the
    order of the PUTs a whole `push` run produced. This fails the moment someone parallelises
    the upload for speed.
    """
    source = tmp_path / "episodes"
    source.mkdir()
    _make_episode(source, "episode_aaaaaaaa")
    _make_episode(source, "episode_bbbbbbbb")

    rc, out, err, transport = _run_push(
        [str(source), "--dataset", "my-ds"], monkeypatch,
    )
    assert rc == 0, f"stderr={err!r}"

    puts = transport.put_paths
    assert puts[-1] == "manifest.json", f"manifest must be last of everything, got {puts}"
    for name in ("episode_aaaaaaaa", "episode_bbbbbbbb"):
        owned = [p for p in puts if p.startswith(f"{name}/")]
        assert len(owned) == 3
        assert owned[-1] == f"{name}/episode.json", f"episode.json must be last of {name}"
    # And the episodes don't interleave: b's first byte goes up only after a is complete.
    assert puts.index("episode_bbbbbbbb/cameras/front/color.mp4") > puts.index(
        "episode_aaaaaaaa/episode.json"
    )
    assert "2/2" in out and "landed" in out


# ---------------------------------------------------------------------------
# push: every refusal arrives before the first signature
# ---------------------------------------------------------------------------

def test_existing_dataset_name_refuses_before_signing_anything(monkeypatch, tmp_path):
    """The store is create-only: one object under a name spends it forever (portal#98).

    So refusing AFTER the first sign+PUT is not a smaller version of this failure, it's a
    different and worse one — it costs the developer the name they were warned about. Assert
    the pre-check's refusal beat the network: zero signs, zero PUTs.
    """
    source = tmp_path / "episodes"
    source.mkdir()
    _make_episode(source, "episode_aaaaaaaa")

    rc, out, err, transport = _run_push(
        [str(source), "--dataset", "my-ds"], monkeypatch,
        objects=[_object("my-ds/manifest.json"), _object("my-ds/episode_zzzzzzzz/data.mcap")],
    )
    assert rc == 1
    assert transport.sign_paths == [], "refused a taken name only after signing — too late"
    assert transport.put_paths == [], "refused a taken name only after uploading — too late"
    assert "my-ds" in err and "write-once" in err


def test_complete_and_partial_collisions_are_not_the_same_problem(monkeypatch, tmp_path):
    """A finished dataset under this name and an abandoned half-push under it lead the
    developer somewhere different — one says the data is already there, the other says the
    name is spent with nothing readable behind it. Rule 12: two causes, never one string."""
    source = tmp_path / "episodes"
    source.mkdir()
    _make_episode(source, "episode_aaaaaaaa")

    _, _, complete_err, complete_t = _run_push(
        [str(source), "--dataset", "my-ds"], monkeypatch,
        objects=[_object("my-ds/manifest.json"), _object("my-ds/episode_zzzzzzzz/data.mcap")],
    )
    _, _, partial_err, partial_t = _run_push(
        [str(source), "--dataset", "my-ds"], monkeypatch,
        objects=[_object("my-ds/episode_zzzzzzzz/data.mcap")],
    )

    assert complete_err != partial_err
    assert "complete" in complete_err and "manifest.json present" in complete_err
    assert "never finished" in partial_err
    # Neither refusal touched the wire.
    assert complete_t.put_paths == [] and partial_t.put_paths == []


def test_a_directory_holding_two_tasks_refuses_before_uploading_either(monkeypatch, tmp_path):
    """`newt record` appends into one directory across runs, so a mixed-task directory is a
    normal thing to hit. The sink alone would only notice on the SECOND episode — after the
    first has landed and spent the name. Catching it in the pre-flight is the whole point, so
    assert nothing was signed AND that the message names both tasks and their episodes.
    """
    source = tmp_path / "episodes"
    source.mkdir()
    _make_episode(source, "episode_aaaaaaaa", task="pick up the cup")
    _make_episode(source, "episode_bbbbbbbb", task="pour the cup")

    rc, out, err, transport = _run_push(
        [str(source), "--dataset", "my-ds"], monkeypatch,
    )
    assert rc == 1
    assert transport.sign_paths == [] and transport.put_paths == []
    assert "pick up the cup" in err and "pour the cup" in err
    assert "episode_aaaaaaaa" in err and "episode_bbbbbbbb" in err
    assert "Nothing was uploaded" in err


# ---------------------------------------------------------------------------
# push: a failure mid-upload never reports success (Rule 10)
# ---------------------------------------------------------------------------

def test_a_failed_put_reports_how_many_episodes_landed_and_fails(monkeypatch, tmp_path):
    """A partial push reporting success is the worst thing this verb could do: the developer
    walks away believing a dataset exists, and can never push into that name again to fix it.
    So the count of what landed has to reach stderr, and the exit code has to be non-zero.
    """
    source = tmp_path / "episodes"
    source.mkdir()
    _make_episode(source, "episode_aaaaaaaa")
    _make_episode(source, "episode_bbbbbbbb")

    rc, out, err, transport = _run_push(
        [str(source), "--dataset", "my-ds"], monkeypatch,
        put_error=URLError("connection reset"),
        fail_put_matching="episode_bbbbbbbb",
    )
    assert rc != 0, "a partial push must never exit 0"
    assert "1 of 2 episode(s) landed" in err
    assert "PARTIAL" in err
    # The first episode really did land, and no manifest claims the dataset is complete.
    assert transport.put_paths == [
        "episode_aaaaaaaa/cameras/front/color.mp4",
        "episode_aaaaaaaa/data.mcap",
        "episode_aaaaaaaa/episode.json",
    ]
    assert "manifest.json" not in transport.put_paths


def test_an_expired_url_mid_push_reads_differently_from_a_taken_name(monkeypatch, tmp_path):
    """Both cost the developer the dataset name, and the fixes diverge: one says get on a
    stronger link, the other says the name was already taken before you started. Rule 12 —
    if these shared a string the developer would retry the wrong thing."""
    source = tmp_path / "episodes"
    source.mkdir()
    _make_episode(source, "episode_aaaaaaaa")

    expired = HTTPError(
        "https://storage.googleapis.com/x", 400, "Bad Request", {},
        io.BytesIO(b"<?xml version='1.0'?><Error><Code>ExpiredToken</Code></Error>"),
    )
    rc, out, expired_err, transport = _run_push(
        [str(source), "--dataset", "my-ds"], monkeypatch, put_error=expired,
    )
    assert rc == 1
    assert "expired" in expired_err

    _, _, collision_err, _ = _run_push(
        [str(source), "--dataset", "my-ds"], monkeypatch,
        objects=[_object("my-ds/manifest.json")],
    )
    assert expired_err != collision_err


# ---------------------------------------------------------------------------
# push: six causes, six strings (Rule 12)
# ---------------------------------------------------------------------------

def test_no_two_push_refusals_share_one_string(monkeypatch, tmp_path):
    """The refusal IS the product here — every one of these ends with the developer holding
    episodes they can't push, and the only thing separating "get a key" from "your directory
    has two tasks in it" is the string. Two causes sharing one string is the Rule 12
    violation this collects the whole surface to catch, including the three different shapes
    of "there is nothing here to push", which are three different mistakes.
    """
    good = tmp_path / "episodes"
    good.mkdir()
    _make_episode(good, "episode_aaaaaaaa")

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    _make_episode(mixed, "episode_aaaaaaaa", task="pick up the cup")
    _make_episode(mixed, "episode_bbbbbbbb", task="pour the cup")

    empty = tmp_path / "empty"
    empty.mkdir()

    uncommitted = tmp_path / "uncommitted"
    uncommitted.mkdir()
    _make_episode(uncommitted, "episode_cccccccc", committed=False)

    messages = {}
    messages["no api key"] = _run_push(
        [str(good), "--dataset", "my-ds"], monkeypatch, key=None)[2]
    messages["name exists, complete"] = _run_push(
        [str(good), "--dataset", "my-ds"], monkeypatch,
        objects=[_object("my-ds/manifest.json")])[2]
    messages["name exists, partial"] = _run_push(
        [str(good), "--dataset", "my-ds"], monkeypatch,
        objects=[_object("my-ds/episode_zzzzzzzz/data.mcap")])[2]
    messages["more than one task"] = _run_push(
        [str(mixed), "--dataset", "my-ds"], monkeypatch)[2]
    messages["no such directory"] = _run_push(
        [str(tmp_path / "ghost"), "--dataset", "my-ds"], monkeypatch)[2]
    messages["no episode dirs"] = _run_push(
        [str(empty), "--dataset", "my-ds"], monkeypatch)[2]
    messages["episode never committed"] = _run_push(
        [str(uncommitted), "--dataset", "my-ds"], monkeypatch)[2]
    messages["a put failed mid-push"] = _run_push(
        [str(good), "--dataset", "my-ds"], monkeypatch,
        put_error=URLError("connection reset"))[2]

    for label, text in messages.items():
        assert text.strip(), f"{label} refused silently"

    labels = list(messages)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            assert messages[a] != messages[b], (
                f"{a!r} and {b!r} produce the identical string — a developer can't tell "
                "which one happened (Rule 12)"
            )

    # Distinct is necessary, not sufficient: each must also NAME its own cause.
    assert "newt login" in messages["no api key"]
    assert "no such directory" in messages["no such directory"]
    assert "no episode_<id> directories" in messages["no episode dirs"]
    assert "no episode.json" in messages["episode never committed"]
    assert "more than one task" in messages["more than one task"]
    assert "0 of 1 episode(s) landed" in messages["a put failed mid-push"]


# ---------------------------------------------------------------------------
# push: --json shape
# ---------------------------------------------------------------------------

def test_push_json_keeps_stdout_pure_and_progress_on_stderr(monkeypatch, tmp_path):
    """`newt episodes push --json` is meant to be piped — `$(...)`, jq, an agent reading the
    result. One stray progress line on stdout breaks every one of those callers, so parse the
    WHOLE of stdout as JSON. The preflight report still has to reach a human: it goes to
    stderr, where a pipe leaves it alone.
    """
    source = tmp_path / "episodes"
    source.mkdir()
    _make_episode(source, "episode_aaaaaaaa")
    _make_episode(source, "episode_bbbbbbbb")

    rc, out, err, transport = _run_push(
        [str(source), "--dataset", "my-ds", "--json"], monkeypatch,
    )
    assert rc == 0, f"stderr={err!r}"

    payload = json.loads(out)  # the WHOLE of stdout, not a line of it
    assert payload["dataset"] == "my-ds"
    assert payload["namespace"] == _NAMESPACE
    assert payload["task"] == "pick up the cup"
    assert payload["episodes"] == 2
    assert payload["files"] == 6
    assert payload["bytes"] > 0

    # The human-facing preflight — including the write-once warning — went to stderr.
    assert "Pushing 2 episode(s)" in err
    assert "write-once" in err
    assert "%" not in out, "progress must never print a percentage it can't source"


# ---------------------------------------------------------------------------
# The verb coexists with `validate` and the help guard
# ---------------------------------------------------------------------------

def test_help_lists_every_subcommand_the_dispatcher_accepts(monkeypatch):
    """`newt episodes --help` is the only place a developer discovers these verbs from inside
    the tool. A subcommand live in the dispatcher and absent from the usage text is a verb
    nobody can find — so drive the assertion off the dispatcher's own accepted set rather than
    a hand-written list that goes stale the next time one is added."""
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    rc = cmd_episodes(["--help"])
    assert rc == 0

    # Read the Subcommands BLOCK, not the whole page: every one of these verbs also appears
    # in the options list ("(push) The dataset name..."), so a substring search over the page
    # passes even with the verb's own line deleted — a test that can't fail.
    body = out.getvalue().split("Subcommands:", 1)[1].split("Options:", 1)[0]
    listed = {line.split()[0] for line in body.splitlines() if line.strip()}
    for sub in ("validate", "push", "pull"):
        assert sub in listed, f"{sub!r} is dispatchable but not listed under Subcommands"
        assert cmd_episodes([sub, "--help"]) == 0, f"{sub!r} is documented but not dispatchable"
    assert "--dataset" in out.getvalue(), "push's required flag has to be discoverable"


def test_unknown_subcommand_still_rejected(monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    rc = cmd_episodes(["frobnicate"])
    assert rc == 1
    assert "unknown subcommand" in err.getvalue()
