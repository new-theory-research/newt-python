"""Unit tests for ``NTCloudSink`` — mocked signed-URL API + mocked GCS PUT.

No network, no real API key, no live bucket. `urlopen` is monkeypatched at
`newt.recording._cloud_sink.urlopen` for both the sign call (POST
/api/uploads/sign) and the upload call (PUT to the signed URL), mirroring the
pattern in `tests/test_cli_login.py`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

_FAKE_KEY = "nt_" + "a" * 40
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


def _sign_response(dataset: str, path: str) -> bytes:
    return json.dumps(
        {
            "url": f"https://storage.googleapis.com/{_BUCKET}/{_NAMESPACE}/{dataset}/{path}?sig=fake",
            "objectPath": f"gs://{_BUCKET}/{_NAMESPACE}/{dataset}/{path}",
            "expiresAt": "2026-07-04T23:00:00.000Z",
        }
    ).encode()


def _install_fake_urlopen(monkeypatch, dataset: str, *, upload_error: Exception | None = None):
    """Every /api/uploads/sign call succeeds; every upload PUT succeeds unless
    `upload_error` is given, in which case every PUT raises it. Returns the list
    of (method, url) calls in order, plus `signed_paths` — the `path` field
    sent to /api/uploads/sign, in the order it was requested."""
    calls: list[tuple[str, str]] = []
    signed_paths: list[str] = []

    def fake_urlopen(req, timeout=None):
        calls.append((req.get_method(), req.full_url))
        if req.full_url.endswith("/api/uploads/sign"):
            body = json.loads(req.data)
            signed_paths.append(body["path"])
            return _FakeHTTPResp(_sign_response(dataset, body["path"]))
        # This is the "upload" PUT to the signed URL.
        if upload_error is not None:
            raise upload_error
        return _FakeHTTPResp(b"")

    monkeypatch.setattr("newt.recording._cloud_sink.urlopen", fake_urlopen)
    return calls, signed_paths


def _make_episode_dir(tmp_path: Path, name: str, *, task: str = "pick up the cup") -> Path:
    ep = tmp_path / name
    (ep / "cameras" / "front").mkdir(parents=True)
    (ep / "data.mcap").write_bytes(b"fake-mcap-bytes")
    (ep / "cameras" / "front" / "color.mp4").write_bytes(b"fake-mp4-bytes")
    # episode.json written LAST on real disk too, but timing doesn't matter here
    # — deliver() reads whatever is on disk when it's called.
    (ep / "episode.json").write_text(
        json.dumps(
            {
                "episode_config": {"task_name": task, "tags": ["alpha"], "duration": 1.0},
                "format_version": "0.0.3",
            }
        )
    )
    return ep


def test_satisfies_sink_protocol(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink, Sink

    sink = NTCloudSink("grasp-cup", api_key=_FAKE_KEY)
    assert isinstance(sink, Sink)


def test_deliver_uploads_under_correct_namespace_dataset_prefix(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "grasp-cup"
    calls, _signed_paths = _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    episode_dir = _make_episode_dir(tmp_path, "episode_deadbeef")
    sink.deliver(episode_dir)

    sign_calls = [c for c in calls if c[1].endswith("/api/uploads/sign")]
    upload_calls = [c for c in calls if not c[1].endswith("/api/uploads/sign")]

    # One sign+upload round trip per file in the episode dir.
    assert len(sign_calls) == 3
    assert len(upload_calls) == 3
    # Every uploaded object lands under gs://<bucket>/<namespace>/<dataset>/episode_<id>/...
    for _, url in upload_calls:
        assert f"/{_NAMESPACE}/{dataset}/episode_deadbeef/" in url


def test_deliver_writes_episode_json_last(monkeypatch, tmp_path):
    """Mirrors the local writer's own rename-last commit: episode.json is the
    last file uploaded for an episode, so a partial upload never looks like a
    complete episode in the cloud either."""
    from newt.recording import NTCloudSink

    dataset = "grasp-cup"
    _calls, signed_paths = _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    episode_dir = _make_episode_dir(tmp_path, "episode_deadbeef")
    sink.deliver(episode_dir)

    assert signed_paths[-1] == "episode_deadbeef/episode.json"
    assert len(signed_paths) == 3


def test_deliver_raises_and_preserves_episode_on_signing_failure(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "grasp-cup"

    def fake_urlopen(req, timeout=None):
        raise HTTPError(req.full_url, 503, "signing_unavailable", None, None)

    monkeypatch.setattr("newt.recording._cloud_sink.urlopen", fake_urlopen)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    episode_dir = _make_episode_dir(tmp_path, "episode_deadbeef")
    with pytest.raises(RuntimeError, match="signed upload URL"):
        sink.deliver(episode_dir)

    # Loud failure never touches the local episode.
    assert episode_dir.exists()
    assert (episode_dir / "episode.json").exists()
    assert (episode_dir / "data.mcap").exists()


def test_deliver_raises_and_preserves_episode_on_upload_failure(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "grasp-cup"
    _install_fake_urlopen(
        monkeypatch,
        dataset,
        upload_error=URLError("connection reset"),
    )
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    episode_dir = _make_episode_dir(tmp_path, "episode_deadbeef")
    with pytest.raises(RuntimeError, match="upload failed"):
        sink.deliver(episode_dir)

    assert episode_dir.exists()
    assert (episode_dir / "episode.json").exists()


def test_deliver_rejects_mixed_tasks_in_one_dataset(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "grasp-cup"
    _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    sink.deliver(_make_episode_dir(tmp_path, "episode_aaaaaaaa", task="pick up the cup"))
    with pytest.raises(RuntimeError, match="mixes tasks"):
        sink.deliver(_make_episode_dir(tmp_path, "episode_bbbbbbbb", task="pour the cup"))


def test_no_api_key_raises_with_actionable_message(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    monkeypatch.delenv("NT_API_KEY", raising=False)
    monkeypatch.setattr("newt.recording._cloud_sink.read_api_key", lambda: None)

    with pytest.raises(RuntimeError, match="newt login"):
        NTCloudSink("grasp-cup")


def test_finalize_writes_manifest_after_all_episodes(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "grasp-cup"
    calls, signed_paths = _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    sink.deliver(_make_episode_dir(tmp_path, "episode_aaaaaaaa"))
    sink.deliver(_make_episode_dir(tmp_path, "episode_bbbbbbbb"))

    manifest_calls_before = [c for c in calls if c[1].endswith("/manifest.json?sig=fake")]
    assert manifest_calls_before == []  # not written before finalize()

    sink.finalize()

    assert signed_paths[-1] == "manifest.json"
    manifest_uploads = [c for c in calls if not c[1].endswith("/api/uploads/sign")]
    manifest_put = manifest_uploads[-1]
    assert manifest_put[0] == "PUT"
    assert f"/{_NAMESPACE}/{dataset}/manifest.json" in manifest_put[1]


def test_finalize_manifest_body_matches_v0_minimal_fields(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "grasp-cup"
    put_bodies: dict[str, bytes] = {}

    def fake_urlopen(req, timeout=None):
        if req.full_url.endswith("/api/uploads/sign"):
            body = json.loads(req.data)
            return _FakeHTTPResp(_sign_response(dataset, body["path"]))
        put_bodies[req.full_url] = req.data
        return _FakeHTTPResp(b"")

    monkeypatch.setattr("newt.recording._cloud_sink.urlopen", fake_urlopen)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)
    sink.deliver(_make_episode_dir(tmp_path, "episode_aaaaaaaa", task="pick up the cup"))
    sink.finalize()

    manifest_body = next(
        body for url, body in put_bodies.items() if url.endswith("/manifest.json?sig=fake")
    )
    manifest = json.loads(manifest_body)
    assert manifest["format_version"] == "0.0.3"
    assert manifest["task"] == "pick up the cup"
    assert manifest["episode_count"] == 1
    assert manifest["attribution"] == _NAMESPACE
    assert "created_at" in manifest


def test_finalize_before_any_deliver_raises_without_writing(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "grasp-cup"
    calls, _signed_paths = _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    with pytest.raises(RuntimeError, match="no episodes were delivered"):
        sink.finalize()

    assert calls == []  # never even attempted to sign a manifest upload


def test_interrupted_dataset_never_gets_a_manifest(monkeypatch, tmp_path):
    """Simulates a mid-dataset crash: the second episode's upload fails, the
    recording session propagates the error and never calls finalize(). No
    manifest.json request is ever made."""
    from newt.recording import NTCloudSink

    dataset = "grasp-cup"
    calls, _signed_paths = _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    sink.deliver(_make_episode_dir(tmp_path, "episode_aaaaaaaa"))

    def failing_urlopen(req, timeout=None):
        raise URLError("connection reset")

    monkeypatch.setattr("newt.recording._cloud_sink.urlopen", failing_urlopen)
    with pytest.raises(RuntimeError, match="connection reset"):
        sink.deliver(_make_episode_dir(tmp_path, "episode_bbbbbbbb"))

    # A real caller would stop here and never call finalize(); confirm no
    # manifest request was ever issued up to the point of the crash.
    assert not any(c[1].endswith("/manifest.json?sig=fake") for c in calls)


def test_finalize_manifest_uses_same_content_type_as_episode_uploads(monkeypatch, tmp_path):
    """The signed URL binds Content-Type into the signature (see module
    docstring) — the manifest PUT must reuse the same octet-stream type as
    episode file uploads, not switch to application/json."""
    from newt.recording import NTCloudSink

    dataset = "grasp-cup"
    put_headers: dict[str, str] = {}

    def fake_urlopen(req, timeout=None):
        if req.full_url.endswith("/api/uploads/sign"):
            body = json.loads(req.data)
            return _FakeHTTPResp(_sign_response(dataset, body["path"]))
        put_headers[req.full_url] = req.get_header("Content-type")
        return _FakeHTTPResp(b"")

    monkeypatch.setattr("newt.recording._cloud_sink.urlopen", fake_urlopen)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)
    sink.deliver(_make_episode_dir(tmp_path, "episode_aaaaaaaa"))
    sink.finalize()

    manifest_url = next(url for url in put_headers if url.endswith("/manifest.json?sig=fake"))
    assert put_headers[manifest_url] == "application/octet-stream"


def test_end_to_end_via_session_delivers_real_episode(monkeypatch, tmp_path):
    """Drives a real Session + SimulatedSource episode through NTCloudSink,
    same as the LocalSink protocol-seam test, to prove NTCloudSink is a real
    drop-in Sink (not just structurally shaped like one)."""
    pytest.importorskip("mcap")
    pytest.importorskip("google.protobuf")
    from newt.recording import NTCloudSink, Session, SimulatedSource, SINGLE_ARM_DESCRIPTOR

    dataset = "grasp-cup"
    calls, _signed_paths = _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    session = Session(
        SimulatedSource(SINGLE_ARM_DESCRIPTOR),
        task="pick up the cup",
        output_dir=tmp_path,
        sink=sink,
    )
    session.start_episode()
    time.sleep(0.15)
    path = session.end_episode(keep=True)
    session.close()

    assert path is not None
    upload_calls = [c for c in calls if not c[1].endswith("/api/uploads/sign")]
    # data.mcap + episode.json, no cameras configured in this test.
    assert len(upload_calls) == 2
    for _, url in upload_calls:
        assert f"/{_NAMESPACE}/{dataset}/{path.name}/" in url


# --------------------------------------------------------------------------- #
# capture-005-cont — upload_directory: the Rerun-exported-directory hand-off.
# Takes an arbitrary already-exported directory (LeRobot-v3, as Rerun
# produces it) as given — no episode.json parsing, no format assumptions —
# and reuses the same sign/PUT plumbing deliver()/finalize() use.
# --------------------------------------------------------------------------- #

def _make_rerun_export_dir(tmp_path: Path, name: str = "minigolf_export") -> Path:
    """A LeRobot-v3-shaped export tree — not an NT episode.json shape at all,
    proving upload_directory takes the directory as given rather than
    expecting NT's own episode format."""
    export = tmp_path / name
    (export / "meta").mkdir(parents=True)
    (export / "data" / "chunk-000").mkdir(parents=True)
    (export / "videos" / "chunk-000").mkdir(parents=True)
    (export / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    (export / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "putt the ball"}) + "\n")
    (export / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"fake-parquet-bytes")
    (export / "videos" / "chunk-000" / "episode_000000.mp4").write_bytes(b"fake-mp4-bytes")
    return export


def test_upload_directory_rerun_export_uploads_every_file_under_namespace(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "minigolf"
    calls, signed_paths = _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    export_dir = _make_rerun_export_dir(tmp_path)
    sink.upload_directory(export_dir)

    upload_calls = [c for c in calls if not c[1].endswith("/api/uploads/sign")]
    # 4 export files + manifest.json written last.
    assert len(upload_calls) == 5
    for _, url in upload_calls:
        assert f"/{_NAMESPACE}/{dataset}/" in url
    # Relative paths preserved under the dataset prefix, no NT episode.json assumed.
    assert "meta/info.json" in signed_paths
    assert "data/chunk-000/episode_000000.parquet" in signed_paths
    assert signed_paths[-1] == "manifest.json"


def test_upload_directory_rerun_export_writes_manifest_last(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "minigolf"
    calls, signed_paths = _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    export_dir = _make_rerun_export_dir(tmp_path)
    sink.upload_directory(export_dir)

    manifest_uploads = [c for c in calls if not c[1].endswith("/api/uploads/sign")]
    manifest_put = manifest_uploads[-1]
    assert manifest_put[0] == "PUT"
    assert f"/{_NAMESPACE}/{dataset}/manifest.json" in manifest_put[1]
    assert sink.namespace == _NAMESPACE


def test_upload_directory_rerun_export_manifest_does_not_invent_episode_fields(monkeypatch, tmp_path):
    """The export isn't NT's episode.json shape, so the manifest must not
    claim NT-episode fields (format_version="0.0.3", episode_count) it has no
    basis for (Rule 10) — it records what's actually known about this upload."""
    from newt.recording import NTCloudSink

    dataset = "minigolf"
    put_bodies: dict[str, bytes] = {}

    def fake_urlopen(req, timeout=None):
        if req.full_url.endswith("/api/uploads/sign"):
            body = json.loads(req.data)
            return _FakeHTTPResp(_sign_response(dataset, body["path"]))
        put_bodies[req.full_url] = req.data
        return _FakeHTTPResp(b"")

    monkeypatch.setattr("newt.recording._cloud_sink.urlopen", fake_urlopen)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)
    export_dir = _make_rerun_export_dir(tmp_path)
    sink.upload_directory(export_dir)

    manifest_body = next(
        body for url, body in put_bodies.items() if url.endswith("/manifest.json?sig=fake")
    )
    manifest = json.loads(manifest_body)
    assert manifest["source_format"] == "lerobot-v3"
    assert manifest["file_count"] == 4
    assert manifest["attribution"] == _NAMESPACE
    assert "created_at" in manifest
    assert "format_version" not in manifest
    assert "episode_count" not in manifest


def test_upload_directory_rerun_export_raises_and_preserves_on_upload_failure(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "minigolf"
    _install_fake_urlopen(monkeypatch, dataset, upload_error=URLError("connection reset"))
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    export_dir = _make_rerun_export_dir(tmp_path)
    with pytest.raises(RuntimeError, match="upload failed"):
        sink.upload_directory(export_dir)

    # Loud failure never touches the local export directory.
    assert export_dir.exists()
    assert (export_dir / "meta" / "info.json").exists()


def test_upload_directory_rerun_export_rejects_empty_directory(tmp_path):
    from newt.recording import NTCloudSink

    sink = NTCloudSink("minigolf", api_key=_FAKE_KEY)
    empty_dir = tmp_path / "empty_export"
    empty_dir.mkdir()

    with pytest.raises(RuntimeError, match="no files found"):
        sink.upload_directory(empty_dir)


def test_upload_directory_rerun_export_rejects_missing_directory(tmp_path):
    from newt.recording import NTCloudSink

    sink = NTCloudSink("minigolf", api_key=_FAKE_KEY)
    with pytest.raises(RuntimeError, match="not a directory"):
        sink.upload_directory(tmp_path / "does_not_exist")


# --------------------------------------------------------------------------- #
# capture-005-cont task 2 — placeholder-task refusal, coordinated with
# training/intake's own check (portal repo, ft-002). meta/tasks.parquet is the
# real LeRobot v3.0 task-storage format (meta/tasks.jsonl is legacy) — see
# training/intake/intake.py::load_task_records.
# --------------------------------------------------------------------------- #

def _write_tasks_parquet(export_dir: Path, task_texts: list[str]) -> None:
    """Write meta/tasks.parquet in the real LeRobot v3.0 shape — a
    task-text-indexed table with an int task_index column, matching
    training/intake/intake.py's write_task_records (portal repo)."""
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    table = pa.table(
        {"task": task_texts, "task_index": list(range(len(task_texts)))}
    )
    pq.write_table(table, export_dir / "meta" / "tasks.parquet")


def test_upload_directory_rerun_export_refuses_placeholder_only_tasks(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "minigolf"
    calls, _signed_paths = _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    export_dir = _make_rerun_export_dir(tmp_path)
    _write_tasks_parquet(export_dir, ["task"])

    with pytest.raises(RuntimeError, match="placeholder"):
        sink.upload_directory(export_dir)

    # Refused before any file was uploaded — no partial upload.
    assert calls == []


def test_upload_directory_rerun_export_allows_real_task_labels(monkeypatch, tmp_path):
    from newt.recording import NTCloudSink

    dataset = "minigolf"
    _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    export_dir = _make_rerun_export_dir(tmp_path)
    _write_tasks_parquet(export_dir, ["putt the ball"])

    sink.upload_directory(export_dir)  # does not raise

    assert sink.namespace == _NAMESPACE


def test_upload_directory_rerun_export_only_refuses_if_every_task_is_placeholder(monkeypatch, tmp_path):
    """Mirrors training/intake/validators.py::validate_task_requirement's
    `all(...)` semantics — a dataset with a mix of placeholder and real task
    strings is not refused, only an all-placeholder dataset is."""
    from newt.recording import NTCloudSink

    dataset = "minigolf"
    _install_fake_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    export_dir = _make_rerun_export_dir(tmp_path)
    _write_tasks_parquet(export_dir, ["task", "putt the ball"])

    sink.upload_directory(export_dir)  # does not raise
