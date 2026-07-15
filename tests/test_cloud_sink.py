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
# validate_lerobot_export + upload_directory (the batch-sign folder
# upload the `newt finetune --dataset ./folder` verb drives). Reuses the SAME
# NTCloudSink class and PUT plumbing as deliver()/finalize(); only the sign call
# is batched (the manifest-of-signed-URLs shape) instead of per-file.
# --------------------------------------------------------------------------- #


def _make_lerobot_export(tmp_path: Path, name: str = "my_export", *, info=None) -> Path:
    """A minimal LeRobot-v3-shaped export: meta/info.json with readable action +
    observation.state feature shapes, plus a data file and a video file."""
    export = tmp_path / name
    (export / "meta").mkdir(parents=True)
    (export / "data" / "chunk-000").mkdir(parents=True)
    (export / "videos" / "chunk-000").mkdir(parents=True)
    if info is None:
        info = {
            "codebase_version": "v3.0",
            "features": {
                "action": {"dtype": "float32", "shape": [6], "names": ["j0"]},
                "observation.state": {"dtype": "float32", "shape": [6]},
            },
        }
    (export / "meta" / "info.json").write_text(json.dumps(info))
    (export / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"fake-parquet")
    (export / "videos" / "chunk-000" / "episode_000000.mp4").write_bytes(b"fake-mp4-bytes")
    return export


def _install_fake_batch_urlopen(monkeypatch, dataset: str, *, upload_error=None):
    """Fake the BATCH sign shape: POST {dataset, paths:[...]} -> {namespace, dataset,
    count, urls:[{path,url,objectPath,expiresAt}]}, plus per-file PUTs. The single
    sign shape (used for manifest.json) is still honored. Returns (calls, batched_paths)
    where batched_paths is the list the batch sign was asked to sign."""
    calls: list[tuple[str, str]] = []
    batched_paths: list[list[str]] = []

    def fake_urlopen(req, timeout=None):
        calls.append((req.get_method(), req.full_url))
        if req.full_url.endswith("/api/uploads/sign"):
            body = json.loads(req.data)
            if "paths" in body:  # batch shape
                batched_paths.append(body["paths"])
                urls = [
                    {
                        "path": p,
                        "url": f"https://storage.googleapis.com/{_BUCKET}/{_NAMESPACE}/{dataset}/{p}?sig=fake",
                        "objectPath": f"gs://{_BUCKET}/{_NAMESPACE}/{dataset}/{p}",
                        "expiresAt": "2026-07-16T23:00:00.000Z",
                    }
                    for p in body["paths"]
                ]
                return _FakeHTTPResp(
                    json.dumps(
                        {"namespace": _NAMESPACE, "dataset": dataset, "count": len(urls), "urls": urls}
                    ).encode()
                )
            # single shape (manifest.json)
            return _FakeHTTPResp(_sign_response(dataset, body["path"]))
        # a PUT to a signed URL
        if upload_error is not None:
            raise upload_error
        return _FakeHTTPResp(b"")

    monkeypatch.setattr("newt.recording._cloud_sink.urlopen", fake_urlopen)
    return calls, batched_paths


def test_validate_lerobot_export_accepts_a_real_export(tmp_path):
    from newt.recording._cloud_sink import validate_lerobot_export

    export = _make_lerobot_export(tmp_path)
    validate_lerobot_export(export)  # must not raise


def test_validate_lerobot_export_missing_info_json_names_the_fix(tmp_path):
    """A folder with no meta/info.json fails with a message a developer can act on —
    the whole point of validating before an 84 MB upload."""
    from newt.recording._cloud_sink import validate_lerobot_export

    bare = tmp_path / "not_an_export"
    bare.mkdir()
    with pytest.raises(RuntimeError, match="meta/info.json"):
        validate_lerobot_export(bare)


def test_validate_lerobot_export_missing_action_shape_is_caught(tmp_path):
    from newt.recording._cloud_sink import validate_lerobot_export

    export = _make_lerobot_export(
        tmp_path, info={"features": {"observation.state": {"shape": [6]}}}
    )
    with pytest.raises(RuntimeError, match="action"):
        validate_lerobot_export(export)


def test_validate_lerobot_export_corrupt_json_is_caught(tmp_path):
    from newt.recording._cloud_sink import validate_lerobot_export

    export = tmp_path / "corrupt"
    (export / "meta").mkdir(parents=True)
    (export / "meta" / "info.json").write_text("{not json")
    with pytest.raises(RuntimeError, match="JSON"):
        validate_lerobot_export(export)


def test_upload_directory_batch_signs_all_files_in_one_round_trip(monkeypatch, tmp_path):
    """upload_directory signs the whole file list in ONE batch call (the
    manifest shape), not one sign per file — then PUTs each under the namespace."""
    from newt.recording._cloud_sink import NTCloudSink

    dataset = "my_export"
    calls, batched_paths = _install_fake_batch_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    export = _make_lerobot_export(tmp_path, dataset)
    namespace = sink.upload_directory(export)

    assert namespace == _NAMESPACE
    # Exactly ONE batch sign for the real files (manifest.json is signed singly after).
    batch_sign_calls = [
        c for c in calls if c[1].endswith("/api/uploads/sign") and c[0] == "POST"
    ]
    # 1 batch sign (files) + 1 single sign (manifest) = 2 sign POSTs.
    assert len(batch_sign_calls) == 2
    assert len(batched_paths) == 1, "the export's files must be signed in one batch"
    assert set(batched_paths[0]) == {
        "meta/info.json",
        "data/chunk-000/episode_000000.parquet",
        "videos/chunk-000/episode_000000.mp4",
    }
    # 3 files + manifest.json = 4 PUTs, all under the namespace/dataset prefix.
    upload_calls = [c for c in calls if not c[1].endswith("/api/uploads/sign")]
    assert len(upload_calls) == 4
    for _, url in upload_calls:
        assert f"/{_NAMESPACE}/{dataset}/" in url


def test_upload_directory_validates_before_any_sign_or_put(monkeypatch, tmp_path):
    """A malformed export raises BEFORE any network call — no byte leaves the machine.
    This is the pre-upload gate: fail on the laptop, not after the transfer."""
    from newt.recording._cloud_sink import NTCloudSink

    calls, _ = _install_fake_batch_urlopen(monkeypatch, "bad")
    sink = NTCloudSink("bad", api_key=_FAKE_KEY)

    bare = tmp_path / "not_an_export"
    bare.mkdir()
    (bare / "data.bin").write_bytes(b"stuff")  # non-empty, but no meta/info.json
    with pytest.raises(RuntimeError, match="meta/info.json"):
        sink.upload_directory(bare)

    assert calls == [], "validation must fail before any sign or upload call"


def test_upload_directory_writes_manifest_last(monkeypatch, tmp_path):
    from newt.recording._cloud_sink import NTCloudSink

    dataset = "my_export"
    calls, _ = _install_fake_batch_urlopen(monkeypatch, dataset)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    export = _make_lerobot_export(tmp_path, dataset)
    sink.upload_directory(export)

    upload_calls = [c for c in calls if not c[1].endswith("/api/uploads/sign")]
    last_method, last_url = upload_calls[-1]
    assert last_method == "PUT"
    assert f"/{_NAMESPACE}/{dataset}/manifest.json" in last_url


def test_upload_directory_raises_and_preserves_on_put_failure(monkeypatch, tmp_path):
    from newt.recording._cloud_sink import NTCloudSink

    dataset = "my_export"
    _install_fake_batch_urlopen(monkeypatch, dataset, upload_error=URLError("reset"))
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)

    export = _make_lerobot_export(tmp_path, dataset)
    with pytest.raises(RuntimeError, match="upload failed"):
        sink.upload_directory(export)
    # A loud failure never touches the local export.
    assert (export / "meta" / "info.json").exists()


def test_upload_directory_rejects_url_count_mismatch(monkeypatch, tmp_path):
    """If the server signs fewer URLs than files requested, refuse — never upload a
    partial set and call it done (Rule 10)."""
    from newt.recording._cloud_sink import NTCloudSink

    dataset = "my_export"

    def short_urlopen(req, timeout=None):
        if req.full_url.endswith("/api/uploads/sign"):
            body = json.loads(req.data)
            # Return only the FIRST path's URL — a short manifest.
            p = body["paths"][0]
            return _FakeHTTPResp(
                json.dumps(
                    {
                        "namespace": _NAMESPACE,
                        "dataset": dataset,
                        "count": 1,
                        "urls": [
                            {
                                "path": p,
                                "url": f"https://x/{p}?sig=fake",
                                "objectPath": f"gs://{_BUCKET}/{_NAMESPACE}/{dataset}/{p}",
                                "expiresAt": "2026-07-16T23:00:00.000Z",
                            }
                        ],
                    }
                ).encode()
            )
        return _FakeHTTPResp(b"")

    monkeypatch.setattr("newt.recording._cloud_sink.urlopen", short_urlopen)
    sink = NTCloudSink(dataset, api_key=_FAKE_KEY)
    export = _make_lerobot_export(tmp_path, dataset)
    with pytest.raises(RuntimeError, match="one URL per file"):
        sink.upload_directory(export)
