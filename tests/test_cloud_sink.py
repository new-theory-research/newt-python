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
