"""Offline unit tests for `newt episodes pull` — download a staged dataset back down.

No network: the two HTTP round-trips (`_fetch_manifest`, `_download_url`) are monkeypatched
and files are written into `tmp_path`. Each test encodes why the behavior matters:
  - the manifest is fetched authed, then every file is downloaded into --dest, recreating the
    dataset's relative layout;
  - progress is reported by FILES COMPLETED, never an invented percentage (Rule 10);
  - a rerun is resumable — a file already present at the manifest's size is SKIPPED, not
    re-downloaded;
  - `--json` emits a machine-readable result (files, bytes) on a clean stdout;
  - a missing key prints the `newt login` hint; a 404 names the dataset.
"""
from __future__ import annotations

import io
import json
import sys

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
# The verb coexists with `validate` and the help guard
# ---------------------------------------------------------------------------

def test_help_lists_both_subcommands(monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    rc = cmd_episodes(["--help"])
    assert rc == 0
    assert "validate" in out.getvalue() and "pull" in out.getvalue()


def test_unknown_subcommand_still_rejected(monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    rc = cmd_episodes(["frobnicate"])
    assert rc == 1
    assert "unknown subcommand" in err.getvalue()
