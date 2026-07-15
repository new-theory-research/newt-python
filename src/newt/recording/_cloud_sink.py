"""``NTCloudSink`` — lands a committed episode in the developer's NT cloud
namespace via NT's API, never touching the ``nt-episodes-writer`` GCS credential.

The write path (per ``wiki/programs/cloud/resource-manifest.md`` and
``wiki/operating-docs/gcp-guardrails.md``):

1. The developer's ``nt_<40hex>`` API key authenticates a request to NT's API.
2. The API mints a short-lived, namespace-scoped signed GCS upload URL
   (``POST /api/uploads/sign`` — see ``apps/console`` in the ``portal`` repo).
3. This sink PUTs the episode's bytes straight to that URL.

The SA credential lives server-side only; it is never requested, held, or
derivable from anything in this module — the signed URL is the only thing that
ever crosses the wire toward GCS.

Featherweight on purpose, same as ``_sink.py``: stdlib only (``urllib``, no
``requests``, no ``google-cloud-storage`` — a signed URL is just an HTTP PUT).
Safe to import (and construct) without the ``recording`` extra installed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from newt._credentials import read_api_key

# Matches newt._cli.login._DEFAULT_CONSOLE — kept as a separate literal rather
# than importing the CLI package (recording must not depend on the CLI layer).
_DEFAULT_CONSOLE_URL = "https://newtheory-console.vercel.app"

# GCS v4 signed URLs bind the content-type into the signature itself
# (apps/console/lib/episodes-storage.ts signs with "application/octet-stream").
# A PUT with any other Content-Type fails the signature check server-side, so
# this is not configurable per upload.
_UPLOAD_CONTENT_TYPE = "application/octet-stream"


def _rfc3339_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _namespace_from_object_path(object_path: str, dataset: str) -> str:
    """Extract ``<namespace>`` from the API's ``gs://<bucket>/<namespace>/<dataset>/...``
    response, rather than re-deriving the key-hash client-side — the API is the
    only place that mints the namespace (data-flywheel.md §9); duplicating that
    derivation here would drift the moment either side changes it."""
    prefix = "gs://"
    if not object_path.startswith(prefix):
        raise RuntimeError(
            f"NTCloudSink: unexpected objectPath from signed-URL API (no {prefix!r} "
            f"prefix): {object_path!r}"
        )
    segments = object_path[len(prefix):].split("/")
    if len(segments) < 3 or segments[2] != dataset:
        raise RuntimeError(
            f"NTCloudSink: unexpected objectPath shape for dataset {dataset!r}: "
            f"{object_path!r}"
        )
    return segments[1]


def validate_lerobot_export(export_dir: Path) -> None:
    """Presence/shape sanity check that ``export_dir`` is a LeRobot dataset export,
    run BEFORE any byte is uploaded so a malformed folder fails on the developer's
    machine instead of after the whole transfer and a server-side intake rejection.

    This is deliberately NOT the authoritative dimension/contract check — the intake
    gate owns that server-side, and duplicating it here would fork one contract into
    two copies that drift. This only spares the developer a pointless upload by
    confirming the cheap, local things:

    - ``meta/info.json`` exists,
    - it parses as JSON,
    - it records readable ``action`` and ``observation.state`` feature shapes.

    Raises ``RuntimeError`` naming the fixable problem; returns ``None`` on success.
    """
    info_path = export_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(
            f"no meta/info.json in {export_dir} — is this a LeRobot export? "
            "The exported dataset folder should contain meta/info.json."
        )
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"meta/info.json in {export_dir} could not be read as JSON ({exc}) — "
            "the export looks corrupt. Re-export the dataset."
        ) from exc

    features = info.get("features") if isinstance(info, dict) else None
    if not isinstance(features, dict):
        raise RuntimeError(
            f"meta/info.json in {export_dir} has no `features` map — this doesn't "
            "look like a LeRobot export. Re-export the dataset."
        )

    for key in ("action", "observation.state"):
        feature = features.get(key)
        shape = feature.get("shape") if isinstance(feature, dict) else None
        if not (
            isinstance(shape, (list, tuple))
            and len(shape) >= 1
            and all(isinstance(dim, int) for dim in shape)
        ):
            raise RuntimeError(
                f"meta/info.json in {export_dir} is missing a readable `{key}` shape "
                "under `features` — a LeRobot export records the action and "
                "observation.state dimensions there. Re-export the dataset."
            )


class NTCloudSink:
    """``Sink`` that uploads each committed episode to the developer's NT cloud
    namespace, one file at a time, via a fresh signed URL per file.

    Construct with the dataset name (the developer-chosen collection this
    session's episodes belong to); the API key defaults through the same
    resolution precedence as the rest of the SDK (arg → ``NT_API_KEY`` env var
    → ``~/.nt/credentials``, see ``newt._credentials``).

    ``deliver`` uploads every file in the episode directory except
    ``episode.json``, then uploads ``episode.json`` last — mirroring the local
    writer's own rename-last commit, so a partially-uploaded episode never
    looks complete in the cloud either. A failed upload raises (Rule 10); the
    episode is never touched or removed locally, so nothing is lost.

    ``finalize`` writes the dataset-level ``manifest.json`` — call it once,
    explicitly, after every episode for the dataset has been delivered. It is
    not part of the ``Sink`` protocol and ``Session`` never calls it, so a
    recording run that dies mid-dataset (and never reaches ``finalize``)
    leaves no manifest behind — the completeness sentinel data-flywheel.md §9
    requires.
    """

    def __init__(
        self,
        dataset: str,
        *,
        api_key: str | None = None,
        console_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("NT_API_KEY") or read_api_key()
        if not api_key:
            raise RuntimeError(
                "NTCloudSink: no API key found. Run `newt login` to authenticate, "
                "or set the NT_API_KEY environment variable."
            )
        self._dataset = dataset
        self._api_key = api_key
        self._console_url = (
            console_url or os.environ.get("NT_CONSOLE_URL") or _DEFAULT_CONSOLE_URL
        ).rstrip("/")
        self._timeout = timeout

        self._episode_count = 0
        self._namespace: str | None = None
        self._task: str | None = None
        self._format_version: str | None = None

    def deliver(self, episode_dir: Path) -> None:
        episode_dir = Path(episode_dir)
        if not episode_dir.exists():
            raise RuntimeError(
                f"NTCloudSink.deliver: episode directory does not exist: {episode_dir}"
            )

        episode_json_path = episode_dir / "episode.json"
        try:
            episode_meta = json.loads(episode_json_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"NTCloudSink.deliver: cannot read {episode_json_path}: {exc}"
            ) from exc

        task = episode_meta.get("episode_config", {}).get("task_name")
        if self._task is None:
            self._task = task
        elif self._task != task:
            raise RuntimeError(
                f"NTCloudSink.deliver: dataset {self._dataset!r} mixes tasks "
                f"({self._task!r} vs {task!r}) — one task per dataset "
                "(data-flywheel.md §9)."
            )
        self._format_version = episode_meta.get("format_version")

        other_files = sorted(
            p.relative_to(episode_dir).as_posix()
            for p in episode_dir.rglob("*")
            if p.is_file() and p != episode_json_path
        )
        for rel in other_files:
            self._upload(episode_dir / rel, f"{episode_dir.name}/{rel}")
        self._upload(episode_json_path, f"{episode_dir.name}/episode.json")

        self._episode_count += 1

    def finalize(self) -> None:
        """Write the dataset-level ``manifest.json`` — the completeness
        sentinel (data-flywheel.md §9): its existence means every
        ``episode_count`` episode has landed and the dataset is safe to read.

        Call this once, after the recording session that owns this sink is
        done delivering episodes. Raises (Rule 10) if no episode was ever
        delivered, rather than writing a manifest claiming a dataset that
        doesn't exist.
        """
        if self._episode_count == 0:
            raise RuntimeError(
                "NTCloudSink.finalize: no episodes were delivered to this sink; "
                "refusing to write a manifest.json for an empty dataset."
            )
        manifest = {
            "format_version": self._format_version,
            "task": self._task,
            "episode_count": self._episode_count,
            "attribution": self._namespace,
            "created_at": _rfc3339_now(),
        }
        self._put(
            json.dumps(manifest).encode(),
            "manifest.json",
            failure_note=(
                "manifest.json was not written; the "
                f"{self._episode_count} already-uploaded episode(s) remain in "
                "place — safe to retry finalize() once the API is reachable."
            ),
        )

    def upload_directory(
        self,
        export_dir: Path,
        *,
        validate: bool = True,
        progress=None,
    ) -> str:
        """Upload every file under an already-exported dataset directory to this
        sink's dataset, then write the completeness ``manifest.json`` last.

        Unlike ``deliver()`` (which parses NT's own per-episode ``episode.json``),
        this takes the directory's contents as given — no format conversion, the
        export goes up as-is under the dataset prefix, relative paths preserved.

        All the files are signed in ONE round-trip against the batch shape of
        ``POST /api/uploads/sign`` (``{dataset, paths: [...]}`` → a manifest of
        signed URLs), then PUT one at a time. A realistic LeRobot export is
        hundreds of files (per-episode video); the batch shape keeps signing them
        to a single console round-trip instead of one per file.

        ``validate`` runs :func:`validate_lerobot_export` first — a malformed
        folder raises before a single byte is signed or uploaded. ``progress``, if
        given, is called as ``progress(done_files, total_files, done_bytes,
        total_bytes)`` after each file so a frontend can render an upload meter;
        the library itself prints nothing.

        Raises (Rule 10) if the directory is missing/empty, if the sign response
        doesn't return one URL per file, or if any PUT fails — never a silent
        partial upload. Returns the namespace the upload landed under (read from
        the server's response, never recomputed client-side).
        """
        export_dir = Path(export_dir)
        if not export_dir.is_dir():
            raise RuntimeError(
                f"NTCloudSink.upload_directory: not a directory: {export_dir}"
            )
        if validate:
            validate_lerobot_export(export_dir)

        files = sorted(
            p.relative_to(export_dir).as_posix()
            for p in export_dir.rglob("*")
            if p.is_file()
        )
        if not files:
            raise RuntimeError(
                f"NTCloudSink.upload_directory: no files found under {export_dir}"
            )

        signed = self._sign_batch(files, failure_note="nothing was uploaded.")
        entries = signed.get("urls") if isinstance(signed, dict) else None
        if not isinstance(entries, list) or len(entries) != len(files):
            raise RuntimeError(
                "NTCloudSink.upload_directory: the sign response did not return one "
                f"URL per file (asked for {len(files)}, got "
                f"{len(entries) if isinstance(entries, list) else entries!r}) — "
                "refusing to upload a partial set (Rule 10)."
            )

        url_by_path = {entry["path"]: entry for entry in entries}
        missing = [rel for rel in files if rel not in url_by_path]
        if missing:
            raise RuntimeError(
                f"NTCloudSink.upload_directory: {len(missing)} file(s) were not "
                f"signed by the server (e.g. {missing[0]!r}) — refusing to upload "
                "a partial set (Rule 10)."
            )

        if self._namespace is None:
            self._namespace = _namespace_from_object_path(
                url_by_path[files[0]]["objectPath"], self._dataset
            )

        total_bytes = sum((export_dir / rel).stat().st_size for rel in files)
        done_bytes = 0
        for i, rel in enumerate(files, start=1):
            data = (export_dir / rel).read_bytes()
            self._put_bytes_to(
                url_by_path[rel]["url"],
                data,
                rel,
                failure_note="the export remains on local disk, untouched.",
            )
            done_bytes += len(data)
            if progress is not None:
                progress(i, len(files), done_bytes, total_bytes)

        # Completeness sentinel, written last: its existence
        # means every file above landed. Signed on the single-file shape — it's one
        # file, added after the batch, so it never needs the batch round-trip.
        manifest = {
            "source_format": "lerobot-v3",
            "file_count": len(files),
            "attribution": self._namespace,
            "created_at": _rfc3339_now(),
        }
        self._put(
            json.dumps(manifest).encode(),
            "manifest.json",
            failure_note=(
                "manifest.json was not written; the "
                f"{len(files)} already-uploaded file(s) remain in place — "
                "safe to retry once the API is reachable."
            ),
        )
        return self._namespace

    # --- signed-URL + upload plumbing ---------------------------------------

    def _sign(self, remote_path: str, *, failure_note: str) -> dict:
        req = Request(
            f"{self._console_url}/api/uploads/sign",
            data=json.dumps({"dataset": self._dataset, "path": remote_path}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except HTTPError as exc:
            raise RuntimeError(
                f"NTCloudSink: failed to get a signed upload URL for {remote_path!r} "
                f"({exc.code} {exc.reason}); {failure_note}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"NTCloudSink: cannot reach {self._console_url} to sign "
                f"{remote_path!r}: {exc.reason}; {failure_note}"
            ) from exc

    def _sign_batch(self, remote_paths: list[str], *, failure_note: str) -> dict:
        """Sign every path in ONE round-trip against the batch shape of the sign
        route (``{dataset, paths: [...]}`` → ``{namespace, dataset, count, urls}``).
        Same Bearer-key auth and the same owner→namespace derivation as ``_sign``;
        only the request shape differs, so a directory upload can no more escape its
        namespace than a single-file one can. Returns the parsed manifest."""
        req = Request(
            f"{self._console_url}/api/uploads/sign",
            data=json.dumps({"dataset": self._dataset, "paths": remote_paths}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except HTTPError as exc:
            raise RuntimeError(
                f"NTCloudSink: failed to get signed upload URLs for "
                f"{len(remote_paths)} file(s) ({exc.code} {exc.reason}); {failure_note}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"NTCloudSink: cannot reach {self._console_url} to sign "
                f"{len(remote_paths)} file(s): {exc.reason}; {failure_note}"
            ) from exc

    def _upload(self, local_path: Path, remote_path: str) -> None:
        self._put(
            local_path.read_bytes(),
            remote_path,
            failure_note="episode remains on local disk, untouched.",
        )

    def _put(self, data: bytes, remote_path: str, *, failure_note: str) -> None:
        signed = self._sign(remote_path, failure_note=failure_note)
        if self._namespace is None:
            self._namespace = _namespace_from_object_path(signed["objectPath"], self._dataset)
        self._put_bytes_to(signed["url"], data, remote_path, failure_note=failure_note)

    def _put_bytes_to(
        self, url: str, data: bytes, remote_path: str, *, failure_note: str
    ) -> None:
        """PUT ``data`` to an already-signed GCS URL. Shared by the single-file
        (``_put``) and directory (``upload_directory``) paths so both cross the wire
        the exact same way — the signature mint differs, the PUT never does."""
        req = Request(
            url,
            data=data,
            headers={"Content-Type": _UPLOAD_CONTENT_TYPE},
            method="PUT",
        )
        try:
            with urlopen(req, timeout=self._timeout):
                pass
        except HTTPError as exc:
            raise RuntimeError(
                f"NTCloudSink: upload failed for {remote_path!r} ({exc.code} {exc.reason}); "
                f"{failure_note}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"NTCloudSink: upload failed for {remote_path!r}: {exc.reason}; {failure_note}"
            ) from exc
