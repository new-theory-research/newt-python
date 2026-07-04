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

    # --- signed-URL + upload plumbing ---------------------------------------

    def _sign(self, remote_path: str) -> dict:
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
                f"({exc.code} {exc.reason}); episode remains on local disk, untouched."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"NTCloudSink: cannot reach {self._console_url} to sign "
                f"{remote_path!r}: {exc.reason}; episode remains on local disk, untouched."
            ) from exc

    def _upload(self, local_path: Path, remote_path: str) -> None:
        signed = self._sign(remote_path)
        if self._namespace is None:
            self._namespace = _namespace_from_object_path(signed["objectPath"], self._dataset)

        req = Request(
            signed["url"],
            data=local_path.read_bytes(),
            headers={"Content-Type": _UPLOAD_CONTENT_TYPE},
            method="PUT",
        )
        try:
            with urlopen(req, timeout=self._timeout):
                pass
        except HTTPError as exc:
            raise RuntimeError(
                f"NTCloudSink: upload failed for {remote_path!r} ({exc.code} {exc.reason}); "
                "episode remains on local disk, untouched."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"NTCloudSink: upload failed for {remote_path!r}: {exc.reason}; "
                "episode remains on local disk, untouched."
            ) from exc
