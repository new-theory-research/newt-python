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
from urllib.parse import quote
from urllib.request import Request, urlopen

from newt._credentials import read_api_key
from newt.recording._lantern import require

# Matches newt._cli.login._DEFAULT_CONSOLE — kept as a separate literal rather
# than importing the CLI package (recording must not depend on the CLI layer).
_DEFAULT_CONSOLE_URL = "https://newtheory-console.vercel.app"

# The Rerun exporter's literal placeholder task string, stamped on every frame
# when no --task is given at capture time. Mirrors
# training/intake/validators.py::PLACEHOLDER_TASK (portal repo, ft-002) — that
# module owns the single definition of "what counts as a placeholder task";
# this is a disclosed copy of the same literal value, not an independent
# definition (coordinate-don't-duplicate, capture-005-cont). newt-python and
# portal are separate repos with no shared package today, so a literal import
# isn't possible — if intake's definition ever needs to diverge from this one,
# that's an escalation, not a silent drift.
_PLACEHOLDER_TASK = "task"

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


def _read_task_texts(export_dir: Path) -> list[str] | None:
    """Read task strings from ``meta/tasks.parquet``, or ``None`` if absent.

    ``meta/tasks.parquet`` (a task-text-indexed table with an int
    ``task_index`` column — ``lerobot.datasets.io_utils.load_tasks``) is the
    real LeRobot v3.0 task-storage format; ``meta/tasks.jsonl`` is legacy,
    written only by the v2.1->v3.0 migration script and never present on a
    real v3.0 export (confirmed against ``training/intake/intake.py``'s own
    ``load_task_records``, portal repo). Absence isn't refused here —
    ``upload_directory`` also accepts directories that aren't LeRobot-v3
    exports, so a missing ``meta/tasks.parquet`` just means this check has
    nothing to check, not that the upload is invalid.
    """
    tasks_path = export_dir / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return None
    pq = require("pyarrow.parquet", "pyarrow")
    table = pq.read_table(tasks_path, columns=["task"])
    return table.column("task").to_pylist()


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
        self._uploaded_paths: list[str] | None = None

    @property
    def namespace(self) -> str | None:
        """The namespace this sink's uploads landed under, read from the
        server's sign response (capture-004) — never recomputed client-side.
        ``None`` until the first file has been signed."""
        return self._namespace

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

    def upload_directory(self, export_dir: Path) -> None:
        """Upload every file under an already-exported directory (e.g. a
        Rerun-exported LeRobot-v3 directory, capture-005-cont) to this sink's
        dataset, then write the dataset manifest last.

        Unlike ``deliver()``, which parses NT's own per-episode
        ``episode.json``, this takes the directory's contents as given: no
        format assumptions, no conversion — the export is uploaded as-is,
        preserving its relative paths under the dataset prefix. It reuses the
        same signed-URL upload plumbing (``_upload``/``_put``/``_sign``), not
        a forked mechanism.

        Raises (Rule 10) if the directory doesn't exist or is empty, or if
        any file's sign/PUT fails — no silent partial upload. Also raises,
        before any file is uploaded, if every task string in
        ``meta/tasks.parquet`` is the literal Rerun-exporter placeholder —
        coordinated with (not a fork of) ``training/intake``'s own refusal,
        which applies the same check later at training intake. Catching it
        here means a developer isn't told "your data is fine" at upload and
        "your task string is broken" only later at train.
        """
        export_dir = Path(export_dir)
        if not export_dir.is_dir():
            raise RuntimeError(
                f"NTCloudSink.upload_directory: not a directory: {export_dir}"
            )

        files = sorted(
            p.relative_to(export_dir).as_posix()
            for p in export_dir.rglob("*")
            if p.is_file()
        )
        if not files:
            raise RuntimeError(
                f"NTCloudSink.upload_directory: no files found under {export_dir}"
            )

        task_texts = _read_task_texts(export_dir)
        if task_texts and all(text == _PLACEHOLDER_TASK for text in task_texts):
            raise RuntimeError(
                f"NTCloudSink.upload_directory: {export_dir}'s task strings in "
                f"meta/tasks.parquet are only the placeholder {_PLACEHOLDER_TASK!r} "
                "— refusing to upload an unlabeled dataset (Rule 10: no silent "
                "placeholder passthrough). Fix: re-export from Rerun with --task, "
                "or run training/intake/intake.py --dataset-dir "
                f"{export_dir} --task \"<your task>\" to inject a real label into "
                "meta/tasks.parquet, then re-run this upload."
            )

        for rel in files:
            self._upload(export_dir / rel, rel)

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

        # Every remote path this call landed, in upload order (manifest last) —
        # the expected set verify_listing() checks the server's listing against.
        self._uploaded_paths = [*files, "manifest.json"]

    def verify_listing(self) -> dict:
        """Confirm — by reading the server back — that the files this sink
        uploaded actually landed under the developer's namespace, attributed.

        This is the objectively-checkable proof the hand-off completed
        (capture-005-cont task 3): it GETs ``/api/uploads/list?dataset=...``
        (the read-side mirror of the sign route, capture-006) and asserts every
        path ``upload_directory`` uploaded appears in the server's listing. The
        namespace is the one the server reports, cross-checked against the one
        the sign responses reported during upload — both come from the server
        (capture-004), neither is recomputed here.

        Returns the parsed listing (``{"namespace", "count", "objects"}``).
        Raises (Rule 10) if nothing was uploaded to verify, if the server
        reports a different namespace than the upload landed under, or if any
        uploaded file is missing from the listing — a missing file means the
        hand-off did NOT fully land, and that must fail loud, never be reported
        as success.
        """
        if self._uploaded_paths is None:
            raise RuntimeError(
                "NTCloudSink.verify_listing: nothing has been uploaded yet; "
                "call upload_directory() before verifying the listing."
            )

        listing = self._list(self._dataset)

        # The listing's namespace and the upload's namespace both come from the
        # server (capture-004); if they disagree, the read-side and write-side
        # resolved different identities — surface it rather than trust either.
        listed_namespace = listing.get("namespace")
        if self._namespace is not None and listed_namespace != self._namespace:
            raise RuntimeError(
                f"NTCloudSink.verify_listing: listing namespace {listed_namespace!r} "
                f"does not match the namespace the upload landed under "
                f"{self._namespace!r} — refusing to report a mismatched attribution."
            )

        # The listing's `path` is <dataset>/<relative-path> (namespace stripped,
        # apps/console/app/api/uploads/list/route.ts) — match against the same
        # <dataset>/<rel> shape our uploads landed at.
        listed_paths = {obj["path"] for obj in listing.get("objects", [])}
        missing = [
            rel for rel in self._uploaded_paths
            if f"{self._dataset}/{rel}" not in listed_paths
        ]
        if missing:
            raise RuntimeError(
                f"NTCloudSink.verify_listing: {len(missing)} uploaded file(s) are "
                f"missing from the server listing for dataset {self._dataset!r} "
                f"(e.g. {missing[0]!r}) — the hand-off did not fully land; "
                "refusing to report success."
            )

        return listing

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

    def _list(self, dataset: str) -> dict:
        """GET ``/api/uploads/list?dataset=...`` — the read-side mirror of
        ``_sign`` (capture-006). Same Bearer-key auth; the server derives the
        namespace from the key's owner and scopes the listing to it (the SDK
        can't name another identity's namespace). Returns the parsed JSON."""
        req = Request(
            f"{self._console_url}/api/uploads/list?dataset={quote(dataset, safe='')}",
            headers={"Authorization": f"Bearer {self._api_key}"},
            method="GET",
        )
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except HTTPError as exc:
            raise RuntimeError(
                f"NTCloudSink: failed to list dataset {dataset!r} to verify the "
                f"upload landed ({exc.code} {exc.reason})."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"NTCloudSink: cannot reach {self._console_url} to list "
                f"dataset {dataset!r}: {exc.reason}."
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

        req = Request(
            signed["url"],
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
