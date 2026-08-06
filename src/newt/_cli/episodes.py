"""newt episodes — inspect recorded episodes, push them up, and pull a staged dataset back down.

Frontend only. ``validate`` calls ``newt.recording.validate`` and renders the verdict; the
invariant logic is the library's. ``push`` drives ``newt.recording.NTCloudSink`` — it holds no
upload behaviour of its own. ``pull`` fetches your staged dataset's download manifest from the
console and downloads each object straight from storage — the bytes go GCS → your machine,
never through the console.

    newt episodes validate <dir>              human-readable PASS/FAIL per check
    newt episodes validate <dir> --json       structured verdict

    newt episodes push --dataset <name>       upload ./episodes into your NT namespace
    newt episodes push <dir> --dataset <name> upload a different recording directory
    newt episodes push ... --json             machine-readable result (episodes, files, bytes)

    newt episodes pull <dataset>              download a staged dataset into ./<dataset>
    newt episodes pull <dataset> --dest DIR   download into DIR instead
    newt episodes pull <dataset> --json       machine-readable result (files, bytes)

``push`` and ``pull`` are authed with your ``nt_`` key (``NT_API_KEY`` or ``newt login``).
``pull`` GETs the owner-scoped manifest of signed READ URLs from
``/api/datasets/<name>/download`` and downloads each one; ``push`` checks the owner-scoped
listing at ``/api/uploads/list`` and then hands each episode to ``NTCloudSink.deliver``, which
signs and PUTs straight to storage. Progress is reported by EPISODES COMPLETED, never an
invented percentage. Rerunning a pull is resumable: a file already present at the manifest's
size is skipped. Rerunning a push under the same name is NOT — the store is write-once, so
``push`` refuses before it signs anything. Featherweight on purpose: stdlib ``urllib`` only,
no argparse.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from newt._cli.finetune import _format_size
from newt._credentials import read_api_key

_DEFAULT_CONSOLE = "https://newtheory-console.vercel.app"

# `newt record --dest` defaults here, so `newt episodes push` defaults to the same place: the
# developer who just ran record types `newt episodes push --dataset <name>` and it works.
_DEFAULT_EPISODES_DIR = "episodes"


def _usage() -> None:
    print("Usage: newt episodes <subcommand> [options]")
    print("")
    print("Subcommands:")
    print("  validate <dir>       Validate an episode_<id> directory against the NT v0.0.3 invariants.")
    print("  push [dir]           Upload recorded episodes (default: ./episodes) into your NT namespace.")
    print("  pull <dataset>       Download a staged dataset back down from your NT namespace.")
    print("")
    print("Options:")
    print("  --dataset <name>     (push) The dataset name to land the episodes under. Required.")
    print("  --dest <dir>         (pull) Where to write the dataset (default: ./<dataset>).")
    print("  --json               Emit machine-readable JSON.")
    print("")
    print("Environment:")
    print("  NT_API_KEY      API key override (overrides ~/.nt/credentials).")
    print("  NT_CONSOLE_URL  Console URL (default: https://newtheory-console.vercel.app)")
    print("")
    print("  Validation needs the extra:  pip install \"newt[recording]\"")


def _console_url() -> str:
    return os.environ.get("NT_CONSOLE_URL", _DEFAULT_CONSOLE).rstrip("/")


def _resolve_key() -> str | None:
    return os.environ.get("NT_API_KEY") or read_api_key()


def _opt_value(args: list[str], name: str) -> str | None:
    """Value for ``--name X`` or ``--name=X``. None if absent or the value is missing / looks
    like another flag (so ``--dest --json`` is a missing value, not a dir literally ``--json``)."""
    for i, a in enumerate(args):
        if a == name:
            nxt = args[i + 1] if i + 1 < len(args) else None
            return nxt if (nxt and not nxt.startswith("-")) else None
        if a.startswith(name + "="):
            return a[len(name) + 1 :] or None
    return None


def _positionals(args: list[str], value_flags: set[str]) -> list[str]:
    """Positional args, skipping flags AND the values of value-taking flags — so
    ``pull my-ds --dest ./out`` yields ``["my-ds"]``, not ``["my-ds", "./out"]``."""
    out: list[str] = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a in value_flags:
            skip = True
            continue
        if a.startswith("-"):
            continue
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# HTTP round-trips (split out so tests exercise the orchestration + rendering
# without a network).
# ---------------------------------------------------------------------------
def _fetch_manifest(console: str, api_key: str, dataset: str, *, timeout: float = 30.0) -> dict:
    url = f"{console}/api/datasets/{quote(dataset, safe='')}/download"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _list_dataset_objects(console: str, api_key: str, dataset: str, *, timeout: float = 30.0) -> dict:
    """The owner-scoped listing for one dataset name (``GET /api/uploads/list?dataset=``).

    This is ``push``'s pre-flight: the store is create-only, so the only way to know a name is
    free is to ask before signing anything. The route derives the namespace from the key's
    OWNER — no parameter here can widen it (apps/console/app/api/uploads/list/route.ts)."""
    url = f"{console}/api/uploads/list?dataset={quote(dataset, safe='')}"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _download_url(url: str, *, timeout: float = 300.0) -> bytes:
    """Download one signed URL's bytes. The signed URL carries its own auth (it's minted for
    this object) — no Bearer header here; adding one would break the GCS signature."""
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _should_skip(dest_path: Path, size: object) -> bool:
    """Resume rule (v1): skip a file already present at the manifest's size. A null/unknown
    size never fabricates a match — the file is re-downloaded rather than assumed complete
    (Rule 10). No partial-byte resume in v1."""
    if not dest_path.is_file():
        return False
    if not isinstance(size, int):
        return False
    return dest_path.stat().st_size == size


def _cmd_pull(rest: list[str]) -> int:
    as_json = "--json" in rest
    dest_opt = _opt_value(rest, "--dest")
    positional = _positionals(rest, {"--dest"})

    # Instructional/progress output goes to stderr in --json mode so stdout carries nothing but
    # the final JSON object (composable with $(...) / jq).
    out = sys.stderr if as_json else sys.stdout

    if not positional:
        print("newt episodes pull: a dataset name is required.", file=sys.stderr)
        print("        Fix: newt episodes pull my-dataset", file=sys.stderr)
        return 1
    dataset = positional[0]

    api_key = _resolve_key()
    if not api_key:
        print(
            "newt: no API key found.\n"
            "  Run `newt login` to authenticate, or set NT_API_KEY.",
            file=sys.stderr,
        )
        return 1

    console = _console_url()
    dest_root = Path(dest_opt) if dest_opt else Path(dataset)

    try:
        manifest = _fetch_manifest(console, api_key, dataset)
    except HTTPError as exc:
        if exc.code == 401:
            print("newt episodes pull: authentication failed — your key was rejected.", file=sys.stderr)
            print("  Rotate your key in the console, or run `newt login` again.", file=sys.stderr)
        elif exc.code == 404:
            print(
                f"newt episodes pull: no dataset named {dataset!r} for your key (404).",
                file=sys.stderr,
            )
            print("  Check the name in the console's Datasets page, or that the upload finished.", file=sys.stderr)
        elif exc.code == 413:
            print(
                f"newt episodes pull: dataset {dataset!r} is too large for a single-page pull (413).",
                file=sys.stderr,
            )
            print("  Paginated pull is a planned follow-up — flag it if you hit this.", file=sys.stderr)
        else:
            print(f"newt episodes pull: download failed ({exc.code}): {exc.reason}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"newt episodes pull: cannot reach {console}: {exc.reason}", file=sys.stderr)
        print("  Set NT_CONSOLE_URL if you're running a local console.", file=sys.stderr)
        return 1

    urls = manifest.get("urls") if isinstance(manifest, dict) else None
    if not isinstance(urls, list):
        # A manifest with no urls array is a contract violation — surface it, don't treat it
        # as "empty dataset" (Rule 10).
        print(f"newt episodes pull: malformed manifest (no urls array): {manifest!r}", file=sys.stderr)
        return 1

    total = len(urls)
    if total == 0:
        print(f"newt episodes pull: dataset {dataset!r} has no files to download.", file=out)
        if as_json:
            print(json.dumps({"dataset": dataset, "dest": str(dest_root), "total_files": 0,
                              "downloaded": 0, "skipped": 0, "bytes": 0}))
        return 0

    print(f"Pulling {dataset!r} → {dest_root}/  ({total} files)", file=out, flush=True)

    downloaded = 0
    skipped = 0
    total_bytes = 0
    completed = 0
    for entry in urls:
        if not isinstance(entry, dict):
            print(f"newt episodes pull: malformed manifest entry: {entry!r}", file=sys.stderr)
            return 1
        rel = entry.get("path")
        url = entry.get("url")
        if not isinstance(rel, str) or not isinstance(url, str):
            print(f"newt episodes pull: manifest entry missing path/url: {entry!r}", file=sys.stderr)
            return 1

        dest_path = dest_root / rel
        completed += 1
        if _should_skip(dest_path, entry.get("size")):
            skipped += 1
            print(f"  {completed}/{total}  {rel}  (already present, skipped)", file=out, flush=True)
            continue

        try:
            data = _download_url(url)
        except (HTTPError, URLError) as exc:
            reason = getattr(exc, "reason", exc)
            print(f"\nnewt episodes pull: failed downloading {rel!r}: {reason}", file=sys.stderr)
            return 1

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(data)
        downloaded += 1
        total_bytes += len(data)
        # Progress by FILES COMPLETED — never an invented percentage (Rule 10).
        print(f"  {completed}/{total}  {rel}", file=out, flush=True)

    if as_json:
        print(json.dumps({
            "dataset": dataset,
            "dest": str(dest_root),
            "total_files": total,
            "downloaded": downloaded,
            "skipped": skipped,
            "bytes": total_bytes,
        }))
    else:
        print(
            f"Done — {downloaded} downloaded, {skipped} already present. "
            f"{dataset!r} is in {dest_root}/",
            file=out,
        )
    return 0


# ---------------------------------------------------------------------------
# push — the local recording directory → your NT namespace.
#
# Every refusal below happens BEFORE the first byte is signed, and that ordering is the
# whole design. The store is create-only: `nt-episodes-writer` holds
# `roles/storage.objectCreator` and nothing else, no delete-scoped SA exists, and
# `deleteEpisodePrefix` unconditionally throws (portal#98, and the ground-truth comment at
# apps/console/app/api/uploads/sign/route.ts). So a dataset name is spent the moment one
# object lands under it. A refusal that arrives mid-upload doesn't just fail — it costs the
# developer a name they can never reuse.
# ---------------------------------------------------------------------------
def _discover_episodes(source: Path) -> list[Path]:
    """The ``episode_<id>`` directories under ``source``, sorted. Matches what the writer
    commits (``newt record`` writes one per kept episode) and nothing else — a stray file or
    a scratch folder in the recording directory is not an episode and is never uploaded."""
    return sorted(
        p for p in source.iterdir() if p.is_dir() and p.name.startswith("episode_")
    )


def _episode_task(meta: object) -> str | None:
    """The task name out of a parsed ``episode.json`` — read exactly the way
    ``NTCloudSink.deliver`` reads it (``episode_config.task_name``), so the pre-flight check
    and the sink's own guard can never disagree about what a directory's tasks are."""
    if not isinstance(meta, dict):
        return None
    config = meta.get("episode_config")
    if not isinstance(config, dict):
        return None
    task = config.get("task_name")
    return task if isinstance(task, str) else None


def _dataset_files(objects: object, dataset: str) -> list[str] | None:
    """The relative paths already under ``<dataset>/`` in the caller's namespace, from an
    ``/api/uploads/list`` response. ``None`` means the response didn't have the documented
    shape — a contract violation the caller must surface, never read as "empty" (Rule 10).

    Mirrors the sign route's own batch guard: directory placeholders (a trailing ``/``) and
    the bare prefix itself don't count as files."""
    if not isinstance(objects, list):
        return None
    prefix = f"{dataset}/"
    files = []
    for entry in objects:
        if not isinstance(entry, dict):
            return None
        path = entry.get("path")
        if not isinstance(path, str):
            return None
        if path.startswith(prefix) and not path.endswith("/") and len(path) > len(prefix):
            files.append(path)
    return files


def _cmd_push(rest: list[str]) -> int:
    as_json = "--json" in rest
    dataset = _opt_value(rest, "--dataset")
    positional = _positionals(rest, {"--dataset"})

    # Same door as pull: in --json mode stdout carries nothing but the final object.
    out = sys.stderr if as_json else sys.stdout

    if not dataset:
        print("newt episodes push: --dataset <name> is required.", file=sys.stderr)
        print(
            "  A dataset name is write-once and can't be renamed later, so it is never "
            "guessed from the directory name.",
            file=sys.stderr,
        )
        print("  Fix: newt episodes push ./episodes --dataset cup-demo", file=sys.stderr)
        return 1

    source = Path(positional[0]) if positional else Path(_DEFAULT_EPISODES_DIR)

    api_key = _resolve_key()
    if not api_key:
        # Cause 1 — worded exactly as `pull` words it. One phrasing for one cause across
        # the CLI, not a seventh variant of "no key".
        print(
            "newt: no API key found.\n"
            "  Run `newt login` to authenticate, or set NT_API_KEY.",
            file=sys.stderr,
        )
        return 1

    console = _console_url()

    # --- what is actually on disk (cause 5, and its three shapes) -------------------------
    if not source.is_dir():
        print(f"newt episodes push: no such directory: {source}", file=sys.stderr)
        print(
            "  Point at the directory `newt record` wrote — its --dest, which defaults to "
            "./episodes.",
            file=sys.stderr,
        )
        print(f"  Fix: newt episodes push ./episodes --dataset {dataset}", file=sys.stderr)
        return 1

    episode_dirs = _discover_episodes(source)
    if not episode_dirs:
        print(
            f"newt episodes push: {source} holds no episode_<id> directories — "
            "nothing to push.",
            file=sys.stderr,
        )
        print(
            "  `newt record` writes one episode_<id> directory per kept episode. Check "
            "you're pointing at the recording destination and not at a single episode. A "
            "training export is a different format and a different verb: `newt finetune "
            "--dataset`.",
            file=sys.stderr,
        )
        return 1

    plan: list[tuple[Path, str | None, int, int]] = []  # (dir, task, file count, bytes)
    for episode_dir in episode_dirs:
        meta_path = episode_dir / "episode.json"
        if not meta_path.is_file():
            print(
                f"newt episodes push: {episode_dir.name} has no episode.json — it was "
                "never committed, so it can't be pushed.",
                file=sys.stderr,
            )
            print(
                "  A recording interrupted before the episode was committed leaves a "
                "directory like this behind.",
                file=sys.stderr,
            )
            print(
                f"  Fix: delete {episode_dir} or re-record it, then push again. Nothing "
                "was uploaded.",
                file=sys.stderr,
            )
            return 1
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"newt episodes push: cannot read {meta_path}: {exc}",
                file=sys.stderr,
            )
            print(
                "  The episode's own metadata is unreadable, so its task can't be "
                "confirmed and it can't be pushed with the others.",
                file=sys.stderr,
            )
            print(
                f"  Fix: run `newt episodes validate {episode_dir}` to see what's wrong. "
                "Nothing was uploaded.",
                file=sys.stderr,
            )
            return 1

        files = [p for p in episode_dir.rglob("*") if p.is_file()]
        plan.append((
            episode_dir,
            _episode_task(meta),
            len(files),
            sum(p.stat().st_size for p in files),
        ))

    # --- one task per dataset (cause 4) ---------------------------------------------------
    # NTCloudSink.deliver already refuses a mixed-task dataset — but it finds out on the
    # SECOND episode, after the first one has already landed, which spends the name. The
    # directory accumulates across `newt record` runs, so this is a normal thing to hit.
    by_task: dict[str | None, list[str]] = {}
    for episode_dir, task, _, _ in plan:
        by_task.setdefault(task, []).append(episode_dir.name)
    if len(by_task) > 1:
        print(
            f"newt episodes push: {source} holds more than one task, and a dataset carries "
            "exactly one.",
            file=sys.stderr,
        )
        for task, names in by_task.items():
            label = repr(task) if task is not None else "(no task recorded)"
            print(f"    {label} — {', '.join(names)}", file=sys.stderr)
        print(
            "  This directory accumulates across `newt record` runs. Fix: push one task at "
            "a time — move each task's episodes into their own directory and pass that "
            "directory. Nothing was uploaded.",
            file=sys.stderr,
        )
        return 1

    task = plan[0][1]
    total_files = sum(n for _, _, n, _ in plan)
    total_bytes = sum(b for _, _, _, b in plan)

    # --- is the name free? (causes 2 and 3) -----------------------------------------------
    try:
        listing = _list_dataset_objects(console, api_key, dataset)
    except HTTPError as exc:
        if exc.code == 401:
            print(
                "newt episodes push: authentication failed — your key was rejected (401).",
                file=sys.stderr,
            )
            print(
                "  Rotate your key in the console, or run `newt login` again. Nothing was "
                "uploaded.",
                file=sys.stderr,
            )
        else:
            print(
                f"newt episodes push: couldn't check whether {dataset!r} is free — the "
                f"console's listing returned {exc.code} {exc.reason}.",
                file=sys.stderr,
            )
            print(
                "  Refusing to upload without that check: in a write-once store, pushing "
                "into a name that already has objects burns the name. This is ours, not "
                "your data — retry in a moment, and report it if it persists.",
                file=sys.stderr,
            )
        return 1
    except URLError as exc:
        print(
            f"newt episodes push: cannot reach {console} to check whether {dataset!r} is "
            f"free: {exc.reason}",
            file=sys.stderr,
        )
        print(
            "  Refusing to upload without that check — a name that already has objects "
            "can't be pushed into or cleaned up. Fix: check your connection, or set "
            "NT_CONSOLE_URL if you're running a local console.",
            file=sys.stderr,
        )
        return 1

    existing = _dataset_files(
        listing.get("objects") if isinstance(listing, dict) else None, dataset
    )
    if existing is None:
        print(
            f"newt episodes push: malformed listing response (no objects array): "
            f"{listing!r}",
            file=sys.stderr,
        )
        print(
            "  That's a contract violation on our side, not a full dataset — refusing to "
            "read it as 'the name is free'. Please report it. Nothing was uploaded.",
            file=sys.stderr,
        )
        return 1

    if existing:
        has_manifest = f"{dataset}/manifest.json" in existing
        if has_manifest:
            print(
                f"newt episodes push: dataset {dataset!r} already exists in your namespace "
                f"and is complete ({len(existing)} files, manifest.json present).",
                file=sys.stderr,
            )
            print(
                "  Uploads are write-once — an existing name can't be overwritten, added "
                "to, or deleted today.",
                file=sys.stderr,
            )
            print(
                f"  Fix: push under a new name, e.g. --dataset {dataset}_v2. Nothing was "
                "uploaded and your local episodes are untouched.",
                file=sys.stderr,
            )
        else:
            print(
                f"newt episodes push: dataset {dataset!r} already holds {len(existing)} "
                "file(s) in your namespace but no manifest.json — an earlier push started "
                "under this name and never finished.",
                file=sys.stderr,
            )
            print(
                "  Those bytes can't be resumed, replaced, or cleaned up today (the store "
                "is create-only and no delete exists yet — portal#98), so this name is "
                "spent. Nothing reads it as a complete dataset.",
                file=sys.stderr,
            )
            print(
                f"  Fix: push under a new name, e.g. --dataset {dataset}_v2. Nothing was "
                "uploaded and your local episodes are untouched.",
                file=sys.stderr,
            )
        return 1

    namespace = listing.get("namespace") if isinstance(listing, dict) else None

    # --- the preflight report: everything the developer needs to stop here ----------------
    print(
        f"Pushing {len(plan)} episode(s) from {source}/ → dataset {dataset!r}",
        file=out,
    )
    print(f"  task:       {task if task is not None else '(none recorded)'}", file=out)
    print(f"  contents:   {total_files} files, {_format_size(total_bytes)}", file=out)
    if isinstance(namespace, str):
        print(f"  namespace:  {namespace}", file=out)
    print(
        "  note:       uploads are write-once — this name can't be reused, added to, or "
        "deleted afterwards.",
        file=out,
        flush=True,
    )

    from newt.recording import NTCloudSink

    sink = NTCloudSink(dataset, api_key=api_key, console_url=console)

    landed = 0
    for i, (episode_dir, _, n_files, n_bytes) in enumerate(plan, start=1):
        # Progress by EPISODES COMPLETED — never an invented percentage (Rule 10). The
        # count is written before the upload starts so a long video transfer isn't silent.
        print(
            f"  {i}/{len(plan)}  {episode_dir.name}  ({n_files} files, "
            f"{_format_size(n_bytes)}) ",
            file=out,
            end="",
            flush=True,
        )
        try:
            sink.deliver(episode_dir)
        except Exception as exc:  # the sink raises RuntimeError; never swallow anything else
            print("failed", file=out, flush=True)
            print(f"\nnewt episodes push: {exc}", file=sys.stderr)
            # What the sink can't know and the push can: how far it got, and what that
            # leaves behind (Rule 10 — a partial push never reports success).
            print(
                f"  {landed} of {len(plan)} episode(s) landed before this. Dataset "
                f"{dataset!r} is now PARTIAL in your namespace: no manifest.json was "
                "written, so nothing reads it as complete — but the name is spent, "
                "because what did land can't be deleted (portal#98).",
                file=sys.stderr,
            )
            print(
                f"  Fix: push the whole directory again under a NEW name, e.g. --dataset "
                f"{dataset}_v2. Your local episodes were not touched.",
                file=sys.stderr,
            )
            return 1
        landed += 1
        print("landed", file=out, flush=True)

    try:
        sink.finalize()
    except Exception as exc:
        print(f"\nnewt episodes push: {exc}", file=sys.stderr)
        print(
            f"  All {landed} episode(s) landed, but manifest.json — the marker that says "
            f"the dataset is complete — did not. {dataset!r} will read as still uploading "
            "in the console and won't be pulled or trained on.",
            file=sys.stderr,
        )
        print(
            "  Re-running this push will refuse (the name now has objects), and there is "
            "no repair path for it today. Please report this. Your local episodes were "
            "not touched.",
            file=sys.stderr,
        )
        return 1

    if as_json:
        print(json.dumps({
            "dataset": dataset,
            "source": str(source),
            "namespace": namespace,
            "task": task,
            "episodes": len(plan),
            "files": total_files,
            "bytes": total_bytes,
        }))
    else:
        print(
            f"Done — {len(plan)} episode(s) landed in {dataset!r}. "
            f"Pull them back with: newt episodes pull {dataset}",
            file=out,
        )
    return 0


def _cmd_validate(rest: list[str]) -> int:
    as_json = "--json" in rest
    positional = [a for a in rest if not a.startswith("-")]
    if not positional:
        print("newt episodes validate: a directory is required.", file=sys.stderr)
        print("        Fix: newt episodes validate ./episodes/episode_abcd1234", file=sys.stderr)
        return 1

    episode_dir = Path(positional[0])

    try:
        from newt.recording import validate
    except Exception as exc:
        print(f"[newt episodes] {exc}", file=sys.stderr)
        return 1

    try:
        result = validate(episode_dir)
    except Exception as exc:
        # Lantern (missing extra) or a read error — surface it, don't trace.
        print(f"[newt episodes] {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        mark = "PASS" if result["valid"] else "FAIL"
        print(f"[{mark}] {result['episode']}")
        for c in result["checks"]:
            print(f"  {'ok ' if c['ok'] else 'BAD'} {c['check']}: {c['detail']}")

    return 0 if result["valid"] else 1


def cmd_episodes(args: list[str]) -> int:
    # Bare `newt episodes` or a top-level -h/--help prints usage.
    if not args or args[0] in ("-h", "--help"):
        _usage()
        return 0

    sub = args[0]
    rest = args[1:]

    if sub not in ("validate", "push", "pull"):
        print(f"newt episodes: unknown subcommand {sub!r}", file=sys.stderr)
        print("Run 'newt episodes --help' for usage.", file=sys.stderr)
        return 1

    if any(a in ("-h", "--help") for a in rest):
        _usage()
        return 0

    if sub == "push":
        return _cmd_push(rest)
    if sub == "pull":
        return _cmd_pull(rest)
    return _cmd_validate(rest)
