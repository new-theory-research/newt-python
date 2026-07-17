"""newt episodes — inspect recorded episodes and pull a staged dataset back down.

Frontend only. ``validate`` calls ``newt.recording.validate`` and renders the verdict; the
invariant logic is the library's. ``pull`` fetches your staged dataset's download manifest
from the console and downloads each object straight from storage — the bytes go GCS → your
machine, never through the console.

    newt episodes validate <dir>              human-readable PASS/FAIL per check
    newt episodes validate <dir> --json       structured verdict

    newt episodes pull <dataset>              download a staged dataset into ./<dataset>
    newt episodes pull <dataset> --dest DIR   download into DIR instead
    newt episodes pull <dataset> --json       machine-readable result (files, bytes)

``pull`` is authed with your ``nt_`` key (``NT_API_KEY`` or ``newt login``); it GETs the
owner-scoped manifest of signed READ URLs from ``/api/datasets/<name>/download`` and downloads
each one. Progress is reported by FILES COMPLETED, never an invented percentage. Rerunning is
resumable: a file already present at the manifest's size is skipped, so an interrupted pull
picks up where it left off. Featherweight on purpose: stdlib ``urllib`` only, no argparse.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from newt._credentials import read_api_key

_DEFAULT_CONSOLE = "https://newtheory-console.vercel.app"


def _usage() -> None:
    print("Usage: newt episodes <subcommand> [options]")
    print("")
    print("Subcommands:")
    print("  validate <dir>       Validate an episode_<id> directory against the NT v0.0.3 invariants.")
    print("  pull <dataset>       Download a staged dataset back down from your NT namespace.")
    print("")
    print("Options:")
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

    if sub not in ("validate", "pull"):
        print(f"newt episodes: unknown subcommand {sub!r}", file=sys.stderr)
        print("Run 'newt episodes --help' for usage.", file=sys.stderr)
        return 1

    if any(a in ("-h", "--help") for a in rest):
        _usage()
        return 0

    if sub == "pull":
        return _cmd_pull(rest)
    return _cmd_validate(rest)
