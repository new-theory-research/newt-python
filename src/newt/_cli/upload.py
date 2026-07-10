"""newt upload — land an already-exported directory (e.g. a Rerun-exported
LeRobot-v3 directory) in your NT cloud namespace.

Frontend only: walks the directory and drives ``NTCloudSink.upload_directory``
— capture-002's proven signed-URL mechanism (key -> /api/uploads/sign -> PUT),
pointed at a directory-of-files input instead of a live recording session. No
new upload protocol, no format conversion — the directory goes up as-is.

    newt upload <dir> --dataset NAME          upload, print the landed namespace
    newt upload <dir> --dataset NAME --json   same, machine-readable
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _usage() -> None:
    print("Usage: newt upload <dir> --dataset NAME [options]")
    print("")
    print("  Upload an already-exported directory (e.g. a Rerun-exported")
    print("  LeRobot-v3 directory) into your NT cloud namespace, via")
    print("  NTCloudSink's signed-URL mechanism (key -> sign -> PUT).")
    print("")
    print("Options:")
    print("  --dataset NAME  Dataset name this directory lands under (required).")
    print("  --json          Emit a machine-readable JSON result.")
    print("")
    print("Environment:")
    print("  NT_API_KEY     API key override (overrides ~/.nt/credentials).")
    print("  NT_CONSOLE_URL Console URL (default: https://newtheory-console.vercel.app).")


def cmd_upload(args: list[str]) -> int:
    if not args or any(a in ("-h", "--help") for a in args):
        _usage()
        return 0

    as_json = "--json" in args
    dataset: str | None = None
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--dataset":
            i += 1
            if i >= len(args):
                print("newt upload: --dataset expects a value", file=sys.stderr)
                return 1
            dataset = args[i]
        elif a == "--json":
            pass
        elif not a.startswith("-"):
            positional.append(a)
        else:
            print(f"newt upload: unknown option {a!r}", file=sys.stderr)
            print("Run 'newt upload --help' for usage.", file=sys.stderr)
            return 1
        i += 1

    if not positional:
        print("newt upload: a directory is required.", file=sys.stderr)
        print("        Fix: newt upload ./exported_dataset --dataset my-task", file=sys.stderr)
        return 1
    if not dataset:
        print("newt upload: --dataset is required.", file=sys.stderr)
        print("        Fix: newt upload ./exported_dataset --dataset my-task", file=sys.stderr)
        return 1

    export_dir = Path(positional[0])

    try:
        from newt.recording import NTCloudSink

        sink = NTCloudSink(dataset)
        sink.upload_directory(export_dir)
    except Exception as exc:
        # Missing key, sign/PUT failure, or a bad directory — surface it, don't trace.
        print(f"[newt upload] {exc}", file=sys.stderr)
        return 1

    namespace = sink.namespace
    if as_json:
        print(json.dumps({"dataset": dataset, "namespace": namespace}))
    else:
        print(f"[newt upload] {export_dir} -> dataset {dataset!r} landed under namespace {namespace}")

    return 0
