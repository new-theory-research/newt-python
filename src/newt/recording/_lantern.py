"""The lantern refusal for using ``newt.recording`` without the ``recording`` extra.

Recording pulls in heavyweight, hardware-adjacent dependencies (``mcap`` for the
episode container, ``opencv-python`` for cameras, ``protobuf`` for the wire
schema). Bare ``import newt`` must stay featherweight — the brief-252 purity
golden is the tripwire — so those deps live behind ``pip install "newt[recording]"``
and are imported lazily, never at ``import newt`` time.

When a recording dep is missing, the user gets a lantern, not a stack trace: the
exact install command, the dep that was missing, and the one thing that fixes it.
"""
from __future__ import annotations


class RecordingExtraMissing(ImportError):
    """Raised when ``newt.recording`` is used without the ``recording`` extra.

    Subclasses ``ImportError`` so an ``except ImportError`` still catches it, but
    carries a message that names the fix instead of a bare missing-module trace.
    """


def require(module_name: str, dep: str) -> "object":
    """Import ``module_name``, or raise a lantern naming the ``recording`` extra.

    ``dep`` is the human-facing distribution that provides the module (e.g.
    ``mcap`` for ``mcap.writer``), used only in the message so the user reads a
    package name they can install, not an internal import path.
    """
    try:
        import importlib

        return importlib.import_module(module_name)
    except ImportError as exc:  # the dep isn't installed — light the lantern.
        raise RecordingExtraMissing(
            "\n".join(
                [
                    f"newt recording needs the '{dep}' package, which is not installed.",
                    "",
                    "Recording is an optional extra so that `import newt` stays light for",
                    "everyone who only runs inference. Install it once and this works:",
                    "",
                    '    pip install "newt[recording]"',
                    "",
                    f"(missing module: {module_name} — underlying error: {exc})",
                ]
            )
        ) from exc
