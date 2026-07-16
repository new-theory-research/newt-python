"""Tests for `newt --version` / `newt -V` / `newt version`.

All three spellings must print the same version line and exit 0 — an agent
or developer checking "what version am I on" shouldn't have to guess which
flag this CLI accepts (issue newt-python#27).
"""
from __future__ import annotations

import io
import re
import sys

import pytest

from newt._cli import main


def _capture_main(argv, monkeypatch):
    """Run main() with sys.argv = ['newt'] + argv, capturing stdout/stderr and SystemExit."""
    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "argv", ["newt"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        main()
    return exc_info.value.code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("spelling", ["--version", "-V", "version"])
def test_version_exits_zero(spelling, monkeypatch):
    """Every spelling exits 0 — this is a query, never an error."""
    rc, out, err = _capture_main([spelling], monkeypatch)
    assert rc == 0, f"expected exit 0 for 'newt {spelling}'; rc={rc}, stderr={err!r}"


@pytest.mark.parametrize("spelling", ["--version", "-V", "version"])
def test_version_output_shape(spelling, monkeypatch):
    """Output is `newt <version>` — a single line, program name + version string."""
    rc, out, err = _capture_main([spelling], monkeypatch)
    assert re.match(r"^newt \S+\n$", out), (
        f"'newt {spelling}' must print 'newt <version>' on one line; got {out!r}"
    )


def test_version_spellings_agree(monkeypatch):
    """--version, -V, and version must report the identical version string."""
    _, out_flag, _ = _capture_main(["--version"], monkeypatch)
    _, out_short, _ = _capture_main(["-V"], monkeypatch)
    _, out_cmd, _ = _capture_main(["version"], monkeypatch)
    assert out_flag == out_short == out_cmd, (
        f"all three spellings must agree: --version={out_flag!r}, "
        f"-V={out_short!r}, version={out_cmd!r}"
    )


def test_global_help_mentions_version(monkeypatch):
    """`newt --help` must document the version command/flags for discoverability."""
    rc, out, err = _capture_main(["--help"], monkeypatch)
    assert rc == 0
    assert "version" in out.lower(), f"help must mention version: {out!r}"
