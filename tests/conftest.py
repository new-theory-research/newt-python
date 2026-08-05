"""Test-wide isolation for the two declaration surfaces ``newt`` now reads.

``resolve_spec`` looks for a rig's site config at ``$NT_SITE_CONFIG``, else
``~/.config/nt/nt.toml``, and for short names it reads the ``newt.sources.*``
entry points of whatever is installed in the environment (newtrino-029). Both
are real state on a real bench machine, which means an unfenced suite would
resolve *the developer's own rig* and pass or fail depending on whose laptop it
ran on — the CI-green, bench-red split that is worse than either.

So: every test starts with no ``NT_SITE_CONFIG``, a ``HOME`` that has no config
in it, and no installed kit declaring anything. A test that wants a declaration
writes one and points the env var at it, or hands ``_declared_sources`` a list,
out loud. Nothing about a source is ever ambient.
"""
from __future__ import annotations

import pytest

from newt._cli import _source_spec


@pytest.fixture(autouse=True)
def _no_ambient_site_config(monkeypatch, tmp_path_factory):
    monkeypatch.delenv("NT_SITE_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.setattr(_source_spec, "_declared_sources", lambda verb: [])
