"""Test-wide isolation for the one file ``newt`` now reads off the machine.

``resolve_spec`` looks for a rig's site config at ``$NT_SITE_CONFIG``, else
``~/.config/nt/nt.toml`` (newtrino-029). That is real config on a real bench
machine, which means an unfenced suite would resolve *the developer's own rig*
and pass or fail depending on whose laptop it ran on — the CI-green,
bench-red split that is worse than either.

So: every test starts with no ``NT_SITE_CONFIG`` and a ``HOME`` that has no
config in it. A test that wants a declaration writes one and points the env var
at it, out loud. Nothing about a source is ever ambient.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_site_config(monkeypatch, tmp_path_factory):
    monkeypatch.delenv("NT_SITE_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
