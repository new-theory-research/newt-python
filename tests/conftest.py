"""Test-wide isolation for the two declaration surfaces ``newt`` now reads.

``resolve_spec`` looks for a rig's site config at ``$NT_SITE_CONFIG``, else
``~/.config/nt/nt.toml``, and for short names it reads the ``newt.sources.*``
entry points of whatever is installed in the environment (newtrino-029). Both
are real state on a real bench machine, which means an unfenced suite would
resolve *the developer's own rig* and pass or fail depending on whose laptop it
ran on — the CI-green, bench-red split that is worse than either.

So: every test starts with no ``NT_SITE_CONFIG``, a ``HOME`` that has no config
in it, no installed kit declaring anything, and no kit in the directory pytest
was launched from. A test that wants a declaration writes one and points the env
var at it, or hands ``_declared_sources`` a list, out loud. Nothing about a
source is ever ambient.
"""
from __future__ import annotations

import pytest

from newt import _credentials
from newt._cli import _lease, _source_spec, logout, status


@pytest.fixture(autouse=True)
def _no_ambient_site_config(monkeypatch, tmp_path_factory):
    monkeypatch.delenv("NT_SITE_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.setattr(_source_spec, "_declared_sources", lambda verb: [])
    # Both entry-point reads, not just the one. ``_declaring_verbs`` answers
    # "is anything publishing to newt here at all", which is a *different*
    # question about the same real registry — patching only the first would
    # leave every nothing-declared refusal reading the developer's own
    # environment and printing a different sentence on a laptop than in CI.
    monkeypatch.setattr(_source_spec, "_declaring_verbs", lambda: [])
    # The third read, and the only one that is not the entry-point registry:
    # a refusal now checks whether the *working directory* holds a kit that
    # publishes this verb, to hand `uv run newt <verb>` instead of a lecture
    # about environments. pytest runs from wherever the developer invoked it —
    # unfenced, a refusal would notice this repo's own pyproject on one machine
    # and nothing on another.
    monkeypatch.setattr(_source_spec, "_cwd_kit_declaring", lambda verb: None)


@pytest.fixture(autouse=True)
def _no_real_credentials_file(monkeypatch, tmp_path_factory):
    """No test can reach the developer's real ``~/.nt/credentials``.

    Setting ``HOME`` above is not enough and it is worth being explicit about why:
    ``CREDENTIALS_DIR``/``CREDENTIALS_PATH`` are ``Path.home()``-derived constants
    evaluated once at *import* time, so a later ``HOME`` monkeypatch leaves them
    pointing at the real file. Every credentials test today also patches those
    constants by hand — which means the suite is safe by discipline, and one test
    written without that line would silently overwrite a real developer's key.

    A stolen afternoon has already been paid for this shape once (portal#89): a
    fixture wrote a placeholder over a real ``~/.nt/credentials`` and the CLI 401'd
    against production for a day. Redirecting here makes it safe by construction,
    so the guard cannot be forgotten rather than merely remembered.

    ``logout`` and ``status`` bind the constants into their own module namespaces
    with ``from ... import``, so they are retargeted too — patching only the
    defining module would leave those two aimed at the real path.
    """
    home = tmp_path_factory.mktemp("nt-home")
    cred_dir = home / ".nt"
    cred_path = cred_dir / "credentials"
    monkeypatch.setattr(_credentials, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(_credentials, "CREDENTIALS_PATH", cred_path)
    monkeypatch.setattr(logout, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(logout, "CREDENTIALS_PATH", cred_path)
    monkeypatch.setattr(status, "CREDENTIALS_PATH", cred_path)


@pytest.fixture(autouse=True)
def _no_real_arms_lease(monkeypatch, tmp_path_factory):
    """No test can take, expire or delete the arms lease on a real bench.

    Same import-time-constant trap as the credentials file above, and a worse
    blast radius: ``~/.nt/arms.lease`` is what tells a running ``newt collect``
    that it still owns the motors. A test that wrote one would refuse a
    developer's next ``newt teleop`` with a session that never existed; a test
    that deleted one would hand a second claimant the arms of a rig that is
    mid-recording.
    """
    home = tmp_path_factory.mktemp("nt-lease-home")
    lease_dir = home / ".nt"
    monkeypatch.setattr(_lease, "LEASE_DIR", lease_dir)
    monkeypatch.setattr(_lease, "LEASE_PATH", lease_dir / "arms.lease")
    monkeypatch.setattr(_lease, "LOCK_PATH", lease_dir / "arms.lease.lock")
