"""The suite must not be able to write a real developer's ``~/.nt/credentials``.

This is a test about the test suite. It exists because the failure it guards is
invisible from inside a green run: a test that writes credentials without
redirecting the path passes, and the damage lands on the machine instead — the
developer's key replaced by a fixture value, and a CLI that 401s against
production until someone thinks to look at the file (portal#89).

So the assertion is not "credentials round-trip correctly" — other tests cover
that. It is "the path a credentials write lands on is not a real home", which is
the property that can only be checked here and can only be broken by deleting
the autouse guard in conftest.
"""
from __future__ import annotations

from newt import _credentials


def test_credentials_path_is_redirected_away_from_any_real_home(tmp_path_factory):
    """The constants must point inside pytest's tmp tree, not at ``Path.home()``.

    Patching ``HOME`` does not achieve this on its own: these are import-time
    constants, so they keep the value they were built with. If the autouse guard
    in conftest is removed, this reads the real developer's path and fails.
    """
    base = tmp_path_factory.getbasetemp()
    assert base in _credentials.CREDENTIALS_PATH.parents, (
        f"CREDENTIALS_PATH is {_credentials.CREDENTIALS_PATH}, which is outside pytest's "
        f"tmp tree ({base}) — a credentials write would land on a real machine."
    )


def test_write_api_key_lands_in_the_redirected_path(tmp_path_factory):
    """A write with no per-test patching must still be contained.

    This is the shape of the test nobody remembers to write safely: it calls the
    real writer and patches nothing itself. It has to be harmless anyway.
    """
    _credentials.write_api_key("nt_isolation_guard")

    written = _credentials.CREDENTIALS_PATH
    assert written.exists(), "write_api_key did not write where the constants point"
    assert tmp_path_factory.getbasetemp() in written.parents
    assert written.read_text().strip() == "api_key = nt_isolation_guard"
