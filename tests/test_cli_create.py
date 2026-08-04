"""Tests for ``newt create`` — the verb, and the registry it resolves against.

Two things are being protected here.

**The verb must stay ignorant of robots.** ``newt create`` exists to know a
template *name* and nothing else; every fact about a rig — how many cameras, what
they are called, which driver version matches its firmware — belongs to the kit
being dispensed. The proof that the line held is that adding a second robot costs
a registry row and no code. ``test_verb_knows_no_robots`` is that proof as an
assertion: rig names may appear in the registry module and nowhere else in the
verb.

**A refusal must name its own cause.** "No template called that" and "that one is
private and needs your key" and "the console did not answer" are three different
problems with three different fixes, and a developer reading one of them must not
have to guess which of the other two it might really be.
"""
from __future__ import annotations

import io
import json
import pathlib
import re
import sys

import pytest

from newt._cli import create as create_mod
from newt._cli._template_registry import (
    FALLBACK_TEMPLATES,
    Template,
    names,
    parse_templates,
)
from newt._cli.create import cmd_create


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _sandboxed(monkeypatch, tmp_path):
    """No test in this file may touch the network or the developer's cwd.

    ``newt create`` writes directories, so an un-stubbed run must not be able to
    scatter one into the repository. And a test that forgot to stub its fetch would
    otherwise reach GitHub for real — slow, flaky, and a different thing than what
    it meant to assert. Both are made impossible here rather than remembered.
    """
    monkeypatch.chdir(tmp_path)

    def blocked(req, timeout):
        from urllib.error import URLError

        raise URLError("the network is disabled in this test")

    monkeypatch.setattr(create_mod, "_read_url", blocked)


def _capture(args, monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    rc = cmd_create(args)
    return rc, out.getvalue(), err.getvalue()


def _no_key(monkeypatch):
    monkeypatch.setattr(create_mod, "_resolve_key", lambda: None)


def _with_key(monkeypatch, key="nt_testkey"):
    monkeypatch.setattr(create_mod, "_resolve_key", lambda: key)


def _console_returns(monkeypatch, payload):
    monkeypatch.setattr(create_mod, "_fetch_registry", lambda *a, **kw: payload)


def _console_down(monkeypatch, reason="Connection refused"):
    from urllib.error import URLError

    def boom(*a, **kw):
        raise URLError(reason)

    monkeypatch.setattr(create_mod, "_fetch_registry", boom)


def _console_rejects_key(monkeypatch, code=401):
    from urllib.error import HTTPError

    def boom(*a, **kw):
        raise HTTPError("http://console/api/cli/templates", code, "Unauthorized", {}, None)

    monkeypatch.setattr(create_mod, "_fetch_registry", boom)


# A registry the console might serve that is deliberately NOT the built-in table —
# so a test asserting "the error printed the registry's real contents" can tell the
# difference between reading the registry and reciting a hardcoded list.
_CONSOLE_PAYLOAD = {
    "templates": [
        {"name": "alpha", "visibility": "public", "repo": "org/alpha", "ref": "a" * 40},
        {"name": "beta-private", "visibility": "private"},
    ]
}


# ---------------------------------------------------------------------------
# Help guard — uniform with every other verb: usage, exit 0, zero network
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_exits_zero_and_makes_no_network_call(flag, monkeypatch):
    calls = []
    monkeypatch.setattr(create_mod, "_fetch_registry", lambda *a, **kw: calls.append(a))
    rc, out, err = _capture([flag], monkeypatch)
    assert rc == 0, f"expected exit 0; stderr={err!r}"
    assert "usage" in out.lower()
    assert calls == [], "newt create --help must not touch the network"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_describes_the_action(flag, monkeypatch):
    rc, out, err = _capture([flag], monkeypatch)
    lowered = out.lower()
    assert "starter" in lowered, f"create help must say what it dispenses: {out!r}"


def test_global_help_lists_create(monkeypatch):
    """A verb nobody can discover is a verb nobody uses."""
    from newt._cli import _usage as global_usage

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    global_usage()
    assert "create" in out.getvalue()


# ---------------------------------------------------------------------------
# Bare `newt create` — usage plus the templates that actually exist
# ---------------------------------------------------------------------------

def test_bare_create_lists_the_console_registry(monkeypatch):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    rc, out, err = _capture([], monkeypatch)
    assert rc == 0
    assert "alpha" in out and "beta-private" in out, out


def test_bare_create_marks_private_templates_as_needing_a_key(monkeypatch):
    """A developer scanning the list must be able to see which kits will ask for a key."""
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    rc, out, err = _capture([], monkeypatch)
    private_line = [ln for ln in out.splitlines() if "beta-private" in ln]
    assert private_line and "private" in private_line[0].lower(), out


def test_bare_create_falls_back_when_console_is_down_and_says_so(monkeypatch):
    """Offline is not empty. The built-in table answers, and the output admits it may be stale."""
    _no_key(monkeypatch)
    _console_down(monkeypatch)
    rc, out, err = _capture([], monkeypatch)
    assert rc == 0
    for name in names(FALLBACK_TEMPLATES):
        assert name in out, f"{name!r} missing from the offline listing: {out!r}"
    assert "console was not reachable" in out, out


def test_json_listing_is_machine_readable(monkeypatch):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    rc, out, err = _capture(["--json"], monkeypatch)
    assert rc == 0
    payload = json.loads(out)
    assert payload["registry_source"] == "console"
    assert [t["name"] for t in payload["templates"]] == ["alpha", "beta-private"]


def test_json_listing_says_when_it_is_offline(monkeypatch):
    """An agent reading JSON needs the same staleness warning a human gets in prose."""
    _no_key(monkeypatch)
    _console_down(monkeypatch)
    rc, out, err = _capture(["--json"], monkeypatch)
    assert rc == 0
    assert json.loads(out)["registry_source"] == "fallback"


# ---------------------------------------------------------------------------
# Unknown template — the error prints the registry's REAL contents
# ---------------------------------------------------------------------------

def test_unknown_template_exits_no_such_template(monkeypatch):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    rc, out, err = _capture(["not-a-real-robot"], monkeypatch)
    assert rc == create_mod.EXIT_NO_SUCH_TEMPLATE
    assert "not-a-real-robot" in err


def test_unknown_template_prints_the_live_registry_not_a_hardcoded_list(monkeypatch):
    """The available-templates line must be read out of the registry that answered.

    The console here serves a set that shares no name with the built-in table, so a
    message assembled from a literal list in the source would fail this.
    """
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    rc, out, err = _capture(["nope"], monkeypatch)
    assert "alpha" in err and "beta-private" in err, err
    for stale in names(FALLBACK_TEMPLATES):
        assert stale not in err, (
            f"{stale!r} came from the built-in table, but the console answered with a "
            f"different registry — the error is reciting, not reading: {err!r}"
        )


def test_unknown_template_offline_prints_the_fallback_contents(monkeypatch):
    _no_key(monkeypatch)
    _console_down(monkeypatch)
    rc, out, err = _capture(["nope"], monkeypatch)
    assert rc == create_mod.EXIT_NO_SUCH_TEMPLATE
    for name in names(FALLBACK_TEMPLATES):
        assert name in err, err


# ---------------------------------------------------------------------------
# The private-template path — three causes, three strings
# ---------------------------------------------------------------------------

def test_private_template_without_a_key_says_it_is_private(monkeypatch):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    rc, out, err = _capture(["beta-private"], monkeypatch)
    assert rc == create_mod.EXIT_NEEDS_KEY
    assert "private" in err.lower()
    assert "newt login" in err, "the refusal must name the next step"


def test_private_template_with_console_down_says_the_console_is_down(monkeypatch):
    """Offline is not 'no such template', and it is not 'you need a key' either."""
    _with_key(monkeypatch)
    _console_down(monkeypatch, "Name or service not known")
    rc, out, err = _capture(["trossen-widowx"], monkeypatch)
    assert rc == create_mod.EXIT_CONSOLE_UNREACHABLE
    assert "cannot reach the console" in err.lower(), err
    assert "Name or service not known" in err, "the refusal must carry the actual reason"


def test_console_answering_an_error_is_not_the_same_as_silence(monkeypatch):
    """A 404 from a console that is up is not a network outage, and the fix differs."""
    from urllib.error import HTTPError

    _with_key(monkeypatch)

    _console_down(monkeypatch, "Connection refused")
    _, _, silent = _capture(["trossen-widowx"], monkeypatch)

    def four_oh_four(*a, **kw):
        raise HTTPError("http://console/api/cli/templates", 404, "Not Found", {}, None)

    monkeypatch.setattr(create_mod, "_fetch_registry", four_oh_four)
    _, _, answered = _capture(["trossen-widowx"], monkeypatch)

    assert silent.splitlines()[0] != answered.splitlines()[0], (
        f"an outage and a 404 share one message: {silent.splitlines()[0]!r}"
    )
    assert "404" in answered, answered


def test_rejected_key_is_its_own_string(monkeypatch):
    _with_key(monkeypatch)
    _console_rejects_key(monkeypatch)
    rc, out, err = _capture(["alpha"], monkeypatch)
    assert rc == create_mod.EXIT_KEY_REJECTED
    assert "rejected your key" in err.lower(), err


def test_public_template_still_resolves_with_the_console_down(monkeypatch):
    """A public kit has somewhere else to come from, so an outage must not block it."""
    _no_key(monkeypatch)
    _console_down(monkeypatch)
    rc, out, err = _capture(["so101"], monkeypatch)
    assert rc != create_mod.EXIT_NO_SUCH_TEMPLATE
    assert rc != create_mod.EXIT_CONSOLE_UNREACHABLE


def test_the_four_refusals_share_no_string(monkeypatch):
    """Rule 12's hard edge: two causes must never produce the same message.

    Collected from real runs of the four failure paths rather than asserted about
    string literals, so a future edit that collapses two messages fails here.
    """
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _, _, not_found = _capture(["nope"], monkeypatch)
    _, _, needs_key = _capture(["beta-private"], monkeypatch)

    _with_key(monkeypatch)
    _console_down(monkeypatch)
    _, _, unreachable = _capture(["trossen-widowx"], monkeypatch)

    _console_rejects_key(monkeypatch)
    _, _, rejected = _capture(["so101"], monkeypatch)

    firsts = [s.splitlines()[0] for s in (not_found, needs_key, unreachable, rejected)]
    assert len(set(firsts)) == 4, f"two causes share a message: {firsts}"


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------

def test_malformed_registry_is_an_error_not_an_empty_list():
    """An unreadable registry must never render as 'you have access to nothing'."""
    for payload in ({}, {"templates": "so101"}, {"templates": [{"visibility": "public"}]}):
        with pytest.raises(ValueError):
            parse_templates(payload)


def test_unknown_visibility_is_refused():
    """A visibility the client does not understand is not quietly treated as public."""
    with pytest.raises(ValueError):
        parse_templates({"templates": [{"name": "x", "visibility": "internal"}]})


def test_fallback_table_publishes_no_private_repository():
    """This package is world-readable. A private kit's location is not ours to publish."""
    for t in FALLBACK_TEMPLATES:
        if t.is_private:
            assert t.repo is None and t.ref is None, (
                f"{t.name!r} is private but its repository is in the public client table"
            )


def test_public_fallback_rows_are_pinned_to_a_commit():
    """A branch name is not a pin. Running the same command next month must produce the
    same directory, and only a SHA promises that."""
    for t in FALLBACK_TEMPLATES:
        if not t.is_private:
            assert t.repo and t.ref, f"{t.name!r} has no pinned location"
            assert re.fullmatch(r"[0-9a-f]{40}", t.ref), (
                f"{t.name!r} is pinned to {t.ref!r}, which is not a commit SHA"
            )


# ---------------------------------------------------------------------------
# The invariant: the SDK stays embodiment-agnostic
# ---------------------------------------------------------------------------

_RIG_WORDS = re.compile(
    r"widowx|trossen|realsense|so101|surrounding[12]|right-wrist|yam|aloha|viperx",
    re.IGNORECASE,
)


def test_verb_knows_no_robots():
    """No robot may be named anywhere in the create verb except the registry table.

    This is the check that makes the second-body claim real: if adding a rig means
    editing ``create.py``, the SDK has learned something about robots and the next
    one costs code instead of a row.
    """
    src = pathlib.Path(create_mod.__file__).read_text(encoding="utf-8")
    hits = _RIG_WORDS.findall(src)
    assert not hits, (
        f"create.py names {sorted(set(h.lower() for h in hits))} — rig knowledge belongs "
        "in the kit, and rig names belong in the registry"
    )


def test_verb_has_no_template_shaped_branch():
    """A ``if template == "<a robot>"`` is the exact shape this verb must never grow."""
    src = pathlib.Path(create_mod.__file__).read_text(encoding="utf-8")
    assert not re.search(r'(name|template)\s*==\s*["\']', src), (
        "create.py compares a template name against a literal — that is per-rig behaviour "
        "living in the SDK"
    )


def test_a_second_robot_costs_a_row(monkeypatch):
    """The second-body proof, in miniature: a template the code has never heard of
    resolves through exactly the same path as the ones that shipped."""
    _no_key(monkeypatch)
    _console_returns(
        monkeypatch,
        {"templates": [{"name": "brand-new-arm", "visibility": "public",
                        "repo": "org/newt-starter-brand-new-arm", "ref": "b" * 40}]},
    )
    rc, out, err = _capture(["brand-new-arm"], monkeypatch)
    assert rc not in (
        create_mod.EXIT_NO_SUCH_TEMPLATE,
        create_mod.EXIT_NEEDS_KEY,
        create_mod.EXIT_CONSOLE_UNREACHABLE,
    ), f"a registry row was not enough to make a new robot resolvable: {err!r}"


def test_registry_rows_carry_no_rig_configuration():
    """A row is a name, a visibility, and where to fetch it. Not a camera list."""
    allowed = {"name", "visibility", "repo", "ref"}
    assert set(Template.__dataclass_fields__) == allowed, (
        f"the registry row grew fields beyond {sorted(allowed)} — rig configuration is "
        "the kit's, not the SDK's"
    )


# ---------------------------------------------------------------------------
# Acquisition — the bytes, and what the directory looks like afterwards
#
# The claim under test is the ownership tenet, made checkable: what lands is a
# project, not a checkout of ours. Everything else here — the stripped root, the
# refusal to unpack onto existing work, the archive that tries to climb out of the
# target — is in service of that directory being safe to call yours.
# ---------------------------------------------------------------------------

import subprocess
import tarfile


def _tarball(root="starter-alpha-abc123", files=None, extra=None):
    """Build a GitHub-shaped archive: everything under one top-level directory."""
    files = files if files is not None else {
        "README.md": "# Alpha starter kit\n",
        "pyproject.toml": "[project]\nname = 'kit'\n",
        "conf/nt.toml.example": "# measure and replace\n",
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel, body in files.items():
            data = body.encode()
            info = tarfile.TarInfo(f"{root}/{rel}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for info in extra or ():
            tar.addfile(info, io.BytesIO(b""))
    return buf.getvalue()


def _serves(monkeypatch, archive, record=None):
    """Stub the one function that touches the wire, capturing what was asked for."""
    def _read(req, timeout):
        if record is not None:
            record["url"] = req.full_url
            record["auth"] = req.headers.get("Authorization")
        return archive

    monkeypatch.setattr(create_mod, "_read_url", _read)


def _raises(monkeypatch, exc):
    def _read(req, timeout):
        raise exc

    monkeypatch.setattr(create_mod, "_read_url", _read)


def _http_error(status, reason, code=None):
    from urllib.error import HTTPError

    payload = json.dumps({"code": code, "error": reason}).encode() if code else b"{}"
    return HTTPError("http://console/x", status, reason, {}, io.BytesIO(payload))


def test_a_public_kit_lands_as_files_with_the_archive_root_stripped(monkeypatch, tmp_path):
    """The directory a developer named holds the kit's files, not a folder named after us."""
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _tarball())
    dest = tmp_path / "my-arm"

    rc, out, err = _capture(["alpha", str(dest)], monkeypatch)

    assert rc == create_mod.EXIT_OK, err
    assert (dest / "README.md").read_text().startswith("# Alpha")
    assert (dest / "conf" / "nt.toml.example").exists()
    assert not (dest / "starter-alpha-abc123").exists()
    assert str(dest) in out


def test_the_scaffolded_directory_has_no_remote(monkeypatch, tmp_path):
    """THE ownership assertion: `git -C <dir> remote -v` prints nothing.

    Not "no origin named ours" — nothing at all. A directory that remembers where it
    came from is a checkout; this one is a project.
    """
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _tarball())
    dest = tmp_path / "mine"

    rc, _, err = _capture(["alpha", str(dest)], monkeypatch)
    assert rc == create_mod.EXIT_OK, err

    proc = subprocess.run(
        ["git", "-C", str(dest), "remote", "-v"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "", f"the scaffold carries a remote: {proc.stdout!r}"
    # And no history of ours to disown: the repository is empty, so the developer's
    # first commit is the first commit.
    log = subprocess.run(
        ["git", "-C", str(dest), "log", "--oneline"], capture_output=True, text=True
    )
    assert log.returncode != 0 or log.stdout == "", log.stdout


def test_no_git_leaves_the_directory_inert_and_says_so(monkeypatch, tmp_path):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _tarball())
    dest = tmp_path / "inert"

    rc, out, err = _capture(["alpha", str(dest), "--no-git"], monkeypatch)

    assert rc == create_mod.EXIT_OK, err
    assert not (dest / ".git").exists()
    assert "NOT a git repository" in out


def test_the_directory_defaults_to_the_template_name(monkeypatch, tmp_path):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _tarball())

    rc, out, err = _capture(["alpha"], monkeypatch)

    assert rc == create_mod.EXIT_OK, err
    assert (tmp_path / "alpha" / "README.md").exists()


def test_an_existing_non_empty_directory_is_refused_by_name_not_overwritten(monkeypatch, tmp_path):
    """Unpacking onto someone's work would mix two trees with no way to tell them apart."""
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _tarball())
    dest = tmp_path / "occupied"
    dest.mkdir()
    (dest / "my-notes.txt").write_text("hours of work")

    rc, out, err = _capture(["alpha", str(dest)], monkeypatch)

    assert rc == create_mod.EXIT_TARGET_EXISTS
    assert str(dest) in err
    assert (dest / "my-notes.txt").read_text() == "hours of work"
    assert not (dest / "README.md").exists()


def test_an_existing_empty_directory_is_fine(monkeypatch, tmp_path):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _tarball())
    dest = tmp_path / "empty"
    dest.mkdir()

    rc, _, err = _capture(["alpha", str(dest)], monkeypatch)
    assert rc == create_mod.EXIT_OK, err


def test_a_public_kit_is_fetched_direct_from_github_at_the_pinned_commit(monkeypatch, tmp_path):
    """The cold path: no key, no console in the download, a commit not a branch."""
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    seen = {}
    _serves(monkeypatch, _tarball(), seen)

    rc, _, err = _capture(["alpha", str(tmp_path / "d")], monkeypatch)

    assert rc == create_mod.EXIT_OK, err
    row = next(t for t in _CONSOLE_PAYLOAD["templates"] if t["name"] == "alpha")
    assert seen["url"] == f"https://codeload.github.com/{row['repo']}/tar.gz/{row['ref']}"
    assert seen["auth"] is None, "a public kit must not need a credential"


def test_a_private_kit_comes_through_the_console_against_the_developers_key(monkeypatch, tmp_path):
    """GitHub never enters the developer's story — the console brokers, and the
    repository path never reaches this machine."""
    _with_key(monkeypatch, "nt_realkey")
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    seen = {}
    _serves(monkeypatch, _tarball(root="newt-starter-widowx-def456"), seen)
    monkeypatch.setenv("NT_CONSOLE_URL", "https://console.example")

    rc, _, err = _capture(["beta-private", str(tmp_path / "p")], monkeypatch)

    assert rc == create_mod.EXIT_OK, err
    assert seen["url"] == "https://console.example/api/cli/templates/beta-private/tarball"
    assert seen["auth"] == "Bearer nt_realkey"
    assert "codeload" not in seen["url"]


def test_a_private_kit_without_a_key_never_reaches_the_wire(monkeypatch, tmp_path):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)

    def never(req, timeout):
        raise AssertionError("asked the console to broker a kit for an anonymous caller")

    monkeypatch.setattr(create_mod, "_read_url", never)

    rc, _, err = _capture(["beta-private", str(tmp_path / "p")], monkeypatch)
    assert rc == create_mod.EXIT_NEEDS_KEY
    assert "private" in err


def test_the_console_rejecting_a_key_mid_download_is_its_own_exit(monkeypatch, tmp_path):
    _with_key(monkeypatch, "nt_stale")
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _raises(monkeypatch, _http_error(401, "Unauthorized", code="key_rejected"))

    rc, _, err = _capture(["beta-private", str(tmp_path / "p")], monkeypatch)
    assert rc == create_mod.EXIT_KEY_REJECTED
    assert not (tmp_path / "p").exists()


def test_a_console_that_cannot_dispense_says_it_is_ours_not_yours(monkeypatch, tmp_path):
    """The failure a developer must never read as 'you may not have this'."""
    _with_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _raises(monkeypatch, _http_error(503, "Service Unavailable", code="dispensing_unconfigured"))

    rc, _, err = _capture(["beta-private", str(tmp_path / "p")], monkeypatch)
    assert rc == create_mod.EXIT_FETCH_FAILED
    assert "ours, not yours" in err
    assert "your key is fine" in err.lower()


def test_github_refusing_the_archive_is_not_a_missing_template(monkeypatch, tmp_path):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _raises(monkeypatch, _http_error(404, "Not Found"))

    rc, _, err = _capture(["alpha", str(tmp_path / "d")], monkeypatch)
    assert rc == create_mod.EXIT_FETCH_FAILED
    assert "no template named" not in err.lower()
    assert not (tmp_path / "d").exists()


def test_a_download_that_dies_midway_writes_nothing(monkeypatch, tmp_path):
    from urllib.error import URLError

    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _raises(monkeypatch, URLError("Connection reset by peer"))

    rc, _, err = _capture(["alpha", str(tmp_path / "d")], monkeypatch)
    assert rc == create_mod.EXIT_FETCH_FAILED
    assert not (tmp_path / "d").exists()


def test_an_archive_that_climbs_out_of_the_target_is_refused_whole(monkeypatch, tmp_path):
    """Not skipped quietly — refused. A partial unpack missing the file that mattered
    is worse than a refusal that says what happened."""
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    escaping = tarfile.TarInfo("starter-alpha-abc123/../../evil.txt")
    escaping.size = 0
    _serves(monkeypatch, _tarball(extra=[escaping]))
    dest = tmp_path / "d"

    rc, _, err = _capture(["alpha", str(dest)], monkeypatch)

    assert rc == create_mod.EXIT_BAD_ARCHIVE
    assert not (tmp_path / "evil.txt").exists()
    assert not (dest / "README.md").exists()


def test_an_archive_without_a_single_root_is_refused(monkeypatch, tmp_path):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    stray = tarfile.TarInfo("second-root/file.txt")
    stray.size = 0
    _serves(monkeypatch, _tarball(extra=[stray]))

    rc, _, err = _capture(["alpha", str(tmp_path / "d")], monkeypatch)
    assert rc == create_mod.EXIT_BAD_ARCHIVE


def test_bytes_that_are_not_an_archive_are_refused_not_unpacked(monkeypatch, tmp_path):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, b"<html>404 not found</html>")

    rc, _, err = _capture(["alpha", str(tmp_path / "d")], monkeypatch)
    assert rc == create_mod.EXIT_BAD_ARCHIVE


def test_json_output_names_the_directory_the_ref_and_the_absent_remote(monkeypatch, tmp_path):
    """The agent door: everything the keyboard path prints, machine-readable."""
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _tarball())
    dest = tmp_path / "j"

    rc, out, err = _capture(["alpha", str(dest), "--json"], monkeypatch)

    assert rc == create_mod.EXIT_OK, err
    payload = json.loads(out)
    row = next(t for t in _CONSOLE_PAYLOAD["templates"] if t["name"] == "alpha")
    assert payload["directory"] == str(dest)
    assert payload["ref"] == row["ref"]
    assert payload["remote"] is None
    assert payload["git"] == "initialized"
    assert payload["files"] == 3


def test_every_acquisition_refusal_still_shares_no_string(monkeypatch, tmp_path):
    """Rule 12 extended over the download surface. Collected from real runs, so a
    future edit that collapses two of these fails here rather than in a developer's
    terminal at 2am."""
    from urllib.error import URLError

    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    collected = []

    _no_key(monkeypatch)
    collected.append(_capture(["nope", str(tmp_path / "a")], monkeypatch)[2])
    collected.append(_capture(["beta-private", str(tmp_path / "b")], monkeypatch)[2])

    _raises(monkeypatch, _http_error(404, "Not Found"))
    collected.append(_capture(["alpha", str(tmp_path / "c")], monkeypatch)[2])

    _raises(monkeypatch, URLError("Connection reset by peer"))
    collected.append(_capture(["alpha", str(tmp_path / "d")], monkeypatch)[2])

    _with_key(monkeypatch)
    _raises(monkeypatch, _http_error(503, "Service Unavailable", code="dispensing_unconfigured"))
    collected.append(_capture(["beta-private", str(tmp_path / "e")], monkeypatch)[2])

    _raises(monkeypatch, _http_error(502, "Bad Gateway", code="upstream_unavailable"))
    collected.append(_capture(["beta-private", str(tmp_path / "f")], monkeypatch)[2])

    _raises(monkeypatch, _http_error(401, "Unauthorized", code="key_rejected"))
    collected.append(_capture(["beta-private", str(tmp_path / "g")], monkeypatch)[2])

    _serves(monkeypatch, b"not-a-tarball")
    collected.append(_capture(["alpha", str(tmp_path / "h")], monkeypatch)[2])

    occupied = tmp_path / "i"
    occupied.mkdir()
    (occupied / "x").write_text("x")
    _serves(monkeypatch, _tarball())
    collected.append(_capture(["alpha", str(occupied)], monkeypatch)[2])

    firsts = [s.splitlines()[0] for s in collected]
    assert len(set(firsts)) == len(firsts), f"two causes share a message: {firsts}"


def test_acquisition_added_no_robot_knowledge_to_the_verb(monkeypatch, tmp_path):
    """The fence, re-checked after the fetch landed: unpacking is the same code for
    every kit, so a second body still costs a registry row and nothing else."""
    _no_key(monkeypatch)
    _console_returns(
        monkeypatch,
        {"templates": [{"name": "brand-new-arm", "visibility": "public",
                        "repo": "org/newt-starter-brand-new-arm", "ref": "b" * 40}]},
    )
    _serves(monkeypatch, _tarball(root="newt-starter-brand-new-arm-bbb"))

    rc, out, err = _capture(["brand-new-arm", str(tmp_path / "new")], monkeypatch)
    assert rc == create_mod.EXIT_OK, err
    assert (tmp_path / "new" / "README.md").exists()
