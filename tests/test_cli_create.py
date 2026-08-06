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


def _console_errors(monkeypatch, status=500, reason="Internal Server Error"):
    """Something answered and said no — a different problem from silence, and the
    CLI is required to tell them apart."""
    from urllib.error import HTTPError

    def boom(*a, **kw):
        raise HTTPError("http://console/api/cli/templates", status, reason, {}, None)

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


def _http_error(status, reason, code=None, error=None):
    """A console error body — its machine-readable ``code`` and the sentence it
    wrote for a human. ``error`` is separable from ``reason`` on purpose: the
    console says things the CLI cannot know, and a test has to be able to prove
    that sentence reached the developer rather than being replaced by a status."""
    from urllib.error import HTTPError

    payload = (
        json.dumps({"code": code, "error": error if error is not None else reason}).encode()
        if code
        else b"{}"
    )
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


def test_the_refusal_suggests_the_first_free_suffixed_name(monkeypatch, tmp_path):
    """The fix line is a command a developer can paste, not just a diagnosis.

    No directory argument was given, so the template name doubles as the dir —
    the suggestion must suffix that, and it must name a directory that is
    actually free on disk right now, not a guess.
    """
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _tarball())
    dest = tmp_path / "alpha"
    dest.mkdir()
    (dest / "x").write_text("x")

    rc, _out, err = _capture(["alpha"], monkeypatch)

    assert rc == create_mod.EXIT_TARGET_EXISTS
    assert "newt create alpha alpha-2" in err
    assert not (tmp_path / "alpha-2").exists()


def test_the_suggestion_skips_a_suffix_that_is_also_taken(monkeypatch, tmp_path):
    """First free wins — alpha-2 occupied too means the fix line offers alpha-3."""
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _tarball())
    dest = tmp_path / "alpha"
    dest.mkdir()
    (dest / "x").write_text("x")
    (tmp_path / "alpha-2").mkdir()

    rc, _out, err = _capture(["alpha"], monkeypatch)

    assert rc == create_mod.EXIT_TARGET_EXISTS
    assert "newt create alpha alpha-3" in err
    assert "alpha-2" not in err.replace("alpha-3", "")


def test_the_suggestion_suffixes_the_given_directory_not_the_template_name(monkeypatch, tmp_path):
    """An explicit directory argument, not the template name, is what gets suffixed."""
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _tarball())
    dest = tmp_path / "mydir"
    dest.mkdir()
    (dest / "x").write_text("x")

    rc, _out, err = _capture(["alpha", "mydir"], monkeypatch)

    assert rc == create_mod.EXIT_TARGET_EXISTS
    assert "newt create alpha mydir-2" in err
    assert not (tmp_path / "mydir-2").exists()


def test_dot_as_the_target_is_refused_not_a_traceback(monkeypatch, tmp_path):
    """``newt create <tmpl> .`` against a non-empty cwd has no basename to suffix.

    ``pathlib.Path(".").name`` is ``""``, and ``Path.with_name`` raises on an
    empty name — the suggestion logic must not let that escape as a traceback.
    The refusal still fires, still exits EXIT_TARGET_EXISTS, and the fallback
    "name a directory that does not exist yet" line takes over from the
    suffixed-name suggestion.
    """
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _tarball())
    (tmp_path / "already-here.txt").write_text("not empty")

    rc, _out, err = _capture(["alpha", "."], monkeypatch)

    assert rc == create_mod.EXIT_TARGET_EXISTS
    assert "Fix:" in err
    assert "Traceback" not in err


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
    assert rc == create_mod.EXIT_CONSOLE_REFUSED
    assert rc != create_mod.EXIT_FETCH_FAILED, (
        "the console refused before GitHub was ever asked — nothing was downloaded"
    )
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


# ---------------------------------------------------------------------------
# The handoff — the kit's setup step, and the arguments the verb never reads
#
# This is where Invariant 1 is most likely to break, because the mandate wants a
# create flow that asks about arm IPs and camera IDs and this verb is forbidden to
# know what either of those is. The resolution is that the asking belongs to the
# kit and the verb's whole job is to hand over an argv list it never looks at.
# What follows tests the seam from both sides: the arguments arrive intact, and
# nothing about a rig ever gets read on the way.
# ---------------------------------------------------------------------------

_SETUP_OK = "#!/bin/sh\nprintf 'kit setup ran: %s\\n' \"$*\"\nexit 0\n"


def _kit_tarball(root="starter-alpha-abc123", setup=None, mode=0o755, files=None):
    """A GitHub-shaped archive that can carry a real, executable setup step.

    ``_tarball`` above cannot set a mode bit, and the exec bit is exactly what
    "this kit declares a setup step" means here — so these tests build their own.
    """
    files = dict(files if files is not None else {"README.md": "# Alpha starter kit\n"})
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel, body in files.items():
            data = body.encode()
            info = tarfile.TarInfo(f"{root}/{rel}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if setup is not None:
            data = setup.encode()
            info = tarfile.TarInfo(f"{root}/scripts/setup")
            info.size = len(data)
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _public_alpha(monkeypatch, archive):
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, archive)


def _records_setup(monkeypatch):
    """Stub the subprocess call, capturing exactly what would have been run."""
    seen = {}
    real = create_mod.subprocess.run

    def _run(cmd, **kw):
        if cmd and str(cmd[0]).endswith("scripts/setup"):
            seen["argv"] = list(cmd)
            seen["cwd"] = kw.get("cwd")
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, **kw)

    monkeypatch.setattr(create_mod.subprocess, "run", _run)
    return seen


# --- the default: say what the next step is, don't take it ------------------

def test_the_setup_step_does_not_run_unless_it_is_asked_for(monkeypatch, tmp_path):
    """A script that arrived seconds ago, which writes configuration outside the
    directory that was asked for, does not get to run on 'make me a directory'."""
    _public_alpha(monkeypatch, _kit_tarball(setup=_SETUP_OK))
    seen = _records_setup(monkeypatch)

    rc, out, err = _capture(["alpha", str(tmp_path / "k")], monkeypatch)
    assert rc == create_mod.EXIT_OK, err
    assert "argv" not in seen, "the setup step ran without being asked"
    assert "./scripts/setup" in out, f"the next step must be named: {out!r}"


def test_a_kit_with_no_setup_step_points_at_its_readme_instead(monkeypatch, tmp_path):
    """Not every kit ships one — the yam starter does not — and that is information,
    not a refusal."""
    _public_alpha(monkeypatch, _kit_tarball())

    rc, out, err = _capture(["alpha", str(tmp_path / "k")], monkeypatch)
    assert rc == create_mod.EXIT_OK, err
    assert "README" in out
    assert "./scripts/setup" not in out, "a kit with no setup step must not be told to run one"


# --- the handoff itself -----------------------------------------------------

def test_setup_runs_in_the_kits_directory_when_asked(monkeypatch, tmp_path):
    _public_alpha(monkeypatch, _kit_tarball(setup=_SETUP_OK))
    seen = _records_setup(monkeypatch)

    rc, out, err = _capture(["alpha", str(tmp_path / "k"), "--setup"], monkeypatch)
    assert rc == create_mod.EXIT_OK, err
    assert seen["argv"] == [str(tmp_path / "k" / "scripts" / "setup")]
    assert seen["cwd"] == str(tmp_path / "k"), "setup must run inside the kit it configures"


def test_rig_flags_reach_the_kit_verbatim_and_in_order(monkeypatch, tmp_path):
    """The invariant, as a string comparison. Every one of these arguments is a fact
    about a robot, and the proof the verb did not learn any of them is that it
    passed them through without changing a character or reordering a pair."""
    _public_alpha(monkeypatch, _kit_tarball(setup=_SETUP_OK))
    seen = _records_setup(monkeypatch)

    rig = ["--leader-ip", "10.0.0.2", "--follower-ip", "10.0.0.3",
           "--camera", "right-wrist=1234", "--non-interactive"]
    rc, out, err = _capture(["alpha", str(tmp_path / "k"), *rig], monkeypatch)
    assert rc == create_mod.EXIT_OK, err
    assert seen["argv"][1:] == rig, f"arguments were not forwarded intact: {seen['argv']!r}"


def test_giving_the_kit_arguments_is_asking_for_it_to_run(monkeypatch, tmp_path):
    """Accepting a flag and then dropping it because --setup was missing is the
    silent failure Rule 10 exists to forbid."""
    _public_alpha(monkeypatch, _kit_tarball(setup=_SETUP_OK))
    seen = _records_setup(monkeypatch)

    rc, out, err = _capture(["alpha", str(tmp_path / "k"), "--leader-ip", "10.0.0.2"], monkeypatch)
    assert rc == create_mod.EXIT_OK, err
    assert seen["argv"][1:] == ["--leader-ip", "10.0.0.2"]


def test_a_double_dash_hands_over_an_argument_this_verb_also_owns(monkeypatch, tmp_path):
    """Both kits' setup steps take --json, and so does this verb. Without an escape
    hatch a kit could never be asked for its own machine-readable report."""
    _public_alpha(monkeypatch, _kit_tarball(setup=_SETUP_OK))
    seen = _records_setup(monkeypatch)

    rc, out, err = _capture(
        ["alpha", str(tmp_path / "k"), "--setup", "--", "--json", "--non-interactive"],
        monkeypatch,
    )
    assert rc == create_mod.EXIT_OK, err
    assert seen["argv"][1:] == ["--json", "--non-interactive"]


def test_the_verb_stops_reading_at_the_first_argument_it_does_not_know(monkeypatch, tmp_path):
    """One sentence of parsing rule, tested: --no-git after a forwarded flag is the
    kit's. A verb clever enough to pick it back out would have to know which
    forwarded flags take values, which is knowing about rigs."""
    _public_alpha(monkeypatch, _kit_tarball(setup=_SETUP_OK))
    seen = _records_setup(monkeypatch)

    rc, out, err = _capture(
        ["alpha", str(tmp_path / "k"), "--leader-ip", "10.0.0.2", "--no-git"], monkeypatch
    )
    assert rc == create_mod.EXIT_OK, err
    assert seen["argv"][1:] == ["--leader-ip", "10.0.0.2", "--no-git"]
    assert (tmp_path / "k" / ".git").exists(), "--no-git after the handoff was the kit's, not ours"


# --- flags only, no TTY, all the way through --------------------------------

def test_a_flags_only_run_completes_with_no_tty(monkeypatch, tmp_path):
    """The agent door, end to end and unstubbed: a real archive, a real unpack, a
    real process, and a file on disk that only the kit's setup step could have
    written. Nothing here reads a keyboard."""
    marker_script = (
        "#!/bin/sh\n"
        "printf '%s' \"$*\" > setup-was-here.txt\n"
    )
    _public_alpha(monkeypatch, _kit_tarball(setup=marker_script))

    dest = tmp_path / "k"
    rc, out, err = _capture(
        ["alpha", str(dest), "--leader-ip", "10.0.0.2", "--non-interactive"], monkeypatch
    )
    assert rc == create_mod.EXIT_OK, err
    written = (dest / "setup-was-here.txt").read_text()
    assert written == "--leader-ip 10.0.0.2 --non-interactive", written


def test_a_non_interactive_run_missing_an_input_names_the_flag_and_never_prompts(
    monkeypatch, tmp_path, capfd
):
    """The kit is what knows an input is required, so the kit is what names it —
    and this verb's job is to let that sentence through and exit non-zero rather
    than reporting a directory it made as a success.

    "Does not prompt" is checked twice: this run has no usable stdin and would
    hang forever if anything asked a question, and
    ``test_the_verb_asks_no_questions_of_its_own`` asserts the verb holds no
    prompt of its own to ask.
    """
    demanding = (
        "#!/bin/sh\n"
        "for a in \"$@\"; do [ \"$a\" = \"--leader-ip\" ] && exit 0; done\n"
        "echo 'setup: --leader-ip is required with --non-interactive' >&2\n"
        "exit 2\n"
    )
    _public_alpha(monkeypatch, _kit_tarball(setup=demanding))

    rc, out, err = _capture(["alpha", str(tmp_path / "k"), "--non-interactive"], monkeypatch)
    assert rc == create_mod.EXIT_SETUP_REFUSED, f"a refused setup step is not a success: {out!r}"
    assert rc != 0
    kit_output = capfd.readouterr().err
    assert "--leader-ip" in kit_output, f"the kit's own sentence was swallowed: {kit_output!r}"
    assert "exited 2" in err, f"newt must say what happened, not just pass it on: {err!r}"
    assert str(tmp_path / "k") in err, "the developer needs to be told the kit is already there"


def test_the_verb_asks_no_questions_of_its_own():
    """No prompt schema in ``newt``, asserted at the source. A prompt here would be
    a prompt with no flag equivalent and, worse, a prompt that had to know what it
    was asking about."""
    source = pathlib.Path(create_mod.__file__).read_text()
    assert "input(" not in source, "newt create must not prompt — the kit's setup step does that"
    for rig_flag in ("--leader-ip", "--follower-ip", "--camera-serial", "--rig-name"):
        assert rig_flag not in source, f"{rig_flag} is a fact about a robot; it cannot live here"


# --- the refusals this task adds --------------------------------------------

def test_asking_a_kit_with_no_setup_step_to_run_one_is_a_named_refusal(monkeypatch, tmp_path):
    _public_alpha(monkeypatch, _kit_tarball())

    rc, out, err = _capture(["alpha", str(tmp_path / "k"), "--setup"], monkeypatch)
    assert rc == create_mod.EXIT_NO_SETUP_STEP
    assert "declares no setup step" in err
    assert "intact" in err, "the kit did land; say so"


def test_arguments_nothing_can_consume_are_named_not_swallowed(monkeypatch, tmp_path):
    """The brief's own case: a flag the verb doesn't recognize is the template's
    business, and a flag *nothing* recognizes has to be said out loud."""
    _public_alpha(monkeypatch, _kit_tarball())

    rc, out, err = _capture(["alpha", str(tmp_path / "k"), "--leader-ip", "10.0.0.2"], monkeypatch)
    assert rc == create_mod.EXIT_NO_SETUP_STEP
    assert "--leader-ip 10.0.0.2" in err, f"dropped arguments must be quoted back: {err!r}"


def test_a_setup_step_that_cannot_be_started_is_not_a_setup_step_that_failed(
    monkeypatch, tmp_path
):
    """Two different problems: one is a kit that shipped a file without its exec bit,
    the other is a kit that ran and said no. They have different fixes."""
    _public_alpha(monkeypatch, _kit_tarball(setup=_SETUP_OK, mode=0o644))

    rc, out, err = _capture(["alpha", str(tmp_path / "k"), "--setup"], monkeypatch)
    assert rc == create_mod.EXIT_SETUP_UNRUNNABLE
    assert "not executable" in err
    assert "never launched" in err, "nothing ran, so say nothing ran"


def test_json_and_the_setup_step_refuse_to_share_stdout(monkeypatch, tmp_path):
    """Both write a report to stdout. Splicing them gives an agent neither, so this
    is refused before anything is fetched rather than producing unparseable output."""
    _public_alpha(monkeypatch, _kit_tarball(setup=_SETUP_OK))
    dest = tmp_path / "k"

    rc, out, err = _capture(["alpha", str(dest), "--json", "--setup"], monkeypatch)
    assert rc == create_mod.EXIT_USAGE
    assert not dest.exists(), "a usage refusal must not leave a directory behind"
    assert "two steps" in err, "name the way through, not just the wall"


def test_json_names_the_setup_command_it_did_not_run(monkeypatch, tmp_path):
    """The agent's second step, in the payload, so it never has to guess the path."""
    _public_alpha(monkeypatch, _kit_tarball(setup=_SETUP_OK))
    dest = tmp_path / "k"

    rc, out, err = _capture(["alpha", str(dest), "--json"], monkeypatch)
    assert rc == create_mod.EXIT_OK, err
    setup = json.loads(out)["setup"]
    assert setup == {
        "declared": "scripts/setup",
        "ran": False,
        "next": f"cd {dest} && ./scripts/setup",
    }


def test_json_says_plainly_when_a_kit_declares_no_setup_step(monkeypatch, tmp_path):
    _public_alpha(monkeypatch, _kit_tarball())

    rc, out, err = _capture(["alpha", str(tmp_path / "k"), "--json"], monkeypatch)
    assert rc == create_mod.EXIT_OK, err
    assert json.loads(out)["setup"] == {"declared": None, "ran": False, "next": None}


def test_kit_arguments_with_no_kit_named_are_refused_before_anything_is_written(
    monkeypatch, tmp_path
):
    _public_alpha(monkeypatch, _kit_tarball(setup=_SETUP_OK))

    rc, out, err = _capture(["--leader-ip", "10.0.0.2"], monkeypatch)
    assert rc == create_mod.EXIT_USAGE
    assert "no kit was asked for" in err
    assert "--leader-ip 10.0.0.2" in err


def test_setup_with_no_kit_named_is_its_own_refusal(monkeypatch):
    rc, out, err = _capture(["--setup"], monkeypatch)
    assert rc == create_mod.EXIT_USAGE
    assert "no kit was named" in err


def test_a_third_bare_word_is_refused_rather_than_dropped(monkeypatch, tmp_path):
    """Forwarding starts at the first flag, so a bare word after the directory
    belongs to nothing at all. Silently ignoring it is how a developer discovers an
    hour later that their argument never arrived."""
    _public_alpha(monkeypatch, _kit_tarball(setup=_SETUP_OK))

    rc, out, err = _capture(["alpha", str(tmp_path / "k"), "extra"], monkeypatch)
    assert rc == create_mod.EXIT_USAGE
    assert "at most one directory" in err
    assert not (tmp_path / "k").exists()


def test_the_console_gets_to_say_why_in_its_own_words(monkeypatch, tmp_path):
    """The console knows things this side cannot — which of its credentials is
    missing, what GitHub told it — and it writes a sentence saying so. Repeating
    ``502 Bad Gateway`` back at a developer instead of relaying that sentence
    throws away the only part of the message that names the cause."""
    _with_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _raises(
        monkeypatch,
        _http_error(
            502,
            "Bad Gateway",
            code="upstream_unavailable",
            error="GitHub answered 500 for the 'beta-private' starter kit at its pinned commit.",
        ),
    )

    rc, out, err = _capture(["beta-private", str(tmp_path / "x")], monkeypatch)

    assert rc == create_mod.EXIT_CONSOLE_REFUSED
    assert "GitHub answered 500" in err, f"the console's own sentence was dropped: {err!r}"
    assert "upstream_unavailable" in err.splitlines()[0], (
        "two console-named causes must not open with the same line"
    )


def test_a_console_that_says_nothing_useful_still_gets_a_full_refusal(monkeypatch, tmp_path):
    """The relay is not a dependency. A console that answers an error with no body
    at all must still produce a message that names the cause and the next step."""
    _with_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _raises(monkeypatch, _http_error(500, "Internal Server Error"))

    rc, out, err = _capture(["beta-private", str(tmp_path / "y")], monkeypatch)

    assert rc == create_mod.EXIT_CONSOLE_REFUSED
    assert "500 Internal Server Error" in err
    assert "Fix:" in err


def test_a_registry_row_with_nowhere_to_fetch_from_is_not_a_failed_download(
    monkeypatch, tmp_path
):
    """A public row missing its repository and pinned commit is our registry being
    broken. Reporting it with the download failure's exit code tells a developer —
    and an agent reading the number — to retry something that has nowhere to retry
    into, and it is not what the help text says that code means."""
    _no_key(monkeypatch)
    _console_returns(
        monkeypatch, {"templates": [{"name": "alpha", "visibility": "public"}]}
    )

    rc, out, err = _capture(["alpha", str(tmp_path / "z")], monkeypatch)

    assert rc == create_mod.EXIT_REGISTRY_BROKEN
    assert rc != create_mod.EXIT_FETCH_FAILED
    assert not (tmp_path / "z").exists(), "nothing may be written for a broken row"
    assert "ours, not yours" in err
    assert "12" in _capture_usage(), "an exit code the help text does not document"


def _capture_usage():
    out = io.StringIO()
    stdout, sys.stdout = sys.stdout, out
    try:
        create_mod._usage()
    finally:
        sys.stdout = stdout
    return out.getvalue()


# --- Rule 12 over the whole surface -----------------------------------------
#
# The three tests below are deliberately checked against the *module*, not against
# the list of cases someone remembered to write. A refusal nobody exercises is a
# string nobody has read, and the way that happens is a new one being added a year
# from now next to a test file that still passes.


def _refusal_names():
    """Every refusal in the verb, read off the module rather than listed here."""
    return sorted(n for n in vars(create_mod) if n.startswith("_say_"))


def _watch_refusals(monkeypatch):
    """Record which refusal each run takes, without changing what it prints."""
    seen: list[str] = []
    for name in _refusal_names():
        original = getattr(create_mod, name)

        def _wrap(*a, _name=name, _original=original, **kw):
            seen.append(_name)
            return _original(*a, **kw)

        monkeypatch.setattr(create_mod, name, _wrap)
    return seen


def _documented_exit_codes():
    """The exit codes ``newt create --help`` promises, parsed out of the help text."""
    return {int(m.group(1)) for m in re.finditer(r"^  (\d+) ", _capture_usage(), re.M)}


def _run_every_refusal(monkeypatch, tmp_path):
    """Take every refusing path the verb has, once, and return what each printed.

    Ordered roughly as a developer would meet them: the name, the key, the
    console, the directory, the download, the archive, the kit's own setup step.
    """
    from urllib.error import URLError

    collected: list[tuple[str, int, str]] = []

    def _run(label, args):
        rc, _out, err = _capture(args, monkeypatch)
        collected.append((label, rc, err))

    # --- resolution: the name, and who is asking -----------------------------
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _serves(monkeypatch, _kit_tarball(setup=_SETUP_OK))
    _run("unknown name", ["nope", str(tmp_path / "a")])
    _run("private, no key", ["beta-private", str(tmp_path / "b")])
    _run("--json with a setup step", ["alpha", str(tmp_path / "c"), "--json", "--setup"])
    _run("a third bare word", ["alpha", str(tmp_path / "d"), "one", "two"])
    _run("kit arguments, no kit", ["--leader-ip", "10.0.0.2"])
    _run("--setup, no kit", ["--setup"])

    _console_returns(
        monkeypatch,
        {"templates": [{"name": "alpha", "visibility": "public"}]},
    )
    _run("public row with nowhere to fetch from", ["alpha", str(tmp_path / "m")])

    # --- the console: silent, answering an error, or refusing the key --------
    _console_down(monkeypatch)
    _run("console silent, private kit", ["trossen-widowx", str(tmp_path / "n")])
    _console_errors(monkeypatch, 500, "Internal Server Error")
    _run("console errored, private kit", ["trossen-widowx", str(tmp_path / "o")])
    _console_rejects_key(monkeypatch)
    _run("key refused while listing", ["alpha", str(tmp_path / "p")])

    # --- the directory, and the download ------------------------------------
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    occupied = tmp_path / "e"
    occupied.mkdir()
    (occupied / "x").write_text("x")
    _run("directory in the way", ["alpha", str(occupied)])

    _raises(monkeypatch, _http_error(404, "Not Found"))
    _run("github refused", ["alpha", str(tmp_path / "f")])
    _raises(monkeypatch, URLError("Connection reset by peer"))
    _run("github unreachable", ["alpha", str(tmp_path / "g")])

    _with_key(monkeypatch)
    _raises(monkeypatch, _http_error(503, "Service Unavailable", code="dispensing_unconfigured"))
    _run("console holds no credential", ["beta-private", str(tmp_path / "q")])
    _raises(
        monkeypatch,
        _http_error(502, "Bad Gateway", code="upstream_unavailable",
                    error="GitHub answered 500 for that kit at its pinned commit."),
    )
    _run("console could not get it", ["beta-private", str(tmp_path / "r")])
    _raises(monkeypatch, URLError("Broken pipe"))
    _run("console lost mid-download", ["beta-private", str(tmp_path / "s")])
    _raises(monkeypatch, _http_error(401, "Unauthorized", code="key_rejected"))
    _run("key refused while downloading", ["beta-private", str(tmp_path / "t")])

    _no_key(monkeypatch)
    _serves(monkeypatch, b"not-a-tarball")
    _run("not an archive", ["alpha", str(tmp_path / "h")])

    # --- the handoff ---------------------------------------------------------
    _serves(monkeypatch, _kit_tarball())
    _run("kit declares no setup step", ["alpha", str(tmp_path / "i"), "--setup"])
    _serves(monkeypatch, _kit_tarball(setup=_SETUP_OK, mode=0o644))
    _run("setup step will not start", ["alpha", str(tmp_path / "j"), "--setup"])
    _serves(monkeypatch, _kit_tarball(setup="#!/bin/sh\nexit 3\n"))
    _run("setup step exited non-zero", ["alpha", str(tmp_path / "l"), "--setup"])

    return collected


def test_every_refusal_across_the_whole_verb_shares_no_string(monkeypatch, tmp_path):
    """Rule 12 over the finished surface — resolution, acquisition, and the handoff.

    Twenty causes, collected from real runs, so that a future edit collapsing two
    of them fails here rather than in someone's terminal at 2am. Two of the runs
    are the *same* cause reached by different routes — a key refused while listing
    templates and a key refused while downloading one — and they are meant to share
    their message: one problem, one fix, one sentence.
    """
    collected = _run_every_refusal(monkeypatch, tmp_path)

    firsts = {}
    for label, _rc, err in collected:
        first = err.splitlines()[0]
        if label == "key refused while downloading":
            assert first == firsts["key refused while listing"], (
                "a refused key is one cause and must read the same either way"
            )
            continue
        assert first not in firsts.values(), f"{label} shares a message with another cause"
        firsts[label] = first


def test_every_refusal_says_what_to_do_next(monkeypatch, tmp_path):
    """The third part of Rule 12. Naming the cause and the owner is not enough —
    a developer reading any of these has to be told what to do with it."""
    for label, _rc, err in _run_every_refusal(monkeypatch, tmp_path):
        assert "Fix" in err, f"{label} names no next step: {err!r}"


def test_no_refusal_goes_unread_and_no_exit_code_is_undocumented(monkeypatch, tmp_path):
    """The surface checked against itself.

    Two ways this file could quietly stop covering the verb: someone adds a
    refusal and no test reaches it, or someone adds an exit code and only the
    help text knows about it. Both fail here.
    """
    seen = _watch_refusals(monkeypatch)
    collected = _run_every_refusal(monkeypatch, tmp_path)

    unreached = set(_refusal_names()) - set(seen)
    assert not unreached, f"refusals no test reaches: {sorted(unreached)}"

    used = {rc for _label, rc, _err in collected}
    documented = _documented_exit_codes()
    assert used - {create_mod.EXIT_OK} == documented - {create_mod.EXIT_OK}, (
        f"exit codes used {sorted(used)} vs documented {sorted(documented)}"
    )


def test_who_refused_decides_the_number_not_what_the_transport_said(monkeypatch, tmp_path):
    """The same HTTP shape from two different owners must not read as one cause.

    Both of these are a failed attempt to get a private kit's bytes, and both would
    once have exited 7 — whose documented meaning is that the archive could not be
    downloaded. That is true when GitHub refused it. When the console refused, no
    archive was ever requested, and an agent reading only the number would retry a
    download that never started. The string above the number says which; the number
    has to say it too.
    """
    _no_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _raises(monkeypatch, _http_error(404, "Not Found"))
    github_rc, _, _ = _capture(["alpha", str(tmp_path / "a")], monkeypatch)

    _with_key(monkeypatch)
    _console_returns(monkeypatch, _CONSOLE_PAYLOAD)
    _raises(monkeypatch, _http_error(503, "Service Unavailable", code="anything_new"))
    console_rc, _, console_err = _capture(["beta-private", str(tmp_path / "b")], monkeypatch)

    assert github_rc == create_mod.EXIT_FETCH_FAILED
    assert console_rc == create_mod.EXIT_CONSOLE_REFUSED
    assert github_rc != console_rc
    # And a console code this SDK has never heard of still arrives named, so the console
    # can grow a cause without an SDK release.
    assert "anything_new" in console_err


def test_the_handoff_added_no_robot_knowledge_to_the_verb(monkeypatch, tmp_path):
    """The fence, checked once more now that rig arguments pass through the verb:
    a second body is still a registry row, and the handoff code is the same code
    for a two-armed teleop rig as for a single-arm follower."""
    _no_key(monkeypatch)
    _console_returns(
        monkeypatch,
        {"templates": [{"name": "brand-new-arm", "visibility": "public",
                        "repo": "org/newt-starter-brand-new-arm", "ref": "b" * 40}]},
    )
    _serves(monkeypatch, _kit_tarball(root="newt-starter-brand-new-arm-bbb", setup=_SETUP_OK))
    seen = _records_setup(monkeypatch)

    rc, out, err = _capture(
        ["brand-new-arm", str(tmp_path / "new"), "--whatever-this-rig-wants", "7"], monkeypatch
    )
    assert rc == create_mod.EXIT_OK, err
    assert seen["argv"][1:] == ["--whatever-this-rig-wants", "7"]
