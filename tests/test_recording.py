"""Goldens for newt.recording — the Session, the validator, the layering invariant.

Each test leads with the plain-English thing a user (or an agent) experiences and
why it matters; the asserts encode the guarantee. The recording-format tests need
the `recording` extra (mcap/protobuf); they skip cleanly when it is absent so
core-only CI stays green. The layering and single-format guards are grep-level and
run everywhere.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _code_tokens(path: Path) -> set[str]:
    """Identifiers, attribute names, and string-literal values that are actual
    CODE in a module — comments and docstrings excluded. Lets the layering and
    single-format guards assert on real behavior, not prose that describes what
    the module deliberately does NOT do.
    """
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    docstrings: set[int] = set()

    # Collect docstring node ids so their string values don't count as code.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and node.body and isinstance(node.body[0], ast.Expr):
                docstrings.add(id(node.body[0].value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                names.add(node.value)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names.add(mod)
            for alias in node.names:
                names.add(alias.name)
    return names

_HAVE_EXTRA = (
    importlib.util.find_spec("mcap") is not None
    and importlib.util.find_spec("google.protobuf") is not None
)
needs_extra = pytest.mark.skipif(not _HAVE_EXTRA, reason="needs the [recording] extra (mcap/protobuf)")

_SRC = Path(__file__).resolve().parent.parent / "src" / "newt"


# ---------------------------------------------------------------------------
# Single-format law + layering invariant — grep-level, run everywhere
# ---------------------------------------------------------------------------


def test_single_format_only_v0_0_3_is_written():
    """The library writes exactly one episode format — there is no second write
    path and no format option.

    Inspect the recording package's CODE (not its prose) for any LeRobot/HDF5
    write path or a `format=` capture option. Their presence in code would mean
    the single-format law (Mattie, 2026-06-12) has been broken. Docstrings that
    describe what we deliberately do NOT do are excluded by design.
    """
    forbidden = {
        "lerobot", "to_lerobot", "hdf5", "h5py", "to_parquet",
        "format_plugin", "rlds", "to_rlds",
    }
    for p in (_SRC / "recording").glob("*.py"):
        tokens = _code_tokens(p)
        hits = forbidden & tokens
        assert hits == set(), (
            f"{p.name} has second-format code tokens {hits}. "
            "Single-format law: NT v0.0.3 only, no plugins, no converter in capture."
        )

    # The one canonical version string lives in the writer, set once.
    assert 'FORMAT_VERSION = "0.0.3"' in (_SRC / "recording" / "_writer.py").read_text()


def test_no_nt_platform_imports_anywhere_in_recording():
    """Recording never imports nt-platform / nt-runway — the format is built
    clean-room from the spec, never a dependency on internal NT repos.
    """
    forbidden = {"nt_platform", "nt_runway"}
    for p in (_SRC / "recording").glob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            if isinstance(node, ast.Import):
                mods.extend(a.name for a in node.names)
            for mod in mods:
                root = mod.split(".")[0]
                assert root not in forbidden, f"{p.name} imports {mod!r} — forbidden"


def test_cli_record_is_frontend_only_no_behavior():
    """`newt record` holds zero recording behavior — it is a keyboard/JSON skin.

    The layering invariant: episode/format/atomicity/timing logic lives only in
    Session. The CLI may translate input to Session calls and render, but it must
    not write MCAP, register channels, write episode.json, or rename temp dirs. We
    assert the format machinery is ABSENT from the CLI code (prose excluded) and
    that the CLI actually calls into the Session library.
    """
    record_tokens = _code_tokens(_SRC / "_cli" / "record.py")
    episodes_tokens = _code_tokens(_SRC / "_cli" / "episodes.py")

    behavior_markers = {
        "register_channel",
        "register_schema",
        "add_message",
        "SerializeToString",
        "robot_state_class",
        "EpisodeWriter",
        "file_descriptor_set",
        "_atomic_write_json",
    }
    leaked_record = behavior_markers & record_tokens
    assert leaked_record == set(), (
        f"newt record has behavior tokens {leaked_record} — that belongs in Session/writer, "
        "not the frontend (layering invariant)."
    )
    leaked_episodes = behavior_markers & episodes_tokens
    assert leaked_episodes == set(), (
        f"newt episodes has behavior tokens {leaked_episodes} — frontend must call the library."
    )

    # The CLI must actually drive the Session library (not reimplement it).
    record_src = (_SRC / "_cli" / "record.py").read_text()
    assert "from newt.recording import" in record_src
    assert "start_episode" in record_tokens and "end_episode" in record_tokens


# ---------------------------------------------------------------------------
# Session describe/preflight — descriptive, never blocks (run everywhere)
# ---------------------------------------------------------------------------


def test_preflight_describes_and_never_exits(tmp_path):
    """The reverse contract is a description, not a gate — preflight() returns the
    contract and never raises or exits, even on a non-writable destination.

    The library DESCRIBES; the frontend decides whether to refuse. A blocking
    preflight in the library would break that layering.
    """
    from newt.recording import Session, SimulatedSource, BIMANUAL_DESCRIPTOR

    session = Session(SimulatedSource(BIMANUAL_DESCRIPTOR), task="fold the towel", output_dir=tmp_path)
    report = session.preflight()  # must not raise

    assert report["format"].startswith("NT episode v0.0.3")
    assert report["task"] == "fold the towel"
    # Bimanual descriptor => four leader/follower channels, descriptor-driven.
    assert report["channels"] == [
        "robot_state/left/leader",
        "robot_state/left/follower",
        "robot_state/right/leader",
        "robot_state/right/follower",
    ]
    assert report["provenance"]["verified"] is False
    session.close()


# ---------------------------------------------------------------------------
# Record / keep / discard / validate (need the extra)
# ---------------------------------------------------------------------------


def _run_simulated_session(tmp_path, script, **kwargs):
    """Drive a Session through a list of ('keep'|'discard') verdicts with a short
    capture each. Returns the Session (closed) so callers can inspect kept count."""
    from newt.recording import Session, SimulatedSource, SINGLE_ARM_DESCRIPTOR

    session = Session(
        SimulatedSource(SINGLE_ARM_DESCRIPTOR, drop_every=kwargs.get("drop_every", 0)),
        task="pick up the cup",
        output_dir=tmp_path,
    )
    for verdict in script:
        session.start_episode()
        time.sleep(0.15)  # let the capture loop write a few frames
        session.end_episode(keep=(verdict == "keep"))
    session.close()
    return session


@needs_extra
def test_simulated_session_produces_valid_episodes(tmp_path):
    """A fresh Session in simulate produces real, valid NT episodes.

    Drive keep,keep — two episodes land on disk, each passes the ported validator
    on every invariant (episode.json present + parses, format_version 0.0.3,
    data.mcap readable, a robot_state channel with messages, monotonic timestamps).
    """
    from newt.recording import validate

    session = _run_simulated_session(tmp_path, ["keep", "keep"])
    assert session.status().kept == 2

    episodes = sorted(tmp_path.glob("episode_*"))
    assert len(episodes) == 2, f"expected 2 kept episode dirs, found {episodes}"
    for ep in episodes:
        result = validate(ep)
        assert result["valid"], f"episode {ep} failed validation: {result['checks']}"
        # provenance + task carried into episode.json
        meta = json.loads((ep / "episode.json").read_text())
        assert meta["episode_config"]["task_name"] == "pick up the cup"
        assert meta["provenance"]["verified"] is False
        assert meta["format_version"] == "0.0.3"


@needs_extra
def test_discard_leaves_no_directory(tmp_path):
    """Discard leaves nothing — a discarded episode never appears on disk.

    Drive keep,discard,keep: exactly two directories survive, and the validator
    finds no partial dir to trip over.
    """
    from newt.recording import validate

    session = _run_simulated_session(tmp_path, ["keep", "discard", "keep"])
    assert session.status().kept == 2

    episodes = sorted(tmp_path.glob("episode_*"))
    assert len(episodes) == 2, f"discard must leave no dir; found {episodes}"
    # No leftover temp dirs either (atomicity: temp dir removed whole on discard).
    leftovers = list(tmp_path.glob(".episode_*"))
    assert leftovers == [], f"discard left a temp dir behind: {leftovers}"
    for ep in episodes:
        assert validate(ep)["valid"]


@needs_extra
def test_dropped_frames_are_counted_and_surfaced(tmp_path):
    """Dropped reads are counted and surfaced — never swallowed.

    With an injected drop cadence, status() reports a non-zero dropped count and
    the kept episode is still valid (a drop is a soft skip, not a corruption).
    """
    from newt.recording import Session, SimulatedSource, SINGLE_ARM_DESCRIPTOR, validate

    session = Session(
        SimulatedSource(SINGLE_ARM_DESCRIPTOR, drop_every=3),
        task="stack the blocks",
        output_dir=tmp_path,
    )
    session.start_episode()
    time.sleep(0.3)
    st = session.status()
    assert st.dropped_state > 0, "injected drops must be counted in status()"
    report = session.dropped_report()
    assert report is not None and "dropped" in report
    path = session.end_episode(keep=True)
    session.close()
    assert validate(path)["valid"]


@needs_extra
def test_kill_mid_episode_leaves_no_directory(tmp_path):
    """A kill mid-episode torques off and leaves no partial directory.

    A real Ctrl+H is a SIGKILL-class event in spirit: the in-flight episode must
    vanish. We exercise Session.kill() directly (the behavior behind the key),
    then confirm no episode dir and no temp dir survive, and the source was
    torque-off'd via disable_all.
    """
    from newt.recording import Session, SimulatedSource, SINGLE_ARM_DESCRIPTOR

    source = SimulatedSource(SINGLE_ARM_DESCRIPTOR)
    session = Session(source, task="t", output_dir=tmp_path)
    session.start_episode()
    time.sleep(0.15)
    session.kill()

    assert sorted(tmp_path.glob("episode_*")) == [], "kill must leave no episode dir"
    assert list(tmp_path.glob(".episode_*")) == [], "kill must leave no temp dir"
    assert source.disabled is True, "kill must torque off via the source's disable_all"


@needs_extra
def test_sigkill_mid_episode_leaves_no_committed_episode(tmp_path):
    """Hard SIGKILL of a recording process leaves no committed episode on disk.

    The atomicity spine: episode.json is written last via rename(2), so a process
    killed -9 mid-capture can leave a temp dir but NEVER a directory the validator
    would accept. We spawn a child that records, SIGKILL it, then assert no
    `episode_*` dir carrying episode.json exists.
    """
    child = (
        "import time, sys\n"
        "sys.path.insert(0, %r)\n"
        "from newt.recording import Session, SimulatedSource, SINGLE_ARM_DESCRIPTOR\n"
        "s = Session(SimulatedSource(SINGLE_ARM_DESCRIPTOR), task='t', output_dir=%r)\n"
        "s.start_episode()\n"
        "print('recording', flush=True)\n"
        "time.sleep(30)\n"
    ) % (str(_SRC.parent), str(tmp_path))

    proc = subprocess.Popen(
        [sys.executable, "-c", child],
        stdout=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(_SRC.parent)},
        text=True,
    )
    # Wait until it's actually recording, then hard-kill.
    assert proc.stdout.readline().strip() == "recording"
    time.sleep(0.2)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)

    committed = [d for d in tmp_path.glob("episode_*") if (d / "episode.json").exists()]
    assert committed == [], (
        f"SIGKILL left a committed episode: {committed}. episode.json-last + rename "
        "must guarantee no killed capture is ever a valid episode."
    )


# ---------------------------------------------------------------------------
# --source — the live-hardware loading path (capture-003)
# ---------------------------------------------------------------------------


def _base_opts(tmp_path, **overrides):
    opts = {
        "task": "pick up the cup",
        "dest": str(tmp_path),
        "simulate": False,
        "source": None,
        "bimanual": False,
        "target": None,
        "hz": 30,
        "author": None,
        "license": None,
        "drop_every": 0,
        "json": False,
    }
    opts.update(overrides)
    return opts


def test_load_source_rejects_a_spec_with_no_colon():
    """A --source value that isn't MODULE:FACTORY shaped fails immediately,
    naming the spec and the expected shape — not a bare traceback."""
    from newt._cli.record import _load_source

    with pytest.raises(ValueError, match="MODULE:FACTORY"):
        _load_source("not-a-valid-spec")


def test_load_source_names_the_module_on_import_failure():
    """An unimportable module in --source names the module and the import
    error, not a bare ModuleNotFoundError."""
    from newt._cli.record import _load_source

    with pytest.raises(RuntimeError, match="no_such_module_xyz"):
        _load_source("no_such_module_xyz:make_source")


def test_load_source_names_the_missing_factory():
    """A real module with no such attribute names both, not an AttributeError
    with no context."""
    from newt._cli.record import _load_source

    with pytest.raises(RuntimeError, match="no_such_factory"):
        _load_source("newt.recording:no_such_factory")


def test_source_and_simulate_are_mutually_exclusive():
    """--source and --simulate together is a grammar collision, not a silent
    pick-one — refuse loudly instead of guessing which the developer meant."""
    from newt._cli.record import _build_session

    opts = _base_opts("/tmp/unused", simulate=True, source="tests.fixtures.fake_source:make_source")
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_session(opts)


@needs_extra
def test_simulate_path_unchanged_by_the_new_source_flag(tmp_path):
    """--simulate takes the exact branch it always did — adding --source must
    not perturb its output. Same source_kind, same descriptor, same channels,
    a validate()-passing episode, byte-identical to pre-brief behavior."""
    from newt._cli.record import _build_session
    from newt.recording import SINGLE_ARM_DESCRIPTOR, SimulatedSource, validate

    session = _build_session(_base_opts(tmp_path, simulate=True))
    try:
        report = session.preflight()
        assert report["source_kind"] == SimulatedSource(SINGLE_ARM_DESCRIPTOR).source_kind
        assert report["channels"] == ["robot_state/sim-arm"]
        session.start_episode()
        time.sleep(0.15)
        path = session.end_episode(keep=True)
    finally:
        session.close()
    assert validate(path)["valid"]


@needs_extra
def test_cli_source_loads_fake_hardware_and_validates(tmp_path):
    """A developer's RecordingSource, loaded via --source MODULE:FACTORY,
    drives `newt record` exactly as --simulate would — the CLI never knows
    it's a fake. This is the live-hardware door: one kept episode, valid, and
    the kill-switch (disable_all) fires on the fake source through the
    unchanged, source-agnostic kill path.
    """
    from newt.recording import validate

    commands = "\n".join(
        [
            json.dumps({"cmd": "start"}),
            json.dumps({"cmd": "stop", "keep": True}),
        ]
    ) + "\n"

    proc = subprocess.run(
        [
            sys.executable, "-m", "newt", "record",
            "--source", "tests.fixtures.fake_source:make_source",
            "--json",
            "--task", "wipe the table",
            "--dest", str(tmp_path),
            "--target", "1",
        ],
        input=commands,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_SRC.parent)},
        timeout=60,
    )
    assert proc.returncode == 0, f"--source record failed: {proc.stderr}"

    events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    kinds = [e["event"] for e in events]
    assert "preflight" in kinds
    assert "target_reached" in kinds

    episodes = sorted(tmp_path.glob("episode_*"))
    assert len(episodes) == 1, f"one keep -> one dir; found {episodes}"
    assert validate(episodes[0])["valid"]


def test_cli_source_raising_factory_produces_loud_runnable_error(tmp_path):
    """A --source factory that raises on construction produces a clear,
    runnable CLI error naming the spec and the failure — no silent fallback
    to simulate, and no episode is ever created (Rule 10).
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "newt", "record",
            "--source", "tests.fixtures.fake_source:make_raising_source",
            "--json",
            "--task", "t",
            "--dest", str(tmp_path),
        ],
        input="",
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_SRC.parent)},
        timeout=30,
    )
    assert proc.returncode == 1, f"expected a refusal exit code; got stdout={proc.stdout!r}"
    assert "make_raising_source" in proc.stderr
    assert "fake hardware initialization failure" in proc.stderr
    assert sorted(tmp_path.glob("episode_*")) == [], "a raising factory must never leave an episode dir"


def test_cli_source_kill_switch_disables_the_loaded_source(tmp_path, monkeypatch):
    """Ctrl+H's kill path (Session.kill() -> disable_all()) fires for a
    --source-loaded RecordingSource exactly as it does for SimulatedSource —
    no new mechanism, no type-based branching (recording.md's safety-parity
    requirement between simulate and live sources). A standalone module (not
    the `newt` package) proves `_load_source` really imports the developer's
    own module rather than something bundled.
    """
    from newt._cli.record import _load_source
    from newt.recording import Session

    fixture_dir = tmp_path / "fixture_mod"
    fixture_dir.mkdir()
    (fixture_dir / "kill_switch_fixture.py").write_text(
        "from newt.recording import SINGLE_ARM_DESCRIPTOR, SimulatedSource\n"
        "def make_source():\n"
        "    return SimulatedSource(SINGLE_ARM_DESCRIPTOR)\n"
    )
    monkeypatch.syspath_prepend(str(fixture_dir))

    output_dir = tmp_path / "episodes"
    source = _load_source("kill_switch_fixture:make_source")
    session = Session(source, task="t", output_dir=output_dir)
    session.start_episode()
    time.sleep(0.15)
    session.kill()

    assert source.disabled is True, "kill must torque off the --source-loaded source via disable_all"
    assert sorted(output_dir.glob("episode_*")) == [], "kill must leave no episode dir"


# ---------------------------------------------------------------------------
# --json agent driving — same Session, stdin commands -> events
# ---------------------------------------------------------------------------


@needs_extra
def test_json_mode_drives_the_same_session_to_valid_episodes(tmp_path):
    """An agent drives `newt record --json` over stdin and gets valid episodes.

    The JSON frontend is the SAME Session as the keyboard one — no second code
    path. We feed line-delimited commands (start/stop keep, start/stop discard,
    start/stop keep) and assert: two episodes land, each valid; the discarded one
    leaves nothing; every action emitted a JSON event line; target_reached fires.
    """
    from newt.recording import validate

    commands = "\n".join(
        [
            json.dumps({"cmd": "start"}),
            json.dumps({"cmd": "stop", "keep": True}),
            json.dumps({"cmd": "start"}),
            json.dumps({"cmd": "stop", "keep": False}),
            json.dumps({"cmd": "start"}),
            json.dumps({"cmd": "stop", "keep": True}),
        ]
    ) + "\n"

    proc = subprocess.run(
        [
            sys.executable, "-m", "newt", "record",
            "--simulate", "--json",
            "--task", "wipe the table",
            "--dest", str(tmp_path),
            "--target", "2",
        ],
        input=commands,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_SRC.parent)},
        timeout=60,
    )
    assert proc.returncode == 0, f"json record failed: {proc.stderr}"

    events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    kinds = [e["event"] for e in events]
    assert "preflight" in kinds
    assert kinds.count("started") == 3
    assert kinds.count("stopped") == 3
    assert "target_reached" in kinds

    episodes = sorted(tmp_path.glob("episode_*"))
    assert len(episodes) == 2, f"two keeps -> two dirs; found {episodes}"
    for ep in episodes:
        assert validate(ep)["valid"]


@needs_extra
def test_episodes_validate_cli_round_trips(tmp_path):
    """`newt episodes validate <dir> --json` passes on a freshly recorded episode
    and fails on a partial (no episode.json) one.

    The validator CLI is a frontend on the library check; it must agree with the
    library and emit a structured verdict an agent can read.
    """
    _run_simulated_session(tmp_path, ["keep"])
    ep = sorted(tmp_path.glob("episode_*"))[0]

    ok = subprocess.run(
        [sys.executable, "-m", "newt", "episodes", "validate", str(ep), "--json"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(_SRC.parent)},
    )
    assert ok.returncode == 0
    verdict = json.loads(ok.stdout)
    assert verdict["valid"] is True

    # A directory with no episode.json is partial -> invalid, exit 1.
    partial = tmp_path / "episode_partial"
    partial.mkdir()
    (partial / "data.mcap").write_bytes(b"")
    bad = subprocess.run(
        [sys.executable, "-m", "newt", "episodes", "validate", str(partial), "--json"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(_SRC.parent)},
    )
    assert bad.returncode == 1
    assert json.loads(bad.stdout)["valid"] is False
