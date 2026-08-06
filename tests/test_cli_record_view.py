"""`newt record --view` — the CLI surface of the live view.

`test_live_view.py` owns the library: the seam, the declaration, the moat. This
file owns the twenty lines of frontend between an operator's command and that
library, which is where the failures a person actually meets live:

- **the two ports are one number.** The help promises "the stream it reads takes
  the next port up", and an operator who moves the page off 9099 to dodge a
  collision has to be able to predict what else moved. Nothing else in the suite
  can fail if that arithmetic drifts.
- **`--json` means JSON.** The verb documents its agent mode as "line-delimited
  JSON events", and a caller that took that literally gets a parse error from the
  first bare `[view] http://…` line — before the stream it was waiting for even
  begins. This is a contract about the *whole* stream, so it is tested on the
  whole stream and not on the one function that used to break it.
- **a view that will not start is a different problem on each path.** Plain
  `record` failed with nothing energized; `--teleop` failed with two arms up. One
  string for both would leave an operator guessing whether there is a torqued rig
  in the next room.

None of this needs the ``view`` extra: what these drive is the frontend, so the
LiveView is a stand-in that remembers how it was built. The real one is exercised
in `test_live_view.py` behind ``requires_view``.
"""
from __future__ import annotations

import importlib
import io
import json
import sys

import pytest

import newt.live
from newt._cli.record import _open_view, _parse, cmd_record


# --------------------------------------------------------------------------- #
# Stand-ins
# --------------------------------------------------------------------------- #

class _FakeView:
    """A LiveView that starts, remembers its construction, and draws nothing."""

    instances: list[_FakeView] = []
    #: On the class, because "this machine cannot name itself on a network" is a
    #: property of the machine and a test sets it for the whole run, not per view.
    network_address = "192.168.1.50"

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        _FakeView.instances.append(self)

    def start(self) -> str:
        self.started = True
        return f"http://localhost:{self.kwargs['port']}/"

    def urls(self):
        port = self.kwargs["port"]
        local = f"http://localhost:{port}/"
        if not self.network_address:
            return local, None
        return local, f"http://{self.network_address}:{port}/"

    def close(self) -> None:
        self.closed = True


class _RefusingView(_FakeView):
    """A LiveView whose start refuses the way the real one refuses."""

    def start(self) -> str:
        raise newt.live.LiveViewUnavailable(
            "The kit declared its robot description at /nowhere/arm.urdf, and there "
            "is no file there.\n"
            "Theirs: the declaration comes from the source this session loaded.\n"
            "Do now: check the kit is fully installed, or record without the view."
        )


@pytest.fixture(autouse=True)
def _fresh_view_instances():
    _FakeView.instances = []
    yield
    _FakeView.instances = []


@pytest.fixture
def fake_view(monkeypatch):
    """Every LiveView the frontend builds is a stand-in that only remembers."""
    monkeypatch.setattr(newt.live, "LiveView", _FakeView)
    return _FakeView


@pytest.fixture
def refusing_view(monkeypatch):
    monkeypatch.setattr(newt.live, "LiveView", _RefusingView)
    return _RefusingView


class _Session:
    """The three things `_open_view` asks a Session for, and nothing else."""

    def __init__(self, *, declaration=None) -> None:
        from newt.recording import SINGLE_ARM_DESCRIPTOR

        self.descriptor = SINGLE_ARM_DESCRIPTOR
        self.view_declaration = declaration
        self.camera_ids = ["cam-a", "cam-b"]
        self.observers = []

    def describe(self) -> dict:
        return {"source_kind": "a stand-in"}

    def attach_observer(self, observer) -> None:
        self.observers.append(observer)


class _Tty(io.StringIO):
    """stdin that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Parsing — the flags exist, and the second port is the first plus one
# --------------------------------------------------------------------------- #

def test_neither_view_flag_is_on_by_default():
    """A rig that did not ask for a page must not start serving one.

    `--view` opens two ports and publishes what the cameras see. Defaulting that
    to on would make every `newt record` a broadcast nobody consented to.
    """
    opts = _parse(["--task", "t", "--simulate"])
    assert opts["view"] is False
    assert opts["view_port"] is None


def test_the_view_flags_parse():
    opts = _parse(["--task", "t", "--view", "--view-port", "9200"])
    assert opts["view"] is True
    assert opts["view_port"] == 9200


def test_a_view_port_that_is_not_a_number_is_refused_at_the_parse():
    """Refused before anything opens, and by the same converter every other
    numeric flag uses — a port that silently became None would fall back to the
    default and serve on a port the operator did not name."""
    with pytest.raises(ValueError):
        _parse(["--task", "t", "--view-port", "nine-thousand"])


def test_the_stream_port_is_the_page_port_plus_one(fake_view):
    """The help makes this promise, so something has to be able to break on it.

    An operator moves `--view-port` because 9099 is taken; the stream port moves
    with it silently. If that arithmetic ever drifts, the page loads and paints
    nothing, and the reason is in a browser console rather than in the terminal.
    """
    view = _open_view(_Session(), _parse(["--task", "t", "--view-port", "9200"]))

    assert view.kwargs["port"] == 9200
    assert view.kwargs["grpc_port"] == 9201


def test_no_view_port_takes_the_librarys_default(fake_view, capsys):
    from newt.live import DEFAULT_PORT

    view = _open_view(_Session(), _parse(["--task", "t", "--view"]))

    assert view.kwargs["port"] == DEFAULT_PORT
    assert view.kwargs["grpc_port"] == DEFAULT_PORT + 1


# --------------------------------------------------------------------------- #
# What a person is told
# --------------------------------------------------------------------------- #

def test_both_urls_are_printed_and_the_second_says_what_it_is_for(fake_view, capsys):
    """Two addresses, and the one that is not this desk is labelled as such.

    An unlabelled pair of URLs is a guess about which one to send to the person
    on the laptop beside you.
    """
    _open_view(_Session(), _parse(["--task", "t", "--view", "--view-port", "9300"]))

    out = capsys.readouterr().out
    assert "[view] http://localhost:9300/" in out
    assert "[view] http://192.168.1.50:9300/  (from another machine on this network)" in out


def test_a_machine_that_cannot_name_itself_prints_one_url(fake_view, capsys, monkeypatch):
    """One true line beats two lines where the second means "this desk" to the
    reader and nothing to anyone else."""
    monkeypatch.setattr(_FakeView, "network_address", "")

    _open_view(_Session(), _parse(["--task", "t", "--view"]))

    out = capsys.readouterr().out
    assert out.count("[view] http") == 1


def test_a_rig_that_declares_no_robot_is_told_why_the_body_is_missing(fake_view, capsys):
    """The absence has to be explained where it is noticed.

    A page with camera panes and an empty 3D view reads as broken. The line says
    it is the kit's declaration that is missing, not the view.
    """
    _open_view(_Session(declaration=None), _parse(["--task", "t", "--view"]))

    out = capsys.readouterr().out
    assert "no robot drawn" in out
    assert "declares no description" in out


def test_a_rig_that_declares_a_robot_is_not_told_anything_about_one(fake_view, capsys):
    from newt.recording import JointDrive, ViewDeclaration

    declaration = ViewDeclaration(
        urdf_path="/somewhere/arm.urdf",
        entity_prefix="rig",
        drives=(JointDrive(urdf_joint="j0", channel="sim-arm", index=0),),
        joint_convention="the kit's own sentence",
    )

    _open_view(_Session(declaration=declaration), _parse(["--task", "t", "--view"]))

    assert "no robot drawn" not in capsys.readouterr().out


def test_the_view_is_attached_to_the_session_it_was_built_from(fake_view):
    """A view that starts and is never attached is a page that draws nothing
    forever — which looks exactly like a rig that is not reporting."""
    session = _Session()

    view = _open_view(session, _parse(["--task", "t", "--view"]))

    assert view.started is True
    assert session.observers == [view]


# --------------------------------------------------------------------------- #
# What an agent is told — the contract is the whole stream
# --------------------------------------------------------------------------- #

def test_the_json_frontend_emits_the_view_as_a_record_not_as_prose(fake_view, capsys):
    """`--json` documents line-delimited JSON events, so this is an event.

    The fields are the same two facts the printed lines carry plus the one the
    prose line carried: whether a body is being drawn. An agent that has to
    regex `[view] …` out of a stream to find the URL was handed a document, not
    an interface.
    """
    _open_view(
        _Session(), _parse(["--task", "t", "--view", "--view-port", "9300"]), as_json=True
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event == {
        "event": "view",
        "url": "http://localhost:9300/",
        "network_url": "http://192.168.1.50:9300/",
        "robot_drawn": False,
        "note": (
            "no robot drawn — this rig's source declares no description. "
            "Cameras and joint traces are live."
        ),
    }


def test_the_json_view_event_drops_the_note_when_there_is_a_robot(fake_view, capsys):
    from newt.recording import JointDrive, ViewDeclaration

    declaration = ViewDeclaration(
        urdf_path="/somewhere/arm.urdf",
        entity_prefix="rig",
        drives=(JointDrive(urdf_joint="j0", channel="sim-arm", index=0),),
        joint_convention="the kit's own sentence",
    )

    _open_view(
        _Session(declaration=declaration),
        _parse(["--task", "t", "--view"]),
        as_json=True,
    )

    event = json.loads(capsys.readouterr().out.strip())
    assert event["robot_drawn"] is True
    assert event["note"] is None


def test_every_stdout_line_of_a_json_view_session_parses_as_json(
    fake_view, monkeypatch, tmp_path, capsys
):
    """The regression, end to end, on the stream an agent actually reads.

    Testing `_open_view` alone would not have caught this: the break was that it
    is called *before* `_run_json`, so its three bare lines arrived ahead of the
    first record and every strict NDJSON reader failed on line one. So the
    assertion is over the whole of stdout for a whole session, and the `view`
    event is asserted to be in it rather than merely to exist somewhere.
    """
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd": "close"}\n'))

    rc = cmd_record(
        [
            "--task", "pick up the cup",
            "--simulate",
            "--json",
            "--view",
            "--dest", str(tmp_path / "episodes"),
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    events = []
    for number, line in enumerate(out.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            pytest.fail(f"stdout line {number} is not JSON ({exc}): {line!r}")
    kinds = [event["event"] for event in events]
    assert "view" in kinds, f"the view never announced itself: {kinds}"
    assert "preflight" in kinds


# --------------------------------------------------------------------------- #
# A view that will not start — two paths, two states of the rig
# --------------------------------------------------------------------------- #

def test_a_refused_view_stops_plain_record_and_closes_the_session(
    refusing_view, monkeypatch, tmp_path, capsys
):
    """Nothing is energized here, so the message says only what went wrong.

    And the session goes with it: a Session that was built and abandoned keeps a
    capture thread running against the source, which on a real rig is a verb that
    exited holding a camera open.
    """
    built = []
    real_build = importlib.import_module("newt._cli.record")._build_session

    def _capture(opts):
        session, receipt = real_build(opts)
        built.append(session)
        return session, receipt

    monkeypatch.setattr("newt._cli.record._build_session", _capture)
    monkeypatch.setattr(sys, "stdin", _Tty())

    rc = cmd_record(
        [
            "--task", "pick up the cup",
            "--simulate",
            "--view",
            "--dest", str(tmp_path / "episodes"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "the live view did not start" in captured.err
    assert "/nowhere/arm.urdf" in captured.err
    assert "Do now:" in captured.err
    assert "The arms are up" not in captured.err, (
        "plain record built no rig — telling an operator to go put arms away sends "
        "them to a bench where nothing is energized"
    )
    assert len(built) == 1
    assert built[0].status().closed is True


_COMPOSED_RIG = """
from newt.recording import JointState, StateDescriptor
from newt.teleop import drives_and_records


class _Part:
    name = "arm"

    def disable(self):
        pass


class _Pair:
    drives_and_records = True
    descriptor = StateDescriptor(
        arms=[{"id": "pair"}],
        channels=["pair/follower"],
        joint_names=["waist"],
        state_fields=["positions"],
    )

    def describe(self):
        return "a stand-in pair"

    def moving_parts(self):
        return [_Part()]

    def read_action(self):
        raise KeyboardInterrupt

    def send_action(self, action):
        pass

    def read_state(self):
        return {"pair/follower": JointState(positions=[0.0])}

    def close(self):
        pass


@drives_and_records
def make_demo():
    return _Pair()
"""


def test_a_refused_view_on_the_composed_path_says_the_arms_are_up(
    refusing_view, monkeypatch, tmp_path, capsys
):
    """Same refusal, one sentence more, because the rig's state is different.

    `--teleop` builds the rig *before* it opens the view — that call is what
    connects and energizes — so a view that fails here fails with two arms
    holding position. An operator who reads the plain message and walks away
    leaves them that way.
    """
    from newt._cli.teleop import KillKey

    (tmp_path / "view_refusal_rig.py").write_text(_COMPOSED_RIG)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "view_refusal_rig", raising=False)
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(KillKey, "arm", lambda self: True)
    monkeypatch.setattr(KillKey, "restore", lambda self: None)

    rc = cmd_record(
        [
            "--task", "pick up the cup",
            "--teleop",
            "--source", "view_refusal_rig:make_demo",
            "--view",
            "--dest", str(tmp_path / "episodes"),
        ]
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "the live view did not start" in err
    assert "The arms are up" in err
    assert "newt rest" in err


def test_the_two_refusals_are_not_the_same_string(
    refusing_view, monkeypatch, tmp_path, capsys
):
    """Rule 12 across the pair: same cause, two rig states, two strings.

    Collapsing them would leave the only actionable difference — is there a
    torqued rig in the next room — out of the only thing the operator reads.
    """
    from newt._cli.teleop import KillKey

    (tmp_path / "view_refusal_rig_b.py").write_text(_COMPOSED_RIG)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "view_refusal_rig_b", raising=False)
    monkeypatch.setattr(KillKey, "arm", lambda self: True)
    monkeypatch.setattr(KillKey, "restore", lambda self: None)
    monkeypatch.setattr(sys, "stdin", _Tty())

    cmd_record(
        ["--task", "t", "--simulate", "--view", "--dest", str(tmp_path / "a")]
    )
    plain = capsys.readouterr().err

    cmd_record(
        [
            "--task", "t",
            "--teleop",
            "--source", "view_refusal_rig_b:make_demo",
            "--view",
            "--dest", str(tmp_path / "b"),
        ]
    )
    composed = capsys.readouterr().err

    assert plain and composed
    assert plain != composed
