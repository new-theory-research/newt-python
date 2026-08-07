"""Session control: the contract a collection page runs on, and its refusals.

These encode WHY the control surface is shaped the way it is:

- it is a translation of the Session's own verbs, not a second implementation of
  them, so what a page can do and what the keyboard can do stay one behaviour;
- a take carries what it was about (dataset, task, tags) into the episode file
  itself, because a sidecar is a thing a later reader has to know to look for;
- the hardware contract is not per-take, so nothing a page sends can re-open a
  camera underneath a live view;
- every refusal names a cause no other refusal names, and no failure of delivery
  is ever allowed to read as a success (Rule 10 / Rule 12);
- a look-only server refuses control by saying it serves no control, which is a
  different answer from "no such route".

The HTTP half runs a real ``ThreadingHTTPServer`` on a real port rather than
poking the handler directly: the routing, the status codes and the JSON bodies are
the contract the portal page consumes, and a test that calls ``SessionControl``
methods would pass while every one of those was wrong.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from newt.live._control import ControlRefused, PushQueue, SessionControl, Take
from newt.live._server import PageDirMissing, serve
from newt.recording import SINGLE_ARM_DESCRIPTOR, Session, SimulatedSource

_HAVE_EXTRA = (
    importlib.util.find_spec("mcap") is not None
    and importlib.util.find_spec("google.protobuf") is not None
)
needs_extra = pytest.mark.skipif(
    not _HAVE_EXTRA, reason="needs the [recording] extra (mcap/protobuf)"
)


def _session(tmp_path) -> Session:
    return Session(
        SimulatedSource(SINGLE_ARM_DESCRIPTOR),
        task="session default task",
        output_dir=tmp_path / "episodes",
        state_hz=60,
    )


def _control(tmp_path, **kwargs) -> SessionControl:
    session = _session(tmp_path)
    return SessionControl(session, dataset_root=tmp_path / "episodes", **kwargs)


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class _StubSession:
    """A Session-shaped object for the paths that must not need a rig or a writer.

    Deliberately not a mock: the point of most of these tests is that the control
    layer forwards rather than decides, and a stub that records its calls is how
    "forwards" is checkable at all.
    """

    def __init__(self, *, recording=False, drives=False):
        self.started: list[dict] = []
        self.ended: list[bool] = []
        self.tagged: list[list[str]] = []
        self._recording = recording
        self.camera_failure = None
        self.drives = drives
        self.drive_failure = None
        self.last_episode_counts = (7, 1)

    def status(self):
        from newt.recording import SessionStatus

        return SessionStatus(
            recording=self._recording,
            episode_id="abcd1234" if self._recording else None,
            state_count=7,
            dropped_state=1,
            kept=0,
            target=None,
            last_positions=None,
            closed=False,
            frame_counts={"wrist": 12},
            dropped_frames={"wrist": 2},
        )

    def describe(self):
        return {
            "source_kind": "StubSource",
            "state_hz": 30,
            "drives": self.drives,
            "task": "session default task",
            "cameras": [{"id": "wrist", "width": 640, "height": 480, "fps": 30}],
        }

    def observer_failures(self):
        return []

    def start_episode(self, *, task=None, dest=None, tags=None):
        self.started.append({"task": task, "dest": dest, "tags": tags})
        self._recording = True
        return "abcd1234"

    def retag_episode(self, tags):
        self.tagged.append(list(tags))

    def end_episode(self, keep):
        self.ended.append(keep)
        self._recording = False
        return (dest_root / "episode_abcd1234") if (dest_root := None) else None


# --- what a take may and may not change -------------------------------------


def test_a_take_carries_its_own_dataset_task_and_tags(tmp_path):
    """The three per-take facts reach the Session, and nothing else does.

    This is the seam the whole design rests on: a page changes what a take is
    about and where it lands, and cannot touch the source, the cameras or the
    state rate. If this test starts passing a fourth key through, a page has
    gained the ability to re-open hardware mid-session.
    """
    session = _StubSession()
    control = SessionControl(session, dataset_root=tmp_path)

    control.start_episode(dataset="cube-picks", task="pick up the cube", tags=["clean"])

    assert session.started == [
        {"task": "pick up the cube", "dest": tmp_path / "cube-picks", "tags": ["clean"]}
    ]


@needs_extra
def test_the_tag_reaches_the_episode_file_not_a_sidecar(tmp_path):
    """A tag chosen at the stop is written into episode.json, where the episode is.

    Tagging happens after watching the take, so the value that matters arrives
    after the episode opened. It still has to end up in the one file that travels
    with the episode — a tag in a sidecar is a tag the next reader does not know
    to look for.
    """
    control = _control(tmp_path)
    control.start_episode(dataset="cube-picks", task="pick up the cube")
    time.sleep(0.2)
    take = control.stop_episode(keep=True, tags=["recovery", "jerky"])

    written = json.loads((tmp_path / "episodes" / "cube-picks" /
                          f"episode_{take['episode_id']}" / "episode.json").read_text())
    assert written["episode_config"]["tags"] == ["recovery", "jerky"]
    assert written["episode_config"]["task_name"] == "pick up the cube"


@needs_extra
def test_a_dataset_is_a_directory_that_exists_once_a_take_lands(tmp_path):
    """There is no create-dataset step, and that is the design.

    A page's "new dataset" is somebody typing a name. If this ever needs a
    registry, the picker and the recorder can disagree about what exists.
    """
    control = _control(tmp_path)
    assert control.session_status()["datasets"] == []

    control.start_episode(dataset="monday-cubes", task="pick up the cube")
    time.sleep(0.2)
    control.stop_episode(keep=True)

    assert control.session_status()["datasets"] == ["monday-cubes"]
    assert (tmp_path / "episodes" / "monday-cubes").is_dir()


@needs_extra
def test_two_datasets_in_one_session_land_in_their_own_directories(tmp_path):
    """Changing dataset must not need a new session — a new session re-opens the rig.

    The live view is drawn from reads the session takes; tearing it down to switch
    dataset would black out the picture for the length of a hardware re-open.
    """
    control = _control(tmp_path)
    for dataset in ("morning", "afternoon"):
        control.start_episode(dataset=dataset, task="pick up the cube")
        time.sleep(0.15)
        control.stop_episode(keep=True)

    assert sorted(control.session_status()["datasets"]) == ["afternoon", "morning"]
    for dataset in ("morning", "afternoon"):
        landed = list((tmp_path / "episodes" / dataset).glob("episode_*"))
        assert len(landed) == 1, f"{dataset} should hold exactly its own take"


@needs_extra
def test_a_discarded_take_leaves_nothing_and_says_so(tmp_path):
    """Discard means no directory, and a take whose state is not a push state.

    ``discarded`` rather than a failed push: nothing was attempted, so reporting a
    delivery outcome would be inventing one.
    """
    control = _control(tmp_path)
    control.start_episode(dataset="cube-picks", task="pick up the cube")
    time.sleep(0.15)
    take = control.stop_episode(keep=False)

    assert take["push"] == "discarded"
    assert take["path"] is None
    assert list((tmp_path / "episodes" / "cube-picks").glob("episode_*")) == []


# --- the refusals ------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"dataset": "", "task": "pick"}, "dataset-missing"),
        ({"dataset": "../escape", "task": "pick"}, "dataset-name-refused"),
        ({"dataset": "ok", "task": "   "}, "task-missing"),
    ],
)
def test_start_refuses_by_name_and_touches_nothing(tmp_path, kwargs, reason):
    """Every bad start has its own slug, and none of them reaches the Session.

    The second half is the one that matters: a refusal that has already called
    ``start_episode`` has opened a writer nobody will close.
    """
    session = _StubSession()
    control = SessionControl(session, dataset_root=tmp_path)

    with pytest.raises(ControlRefused) as caught:
        control.start_episode(**kwargs)

    assert caught.value.reason == reason
    assert session.started == []


def test_a_dataset_name_cannot_climb_out_of_the_session_directory(tmp_path):
    """The name becomes a directory, so the guard is about the filesystem.

    A ``..`` accepted here writes episodes outside the tree the session's sink
    verifies against, and the failure would surface at delivery time as a
    confusing sink error rather than here as a naming one.
    """
    session = _StubSession()
    control = SessionControl(session, dataset_root=tmp_path)
    for name in ("../up", "a/b", "/absolute", ".hidden", "with space"):
        with pytest.raises(ControlRefused) as caught:
            control.start_episode(dataset=name, task="pick")
        assert caught.value.reason == "dataset-name-refused", name


def test_two_causes_never_share_a_refusal_string(tmp_path):
    """Rule 12's hard half: distinct causes, distinct sentences AND distinct slugs."""
    session = _StubSession()
    control = SessionControl(session, dataset_root=tmp_path)

    seen: list[ControlRefused] = []
    for kwargs in ({"dataset": "", "task": "p"},
                   {"dataset": "../x", "task": "p"},
                   {"dataset": "ok", "task": ""}):
        with pytest.raises(ControlRefused) as caught:
            control.start_episode(**kwargs)
        seen.append(caught.value)
    recording = _StubSession(recording=True)
    with pytest.raises(ControlRefused) as caught:
        SessionControl(recording, dataset_root=tmp_path).start_episode(
            dataset="ok", task="p"
        )
    seen.append(caught.value)
    with pytest.raises(ControlRefused) as caught:
        control.stop_episode()
    seen.append(caught.value)

    assert len({exc.reason for exc in seen}) == len(seen)
    assert len({exc.message for exc in seen}) == len(seen)
    for exc in seen:
        # Three-part shape: what went wrong, whose it is, what to do next.
        assert exc.message.count("\n") >= 2, exc.message
        assert "Do now:" in exc.message


def test_starting_a_second_take_is_refused_and_the_first_is_untouched(tmp_path):
    session = _StubSession(recording=True)
    control = SessionControl(session, dataset_root=tmp_path)

    with pytest.raises(ControlRefused) as caught:
        control.start_episode(dataset="ok", task="pick")

    assert caught.value.reason == "already-recording"
    assert caught.value.status == 409
    assert session.started == []


def test_tags_of_the_wrong_shape_are_refused_rather_than_coerced(tmp_path):
    """A coerced tag is a value nobody typed, written into a file that outlives it."""
    from newt.live._control import string_list

    with pytest.raises(ControlRefused) as caught:
        string_list("clean", "tags")
    assert caught.value.reason == "tags-not-a-string-list"
    assert string_list(["clean", " jerky "], "tags") == ["clean", "jerky"]
    assert string_list(None, "tags") is None


# --- delivery is visible per take -------------------------------------------


class _Refusing:
    """A sink that fails a stated number of times, then succeeds (or never does)."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def deliver(self, episode_dir):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("the store said no")


def test_a_delivery_that_never_succeeds_ends_as_failed_and_names_the_cause(monkeypatch):
    """The state a page shows must never be a hopeful one.

    A queue that gave up quietly, or one that sat in ``pushing`` forever, both
    render as "probably fine" in a list — which is the exact gap Rule 10 is about.
    """
    monkeypatch.setattr("newt.live._control.PUSH_BACKOFF_S", (0.0, 0.0))
    take = Take(episode_id="x", dataset="d", task="t", tags=[], started_at=0.0)
    sink = _Refusing(failures=99)
    queue = PushQueue()
    queue.submit(take, "/nowhere", sink)

    assert _wait_for(lambda: take.push == "failed")
    assert sink.calls == 3, "the stated attempt count, not an unbounded retry"
    assert "the store said no" in take.push_detail
    assert "nothing about it was lost" in take.push_detail
    queue.close()


def test_a_delivery_that_succeeds_after_a_retry_lands_and_clears_its_detail(monkeypatch):
    monkeypatch.setattr("newt.live._control.PUSH_BACKOFF_S", (0.0, 0.0))
    take = Take(episode_id="x", dataset="d", task="t", tags=[], started_at=0.0)
    queue = PushQueue()
    queue.submit(take, "/nowhere", _Refusing(failures=1))

    assert _wait_for(lambda: take.push == "landed")
    assert take.push_detail is None
    assert take.push_attempts == 2
    queue.close()


@needs_extra
def test_a_session_with_no_destination_says_local_only_not_landed(tmp_path):
    """No destination is a fact about the session, and the take says which one.

    ``landed`` here would tell an operator their data is safe off the rig when it
    is on the rig only.
    """
    control = _control(tmp_path)
    control.start_episode(dataset="cube-picks", task="pick up the cube")
    time.sleep(0.15)
    take = control.stop_episode(keep=True)

    assert take["push"] == "local-only"
    assert "stays on this rig" in take["push_detail"]


@needs_extra
def test_the_store_is_asked_once_per_dataset_and_the_answer_is_kept(tmp_path):
    """A cloud sink counts what it delivered and writes a completeness marker from
    that count — one per episode would each claim a dataset of one."""
    built: list[tuple[str, str]] = []

    def sink_for(dataset, task):
        built.append((dataset, task))
        return _Refusing(failures=0)

    session = _session(tmp_path)
    control = SessionControl(
        session, dataset_root=tmp_path / "episodes", sink_for=sink_for
    )
    for _ in range(2):
        control.start_episode(dataset="cube-picks", task="pick up the cube")
        time.sleep(0.15)
        control.stop_episode(keep=True)

    assert _wait_for(lambda: all(t["push"] == "landed" for t in control.episodes()))
    assert built == [("cube-picks", "pick up the cube")]
    control.close()


# --- what the page reads -----------------------------------------------------


def test_session_status_reports_camera_health_it_actually_observed(tmp_path):
    """Camera health is reads already taken, never a probe this layer invented.

    The ``basis`` line ships with the numbers so a page cannot quietly present
    "ok between takes" as a liveness check somebody performed.
    """
    session = _StubSession()
    control = SessionControl(session, dataset_root=tmp_path)

    cameras = control.session_status()["cameras"]

    assert cameras["bridge"] == "ok"
    assert cameras["declared"][0]["id"] == "wrist"
    assert cameras["declared"][0]["frames"] == 12
    assert cameras["declared"][0]["frames_dropped"] == 2
    assert "not a probe" in cameras["basis"]


def test_a_dead_camera_bridge_is_reported_before_the_next_take(tmp_path):
    """``end_episode`` already refuses on this. The point of surfacing it here is
    the three minutes somebody would otherwise spend recording into it first."""
    session = _StubSession()
    session.camera_failure = ("stopped_answering", RuntimeError("usb reset"))
    control = SessionControl(session, dataset_root=tmp_path)

    cameras = control.session_status()["cameras"]

    assert cameras["bridge"] == "failed"
    assert "stopped_answering" in cameras["bridge_detail"]
    assert "usb reset" in cameras["bridge_detail"]


def test_a_driving_session_reports_that_it_drives_and_that_it_still_does(tmp_path):
    """A page showing a live pose looks the same whether the follower is tracking
    or standing still, so the payload says which — and says it for a session that
    drives nothing too, rather than leaving the row absent to be read as either."""
    watching = SessionControl(_StubSession(), dataset_root=tmp_path)
    driving = SessionControl(_StubSession(drives=True), dataset_root=tmp_path)

    look_only = watching.session_status()["drive"]
    assert look_only["declares"] is False
    assert look_only["state"] == "not-driven"
    assert look_only["detail"] is None

    moving = driving.session_status()["drive"]
    assert moving["declares"] is True
    assert moving["state"] == "ok"
    assert "not a probe" in moving["basis"]


def test_driving_that_died_between_takes_reaches_the_page(tmp_path):
    """The gap is where driving dies unwatched: nothing is recording, no readout
    is printing, and the next take would be three minutes of a rig that stopped
    moving before it started. ``end_episode`` cannot say it — no episode is open."""
    session = _StubSession(drives=True)
    session.drive_failure = ConnectionError("the follower stopped answering")
    control = SessionControl(session, dataset_root=tmp_path)

    drive = control.session_status()["drive"]

    assert drive["declares"] is True
    assert drive["state"] == "stopped"
    assert "ConnectionError" in drive["detail"]
    assert "the follower stopped answering" in drive["detail"]


@needs_extra
def test_episodes_lists_every_take_including_the_discarded_ones(tmp_path):
    """A take that was thrown away still happened, and a completed list that hides
    it makes the episode counter and the list disagree."""
    control = _control(tmp_path)
    control.start_episode(dataset="d", task="t")
    time.sleep(0.1)
    control.stop_episode(keep=True)
    control.start_episode(dataset="d", task="t")
    time.sleep(0.1)
    control.stop_episode(keep=False)

    takes = control.episodes()
    assert [t["push"] for t in takes] == ["local-only", "discarded"]
    status = control.session_status()
    assert status["episodes_started"] == 2
    assert status["episodes_kept"] == 1
    assert status["episodes_discarded"] == 1


# --- the HTTP contract -------------------------------------------------------


class _Serving:
    """A real server on a real port, torn down whatever the test does."""

    def __init__(self, **kwargs):
        self.port = _free_port()
        self.server = serve(
            self.port, {}, lambda: {"task": "t"}, **kwargs
        )

    def get(self, path):
        return _request("GET", f"http://127.0.0.1:{self.port}{path}")

    def post(self, path, body):
        return _request(
            "POST",
            f"http://127.0.0.1:{self.port}{path}",
            json.dumps(body).encode(),
        )

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _request(method, url, body=None):
    request = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@needs_extra
def test_the_four_routes_drive_a_real_session_over_http(tmp_path):
    """The whole contract, end to end, as the page consumes it.

    Not a handler unit test: the routing, the codes and the JSON shapes are what a
    front-end is written against, and every one of them can be wrong while the
    ``SessionControl`` methods underneath are right.
    """
    session = _session(tmp_path)
    control = SessionControl(session, dataset_root=tmp_path / "episodes")
    serving = _Serving(control=control)
    try:
        code, body = serving.get("/api/session")
        assert code == 200
        assert json.loads(body)["recording"] is False

        code, body = serving.post(
            "/api/episode/start",
            {"dataset": "cube-picks", "task": "pick up the cube"},
        )
        assert code == 200
        episode_id = json.loads(body)["episode_id"]

        code, body = serving.get("/api/session")
        assert json.loads(body)["recording"] is True
        assert json.loads(body)["dataset"] == "cube-picks"

        time.sleep(0.2)
        code, body = serving.post("/api/episode/stop", {"keep": True, "tags": ["clean"]})
        assert code == 200
        stopped = json.loads(body)
        assert stopped["episode_id"] == episode_id
        assert stopped["tags"] == ["clean"]
        assert stopped["state_count"] > 0

        code, body = serving.get("/api/episodes")
        listed = json.loads(body)["episodes"]
        assert [t["episode_id"] for t in listed] == [episode_id]
    finally:
        serving.close()
        session.close()


def test_a_look_only_server_refuses_control_by_saying_it_serves_none():
    """A bare 404 would read as a typo. The page needs to know the route exists and
    this session simply is not driving anything."""
    serving = _Serving()
    try:
        for method, path in (
            ("GET", "/api/session"),
            ("GET", "/api/episodes"),
            ("POST", "/api/episode/start"),
            ("POST", "/api/episode/stop"),
        ):
            code, body = (
                serving.get(path) if method == "GET" else serving.post(path, {})
            )
            assert code == 404, path
            payload = json.loads(body)
            assert payload["error"] == "control-not-served", path
            assert "Do now:" in payload["message"]
    finally:
        serving.close()


def test_a_malformed_body_is_refused_without_touching_the_session(tmp_path):
    session = _StubSession()
    serving = _Serving(control=SessionControl(session, dataset_root=tmp_path))
    try:
        url = f"http://127.0.0.1:{serving.port}/api/episode/start"
        code, body = _request("POST", url, b"{not json")
        assert code == 400
        assert json.loads(body)["error"] == "body-not-json"

        code, body = _request("POST", url, b'["a", "list"]')
        assert code == 400
        assert json.loads(body)["error"] == "body-not-an-object"
    finally:
        serving.close()
    assert session.started == []


# --- --page-dir --------------------------------------------------------------


def test_a_page_dir_takes_over_the_root_and_the_view_keeps_its_own_address(tmp_path):
    """The built-in page has to stay reachable, because the collection page embeds
    it. If /view stops answering, a page dir means choosing between your layout and
    your live view."""
    page = tmp_path / "dist"
    (page / "assets").mkdir(parents=True)
    (page / "index.html").write_text("<h1>the clay</h1>")
    (page / "assets" / "app.js").write_text("// built")

    serving = _Serving(page_dir=page)
    try:
        code, body = serving.get("/")
        assert code == 200
        assert b"the clay" in body

        code, body = serving.get("/assets/app.js")
        assert code == 200
        assert b"// built" in body

        code, body = serving.get("/view")
        assert code == 200
        assert b"newt" in body, "the built-in lean page still answers at /view"
    finally:
        serving.close()


def test_without_a_page_dir_the_built_in_page_still_owns_the_root():
    serving = _Serving()
    try:
        code, body = serving.get("/")
        assert code == 200
        assert b"live session" in body
    finally:
        serving.close()


def test_a_page_dir_cannot_be_climbed_out_of(tmp_path):
    """One rule, not four guards: resolve, then check the resolved path is inside.

    A rig serves this on every interface. A traversal here reads the operator's
    home directory to anyone on the network.
    """
    page = tmp_path / "dist"
    page.mkdir()
    (page / "index.html").write_text("<h1>fine</h1>")
    (tmp_path / "secret.txt").write_text("not yours")

    serving = _Serving(page_dir=page)
    try:
        for attempt in ("/../secret.txt", "/..%2Fsecret.txt", "/dist/../../secret.txt"):
            code, body = serving.get(attempt)
            assert b"not yours" not in body, attempt
    finally:
        serving.close()


def test_a_page_dir_that_is_not_there_refuses_before_a_port_is_bound(tmp_path):
    """Refuse at start, not at first request: a session that printed a URL and then
    404s every page has already cost somebody the walk to the rig."""
    with pytest.raises(PageDirMissing) as caught:
        serve(_free_port(), {}, dict, page_dir=tmp_path / "never-built")
    assert "Do now:" in str(caught.value)

    unbuilt = tmp_path / "src"
    unbuilt.mkdir()
    with pytest.raises(PageDirMissing) as caught:
        serve(_free_port(), {}, dict, page_dir=unbuilt)
    assert "no index.html" in str(caught.value)


def test_the_two_page_dir_refusals_do_not_share_a_string(tmp_path):
    missing = tmp_path / "gone"
    unbuilt = tmp_path / "src"
    unbuilt.mkdir()
    with pytest.raises(PageDirMissing) as first:
        serve(_free_port(), {}, dict, page_dir=missing)
    with pytest.raises(PageDirMissing) as second:
        serve(_free_port(), {}, dict, page_dir=unbuilt)
    assert str(first.value) != str(second.value)


# --- the moat ----------------------------------------------------------------


def test_the_control_module_names_no_robot():
    """The contract has to be consumable by a collect UI for a rig newt has never
    heard of. A joint name or a camera brand in the *code* is that promise breaking.

    Docstrings are stripped before the scan, and deliberately: the module's own
    prose says what it refuses to know ("never reads a URDF"), and a check that
    could not tell that sentence from an import of one would push the design note
    out of the file to keep a test green.
    """
    import ast
    from pathlib import Path as _Path

    import newt.live._control as control_module

    tree = ast.parse(_Path(control_module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef) and (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
    code = ast.unparse(tree).lower()

    for word in ("urdf", "servo", "gripper", "leader", "follower", "trossen",
                 "realsense", "dynamixel", "joint_names", "arm"):
        assert word not in code, f"{word!r} is a robot fact and does not live here"


def test_the_control_layer_never_reaches_past_the_session(tmp_path):
    """Everything it can do, the Session already offered. The stub has no other
    methods, so a control layer that grew a second path to the rig fails here."""
    session = _StubSession()
    control = SessionControl(session, dataset_root=tmp_path)
    control.session_status()
    control.episodes()
    control.start_episode(dataset="d", task="t")
    assert set(session.started[0]) == {"task", "dest", "tags"}


def test_the_push_worker_is_a_daemon_thread():
    """A rig operator hitting Ctrl+C must not be held by a queue that outlives the
    session — the episodes are already committed on disk either way."""
    queue = PushQueue()
    names = [t.name for t in threading.enumerate() if t.name == "newt-push"]
    assert names, "the worker should be running and named"
    assert all(t.daemon for t in threading.enumerate() if t.name == "newt-push")
    queue.close()
