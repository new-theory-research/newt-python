"""The live view: the second sink, the declaration seam, and the robot moat.

These encode WHY the view is shaped the way it is, not merely that its methods
run:

- recording is a second sink on one read, so a take cannot dim the picture and a
  view cannot cost the episode a frame;
- what gets drawn is the kit's declaration and only the kit's declaration, so a
  rig newt has never heard of draws correctly and a rig that declares nothing
  draws nothing rather than a stand-in;
- an observer is never allowed to become a gate on capture.

Every skip here is per test, never per module. A module-level ``importorskip``
aborts collection at import, which reports *zero* tests from this file rather than
the ones that could have run — and the robot-name invariant at the bottom, which
needs no extra at all, would silently stop being checked on every bare install.
So: ``needs_extra`` marks what writes an episode, ``requires_view`` marks what
needs the Rerun SDK, and CI installs each extra on its own leg so both halves
execute somewhere.
"""
from __future__ import annotations

import importlib.util
import threading
import time

import pytest

from newt.recording import (
    SINGLE_ARM_DESCRIPTOR,
    JointDrive,
    JointState,
    Session,
    SimulatedSource,
    ViewDeclaration,
)

#: Marks only what actually needs the SDK. ``find_spec`` rather than an import:
#: asking whether rerun is installed must not cost the rest of the file the SDK's
#: import time, and must not fail collection if importing it raises.
requires_view = pytest.mark.skipif(
    importlib.util.find_spec("rerun") is None,
    reason='the view extra is not installed — pip install "newt[view]"',
)

#: Same shape as the rest of the suite: a test that ends an episode needs a writer,
#: and a writer needs mcap/protobuf.
_HAVE_EXTRA = (
    importlib.util.find_spec("mcap") is not None
    and importlib.util.find_spec("google.protobuf") is not None
)
needs_extra = pytest.mark.skipif(
    not _HAVE_EXTRA, reason="needs the [recording] extra (mcap/protobuf)"
)


class _Recorder:
    """An observer that only remembers. Stands in for a LiveView everywhere the
    question is about the Session's side of the seam."""

    def __init__(self) -> None:
        self.states: list[tuple[dict, int]] = []
        self.frames: list[tuple[dict, int]] = []
        self.episodes: list[tuple[str, str]] = []

    def on_state(self, channels, ts_ns):
        self.states.append((channels, ts_ns))

    def on_frames(self, frames, ts_ns):
        self.frames.append((frames, ts_ns))

    def on_episode(self, event, episode_id):
        self.episodes.append((event, episode_id))


def _session(tmp_path, **kwargs) -> Session:
    return Session(
        SimulatedSource(SINGLE_ARM_DESCRIPTOR),
        task="observer tests",
        output_dir=tmp_path / "episodes",
        state_hz=60,
        **kwargs,
    )


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# --- the view is live before, during and after a take -----------------------


def test_view_receives_state_with_no_episode_recording(tmp_path):
    """A preview that only exists while recording is not a preview.

    The whole reason the loop moved off the episode boundary: an operator points
    a camera and aims an arm BEFORE they press record, and a picture that appears
    only once the file is already being written is useless for the part where you
    decide whether to write one.
    """
    session = _session(tmp_path)
    observer = _Recorder()
    session.attach_observer(observer)
    try:
        assert _wait_for(lambda: len(observer.states) > 3), (
            "no state reached the observer with no episode in flight — the capture "
            "loop is still tied to the episode boundary"
        )
        assert not session.status().recording
    finally:
        session.close()


@needs_extra
def test_recording_adds_a_writer_and_does_not_interrupt_the_view(tmp_path):
    """One source, two sinks: starting an episode must not cost the view a beat.

    Failing this looks like a preview that stutters or blanks the moment somebody
    presses record — the exact thing the split exists to prevent — so the test
    watches the observer's own count across the boundary rather than trusting
    that the code path looks right.
    """
    session = _session(tmp_path)
    observer = _Recorder()
    session.attach_observer(observer)
    try:
        assert _wait_for(lambda: len(observer.states) > 3)
        before = len(observer.states)
        episode_id = session.start_episode()
        assert _wait_for(lambda: len(observer.states) > before + 5), (
            "the view stopped receiving state when an episode started"
        )
        during = len(observer.states)
        path = session.end_episode(keep=True)
        assert path is not None and path.is_dir()
        assert _wait_for(lambda: len(observer.states) > during + 5), (
            "the view stopped receiving state when the episode ended — the loop is "
            "being torn down with the writer"
        )
        assert ("started", episode_id) in observer.episodes
        assert ("stopped", episode_id) in observer.episodes
    finally:
        session.close()


@needs_extra
def test_the_episode_still_records_everything_it_used_to(tmp_path):
    """The second sink must not become a filter on the first.

    An observer that could drop, reorder or delay what reaches the writer would
    make the picture and the file two different recordings of one event.
    """
    session = _session(tmp_path)
    observer = _Recorder()
    session.attach_observer(observer)
    try:
        session.start_episode()
        time.sleep(0.4)
        session.end_episode(keep=True)
        state_count, dropped = session.last_episode_counts
        assert state_count > 5, f"the episode recorded only {state_count} state frames"
        assert dropped == 0
    finally:
        session.close()


# --- an observer is never a gate --------------------------------------------


@needs_extra
def test_an_observer_that_raises_is_dropped_and_named(tmp_path):
    """A view that dies must not take the episode with it, and must not go quiet.

    Both halves matter. Letting the exception through would mean a broken browser
    page can end a recording session; swallowing it would mean a view that
    silently stopped drawing and an operator staring at a frozen robot deciding
    the arm is stuck.
    """

    class Exploding:
        def on_state(self, channels, ts_ns):
            raise ValueError("this observer is broken on purpose")

    session = _session(tmp_path)
    session.attach_observer(Exploding())
    try:
        session.start_episode()
        assert _wait_for(lambda: bool(session.observer_failures()))
        time.sleep(0.2)
        session.end_episode(keep=True)
        state_count, _ = session.last_episode_counts
        assert state_count > 3, "a raising observer stopped the episode being written"
        failures = session.observer_failures()
        assert [name for name, _ in failures] == ["Exploding"]
        assert isinstance(failures[0][1], ValueError)
    finally:
        session.close()


def test_a_slow_observer_does_not_hold_the_writer_lock(tmp_path):
    """The second sink runs outside the lock the writer takes.

    If it did not, a view stalling on a slow frame encode would stall the episode
    — the file would thin out because a browser was busy, which is the opposite
    of the guarantee.
    """
    holding = threading.Event()

    class Slow:
        def on_state(self, channels, ts_ns):
            holding.set()
            time.sleep(0.5)

    session = _session(tmp_path)
    session.attach_observer(Slow())
    try:
        assert _wait_for(holding.is_set)
        # status() takes the same lock the writer does. If the observer were
        # called under it, this would block for the observer's whole sleep.
        started = time.monotonic()
        session.status()
        assert time.monotonic() - started < 0.2
    finally:
        session.close()


@needs_extra
def test_pushed_state_reaches_observers_too(tmp_path):
    """The composed path shares the seam rather than growing a second one.

    ``newt record --teleop`` pushes each tick instead of being polled, and a view
    that worked on one path and not the other would be two implementations of
    "what the session is doing" that can disagree.
    """
    session = _session(tmp_path, state_pushed=True)
    observer = _Recorder()
    session.attach_observer(observer)
    try:
        session.start_episode()
        session.feed_state({"sim-arm": JointState(positions=[0.1] * 7)}, ts_ns=1234)
        session.end_episode(keep=True)
        assert observer.states, "a pushed tick never reached the observer"
        channels, ts_ns = observer.states[-1]
        assert ts_ns == 1234, "the observer was handed a timestamp the tick did not carry"
        assert channels["sim-arm"].positions == [0.1] * 7
    finally:
        session.close()


# --- the declaration seam, and the robot moat -------------------------------


def test_a_session_with_no_declaration_reports_none(tmp_path):
    """A source that declares no robot must read as None, not as a default.

    This is the moat: the only alternative to None here would be newt supplying a
    body of its own, which is a robot name in newt's source and a drawing nobody
    can vouch for.
    """
    session = _session(tmp_path)
    try:
        assert session.view_declaration is None
    finally:
        session.close()


def test_the_declaration_passes_through_untouched(tmp_path):
    """What the kit declares is what the session hands over — nothing normalised,
    defaulted or filled in on the way."""
    declaration = ViewDeclaration(
        urdf_path="/somewhere/arm.urdf",
        entity_prefix="rig",
        drives=(JointDrive(urdf_joint="j0", channel="sim-arm", index=0),),
        joint_convention="the kit's own sentence",
    )
    source = SimulatedSource(SINGLE_ARM_DESCRIPTOR)
    source.view_declaration = declaration
    session = Session(source, task="t", output_dir=tmp_path / "e")
    try:
        assert session.view_declaration is declaration
    finally:
        session.close()


def test_a_joint_drive_carries_no_sign_scale_or_offset():
    """The declaration has no field for a correction, and that is deliberate.

    A sign or scale on this dataclass would be a place for somebody to type a
    number that makes a render look right, which is the fabricated input Rule 10
    exists for. The kit says in words what it can vouch for; it does not get a
    numeric field to say it with.
    """
    fields = set(JointDrive.__dataclass_fields__)
    assert fields == {"urdf_joint", "channel", "index"}, (
        f"JointDrive grew fields: {fields}. If one of them is a correction factor, "
        "the derivation belongs at a bench, not in a default."
    )


def test_the_convention_sentence_is_required():
    """A declaration cannot be built without the kit saying how far it vouches.

    Making this optional would let a kit ship a robot drawing with nothing said
    about whether its joints turn the right way, which is precisely the claim
    nobody may make silently.
    """
    with pytest.raises(TypeError):
        ViewDeclaration(  # type: ignore[call-arg]
            urdf_path="/somewhere/arm.urdf",
            entity_prefix="rig",
            drives=(),
        )


# --- newt carries no robot ---------------------------------------------------


def test_live_view_module_names_no_robot():
    """The live view is the newest surface that could smuggle a robot name in.

    It handles descriptions, joints and meshes, which is exactly the code that
    wants to hardcode "the arm we have". Checked here as well as in the
    repo-wide sweep so this surface owns its own invariant.
    """
    import pathlib

    import newt.live

    root = pathlib.Path(newt.live.__file__).parent
    banned = ("widowx", "trossen", "so100", "so-101", "so101", "aloha", "franka", "wxai")
    # The upstream repository this page is lifted from is called so100-hackathon,
    # and both its licences require the attribution to name it. That is the name
    # of a repository we credit, not a robot this code knows about — so the slug
    # is removed before the scan and a bare "so100" anywhere else still fails.
    attribution = (
        "mission-robotics-ai/so100-hackathon",
        "rerun-io/so100-hackathon",
        "src/so100_hackathon/web_assets.py",
    )
    offenders = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.suffix not in {".py", ".js", ".html", ".css"}:
            continue
        text = path.read_text(errors="ignore").lower()
        for slug in attribution:
            text = text.replace(slug, "")
        offenders += [f"{path.name}: {word}" for word in banned if word in text]
    assert not offenders, (
        f"newt/live names a specific robot: {offenders}. The description, the "
        "entity prefix and the joint mapping arrive at runtime from the kit's "
        "ViewDeclaration; nothing in newt may know which arm it is drawing."
    )


# --- the rerun half ----------------------------------------------------------


@pytest.fixture
def cached_viewer_assets(tmp_path, monkeypatch):
    """Put a complete-looking viewer cache where ``ensure_assets`` will find it.

    ``LiveView.start`` fetches the viewer's 46 MB payload before it reads the kit's
    declaration, so a test about the declaration would otherwise download a
    browser build from npm to find out the URDF is missing. Seeding the cache is
    what keeps that test about the refusal and off the network — the bytes are
    stubs because nothing in this test opens them.
    """
    from newt.live import _assets

    version = importlib.import_module("rerun").__version__
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    directory = _assets.cache_root() / version
    directory.mkdir(parents=True)
    for name in _assets.WEB_VIEWER_FILES:
        (directory / name).write_bytes(b"stub")
    return directory


@requires_view
def test_view_refuses_a_description_that_is_not_there(tmp_path, cached_viewer_assets):
    """The refusal names the kit, because the kit is whose statement was wrong.

    An operator who typed a correct command and got told to check their own
    spelling would go looking in the wrong place.
    """
    from newt.live import LiveView, LiveViewUnavailable

    view = LiveView(
        descriptor=SINGLE_ARM_DESCRIPTOR,
        task="t",
        declaration=ViewDeclaration(
            urdf_path=str(tmp_path / "absent.urdf"),
            entity_prefix="rig",
            drives=(JointDrive(urdf_joint="j0", channel="sim-arm", index=0),),
            joint_convention="none",
        ),
        port=9401,
        grpc_port=9402,
    )
    with pytest.raises(LiveViewUnavailable) as caught:
        view.start()
    message = str(caught.value)
    assert "absent.urdf" in message
    assert "kit" in message.lower()


def test_two_causes_never_share_a_string(tmp_path):
    """Rule 12, checked rather than asserted in prose.

    Every refusal this surface can raise must be distinguishable from every other
    one by its text alone, because the text is all a reader gets.
    """
    from newt.live import _assets, _server

    messages = []
    try:
        _assets.ensure_assets("0.0.0-not-a-version", allow_fetch=False)
    except _assets.ViewerAssetsUnavailable as exc:
        messages.append(str(exc))

    import socket

    held = socket.socket()
    # Every interface, because that is what the view binds and therefore what a
    # collision really looks like — a loopback-only holder is a different (and
    # weaker) test than the one this refusal exists for.
    held.bind(("0.0.0.0", 0))
    held.listen(1)
    try:
        _server.claim_port(held.getsockname()[1], "a port under test")
    except _server.PortUnavailable as exc:
        messages.append(str(exc))
    finally:
        held.close()

    assert len(messages) == 2
    assert messages[0] != messages[1]
    for message in messages:
        assert "Do now:" in message, f"a refusal with no next step: {message}"


# --------------------------------------------------------------------------- #
# The chrome fence — what the embedded pane is allowed to put on screen
# --------------------------------------------------------------------------- #

@requires_view
def test_the_layout_names_every_panel_it_refuses():
    """"Hackathon-lean, not Rerun's application" is a claim about four panels, so
    the blueprint states all four rather than relying on defaults.

    This was written after a screenshot taken to prove the claim showed the Rerun
    wordmark, a Share button, three panel toggles and a full transport bar. Every
    one of those came from a panel nobody had named. A default is not a decision,
    and the difference is only visible in a picture — which is exactly the kind of
    drift a test has to hold instead.
    """
    import rerun.blueprint as rrb

    from newt.live._view import LiveView

    view = LiveView(
        descriptor=SINGLE_ARM_DESCRIPTOR,
        task="t",
        camera_ids=["cam-a"],
        port=9411,
        grpc_port=9412,
    )
    blueprint = view._blueprint()

    # Each attribute exists only because the matching part was passed — rerun's
    # Blueprint sets them per-part — so `getattr(..., None)` is what catches a
    # panel that stopped being named at all, rather than one named differently.
    def state(attribute: str):
        panel = getattr(blueprint, attribute, None)
        assert panel is not None, f"{attribute} is left at rerun's default"
        return panel.state

    assert state("blueprint_panel") == rrb.PanelState.Hidden
    assert state("selection_panel") == rrb.PanelState.Hidden
    # The wordmark, the menu, Share, the bell and the toggles that put the two
    # panels above back on screen all live on this one.
    assert state("top_panel") == rrb.PanelState.Hidden
    # The transport, and the scrubber that offers to leave the present. The joint
    # plot's own axis is what a screenshot reads its time off instead.
    assert state("time_panel") == rrb.PanelState.Hidden
