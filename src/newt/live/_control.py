"""Session control over HTTP: start a take, stop a take, read what the session did.

This is the seam a collection page runs on, and it is deliberately the smallest
thing that can be one. Four operations, all of them things a person at the
keyboard can already do — ``newt record``'s SPACE, its second SPACE, its readout,
its kept count. Nothing here is new authority; it is the same authority reachable
from a browser on the rig instead of from the terminal that started the session.

**Nothing in this module knows what a robot is.** It never opens a camera, never
names a joint, never reads a URDF, never touches a serial port. It holds a
``newt.recording.Session`` — which is itself robot-ignorant, the embodiment lives
behind the source seam — and calls four of its public methods. A rig with two arms
and a rig with none produce the same JSON here, differing only in what the session
they wrap reports about itself. That neutrality is the point: the contract below is
the one *any* collect UI consumes, not the one ours happens to need.

**What a take may change.** A dataset, a task and tags are per-take and travel in
the start request. The hardware contract — source, cameras, state rate — belongs to
the session and cannot be changed from here at all, because changing it means
re-opening hardware and a live view drawn from those reads would go dark. The three
things that do change touch nothing on the rig.

**Datasets are directories, and they exist when a take lands in one.** There is no
create-dataset call and there is deliberately no dataset registry: a dataset is a
subdirectory of the session's output directory, made by the writer the first time
something is written into it. A page's "new dataset" is a person typing a name.

**Saving is one motion.** A kept take goes to the session's sink without anybody
pressing a second button — and because a network upload can take longer than the
next take, delivery runs on its own thread and every take carries its state:
``pending``, ``pushing``, ``landed``, ``retrying``, ``failed``. There is no state
that means "we stopped looking". A take whose push failed for good says so and
names the cause; it never quietly reads as landed and never disappears from the
list (Rule 10).

The full set a take's ``push`` field can hold, and each means one thing only:

===============  ===========================================================
``pending``      queued for delivery; the worker has not reached it yet
``pushing``      a delivery attempt is in flight right now
``retrying``     an attempt failed and another is scheduled; detail says why
``landed``       the sink accepted it and did not raise
``failed``       every attempt failed, or no destination could be built
``discarded``    the person threw this take away; nothing was delivered
``local-only``   no destination was configured; the episode is on the rig
``not-committed``  the take never became an episode, so there is nothing to send
===============  ===========================================================
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

#: How long the push worker waits before re-trying a delivery that raised, per
#: attempt. Five attempts over about half a minute, and then it stops and says so.
#: Deliberately short and deliberately finite: a rig operator needs to know inside
#: one take whether the last one landed, and a queue that retries forever is a
#: queue that never reports a failure.
PUSH_BACKOFF_S = (1.0, 2.0, 4.0, 8.0)

#: What a dataset name may be. Not a taste rule — this string becomes a directory
#: name under the session's output directory, so anything that could climb out of
#: it or collide with the writer's own temp-directory prefix is refused by name.
DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ControlRefused(Exception):
    """A control request that cannot be honoured, carrying the sentence to show.

    ``status`` is the HTTP code the handler answers with; ``reason`` is a stable
    machine-readable slug so a page can branch without matching on prose, and no
    two causes in this module share one.
    """

    def __init__(self, status: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


@dataclass
class Take:
    """One attempt at recording, and everything the page knows about it.

    A take exists from the moment it starts, not from the moment it commits, which
    is why ``episode_id`` is present and ``path`` is None while it is in flight.
    """

    episode_id: str
    dataset: str
    task: str
    tags: list[str]
    started_at: float
    ended_at: float | None = None
    kept: bool | None = None
    path: str | None = None
    state_count: int = 0
    dropped_state: int = 0
    #: One of the states in this module's docstring table. Never blank, and never
    #: a state that means the delivery stopped being watched.
    push: str = "pending"
    push_detail: str | None = None
    push_attempts: int = 0

    def public(self) -> dict:
        return asdict(self)


class PushQueue:
    """Deliver committed episodes to their sink off the caller's thread, visibly.

    A sink here is the ``newt.recording.Sink`` protocol — one method, ``deliver``,
    which raises if it could not. This class adds exactly two things to it: the
    delivery does not block the person recording, and the outcome of every
    delivery is written down where the page can read it.

    It does not know what a sink delivers to. A local sink that verifies a path and
    a cloud sink that uploads a directory are the same object to this queue, and
    each queued take carries its own — because a destination's grain is the
    destination's business, and at least one real one (NT's store) is per-dataset.

    Order is FIFO across every sink, one delivery at a time. Not for throughput:
    a store that is create-only per dataset makes concurrent deliveries into one
    dataset a race over which of them creates it, and a rig uploading two videos at
    once on venue wifi finishes both later than it would have finished either.
    """

    def __init__(self, *, on_change: Callable[[], None] | None = None) -> None:
        self._on_change = on_change
        self._queue: list[tuple[Take, Path, object]] = []
        self._wake = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="newt-push", daemon=True)
        self._thread.start()

    def submit(self, take: Take, episode_dir: Path, sink) -> None:
        with self._wake:
            take.push = "pending"
            self._queue.append((take, episode_dir, sink))
            self._wake.notify()

    def close(self, *, drain_timeout: float = 0.0) -> None:
        """Stop the worker. With a drain timeout, wait that long for what is queued.

        Never silently drops: whatever is still queued when the timeout expires
        keeps whatever state it had, and that state is not ``landed``.
        """
        if drain_timeout > 0:
            deadline = time.monotonic() + drain_timeout
            while time.monotonic() < deadline:
                with self._wake:
                    if not self._queue:
                        break
                time.sleep(0.05)
        with self._wake:
            self._stop = True
            self._wake.notify_all()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._wake:
                while not self._queue and not self._stop:
                    self._wake.wait(timeout=0.5)
                if self._stop and not self._queue:
                    return
                take, episode_dir, sink = self._queue.pop(0)
            self._deliver(take, episode_dir, sink)

    def _deliver(self, take: Take, episode_dir: Path, sink) -> None:
        for attempt in range(len(PUSH_BACKOFF_S) + 1):
            take.push = "pushing"
            take.push_attempts = attempt + 1
            self._changed()
            try:
                sink.deliver(episode_dir)
            except Exception as exc:  # noqa: BLE001 — a sink is allowed to fail for
                # any reason its destination invents, and none of them is a reason to
                # take down the thread that is the only thing reporting on it. What
                # every failure has in common is that it must end up on the take.
                detail = f"{type(exc).__name__}: {exc}"
                if attempt < len(PUSH_BACKOFF_S):
                    take.push = "retrying"
                    take.push_detail = (
                        f"attempt {attempt + 1} failed ({detail}); retrying in "
                        f"{PUSH_BACKOFF_S[attempt]:g}s"
                    )
                    self._changed()
                    time.sleep(PUSH_BACKOFF_S[attempt])
                    continue
                take.push = "failed"
                take.push_detail = (
                    f"{detail}. The episode is committed on this rig at "
                    f"{episode_dir} and nothing about it was lost — only the "
                    f"delivery failed, after {attempt + 1} attempts."
                )
                self._changed()
                return
            else:
                take.push = "landed"
                take.push_detail = None
                self._changed()
                return

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()


class SessionControl:
    """The four operations a collection page needs, over one Session.

    Construct it around a live ``Session`` and call the four methods; the HTTP
    routes in ``newt.live._server`` are a thin translation of exactly these, so a
    caller embedding newt in something else gets the same surface without a server.

    ``dataset_root`` must be the session's own output directory. Every dataset is a
    directory under it, which is what keeps a ``LocalSink`` verification honest —
    the sink checks the committed episode is under the directory it was given, and
    a dataset outside that tree would fail that check at delivery time rather than
    at the moment somebody typed the name.

    ``sink_for`` is a callable ``(dataset, task) -> Sink | None`` and it is a
    factory rather than one sink because a destination gets to pick its own grain:
    NT's store is one namespace per dataset and refuses a name it has already seen,
    so a single sink for a session that records into two datasets would be wrong at
    the second one. It is called once per dataset name and the answer is kept. A
    caller with a destination that does not care about datasets returns the same
    sink every time; a caller with no destination passes nothing, and every take
    reports ``not-delivered`` with the reason rather than a hopeful ``landed``.
    """

    def __init__(
        self,
        session,
        *,
        dataset_root: str | Path,
        sink_for: Callable[[str, str], object] | None = None,
        dataset: str | None = None,
    ) -> None:
        self._session = session
        self._root = Path(dataset_root)
        self._takes: list[Take] = []
        self._lock = threading.Lock()
        self._dataset = dataset
        self._task = ""
        self._sink_for = sink_for
        self._sinks: dict[str, object] = {}
        self._pushes = PushQueue() if sink_for is not None else None

    # --- what the page reads -------------------------------------------------

    def session_status(self) -> dict:
        """State, counts, current dataset, camera health — one document, no writes."""
        status = self._session.status()
        described = self._session.describe()
        with self._lock:
            takes = list(self._takes)
            dataset = self._dataset
        kept = [t for t in takes if t.kept]
        return {
            "recording": status.recording,
            "episode_id": status.episode_id,
            "state_count": status.state_count,
            "dropped_state": status.dropped_state,
            "closed": status.closed,
            "dataset": dataset,
            "datasets": self._datasets(),
            "task": self._task or described.get("task") or "",
            "episodes_started": len(takes),
            "episodes_kept": len(kept),
            "episodes_discarded": len([t for t in takes if t.kept is False]),
            "source_kind": described.get("source_kind"),
            "state_hz": described.get("state_hz"),
            "destination": str(self._root.resolve()),
            "cameras": self._camera_health(status),
            "drive": self._drive_health(described),
            "observer_failures": [
                {"observer": name, "detail": f"{type(exc).__name__}: {exc}"}
                for name, exc in self._session.observer_failures()
            ],
        }

    def _drive_health(self, described: dict) -> dict:
        """Whether this session moves the rig, and whether it still does.

        The camera half's twin, and it exists for the sharper version of the same
        reason: a session that drives keeps driving between takes, so the gap is
        exactly where driving dies with nobody watching — no readout printing, no
        episode open, and a page that shows a live pose looking identical whether
        the follower is tracking or standing still. ``declares`` is what the
        source said at construction; ``state`` is what the capture loop has
        actually seen, never a probe this layer invented.
        """
        drives = described["drives"]
        failure = self._session.drive_failure
        if not drives:
            state, detail = "not-driven", None
        elif failure is None:
            state, detail = "ok", None
        else:
            state, detail = "stopped", f"{type(failure).__name__}: {failure}"
        return {
            "declares": drives,
            "state": state,
            "detail": detail,
            "basis": (
                "Ticks already taken by this session — not a probe. `ok` means no "
                "drive() has raised, including in the gaps between takes; `stopped` "
                "means one did and this session will not open another episode."
            ),
        }

    def _camera_health(self, status) -> dict:
        """What is actually known about the cameras, and nothing more.

        ``declared`` is what the source opened and the episode will describe.
        ``bridge`` is the only liveness statement available without opening a
        camera ourselves, which this layer will not do: the capture loop either has
        or has not seen the source's frame reader fail. A rig between takes whose
        bridge has never faltered reports ``ok`` and that is a claim about reads
        already taken, not a probe — so it is spelled out in ``basis`` rather than
        left for a page to over-read.
        """
        failure = getattr(self._session, "camera_failure", None)
        declared = [
            {
                "id": cam["id"],
                "width": cam["width"],
                "height": cam["height"],
                "fps": cam["fps"],
                "frames": status.frame_counts.get(cam["id"], 0),
                "frames_dropped": status.dropped_frames.get(cam["id"], 0),
            }
            for cam in self._session.describe()["cameras"]
        ]
        if failure is None:
            bridge, detail = "ok", None
        else:
            cause, exc = failure
            bridge = "failed"
            detail = f"{cause}: {type(exc).__name__}: {exc}"
        return {
            "declared": declared,
            "bridge": bridge,
            "bridge_detail": detail,
            "basis": (
                "Reads already taken by this session — not a probe. A camera this "
                "session has not yet read from is neither healthy nor broken here."
            ),
        }

    def _datasets(self) -> list[str]:
        """Every dataset directory under the session's output directory, sorted.

        Read off disk each time rather than remembered, so a dataset another
        session on this rig created is offered to the picker too. Directories the
        writer is still building (its temp prefix is a dot) are not datasets.
        """
        if not self._root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self._root.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )

    def episodes(self) -> list[dict]:
        """Every take this session started, newest last, each with its push state."""
        with self._lock:
            return [take.public() for take in self._takes]

    # --- what the page does --------------------------------------------------

    def start_episode(self, *, dataset: str, task: str, tags: list[str] | None = None) -> dict:
        """Begin a take in ``dataset``, about ``task``. Exactly what SPACE does.

        Both ``dataset`` and ``task`` are required and neither has a default,
        because both are written into the episode file and a value this layer
        invented would be indistinguishable there from one a person meant.
        """
        dataset = (dataset or "").strip()
        task = (task or "").strip()
        if not dataset:
            raise ControlRefused(
                400,
                "dataset-missing",
                "This take names no dataset, and a take has to land somewhere.\n"
                "Yours: the request carried no `dataset`. Nothing was started and "
                "the rig was not touched.\n"
                "Do now: send `dataset` — the name of the group these takes belong "
                "to. It does not have to exist yet.",
            )
        if not DATASET_NAME.match(dataset):
            raise ControlRefused(
                400,
                "dataset-name-refused",
                f"{dataset!r} cannot be a dataset name.\n"
                f"Yours: a dataset is a directory under {self._root}, so the name "
                f"has to be one path segment — letters, digits, dot, dash and "
                f"underscore, starting with a letter or digit, up to 64 characters. "
                f"Nothing was started.\n"
                f"Do now: send a name matching that shape.",
            )
        if not task:
            raise ControlRefused(
                400,
                "task-missing",
                "This take names no task, and the task is what the episode is of.\n"
                "Yours: the request carried no `task`. Nothing was started and the "
                "rig was not touched.\n"
                "Do now: send `task` — the language prompt describing what is about "
                "to be demonstrated. It is written into every episode as recorded.",
            )
        if self._session.status().recording:
            raise ControlRefused(
                409,
                "already-recording",
                "A take is already in flight on this session.\n"
                "Yours: something asked to start a second one. The take that was "
                "already running is untouched and still recording.\n"
                "Do now: stop the current take first, then start the next.",
            )

        try:
            episode_id = self._session.start_episode(
                task=task, dest=self._root / dataset, tags=tags or None
            )
        except RuntimeError as exc:
            raise ControlRefused(
                409,
                "session-refused-start",
                f"The session would not start a take: {exc}\n"
                f"Ours: this is the recording session refusing, not the request "
                f"being malformed — most often a session that has already been "
                f"closed.\n"
                f"Do now: restart the session command on the rig.",
            ) from exc

        take = Take(
            episode_id=episode_id,
            dataset=dataset,
            task=task,
            tags=list(tags or []),
            started_at=time.time(),
        )
        with self._lock:
            self._takes.append(take)
            self._dataset = dataset
            self._task = task
        return take.public()

    def stop_episode(self, *, keep: bool = True, tags: list[str] | None = None) -> dict:
        """Close the take in flight. Returns it, with its id and committed path.

        ``tags`` here is the second chance the record grammar needs: a person knows
        what a take was only after watching it, so a tag chosen at the stop replaces
        whatever the start carried. It is written into the episode file, not
        alongside it.
        """
        status = self._session.status()
        if not status.recording:
            raise ControlRefused(
                409,
                "not-recording",
                "There is no take in flight to stop.\n"
                "Yours: something asked to stop a take that is not running. The "
                "session is idle and unchanged.\n"
                "Do now: start a take before stopping one.",
            )
        with self._lock:
            take = self._takes[-1] if self._takes else None
        if tags is not None and take is not None:
            take.tags = list(tags)
            self._session.retag_episode(list(tags))

        try:
            path = self._session.end_episode(keep=keep)
        except Exception as exc:
            # Caught broadly on purpose: the session raises for reasons that are all
            # about the rig (a camera bridge that died mid-take is the designed one),
            # and each already carries its own three-part sentence. Nothing is
            # swallowed — every path below re-raises as a ControlRefused.
            # Wrapping it keeps the page's error shape uniform without editing the
            # message the session wrote.
            if take is not None:
                take.ended_at = time.time()
                take.kept = False
                take.push = "not-committed"
                take.push_detail = f"{type(exc).__name__}: {exc}"
            raise ControlRefused(
                500,
                "episode-refused-commit",
                f"The take was not committed: {exc}\n"
                f"Ours: the recording session refused to keep it, and nothing "
                f"partial was written. The reason above is the session's own.\n"
                f"Do now: fix what that names, then record the take again.",
            ) from exc

        state_count, dropped = self._session.last_episode_counts
        if take is None:
            # A session driven from two places at once — the keyboard started this
            # one and the page stopped it. Say so rather than reporting a take this
            # layer never saw as though it had.
            raise ControlRefused(
                409,
                "take-not-this-controls",
                "That take was stopped, and it was not one this page started.\n"
                "Ours: the session is being driven from somewhere else as well as "
                "from here, so this page cannot report on it. The episode itself is "
                "fine and was handled normally.\n"
                "Do now: drive the session from one place at a time.",
            )
        take.ended_at = time.time()
        take.kept = keep
        take.state_count = state_count
        take.dropped_state = dropped
        take.path = str(path) if path is not None else None

        if not keep:
            take.push = "discarded"
        elif self._pushes is None:
            take.push = "local-only"
            take.push_detail = (
                "This session was started without a delivery destination, so the "
                f"episode stays on this rig, committed at {path}."
            )
        else:
            try:
                sink = self._sink(take.dataset, take.task)
            except Exception as exc:  # noqa: BLE001 — building a destination can
                # fail for its own reasons (a missing key is the common one), and a
                # take that recorded fine must not read as a recording failure.
                take.push = "failed"
                take.push_detail = (
                    f"No destination could be built for dataset {take.dataset!r} "
                    f"({type(exc).__name__}: {exc}). The episode is committed on "
                    f"this rig at {path} and nothing about it was lost."
                )
            else:
                self._pushes.submit(take, path, sink)
        return take.public()

    def _sink(self, dataset: str, task: str):
        """This dataset's destination, built once and kept.

        Kept rather than rebuilt because a cloud sink counts what it has delivered
        and writes a completeness marker from that count — a fresh one per episode
        would each claim a dataset of one.
        """
        if dataset not in self._sinks:
            self._sinks[dataset] = self._sink_for(dataset, task)
        return self._sinks[dataset]

    def close(self, *, drain_timeout: float = 0.0) -> None:
        """Drain the push queue, then close every destination that asks to be.

        ``finalize`` is a sink's own word for "that was all of them" — NT's store
        writes its completeness marker there. It is called after the drain and only
        for sinks that have one, and a sink that refuses to finalize says so on
        stderr rather than silently: a dataset missing its marker reads as
        incomplete to everything downstream, and the operator should hear it here
        rather than discover it from a training run.
        """
        if self._pushes is not None:
            self._pushes.close(drain_timeout=drain_timeout)
        for dataset, sink in self._sinks.items():
            finalize = getattr(sink, "finalize", None)
            if finalize is None:
                continue
            try:
                finalize()
            except Exception as exc:  # noqa: BLE001 — teardown, and the whole point
                # of catching is to get the sentence out rather than to hide it.
                print(
                    f"[newt] dataset {dataset!r} was not finalized "
                    f"({type(exc).__name__}: {exc}). Its episodes are uploaded; the "
                    f"marker that says the set is complete is not. Anything reading "
                    f"the dataset will treat it as still arriving.",
                    file=sys.stderr,
                )


def read_json_body(raw: bytes) -> dict:
    """Parse a request body as a JSON object, refusing anything else by name."""
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlRefused(
            400,
            "body-not-json",
            f"The request body did not parse as JSON ({type(exc).__name__}: {exc}).\n"
            f"Yours: whatever sent this request built a malformed body. Nothing was "
            f"started and the rig was not touched.\n"
            f"Do now: send a JSON object, `Content-Type: application/json`.",
        ) from exc
    if not isinstance(payload, dict):
        raise ControlRefused(
            400,
            "body-not-an-object",
            f"The request body parsed as {type(payload).__name__}, and these routes "
            f"read a JSON object.\n"
            f"Yours: the body is valid JSON of the wrong shape. Nothing was started.\n"
            f"Do now: send an object, e.g. "
            f'{{"dataset": "cube-picks", "task": "pick up the cube"}}.',
        )
    return payload


def string_list(value, field_name: str) -> list[str] | None:
    """Read an optional list-of-strings field, refusing a wrong shape rather than
    coercing one. A tag list quietly turned into ``["['a', 'b']"]`` is a value
    nobody typed, written into an episode file that outlives the mistake."""
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item.strip() for item in value if item.strip()]
    raise ControlRefused(
        400,
        "tags-not-a-string-list",
        f"`{field_name}` has to be a list of strings, and this request sent "
        f"{type(value).__name__}.\n"
        f"Yours: the request body is the wrong shape. Nothing was started.\n"
        f'Do now: send `"{field_name}": ["one", "two"]`, or leave it out entirely.',
    )
