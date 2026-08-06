"""``newt.live.LiveView`` — the session, on a screen, while it is happening.

What this is, plainly: a second sink on the same reads the episode file gets. The
session polls the rig once; the writer takes that read and so does this, and this
one turns it into a robot moving, camera panes and joint traces in a page the rig
serves itself. Recording adds a writer. It does not add a read, and it does not
take anything away from the picture.

**The view is not the Rerun application.** The blueprint below collapses the
blueprint and selection panels and the served page hides the welcome screen, so
what a person gets is the 3D robot, their cameras, their joint traces and a
scrubber. Shipping Rerun's full app chrome instead would be a different product
than the one this exists to be.

**Nothing here knows what a robot is.** The description file, where it hangs in the
entity tree, which joints move and which reported number moves each one all arrive
at runtime in a ``ViewDeclaration`` the kit wrote. A rig that declares none gets no
robot drawn and a sentence saying why — never a stand-in body, and never a joint
matched by guesswork.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from newt.recording._seam import StateDescriptor, ViewDeclaration

#: The timeline every row lands on: the wall clock at the instant the rig produced
#: it. One clock, because a live view has exactly one thing to be honest about —
#: when this was true.
TIMELINE = "live"

#: Where the joint transforms go. Any single path works: a Transform3D from a URDF
#: joint carries its own parent and child frame names, and Rerun resolves the
#: skeleton from those rather than from where the row was logged.
TF_PATH = "tf"

DEFAULT_PORT = 9099
DEFAULT_GRPC_PORT = 9100

#: Bounded because a browser that is not attached must never become back-pressure
#: on the loop that is recording an episode. The proxy drops its oldest rows at
#: this limit; a viewer connecting late gets what is still in the buffer.
SERVER_MEMORY_LIMIT = "512MB"

#: How many camera reads a second reach the page. The episode file still gets
#: every one; this is the preview's rate, and it is a number this module picks
#: rather than one the hardware stated — so the page says it out loud.
DEFAULT_VIEW_FPS = 10.0

#: Keep every Nth pixel of a camera frame for the preview. 2 quarters the bytes.
#: Same reason as the rate above and the same honesty: the file keeps every pixel,
#: the pane is coarser, and the page says so rather than letting somebody conclude
#: their camera is soft.
DEFAULT_VIEW_SCALE = 2


class LiveViewUnavailable(RuntimeError):
    """The view cannot be built. Every instance names which of its inputs was missing."""


def _require_rerun():
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise LiveViewUnavailable(
            "The live view needs the Rerun SDK, which is not installed.\n"
            "Yours: recording works without it; only the on-screen view needs it.\n"
            'Do now: pip install "newt[view]" — then run the same command again.'
        ) from exc
    return rr


class LiveView:
    """One live view of one session. Attach it to a Session as an observer.

    Construct, call :meth:`start` (which returns the URL to open), attach, and call
    :meth:`close` when the session ends. It reads; it cannot write to the rig, stop
    a session, or start one.
    """

    def __init__(
        self,
        *,
        descriptor: StateDescriptor,
        task: str,
        source_kind: str = "",
        view_fps: float = DEFAULT_VIEW_FPS,
        view_scale: int = DEFAULT_VIEW_SCALE,
        camera_ids: list[str] | None = None,
        declaration: ViewDeclaration | None = None,
        port: int = DEFAULT_PORT,
        grpc_port: int = DEFAULT_GRPC_PORT,
        application_id: str = "newt-live-session",
    ) -> None:
        self._descriptor = descriptor
        self._task = task
        self._source_kind = source_kind
        self._view_fps = view_fps
        self._view_scale = max(1, int(view_scale))
        self._frame_period = 1.0 / view_fps if view_fps > 0 else 0.0
        self._last_frame_published = 0.0
        self._frames_skipped = 0
        self._camera_ids = list(camera_ids or [])
        self._declaration = declaration
        self._port = port
        self._grpc_port = grpc_port
        self._application_id = application_id

        self._stream = None
        self._tree = None
        self._drives: list = []
        self._mimic_joints: dict = {}
        self._mimics: list[tuple[str, str, float, float]] = []
        self._server = None
        self._assets = None
        self._url = ""
        self._grpc_url = ""
        self._lock = threading.Lock()
        self._recording_episode: str | None = None
        self._state_rows = 0
        self._frame_rows = 0
        self._started_at = 0.0

    # --- start ---------------------------------------------------------------

    def start(self) -> str:
        """Bind both ports, publish the robot, serve the page. Returns the URL.

        Order matters: both ports are claimed before anything is logged, because
        ``serve_grpc`` returns a healthy-looking URL for a proxy whose port was
        taken and reports the crash only on a background thread — a link printed
        without the check points at nothing.
        """
        rr = _require_rerun()
        from newt.live._assets import ensure_assets
        from newt.live._server import claim_port, local_address, serve

        claim_port(self._grpc_port, "the stream the browser subscribes to")
        claim_port(self._port, "the web server that hands out the page")

        self._assets = ensure_assets(rr.__version__)

        self._stream = rr.RecordingStream(self._application_id)
        self._grpc_url = self._stream.serve_grpc(
            grpc_port=self._grpc_port,
            server_memory_limit=SERVER_MEMORY_LIMIT,
            cors_allow_origin=self._allowed_origins(local_address()),
        )

        if self._declaration is not None:
            self._load_robot()
        self._log_provenance(rr)
        self._stream.send_blueprint(self._blueprint())

        self._server = serve(self._port, self._assets, self.session_json)
        self._started_at = time.monotonic()
        self._url = f"http://localhost:{self._port}/"
        return self._url

    def _allowed_origins(self, address: str | None) -> list[str]:
        """Which pages this rig lets read its stream.

        The page and the stream are two ports, so they are two origins, so every
        read the viewer makes is cross-origin. Left undeclared, the SDK permits a
        loopback origin and refuses a LAN one — the page loads, paints nothing, and
        the reason arrives only in the browser's console. Naming both here is what
        makes the same page work for the browser on the rig and the browser on a
        laptop beside it.
        """
        origins = [
            f"http://localhost:{self._port}",
            f"http://127.0.0.1:{self._port}",
        ]
        if address:
            origins.append(f"http://{address}:{self._port}")
        return origins

    def urls(self) -> tuple[str, str | None]:
        """(the URL to open on this machine, the URL another machine can use).

        The second is None when this machine could not name itself on a network —
        which is a real answer, and better than printing an address that means
        "this desk" to somebody sitting at another one.
        """
        from newt.live._server import local_address

        address = local_address()
        return self._url, (f"http://{address}:{self._port}/" if address else None)

    # --- the robot -----------------------------------------------------------

    def _load_robot(self) -> None:
        """Load the kit's description, check every declared joint against it, and
        log the geometry once.

        Every refusal here names the kit's declaration rather than the operator's
        command, because that is whose statement was wrong.
        """
        from rerun.urdf import UrdfTree

        declaration = self._declaration
        path = Path(declaration.urdf_path).expanduser()
        if not path.is_file():
            raise LiveViewUnavailable(
                f"The kit declared its robot description at {path}, and there is no "
                f"file there.\n"
                f"Theirs: the declaration comes from the source this session loaded, "
                f"not from anything you typed.\n"
                f"Do now: check the kit is fully installed (a description shipped as "
                f"package data is missing from an incomplete checkout), or record "
                f"without the view."
            )
        try:
            tree = UrdfTree.from_file_path(
                path,
                entity_path_prefix=declaration.entity_prefix,
                frame_prefix=f"{declaration.entity_prefix}/",
            )
        except Exception as exc:
            raise LiveViewUnavailable(
                f"The robot description at {path} would not parse "
                f"({type(exc).__name__}: {exc}).\n"
                f"Theirs: the file is the kit's, and it is either malformed or uses "
                f"URDF this SDK version does not read.\n"
                f"Do now: open the file at the line the error names, or record "
                f"without the view."
            ) from exc

        channels = set(self._descriptor.channels)
        joint_count = len(self._descriptor.joint_names)
        drives = []
        for drive in declaration.drives:
            joint = tree.get_joint_by_name(drive.urdf_joint)
            if joint is None:
                available = ", ".join(j.name for j in tree.joints())
                raise LiveViewUnavailable(
                    f"The kit declared a live view driving {drive.urdf_joint!r}, and "
                    f"{path} has no joint by that name.\n"
                    f"Theirs: the declaration and the description disagree — most "
                    f"often a description re-vendored from a newer upstream.\n"
                    f"Do now: reconcile the kit's declaration against the joints the "
                    f"file actually has: {available}."
                )
            if drive.channel not in channels:
                raise LiveViewUnavailable(
                    f"The kit declared {drive.urdf_joint!r} driven from channel "
                    f"{drive.channel!r}, and this source's stream has no such channel "
                    f"(it has: {', '.join(sorted(channels))}).\n"
                    f"Theirs: the view declaration and the state descriptor come from "
                    f"the same kit and do not agree.\n"
                    f"Do now: fix whichever of the two is stale in the kit."
                )
            if not 0 <= drive.index < joint_count:
                raise LiveViewUnavailable(
                    f"The kit declared {drive.urdf_joint!r} driven from index "
                    f"{drive.index} of channel {drive.channel!r}, and that channel "
                    f"reports {joint_count} value(s).\n"
                    f"Theirs: the index is out of the range the descriptor declares. "
                    f"Reading past the end would either crash mid-session or draw "
                    f"whatever happened to be next.\n"
                    f"Do now: correct the index in the kit's view declaration."
                )
            drives.append((drive, joint))
        self._drives = drives

        # Mimic joints: the description states the relation, and this SDK version
        # parses `<mimic>` without acting on it. Reading multiplier and offset off
        # the file is one line; assuming a mimic is a mirror is how a gripper opens
        # by the wrong amount and still looks plausible.
        driven = {drive.urdf_joint for drive, _ in drives}
        for joint in tree.joints():
            mimic = joint.mimic
            if mimic is not None and mimic.joint in driven:
                self._mimics.append(
                    (joint.name, mimic.joint, float(mimic.multiplier), float(mimic.offset))
                )
        self._mimic_joints = {name: tree.get_joint_by_name(name) for name, _, _, _ in self._mimics}

        tree.log_urdf_to_recording(recording=self._stream)
        self._tree = tree

    def _log_provenance(self, rr) -> None:
        """The sentence the kit wrote about its own joint convention, pinned into
        the layout so it survives into a screenshot with no terminal beside it."""
        if self._declaration is None:
            body = (
                "# No robot drawn\n\n"
                "The source driving this session declares no robot description, so "
                "there is no model to draw. Joint values and camera frames are real "
                "and are shown as they arrive.\n\n"
                "A body drawn from a description nobody declared would be a guess "
                "with a shape, which is worse than an empty pane."
            )
        else:
            body = (
                "# What this drawing is\n\n"
                f"Description: `{self._declaration.urdf_path}`\n\n"
                "**How far the kit vouches for it, in the kit's own words:**\n\n"
                f"{self._declaration.joint_convention}\n\n"
                "Joint values are drawn exactly as the arm reported them. Nothing "
                "here applies a sign, scale or offset that was not measured."
            )
        self._stream.log(
            "provenance", rr.TextDocument(body, media_type=rr.MediaType.MARKDOWN), static=True
        )

    # --- the layout ----------------------------------------------------------

    def _blueprint(self):
        """The lean layout: robot, cameras, joint traces, and the provenance note.

        ``collapse_panels=True`` is the single line that makes this a view rather
        than an application — it hides the blueprint and selection panels outright
        and swaps the full timeline for a simple scrubber. It is also the supported
        route: the option that would do the same from JavaScript is marked private
        in the viewer's own types.
        """
        import rerun.blueprint as rrb

        prefix = self._declaration.entity_prefix if self._declaration else None
        side: list = []
        for cam_id in self._camera_ids:
            side.append(rrb.Spatial2DView(name=cam_id, origin=f"cameras/{cam_id}"))
        side.append(
            rrb.TimeSeriesView(name="joints", origin="joints"),
        )

        if prefix:
            contents = [f"{prefix}/**", f"- {prefix}/**/collision_geometries/**"]
            if not self._declaration.visual_only:
                contents = [f"{prefix}/**"]
            main = rrb.Spatial3DView(name="robot", contents=contents)
        else:
            main = rrb.TextDocumentView(name="what this is", origin="provenance")

        if prefix:
            side.append(rrb.TextDocumentView(name="what this is", origin="provenance"))

        return rrb.Blueprint(
            rrb.Horizontal(main, rrb.Vertical(*side), column_shares=[3, 2]),
            collapse_panels=True,
        )

    # --- the observer hooks --------------------------------------------------

    def on_state(self, channels: dict, ts_ns: int) -> None:
        """One tick of joint state, from the same read the episode file took."""
        if self._stream is None:
            return
        self._stream.set_time(TIMELINE, timestamp=np.datetime64(ts_ns, "ns"))
        values: dict[tuple[str, int], float] = {}
        for name, state in channels.items():
            if state is None:
                continue  # a dropped read draws nothing; it does not repeat the last pose
            for index, position in enumerate(state.positions):
                values[(name, index)] = float(position)
            joint_names = self._descriptor.joint_names
            for index, position in enumerate(state.positions):
                if index < len(joint_names):
                    self._log_scalar(f"joints/{name}/{joint_names[index]}", float(position))

        driven_values: dict[str, float] = {}
        for drive, joint in self._drives:
            value = values.get((drive.channel, drive.index))
            if value is None:
                continue
            driven_values[drive.urdf_joint] = value
            # clamp=False: a clamped value is a number the arm never reported, drawn
            # as though it had — and it warns on every frame of an uncalibrated arm.
            self._stream.log(TF_PATH, joint.compute_transform(value, clamp=False))
        for name, source_joint, multiplier, offset in self._mimics:
            value = driven_values.get(source_joint)
            if value is None:
                continue
            joint = self._mimic_joints.get(name)
            if joint is None:
                continue
            self._stream.log(TF_PATH, joint.compute_transform(multiplier * value + offset, clamp=False))
        with self._lock:
            self._state_rows += 1

    def _log_scalar(self, path: str, value: float) -> None:
        import rerun as rr

        self._stream.log(path, rr.Scalars(value))

    def on_frames(self, frames: dict, ts_ns: int) -> None:
        """Publish a read of the cameras — at the view's rate, not the camera's.

        **The view is sampled and the episode is not, deliberately and out loud.**
        The file gets every frame the bridge delivers; this pane gets
        ``view_fps`` of them and the page says so. Publishing all of them means
        raw frames at the camera's full rate through a browser, which fills the
        stream buffer in seconds and starves a viewer that attaches late — a
        preview that fights the recording for room, which is the exact thing this
        split exists to prevent.

        The frames are raw and not JPEG, which costs the bandwidth the sampling
        buys back. ``rr.Image(...).compress()`` produces an ``EncodedImage``, and
        the embedded web viewer at 0.34.1 never decodes one: every camera pane
        renders a loading spinner forever, with no error anywhere — measured on
        the bench, 2026-08-06. Do not put the compress back without a screenshot
        showing a pane that resolves.
        """
        if self._stream is None:
            return
        import rerun as rr

        now = time.monotonic()
        if now - self._last_frame_published < self._frame_period:
            with self._lock:
                self._frames_skipped += 1
            return
        self._last_frame_published = now

        self._stream.set_time(TIMELINE, timestamp=np.datetime64(ts_ns, "ns"))
        for cam_id in self._camera_ids:
            frame = frames.get(cam_id)
            if frame is None:
                continue  # a dropped frame shows nothing new; the last one stays up
            # Two transforms, both named, neither inventing a pixel.
            #
            # The bridge delivers bgr24 — what the episode's encoder wants and the
            # opposite of what an image viewer wants — so the channels reverse.
            #
            # Then every Nth pixel is kept. Not a resize: no interpolation, no
            # blending, nothing on screen that no sensor produced.
            #
            # It is here for buffer depth, and that is the whole of the reason.
            # The frames go raw (see the EncodedImage note above), so three
            # 640x480 cameras at the rate above are ~27 MB/s, which fills
            # SERVER_MEMORY_LIMIT in about nineteen seconds — and the proxy then
            # drops its oldest rows, which is what a browser attaching late finds
            # missing. A quarter of the bytes is about seventy seconds of buffer.
            #
            # It is NOT here to fix the viewer's latency readout, which was the
            # first guess and was wrong: that number tracks how long the page has
            # been open, not any lag in the stream. Bench receipt, 2026-08-06 —
            # one session, two screenshots: settling 10 s read 14.7 s, settling
            # 45 s read 44.1 s, with both showing the same live scene.
            rgb = np.asarray(frame)[:: self._view_scale, :: self._view_scale, ::-1]
            self._stream.log(f"cameras/{cam_id}", rr.Image(np.ascontiguousarray(rgb)))
        with self._lock:
            self._frame_rows += 1

    def on_episode(self, event: str, episode_id: str) -> None:
        with self._lock:
            self._recording_episode = episode_id if event == "started" else None

    # --- what the page reads -------------------------------------------------

    def session_json(self) -> dict:
        with self._lock:
            episode = self._recording_episode
            state_rows = self._state_rows
            frame_rows = self._frame_rows
            skipped = self._frames_skipped
        _, shareable = self.urls()
        return {
            "stream": self._browser_stream_url(),
            "task": self._task,
            "recording": episode is not None,
            "episode_id": episode,
            "state_rows": state_rows,
            "frame_rows": frame_rows,
            "view_fps": self._view_fps,
            "view_scale": self._view_scale,
            "cameras": self._camera_ids,
            "robot": self._declaration.urdf_path if self._declaration else None,
            "frames_not_shown": skipped,
            "shareable_url": shareable,
            "note": self._note(),
        }

    def _browser_stream_url(self) -> str:
        """The stream address as the browser must ask for it.

        Built from the host the page was served on rather than from the address the
        SDK printed: the same session is opened from the rig's own browser and from
        a laptop beside it, and a URL naming 127.0.0.1 is only true for one of them.
        The page substitutes its own hostname into this at load.
        """
        return f"rerun+http://{{host}}:{self._grpc_port}/proxy"

    def _note(self) -> str:
        """The footer line, and it leads with what is producing the numbers.

        The source describes itself and that description goes out unedited. A
        session whose joints are a sine wave and whose cameras are real says so
        here, in the page, rather than only in the terminal that started it —
        somebody looking at a screenshot has no terminal.
        """
        parts = []
        if self._source_kind:
            parts.append(f"Source: {self._source_kind}.")
        if self._camera_ids:
            sampled = f"{self._view_fps:g} fps"
            if self._view_scale > 1:
                sampled += f", every {self._view_scale}nd pixel"
            parts.append(
                f"Cameras shown at {sampled} — the episode file records every frame "
                "at full size; this pane samples them."
            )
        if self._declaration is None:
            parts.append(
                "No robot description declared by this rig's source, so no model is drawn."
            )
        else:
            parts.append(
                f"Robot drawn from {self._declaration.urdf_path}, from values passed "
                "through unchanged — see the provenance pane for what the kit vouches for."
            )
        return " ".join(parts)

    # --- teardown ------------------------------------------------------------

    def close(self) -> None:
        """Stop serving. Idempotent, and safe to call from a kill path."""
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:  # noqa: BLE001, S110 — teardown must not raise over the
                # session's own teardown; a socket that will not close cleanly is not
                # a reason to leave arms energized behind it.
                pass
            self._server = None
        self._stream = None
