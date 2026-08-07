"""The rig's own little web server: the viewer's files, a status route, and — when
a session hands it one — four session-control routes.

Deliberately a stdlib ``ThreadingHTTPServer`` and deliberately dumb. It serves
files that are already on disk, one JSON document a callback produces, and the four
operations in ``newt.live._control``. It holds no session state of its own: every
control route is a two-line translation of a ``SessionControl`` method, and a
server constructed without one answers 404 on all four rather than pretending.

**What a page served from here can do.** With no control attached: look, and that
is all — this is what ``newt record --view`` serves. With one attached: start a
take, stop a take, read the session's state and the list of takes. That is the same
authority the keyboard already has, reachable from a browser on the rig. It is
still true that no route here reaches hardware: control goes through the Session,
and the Session reaches the rig only through the source seam.

**Two page directories, one server.** The built-in lean page is what ``/`` serves
by default. ``--page-dir`` puts somebody else's built page there instead, and the
built-in one stays reachable at ``/view`` — which is how a collection UI embeds the
live session as one pane of its own layout without rebuilding the viewer.

The route table for the viewer's three files is the hackathon's, including the two
routes that point at one file; see ``_assets.VIEWER_ROUTES`` for why that is the
whole reason no bundler is needed.
"""
from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from newt.live._assets import VIEWER_ROUTES
from newt.live._control import ControlRefused, read_json_body, string_list

PAGE_DIR = Path(__file__).resolve().parent / "page"

#: Extensions a served page directory may hand out, and what to call them. An
#: allow-list rather than ``mimetypes.guess_type``: this server sits on a rig, and
#: the set of things a built front-end is made of is small and known.
_SERVED_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".wasm": "application/wasm",
}

_PAGE_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
}


class PortUnavailable(RuntimeError):
    """A port this session needs is already held by something else."""


class PageDirMissing(RuntimeError):
    """``--page-dir`` names something this server cannot serve a page out of."""


def claim_port(port: int, role: str) -> None:
    """Refuse now, loudly, rather than print a link to a server that never bound.

    Bind-test-then-serve leaves a microsecond window where another process can take
    the port; this closes it from "always" to "almost never" and, when it does lose
    the race, the server's own bind failure is what surfaces. What it buys is the
    common case: a viewer or an earlier session still holding the port gets named
    here, before a URL is printed, instead of the operator opening a dead page.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("0.0.0.0", port))
    except OSError as exc:
        raise PortUnavailable(
            f"Port {port} is already in use, and that is the port this session wants "
            f"for {role} ({type(exc).__name__}: {exc}).\n"
            f"Yours: another process holds it — most often an earlier newt session or "
            f"a Rerun viewer left running. Nothing about the rig or the episode is "
            f"wrong.\n"
            f"Do now: `lsof -nP -iTCP:{port} -sTCP:LISTEN` names the holder. Stop it, "
            f"or pass a different --view-port."
        ) from exc
    finally:
        probe.close()


def local_address() -> str | None:
    """This machine's address on the network it routes through, or None.

    Asks the kernel which source address it *would* use for an outbound route — a
    UDP socket that is connected and never written to, so nothing leaves the
    machine. ``gethostbyname(gethostname())`` is the other idiom and it lies: on
    some hosts it answers 127.0.0.1 whenever name resolution is having a day, and
    an address printed from that is unreachable from anywhere but this desk.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1, guaranteed unroutable
        address = probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
    return None if address.startswith(("127.", "::1")) else address


def _resolve_under(root: Path, request_path: str) -> Path | None:
    """The file ``request_path`` names inside ``root``, or None if it names none.

    Resolved and then checked against the resolved root, so ``..`` segments, an
    absolute path, a symlink pointing out of the tree and a URL-encoded climb all
    come back None by the same rule rather than by four separate guards.
    """
    from urllib.parse import unquote

    relative = unquote(request_path).lstrip("/")
    if not relative:
        relative = "index.html"
    try:
        candidate = (root / relative).resolve()
        candidate.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return candidate if candidate.is_file() else None


def _handler(
    assets: dict[str, Path],
    session_json: Callable[[], dict],
    control=None,
    page_dir: Path | None = None,
):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            # The operator's terminal is the session's readout. A request log
            # scrolling underneath it would bury the thing they are watching.
            pass

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict | list) -> None:
            self._send(code, "application/json", json.dumps(payload).encode())

        def _refused(self, exc: ControlRefused) -> None:
            self._send_json(exc.status, {"error": exc.reason, "message": exc.message})

        def _no_control(self) -> None:
            """The distinct refusal for a control route on a look-only server.

            Never a bare 404: a page that got this needs to know the route exists
            and this session simply is not driving anything, which is a different
            problem from a misspelled path.
            """
            self._send_json(
                404,
                {
                    "error": "control-not-served",
                    "message": (
                        "This session serves the live view only, so it has no "
                        "session-control routes.\n"
                        "Yours: the page is asking a look-only session to record. "
                        "Nothing was started and the rig was not touched.\n"
                        "Do now: start the session with a command that serves "
                        "control, or use the page this session does serve, at /."
                    ),
                },
            )

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in VIEWER_ROUTES:
                name, content_type = VIEWER_ROUTES[path]
                self._send(200, content_type, assets[name].read_bytes())
                return
            if path == "/session.json":
                self._send(
                    200,
                    "application/json",
                    json.dumps(session_json()).encode(),
                )
                return
            if path in ("/api/session", "/api/episodes"):
                if control is None:
                    self._no_control()
                    return
                if path == "/api/session":
                    self._send_json(200, control.session_status())
                else:
                    self._send_json(200, {"episodes": control.episodes()})
                return
            if path.startswith("/static/"):
                # Path().name flattens the request, so no traversal reaches out of
                # the page directory.
                candidate = PAGE_DIR / Path(path).name
                if candidate.is_file():
                    self._send(
                        200,
                        _PAGE_TYPES.get(candidate.suffix, "application/octet-stream"),
                        candidate.read_bytes(),
                    )
                    return
                self._send(404, "text/plain; charset=utf-8", b"no such page asset")
                return
            # The built-in page keeps its own address whether or not something else
            # took over /. A page dir that wants the live session as one pane of its
            # layout embeds this; nothing has to be rebuilt for it to work.
            if path == "/view" or (path == "/" and page_dir is None):
                self._send(
                    200,
                    "text/html; charset=utf-8",
                    (PAGE_DIR / "index.html").read_bytes(),
                )
                return
            if page_dir is not None:
                served = _resolve_under(page_dir, path)
                if served is not None:
                    self._send(
                        200,
                        _SERVED_TYPES.get(served.suffix, "application/octet-stream"),
                        served.read_bytes(),
                    )
                    return
                self._send(
                    404,
                    "text/plain; charset=utf-8",
                    (
                        f"No such file in the page directory this session was given "
                        f"({page_dir}). The live view is at /view.\n"
                    ).encode(),
                )
                return
            self._send(404, "text/plain; charset=utf-8", b"this session serves one page, at /")

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path not in ("/api/episode/start", "/api/episode/stop"):
                self._send(
                    404,
                    "text/plain; charset=utf-8",
                    b"this session takes POST at /api/episode/start and /api/episode/stop",
                )
                return
            if control is None:
                self._no_control()
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = read_json_body(self.rfile.read(length) if length else b"")
                if path == "/api/episode/start":
                    result = control.start_episode(
                        dataset=body.get("dataset", ""),
                        task=body.get("task", ""),
                        tags=string_list(body.get("tags"), "tags"),
                    )
                else:
                    result = control.stop_episode(
                        keep=bool(body.get("keep", True)),
                        tags=string_list(body.get("tags"), "tags"),
                    )
            except ControlRefused as exc:
                self._refused(exc)
                return
            self._send_json(200, result)

    return Handler


def serve(
    port: int,
    assets: dict[str, Path],
    session_json: Callable[[], dict],
    *,
    control=None,
    page_dir: str | Path | None = None,
) -> ThreadingHTTPServer:
    """Bind every interface on ``port`` and serve until ``shutdown()``.

    Every interface, not loopback: the browser that matters most is the one on the
    rig itself (``http://localhost`` is a secure context, so cameras decode), and
    the next one is a laptop on the same network. Binding loopback only would serve
    the first and silently refuse the second.

    ``control`` is a ``newt.live._control.SessionControl`` or None. None is the
    look-only server, and the four control routes then refuse by name rather than
    404ing as though they had been misspelled.

    ``page_dir`` is an already-built front-end to serve at ``/`` instead of the
    built-in lean page, which moves to ``/view``. Nothing is built here and nothing
    is validated beyond the directory existing: what a page directory contains is
    its author's business, and this server's whole job is to hand its bytes over.
    """
    resolved_page_dir = None
    if page_dir is not None:
        resolved_page_dir = Path(page_dir).expanduser().resolve()
        if not resolved_page_dir.is_dir():
            raise PageDirMissing(
                f"There is no directory at {resolved_page_dir}, and that is the page "
                f"directory this session was told to serve.\n"
                f"Yours: the path came from the command that started this session, "
                f"not from the rig. Nothing was started and no port was bound.\n"
                f"Do now: build the page first, or point --page-dir at the directory "
                f"the build wrote."
            )
        if not (resolved_page_dir / "index.html").is_file():
            raise PageDirMissing(
                f"{resolved_page_dir} exists but has no index.html, so a browser "
                f"opening / would get nothing.\n"
                f"Yours: this looks like a source directory rather than a built one "
                f"— most front-end builds write index.html into their output "
                f"directory. Nothing was started and no port was bound.\n"
                f"Do now: point --page-dir at the build output, e.g. that project's "
                f"dist/ rather than its src/."
            )
    server = ThreadingHTTPServer(
        ("0.0.0.0", port), _handler(assets, session_json, control, resolved_page_dir)
    )
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="newt-live-http", daemon=True).start()
    return server
