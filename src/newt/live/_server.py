"""The rig's own little web server: one page, three viewer files, one status route.

Deliberately a stdlib ``ThreadingHTTPServer`` and deliberately dumb. It serves
files that are already on disk and one JSON document a callback produces. It holds
no session state, cannot start or stop anything, and has no route that reaches
hardware — a page served from here can look, and that is all it can do.

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

PAGE_DIR = Path(__file__).resolve().parent / "page"

_PAGE_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
}


class PortUnavailable(RuntimeError):
    """A port this session needs is already held by something else."""


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


def _handler(assets: dict[str, Path], session_json: Callable[[], dict]):
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
            if path == "/":
                self._send(
                    200,
                    "text/html; charset=utf-8",
                    (PAGE_DIR / "index.html").read_bytes(),
                )
                return
            self._send(404, "text/plain; charset=utf-8", b"this session serves one page, at /")

    return Handler


def serve(
    port: int,
    assets: dict[str, Path],
    session_json: Callable[[], dict],
) -> ThreadingHTTPServer:
    """Bind every interface on ``port`` and serve until ``shutdown()``.

    Every interface, not loopback: the browser that matters most is the one on the
    rig itself (``http://localhost`` is a secure context, so cameras decode), and
    the next one is a laptop on the same network. Binding loopback only would serve
    the first and silently refuse the second.
    """
    server = ThreadingHTTPServer(("0.0.0.0", port), _handler(assets, session_json))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="newt-live-http", daemon=True).start()
    return server
