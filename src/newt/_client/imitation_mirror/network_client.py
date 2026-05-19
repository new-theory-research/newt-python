from typing import Any, Optional

import httpx
import numpy as np
import websockets

from nt._client.imitation_mirror import pack, unpack


class NetworkClient:
    """Client for communicating with the remote policy server.

    Two transports, auto-selected from the URL scheme:
      - WebSocket: ``ws://.../ws/inference`` or ``wss://``. Used for
        non-Modal / direct deployments. Single frame per request.
      - HTTP POST: ``http(s)://.../infer``. Used for the Modal Tunnel
        backend; single ``POST /infer`` per call over a persistent
        ``httpx.AsyncClient``.
    """

    def __init__(self, url: str, *, http_mode: bool | None = None):
        self.url = url
        # Auto-detect transport from URL scheme unless the caller pins it.
        if http_mode is None:
            http_mode = url.startswith(("http://", "https://"))
        self.http_mode = http_mode
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    def _origin(self) -> str:
        """Return scheme://host[:port] with the original scheme preserved."""
        url = self.url
        scheme_end = url.find("://") + 3
        slash = url.find("/", scheme_end)
        return url if slash == -1 else url[:slash]

    def _http_origin(self) -> str:
        """Return scheme://host[:port] with ``ws``→``http`` / ``wss``→``https``."""
        origin = self._origin()
        for prefix_ws, prefix_http in (("wss://", "https://"), ("ws://", "http://")):
            if origin.startswith(prefix_ws):
                return prefix_http + origin[len(prefix_ws) :]
        return origin

    async def connect(self):
        if self.http_mode:
            if self._http_client is None:
                # Persistent pool: TCP + TLS paid once, amortized across calls.
                # HTTP/2 falls back to 1.1 when the optional ``h2`` package is
                # missing; try/except keeps the client running on hosts where
                # it isn't installed. HTTP/2 lets us multiplex future requests
                # on one connection and negotiates cheaper framing.
                try:
                    self._http_client = httpx.AsyncClient(
                        base_url=self._http_origin(),
                        timeout=60.0,
                        http2=True,
                    )
                except ImportError:
                    self._http_client = httpx.AsyncClient(
                        base_url=self._http_origin(),
                        timeout=60.0,
                    )
            print(f"HTTP client ready for {self._http_origin()}")
            return
        print(f"Connecting to {self.url}...")
        self.ws = await websockets.connect(
            self.url, max_size=None
        )  # max_size=None to allow large payloads
        print("Connected.")

    async def disconnect(self):
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def get_config(self) -> dict:
        # Always hit ``/config`` at the HTTP origin — httpx rejects ws/wss schemes.
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self._http_origin()}/config")
            resp.raise_for_status()
            return resp.json()

    async def infer(
        self,
        sample: dict[str, Any],
        run_inference: bool = True,
    ) -> tuple[np.ndarray | None, float]:
        """Request inference from the server.

        Args:
            sample: A batched DataSample-like dict matching the wire schema.
            run_inference: Whether to run inference or just update buffer (server-dependent).
        """
        payload = {"sample": sample, "run_inference": run_inference}
        data = pack(payload)

        if self.http_mode:
            if self._http_client is None:
                raise RuntimeError("Not connected (http_mode). Call connect() first.")
            resp = await self._http_client.post(
                "/infer",
                content=data,
                headers={"content-type": "application/msgpack"},
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"/infer returned HTTP {resp.status_code}: {resp.text[:200]}"
                )
            response = unpack(resp.content)
            if "error" in response:
                raise RuntimeError(f"server error: {response['error']}")
            return response.get("actions"), response.get("inference_time")

        if not self.ws:
            raise RuntimeError("Not connected")

        await self.ws.send(data)
        response_data = await self.ws.recv()
        response = unpack(response_data)

        return response.get("actions"), response.get("inference_time")

    async def start_episode(
        self,
        session_id: str,
        client_id: str,
        metadata: dict | None = None,
    ) -> dict | None:
        """Signal the server to begin a new eval recording episode.

        Sent over HTTP (``POST /episode/start``) in http mode and as a
        ``message_type=episode_start`` WS frame otherwise. ``metadata`` is a
        free-form dict forwarded to the server and embedded in the recorded
        ``.rrd`` + ``.json`` sidecar (e.g. task label, language instruction,
        session label). Returns the server's ``episode`` dict on success, or
        ``None`` when the server does not support the message (older deploys).
        Never raises for transport errors — recording is opt-in and
        best-effort.
        """
        return await self._episode_control("start", session_id, client_id, metadata)

    async def end_episode(self) -> dict | None:
        """Signal the server to finalize + upload the current episode."""
        return await self._episode_control("end", None, None, None)

    async def _episode_control(
        self,
        action: str,
        session_id: str | None,
        client_id: str | None,
        metadata: dict | None,
    ) -> dict | None:
        if action not in ("start", "end"):
            raise ValueError(f"episode control action must be start|end, got {action}")

        if self.http_mode:
            if self._http_client is None:
                return None
            path = f"/episode/{action}"
            json_body: dict[str, Any] = {}
            if action == "start":
                json_body = {
                    "session_id": session_id or "",
                    "client_id": client_id or "",
                }
                if metadata:
                    json_body["metadata"] = metadata
            try:
                resp = await self._http_client.post(path, json=json_body)
            except Exception as e:
                print(f"[episode-{action}] HTTP request failed: {e}")
                return None
            if resp.status_code == 404:
                # Server too old — silently accept.
                return None
            if resp.status_code != 200:
                print(
                    f"[episode-{action}] server returned HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
                return None
            try:
                response = unpack(resp.content)
            except Exception as e:
                print(f"[episode-{action}] failed to unpack response: {e}")
                return None
            return response.get("episode")

        if not self.ws:
            return None

        payload: dict[str, Any] = {"message_type": f"episode_{action}"}
        if action == "start":
            payload.update(
                {
                    "session_id": session_id or "",
                    "client_id": client_id or "",
                }
            )
            if metadata:
                payload["metadata"] = metadata
        try:
            await self.ws.send(pack(payload))
            response_data = await self.ws.recv()
        except Exception as e:
            print(f"[episode-{action}] WS exchange failed: {e}")
            return None
        try:
            response = unpack(response_data)
        except Exception as e:
            print(f"[episode-{action}] failed to unpack WS response: {e}")
            return None
        return response.get("episode")
