"""The dashboard's HTTP surface: **standard library only**, same call as ``order_api``.

No FastAPI, no CDN, no external asset: one process, one port, and a page that renders
with the network cable pulled out. The handler is deliberately
thin: it parses, delegates to :class:`src.gui.app.DashboardApp`, and serializes.

Routes:
    GET  /            -> the single self-contained HTML page
    GET  /state       -> the whole document the page polls (JSON)
    GET  /frame.jpg   -> the current overhead frame (JPEG bytes, never cached)
    GET  /health      -> {"status": "ok", ...} for a curl smoke test
    POST /events      -> feed events in: one dict, a list, or {"events": [...]}
    POST /mode        -> {"mode": "order"|"qc"|"fusion"}
    POST /replay      -> {"outcome": "auto"|"ok"|"retry"|"fail"} (mock only)

Nothing here can move the robot: the dashboard is a viewer. The only writes it performs
are into its own in-memory state.
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .page import render_page

SERVICE = "picking-cell-dashboard"

#: A frame POST carries a path, not pixels; an event batch is small. Refuse anything bigger.
MAX_BODY_BYTES = 1024 * 1024

ROUTES: dict[str, list[str]] = {
    "/": ["GET"],
    "/state": ["GET"],
    "/frame.jpg": ["GET"],
    "/health": ["GET"],
    "/events": ["POST"],
    "/mode": ["POST"],
    "/replay": ["POST"],
}


def parse_events(data: Any) -> list[dict[str, Any]]:
    """Accept the three shapes a producer might reasonably send. Never raises."""
    if isinstance(data, dict):
        if isinstance(data.get("events"), list):
            return [e for e in data["events"] if isinstance(e, dict)]
        return [data] if data.get("kind") else []
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    return []


def make_handler(app: Any, *, quiet: bool = True) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to a :class:`DashboardApp`."""

    class DashboardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: Content-Length is always set
        server_version = f"{SERVICE}/1.0"

        # --- routing ----------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802, stdlib naming
            path = self._path()
            if path in ("/", "/index.html"):
                self._bytes(HTTPStatus.OK, render_page(), "text/html; charset=utf-8")
            elif path == "/state":
                self._json(HTTPStatus.OK, app.snapshot())
            elif path == "/frame.jpg":
                self._frame()
            elif path == "/health":
                self._json(
                    HTTPStatus.OK,
                    {"status": "ok", "service": SERVICE, "source": app.state.source,
                     "mode": app.state.mode, "routes": ROUTES},
                )
            else:
                self._reject(path, "GET")

        def do_POST(self) -> None:  # noqa: N802, stdlib naming
            raw = self._read_body()  # always drain first (HTTP/1.1 keep-alive)
            if raw is None:
                return
            path = self._path()
            if path not in ("/events", "/mode", "/replay"):
                self._reject(path, "POST")
                return
            try:
                data = json.loads(raw.decode("utf-8") or "null")
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": f"invalid JSON: {e}"})
                return
            if path == "/events":
                self._json(HTTPStatus.OK, app.ingest(parse_events(data)))
            elif path == "/mode":
                mode = (data or {}).get("mode") if isinstance(data, dict) else None
                try:
                    self._json(HTTPStatus.OK, {"mode": app.set_mode(mode)})
                except Exception as e:  # noqa: BLE001, a bad mode is a 422, not a 500
                    self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(e)})
            else:
                outcome = (data or {}).get("outcome") if isinstance(data, dict) else None
                self._json(HTTPStatus.OK, {"replayed": app.replay(outcome)})

        # --- plumbing ---------------------------------------------------
        def _frame(self) -> None:
            try:
                jpeg, token = app.frame_jpeg()
            except Exception as e:  # noqa: BLE001, no cv2 / no frame must not 500 the page
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": f"no frame: {e}"})
                return
            if not jpeg:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "no frame yet"})
                return
            self._bytes(HTTPStatus.OK, jpeg, "image/jpeg", headers={"X-Frame-Token": token})

        def _path(self) -> str:
            path = urlparse(self.path).path
            return path if path == "/" else (path.rstrip("/") or "/")

        def _read_body(self) -> bytes | None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._json(HTTPStatus.LENGTH_REQUIRED, {"error": "invalid Content-Length"})
                return None
            if length > MAX_BODY_BYTES:
                self._json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": f"body larger than {MAX_BODY_BYTES} bytes"},
                )
                return None
            return self.rfile.read(length) if length > 0 else b""

        def _reject(self, path: str, method: str) -> None:
            allowed = ROUTES.get(path)
            if allowed:
                self._json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"error": f"{method} not allowed on {path}"},
                    headers={"Allow": ", ".join(allowed)},
                )
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": f"unknown route {path}",
                                                  "routes": ROUTES})

        def _json(
            self,
            code: HTTPStatus,
            payload: dict[str, Any],
            *,
            headers: dict[str, str] | None = None,
        ) -> None:
            blob = json.dumps(payload, default=str).encode("utf-8")
            self._bytes(code, blob, "application/json", headers=headers)

        def _bytes(
            self,
            code: HTTPStatus,
            blob: bytes,
            content_type: str,
            *,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(int(code))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(blob)))
            # The page polls hard; a cached /state or /frame.jpg would freeze the demo.
            self.send_header("Cache-Control", "no-store, max-age=0")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(blob)
            except (BrokenPipeError, ConnectionResetError):  # browser navigated away
                pass

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN401, stdlib signature
            if not quiet:
                print(f"[dashboard] {fmt % args}", flush=True)

    return DashboardHandler


def make_server(
    app: Any, *, host: str = "127.0.0.1", port: int = 8090, quiet: bool = True
) -> ThreadingHTTPServer:
    """A ready-to-``serve_forever`` dashboard server. ``port=0`` picks a free port (tests)."""
    return ThreadingHTTPServer((host, port), make_handler(app, quiet=quiet))


def serve_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    """Run ``serve_forever`` on a daemon thread (used by tests and ``--open``)."""
    thread = threading.Thread(target=server.serve_forever, name="dashboard-http", daemon=True)
    thread.start()
    return thread
