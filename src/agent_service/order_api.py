"""HTTP webhook receiver — the **"order-driven"** half of "Order-Driven Zero-Shot Picking Cell".

Until now orders only came from the in-process mock list (``wms_mock.OrderSource``), so the
project name was only half true: nothing outside the process could *trigger* the agent.
This module closes that gap — a mock WMS ``POST``s an order and the arm moves
(docs/demo-scenario.md §"How it drives the agentic loop", step 1):

    curl -X POST /orders {"order_id","item","bin"} → Order.from_dict → fulfill_order → JSON

**Standard library only** (``http.server`` + ``json``): no FastAPI/uvicorn/flask in the
venv and adding a web stack days before submission is unacceptable risk (CLAUDE.md rule 6
— bias to shipping, log the debt).
TODO(post-cohort): re-implement as a FastAPI app (pydantic request model, async, OpenAPI
schema, TestClient) and mount it behind the Cyberwave webhook workflow; the seam here is
already the right one — ``make_handler(runner)`` — so only the transport changes.

Design notes:
- The handler is **injected with a runner callable** ``(Order) -> Fulfillment`` and never
  builds a robot itself, so tests drive it with a fake and zero hardware (same duck-typed
  style as ``loop.fulfill_order``'s ``verify``).
- Fulfilment is **serialized behind a lock**: there is exactly one physical arm, so two
  concurrent orders would interleave joint commands and wreck both. The server stays
  threaded only so ``/health`` and error replies never block behind a running order.
- Any exception from the runner becomes a JSON **500**, never a stack trace on the wire
  and never a dead server.

Routes:
    POST /orders   → 200 fulfilled · 409 well-formed but not fulfilled · 422 malformed
                     · 413 body too large · 500 runner exploded
    GET  /health   → 200 {"status": "ok", ...}
    (other)        → 404 · wrong method on a known route → 405 + Allow

SAFETY (CLAUDE.md rule 2): default is **dry-run** (no connection, no motion). ``--sim``
goes through ``control.sim_session``, which pins ``cw.affect("simulation")``; there is no
live-hardware path here.

Usage:
    python -m src.agent_service.order_api                  # dry-run receiver on :8080
    python -m src.agent_service.order_api --verify         # + closed loop (stub verifier)
    python -m src.agent_service.order_api --sim            # POSTs move the twin in SIMULATION
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from src.control import MotionExecutor
from src.wms_mock import Order

from .loop import Fulfillment, fulfill_order
from .poses import PICK_POSES, PLACE_POSES

# Reused from the single-order CLI so the two entrypoints behave identically (no
# duplicated offline stubs — CLAUDE.md rule 5). Private-but-sibling on purpose.
from .run_order import _NullRobot, _live_verifier, _stub_verifier

SERVICE = "order-webhook"

#: ``(Order) -> Fulfillment``-ish. Duck-typed: anything with ``status``/``stages`` works.
Runner = Callable[[Order], Any]

#: Refuse absurd bodies outright — an order is ~80 bytes; this is a mock WMS, not an API.
MAX_BODY_BYTES = 64 * 1024

ROUTES: dict[str, list[str]] = {"/orders": ["POST"], "/health": ["GET"]}

# HTTP code for "your order was fine, the robot could not fulfil it". Documented choice:
# 409 Conflict (the request conflicts with the state of the cell — item missing, placement
# unverified) rather than 200-with-failed-status, so `curl -f` / the demo script can tell
# success from failure without parsing the body. The body still carries the full outcome.
FAILED_STATUS = HTTPStatus.CONFLICT


def outcome_payload(result: Any, order: Order, order_id: str | None = None) -> dict[str, Any]:
    """JSON-safe view of a :class:`Fulfillment` (read via ``getattr`` to stay duck-typed)."""
    status = str(getattr(result, "status", "failed"))
    return {
        "status": status,
        "order_id": order_id,
        "order": order.to_dict(),
        "stages": list(getattr(result, "stages", []) or []),
        "attempts": int(getattr(result, "attempts", 0) or 0),
        "verification": getattr(result, "verification", None),
        "alert": getattr(result, "alert", None),
        "reason": getattr(result, "error", None),
    }


def make_handler(
    runner: Runner, *, lock: threading.Lock | None = None
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to ``runner`` (and a serialization lock)."""
    run_lock = lock if lock is not None else threading.Lock()

    class OrderHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: Content-Length is always set below
        server_version = f"{SERVICE}/0.1"

        # --- routing ----------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 — stdlib naming
            path = self._path()
            if path == "/health":
                self._json(HTTPStatus.OK, {"status": "ok", "service": SERVICE, "routes": ROUTES})
            else:
                self._reject_route(path, "GET")

        def do_POST(self) -> None:  # noqa: N802 — stdlib naming
            # Always drain the body first: an undrained request breaks HTTP/1.1 keep-alive.
            raw = self._read_body()
            if raw is None:
                return  # 411/413 already sent
            path = self._path()
            if path != "/orders":
                self._reject_route(path, "POST")
                return
            self._handle_order(raw)

        # --- POST /orders -----------------------------------------------

        def _handle_order(self, raw: bytes) -> None:
            try:
                data = json.loads(raw.decode("utf-8") or "null")
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"invalid JSON body: {e}")
                return
            if not isinstance(data, dict):
                self._error(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    f"order body must be a JSON object, got {type(data).__name__}",
                )
                return
            try:
                order = Order.from_dict(data)  # the one validator, shared with the CLI
            except ValueError as e:
                self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(e))
                return
            order_id = str(data.get("order_id") or "").strip() or None

            # One arm ⇒ one order at a time. Serialize fulfilment; reply outside the lock.
            with run_lock:
                self._log(f"▶ {order_id or '(no id)'}: {order.item!r} → bin {order.bin!r}")
                try:
                    result = runner(order)
                except Exception as e:  # noqa: BLE001 — a broken runner must not kill the server
                    self._log(f"‼ runner error: {type(e).__name__}: {e}")
                    self._error(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        f"fulfilment crashed: {type(e).__name__}: {e}",
                        extra={"order": order.to_dict(), "order_id": order_id, "status": "error"},
                    )
                    return

            payload = outcome_payload(result, order, order_id)
            ok = payload["status"] == "fulfilled"
            self._json(HTTPStatus.OK if ok else FAILED_STATUS, payload)

        # --- plumbing ---------------------------------------------------

        def _path(self) -> str:
            return urlparse(self.path).path.rstrip("/") or "/"

        def _read_body(self) -> bytes | None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._error(HTTPStatus.LENGTH_REQUIRED, "invalid Content-Length header")
                return None
            if length > MAX_BODY_BYTES:
                self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    f"body larger than {MAX_BODY_BYTES} bytes",
                )
                return None
            return self.rfile.read(length) if length > 0 else b""

        def _reject_route(self, path: str, method: str) -> None:
            allowed = ROUTES.get(path)
            if allowed:
                self._error(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    f"{method} not allowed on {path}; use {'/'.join(allowed)}",
                    headers={"Allow": ", ".join(allowed)},
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, f"unknown route {path}", extra={"routes": ROUTES})

        def _error(
            self,
            code: HTTPStatus,
            reason: str,
            *,
            extra: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            body: dict[str, Any] = {"status": "rejected", "error": reason}
            body.update(extra or {})
            self._json(code, body, headers=headers)

        def _json(
            self,
            code: HTTPStatus,
            payload: dict[str, Any],
            *,
            headers: dict[str, str] | None = None,
        ) -> None:
            blob = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(int(code))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(blob)

        def _log(self, message: str) -> None:
            print(f"[order_api] {message}", flush=True)

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN401 — stdlib signature
            self._log(fmt % args)

    return OrderHandler


def make_server(
    runner: Runner, *, host: str = "127.0.0.1", port: int = 8080
) -> ThreadingHTTPServer:
    """A ready-to-``serve_forever`` webhook server. ``port=0`` picks a free port (tests)."""
    return ThreadingHTTPServer((host, port), make_handler(runner))


def make_runner(
    executor: MotionExecutor,
    *,
    robot: Any = None,
    dry_run_alert: bool = False,
    verifier_for: Callable[[Order], Any] | None = None,
) -> Runner:
    """Bind the existing loop into a ``(Order) -> Fulfillment`` callable for the handler.

    ``verifier_for(order) -> Verifier | None`` is built per order because the verifier
    needs the order's bin region (see ``run_order._live_verifier``).
    """

    def run(order: Order) -> Fulfillment:
        verify = verifier_for(order) if verifier_for is not None else None
        return fulfill_order(
            order, executor, robot=robot, dry_run_alert=dry_run_alert, verify=verify
        )

    return run


def demo_curl(host: str, port: int) -> str:
    """The exact command to paste in the demo — built from an item/bin that really works."""
    item = "red marker" if "red marker" in PICK_POSES else sorted(PICK_POSES)[0]
    bin_ = "A" if "A" in PLACE_POSES else sorted(PLACE_POSES)[0]
    body = json.dumps({"order_id": "SO-1042", "item": item, "bin": bin_})
    return (
        f"curl -sS -X POST http://{host}:{port}/orders "
        f"-H 'Content-Type: application/json' -d '{body}'"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mock-WMS webhook receiver: POST /orders drives the agent loop (sim only)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback).")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080).")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Actually move the twin in SIMULATION (default is dry-run: no connection/motion).",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=3.0,
        help="Seconds to wait after warm-up so the controller finishes attaching (--sim only).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Close the loop: verify the placement, retry once, else ERROR alert (D9 thesis).",
    )
    parser.add_argument(
        "--verify-fail",
        type=int,
        default=0,
        metavar="N",
        help="Dry-run only: make the stub verifier fail its first N checks (per order).",
    )
    parser.add_argument(
        "--camera", help="Camera device for the verification re-capture (--sim --verify only)."
    )
    args = parser.parse_args(argv)

    if args.sim:
        from src.control.sim_session import load_env, open_sim_executor

        load_env()
        twin_id = os.getenv("CYBERWAVE_TWIN_ID")
        if not twin_id:
            print("ERROR: CYBERWAVE_TWIN_ID is not set (see .env).", file=sys.stderr)
            return 2
        executor, robot = open_sim_executor(twin_id, settle=args.settle)
        verifier_for = None
        if args.verify:
            verifier_for = lambda order: _live_verifier(order.bin, device=args.camera)  # noqa: E731
        runner = make_runner(executor, robot=robot, verifier_for=verifier_for)
        mode = "SIMULATION (twin will move)"
    else:
        null = _NullRobot()
        executor = MotionExecutor(null, dry_run=True)
        # Fresh stub per order so --verify-fail replays the retry path on every POST.
        verifier_for = None
        if args.verify:
            verifier_for = lambda order: _stub_verifier(args.verify_fail)  # noqa: E731
        runner = make_runner(executor, robot=null, dry_run_alert=True, verifier_for=verifier_for)
        mode = "DRY RUN (no connection, no motion)"

    server = make_server(runner, host=args.host, port=args.port)
    host, port = server.server_address[0], server.server_address[1]
    banner = [f"[order_api] {mode}"]
    if args.verify:
        kind = "live" if args.sim else "stub"
        banner.append(f"[order_api] closed loop enabled ({kind} verifier)")
    if host not in ("127.0.0.1", "::1", "localhost"):
        banner.append(f"[order_api] ⚠ bound to {host} — reachable from the network.")
    banner += [
        f"[order_api] listening on http://{host}:{port}  (Ctrl-C to stop)",
        f"[order_api] health : curl -sS http://{host}:{port}/health",
        f"[order_api] DEMO   : {demo_curl(host, port)}",
    ]
    print("\n".join(banner), flush=True)  # flush: visible immediately even when piped/logged
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[order_api] stopping …")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
