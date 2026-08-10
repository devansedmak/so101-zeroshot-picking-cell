"""Unit tests for the mock-WMS webhook receiver (``POST /orders`` → the agent loop).

Hermetic: a real :class:`ThreadingHTTPServer` bound to **port 0 on loopback** (so the OS
picks a free port and nothing leaves the machine) driven with ``http.client``. No
hardware, no Cyberwave SDK, no outbound network — the runner is injected, exactly the
seam ``order_api.make_handler`` exists for.

Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (see runbook).
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from src.agent_service.loop import Fulfillment, fulfill_order
from src.agent_service.order_api import FAILED_STATUS, demo_curl, make_server
from src.agent_service.poses import PICK_POSES, PLACE_POSES
from src.control import MotionExecutor
from src.wms_mock import Order

GOOD_ORDER = {"order_id": "SO-1042", "item": "red marker", "bin": "A"}


class RecordingRunner:
    """Fake runner: records the orders it was handed, returns a canned outcome."""

    def __init__(self, status: str = "fulfilled", error: str | None = None) -> None:
        self.orders: list[Order] = []
        self.status = status
        self.error = error

    def __call__(self, order: Order) -> Fulfillment:
        self.orders.append(order)
        return Fulfillment(
            order=order,
            status=self.status,
            stages=["picked", "placed", "alerted"],
            attempts=1,
            error=self.error,
            alert={"dispatched": False},
        )


class FakeJoints:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, bool]] = []

    def set(self, joint_name: str, position: float, degrees: bool = True):
        self.calls.append((joint_name, position, degrees))


class FakeAlerts:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return {"id": "alert-123", **kwargs}


class FakeRobot:
    def __init__(self) -> None:
        self.joints = FakeJoints()
        self.alerts = FakeAlerts()


@contextmanager
def serving(runner: Any) -> Iterator[tuple[str, int]]:
    """Start the webhook server on an ephemeral loopback port for the duration of a test."""
    server = make_server(runner, host="127.0.0.1", port=0)
    # Small poll interval: shutdown() waits up to one interval, and the default 0.5s
    # would make every test in this file half a second slower for nothing.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server.server_address[0], server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(
    addr: tuple[str, int], method: str, path: str, body: str | None = None
) -> tuple[int, dict[str, Any], dict[str, str]]:
    """Fire one request, return ``(status, parsed JSON body, headers)``."""
    conn = http.client.HTTPConnection(addr[0], addr[1], timeout=5)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw), dict(resp.getheaders())
    finally:
        conn.close()


def post_order(addr: tuple[str, int], payload: Any) -> tuple[int, dict[str, Any]]:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    status, data, _ = request(addr, "POST", "/orders", body)
    return status, data


# --- happy path ----------------------------------------------------------


def test_post_order_fulfilled_returns_200_and_outcome():
    runner = RecordingRunner()
    with serving(runner) as addr:
        status, data = post_order(addr, GOOD_ORDER)

    assert status == 200
    assert data["status"] == "fulfilled"
    assert data["order_id"] == "SO-1042"
    assert data["order"] == {"item": "red marker", "bin": "A"}
    assert data["stages"] == ["picked", "placed", "alerted"]
    assert data["attempts"] == 1
    assert data["reason"] is None


def test_post_order_invokes_runner_once_with_parsed_order():
    runner = RecordingRunner()
    with serving(runner) as addr:
        status, _ = post_order(addr, {"item": "  blue marker ", "bin": " B "})

    assert status == 200
    assert runner.orders == [Order(item="blue marker", bin="B")]  # exactly once, stripped


def test_response_is_json_content_type():
    with serving(RecordingRunner()) as addr:
        _, _, headers = request(addr, "POST", "/orders", json.dumps(GOOD_ORDER))
    assert headers["Content-Type"] == "application/json"


def test_health_reports_ok():
    with serving(RecordingRunner()) as addr:
        status, data, _ = request(addr, "GET", "/health")
    assert status == 200
    assert data["status"] == "ok"
    assert "/orders" in data["routes"]


# --- rejected requests (422) ---------------------------------------------


def test_malformed_json_returns_422():
    runner = RecordingRunner()
    with serving(runner) as addr:
        status, data = post_order(addr, '{"item": "red marker", ')
    assert status == 422
    assert "invalid JSON" in data["error"]
    assert runner.orders == []  # the arm was never asked to do anything


def test_missing_required_field_returns_422():
    runner = RecordingRunner()
    with serving(runner) as addr:
        status, data = post_order(addr, {"order_id": "SO-1", "item": "red marker"})
    assert status == 422
    assert "bin" in data["error"]
    assert runner.orders == []


def test_blank_required_field_returns_422():
    runner = RecordingRunner()
    with serving(runner) as addr:
        status, data = post_order(addr, {"item": "   ", "bin": "A"})
    assert status == 422
    assert "item" in data["error"]
    assert runner.orders == []


def test_non_object_body_returns_422():
    with serving(RecordingRunner()) as addr:
        status, data = post_order(addr, [GOOD_ORDER])
    assert status == 422
    assert "JSON object" in data["error"]


# --- well-formed but unfulfillable (409) ---------------------------------


def test_unknown_item_returns_conflict_with_reason():
    """End-to-end through the REAL loop: a valid order the cell can't fulfil ⇒ 409."""
    robot = FakeRobot()
    executor = MotionExecutor(robot, dry_run=True)

    def runner(order: Order) -> Fulfillment:
        return fulfill_order(order, executor, robot=robot, dry_run_alert=True)

    with serving(runner) as addr:
        status, data = post_order(addr, {"order_id": "SO-9", "item": "ghost pen", "bin": "A"})

    assert status == int(FAILED_STATUS) == 409
    assert data["status"] == "failed"
    assert "UnknownItem" in data["reason"]
    assert data["order_id"] == "SO-9"
    assert robot.joints.calls == []  # dry-run: nothing moved


# --- routing -------------------------------------------------------------


def test_unknown_route_returns_404():
    with serving(RecordingRunner()) as addr:
        status, data, _ = request(addr, "GET", "/nope")
    assert status == 404
    assert "unknown route" in data["error"]


def test_get_orders_returns_405_with_allow_header():
    with serving(RecordingRunner()) as addr:
        status, data, headers = request(addr, "GET", "/orders")
    assert status == 405
    assert headers.get("Allow") == "POST"
    assert "POST" in data["error"]


def test_post_health_returns_405():
    with serving(RecordingRunner()) as addr:
        status, _, headers = request(addr, "POST", "/health", "{}")
    assert status == 405
    assert headers.get("Allow") == "GET"


def test_post_to_unknown_route_returns_404_without_running_the_order():
    runner = RecordingRunner()
    with serving(runner) as addr:
        status, data, _ = request(addr, "POST", "/orders/extra", json.dumps(GOOD_ORDER))
    assert status == 404
    assert runner.orders == []


# --- crashing runner (500, server survives) ------------------------------


def test_runner_exception_returns_500_json_and_server_stays_alive():
    def exploding_runner(order: Order) -> Fulfillment:
        raise RuntimeError("USB port vanished")

    with serving(exploding_runner) as addr:
        status, data = post_order(addr, GOOD_ORDER)
        assert status == 500
        assert data["status"] == "error"
        assert "RuntimeError" in data["error"] and "USB port vanished" in data["error"]
        assert "Traceback" not in json.dumps(data)

        # still serving after the blow-up
        health_status, health, _ = request(addr, "GET", "/health")
        assert health_status == 200 and health["status"] == "ok"


# --- serialization (one arm, one order at a time) ------------------------


def test_concurrent_posts_are_serialized():
    overlaps: list[str] = []
    active = {"n": 0}

    def slow_runner(order: Order) -> Fulfillment:
        active["n"] += 1
        if active["n"] > 1:
            overlaps.append(order.item)
        time.sleep(0.05)
        active["n"] -= 1
        return Fulfillment(order=order, status="fulfilled", stages=["picked", "placed"])

    with serving(slow_runner) as addr:
        results: list[int] = []
        threads = [
            threading.Thread(target=lambda: results.append(post_order(addr, GOOD_ORDER)[0]))
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert results == [200, 200, 200, 200]
    assert overlaps == [], "two orders ran at once — a single arm cannot do that"


# --- demo ergonomics -----------------------------------------------------


def test_demo_curl_advertises_a_fulfillable_order():
    cmd = demo_curl("127.0.0.1", 8080)
    assert cmd.startswith("curl -sS -X POST http://127.0.0.1:8080/orders")
    body = json.loads(cmd.split("-d '", 1)[1].rstrip("'"))
    # the command printed at startup must be one the cell can actually fulfil
    assert body["item"] in PICK_POSES
    assert body["bin"] in PLACE_POSES


@pytest.mark.parametrize("path", ["/health/", "/orders/"])
def test_trailing_slash_is_tolerated(path):
    with serving(RecordingRunner()) as addr:
        method = "GET" if "health" in path else "POST"
        body = None if method == "GET" else json.dumps(GOOD_ORDER)
        status, _, _ = request(addr, method, path, body)
    assert status == 200
