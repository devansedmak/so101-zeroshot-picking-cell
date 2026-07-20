"""Unit tests for the walking-skeleton loop: order parsing + order→pick→place→alert.

Pure logic, no hardware/network. Uses a fake robot that records both joint commands
(mirrors tests/test_motion.py) and alert calls, so the whole loop is offline-testable.
Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (see runbook).
"""

from __future__ import annotations

import pytest

from src.agent_service import (
    build_fulfilled_alert,
    fulfill_order,
    send_alert,
)
from src.agent_service.poses import GRIPPER_CLOSE, PICK_POSES, PLACE_POSES, UnknownItem, pick_plan
from src.control import MotionExecutor
from src.wms_mock import DEMO_ORDERS, Order, OrderSource


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


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make ramping instantaneous so tests don't wait on wall-clock time."""
    monkeypatch.setattr("src.control.motion.time.sleep", lambda _s: None)


# --- order parsing / source ---------------------------------------------


def test_order_from_dict_parses_and_strips():
    order = Order.from_dict({"item": "  red marker ", "bin": " A "})
    assert order == Order(item="red marker", bin="A")
    assert order.to_dict() == {"item": "red marker", "bin": "A"}


def test_order_missing_field_rejected():
    with pytest.raises(ValueError):
        Order.from_dict({"item": "red marker"})


def test_order_empty_field_rejected():
    with pytest.raises(ValueError):
        Order.from_dict({"item": "", "bin": "A"})


def test_order_source_yields_validated_orders():
    src = OrderSource()
    orders = list(src)
    assert len(orders) == len(DEMO_ORDERS)
    assert all(isinstance(o, Order) for o in orders)
    assert src.first() == Order.from_dict(DEMO_ORDERS[0])


def test_order_source_custom_list_and_empty_first():
    src = OrderSource([{"item": "x", "bin": "B"}])
    assert src.first() == Order(item="x", bin="B")
    with pytest.raises(ValueError):
        OrderSource([]).first()


# --- pose lookup ---------------------------------------------------------


def test_pick_plan_grasps_after_approach():
    plan = pick_plan("red marker")
    types = [a.type for a in plan.actions]
    assert types == ["home", "set_pose", "set_joint"]
    grasp = plan.actions[-1]
    assert grasp.joint == "6" and grasp.angle == GRIPPER_CLOSE


def test_pick_plan_unknown_item_raises():
    with pytest.raises(UnknownItem):
        pick_plan("no such item")


def test_every_demo_order_has_poses():
    for raw in DEMO_ORDERS:
        o = Order.from_dict(raw)
        assert o.item in PICK_POSES
        assert o.bin in PLACE_POSES


# --- alert stub ----------------------------------------------------------


def test_build_fulfilled_alert_payload():
    alert = build_fulfilled_alert(Order(item="red marker", bin="A"))
    assert alert["alert_type"] == "order_fulfilled"
    assert alert["severity"] == "info"
    assert alert["category"] == "business"
    assert "message" not in alert  # real SDK has no 'message' kwarg — details go in description
    assert "red marker" in alert["description"] and "A" in alert["description"]


def test_send_alert_dry_run_does_not_touch_robot():
    robot = FakeRobot()
    result = send_alert(robot, build_fulfilled_alert(Order("x", "A")), dry_run=True)
    assert result["dispatched"] is False
    assert robot.alerts.created == []


def test_send_alert_live_calls_sdk():
    robot = FakeRobot()
    result = send_alert(robot, build_fulfilled_alert(Order("x", "A")))
    assert result["dispatched"] is True
    assert len(robot.alerts.created) == 1
    created = robot.alerts.created[0]
    assert created["alert_type"] == "order_fulfilled"
    assert created["source_type"] == "simulation"  # added at send time


# --- full loop wiring ----------------------------------------------------


def test_fulfill_order_moves_arm_and_alerts():
    robot = FakeRobot()
    ex = MotionExecutor(robot)
    result = fulfill_order(Order("red marker", "A"), ex, robot=robot)

    assert result.ok
    assert result.status == "fulfilled"
    assert result.stages == ["picked", "placed", "alerted"]
    # arm was actually driven and every command stayed within joint limits
    assert robot.joints.calls, "expected joint commands"
    # the gripper (joint "6") was actuated during the pick
    assert any(name == "6" for (name, _, _) in robot.joints.calls)
    # a fulfilled alert was dispatched
    assert len(robot.alerts.created) == 1
    assert result.alert and result.alert["dispatched"] is True


def test_fulfill_order_dry_run_moves_nothing_but_fulfills():
    robot = FakeRobot()
    ex = MotionExecutor(robot, dry_run=True)
    result = fulfill_order(Order("eraser", "A"), ex, robot=robot, dry_run_alert=True)

    assert result.ok
    assert robot.joints.calls == []
    assert robot.alerts.created == []


def test_fulfill_order_unknown_item_fails_without_raising():
    robot = FakeRobot()
    ex = MotionExecutor(robot)
    result = fulfill_order(Order("ghost", "A"), ex, robot=robot)

    assert not result.ok
    assert result.status == "failed"
    assert "UnknownItem" in (result.error or "")
    # nothing was picked, nothing alerted
    assert result.stages == []
    assert robot.alerts.created == []


def test_fulfill_order_unknown_bin_fails_after_no_placement():
    robot = FakeRobot()
    ex = MotionExecutor(robot)
    result = fulfill_order(Order("red marker", "Z"), ex, robot=robot)

    assert not result.ok
    assert "UnknownBin" in (result.error or "")
    # planning both plans happens before any motion, so the arm never moved
    assert robot.joints.calls == []
    assert robot.alerts.created == []
