"""Unit tests for the SO-101 motion executor (pure logic, no hardware/network).

Covers the safety contract that matters for a public-demo arm: clamping,
all-or-nothing validation, smooth ramping, and in-memory pose tracking.
"""

from __future__ import annotations

import pytest

from src.control.motion import (
    Action,
    DEFAULT_JOINT_LIMITS,
    JOINTS,
    MotionExecutor,
    MotionPlan,
    clamp,
    validate_plan,
)


class FakeJoints:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, bool]] = []

    def set(self, joint_name: str, position: float, degrees: bool = True):
        self.calls.append((joint_name, position, degrees))


class FakeRobot:
    def __init__(self) -> None:
        self.joints = FakeJoints()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make ramping instantaneous so tests don't wait on wall-clock time."""
    monkeypatch.setattr("src.control.motion.time.sleep", lambda _s: None)


# --- clamp ---------------------------------------------------------------


def test_clamp_within_bounds_is_unchanged():
    assert clamp("1", 45) == 45


def test_clamp_squeezes_out_of_range():
    # Assert against the constant, not a copy of it: the limits are derived from the twin's
    # own calibration (hardware/config/joint-ranges.md) and are expected to change if the
    # arm is ever re-calibrated. What must hold is that clamping lands exactly on the bound.
    for joint, (lo, hi) in DEFAULT_JOINT_LIMITS.items():
        assert clamp(joint, 9999) == hi
        assert clamp(joint, -9999) == lo
    # and the base really is the widest sweep
    assert DEFAULT_JOINT_LIMITS["1"][1] > DEFAULT_JOINT_LIMITS["2"][1]


def test_clamp_unknown_joint_uses_conservative_default():
    assert clamp("99", 9999) == 60
    assert clamp("99", -9999) == -60


# --- validation ----------------------------------------------------------


def test_valid_plan_has_no_errors():
    plan = MotionPlan.from_dict(
        {
            "say": "wave",
            "actions": [
                {"type": "set_joint", "joint": "1", "angle": 30, "duration": 0.5},
                {"type": "home", "duration": 1.0},
                {"type": "wait", "duration": 0.2},
                {"type": "set_pose", "pose": {"1": 10, "2": -10}, "duration": 0.5},
            ],
        }
    )
    assert validate_plan(plan) == []


def test_too_many_actions_rejected():
    plan = MotionPlan(actions=[Action(type="home") for _ in range(9)])
    assert any("too many actions" in e for e in validate_plan(plan))


def test_unknown_action_type_rejected():
    plan = MotionPlan(actions=[Action(type="teleport")])  # type: ignore[arg-type]
    assert any("unknown type" in e for e in validate_plan(plan))


def test_negative_and_excessive_duration_rejected():
    plan = MotionPlan(
        actions=[
            Action(type="home", duration=-1.0),
            Action(type="home", duration=99.0),
        ]
    )
    errors = validate_plan(plan)
    assert any("duration must be ≥ 0" in e for e in errors)
    assert any("exceeds MAX_DURATION_S" in e for e in errors)


def test_set_joint_unknown_joint_and_missing_angle_rejected():
    plan = MotionPlan(
        actions=[
            Action(type="set_joint", joint="7", angle=10),
            Action(type="set_joint", joint="1", angle=None),
        ]
    )
    errors = validate_plan(plan)
    assert any("unknown joint" in e for e in errors)
    assert any("missing 'angle'" in e for e in errors)


def test_set_pose_empty_and_unknown_joint_rejected():
    plan = MotionPlan(
        actions=[
            Action(type="set_pose", pose={}),
            Action(type="set_pose", pose={"9": 10}),
        ]
    )
    errors = validate_plan(plan)
    assert any("missing or empty 'pose'" in e for e in errors)
    assert any("unknown joint" in e for e in errors)


# --- execution / safety --------------------------------------------------


def test_execute_rejects_invalid_plan_and_moves_nothing():
    robot = FakeRobot()
    ex = MotionExecutor(robot)
    bad = MotionPlan(actions=[Action(type="set_joint", joint="7", angle=10)])
    with pytest.raises(ValueError):
        ex.execute(bad)
    assert robot.joints.calls == []


def test_dry_run_tracks_pose_without_calling_robot():
    robot = FakeRobot()
    ex = MotionExecutor(robot, dry_run=True)
    ex.execute(MotionPlan(actions=[Action(type="set_joint", joint="1", angle=30, duration=0.5)]))
    assert robot.joints.calls == []
    assert ex._current_pose["1"] == pytest.approx(30)


def _positions_for(robot: FakeRobot, joint: str) -> list[float]:
    """Commands issued for a single joint (the executor re-sends the full pose each step)."""
    return [pos for (name, pos, _) in robot.joints.calls if name == joint]


def test_out_of_range_angle_is_clamped_before_sdk_call():
    robot = FakeRobot()
    ex = MotionExecutor(robot)
    ex.execute(MotionPlan(actions=[Action(type="set_joint", joint="1", angle=9999, duration=0.5)]))
    # every command ever issued for any joint must be within that joint's limits
    assert robot.joints.calls, "expected at least one joint command"
    for name, pos, _ in robot.joints.calls:
        lo, hi = DEFAULT_JOINT_LIMITS[name]
        assert lo <= pos <= hi
    # ends exactly at joint 1's upper limit, whatever the calibration says it is
    assert _positions_for(robot, "1")[-1] == pytest.approx(DEFAULT_JOINT_LIMITS["1"][1])


def test_ramp_step_count_and_monotonic_arrival():
    robot = FakeRobot()
    ex = MotionExecutor(robot, ramp_hz=20)
    ex.execute(MotionPlan(actions=[Action(type="set_joint", joint="1", angle=40, duration=1.0)]))
    positions = _positions_for(robot, "1")
    assert len(positions) == 20  # duration * ramp_hz steps for the moving joint
    assert positions == sorted(positions)  # monotonic increase toward target
    assert positions[-1] == pytest.approx(40)


def test_set_pose_multijoint_arrives_together():
    robot = FakeRobot()
    ex = MotionExecutor(robot, ramp_hz=10)
    ex.execute(
        MotionPlan(actions=[Action(type="set_pose", pose={"1": 20, "2": -30}, duration=1.0)])
    )
    assert ex._current_pose["1"] == pytest.approx(20)
    assert ex._current_pose["2"] == pytest.approx(-30)


def test_home_returns_all_joints_to_zero():
    robot = FakeRobot()
    ex = MotionExecutor(robot, dry_run=True)
    ex.execute(MotionPlan(actions=[Action(type="set_pose", pose={"1": 20, "3": 15}, duration=0.5)]))
    ex.home(duration=0.5)
    assert all(ex._current_pose[j] == pytest.approx(0.0) for j in JOINTS)


def test_name_map_translates_at_sdk_boundary_only():
    robot = FakeRobot()
    # canonical "1".."6" → twin's actual "_1".."_6"
    name_map = {j: f"_{j}" for j in JOINTS}
    ex = MotionExecutor(robot, name_map=name_map)
    ex.execute(MotionPlan(actions=[Action(type="set_joint", joint="1", angle=20, duration=0.5)]))
    # SDK receives the mapped names; canonical keys are unchanged internally
    assert all(name.startswith("_") for (name, _, _) in robot.joints.calls)
    assert ("_1", pytest.approx(20), True) in [(n, p, d) for (n, p, d) in robot.joints.calls]
    assert ex._current_pose["1"] == pytest.approx(20)  # tracked under canonical key


def test_zero_duration_snaps_in_one_step():
    robot = FakeRobot()
    ex = MotionExecutor(robot)
    ex.execute(MotionPlan(actions=[Action(type="set_joint", joint="1", angle=25, duration=0.0)]))
    # duration 0 → single snap: each joint commanded exactly once
    assert _positions_for(robot, "1") == [pytest.approx(25)]
    assert len(robot.joints.calls) == len(JOINTS)
