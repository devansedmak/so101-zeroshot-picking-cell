"""Unit tests for the SO-101 analytic IK (pure math, no hardware/network).

The correctness invariant is the FK∘IK round-trip: solving a reachable table
target and running it back through forward kinematics must land on the target.
Also covers base-pan orientation, reach guards, output shape, and determinism.
Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (see runbook).
"""

from __future__ import annotations

import math

import pytest

from src.control.ik import (
    L1_SHOULDER_TO_ELBOW_MM,
    L2_ELBOW_TO_WRIST_MM,
    Unreachable,
    forward_kinematics,
    in_limits,
    solve_ik,
)

# In-reach shell for the wrist (mm); used to filter the round-trip grid.
_LO = abs(L1_SHOULDER_TO_ELBOW_MM - L2_ELBOW_TO_WRIST_MM)
_HI = L1_SHOULDER_TO_ELBOW_MM + L2_ELBOW_TO_WRIST_MM


def _reachable(x: float, y: float, z: float) -> bool:
    # Mirror solve_ik's planar reach test (tip held L3 below the wrist).
    from src.control.ik import BASE_HEIGHT_MM, L3_WRIST_TO_TIP_MM

    reach = math.hypot(math.hypot(x, y), (z + L3_WRIST_TO_TIP_MM) - BASE_HEIGHT_MM)
    return _LO <= reach <= _HI


# --- FK∘IK round-trip ----------------------------------------------------


def test_fk_of_ik_round_trips_on_a_grid():
    tested = 0
    for x in (90, 130, 170, 210):
        for y in (-120, -40, 0, 40, 120):
            for z in (-10, 0, 20, 40):
                if not _reachable(x, y, z):
                    continue
                tip = forward_kinematics(solve_ik(x, y, z))
                assert tip == pytest.approx((x, y, z), abs=1e-6)
                tested += 1
    assert tested > 20  # the grid must actually exercise reachable targets


def test_gripper_is_held_vertical():
    # q2+q3+q4 == -90° ⇒ tool points straight down; tip sits directly below wrist.
    j = solve_ik(150, 30, 0)
    assert j["2"] + j["3"] + j["4"] == pytest.approx(-90.0, abs=1e-9)


# --- base-pan orientation ------------------------------------------------


def test_target_on_positive_x_axis_gives_zero_pan():
    assert solve_ik(160, 0, 0)["1"] == pytest.approx(0.0, abs=1e-9)


def test_target_on_positive_y_axis_gives_ninety_pan():
    assert solve_ik(0, 160, 0)["1"] == pytest.approx(90.0, abs=1e-9)


def test_target_on_negative_y_axis_gives_minus_ninety_pan():
    assert solve_ik(0, -160, 0)["1"] == pytest.approx(-90.0, abs=1e-9)


# --- reach guards --------------------------------------------------------


def test_too_far_target_raises_unreachable():
    with pytest.raises(Unreachable, match="reach"):
        solve_ik(400, 0, 0)  # planar reach > L1+L2


def test_too_near_target_raises_unreachable():
    with pytest.raises(Unreachable):
        solve_ik(0, 0, 20)  # wrist collapses onto the shoulder ⇒ reach < |L1-L2|


def test_unreachable_is_a_valueerror():
    assert issubclass(Unreachable, ValueError)


# --- output shape / branch / determinism ---------------------------------


def test_output_keys_are_exactly_the_six_servos():
    j = solve_ik(150, 0, 0)
    assert set(j) == {"1", "2", "3", "4", "5", "6"}
    assert all(isinstance(v, float) for v in j.values())


def test_elbow_up_branch_and_static_joints():
    j = solve_ik(150, 20, 0)
    assert j["3"] <= 0.0  # elbow-up ⇒ q3 ≤ 0 (matches the old hardcoded poses' sign)
    assert j["5"] == 0.0  # wrist_roll fixed
    assert j["6"] == pytest.approx(-40.0)  # gripper open


def test_solution_is_deterministic():
    assert solve_ik(170, -30, 10) == solve_ik(170, -30, 10)


# --- in_limits helper ----------------------------------------------------


def test_in_limits_reports_without_clamping():
    # A bent-elbow target needs an elbow angle past the conservative skeleton limit,
    # but solve_ik must return it untouched (in_limits just reports the fact).
    j = solve_ik(140, 0, 0)
    assert j["3"] < -60.0  # raw solution genuinely exceeds DEFAULT_JOINT_LIMITS["3"]
    assert in_limits(j) is False
    # A hand-built in-range pose passes.
    assert in_limits({"1": 10, "2": 20, "3": -30, "4": 15, "5": 0, "6": -40}) is True
