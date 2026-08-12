"""Unit tests for the SO-101 analytic IK (pure math, no hardware/network).

The correctness invariant is the FK∘IK round-trip: solving a reachable table
target and running it back through forward kinematics must land on the target.
Also covers base-pan orientation, reach guards, output shape, and determinism.
Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import math

import pytest

from src.control.ik import (
    BASE_HEIGHT_MM,
    GRIPPER_OPEN,
    JAW_OFFSET_DEG,
    L1_SHOULDER_TO_ELBOW_MM,
    L2_ELBOW_TO_WRIST_MM,
    L3_WRIST_TO_TIP_MM,
    GraspAngleUnreachable,
    Unreachable,
    forward_kinematics,
    in_limits,
    solve_ik,
)
from src.control.motion import DEFAULT_JOINT_LIMITS

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
    # q2+q3-q4 == -90° means the tool points straight down; tip sits directly below wrist.
    #
    # NOTE: the MINUS on q4 is not a typo and must not be "tidied" back to a plus.
    # wrist_flex's positive direction is INVERTED relative to shoulder_lift/elbow_flex.
    # VERIFIED ON THE PHYSICAL ARM 2026-08-11: commanding
    #     (q1, q2, q3, q4) = (11.89, 59.24, -62.00, -87.25)°
    # read back on the servo encoders as (11.5, 59.6, -60.2, -87.1)°, i.e. the arm
    # tracked the command to within half a degree, and yet the gripper pointed straight
    # UP, with the wrist straining against a mechanical end stop.
    #   old convention:  59.6 - 60.2 - 87.1 = -87.7° -> "down"  (contradicted by the arm)
    #   this convention: 59.6 - 60.2 + 87.1 = +86.5° -> "up"    (what was observed)
    # forward_kinematics and solve_ik were flipped together, so the FK∘IK round-trip
    # above passes under either sign; this test is the only thing pinning the one the
    # hardware actually has. Reverting it re-introduces a gripper that points at the
    # ceiling on every pick.
    j = solve_ik(150, 30, 0)
    assert j["2"] + j["3"] - j["4"] == pytest.approx(-90.0, abs=1e-9)


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
    # Tip height that puts the wrist exactly ON the shoulder, so reach 0 < |L1-L2|.
    # Expressed via the constants: with the real URDF lengths no *table-level*
    # point is too near any more, so a hardcoded z would silently stop testing
    # the guard the moment a link length changes.
    z_wrist_on_shoulder = BASE_HEIGHT_MM - L3_WRIST_TO_TIP_MM
    with pytest.raises(Unreachable):
        solve_ik(0, 0, z_wrist_on_shoulder)


def test_unreachable_is_a_valueerror():
    assert issubclass(Unreachable, ValueError)


# --- output shape / branch / determinism ---------------------------------


def test_output_keys_are_exactly_the_six_servos():
    j = solve_ik(150, 0, 0)
    assert set(j) == {"1", "2", "3", "4", "5", "6"}
    assert all(isinstance(v, float) for v in j.values())


def test_elbow_up_branch_and_static_joints():
    j = solve_ik(150, 20, 0)
    assert j["3"] <= 0.0  # elbow-up means q3 ≤ 0 (matches the old hardcoded poses' sign)
    assert j["5"] == 0.0  # wrist_roll fixed when no object axis is given
    assert j["6"] == pytest.approx(GRIPPER_OPEN)  # gripper open


def test_solution_is_deterministic():
    assert solve_ik(170, -30, 10) == solve_ik(170, -30, 10)


# --- oriented grasping: wrist_roll follows the object's long axis --------


def _jaw_direction_deg(joints: dict[str, float]) -> float:
    """Direction the jaws close along, in TABLE degrees mod 180.

    At ``q5 = 0`` the jaws close along the arm's radial direction (``q1``) plus the
    mechanical ``JAW_OFFSET_DEG``; rolling the wrist adds ``q5`` on top. This is the
    independent restatement of the model that the assertions below are checked against,
    the oriented-grasp analogue of the FK∘IK round-trip.
    """
    return (joints["1"] + joints["5"] + JAW_OFFSET_DEG) % 180.0


def _axis_error(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def test_no_axis_keeps_the_old_fixed_roll_exactly():
    # Back-compat: every caller that never heard of axis_deg must get today's answer.
    assert solve_ik(150, 20, 0)["5"] == 0.0
    assert solve_ik(150, 20, 0, None) == solve_ik(150, 20, 0)


@pytest.mark.parametrize(
    ("x", "y", "axis_deg", "expected_q5"),
    [
        (160, 0, 90.0, 0.0),     # object across the reach direction, no roll needed
        (160, 0, 0.0, -90.0),    # object along the reach direction, a quarter turn
        (160, 0, 45.0, -45.0),
        (160, 0, 135.0, 45.0),
        (0, 160, 0.0, 0.0),      # arm panned 90°: the same geometry, rotated with it
        (0, 160, 90.0, -90.0),
    ],
)
def test_wrist_rolls_to_the_expected_angle(x, y, axis_deg, expected_q5):
    assert solve_ik(x, y, 0, axis_deg)["5"] == pytest.approx(expected_q5, abs=1e-9)


@pytest.mark.parametrize("axis_deg", [0.0, 17.0, 45.0, 90.0, 123.0, 179.0])
@pytest.mark.parametrize(("x", "y"), [(160, 0), (150, 60), (120, -110), (60, 150)])
def test_the_jaws_close_perpendicular_to_the_object_axis(x, y, axis_deg):
    # THE invariant: whatever the pan, the closing line ends up 90° off the long axis,
    # which is what stops a marker squirting out of an end-on grasp.
    joints = solve_ik(x, y, 0, axis_deg)
    assert _axis_error(_jaw_direction_deg(joints), axis_deg + 90.0) < 1e-6


@pytest.mark.parametrize("axis_deg", [0.0, 37.0, 90.0, 151.0])
def test_an_axis_and_the_same_axis_turned_180_give_the_same_roll(axis_deg):
    # A parallel jaw cannot tell an object's head from its tail, and neither may we:
    # this symmetry is what lets a ±90° wrist cover every orientation.
    assert solve_ik(150, 40, 0, axis_deg)["5"] == pytest.approx(
        solve_ik(150, 40, 0, axis_deg + 180.0)["5"], abs=1e-9
    )
    assert solve_ik(150, 40, 0, axis_deg)["5"] == pytest.approx(
        solve_ik(150, 40, 0, axis_deg - 180.0)["5"], abs=1e-9
    )


def test_the_axis_never_moves_the_tip():
    # Rolling the wrist is a pure re-orientation: the FK∘IK round-trip must be untouched.
    for axis_deg in (0.0, 30.0, 95.0):
        assert forward_kinematics(solve_ik(170, -30, 10, axis_deg)) == pytest.approx(
            (170, -30, 10), abs=1e-6
        )


@pytest.mark.parametrize("axis_deg", [float(a) for a in range(0, 180, 15)])
@pytest.mark.parametrize(("x", "y"), [(160, 0), (150, 60), (120, -110)])
def test_every_orientation_is_reachable_with_the_widened_roll_limit(x, y, axis_deg):
    # The point of widening DEFAULT_JOINT_LIMITS["5"] to ±90 (joint-ranges.md): mod-180
    # symmetry means half a turn covers every object orientation, so there is NO dead
    # wedge left; no angle raises, and none gets clamped by the executor either.
    joints = solve_ik(x, y, 0, axis_deg)
    lo, hi = DEFAULT_JOINT_LIMITS["5"]
    assert lo <= joints["5"] <= hi


def test_an_out_of_range_grasp_angle_raises_instead_of_clamping(monkeypatch):
    # Reproduces the pre-widening limit (±60°), where a 60° wedge of orientations was
    # unreachable: the solver must SAY so, because MotionExecutor would otherwise clamp
    # and execute a confident grasp at the wrong angle.
    monkeypatch.setitem(DEFAULT_JOINT_LIMITS, "5", (-60, 60))
    with pytest.raises(GraspAngleUnreachable) as excinfo:
        solve_ik(160, 0, 0, 0.0)  # needs -90° of roll

    message = str(excinfo.value)
    assert "-90" in message  # the angle it wanted
    assert "+60" in message and "-60" in message  # the limit it hit
    assert "rotate the object" in message  # and what the operator can do about it


def test_grasp_angle_unreachable_is_a_valueerror():
    # Callers that already catch Unreachable/ValueError keep working.
    assert issubclass(GraspAngleUnreachable, ValueError)


# --- gripper convention (measured on hardware) ---------------------------


def test_gripper_constants_match_the_measured_hardware_convention():
    """Pins open/closed to the follower's own calibrated 0° to 128.9° gripper span.

    MEASURED 2026-08-11: jaws pushed shut by hand read 0.107 rad = 6.1° on the encoder,
    and a commanded 110° -> 60° -> 10° -> 110° sweep was observed as fully open -> about half
    closed -> almost fully shut -> fully open. HIGH = OPEN, LOW = SHUT, no negative angles.
    The previous values (open -40°, closed +20°) were guesses from before any gripper
    calibration, and -40° was outside the range the servo physically has.
    """
    from src.agent_service.poses import GRIPPER_CLOSE
    from src.agent_service.poses import GRIPPER_OPEN as POSES_GRIPPER_OPEN

    # ik.GRIPPER_OPEN is a deliberate copy (avoids a control -> agent_service cycle).
    assert GRIPPER_OPEN == POSES_GRIPPER_OPEN

    assert GRIPPER_OPEN > GRIPPER_CLOSE  # the direction itself: higher angle = wider jaws
    lo, hi = DEFAULT_JOINT_LIMITS["6"]
    for angle in (GRIPPER_OPEN, GRIPPER_CLOSE):
        # Must survive the executor's clamp untouched, or "open" silently becomes "ajar".
        assert lo <= angle <= hi
        assert 0.0 <= angle <= 124.0  # inside the calibrated span, with the usual margin
    assert GRIPPER_CLOSE >= 6.1  # don't drive past the hand-measured hard-shut reading


# --- in_limits helper ----------------------------------------------------


def test_in_limits_reports_without_clamping():
    # A bent-elbow target needs an elbow angle past the conservative skeleton limit,
    # but solve_ik must return it untouched (in_limits just reports the fact).
    j = solve_ik(140, 0, 0)
    assert j["3"] < -60.0  # raw solution genuinely exceeds DEFAULT_JOINT_LIMITS["3"]
    assert in_limits(j) is False
    # A hand-built in-range pose passes. Joint 6 is a POSITIVE angle: the follower's
    # calibrated gripper span is 0° to 128.9° (0 = shut), so a negative "open" is not a
    # thing the servo has. See GRIPPER_OPEN / hardware/config/joint-ranges.md.
    assert in_limits({"1": 10, "2": 20, "3": -30, "4": 15, "5": 0, "6": GRIPPER_OPEN}) is True
