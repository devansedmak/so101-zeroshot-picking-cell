"""SO-101 analytic inverse kinematics: table (X, Y, Z) mm -> joint targets (deg).

The missing piece of the loop seam: homography gives a table
point in mm, this turns it into servo angles. Deliberately the *simplest* solver
that closes the loop (a base-pan + planar 2-link reach, gripper held vertical),
not a full 6-DOF numeric IK. Pure math (no SDK / robot / network), so the whole
FK∘IK round-trip is unit-testable offline.

Frame (ASSUMED, see TODO(calibration)): the table (X, Y) handed in is already in
the **robot base frame**: origin under the base pivot, +X forward, +Y left, +Z
up, angles per calibration. The real camera-to-base alignment is a future
calibration offset that will pre-transform (X, Y) before it reaches here.

Model:
- ``q1`` (shoulder_pan) rotates the whole arm about +Z: ``q1 = atan2(y, x)``.
- In the resulting vertical plane, ``(r=hypot(x,y), z)`` is reached by a 2-link
  arm ``L1`` (shoulder -> elbow) + ``L2`` (elbow -> wrist), law-of-cosines, **elbow-up**
  branch (``q3 ≤ 0``, matching the sign of the old hardcoded poses).
- The gripper is held pointing **straight down**; ``q4`` (wrist_flex) is derived
  from ``q2, q3`` so the tool axis is vertical, and the tip sits ``L3`` below the
  wrist. ``q6`` (gripper) = open.
- ``q5`` (wrist_roll) = 0 unless the caller passes ``axis_deg``, the object's long
  axis on the table (perception/orient.py). Then the wrist rolls so the jaws close
  **across** that axis instead of along it. See :func:`solve_ik`.

Angles in DEGREES, lengths in mm. Sign/offset conventions are internally
consistent so that ``forward_kinematics(solve_ik(x, y, z)) ≈ (x, y, z)`` for any
reachable target. That round-trip is THE correctness invariant.

``solve_ik`` does NOT clamp to ``DEFAULT_JOINT_LIMITS`` (the conservative skeleton
limits would silently corrupt a valid grasp). It returns the truthful raw
solution and raises :class:`Unreachable` when the target is out of reach; use
:func:`in_limits` to check a solution against the executor's limits. The
MotionExecutor clamps again at the SDK boundary, so an out-of-limit angle can
never physically leave the arm.
"""

from __future__ import annotations

import math

from .motion import DEFAULT_JOINT_LIMITS, JOINTS

# --- geometry (mm) -------------------------------------------------------
# Derived 2026-08-11 from the official SO-101 URDF (TheRobotStudio/SO-ARM100,
# Simulation/SO101/so101_new_calib.urdf) by chaining the joint origins. See
# hardware/config/link-lengths.md. L1/L2 confirmed the old placeholders exactly;
# BASE_HEIGHT was 3.4 mm high; L3 was 60 mm SHORT, which would have driven the
# gripper into the table on the first live pick.
L1_SHOULDER_TO_ELBOW_MM: float = 116.0  # shoulder_lift pivot -> elbow_flex pivot
L2_ELBOW_TO_WRIST_MM: float = 135.0     # elbow_flex pivot -> wrist_flex pivot
L3_WRIST_TO_TIP_MM: float = 159.8       # wrist_flex pivot -> gripper_frame (TCP)
BASE_HEIGHT_MM: float = 116.6           # table plane -> shoulder_lift pivot

# Default tip height for a table pick: gripper tip at the table surface (Z=0).
PICK_Z: float = 0.0

# Gripper open angle: a local mirror of agent_service.poses.GRIPPER_OPEN, kept
# here to avoid a control -> agent_service import cycle. Keep the two in sync; the
# measurement that fixed the value (hardware 2026-08-11: HIGH = OPEN, LOW = SHUT,
# jaws-touching reads 6.1°) is written up there and in hardware/config/joint-ranges.md.
GRIPPER_OPEN: float = 105.0

# Angle between the JAW CLOSING LINE and the arm's RADIAL direction (base -> tip), at
# wrist_roll q5 = 0°. The current 0.0 asserts "at zero roll the jaws close along the line
# from the base out to the tip". That is an ASSUMPTION, not a measurement.
#
# HOW TO MEASURE IT (still TODO: needs the arm and the overhead camera, ~5 min):
#   1. Move the arm to a known pose over the table with q5 = 0, e.g.
#      `tools/live_check.py hover --x ... --y ...`, which solves IK with no axis_deg and so
#      leaves the roll at zero. Note the commanded (x, y): the radial direction in the
#      overhead frame is the line from the base's image position to that point.
#   2. WITHOUT moving the arm, photograph the jaws from the overhead camera:
#      `.venv/bin/python -m src.perception.capture --save /tmp/jaws.jpg -w 12`.
#   3. In that image, measure the angle from the radial direction (step 1) to the line
#      the two jaws close along (the line joining the fingertips, perpendicular to the
#      jaw faces). Signed the same way as q5, i.e. counter-clockwise positive.
#   4. Put that number here, in degrees. Re-run tests/test_ik.py. The oriented-grasp
#      tests derive the expected jaw direction from this constant, so they follow it.
#
# WHY IT MATTERS: this is a pure constant OFFSET on the wrist roll. A wrong value rotates
# EVERY oriented grasp by exactly the same amount: a systematic bias, identical on every
# object, with no scatter to make it visible. So the jaws end up consistently off the
# perpendicular and long thin items (marker, pen) keep squirting out end-on: precisely the
# failure the axis logic exists to remove, silently reintroduced. Note it is also
# indistinguishable, from the outside, from a constant bias in the perception axis
# estimate (perception/orient.py), so measure it independently rather than tuning it to
# make picks work, or the two errors will be fitted against each other.
JAW_OFFSET_DEG: float = 0.0

_EPS = 1e-6  # mm slack on the reach envelope / float guard on acos


class Unreachable(ValueError):
    """Target lies outside the 2-link reach envelope (too far or too near)."""


class GraspAngleUnreachable(ValueError):
    """The wrist cannot roll far enough to close the jaws across the object's axis.

    Raised instead of silently clamping: ``MotionExecutor._clamp`` would otherwise turn
    an impossible roll into a confident grasp at the WRONG angle, the exact failure
    (long object squirting out of end-on jaws) the axis logic exists to prevent.
    """


def forward_kinematics(joints: dict[str, float]) -> tuple[float, float, float]:
    """Gripper-tip (X, Y, Z) in table/base mm from joint angles (degrees).

    Uses joints "1".."4" (pan, shoulder_lift, elbow_flex, wrist_flex); "5"/"6"
    (roll/gripper) don't move the tip in this planar model. Inverse of
    :func:`solve_ik`. See the module round-trip invariant.
    """
    q1 = math.radians(joints["1"])
    q2 = math.radians(joints["2"])
    q3 = math.radians(joints["3"])
    q4 = math.radians(joints["4"])

    # Planar chain in the (radial, vertical) plane; angles are absolute-from-horizontal.
    a_link2 = q2 + q3          # link2 (forearm) heading
    # wrist_flex's positive direction is INVERTED relative to shoulder/elbow, hence the
    # minus. Verified on hardware 2026-08-11: commanding (11.89, 59.24, -62.00, -87.25)°
    # read back on the encoders as (11.5, 59.6, -60.2, -87.1)°, so the arm tracked the
    # command exactly, yet the tool pointed straight UP. With this sign the same pose
    # gives 59.6 - 60.2 + 87.1 = +86.5° (up), which is what was actually observed.
    a_tool = q2 + q3 - q4      # tool heading (-90° means straight down)
    wrist_r = L1_SHOULDER_TO_ELBOW_MM * math.cos(q2) + L2_ELBOW_TO_WRIST_MM * math.cos(a_link2)
    wrist_z = (
        BASE_HEIGHT_MM
        + L1_SHOULDER_TO_ELBOW_MM * math.sin(q2)
        + L2_ELBOW_TO_WRIST_MM * math.sin(a_link2)
    )
    tip_r = wrist_r + L3_WRIST_TO_TIP_MM * math.cos(a_tool)
    tip_z = wrist_z + L3_WRIST_TO_TIP_MM * math.sin(a_tool)

    return tip_r * math.cos(q1), tip_r * math.sin(q1), tip_z


def _wrist_roll_for_axis(axis_deg: float, q1_deg: float) -> float:
    """Roll (deg) that closes the jaws ACROSS an object lying at ``axis_deg``.

    ``axis_deg`` is the object's long axis in the table frame (mod 180, from
    perception.orient.axis_angle_table); ``q1_deg`` is the base pan, i.e. how far the
    whole arm has already been rotated about +Z. The jaws must close **perpendicular**
    to the long axis, hence the +90°, and the base rotation and the mechanical jaw
    offset are then subtracted to get the angle the wrist itself must supply.

    The result is folded into ``[-90, +90)``: a parallel-jaw gripper is symmetric under
    a 180° roll (swapping which finger is which grasps identically), so every possible
    object orientation is reachable inside half a turn. Raises
    :class:`GraspAngleUnreachable` if even that folded angle is outside the executor's
    wrist_roll limit.
    """
    rel = (axis_deg + 90.0) - q1_deg - JAW_OFFSET_DEG
    q5 = (rel + 90.0) % 180.0 - 90.0

    lo, hi = DEFAULT_JOINT_LIMITS["5"]
    if not (lo <= q5 <= hi):
        raise GraspAngleUnreachable(
            f"object axis {axis_deg:.1f}° at pan {q1_deg:.1f}° needs wrist_roll "
            f"{q5:+.1f}°, outside the limit [{lo:+.0f}, {hi:+.0f}]° — rotate the object "
            f"(or widen DEFAULT_JOINT_LIMITS['5'], see hardware/config/joint-ranges.md)"
        )
    return q5


def solve_ik(
    x: float,
    y: float,
    z: float = PICK_Z,
    axis_deg: float | None = None,
) -> dict[str, float]:
    """Table/base target (mm) -> joint targets in DEGREES, keyed "1".."6".

    Holds the gripper vertical (tip ``L3`` below the wrist), solves the 2-link
    reach on the **elbow-up** branch, and sets gripper=open. Raises
    :class:`Unreachable` if the (radial, vertical) target is outside
    ``[|L1-L2|, L1+L2]`` (z is folded into this planar reach check). The result
    is NOT clamped to ``DEFAULT_JOINT_LIMITS``. See the module docstring and
    :func:`in_limits`.

    ``axis_deg`` is the object's long axis on the table in degrees (mod 180, from
    perception.orient). ``None`` means roll stays at 0° exactly as before, so a scene
    with no measurable axis behaves like it always did. Given one, the wrist rolls
    so the jaws close across the object (:func:`_wrist_roll_for_axis`), which can
    raise :class:`GraspAngleUnreachable`.
    """
    q1 = math.atan2(y, x)

    # Gripper vertical means the tip is straight below the wrist by L3; solve for the wrist.
    r = math.hypot(x, y)
    wrist_r = r
    dz = (z + L3_WRIST_TO_TIP_MM) - BASE_HEIGHT_MM

    reach = math.hypot(wrist_r, dz)
    lo = abs(L1_SHOULDER_TO_ELBOW_MM - L2_ELBOW_TO_WRIST_MM)
    hi = L1_SHOULDER_TO_ELBOW_MM + L2_ELBOW_TO_WRIST_MM
    if reach < lo - _EPS or reach > hi + _EPS:
        raise Unreachable(
            f"target ({x:.1f}, {y:.1f}, {z:.1f}) mm → planar reach {reach:.1f} mm "
            f"outside [{lo:.1f}, {hi:.1f}] mm"
        )

    # Law of cosines for the 2-link arm; elbow-up means q3 ≤ 0.
    cos_q3 = (wrist_r**2 + dz**2 - L1_SHOULDER_TO_ELBOW_MM**2 - L2_ELBOW_TO_WRIST_MM**2) / (
        2 * L1_SHOULDER_TO_ELBOW_MM * L2_ELBOW_TO_WRIST_MM
    )
    cos_q3 = max(-1.0, min(1.0, cos_q3))  # guard float drift at the envelope edge
    q3 = -math.acos(cos_q3)

    q2 = math.atan2(dz, wrist_r) - math.atan2(
        L2_ELBOW_TO_WRIST_MM * math.sin(q3),
        L1_SHOULDER_TO_ELBOW_MM + L2_ELBOW_TO_WRIST_MM * math.cos(q3),
    )
    # Tool absolute heading must be -90° (straight down). With wrist_flex inverted
    # (see forward_kinematics) that is q2 + q3 - q4 = -pi/2, so:
    q4 = (q2 + q3) + math.pi / 2

    q1_deg = math.degrees(q1)
    q5_deg = 0.0 if axis_deg is None else _wrist_roll_for_axis(float(axis_deg), q1_deg)

    return {
        "1": q1_deg,
        "2": math.degrees(q2),
        "3": math.degrees(q3),
        "4": math.degrees(q4),
        "5": q5_deg,
        "6": GRIPPER_OPEN,
    }


def in_limits(
    joints: dict[str, float],
    limits: dict[str, tuple[float, float]] | None = None,
) -> bool:
    """True iff every joint in ``joints`` is within ``limits`` (default: executor's).

    Lets a caller decide what to do with a raw IK solution that exceeds the
    conservative skeleton limits, instead of silently clamping it.
    """
    bounds = limits or DEFAULT_JOINT_LIMITS
    for j in JOINTS:
        if j not in joints:
            continue
        lo, hi = bounds[j]
        if not (lo <= joints[j] <= hi):
            return False
    return True
