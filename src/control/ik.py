"""SO-101 analytic inverse kinematics — table (X, Y, Z) mm → joint targets (deg).

The missing ⬜ of the loop seam (progress.md Part E): homography gives a table
point in mm, this turns it into servo angles. Deliberately the *simplest* solver
that closes the loop — a base-pan + planar 2-link reach, gripper held vertical —
not a full 6-DOF numeric IK. Pure math (no SDK / robot / network), so the whole
FK∘IK round-trip is unit-testable offline.

Frame (ASSUMED, see TODO(calibration)): the table (X, Y) handed in is already in
the **robot base frame** — origin under the base pivot, +X forward, +Y left, +Z
up, angles per calibration. The real camera→base alignment is a future
calibration offset that will pre-transform (X, Y) before it reaches here.

Model:
- ``q1`` (shoulder_pan) rotates the whole arm about +Z: ``q1 = atan2(y, x)``.
- In the resulting vertical plane, ``(r=hypot(x,y), z)`` is reached by a 2-link
  arm ``L1`` (shoulder→elbow) + ``L2`` (elbow→wrist), law-of-cosines, **elbow-up**
  branch (``q3 ≤ 0``, matching the sign of the old hardcoded poses).
- The gripper is held pointing **straight down**; ``q4`` (wrist_flex) is derived
  from ``q2, q3`` so the tool axis is vertical, and the tip sits ``L3`` below the
  wrist. ``q5`` (wrist_roll) = 0, ``q6`` (gripper) = open.

Angles in DEGREES, lengths in mm. Sign/offset conventions are internally
consistent so that ``forward_kinematics(solve_ik(x, y, z)) ≈ (x, y, z)`` for any
reachable target — that round-trip is THE correctness invariant.

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
# TODO(verify): measure on the real SO-101 — these are PLACEHOLDER values in the
# SO-ARM100 ballpark, NOT measured. Exact lengths asked in questions-discord.md.
# Kept as module constants so real numbers slot straight in.
L1_SHOULDER_TO_ELBOW_MM: float = 116.0  # shoulder pivot → elbow pivot
L2_ELBOW_TO_WRIST_MM: float = 135.0     # elbow pivot → wrist pivot
L3_WRIST_TO_TIP_MM: float = 100.0       # wrist pivot → gripper tip (tool length)
BASE_HEIGHT_MM: float = 120.0           # table plane → shoulder pivot (base height)

# Default tip height for a table pick: gripper tip at the table surface (Z=0).
PICK_Z: float = 0.0

# Gripper open angle — a local mirror of agent_service.poses.GRIPPER_OPEN, kept
# here to avoid a control→agent_service import cycle. Keep the two in sync.
GRIPPER_OPEN: float = -40.0

_EPS = 1e-6  # mm slack on the reach envelope / float guard on acos


class Unreachable(ValueError):
    """Target lies outside the 2-link reach envelope (too far or too near)."""


def forward_kinematics(joints: dict[str, float]) -> tuple[float, float, float]:
    """Gripper-tip (X, Y, Z) in table/base mm from joint angles (degrees).

    Uses joints "1".."4" (pan, shoulder_lift, elbow_flex, wrist_flex); "5"/"6"
    (roll/gripper) don't move the tip in this planar model. Inverse of
    :func:`solve_ik` — see the module round-trip invariant.
    """
    q1 = math.radians(joints["1"])
    q2 = math.radians(joints["2"])
    q3 = math.radians(joints["3"])
    q4 = math.radians(joints["4"])

    # Planar chain in the (radial, vertical) plane; angles are absolute-from-horizontal.
    a_link2 = q2 + q3          # link2 (forearm) heading
    a_tool = q2 + q3 + q4      # tool heading (‑90° ⇒ straight down)
    wrist_r = L1_SHOULDER_TO_ELBOW_MM * math.cos(q2) + L2_ELBOW_TO_WRIST_MM * math.cos(a_link2)
    wrist_z = (
        BASE_HEIGHT_MM
        + L1_SHOULDER_TO_ELBOW_MM * math.sin(q2)
        + L2_ELBOW_TO_WRIST_MM * math.sin(a_link2)
    )
    tip_r = wrist_r + L3_WRIST_TO_TIP_MM * math.cos(a_tool)
    tip_z = wrist_z + L3_WRIST_TO_TIP_MM * math.sin(a_tool)

    return tip_r * math.cos(q1), tip_r * math.sin(q1), tip_z


def solve_ik(x: float, y: float, z: float = PICK_Z) -> dict[str, float]:
    """Table/base target (mm) → joint targets in DEGREES, keyed "1".."6".

    Holds the gripper vertical (tip ``L3`` below the wrist), solves the 2-link
    reach on the **elbow-up** branch, and sets roll=0, gripper=open. Raises
    :class:`Unreachable` if the (radial, vertical) target is outside
    ``[|L1-L2|, L1+L2]`` (z is folded into this planar reach check). The result
    is NOT clamped to ``DEFAULT_JOINT_LIMITS`` — see the module docstring and
    :func:`in_limits`.
    """
    q1 = math.atan2(y, x)

    # Gripper vertical ⇒ tip is straight below the wrist by L3; solve for the wrist.
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

    # Law of cosines for the 2-link arm; elbow-up ⇒ q3 ≤ 0.
    cos_q3 = (wrist_r**2 + dz**2 - L1_SHOULDER_TO_ELBOW_MM**2 - L2_ELBOW_TO_WRIST_MM**2) / (
        2 * L1_SHOULDER_TO_ELBOW_MM * L2_ELBOW_TO_WRIST_MM
    )
    cos_q3 = max(-1.0, min(1.0, cos_q3))  # guard float drift at the envelope edge
    q3 = -math.acos(cos_q3)

    q2 = math.atan2(dz, wrist_r) - math.atan2(
        L2_ELBOW_TO_WRIST_MM * math.sin(q3),
        L1_SHOULDER_TO_ELBOW_MM + L2_ELBOW_TO_WRIST_MM * math.cos(q3),
    )
    # Tool absolute heading must be ‑90° (straight down): q2+q3+q4 = -pi/2.
    q4 = -math.pi / 2 - (q2 + q3)

    return {
        "1": math.degrees(q1),
        "2": math.degrees(q2),
        "3": math.degrees(q3),
        "4": math.degrees(q4),
        "5": 0.0,
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
