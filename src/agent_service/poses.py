"""Hardcoded pick/place poses — the perception-free stand-in for the walking skeleton.

⚠ THROWAWAY by design. These joint targets are *not* real grasps: they're plausible,
in-limits poses so the loop moves the arm to a distinct spot per item/bin and back.
They exist only so order→pick→place→alert runs end-to-end **before** perception.

They get deleted the moment the overhead camera is mounted: the pick pose becomes
VLM-detected pixel → planar homography → IK, and the place pose becomes the bin's
surveyed location (roadmap.md / decisions.md D9). Do not tune these for accuracy.

Poses are dicts of canonical joint key ("1".."6") → angle°, consumed by
control.MotionExecutor (which clamps every value again at the SDK boundary).

``pick_plan_from_table`` is the camera-gated successor to the hardcoded ``PICK_POSES``:
it takes a table (X, Y) mm (from detect→homography) and solves IK for the pose. Once
the overhead camera + real homography calibration are live, ``PICK_POSES`` /
``pick_plan`` become obsolete and this becomes the only pick path.

:class:`PickChoice` is the seam the loop resolves a pick through: it carries the plan
*plus where it came from* ("hardcoded" vs "perceived"), so a run can prove which path it
actually took — the point of ``run_order --perceive`` (see run_order._perceiving_resolver).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.control import MotionPlan, solve_ik

# Gripper (joint "6") symbolic open/close angles, in the follower's own calibrated
# 0°→128.9° span (hardware/config/joint-ranges.md) and inside DEFAULT_JOINT_LIMITS["6"].
#
# MEASURED ON HARDWARE 2026-08-11 — these were −40°/+20°, guesses that predate any
# gripper calibration and put "open" *outside* the servo's range entirely:
#   * jaws pushed shut by hand read 0.107 rad = 6.1° on the encoder → LOW = SHUT;
#   * a commanded sweep 110° → 60° → 10° → 110° was observed as fully open → about
#     half closed → almost fully shut → fully open again.
# So the axis is unambiguous: HIGH = OPEN, LOW = SHUT, and the old −40° "open" would
# have driven hard into (or been clamped to) the shut end.
#
# 10° sits just above the 6.1° hard-shut reading, so a grasp squeezes rather than stalls
# against the stop; 105° is comfortably open without reaching the 128.9° mechanical end.
# ⚠ One constant angle for EVERY object — grasp force/width are not modelled at all
# (see hardware/config/joint-ranges.md). Keep in sync with control.ik.GRIPPER_OPEN.
GRIPPER_OPEN = 105.0
GRIPPER_CLOSE = 10.0

# item → arm pose hovering "over" that item, gripper open. Placeholder geometry.
PICK_POSES: dict[str, dict[str, float]] = {
    "red marker": {"1": 30, "2": 25, "3": -20, "4": 15, "6": GRIPPER_OPEN},
    "blue marker": {"1": 15, "2": 30, "3": -25, "4": 20, "6": GRIPPER_OPEN},
    "eraser": {"1": -20, "2": 20, "3": -15, "4": 10, "6": GRIPPER_OPEN},
}

# bin → arm pose hovering "over" that drop bin. Placeholder geometry.
PLACE_POSES: dict[str, dict[str, float]] = {
    "A": {"1": -60, "2": 30, "3": -20, "4": 15},
    "B": {"1": 60, "2": 30, "3": -20, "4": 15},
}


class UnknownItem(KeyError):
    """No hardcoded pick pose for the ordered item."""


class UnknownBin(KeyError):
    """No hardcoded place pose for the ordered bin."""


def pick_plan(item: str) -> MotionPlan:
    """Approach the item (gripper open) then close the gripper to grasp it."""
    pose = PICK_POSES.get(item)
    if pose is None:
        raise UnknownItem(f"no pick pose for item {item!r}; known: {sorted(PICK_POSES)}")
    return MotionPlan.from_dict(
        {
            "say": f"picking {item}",
            "actions": [
                {"type": "home", "duration": 1.0},
                {"type": "set_pose", "pose": pose, "duration": 1.5},
                {"type": "set_joint", "joint": "6", "angle": GRIPPER_CLOSE, "duration": 0.6},
            ],
        }
    )


def pick_plan_from_table(
    x_mm: float,
    y_mm: float,
    axis_deg: float | None = None,
) -> MotionPlan:
    """IK-driven pick at a table (X, Y) mm — the camera-gated successor to ``pick_plan``.

    Solves IK for the target (gripper vertical, open) then closes to grasp; same
    shape as ``pick_plan`` but the pose comes from detect→homography→IK rather than
    a hardcoded lookup. Raises ``control.Unreachable`` if the target is out of reach.

    ``axis_deg`` is the object's long axis on the table (perception.orient): given one,
    the wrist rolls so the jaws close across it instead of along it. ``None`` ⇒ the
    previous fixed-roll grasp. Raises ``control.GraspAngleUnreachable`` if the wrist
    cannot supply that roll.
    """
    joints = solve_ik(x_mm, y_mm, axis_deg=axis_deg)
    return MotionPlan.from_dict(
        {
            "say": f"picking at table ({x_mm:.0f}, {y_mm:.0f}) mm",
            "actions": [
                {"type": "home", "duration": 1.0},
                {"type": "set_pose", "pose": joints, "duration": 1.5},
                {"type": "set_joint", "joint": "6", "angle": GRIPPER_CLOSE, "duration": 0.6},
            ],
        }
    )


# ------------------------------------------------------- the pick-target seam
HARDCODED = "hardcoded"
PERCEIVED = "perceived"


@dataclass(frozen=True)
class PickChoice:
    """A pick plan **plus its provenance** — what the loop records about the grasp.

    ``source`` is :data:`HARDCODED` (the throwaway pose table) or :data:`PERCEIVED`
    (VLM → homography → IK). The extra fields are only set on the perceived path and
    exist for logs/alerts/demo narration, not for control. ``axis_deg`` records the
    measured object orientation the wrist roll was derived from (``None`` = none
    measured, fixed-roll grasp) — the provenance that explains a rotated wrist.
    """

    plan: MotionPlan
    source: str = HARDCODED
    table_mm: tuple[float, float] | None = None
    pixel: tuple[float, float] | None = None
    frame: str | None = None
    axis_deg: float | None = None

    @property
    def perceived(self) -> bool:
        return self.source == PERCEIVED

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe provenance (no MotionPlan — that's the executor's business)."""
        return {
            "source": self.source,
            "table_mm": list(self.table_mm) if self.table_mm else None,
            "pixel": list(self.pixel) if self.pixel else None,
            "frame": self.frame,
            "axis_deg": self.axis_deg,
        }


def pick_choice(item: str) -> PickChoice:
    """The default (unchanged) pick: hardcoded pose table. Raises ``UnknownItem``."""
    return PickChoice(pick_plan(item), HARDCODED)


def pick_choice_from_table(
    x_mm: float,
    y_mm: float,
    *,
    pixel: tuple[float, float] | None = None,
    frame: str | None = None,
    axis_deg: float | None = None,
) -> PickChoice:
    """The perceived pick: IK at a detected table (X, Y) mm.

    Raises ``Unreachable`` (target out of reach) or ``GraspAngleUnreachable``
    (``axis_deg`` given but the wrist cannot roll that far).
    """
    return PickChoice(
        pick_plan_from_table(x_mm, y_mm, axis_deg),
        PERCEIVED,
        table_mm=(float(x_mm), float(y_mm)),
        pixel=pixel,
        frame=frame,
        axis_deg=float(axis_deg) if axis_deg is not None else None,
    )


def place_plan(bin_: str) -> MotionPlan:
    """Carry to the bin, release (gripper open), then return home."""
    pose = PLACE_POSES.get(bin_)
    if pose is None:
        raise UnknownBin(f"no place pose for bin {bin_!r}; known: {sorted(PLACE_POSES)}")
    return MotionPlan.from_dict(
        {
            "say": f"placing in bin {bin_}",
            "actions": [
                {"type": "set_pose", "pose": pose, "duration": 1.5},
                {"type": "set_joint", "joint": "6", "angle": GRIPPER_OPEN, "duration": 0.6},
                {"type": "home", "duration": 1.5},
            ],
        }
    )
