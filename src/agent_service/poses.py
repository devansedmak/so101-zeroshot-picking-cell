"""Hardcoded pick/place poses — the perception-free stand-in for the walking skeleton.

⚠ THROWAWAY by design. These joint targets are *not* real grasps: they're plausible,
in-limits poses so the loop moves the arm to a distinct spot per item/bin and back.
They exist only so order→pick→place→alert runs end-to-end **before** perception.

They get deleted the moment the overhead camera is mounted: the pick pose becomes
VLM-detected pixel → planar homography → IK, and the place pose becomes the bin's
surveyed location (roadmap.md / decisions.md D9). Do not tune these for accuracy.

Poses are dicts of canonical joint key ("1".."6") → angle°, consumed by
control.MotionExecutor (which clamps every value again at the SDK boundary).
"""

from __future__ import annotations

from src.control import MotionPlan

# Gripper (joint "6") symbolic open/close angles, well inside DEFAULT_JOINT_LIMITS.
GRIPPER_OPEN = -40.0
GRIPPER_CLOSE = 20.0

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
