"""SO-101 motion executor: validated plans -> clamped, ramped joint commands.

Safety layer between a planner (LLM / zero-shot pipeline) and the raw SDK. The
executor never commands a target in one step: each action is split into
``duration × ramp_hz`` linear-interpolation steps so the arm sweeps smoothly and
the gearbox is spared. Every commanded angle is clamped to a conservative
per-joint limit *right before* the SDK call, so an out-of-range request is
physically impossible to issue.

Ported from the Cyberwave reference example ``nl_arm_controller/motion.py`` and adapted to the
SO-101: canonical joint keys are the servo IDs "1".."6" (the platform accepts
these directly, e.g. ``robot.joints.set("1", 10, degrees=True)``); JOINT_LABELS
maps them to the semantic names used during calibration.

Design notes:
- Pure logic + a duck-typed ``robot`` (anything exposing ``joints.set``), so the
  whole module is unit-testable with a fake robot and no hardware/network.
- ``dry_run=True`` tracks the in-memory pose without issuing any SDK call.
- Limits here are DEGREES and deliberately conservative for the walking skeleton;
  tighten/replace from real calibration once live teleop is validated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

JointName = str  # canonical SO-101 keys: servo IDs "1".."6"

ActionType = Literal["set_joint", "set_pose", "wait", "home"]
ALLOWED_ACTION_TYPES: set[str] = {"set_joint", "set_pose", "wait", "home"}

# SO-101 servo IDs (canonical control keys) and their calibration-time names.
JOINTS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")
JOINT_LABELS: dict[str, str] = {
    "1": "shoulder_pan",
    "2": "shoulder_lift",
    "3": "elbow_flex",
    "4": "wrist_flex",
    "5": "wrist_roll",
    "6": "gripper",
}

# Degree limits, derived 2026-08-10 from OUR twin's own guided calibration
# (`joint_calibration.follower` on twin aa1dd0ad-...), keeping a ~5° margin inside the
# measured mechanical stops. Full table + provenance: hardware/config/joint-ranges.md.
#
# These were hand-guessed (±60° everywhere) until a perceived pick in the middle of the
# table solved to elbow = -83° and got silently clamped to -60°, landing the tip short of
# the object. The arm reaches ±96.6° there; only our guess said otherwise. Clamping is
# still always on; it just no longer clamps away real, calibrated reach.
#
# wrist_roll (5) is ±90°, not the measured ±179.5°. It was ±60° while a top-down pick never
# needed roll; oriented grasping (ik.solve_ik(axis_deg=...)) needs it, because the jaws must
# close ACROSS the object's long axis. ±90° is the *exact* amount required: the parallel jaw
# is symmetric under a 180° roll, so every object orientation folds into [-90, +90), which
# means this range has no dead wedge at all, while still staying far from the encoder
# 0↔4095 seam near ±180° that motivated the original narrowing (calibration-notes.md).
#
# gripper (6) was the last shaky guess (±60°, convention unknown) and is now MEASURED, on
# hardware 2026-08-11: jaws shut by hand read 6.1°, and a 110 -> 60 -> 10 -> 110° sweep read as
# open -> half closed -> nearly shut -> open. So LOW = SHUT, HIGH = OPEN on the follower's own
# calibrated 0 to 128.9° span, and the old range was doubly wrong: it admitted negative angles
# the servo does not have, and capped "open" at 60° (half closed). 124° keeps the usual ~5°
# margin off the mechanical end. Details: hardware/config/joint-ranges.md §gripper.
DEFAULT_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "1": (-113, 113),  # measured ±118.9
    "2": (-97, 97),    # measured ±102.8
    "3": (-91, 91),    # measured ±96.6
    "4": (-96, 96),    # measured ±101.6
    "5": (-90, 90),    # measured ±179.5, still narrowed (encoder seam), see above
    "6": (0, 124),     # measured 0 to 128.9 (0 = shut, 124 = open)
}

MAX_DURATION_S: float = 5.0
MAX_ACTIONS_PER_PLAN: int = 8
DEFAULT_DURATION_S: float = 1.0
RAMP_HZ: int = 20


class _RobotJoints(Protocol):
    def set(self, joint_name: str, position: float, degrees: bool = True) -> Any: ...


class _Robot(Protocol):
    @property
    def joints(self) -> _RobotJoints: ...


@dataclass
class Action:
    """One step of a motion plan.

    ``type`` determines which fields matter:
      * ``set_joint`` -> joint, angle, [duration]
      * ``set_pose``  -> pose,  [duration]
      * ``wait``      -> duration
      * ``home``      -> [duration]
    """

    type: ActionType
    joint: str | None = None
    angle: float | None = None
    pose: dict[str, float] | None = None
    duration: float = DEFAULT_DURATION_S

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        return cls(
            type=data["type"],
            joint=data.get("joint"),
            angle=data.get("angle"),
            pose=data.get("pose"),
            duration=float(data.get("duration", DEFAULT_DURATION_S)),
        )


@dataclass
class MotionPlan:
    """A short, validated sequence of motion actions."""

    say: str = ""
    actions: list[Action] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionPlan":
        return cls(
            say=str(data.get("say", "")),
            actions=[Action.from_dict(a) for a in data.get("actions", [])],
        )


def clamp(
    joint: str,
    angle: float,
    limits: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Clamp ``angle`` to the safe range for ``joint``."""
    bounds = (limits or DEFAULT_JOINT_LIMITS).get(joint, (-60.0, 60.0))
    return max(bounds[0], min(bounds[1], float(angle)))


def validate_plan(plan: MotionPlan) -> list[str]:
    """Return a list of human-readable error strings; empty list = valid.

    All-or-nothing by contract: the caller must refuse to execute a plan with any
    error (the executor raises on a non-empty result). Clamping is a separate,
    always-on safety net applied per command.
    """
    errors: list[str] = []

    if len(plan.actions) > MAX_ACTIONS_PER_PLAN:
        errors.append(f"too many actions ({len(plan.actions)} > {MAX_ACTIONS_PER_PLAN})")

    for i, a in enumerate(plan.actions):
        prefix = f"action[{i}] ({a.type})"

        if a.type not in ALLOWED_ACTION_TYPES:
            errors.append(f"{prefix}: unknown type, must be one of {ALLOWED_ACTION_TYPES}")
            continue

        if a.duration < 0:
            errors.append(f"{prefix}: duration must be ≥ 0, got {a.duration}")
        if a.duration > MAX_DURATION_S:
            errors.append(
                f"{prefix}: duration {a.duration}s exceeds MAX_DURATION_S ({MAX_DURATION_S}s)"
            )

        if a.type == "set_joint":
            if a.joint not in DEFAULT_JOINT_LIMITS:
                errors.append(f"{prefix}: unknown joint {a.joint!r}, expected one of {JOINTS}")
            if a.angle is None:
                errors.append(f"{prefix}: missing 'angle'")

        elif a.type == "set_pose":
            if not a.pose:
                errors.append(f"{prefix}: missing or empty 'pose'")
            else:
                for j in a.pose:
                    if j not in DEFAULT_JOINT_LIMITS:
                        errors.append(f"{prefix}: pose contains unknown joint {j!r}")

    return errors


class MotionExecutor:
    """Runs validated MotionPlans on a Cyberwave robot twin.

    Tracks an in-memory ``_current_pose`` so it can interpolate from the last
    commanded pose, independent of the robot's actual physical state. Pair with
    ``cw.affect("simulation")`` to develop against the twin without moving the
    real arm; live execution needs explicit confirmation.
    """

    def __init__(
        self,
        robot: _Robot,
        *,
        joint_limits: dict[str, tuple[float, float]] | None = None,
        name_map: dict[str, str] | None = None,
        ramp_hz: int = RAMP_HZ,
        dry_run: bool = False,
    ) -> None:
        self.robot = robot
        self.joint_limits = joint_limits or DEFAULT_JOINT_LIMITS
        # Canonical key ("1".."6") -> the twin's actual SDK joint name. Some SO-101
        # assets expose "_1".."_6"; validation/clamping stay on canonical keys and
        # we translate only at the SDK boundary. Default identity.
        self.name_map = name_map or {}
        self.ramp_hz = ramp_hz
        self.dry_run = dry_run
        self._current_pose: dict[str, float] = {j: 0.0 for j in JOINTS}

    def home(self, duration: float = 1.5) -> None:
        """Convenience: ramp every joint back to 0°."""
        self._ramp_to({j: 0.0 for j in JOINTS}, duration)

    def execute(self, plan: MotionPlan) -> None:
        """Validate, log, then execute every action in ``plan``."""
        errors = validate_plan(plan)
        if errors:
            raise ValueError("invalid plan:\n  - " + "\n  - ".join(errors))

        if plan.say:
            print(f"  💬  {plan.say}")

        for i, action in enumerate(plan.actions, 1):
            print(f"  ▶  [{i}/{len(plan.actions)}] {self._describe(action)}", flush=True)
            self._run(action)

        print(f"  ✅  plan complete  (final pose: {self._format_pose()})")

    def _describe(self, a: Action) -> str:
        if a.type == "wait":
            return f"wait {a.duration:.2f}s"
        if a.type == "home":
            return f"home over {a.duration:.2f}s"
        if a.type == "set_joint":
            clamped = clamp(a.joint or "", a.angle or 0.0, self.joint_limits)
            return f"joint {a.joint} ({JOINT_LABELS.get(a.joint or '', '?')}) → {clamped:+.1f}° over {a.duration:.2f}s"
        if a.type == "set_pose":
            parts = ", ".join(
                f"{j}={clamp(j, v, self.joint_limits):+.1f}°" for j, v in (a.pose or {}).items()
            )
            return f"pose {{{parts}}} over {a.duration:.2f}s"
        return f"<unknown:{a.type}>"

    def _run(self, action: Action) -> None:
        if action.type == "wait":
            time.sleep(action.duration)
            return

        if action.type == "home":
            self._ramp_to({j: 0.0 for j in JOINTS}, action.duration)
            return

        if action.type == "set_joint":
            assert action.joint is not None and action.angle is not None
            target = clamp(action.joint, action.angle, self.joint_limits)
            self._ramp_to({action.joint: target}, action.duration)
            return

        if action.type == "set_pose":
            assert action.pose is not None
            self._ramp_to(
                {j: clamp(j, v, self.joint_limits) for j, v in action.pose.items()},
                action.duration,
            )
            return

        raise ValueError(f"unknown action type: {action.type}")

    def _ramp_to(self, target_pose: dict[str, float], duration: float) -> None:
        """Interpolate the joints **named in ``target_pose``** from current -> target.

        NOTE: the key set of ``target_pose`` is the *command scope*: a joint that is not a
        key here is never written to, at any step. Callers must therefore pass only the
        joints they mean to move. ``home`` legitimately names all six, ``set_pose``
        names exactly the keys the plan listed.

        This is a safety property, not a micro-optimisation. It used to merge the action
        into ``_current_pose`` and ramp all six every time, which meant an action that
        said nothing about the gripper still commanded it, to ``_current_pose``'s
        initial 0°, and 0° is the SHUT end of the follower's calibrated 0 to 128.9° gripper
        span (hardware/config/joint-ranges.md). A fresh executor's very first ``set_pose``
        therefore closed the jaws on whatever was between them; observed on hardware
        2026-08-11 via ``tools/live_check.py``, whose pose deliberately omitted joint 6.

        Known residual: ``_current_pose`` is only the last *commanded* pose, seeded with
        0° for joints this executor has never written. The ramp for a joint's first
        command therefore starts from an assumed 0°, not the arm's true angle. It only
        affects the path, never the endpoint, but it means the first gripper command of
        a session still passes near the shut end on its way out. Fixing it properly needs
        a read-back of the servos' actual positions at construction time.
        """
        if duration <= 0:
            self._snap_to(target_pose)
            return

        steps = max(2, int(duration * self.ramp_hz))
        start = dict(self._current_pose)
        dt = duration / steps

        for s in range(1, steps + 1):
            t = s / steps
            interp = {
                j: start.get(j, 0.0) + (target_pose[j] - start.get(j, 0.0)) * t
                for j in target_pose
            }
            self._snap_to(interp)
            time.sleep(dt)

    def _snap_to(self, pose: dict[str, float]) -> None:
        for joint, angle in pose.items():
            if not self.dry_run:
                self.robot.joints.set(self.name_map.get(joint, joint), angle, degrees=True)
            self._current_pose[joint] = angle

    def _format_pose(self) -> str:
        return ", ".join(f"{j}={self._current_pose[j]:+.1f}°" for j in JOINTS)
