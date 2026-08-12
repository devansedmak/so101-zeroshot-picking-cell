"""LIVE hardware bring-up checks for the SO-101 follower. RUN THIS YOURSELF.

Three deliberately tiny steps, each a separate invocation, in this order:

    # 1. connect + read state. MOVES NOTHING.
    .venv/bin/python tools/live_check.py read

    # 2. gripper ONLY: open, pause, close, pause, re-open. No arm travel.
    .venv/bin/python tools/live_check.py gripper --yes

    # 3. hover the tip over a table point (default: the printed mark at 190,40),
    #    5 cm above the table so a frame error cannot press into the surface.
    .venv/bin/python tools/live_check.py hover --yes
    .venv/bin/python tools/live_check.py hover --x 190 --y 40 --z 50 --yes

SAFETY: every motion subcommand refuses to run without ``--yes``,
prints exactly what it is about to do, and pauses for a typed confirmation. Steps are
never batched: one invocation, one small motion. Keep a hand near the follower's PSU
switch. If anything looks wrong, Ctrl-C and cut power.

Why hover at z=+50 by default: the kinematic frame is not yet validated against the
real servo zeros, so the tip may land tens of mm from where the model believes. A
5 cm standoff turns a frame error into a visible offset instead of a table collision.
The point of the test is precisely to *measure* that offset: capture an overhead frame
while the arm holds the pose and compare the tip against the printed mark it is
supposed to be above.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.ik import Unreachable, forward_kinematics, in_limits, solve_ik  # noqa: E402
from src.control.live_session import (  # noqa: E402
    canonical_joints,
    open_live_executor,
    read_joints,
    sync_executor_pose,
    verify_pose,
)
from src.control.motion import (  # noqa: E402
    DEFAULT_JOINT_LIMITS,
    JOINT_LABELS,
    Action,
    MotionPlan,
)
from src.control.sim_session import load_env  # noqa: E402

# This probe ANSWERED the gripper-direction question on 2026-08-11: the sweep below was
# observed as fully open (110°) -> half closed (60°) -> almost shut (10°) -> fully open, and
# the jaws pushed shut by hand read 6.1° on the encoder. HIGH = OPEN, LOW = SHUT, on the
# follower's calibrated 0° to 128.9° span. poses.GRIPPER_OPEN/CLOSE are now 105°/10° and
# DEFAULT_JOINT_LIMITS["6"] is 0 to 124 (hardware/config/joint-ranges.md §gripper), so this
# probe is kept only as a re-verification tool after a re-calibration or a re-pair.
GRIPPER_LIMITS = (0.0, 124.0)
GRIPPER_PROBE_ANGLES = (110.0, 60.0, 10.0, 110.0)


def _twin_id() -> str:
    load_env()
    twin = os.environ.get("CYBERWAVE_TWIN_ID", "").strip()
    if not twin:
        sys.exit("CYBERWAVE_TWIN_ID is not set (see .env)")
    return twin


def _confirm(what: str, *, yes: bool) -> None:
    """Refuse to move real hardware without --yes AND a typed confirmation."""
    print("\n⚠  ABOUT TO MOVE THE REAL ROBOT")
    print(f"   {what}")
    if not yes:
        sys.exit("\nrefused: re-run with --yes once you have read the above.")
    reply = input("\n   type 'go' to execute (anything else aborts): ").strip().lower()
    if reply != "go":
        sys.exit("aborted by operator.")


def cmd_read(args: argparse.Namespace) -> None:
    executor, robot = open_live_executor(
        _twin_id(), settle=0.0, i_understand_this_moves_real_hardware=True
    )
    print(f"[live] executor ready (joint map: {executor.name_map or 'identity'})")
    state = read_joints(robot)
    if state:
        print("[live] current joint angles:")
        for name, value in state.items():
            print(f"        {name:>4} = {value:+8.2f}")
    else:
        print(
            "[live] no joint state returned.\n"
            "        Usually means the follower's servos are unpowered (USB gives logic\n"
            "        power, the 12V/8A PSU gives servo power) or telemetry has not started.\n"
            "        Check the PSU before running any motion step."
        )


def cmd_gripper(args: argparse.Namespace) -> None:
    steps = " → ".join(f"{a:.0f}°" for a in GRIPPER_PROBE_ANGLES)
    _confirm(
        f"Gripper ONLY (joint 6). No arm travel. Sweep {steps}, pausing 2s at each.\n"
        f"   Angles stay inside the CALIBRATED range {GRIPPER_LIMITS[0]:.0f}–{GRIPPER_LIMITS[1]:.0f}°.\n"
        "   NOTHING must be in the jaws — we do not yet know which direction closes.\n"
        "   Watch and note: (a) does a HIGHER angle open or close the jaws?\n"
        "                   (b) at which angle are the jaws fully open / fully shut?",
        yes=args.yes,
    )
    # Joint 6 only: widen its bound to the calibrated span for this probe. Every other
    # joint keeps the normal conservative limit, and nothing else is commanded anyway.
    limits = dict(DEFAULT_JOINT_LIMITS)
    limits["6"] = GRIPPER_LIMITS

    executor, _ = open_live_executor(
        _twin_id(), i_understand_this_moves_real_hardware=True
    )
    executor.joint_limits = limits

    actions: list[Action] = []
    for angle in GRIPPER_PROBE_ANGLES:
        actions.append(Action("set_joint", joint="6", angle=angle, duration=1.2))
        actions.append(Action("wait", duration=2.0))
    executor.execute(MotionPlan(say="gripper convention probe", actions=actions))
    print(
        "\nReport back: which angle was OPEN and which was SHUT. Those two numbers\n"
        "replace GRIPPER_OPEN / GRIPPER_CLOSE in src/agent_service/poses.py, and\n"
        "close the open risk recorded in hardware/config/joint-ranges.md."
    )


def cmd_hover(args: argparse.Namespace) -> None:
    try:
        joints = solve_ik(args.x, args.y, args.z)
    except Unreachable as e:
        sys.exit(
            f"\nIK says unreachable, so NOTHING was sent to the robot:\n  {e}\n\n"
            "This is the safe failure: an out-of-envelope target can never become a\n"
            "motion command. Move the target closer to the base and retry."
        )

    tip = forward_kinematics(joints)
    out_of_range = [
        j
        for j, v in joints.items()
        if j in DEFAULT_JOINT_LIMITS and not (DEFAULT_JOINT_LIMITS[j][0] <= v <= DEFAULT_JOINT_LIMITS[j][1])
    ]
    print(f"\ntarget table point : ({args.x:.1f}, {args.y:.1f}, {args.z:.1f}) mm")
    print(f"FK check of solution: ({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f}) mm")
    print("joint solution:")
    for j in sorted(joints):
        flag = "  ← OUT OF LIMITS" if j in out_of_range else ""
        print(f"   {j} {JOINT_LABELS.get(j, ''):>13} = {joints[j]:+8.2f}°{flag}")

    if out_of_range:
        sys.exit(
            f"\nrefusing to move: joints {out_of_range} exceed the executor limits.\n"
            "The executor would clamp them, which would silently send the tip somewhere\n"
            "other than the requested point. Pick a different target."
        )
    if not in_limits(joints):
        sys.exit("\nrefusing to move: in_limits() rejected the solution.")

    _confirm(
        f"Move the arm to hover its gripper tip {args.z:.0f} mm above table point "
        f"({args.x:.0f}, {args.y:.0f}).\n"
        f"   Single ramped move over {args.duration:.1f}s. The gripper is NOT commanded —\n"
        "   the jaws stay exactly where you left them.\n"
        "   EXPECT the tip to be off by some mm — measuring that offset is the point.",
        yes=args.yes,
    )
    executor, robot = open_live_executor(
        _twin_id(), i_understand_this_moves_real_hardware=True
    )
    # Ramp from where the arm ACTUALLY is, not from the executor's assumed all-zeros;
    # otherwise every fresh session's first move is a step command, not a ramp.
    sync_executor_pose(executor, robot)
    # Drop joint 6 (gripper): solve_ik fills it with GRIPPER_OPEN, and this test is about
    # where the TIP lands, not about the jaws. Omitting it leaves them where the operator
    # left them, which the executor now actually honours: until 2026-08-11 it merged the
    # pose into its assumed all-zeros state and commanded joint 6 to 0° (= SHUT) anyway,
    # which is how this very call was observed closing the gripper. See motion._ramp_to.
    arm_pose = {j: v for j, v in joints.items() if j != "6"}
    executor.execute(
        MotionPlan(
            say=f"hover over ({args.x:.0f}, {args.y:.0f}) at z={args.z:.0f}mm",
            actions=[Action("set_pose", pose=arm_pose, duration=args.duration)],
        )
    )
    _report_drift(robot, executor, arm_pose)
    print(
        "\nNow capture an overhead frame WITHOUT moving the arm:\n"
        "  .venv/bin/python -m src.perception.capture --save /tmp/hover.jpg -w 12\n"
        "Then compare the tip against the printed mark it should be above."
    )


def cmd_home(args: argparse.Namespace) -> None:
    """Ease joints 1-5 back to 0°: the arm's stable extended-horizontal rest pose.

    Deliberately leaves joint 6 alone: 0° is the SHUT end of the gripper's calibrated
    0 to 128.9° span (confirmed 2026-08-11), so homing it would clamp the jaws on whatever
    they happen to be around, including an item mid-carry.
    """
    _confirm(
        f"Ease joints 1–5 back to 0° over {args.duration:.1f}s (arm extended, horizontal).\n"
        "   The gripper (joint 6) is NOT commanded.\n"
        "   Use this to leave a strained pose without cutting power — a powered-off\n"
        "   arm goes limp and DROPS onto the table.",
        yes=args.yes,
    )
    executor, robot = open_live_executor(
        _twin_id(), i_understand_this_moves_real_hardware=True
    )
    actual = sync_executor_pose(executor, robot)
    if actual:
        print(f"[live] synced from measured pose: { {k: round(v,1) for k,v in actual.items()} }")
    else:
        print("[live] ⚠ no telemetry — ramping from the assumed pose; expect a brisker move.")
    target = {j: 0.0 for j in ("1", "2", "3", "4", "5")}
    executor.execute(
        MotionPlan(say="return to rest", actions=[Action("set_pose", pose=target, duration=args.duration)])
    )
    _report_drift(robot, executor, target)


def cmd_joint(args: argparse.Namespace) -> None:
    """Move ONE joint to ONE angle: the probe for resolving a sign/zero convention."""
    lo, hi = DEFAULT_JOINT_LIMITS[args.joint]
    if not (lo <= args.angle <= hi):
        sys.exit(f"refusing: joint {args.joint} angle {args.angle}° is outside limits [{lo}, {hi}]")
    _confirm(
        f"Move ONLY joint {args.joint} ({JOINT_LABELS.get(args.joint, '?')}) to "
        f"{args.angle:+.1f}° over {args.duration:.1f}s. No other joint is commanded.",
        yes=args.yes,
    )
    executor, robot = open_live_executor(
        _twin_id(), i_understand_this_moves_real_hardware=True
    )
    sync_executor_pose(executor, robot)
    executor.execute(
        MotionPlan(
            say=f"single-joint probe: {args.joint} → {args.angle:+.1f}°",
            actions=[Action("set_joint", joint=args.joint, angle=args.angle, duration=args.duration)],
        )
    )
    _report_drift(robot, executor, {args.joint: args.angle})


def _report_drift(robot: object, executor: object, expected: dict[str, float]) -> None:
    """Read the arm back and say whether it actually went where it was told."""
    import time as _time

    _time.sleep(1.0)  # let the servos settle before believing the encoders
    try:
        drift = verify_pose(robot, expected, name_map=getattr(executor, "name_map", {}))
    except RuntimeError as e:
        print(f"\n⚠  CANNOT CONFIRM THE MOVE: {e}")
        return
    if not drift:
        print("\n✅  verified against servo encoders: every joint landed within 5°.")
        return
    print("\n⚠  COMMANDED ≠ MEASURED — the arm did not go where it was told:")
    for joint, error in sorted(drift.items()):
        print(f"      joint {joint} ({JOINT_LABELS.get(joint,'?'):>13}) off by {error:+.1f}°")
    print("   Causes: wrong sign/zero convention, a servo at its mechanical end, or a clamp.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("read", help="connect live and read joint state (moves nothing)").set_defaults(func=cmd_read)

    p_home = sub.add_parser("home", help="ease joints 1-5 back to 0 (safe recovery)")
    p_home.add_argument("--duration", type=float, default=4.0)
    p_home.add_argument("--yes", action="store_true", help="acknowledge real-hardware motion")
    p_home.set_defaults(func=cmd_home)

    p_joint = sub.add_parser("joint", help="move ONE joint (sign/zero convention probe)")
    p_joint.add_argument("joint", choices=sorted(DEFAULT_JOINT_LIMITS))
    p_joint.add_argument("angle", type=float)
    p_joint.add_argument("--duration", type=float, default=2.0)
    p_joint.add_argument("--yes", action="store_true", help="acknowledge real-hardware motion")
    p_joint.set_defaults(func=cmd_joint)

    p_grip = sub.add_parser("gripper", help="open/close the gripper only")
    p_grip.add_argument("--yes", action="store_true", help="acknowledge real-hardware motion")
    p_grip.set_defaults(func=cmd_gripper)

    p_hover = sub.add_parser("hover", help="hover the tip above a table point")
    p_hover.add_argument("--x", type=float, default=190.0, help="table X in mm (default: printed mark)")
    p_hover.add_argument("--y", type=float, default=40.0, help="table Y in mm (default: printed mark)")
    p_hover.add_argument("--z", type=float, default=50.0, help="tip height above the table in mm")
    p_hover.add_argument("--duration", type=float, default=3.0, help="ramp duration in seconds")
    p_hover.add_argument("--yes", action="store_true", help="acknowledge real-hardware motion")
    p_hover.set_defaults(func=cmd_hover)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
