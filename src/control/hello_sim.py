"""Sim 'hello world': move the SO-101 twin in SIMULATION via the motion executor.

This is the walking-skeleton smoke test for the control layer. It connects to
Cyberwave, pins the runtime to **simulation** (``cw.affect("simulation")``), fetches
our calibrated twin by ``CYBERWAVE_TWIN_ID``, and runs a short, clamped, ramped
demo plan through :class:`MotionExecutor`. Watch the digital twin move in the
dashboard — no servos are written.

SAFETY (CLAUDE.md rule 2): this script is **simulation-only by construction** —
it hard-codes ``affect("simulation")`` and has no live path. ``--dry-run`` goes
further and issues no SDK calls at all (prints the plan and exits). Live hardware
motion is a separate, explicitly-confirmed step and is intentionally not wired here.

Usage:
    python -m src.control.hello_sim            # move the twin in simulation
    python -m src.control.hello_sim --dry-run  # print the plan, no connection/motion
"""

from __future__ import annotations

import argparse
import os
import sys

from .motion import MotionExecutor, MotionPlan
from .sim_session import load_env, open_sim_executor

# A gentle, obviously-visible demo: pan the base, nod the shoulder, wave, home.
# Every angle is well inside DEFAULT_JOINT_LIMITS; the executor clamps anyway.
DEMO_PLAN = MotionPlan.from_dict(
    {
        "say": "Hello from simulation — SO-101 walking skeleton.",
        "actions": [
            {"type": "home", "duration": 1.0},
            {"type": "set_joint", "joint": "1", "angle": 30, "duration": 1.0},
            {"type": "set_joint", "joint": "1", "angle": -30, "duration": 1.5},
            {"type": "set_pose", "pose": {"1": 0, "2": 20}, "duration": 1.0},
            {"type": "set_joint", "joint": "4", "angle": 25, "duration": 0.7},
            {"type": "set_joint", "joint": "4", "angle": -25, "duration": 0.7},
            {"type": "home", "duration": 1.5},
        ],
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SO-101 sim hello-world (simulation only).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without connecting or issuing any SDK call.",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=3.0,
        help="Seconds to wait after the warm-up command so the controller finishes attaching.",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Move to one bold, static pose and STAY there (skip the gesture + final home). "
        "Use to visually confirm the twin moved in the Simulate view.",
    )
    args = parser.parse_args(argv)

    load_env()

    if args.dry_run:
        # No network, no SDK: exercise the executor against a no-op robot.
        class _NullJoints:
            def set(self, *a, **k):  # noqa: D401,ANN002,ANN003
                return None

        class _NullRobot:
            joints = _NullJoints()

        print("[hello_sim] DRY RUN — no connection, no motion.\n")
        MotionExecutor(_NullRobot(), dry_run=True).execute(DEMO_PLAN)
        return 0

    twin_id = os.getenv("CYBERWAVE_TWIN_ID")
    if not twin_id:
        print("ERROR: CYBERWAVE_TWIN_ID is not set (see .env).", file=sys.stderr)
        return 2

    # Connect + warm up via the shared sim path (see control.sim_session).
    executor, _robot = open_sim_executor(twin_id, settle=args.settle)
    print()

    if args.hold:
        hold_plan = MotionPlan.from_dict(
            {
                "say": "Holding a bold pose — the twin should sit here in the Simulate view.",
                "actions": [{"type": "set_pose", "pose": {"1": 60, "2": 40, "4": 40}, "duration": 2.0}],
            }
        )
        executor.execute(hold_plan)
        print(f"\n[hello_sim] HOLDING at {executor._format_pose()} — not homing. Check the Simulate view.")
        return 0

    try:
        executor.execute(DEMO_PLAN)
    finally:
        # Always leave the (simulated) arm at home.
        print("[hello_sim] homing …")
        executor.home(duration=1.0)

    print("\n[hello_sim] done. Watch the twin replay in the dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
