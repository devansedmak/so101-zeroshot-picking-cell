"""Run ONE order end-to-end (order → pick → place → "fulfilled") — sim entrypoint.

The walking-skeleton demo: pull an order from the mock WMS, drive the hardcoded
pick/place poses through control.MotionExecutor, emit a fulfilled alert. Default is
**dry-run** (no connection, no motion); the actual SIMULATION run is behind ``--sim``
and reuses the proven connect/warm-up path (control.sim_session).

SAFETY (CLAUDE.md rule 2 / D9): SIMULATION-ONLY — ``--sim`` pins
``cw.affect("simulation")`` via sim_session; there is no live-hardware path here.

Usage:
    python -m src.agent_service.run_order                 # dry-run first demo order
    python -m src.agent_service.run_order --item "eraser" --bin A
    python -m src.agent_service.run_order --sim           # move the twin in SIMULATION
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from src.control import MotionExecutor
from src.control.sim_session import load_env, open_sim_executor
from src.wms_mock import Order, OrderSource

from .loop import fulfill_order


class _NullJoints:
    def set(self, *a: Any, **k: Any) -> None:  # noqa: ANN401
        return None


class _NullAlerts:
    def create(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        return {"stub": True, **kwargs}


class _NullRobot:
    """Offline stand-in: satisfies joints.set + alerts.create without a network."""

    joints = _NullJoints()
    alerts = _NullAlerts()


def _pick_order(args: argparse.Namespace) -> Order:
    if args.item and args.bin:
        return Order(item=args.item, bin=args.bin)
    return OrderSource().first()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one order end-to-end (simulation only).")
    parser.add_argument("--item", help="Override order item (with --bin).")
    parser.add_argument("--bin", help="Override order bin (with --item).")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Actually move the twin in SIMULATION (default is dry-run: no connection/motion).",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=3.0,
        help="Seconds to wait after warm-up so the controller finishes attaching (--sim only).",
    )
    args = parser.parse_args(argv)

    load_env()
    order = _pick_order(args)

    if not args.sim:
        print("[run_order] DRY RUN — no connection, no motion.\n")
        executor = MotionExecutor(_NullRobot(), dry_run=True)
        result = fulfill_order(order, executor, robot=_NullRobot(), dry_run_alert=True)
        return 0 if result.ok else 1

    twin_id = os.getenv("CYBERWAVE_TWIN_ID")
    if not twin_id:
        print("ERROR: CYBERWAVE_TWIN_ID is not set (see .env).", file=sys.stderr)
        return 2

    executor, robot = open_sim_executor(twin_id, settle=args.settle)
    print()
    try:
        result = fulfill_order(order, executor, robot=robot)
    finally:
        # Always leave the (simulated) arm at home.
        print("[run_order] homing …")
        executor.home(duration=1.0)

    print("\n[run_order] done. Watch the twin replay in the dashboard's Simulate view.")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
