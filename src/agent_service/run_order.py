"""Run ONE order end-to-end (order → pick → place → verify → alert) — sim entrypoint.

Pull an order from the mock WMS, drive the pick/place poses through
control.MotionExecutor, then (with ``--verify``) close the loop: re-look at the scene,
confirm the item is in the bin, retry once if not, else raise an ERROR alert (D9 thesis).
Default is **dry-run** (no connection, no motion, no credits); the actual SIMULATION run
is behind ``--sim`` and reuses the proven connect/warm-up path (control.sim_session).

``--verify`` in dry-run uses a STUB verifier (``--verify-fail N`` makes the first N
checks fail) so the whole closed loop — including the retry and the error alert — is
demoable offline with zero credits and zero hardware. Only ``--sim --verify`` touches
the real camera + hosted VLM.

SAFETY (CLAUDE.md rule 2 / D9): SIMULATION-ONLY — ``--sim`` pins
``cw.affect("simulation")`` via sim_session; there is no live-hardware path here.

Usage:
    python -m src.agent_service.run_order                        # dry-run first demo order
    python -m src.agent_service.run_order --item "eraser" --bin A
    python -m src.agent_service.run_order --verify               # closed loop, stub verifier
    python -m src.agent_service.run_order --verify --verify-fail 1   # retry path
    python -m src.agent_service.run_order --verify --verify-fail 2   # failure + ERROR alert
    python -m src.agent_service.run_order --sim                  # move the twin in SIMULATION
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from src.control import MotionExecutor
from src.control.sim_session import load_env, open_sim_executor
from src.wms_mock import Order, OrderSource

from .loop import Verifier, fulfill_order


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


class _StubVerdict:
    """Offline stand-in for perception.verify.VerifyResult (same duck-type)."""

    def __init__(self, ok: bool, reason: str) -> None:
        self.ok = ok
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "stub": True}


def _stub_verifier(fail_first: int = 0) -> Verifier:
    """A verifier that fails its first ``fail_first`` checks, then passes.

    Lets the demo show all three branches of the closed loop (ok / retry / give up)
    with no camera, no VLM and no credits.
    """
    state = {"calls": 0}

    def verify(order: Order) -> _StubVerdict:
        state["calls"] += 1
        if state["calls"] <= fail_first:
            return _StubVerdict(False, f"(stub) {order.item} not seen in bin {order.bin}")
        return _StubVerdict(True, f"(stub) {order.item} seen inside bin {order.bin}")

    return verify


def _live_verifier(
    bin_label: str,
    *,
    device: str | None = None,
    client: Any = None,
) -> Verifier | None:
    """Real closed loop: re-capture a frame → hosted VLM → is the item in the bin?

    Returns ``None`` (loop stays open, warning printed) if the bin pixel rectangles
    haven't been surveyed yet — missing calibration must not abort a demo run.
    ``client`` is anything with ``mlmodels.run`` (defaults to a fresh ``Cyberwave()``;
    note the VLM client is NOT the twin). Everything network/camera is imported lazily
    so the dry-run path stays dependency-free.
    """
    from src.perception.verify import (
        BinRegionsMissing,
        bin_region_for,
        load_bin_regions,
        verify_placement,
    )

    try:
        region = bin_region_for(load_bin_regions(), bin_label)
    except (BinRegionsMissing, KeyError) as e:
        print(f"[run_order] ⚠ verification disabled: {e}")
        return None

    def verify(order: Order) -> Any:  # noqa: ANN401
        from src.perception.capture import capture_still  # lazy: camera-gated module

        cw = client
        if cw is None:
            from cyberwave import Cyberwave  # lazy: no credentials needed for dry-run

            cw = Cyberwave()
        frame = capture_still(device=device)
        from PIL import Image  # lazy: only the real run needs Pillow

        with Image.open(frame) as im:
            size = im.size  # (w, h) — bin rectangles are in these pixels
        return verify_placement(cw, str(frame), order.item, region, image_size=size)

    return verify


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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Close the loop: verify the placement, retry once, else ERROR alert (D9 thesis).",
    )
    parser.add_argument(
        "--verify-fail",
        type=int,
        default=0,
        metavar="N",
        help="Dry-run only: make the stub verifier fail its first N checks (demo the retry/failure).",
    )
    parser.add_argument(
        "--camera",
        help="Camera device for the verification re-capture (--sim --verify only).",
    )
    args = parser.parse_args(argv)

    load_env()
    order = _pick_order(args)

    if not args.sim:
        print("[run_order] DRY RUN — no connection, no motion.\n")
        executor = MotionExecutor(_NullRobot(), dry_run=True)
        verify = _stub_verifier(args.verify_fail) if args.verify else None
        if verify:
            print("[run_order] closed loop with a STUB verifier (no camera, no credits).\n")
        result = fulfill_order(
            order, executor, robot=_NullRobot(), dry_run_alert=True, verify=verify
        )
        return 0 if result.ok else 1

    twin_id = os.getenv("CYBERWAVE_TWIN_ID")
    if not twin_id:
        print("ERROR: CYBERWAVE_TWIN_ID is not set (see .env).", file=sys.stderr)
        return 2

    executor, robot = open_sim_executor(twin_id, settle=args.settle)
    verify = _live_verifier(order.bin, device=args.camera) if args.verify else None
    print()
    try:
        result = fulfill_order(order, executor, robot=robot, verify=verify)
    finally:
        # Always leave the (simulated) arm at home.
        print("[run_order] homing …")
        executor.home(duration=1.0)

    print("\n[run_order] done. Watch the twin replay in the dashboard's Simulate view.")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
