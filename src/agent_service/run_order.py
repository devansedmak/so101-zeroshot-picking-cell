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

``--perceive`` is the **headline** switch: pick where the camera actually sees the item
(VLM → homography → IK) instead of the throwaway hardcoded pose table. It degrades
loudly, never fatally — resolution order per pick:

    1. --perceive AND the calibration loads AND the item is detected AND IK reaches it
       → poses.pick_plan_from_table(x, y)            [perceived]
    2. anything above missing/failing → poses.pick_plan(item) + a printed ⚠ saying why
    3. no hardcoded pose either → a recorded FAILED Fulfillment (never an exception)

Without ``--perceive`` nothing changes at all. Offline, ``--frame`` + ``--detections``
replay a saved frame and canned VLM output (the pick-side analogue of the stub verifier),
so the perceived path is demoable with no camera, no credits and no network.

SAFETY (CLAUDE.md rule 2 / D9): SIMULATION-ONLY — ``--sim`` pins
``cw.affect("simulation")`` via sim_session; there is no live-hardware path here.

Usage:
    python -m src.agent_service.run_order                        # dry-run first demo order
    python -m src.agent_service.run_order --item "eraser" --bin A
    python -m src.agent_service.run_order --verify               # closed loop, stub verifier
    python -m src.agent_service.run_order --verify --verify-fail 1   # retry path
    python -m src.agent_service.run_order --verify --verify-fail 2   # failure + ERROR alert
    python -m src.agent_service.run_order --sim                  # move the twin in SIMULATION
    python -m src.agent_service.run_order --perceive \
        --frame frame.jpg --frame-size 640x480 --detections dets.json   # perceived, offline
    python -m src.agent_service.run_order --sim --perceive        # perceived, real camera+VLM
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.control import MotionExecutor
from src.control.sim_session import load_env, open_sim_executor
from src.wms_mock import Order, OrderSource

from .loop import PickResolver, Verifier, fulfill_order
from .poses import pick_choice, pick_choice_from_table


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

            from src.control.sim_session import load_env as _load_env

            _load_env()  # cheap + idempotent: this helper is importable on its own, so it
            # cannot assume main() (which also calls load_env) is what invoked it.
            cw = Cyberwave()
        frame = capture_still(device=device)
        from PIL import Image  # lazy: only the real run needs Pillow

        with Image.open(frame) as im:
            size = im.size  # (w, h) — bin rectangles are in these pixels
        return verify_placement(cw, str(frame), order.item, region, image_size=size)

    return verify


class _CannedVLM:
    """Offline stand-in for the hosted VLM: replays a saved ``detect_points`` output.

    Satisfies the only surface ``perception.detect`` needs (``mlmodels.run``), so the
    whole perceived pick — detection → homography → IK → plan — is demoable and
    re-runnable with no camera, no network and zero credits. The pick-side twin of
    ``_stub_verifier``. ``output=None`` ⇒ "the VLM saw nothing", which is exactly how
    the graceful-degradation path is exercised offline.
    """

    def __init__(self, output: Any = None) -> None:  # noqa: ANN401
        self.output = output
        self.calls: list[dict[str, Any]] = []

    @property
    def mlmodels(self) -> "_CannedVLM":
        return self

    def run(self, model: Any, **kwargs: Any) -> "_CannedVLM":  # noqa: ANN401
        self.calls.append({"model": model, **kwargs})
        return self


def _load_canned_detections(path: str | Path) -> Any:  # noqa: ANN401
    """Read canned VLM output: a bare list, or ``{"output": [...]}`` (as logged)."""
    data = json.loads(Path(path).read_text())
    return data.get("output", data) if isinstance(data, dict) else data


def _parse_frame_size(spec: str) -> tuple[int, int]:
    """``"640x480"`` → ``(640, 480)``."""
    parts = [p for p in spec.replace("x", " ").replace(",", " ").split() if p]
    if len(parts) != 2:
        raise ValueError(f"bad --frame-size {spec!r}: expected WxH, e.g. 640x480")
    return int(parts[0]), int(parts[1])


def _warn_if_clamped(choice: Any) -> None:  # noqa: ANN401 — duck-typed PickChoice
    """Say it out loud when the IK solution exceeds the executor's joint limits.

    ``solve_ik`` returns the truthful raw solution; ``MotionExecutor`` then *clamps* at
    the SDK boundary — safe, but it silently moves the tip away from the perceived point.
    With the current conservative skeleton limits (± 60° on joints 2-4) a real table
    reach needs more elbow travel than that, so this is expected to fire until the
    limits are widened against measured link lengths (hardware/config/link-lengths.md).
    """
    from src.control import DEFAULT_JOINT_LIMITS, in_limits  # cheap, no SDK

    pose = next((a.pose for a in choice.plan.actions if a.type == "set_pose"), None)
    if not pose or in_limits(pose):
        return
    offenders = []
    for joint, angle in pose.items():
        lo, hi = DEFAULT_JOINT_LIMITS.get(joint, (angle, angle))
        if not lo <= angle <= hi:
            offenders.append(f"joint {joint}={angle:+.1f}° (limit {lo:+.0f}..{hi:+.0f})")
    over = ", ".join(offenders)
    print(f"[run_order] ⚠ IK solution outside the executor's limits: {over}")
    print("[run_order]   → the executor will CLAMP, so the tip lands short of the "
          "perceived point. Widen DEFAULT_JOINT_LIMITS once the link lengths are measured.")


def _perceiving_resolver(
    *,
    client: Any,
    homography_path: str | Path | None = None,
    frame: str | Path | None = None,
    frame_size: tuple[int, int] | None = None,
    camera: str | None = None,
) -> PickResolver | None:
    """``--perceive``: resolve each pick from what the camera sees, degrading loudly.

    Returns ``None`` — meaning "loop keeps its hardcoded pose table" — when the
    pixel→table calibration isn't there yet, after printing why. Same shape and wording
    as ``_live_verifier``'s missing-calibration branch: **missing calibration must not
    abort a demo run**. Per-order failures (item not detected, capture failed, target
    unreachable) also fall back with a ⚠ instead of failing the order.

    Everything perception is imported lazily so the default dry-run path stays
    dependency-free.
    """
    from src.perception.locate import DEFAULT_HOMOGRAPHY_PATH, load_homography, locate_item

    path = homography_path or DEFAULT_HOMOGRAPHY_PATH
    try:
        homography = load_homography(path)
    except (OSError, ValueError, KeyError) as e:  # HomographyMissing is a FileNotFoundError
        print(f"[run_order] ⚠ perceived picking disabled: {e}")
        print("[run_order]   → falling back to the hardcoded pick poses (poses.PICK_POSES).")
        return None

    where = f"frame {frame}" if frame else f"camera {camera or 'auto'}"
    print(f"[run_order] 👁 perceived picking enabled (calibration {path}, {where}).")

    def resolve(order: Order) -> Any:  # noqa: ANN401
        try:
            located = locate_item(
                order.item,
                client=client,
                homography=homography,
                image=frame,
                camera=camera,
                image_size=frame_size,
            )
            choice = pick_choice_from_table(
                located.x_mm,
                located.y_mm,
                pixel=located.pixel,
                frame=str(located.frame) if located.frame else None,
            )
        except Exception as e:  # noqa: BLE001 — perception must never fail an order
            print(f"[run_order] ⚠ perceived pick unavailable: {type(e).__name__}: {e}")
            print(f"[run_order]   → falling back to the hardcoded pose for {order.item!r}.")
            return pick_choice(order.item)  # UnknownItem here ⇒ recorded failed Fulfillment
        print(f"[run_order] 👁 {located.describe()}")
        _warn_if_clamped(choice)
        return choice

    return resolve


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
        help="Camera device for the capture/re-capture (--perceive, or --sim --verify).",
    )
    parser.add_argument(
        "--perceive",
        action="store_true",
        help="Pick where the item is actually SEEN (VLM → homography → IK) instead of the "
        "hardcoded pose table; warns and falls back if uncalibrated/undetected.",
    )
    parser.add_argument(
        "--homography",
        metavar="PATH",
        help="Pixel→table calibration JSON (default: the file calibrate_homography.py writes).",
    )
    parser.add_argument(
        "--frame",
        metavar="PATH",
        help="--perceive on a SAVED frame instead of capturing one (offline, re-runnable).",
    )
    parser.add_argument(
        "--frame-size",
        metavar="WxH",
        help="Pixel size of --frame when it should not be read from the file (e.g. 640x480).",
    )
    parser.add_argument(
        "--detections",
        metavar="PATH",
        help="Dry-run only: canned VLM output JSON to replay instead of calling the hosted "
        "VLM (zero credits) — e.g. [{\"point\": [y, x], \"label\": \"red marker\"}].",
    )
    args = parser.parse_args(argv)

    load_env()
    order = _pick_order(args)
    frame_size = _parse_frame_size(args.frame_size) if args.frame_size else None

    if not args.sim:
        print("[run_order] DRY RUN — no connection, no motion.\n")
        executor = MotionExecutor(_NullRobot(), dry_run=True)
        verify = _stub_verifier(args.verify_fail) if args.verify else None
        if verify:
            print("[run_order] closed loop with a STUB verifier (no camera, no credits).\n")
        resolve_pick = None
        if args.perceive:
            # NEVER a real VLM call in dry-run: replay canned output, or see nothing.
            canned = _load_canned_detections(args.detections) if args.detections else None
            resolve_pick = _perceiving_resolver(
                client=_CannedVLM(canned),
                homography_path=args.homography,
                frame=args.frame,
                frame_size=frame_size,
                camera=args.camera,
            )
            if resolve_pick is not None and not args.detections:
                print(
                    "[run_order] ⚠ dry-run --perceive has no canned VLM output "
                    "(--detections PATH); detection will find nothing."
                )
            print()
        result = fulfill_order(
            order,
            executor,
            robot=_NullRobot(),
            dry_run_alert=True,
            verify=verify,
            resolve_pick=resolve_pick,
        )
        return 0 if result.ok else 1

    twin_id = os.getenv("CYBERWAVE_TWIN_ID")
    if not twin_id:
        print("ERROR: CYBERWAVE_TWIN_ID is not set (see .env).", file=sys.stderr)
        return 2

    executor, robot = open_sim_executor(twin_id, settle=args.settle)
    verify = _live_verifier(order.bin, device=args.camera) if args.verify else None
    resolve_pick = None
    if args.perceive:
        # Canned output stays available under --sim (dress rehearsal, zero credits);
        # otherwise this is the real hosted VLM. The twin is NOT the VLM client.
        if args.detections:
            client: Any = _CannedVLM(_load_canned_detections(args.detections))
            print("[run_order] 👁 replaying canned detections (no VLM call, zero credits).")
        else:
            from cyberwave import Cyberwave  # lazy: no credentials needed for dry-run

            client = Cyberwave()  # main() has already called load_env()
        resolve_pick = _perceiving_resolver(
            client=client,
            homography_path=args.homography,
            frame=args.frame,
            frame_size=frame_size,
            camera=args.camera,
        )
    print()
    try:
        result = fulfill_order(
            order, executor, robot=robot, verify=verify, resolve_pick=resolve_pick
        )
    finally:
        # Always leave the (simulated) arm at home.
        print("[run_order] homing …")
        executor.home(duration=1.0)

    print("\n[run_order] done. Watch the twin replay in the dashboard's Simulate view.")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
