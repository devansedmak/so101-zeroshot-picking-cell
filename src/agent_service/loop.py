"""The thin order→pick→place→alert loop (perception-free walking skeleton).

``fulfill_order`` is the whole agent for now: take a validated :class:`Order`, look
up a HARDCODED pick pose for the item + place pose for the bin (poses.py), drive both
moves through :class:`control.MotionExecutor`, then emit a "fulfilled" alert (alerts.py).

NO vision, NO IK, NO homography — those are deferred until the overhead camera is
mounted (decisions.md D9). This module is the seam they slot into: swap ``pick_plan``
for VLM+homography and ``send_alert`` for the visually-verified fulfillment signal.

Pure + duck-typed: works with control.MotionExecutor's ``dry_run`` and any robot
exposing ``joints.set`` / ``alerts.create`` — so it unit-tests with a fake robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.control import MotionExecutor
from src.wms_mock import Order

from .alerts import build_fulfilled_alert, send_alert
from .poses import pick_plan, place_plan


@dataclass
class Fulfillment:
    """Outcome of running one order through the loop."""

    order: Order
    status: str  # "fulfilled" | "failed"
    alert: dict[str, Any] | None = None
    error: str | None = None
    stages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "fulfilled"


def fulfill_order(
    order: Order,
    executor: MotionExecutor,
    *,
    robot: Any = None,
    dry_run_alert: bool = False,
) -> Fulfillment:
    """Run one order end-to-end: pick → place → "fulfilled" alert.

    ``executor`` carries its own dry-run/sim setting (so motion is controlled there).
    ``robot`` is passed to the alert stub for ``alerts.create``; pass ``None`` (or set
    ``dry_run_alert``) to keep the alert offline. Unknown item/bin → a ``failed``
    :class:`Fulfillment` (never raises for that), so a batch loop keeps going.
    """
    result = Fulfillment(order=order, status="fulfilled")
    try:
        pick = pick_plan(order.item)
        place = place_plan(order.bin)

        print(f"▶ order: pick {order.item!r} → bin {order.bin!r}")
        executor.execute(pick)
        result.stages.append("picked")
        executor.execute(place)
        result.stages.append("placed")

        result.alert = send_alert(robot, build_fulfilled_alert(order), dry_run=dry_run_alert)
        result.stages.append("alerted")
        print(f"✅ fulfilled: {order.item} → bin {order.bin}\n")
    except Exception as e:  # noqa: BLE001 — surface as a failed outcome, don't crash the batch
        result.status = "failed"
        result.error = f"{type(e).__name__}: {e}"
        print(f"❌ failed: {order.item} → bin {order.bin}  ({result.error})\n")
    return result
