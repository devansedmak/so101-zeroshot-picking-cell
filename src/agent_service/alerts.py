"""Fulfillment alerts — the "fulfilled" signal, stubbed behind a testable function.

Real SDK surface (verified against cyberwave.alerts.TwinAlertManager.create in S5):
    robot.alerts.create(name, *, description="", alert_type="", severity="warning",
                        source_type="edge", category="technical", metadata=None, ...)
    severity ∈ {info, warning, error, critical}; source_type ∈ {edge, simulation,
    cloud, workflow}; category ∈ {technical, business}. (⚠ NB: no ``message`` kwarg —
    details go in ``description``.)

We keep the *payload builder* pure (so tests assert on it with no robot) and the
*send* thin: offline / dry-run it just returns the payload without touching the SDK,
so the whole loop is unit-testable with a fake robot (mirrors tests/test_motion.py).
``source_type`` is a runtime property (sim vs live) so it's a send-time argument,
not part of the pure payload — default "simulation" (our only entrypoint today; pass
"edge" once the loop runs on real hardware, D9).

The closed-loop visual verification that will gate this alert is the project thesis
(decisions.md D9); for the skeleton, "arm completed both moves" == fulfilled.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.wms_mock import Order


class _Alerts(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


def build_fulfilled_alert(order: Order) -> dict[str, str]:
    """The alert payload for a successfully fulfilled order (pure — no SDK call).

    Only kwargs accepted by ``alerts.create``; ``source_type`` is added at send time.
    """
    return {
        "name": f"Order fulfilled: {order.item} → bin {order.bin}",
        "description": f"Picked {order.item} and placed it in bin {order.bin}.",
        "alert_type": "order_fulfilled",
        "severity": "info",
        "category": "business",
    }


def send_alert(
    robot: Any,
    alert: dict[str, str],
    *,
    dry_run: bool = False,
    source_type: str = "simulation",
) -> dict[str, Any]:
    """Dispatch ``alert`` via ``robot.alerts.create`` unless dry-run/no robot.

    Returns a small result dict (``dispatched`` + echoed payload) so callers and
    tests can assert what happened without a live platform. ``robot`` only needs an
    ``alerts.create(**kwargs)`` — a fake satisfies it offline. ``source_type`` tags
    the alert's origin (default "simulation"; pass "edge" for real hardware).
    """
    payload = {**alert, "source_type": source_type}
    if dry_run or robot is None or not hasattr(robot, "alerts"):
        print(f"  🔔  (stub) {alert['name']}")
        return {"dispatched": False, **payload}

    result = robot.alerts.create(**payload)
    print(f"  🔔  alert dispatched: {alert['name']}")
    return {"dispatched": True, "result": result, **payload}
