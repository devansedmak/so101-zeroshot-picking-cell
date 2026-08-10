"""Agent service: the order→pick→place→verify→alert loop.

``fulfill_order`` closes the loop when a ``verify`` callable is injected (the thesis,
decisions.md D9): failed verification retries once, then raises an ERROR alert. Without
a verifier it stays the perception-free walking skeleton on hardcoded poses.
"""

from .alerts import build_failed_alert, build_fulfilled_alert, send_alert
from .loop import MAX_ATTEMPTS, Fulfillment, Verifier, VerifyOutcome, fulfill_order
from .poses import UnknownBin, UnknownItem, pick_plan, pick_plan_from_table, place_plan

__all__ = [
    "MAX_ATTEMPTS",
    "Fulfillment",
    "UnknownBin",
    "UnknownItem",
    "VerifyOutcome",
    "Verifier",
    "build_failed_alert",
    "build_fulfilled_alert",
    "fulfill_order",
    "pick_plan",
    "pick_plan_from_table",
    "place_plan",
    "send_alert",
]
