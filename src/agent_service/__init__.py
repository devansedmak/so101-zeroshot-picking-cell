"""Agent service: the order→pick→place→alert loop (walking skeleton, perception-free).

Future: receive orders via a Cyberwave workflow webhook and close the loop with
visual verification (decisions.md D9). For now ``fulfill_order`` runs hardcoded poses.
"""

from .alerts import build_fulfilled_alert, send_alert
from .loop import Fulfillment, fulfill_order
from .poses import UnknownBin, UnknownItem, pick_plan, place_plan

__all__ = [
    "Fulfillment",
    "UnknownBin",
    "UnknownItem",
    "build_fulfilled_alert",
    "fulfill_order",
    "pick_plan",
    "place_plan",
    "send_alert",
]
