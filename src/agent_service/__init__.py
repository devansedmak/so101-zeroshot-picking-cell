"""Agent service: the order→pick→place→verify→alert loop.

``fulfill_order`` closes the loop when a ``verify`` callable is injected (the thesis,
decisions.md D9): failed verification retries once, then raises an ERROR alert. Without
a verifier it stays the perception-free walking skeleton on hardcoded poses.

Where it picks is injected the same way: a ``resolve_pick`` callable ``(order) ->
PickChoice``. ``run_order --perceive`` supplies one backed by ``perception.locate_item``
(VLM → homography → IK) that degrades to the hardcoded pose table with a warning.

``order_api`` is the *order-driven* front door: a stdlib webhook receiver whose
``POST /orders`` runs one order through this same loop (``python -m
src.agent_service.order_api``).
"""

from typing import Any

from .alerts import build_failed_alert, build_fulfilled_alert, send_alert
from .loop import (
    MAX_ATTEMPTS,
    Fulfillment,
    PickResolver,
    Verifier,
    VerifyOutcome,
    fulfill_order,
)
from .poses import (
    HARDCODED,
    PERCEIVED,
    PickChoice,
    UnknownBin,
    UnknownItem,
    pick_choice,
    pick_choice_from_table,
    pick_plan,
    pick_plan_from_table,
    place_plan,
)

# order_api is exported LAZILY (PEP 562): importing it here eagerly would make
# `python -m src.agent_service.order_api` emit a "found in sys.modules" RuntimeWarning
# right before the demo, and would pull http.server into every import of this package.
_LAZY = {"Runner", "make_handler", "make_runner", "make_server"}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name in _LAZY:
        from . import order_api

        return getattr(order_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HARDCODED",
    "MAX_ATTEMPTS",
    "PERCEIVED",
    "Fulfillment",
    "PickChoice",
    "PickResolver",
    "Runner",
    "UnknownBin",
    "UnknownItem",
    "VerifyOutcome",
    "Verifier",
    "build_failed_alert",
    "build_fulfilled_alert",
    "fulfill_order",
    "make_handler",
    "make_runner",
    "make_server",
    "pick_choice",
    "pick_choice_from_table",
    "pick_plan",
    "pick_plan_from_table",
    "place_plan",
    "send_alert",
]
