"""The order→pick→place→**verify**→alert loop (the closed agentic loop).

``fulfill_order`` is the whole agent: take a validated :class:`Order`, look up a pick
pose for the item + place pose for the bin (poses.py), drive both moves through
:class:`control.MotionExecutor`, then — if a verifier is supplied — ask perception
whether the item actually landed in the bin and act on the answer:

    pick → place → verify ─ ok ──→ "fulfilled" alert (info)
                          └ not ok → retry pick+place ONCE → verify again
                                     └ still not ok → status="failed" + ERROR alert

That feedback edge is the project thesis (decisions.md D9, docs/demo-scenario.md step 5)
— it's what makes this an agent rather than an open-loop replay.

Verification is **opt-in and injected**: ``verify`` is any callable ``(order) -> result``
with an ``ok`` flag (perception.verify.VerifyResult satisfies it). This module therefore
imports NO network/camera code — with ``verify=None`` the loop behaves exactly like the
walking skeleton it grew from, which keeps the perception-free entrypoints/tests valid.

Pure + duck-typed: works with control.MotionExecutor's ``dry_run`` and any robot
exposing ``joints.set`` / ``alerts.create`` — so it unit-tests with a fake robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from src.control import MotionExecutor
from src.wms_mock import Order

from .alerts import build_failed_alert, build_fulfilled_alert, send_alert
from .poses import pick_plan, place_plan

# Pick+place is attempted at most twice before we give up and alert (demo-scenario §5).
MAX_ATTEMPTS = 2


class VerifyOutcome(Protocol):
    """What a verifier must return: an ``ok`` verdict (``reason``/``to_dict`` optional)."""

    ok: bool


# ``verify(order) -> VerifyOutcome``. Kept as a plain callable so a lambda, a fake, or
# ``functools.partial(verify_placement, client, ...)`` all work without a base class.
Verifier = Callable[[Order], Any]


@dataclass
class Fulfillment:
    """Outcome of running one order through the loop."""

    order: Order
    status: str  # "fulfilled" | "failed"
    alert: dict[str, Any] | None = None
    error: str | None = None
    stages: list[str] = field(default_factory=list)
    attempts: int = 0  # pick+place attempts actually executed
    verification: dict[str, Any] | None = None  # last verdict, JSON-safe (for logs/alerts)

    @property
    def ok(self) -> bool:
        return self.status == "fulfilled"


def _verdict(verify: Verifier, order: Order) -> tuple[bool, str, dict[str, Any] | None]:
    """Run the verifier and normalize its answer to ``(ok, reason, payload)``.

    A verifier that raises (VLM/network/camera error) is a *failed verification*, not a
    crash: the loop's contract is "never raise, return an outcome" (extended to
    verification per this module's docstring).
    """
    try:
        result = verify(order)
    except Exception as e:  # noqa: BLE001 — a broken verifier must not kill the run
        return False, f"verification error: {type(e).__name__}: {e}", None
    ok = bool(getattr(result, "ok", False))
    reason = str(getattr(result, "reason", "") or ("verified" if ok else "not verified"))
    to_dict = getattr(result, "to_dict", None)
    payload = to_dict() if callable(to_dict) else None
    return ok, reason, payload


def fulfill_order(
    order: Order,
    executor: MotionExecutor,
    *,
    robot: Any = None,
    dry_run_alert: bool = False,
    verify: Verifier | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> Fulfillment:
    """Run one order end-to-end: pick → place → (verify) → alert.

    ``executor`` carries its own dry-run/sim setting (so motion is controlled there).
    ``robot`` is passed to the alert stub for ``alerts.create``; pass ``None`` (or set
    ``dry_run_alert``) to keep the alert offline.

    ``verify`` closes the loop: a callable ``(order) -> result-with-.ok`` (typically
    ``perception.verify.verify_placement`` bound to a client + fresh frame). When it is
    ``None`` the loop stays open (skeleton behaviour: both moves done == fulfilled).
    Failed verification retries pick+place up to ``max_attempts`` times, then returns
    ``status="failed"`` and dispatches an ERROR alert.

    Never raises: unknown item/bin, motion errors and verification errors all come back
    as a ``failed`` :class:`Fulfillment`, so a batch loop keeps going.
    """
    result = Fulfillment(order=order, status="fulfilled")
    try:
        pick = pick_plan(order.item)
        place = place_plan(order.bin)

        print(f"▶ order: pick {order.item!r} → bin {order.bin!r}")
        attempts = 1 if verify is None else max(1, max_attempts)
        verified = verify is None  # no verifier ⇒ nothing to prove
        reason = ""

        for attempt in range(1, attempts + 1):
            if attempt > 1:
                result.stages.append("retried")
                print(f"  ↻ retry {attempt}/{attempts}: {reason}")
            executor.execute(pick)
            result.stages.append("picked")
            executor.execute(place)
            result.stages.append("placed")
            result.attempts = attempt

            if verify is None:
                break
            verified, reason, payload = _verdict(verify, order)
            result.verification = payload or {"ok": verified, "reason": reason}
            result.stages.append("verified" if verified else "verify-failed")
            print(f"  {'👁 verified' if verified else '👁 NOT verified'}: {reason}")
            if verified:
                break

        if not verified:
            result.status = "failed"
            result.error = f"verification failed after {result.attempts} attempt(s): {reason}"
            result.alert = send_alert(
                robot, build_failed_alert(order, reason), dry_run=dry_run_alert
            )
            result.stages.append("alerted")
            print(f"❌ failed: {order.item} → bin {order.bin}  ({reason})\n")
            return result

        result.alert = send_alert(robot, build_fulfilled_alert(order), dry_run=dry_run_alert)
        result.stages.append("alerted")
        print(f"✅ fulfilled: {order.item} → bin {order.bin}\n")
    except Exception as e:  # noqa: BLE001 — surface as a failed outcome, don't crash the batch
        result.status = "failed"
        result.error = f"{type(e).__name__}: {e}"
        print(f"❌ failed: {order.item} → bin {order.bin}  ({result.error})\n")
    return result
