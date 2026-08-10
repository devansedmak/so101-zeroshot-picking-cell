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

**Where the pick comes from is injected the same way**: ``resolve_pick`` is any callable
``(order) -> poses.PickChoice``. ``None`` ⇒ the hardcoded pose table, i.e. today's exact
behaviour; ``run_order --perceive`` passes a resolver that asks perception where the item
really is (VLM → homography → IK) and *falls back to the hardcoded pose with a warning*
when the calibration or the detection isn't there. Injection is what keeps this module
camera-free and lets the degradation policy live in one obvious place (run_order).

Pure + duck-typed: works with control.MotionExecutor's ``dry_run`` and any robot
exposing ``joints.set`` / ``alerts.create`` — so it unit-tests with a fake robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from src.control import MotionExecutor
from src.wms_mock import Order

from .alerts import build_failed_alert, build_fulfilled_alert, send_alert
from .poses import HARDCODED, PickChoice, pick_choice, place_plan

# Pick+place is attempted at most twice before we give up and alert (demo-scenario §5).
MAX_ATTEMPTS = 2


class VerifyOutcome(Protocol):
    """What a verifier must return: an ``ok`` verdict (``reason``/``to_dict`` optional)."""

    ok: bool


# ``verify(order) -> VerifyOutcome``. Kept as a plain callable so a lambda, a fake, or
# ``functools.partial(verify_placement, client, ...)`` all work without a base class.
Verifier = Callable[[Order], Any]

# ``resolve_pick(order) -> PickChoice`` (a bare MotionPlan is accepted too — see
# ``_as_choice``). Its contract: it must NOT raise for a recoverable perception problem;
# degrading to the hardcoded pose is the resolver's job, not the loop's.
PickResolver = Callable[[Order], Any]


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
    pick_source: str = HARDCODED  # "hardcoded" | "perceived" — which pick path ran
    pick_target_mm: tuple[float, float] | None = None  # perceived table coord, if any

    @property
    def ok(self) -> bool:
        return self.status == "fulfilled"

    @property
    def perceived(self) -> bool:
        """Did this run pick where perception said, rather than from the pose table?"""
        return self.pick_source == "perceived"


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


def _as_choice(resolved: Any) -> PickChoice:  # noqa: ANN401 — duck-typed on purpose
    """Normalize a resolver's answer to a :class:`PickChoice` (bare plans allowed)."""
    return resolved if isinstance(resolved, PickChoice) else PickChoice(resolved)


def fulfill_order(
    order: Order,
    executor: MotionExecutor,
    *,
    robot: Any = None,
    dry_run_alert: bool = False,
    verify: Verifier | None = None,
    resolve_pick: PickResolver | None = None,
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

    ``resolve_pick`` decides *where* to pick: ``(order) -> PickChoice``. ``None`` ⇒ the
    hardcoded pose table (unchanged default). The chosen path is recorded on the result
    as ``pick_source`` / ``pick_target_mm``.

    Never raises: unknown item/bin, motion errors and verification errors all come back
    as a ``failed`` :class:`Fulfillment`, so a batch loop keeps going.
    """
    result = Fulfillment(order=order, status="fulfilled")
    try:
        print(f"▶ order: pick {order.item!r} → bin {order.bin!r}")
        # Resolve the pick target BEFORE any motion (so does place) — a planning error
        # must never leave the arm half-way through an order.
        choice = _as_choice(resolve_pick(order)) if resolve_pick else pick_choice(order.item)
        pick = choice.plan
        place = place_plan(order.bin)
        result.pick_source = choice.source
        result.pick_target_mm = choice.table_mm

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
