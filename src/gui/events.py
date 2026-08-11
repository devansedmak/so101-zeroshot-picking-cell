"""The dashboard's event vocabulary and the state machine the page renders.

One direction of data flow: *producers* (the mock driver, or a real ``run_order`` run via
:mod:`src.gui.emit`) append **events**; :class:`DashboardState` folds them into the single
JSON document the browser polls. The page itself is dumb — it only draws
``DashboardState.to_dict()``.

An event is a plain JSON dict with a ``kind``:

    {"kind": "reset",        "mode": "order"}
    {"kind": "order",        "order_id": "SO-1042", "item": "red marker", "bin": "A"}
    {"kind": "stage",        "stage": "DETECT", "status": "active", "detail": "..."}
    {"kind": "detection",    "pixel": [x, y], "table_mm": [x, y], "label": "red marker"}
    {"kind": "qc",           "grade": "reject", "defect": "cracked barrel", ...}
    {"kind": "routing",      "bin": "C", "requested_bin": "A", "diverted": true, ...}
    {"kind": "motion",       "stage": "PICK", "state": "mismatch", "errors": {"elbow": 7.4}}
    {"kind": "verification", "ok": false, "reason": "...", "point": [x, y], "bin": "A"}
    {"kind": "alert",        "severity": "error", "name": "...", "description": "..."}
    {"kind": "outcome",      "status": "failed", "detail": "..."}
    {"kind": "frame",        "token": "after-1"}     # the frame bytes changed
    {"kind": "log",          "text": "..."}

Design rules that matter at 23:00 with a camera pointed at the screen:

* ``apply`` **never raises**. A malformed event from a POST is dropped and counted, not a
  500 and certainly not a dead dashboard.
* Stage timing is derived here, not by the producer: a stage going ``active`` closes the
  previous one. So a producer that forgets a ``done`` still yields sane per-stage times.
* Every field the page needs has a value from the very first paint (no ``undefined``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .modes import DEFAULT_MODE, MODE_BLURB, ZONES, QCVerdict, check_mode, zone_for

#: The pipeline, left to right, exactly as the video should read.
STAGES: tuple[str, ...] = ("ORDER", "DETECT", "LOCATE", "PICK", "PLACE", "VERIFY", "ALERT")

#: What each stage *is*, in one line — printed under the chip so a viewer needs no narration.
STAGE_BLURB: dict[str, str] = {
    "ORDER": "webhook POST /orders",
    "DETECT": "hosted VLM · detect_points",
    "LOCATE": "homography px → mm · IK",
    "PICK": "MotionExecutor · grasp",
    "PLACE": "MotionExecutor · release",
    "VERIFY": "re-capture · point-in-zone",
    "ALERT": "platform alert",
}

STAGE_STATES: tuple[str, ...] = ("idle", "active", "done", "failed", "skipped")

# ---------------------------------------------------------------- motion honesty
# The stages that actually command servos. Everything else is compute, and compute
# either returned a value or it didn't — there is nothing to "confirm".
MOTION_STAGES: tuple[str, ...] = ("PICK", "PLACE")

#: The FOUR states a commanded move can be in. Two would be a lie.
#:
#: On 2026-08-11 an hour of live session was lost because ``MotionExecutor.execute``
#: publishes to MQTT and prints "plan complete" with **no servo acknowledgement**: the
#: edge driver had lost its serial binding, so every command "succeeded" while the arm
#: stood still. ``control.live_session.verify_pose`` closed that gap (it returns
#: ``{joint: error_deg}`` for joints outside tolerance, empty ⇒ all good, and *raises*
#: when there is no telemetry at all). This vocabulary is that helper's three answers
#: plus the state before it has answered — and the page must render all four
#: differently, because "commanded" looking like "verified" IS the bug.
MOTION_STATES: tuple[str, ...] = ("commanded", "verified", "mismatch", "unverified")

#: state → (short label, one-line meaning, "good"|"warn"|"bad")
MOTION_MEANING: dict[str, tuple[str, str, str]] = {
    "commanded": (
        "COMMANDED",
        "plan published to MQTT — the servos have NOT acknowledged it yet",
        "warn",
    ),
    "verified": (
        "VERIFIED",
        "encoders read back and agree with the commanded pose",
        "good",
    ),
    "mismatch": (
        "MISMATCH",
        "encoders disagree with the commanded pose — the arm is not where we asked",
        "bad",
    ),
    "unverified": (
        "UNVERIFIED",
        "no joint telemetry — we CANNOT tell whether the arm moved (never 'green')",
        "warn",
    ),
}

#: verify_pose's default tolerance, echoed on the page so the numbers mean something.
DEFAULT_TOLERANCE_DEG = 5.0

#: Terminal outcomes. ``retried`` is "fulfilled, but it took two attempts".
OUTCOMES: tuple[str, ...] = ("running", "fulfilled", "retried", "failed")


def _now() -> float:
    return time.time()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _point(value: Any) -> list[float] | None:
    """``[x, y]``-ish → ``[x, y]`` floats, or ``None``."""
    if isinstance(value, Mapping):
        value = [value.get("x"), value.get("y")]
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def motion_report(
    stage: str,
    state: str,
    *,
    errors: Mapping[str, Any] | None = None,
    detail: str = "",
    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
    attempt: int = 0,
    t: float | None = None,
) -> dict[str, Any]:
    """Normalize one motion-confirmation report into what the page draws.

    The rule this whole module exists for: a state is only ``verified`` when telemetry
    came back and agreed. ``errors`` non-empty forces ``mismatch`` even if the producer
    claimed otherwise, so an optimistic caller cannot paint a bad move green.
    """
    name = str(stage or "").strip().upper()
    st = str(state or "").strip().lower()
    if st not in MOTION_STATES:
        raise ValueError(f"unknown motion state {state!r}; expected one of {MOTION_STATES}")
    drift: dict[str, float] = {}
    for joint, value in dict(errors or {}).items():
        try:
            drift[str(joint)] = round(float(value), 2)
        except (TypeError, ValueError):
            continue
    if drift and st == "verified":  # telemetry disagreed — it is NOT verified.
        st = "mismatch"
    label, meaning, tone = MOTION_MEANING[st]
    worst = max((abs(v) for v in drift.values()), default=0.0)
    return {
        "stage": name,
        "state": st,
        "label": label,
        "meaning": meaning,
        "tone": tone,
        "confirmed": st == "verified",
        "errors": drift,
        "max_error_deg": round(worst, 2),
        "tolerance_deg": round(float(tolerance_deg), 2),
        "detail": str(detail or ""),
        "attempt": int(attempt or 0),
        "t": _now() if t is None else float(t),
    }


def motion_summary(errors: Mapping[str, float]) -> str:
    """``{"elbow": 7.4}`` → ``"elbow +7.4°"`` — the per-joint error, worst first."""
    items = sorted(errors.items(), key=lambda kv: -abs(kv[1]))
    return " · ".join(f"{j} {v:+.1f}°" for j, v in items)


@dataclass
class StageState:
    """One chip in the pipeline strip."""

    name: str
    status: str = "idle"
    detail: str = ""
    started: float | None = None
    ended: float | None = None
    ms: float | None = None
    motion: dict[str, Any] | None = None  # only ever set for MOTION_STAGES

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "ms": None if self.ms is None else round(self.ms, 1),
            "blurb": STAGE_BLURB.get(self.name, ""),
            "motion": self.motion,
            "moves": self.name in MOTION_STAGES,
        }


@dataclass
class DashboardState:
    """Everything the page draws. Folded from events; JSON-safe via :meth:`to_dict`."""

    mode: str = DEFAULT_MODE
    source: str = "mock"  # "mock" | "live" — shown as a badge, never guessed by the page
    stages: dict[str, StageState] = field(default_factory=dict)
    order: dict[str, Any] | None = None
    detection: dict[str, Any] | None = None
    qc: dict[str, Any] | None = None
    routing: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    alerts: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = "running"
    outcome_detail: str = ""
    attempt: int = 0
    attempts: int = 0
    frame_token: str = "0"
    frame_size: list[int] = field(default_factory=lambda: [1920, 1080])
    log: list[dict[str, Any]] = field(default_factory=list)
    seq: int = 0
    dropped: int = 0
    started: float = field(default_factory=_now)
    updated: float = field(default_factory=_now)
    run_id: int = 0

    #: Keep the log bounded — a dashboard left running all evening must not eat RAM.
    log_limit: int = 60

    def __post_init__(self) -> None:
        if not self.stages:
            self.reset(self.mode)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, mode: str | None = None, *, source: str | None = None) -> None:
        """Start a fresh run. Keeps ``mode``/``source`` unless overridden."""
        if mode is not None:
            self.mode = check_mode(mode)
        if source is not None:
            self.source = source
        self.stages = {name: StageState(name) for name in STAGES}
        self.order = None
        self.detection = None
        self.qc = None
        self.routing = None
        self.verification = None
        self.alerts = []
        self.outcome = "running"
        self.outcome_detail = ""
        self.attempt = 0
        self.attempts = 0
        self.log = []
        self.started = _now()
        self.updated = self.started
        self.run_id += 1

    # ------------------------------------------------------------------ folding
    def apply_all(self, events: Iterable[Mapping[str, Any]]) -> int:
        """Apply many events; returns how many were understood."""
        return sum(1 for e in events if self.apply(e))

    def apply(self, event: Mapping[str, Any]) -> bool:
        """Fold one event in. Returns ``False`` (and counts it) if it made no sense.

        Never raises: this is fed straight from an HTTP body.
        """
        if not isinstance(event, Mapping):
            self.dropped += 1
            return False
        kind = str(event.get("kind") or "").strip().lower()
        handler = getattr(self, f"_on_{kind}", None) if kind else None
        if handler is None:
            self.dropped += 1
            return False
        try:
            handler(event)
        except Exception:  # noqa: BLE001 — a bad event must never kill the dashboard
            self.dropped += 1
            return False
        self.seq += 1
        self.updated = _now()
        return True

    # ------------------------------------------------------------------ handlers
    def _on_reset(self, e: Mapping[str, Any]) -> None:
        self.reset(e.get("mode"), source=e.get("source"))

    def _on_mode(self, e: Mapping[str, Any]) -> None:
        self.mode = check_mode(e.get("mode"))

    def _on_order(self, e: Mapping[str, Any]) -> None:
        order = {
            "order_id": str(e.get("order_id") or "").strip() or None,
            "item": str(e.get("item") or "").strip(),
            "bin": (str(e.get("bin")).strip().upper() if e.get("bin") else None),
        }
        self.order = order
        self._note(f"order {order['order_id'] or ''} {order['item']} → {order['bin'] or 'QC decides'}")

    def _on_stage(self, e: Mapping[str, Any]) -> None:
        name = str(e.get("stage") or "").strip().upper()
        if name not in self.stages:
            raise KeyError(name)
        status = str(e.get("status") or "active").strip().lower()
        if status not in STAGE_STATES:
            status = "active"
        t = _num(e.get("t"), _now())
        stage = self.stages[name]
        if status == "active":
            self._close_active(t, before=name)
            stage.started = t
            stage.ended = None
            stage.ms = None
            # A re-commanded move starts unconfirmed again: never carry attempt 1's
            # "VERIFIED" badge into attempt 2.
            stage.motion = None
        elif stage.started is None:
            stage.started = t
        if status in ("done", "failed", "skipped"):
            stage.ended = t
            stage.ms = max(0.0, (t - (stage.started or t))) * 1000.0
        stage.status = status
        detail = str(e.get("detail") or "")
        if detail:
            stage.detail = detail
            self._note(f"{name}: {detail}")
        if e.get("attempt"):
            self.attempt = int(_num(e.get("attempt"), self.attempt))
            self.attempts = max(self.attempts, self.attempt)

    def _close_active(self, t: float, *, before: str) -> None:
        """A stage going active implicitly finishes every earlier still-active stage."""
        for name in STAGES:
            if name == before:
                break
            stage = self.stages[name]
            if stage.status == "active":
                stage.status = "done"
                stage.ended = t
                stage.ms = max(0.0, (t - (stage.started or t))) * 1000.0

    def _on_motion(self, e: Mapping[str, Any]) -> None:
        report = motion_report(
            e.get("stage"),
            e.get("state"),
            errors=e.get("errors") if isinstance(e.get("errors"), Mapping) else None,
            detail=str(e.get("detail") or ""),
            tolerance_deg=_num(e.get("tolerance_deg"), DEFAULT_TOLERANCE_DEG),
            attempt=int(_num(e.get("attempt"), self.attempt)),
            t=_num(e.get("t"), _now()),
        )
        stage = self.stages.get(report["stage"])
        if stage is None:
            raise KeyError(report["stage"])
        stage.motion = report
        line = f"{report['stage']} motion {report['label']}"
        if report["errors"]:
            line += f" — {motion_summary(report['errors'])} (tol ±{report['tolerance_deg']:.0f}°)"
        elif report["detail"]:
            line += f" — {report['detail']}"
        self._note(line)

    def _on_detection(self, e: Mapping[str, Any]) -> None:
        self.detection = {
            "pixel": _point(e.get("pixel")),
            "table_mm": _point(e.get("table_mm")),
            "label": str(e.get("label") or ""),
            "confidence": _num(e.get("confidence"), 0.0) or None,
        }
        size = _point(e.get("frame_size"))
        if size:
            self.frame_size = [int(size[0]), int(size[1])]

    def _on_qc(self, e: Mapping[str, Any]) -> None:
        self.qc = QCVerdict.from_dict(e).to_dict()

    def _on_routing(self, e: Mapping[str, Any]) -> None:
        label = str(e.get("bin") or "").strip().upper()
        zone = zone_for(label)
        self.routing = {
            "mode": str(e.get("mode") or self.mode),
            "bin": label,
            "requested_bin": (
                str(e.get("requested_bin")).strip().upper() if e.get("requested_bin") else None
            ),
            "reason": str(e.get("reason") or ""),
            "diverted": bool(e.get("diverted")),
            "zone": zone.to_dict() if zone else None,
            "qc": self.qc,
        }
        self._note(self.routing["reason"])

    def _on_verification(self, e: Mapping[str, Any]) -> None:
        ok = bool(e.get("ok"))
        self.verification = {
            "ok": ok,
            "reason": str(e.get("reason") or ("verified" if ok else "not verified")),
            "point": _point(e.get("point")),
            "bin": (str(e.get("bin")).strip().upper() if e.get("bin") else None),
            "attempt": int(_num(e.get("attempt"), self.attempt or 1)),
        }
        self._note(("VERIFIED: " if ok else "NOT VERIFIED: ") + self.verification["reason"])

    def _on_alert(self, e: Mapping[str, Any]) -> None:
        alert = {
            "severity": str(e.get("severity") or "info").strip().lower(),
            "name": str(e.get("name") or "alert"),
            "description": str(e.get("description") or ""),
            "alert_type": str(e.get("alert_type") or ""),
            "dispatched": bool(e.get("dispatched", True)),
            # Who raised it. Anything other than the dashboard itself (e.g. "run_order")
            # means the producer already owns the platform call — see app._dispatch_alert.
            "origin": str(e.get("origin") or "dashboard"),
            "t": _num(e.get("t"), _now()),
        }
        self.alerts.append(alert)
        self._note(f"alert [{alert['severity']}] {alert['name']}")

    def _on_outcome(self, e: Mapping[str, Any]) -> None:
        status = str(e.get("status") or "").strip().lower()
        if status not in OUTCOMES:
            status = "failed"
        self.outcome = status
        self.outcome_detail = str(e.get("detail") or "")
        if e.get("attempts"):
            self.attempts = int(_num(e.get("attempts"), self.attempts))
        self._close_active(_num(e.get("t"), _now()), before="")
        self._note(f"outcome: {status.upper()} {self.outcome_detail}".strip())

    def _on_frame(self, e: Mapping[str, Any]) -> None:
        self.frame_token = str(e.get("token") or (int(_num(self.frame_token, 0)) + 1))
        size = _point(e.get("frame_size"))
        if size:
            self.frame_size = [int(size[0]), int(size[1])]

    def _on_log(self, e: Mapping[str, Any]) -> None:
        self._note(str(e.get("text") or ""))

    def _note(self, text: str) -> None:
        if not text:
            return
        self.log.append({"t": _now() - self.started, "text": text})
        if len(self.log) > self.log_limit:
            del self.log[: len(self.log) - self.log_limit]

    # ------------------------------------------------------------------ output
    def motion_panel(self) -> list[dict[str, Any]]:
        """One row per servo-commanding stage — the "did it actually move?" panel.

        A stage that has never been commanded reports ``state: null`` (an empty slot),
        which is different from every one of the four post-command states.
        """
        rows: list[dict[str, Any]] = []
        for name in MOTION_STAGES:
            stage = self.stages.get(name)
            report = stage.motion if stage else None
            if report is None:
                rows.append(
                    {"stage": name, "state": None, "label": "—", "tone": "idle",
                     "meaning": "not commanded yet", "errors": {}, "confirmed": False,
                     "summary": "", "detail": "", "max_error_deg": 0.0,
                     "tolerance_deg": DEFAULT_TOLERANCE_DEG}
                )
            else:
                rows.append({**report, "summary": motion_summary(report["errors"])})
        return rows

    def to_dict(self) -> dict[str, Any]:
        """The whole document the page polls. Stable keys, no ``None`` surprises."""
        return {
            "mode": self.mode,
            "mode_blurb": MODE_BLURB.get(self.mode, ""),
            "source": self.source,
            "modes": list(MODE_BLURB),
            "stages": [self.stages[name].to_dict() for name in STAGES],
            "order": self.order,
            "detection": self.detection,
            "qc": self.qc,
            "routing": self.routing,
            "verification": self.verification,
            "alerts": self.alerts,
            "outcome": self.outcome,
            "outcome_detail": self.outcome_detail,
            "attempt": self.attempt,
            "attempts": self.attempts,
            "zones": [z.to_dict() for z in ZONES.values()],
            "motion": self.motion_panel(),
            "motion_legend": [
                {"state": s, "label": MOTION_MEANING[s][0], "meaning": MOTION_MEANING[s][1],
                 "tone": MOTION_MEANING[s][2]}
                for s in MOTION_STATES
            ],
            "frame_token": self.frame_token,
            "frame_size": self.frame_size,
            "elapsed": round(self.updated - self.started, 2),
            "log": self.log[-14:],
            "seq": self.seq,
            "dropped": self.dropped,
            "run_id": self.run_id,
        }


# ---------------------------------------------------------------- event builders
# Small constructors so producers (mock driver, live bridge) never hand-write dicts
# and therefore never drift from the handlers above.


def ev(kind: str, **fields: Any) -> dict[str, Any]:
    """Build an event with a timestamp (``t`` is what stage timing is derived from)."""
    return {"kind": kind, "t": _now(), **fields}


def stage_event(stage: str, status: str = "active", detail: str = "", **extra: Any) -> dict[str, Any]:
    return ev("stage", stage=stage, status=status, detail=detail, **extra)


def motion_event(
    stage: str,
    state: str,
    *,
    errors: Mapping[str, float] | None = None,
    detail: str = "",
    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
    **extra: Any,
) -> dict[str, Any]:
    """Build a motion-confirmation event (see :data:`MOTION_STATES`).

    ``errors`` is exactly what ``control.live_session.verify_pose`` returns:
    ``{joint: error_deg}``, empty ⇒ every joint landed inside ``tolerance_deg``.
    """
    return ev(
        "motion",
        stage=stage,
        state=state,
        errors=dict(errors or {}),
        detail=detail,
        tolerance_deg=tolerance_deg,
        **extra,
    )
