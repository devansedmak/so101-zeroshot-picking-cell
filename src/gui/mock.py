"""Canned runs that drive the whole dashboard with **no hardware and no network**.

This is the fallback if the live demo misbehaves, so it is built to be convincing on its
own rather than to be a placeholder:

* the detected pixel is a real prop position in the synthetic frame, and it is turned
  into table millimetres by the **real** ``hardware/config/homography.json`` (so the mm
  the page shows are the mm the arm would be given);
* the zone rectangles are the **real** surveyed ``bin-regions.json``;
* routing goes through the same :func:`src.gui.modes.route` a live run uses;
* the verification verdict is the same point-in-rectangle test as
  ``perception.verify.evaluate_placement``, run on the point drawn in the frame.

The script is built as data (:func:`build_script`) and executed by :class:`MockDriver`,
so the sequencing is unit-testable without threads, sleeps or a browser.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from . import frames as F
from .events import DEFAULT_TOLERANCE_DEG, ev, motion_event, motion_summary, stage_event
from .modes import DEFAULT_MODE, QCVerdict, check_mode, names_bin, needs_qc, route

#: How a run ends. ``ok`` first try; ``retry`` fails once then succeeds; ``fail`` never verifies.
OUTCOMES: tuple[str, ...] = ("ok", "retry", "fail")
DEFAULT_OUTCOME = "ok"

#: Driver-only: rotate through the three endings so an unattended loop tells the whole
#: story (clean run -> mismatch + retry -> ghost arm + ERROR alert).
AUTO_OUTCOME = "auto"
OUTCOME_ROTATION: tuple[str, ...] = ("ok", "retry", "fail")

#: Seconds the finished outcome stays on screen before the loop restarts.
HOLD_SECONDS = 6.0

# ------------------------------------------------------------------ motion honesty
# What the encoders say about each commanded move, per canned ending. This is the one
# part of the mock that is *not* decoration: MotionExecutor reports success with no servo
# acknowledgement (the "ghost arm" failure), so the dashboard has to be able
# to draw COMMANDED / VERIFIED / MISMATCH / UNVERIFIED differently, and the only way to
# prove it does is to script a run through each of them.
#
#   ("STAGE", attempt) -> (state, {joint: error_deg}, detail)
MOTION_PLAN: dict[str, dict[tuple[str, int], tuple[str, dict[str, float], str]]] = {
    "ok": {
        ("PICK", 1): ("verified", {}, "6/6 joints read back within ±5.0°"),
        ("PLACE", 1): ("verified", {}, "6/6 joints read back within ±5.0°"),
    },
    "retry": {
        ("PICK", 1): ("verified", {}, "6/6 joints read back within ±5.0°"),
        ("PLACE", 1): (
            "mismatch",
            {"elbow": 7.4, "wrist_flex": -6.1},
            "encoders disagree — the gripper opened short of the zone",
        ),
        ("PICK", 2): ("verified", {}, "6/6 joints read back within ±5.0°"),
        ("PLACE", 2): ("verified", {}, "6/6 joints read back within ±5.0°"),
    },
    "fail": {
        ("PICK", 1): ("verified", {}, "6/6 joints read back within ±5.0°"),
        ("PLACE", 1): (
            "unverified",
            {},
            "no joint telemetry — cannot tell whether the arm moved",
        ),
        ("PICK", 2): ("unverified", {}, "no joint telemetry — driver lost its serial binding"),
        ("PLACE", 2): ("unverified", {}, "no joint telemetry — NOT a success, NOT green"),
    },
}

COMMANDED_DETAIL = "plan published to MQTT · awaiting encoder read-back"


@dataclass(frozen=True)
class Scenario:
    """One canned order: what was asked for, and what the VLM will say about it."""

    order_id: str
    item: str
    bin: str | None = None
    grade: str = "pass"
    defect: str | None = None
    confidence: float = 0.93
    qc_detail: str = ""


#: Rotated across repeated runs so a looping dashboard never shows the same clip twice.
SCENARIOS: dict[str, tuple[Scenario, ...]] = {
    "order": (
        Scenario("SO-1042", "red marker", "A"),
        Scenario("SO-1043", "blue marker", "B"),
        Scenario("SO-1044", "eraser", "C"),
    ),
    "qc": (
        Scenario("QC-2201", "red marker", None, "pass", None, 0.96, "barrel intact, cap seated"),
        Scenario("QC-2202", "blue marker", None, "review", "scuffed label", 0.71, "label legible but damaged"),
        Scenario("QC-2203", "eraser", None, "reject", "torn sleeve", 0.89, "sleeve torn, product soiled"),
    ),
    "fusion": (
        Scenario("SO-1051", "red marker", "A", "pass", None, 0.95, "no defect found"),
        Scenario("SO-1052", "blue marker", "A", "reject", "cracked barrel", 0.91, "hairline crack, ink leak risk"),
        Scenario("SO-1053", "eraser", "B", "review", "scuffed label", 0.68, "needs a human eye"),
    ),
}


@dataclass
class Beat:
    """One step of the script: emit these events, maybe swap the frame, then wait."""

    events: list[dict[str, Any]] = field(default_factory=list)
    frame: dict[str, Any] | None = None  # kwargs for frames.synth_scene
    delay: float = 0.0
    note: str = ""


def _homography() -> Any:
    """The real calibration if it is there, else ``None`` (mm are then reported as unknown)."""
    try:
        from src.perception.homography import Homography

        return Homography.load(Path("hardware/config/homography.json"))
    except Exception:  # noqa: BLE001, a missing/odd calibration must not break the demo
        return None


def _table_mm(homography: Any, px: float, py: float) -> list[float] | None:
    if homography is None:
        return None
    try:
        x, y = homography.project((px, py))
        return [round(float(x), 1), round(float(y), 1)]
    except Exception:  # noqa: BLE001
        return None


def commanded(stage: str, attempt: int = 1) -> dict[str, Any]:
    """"Sent, unconfirmed": what every move looks like the instant MQTT accepts it."""
    return motion_event(stage, "commanded", detail=COMMANDED_DETAIL, attempt=attempt)


def confirmed(stage: str, attempt: int, outcome: str) -> dict[str, Any]:
    """The encoder read-back for one commanded move under one canned ending."""
    plan = MOTION_PLAN.get(outcome, MOTION_PLAN[DEFAULT_OUTCOME])
    state, errors, detail = plan.get((stage, attempt), ("verified", {}, "encoders agree"))
    return motion_event(
        stage,
        state,
        errors=errors,
        detail=detail,
        tolerance_deg=DEFAULT_TOLERANCE_DEG,
        attempt=attempt,
    )


def build_script(
    mode: str = DEFAULT_MODE,
    outcome: str = DEFAULT_OUTCOME,
    scenario: Scenario | None = None,
    *,
    rects: dict[str, tuple[float, ...]] | None = None,
    homography: Any = None,
    speed: float = 1.0,
) -> list[Beat]:
    """Build the full beat list for one canned run. Pure: no threads, no sleeping."""
    m = check_mode(mode)
    out = outcome if outcome in OUTCOMES else DEFAULT_OUTCOME
    sc = scenario or SCENARIOS[m][0]
    rects = rects or F.load_zone_rects()
    homography = homography if homography is not None else _homography()
    scale = 1.0 / max(0.05, float(speed))

    def beat(*events: dict[str, Any], frame: dict[str, Any] | None = None, delay: float = 1.0,
             note: str = "") -> Beat:
        return Beat(list(events), frame, delay * scale, note)

    item = sc.item
    home = F.PROP_HOME.get(item, (640, 500))
    det_px = F.jitter(home[0], home[1], 6.0, seed=len(item))
    mm = _table_mm(homography, *det_px)

    qc = (
        QCVerdict(sc.grade, sc.defect, sc.confidence, sc.qc_detail or "")
        if needs_qc(m)
        else None
    )
    order_bin = sc.bin if names_bin(m) else None
    routing = route(m, item=item, order_bin=order_bin, qc=qc)
    dest_rect = rects.get(routing.bin)
    dest_centre = F.zone_centre(dest_rect) if dest_rect else (900.0, 560.0)
    good_point = F.jitter(dest_centre[0], dest_centre[1], 9.0, seed=3)
    # A miss the verifier must catch: on the table, clearly outside the destination.
    miss_point = (dest_centre[0] - 150.0, dest_centre[1] + 40.0)

    script: list[Beat] = []

    # ---------------------------------------------------------------- ORDER
    script.append(
        beat(
            ev("reset", mode=m, source="mock"),
            ev("frame", token=f"{sc.order_id}-idle", frame_size=list(F.load_frame_size())),
            frame={},
            delay=0.6,
            note="idle table",
        )
    )
    script.append(
        beat(
            stage_event("ORDER", "active", "webhook: POST /orders"),
            ev(
                "order",
                order_id=sc.order_id,
                item=item,
                bin=order_bin,
            ),
            delay=1.3,
            note="order in",
        )
    )
    dest_line = (
        f"bin {routing.bin}" if m == "order" else "destination decided after QC"
    )
    script.append(
        beat(
            stage_event("ORDER", "done", f"accepted · {dest_line}"),
            stage_event("DETECT", "active", f"hosted VLM · detect_points({item!r})"),
            delay=1.7,
            note="vlm call",
        )
    )

    # ---------------------------------------------------------------- DETECT
    script.append(
        beat(
            ev(
                "detection",
                pixel=[round(det_px[0], 1), round(det_px[1], 1)],
                table_mm=mm,
                label=item,
                confidence=0.94,
                frame_size=list(F.load_frame_size()),
            ),
            stage_event("DETECT", "done", f"1 point · px ({det_px[0]:.0f}, {det_px[1]:.0f})"),
            stage_event("LOCATE", "active", "homography px → table mm"),
            delay=1.2,
            note="point found",
        )
    )
    locate_detail = (
        f"table ({mm[0]:.0f}, {mm[1]:.0f}) mm · IK solved, joints in limits"
        if mm
        else "no homography loaded — pose table fallback"
    )
    script.append(
        beat(
            stage_event("LOCATE", "done", locate_detail),
            stage_event("PICK", "active", "MotionExecutor · approach → close → lift", attempt=1),
            commanded("PICK", 1),
            delay=1.6,
            note="pick",
        )
    )

    # ---------------------------------------------------------------- PICK (+QC)
    if needs_qc(m) and qc is not None:
        script.append(
            beat(
                confirmed("PICK", 1, out),
                ev("frame", token=f"{sc.order_id}-held"),
                stage_event("PICK", "done", "item in gripper · held up to the camera"),
                stage_event("DETECT", "active", f"hosted VLM · QC classify({item!r})"),
                frame={"held": item},
                delay=1.6,
                note="qc look",
            )
        )
        script.append(
            beat(
                ev("qc", **qc.to_dict()),
                ev("routing", **routing.to_dict()),
                stage_event("DETECT", "done", f"QC {qc.grade.upper()} · {qc.defect or 'no defect'}"),
                stage_event("PLACE", "active", f"MotionExecutor · release over zone {routing.bin}"),
                commanded("PLACE", 1),
                delay=1.8,
                note="qc verdict",
            )
        )
    else:
        script.append(
            beat(
                confirmed("PICK", 1, out),
                ev("routing", **routing.to_dict()),
                ev("frame", token=f"{sc.order_id}-held"),
                stage_event("PICK", "done", "item in gripper"),
                stage_event("PLACE", "active", f"MotionExecutor · release over bin {routing.bin}"),
                commanded("PLACE", 1),
                frame={"held": item},
                delay=1.7,
                note="place",
            )
        )

    # ---------------------------------------------------------------- PLACE -> VERIFY
    first_ok = out == "ok"
    first_point = good_point if first_ok else miss_point
    script.append(
        beat(
            confirmed("PLACE", 1, out),
            ev("frame", token=f"{sc.order_id}-placed-1"),
            stage_event("PLACE", "done", f"released over zone {routing.bin}"),
            stage_event("VERIFY", "active", "re-capture → detect_points → point-in-zone"),
            frame=(
                {"placed": (item, routing.bin)}
                if first_ok
                else {"dropped": (item, miss_point)}
            ),
            delay=1.7,
            note="verify 1",
        )
    )
    script += _verify_beats(
        beat, sc, routing.bin, first_point, dest_rect, ok=first_ok, attempt=1
    )

    if not first_ok:
        # --- retry: pick + place once more, then verify again (loop.MAX_ATTEMPTS = 2)
        second_ok = out == "retry"
        second_point = good_point if second_ok else miss_point
        script.append(
            beat(
                stage_event("PICK", "active", "retry 2/2 · re-grasp from the table", attempt=2),
                commanded("PICK", 2),
                frame={"held": item, "dropped": None},
                delay=1.5,
                note="retry pick",
            )
        )
        script.append(
            beat(
                confirmed("PICK", 2, out),
                stage_event("PICK", "done", "item in gripper (attempt 2)"),
                stage_event("PLACE", "active", f"MotionExecutor · release over zone {routing.bin}"),
                commanded("PLACE", 2),
                delay=1.4,
                note="retry place",
            )
        )
        script.append(
            beat(
                confirmed("PLACE", 2, out),
                ev("frame", token=f"{sc.order_id}-placed-2"),
                stage_event("PLACE", "done", f"released over zone {routing.bin} (attempt 2)"),
                stage_event("VERIFY", "active", "re-capture → detect_points → point-in-zone"),
                frame=(
                    {"placed": (item, routing.bin)}
                    if second_ok
                    else {"dropped": (item, miss_point)}
                ),
                delay=1.7,
                note="verify 2",
            )
        )
        script += _verify_beats(
            beat, sc, routing.bin, second_point, dest_rect, ok=second_ok, attempt=2
        )

    # ---------------------------------------------------------------- ALERT + outcome
    verified = out in ("ok", "retry")
    alerts: list[dict[str, Any]] = []
    if routing.alert:
        alerts.append(ev("alert", **routing.alert))
    motion_alert = _motion_alert(out, item, routing.bin)
    if motion_alert:
        alerts.append(motion_alert)
    if verified:
        alerts.append(
            ev(
                "alert",
                name=f"Order fulfilled: {item} → zone {routing.bin}",
                description=(
                    f"Picked {item} and placed it in zone {routing.bin}; "
                    f"visual verification passed on attempt {2 if out == 'retry' else 1}."
                ),
                alert_type="order_fulfilled",
                severity="info",
            )
        )
    else:
        alerts.append(
            ev(
                "alert",
                name=f"Order FAILED: {item} → zone {routing.bin}",
                description=(
                    f"Could not verify {item} in zone {routing.bin} after 2 attempts — "
                    f"a human has to look."
                ),
                alert_type="order_failed",
                severity="error",
            )
        )
    status = {"ok": "fulfilled", "retry": "retried", "fail": "failed"}[out]
    detail = {
        "ok": f"{item} verified in zone {routing.bin} on the first attempt",
        "retry": f"{item} verified in zone {routing.bin} after one retry",
        "fail": f"{item} never verified in zone {routing.bin} — ERROR alert raised",
    }[out]
    script.append(
        beat(
            stage_event("ALERT", "active", "platform alert · robot.alerts.create"),
            delay=0.9,
            note="alert",
        )
    )
    script.append(
        beat(
            *alerts,
            stage_event("ALERT", "done" if verified else "failed", f"{len(alerts)} alert(s) dispatched"),
            ev("outcome", status=status, detail=detail, attempts=1 if out == "ok" else 2),
            delay=HOLD_SECONDS,  # `beat` scales it like every other delay
            note="outcome",
        )
    )
    return script


def _motion_alert(outcome: str, item: str, bin_label: str) -> dict[str, Any] | None:
    """An alert for a move we could not confirm: the honest half of "plan complete".

    Silence here would be the exact failure mode of the 2026-08-11 session: a run that
    reports success while nothing in the cell has proved the arm moved.
    """
    plan = MOTION_PLAN.get(outcome, {})
    bad = [(stage, att, state, errs) for (stage, att), (state, errs, _d) in plan.items()
           if state != "verified"]
    if not bad:
        return None
    mismatch = next((b for b in bad if b[2] == "mismatch"), None)
    if mismatch:
        stage, attempt, _state, errs = mismatch
        return ev(
            "alert",
            name=f"Motion MISMATCH on {stage} (attempt {attempt}) — {item}",
            description=(
                f"Commanded {stage} for {item} → zone {bin_label} completed on the wire, but "
                f"the encoders disagree: {motion_summary(errs)} (tolerance "
                f"±{DEFAULT_TOLERANCE_DEG:.0f}°). The move was NOT where it was asked to be."
            ),
            alert_type="motion_mismatch",
            severity="warning",
            category="technical",
        )
    stages = ", ".join(sorted({f"{s} #{a}" for s, a, st, _e in bad if st == "unverified"}))
    return ev(
        "alert",
        name="Motion UNVERIFIED — no servo telemetry",
        description=(
            f"{stages}: MotionExecutor published the plan and reported completion, but no "
            f"joint telemetry came back, so the cell CANNOT confirm the arm moved. This is "
            f"the 'ghost arm' signature (edge driver lost its serial binding). Treated as "
            f"not-completed, never as success."
        ),
        alert_type="motion_unverified",
        severity="warning",
        category="technical",
    )


def _verify_beats(
    beat: Callable[..., Beat],
    sc: Scenario,
    bin_label: str,
    point: Sequence[float],
    rect: Sequence[float] | None,
    *,
    ok: bool,
    attempt: int,
) -> list[Beat]:
    """The verification verdict, computed by the same point-in-rectangle test as perception."""
    inside = _inside(rect, point) if rect else ok
    reason = (
        f"{sc.item!r} verified inside zone {bin_label} at px=({point[0]:.0f}, {point[1]:.0f})"
        if inside
        else (
            f"{sc.item!r} is at px=({point[0]:.0f}, {point[1]:.0f}), OUTSIDE zone {bin_label}"
            f"{_rect_text(rect)}"
        )
    )
    return [
        beat(
            ev(
                "verification",
                ok=inside,
                reason=reason,
                point=[round(point[0], 1), round(point[1], 1)],
                bin=bin_label,
                attempt=attempt,
            ),
            stage_event(
                "VERIFY",
                "done" if inside else "failed",
                ("verdict: IN ZONE" if inside else "verdict: NOT IN ZONE → retry") ,
                attempt=attempt,
            ),
            delay=1.5,
            note=f"verdict {attempt}",
        )
    ]


def _inside(rect: Sequence[float], point: Sequence[float]) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= point[0] <= x1 and y0 <= point[1] <= y1


def _rect_text(rect: Sequence[float] | None) -> str:
    if not rect:
        return ""
    return f" [{rect[0]:.0f},{rect[1]:.0f} → {rect[2]:.0f},{rect[3]:.0f}]"


class MockDriver:
    """Runs :func:`build_script` against an app in a daemon thread, optionally forever."""

    def __init__(
        self,
        app: Any,
        *,
        mode: str = DEFAULT_MODE,
        outcome: str = DEFAULT_OUTCOME,
        loop: bool = True,
        speed: float = 1.0,
    ) -> None:
        self.app = app
        self.mode = check_mode(mode)
        self.outcome = outcome if outcome in OUTCOMES + (AUTO_OUTCOME,) else DEFAULT_OUTCOME
        self.loop = loop
        self.speed = speed
        #: The ending actually playing right now (differs from ``outcome`` under "auto").
        self.playing = self.outcome if self.outcome in OUTCOMES else OUTCOME_ROTATION[0]
        self._stop = threading.Event()
        self._restart = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._cycle: Iterator[int] = itertools.count()

    # ---------------------------------------------------------------- control
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mock-driver", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._restart.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def configure(self, *, mode: str | None = None, outcome: str | None = None) -> None:
        """Switch mode/outcome from the UI. Takes effect on the next (immediate) run."""
        with self._lock:
            if mode:
                self.mode = check_mode(mode)
            if outcome in OUTCOMES + (AUTO_OUTCOME,):
                self.outcome = outcome
        self.restart()

    def restart(self) -> None:
        self._restart.set()

    # ---------------------------------------------------------------- engine
    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                mode, outcome = self.mode, self.outcome
            index = next(self._cycle)
            pool = SCENARIOS[mode]
            scenario = pool[index % len(pool)]
            if outcome == AUTO_OUTCOME:
                # Unattended loop: tell the whole story rather than one third of it.
                outcome = OUTCOME_ROTATION[index % len(OUTCOME_ROTATION)]
            self.playing = outcome
            self._restart.clear()
            script = build_script(mode, outcome, scenario, speed=self.speed)
            for b in script:
                if self._stop.is_set() or self._restart.is_set():
                    break
                self.app.play(b)
                if self._restart.wait(timeout=max(0.0, b.delay)):
                    break
            if not self.loop and not self._restart.is_set():
                # One-shot: hold the final frame until told otherwise.
                while not self._stop.is_set() and not self._restart.wait(timeout=0.2):
                    pass
