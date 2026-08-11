"""The three interchangeable demo modes — *same interface, swappable logic*.

The cell has three physical paper zones (A/B/C in ``hardware/config/bin-regions.json``)
which are also a **traffic light**: A green, B orange, C red. That one fact is what lets
the same arm, the same homography and the same verifier tell three different stories:

    order   the order names the destination bin              → route to ``order.bin``
    qc      the order names no bin; the VLM grades the item  → route by the grade
    fusion  the order names a bin AND a QC check runs        → a defect DIVERTS to red
            and raises an alert explaining why the item did not go where it was ordered

Everything here is **pure** — no I/O, no SDK, no camera — so the routing policy is
unit-testable on its own and the mock driver and a real run share exactly one
implementation of "where does this item go?" (:func:`route`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

#: The three modes, in demo order. ``order`` is today's shipped behaviour.
MODES: tuple[str, ...] = ("order", "qc", "fusion")
DEFAULT_MODE = "order"

#: QC grades, best → worst. ``pass`` is the only one that is not a defect.
GRADES: tuple[str, ...] = ("pass", "review", "reject")


class ModeError(ValueError):
    """Unknown mode, or a mode fed inputs it cannot route with."""


@dataclass(frozen=True)
class Zone:
    """A physical paper zone: a bin label the loop already understands + its QC meaning."""

    label: str  # matches bin-regions.json / Order.bin
    color: str  # "green" | "orange" | "red"
    css: str  # hex, used by the page and the synthetic frame so they agree
    grade: str  # the QC grade routed here
    meaning: str  # human line shown on the dashboard

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "color": self.color,
            "css": self.css,
            "grade": self.grade,
            "meaning": self.meaning,
        }


#: label → Zone. Labels are the *existing* bin labels, so QC mode needs no new calibration.
ZONES: dict[str, Zone] = {
    "A": Zone("A", "green", "#22c55e", "pass", "PASS — ready to ship"),
    "B": Zone("B", "orange", "#f59e0b", "review", "REVIEW — needs a human"),
    "C": Zone("C", "red", "#ef4444", "reject", "REJECT — quarantine"),
}

#: grade → zone label (the traffic light).
ZONE_FOR_GRADE: dict[str, str] = {z.grade: label for label, z in ZONES.items()}

REJECT_ZONE = ZONE_FOR_GRADE["reject"]


def zone_for(label: str | None) -> Zone | None:
    """The :class:`Zone` for a bin label, or ``None`` for an unmapped bin."""
    return ZONES.get(str(label).strip().upper()) if label else None


@dataclass(frozen=True)
class QCVerdict:
    """What the VLM said about the item it is holding up to the camera."""

    grade: str
    defect: str | None = None
    confidence: float = 0.0
    detail: str = ""

    def __post_init__(self) -> None:
        if self.grade not in GRADES:
            raise ModeError(f"unknown QC grade {self.grade!r}; expected one of {GRADES}")

    @property
    def defective(self) -> bool:
        """Anything that is not a clean pass is a defect for routing purposes."""
        return self.grade != "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade,
            "defect": self.defect,
            "confidence": round(float(self.confidence), 3),
            "detail": self.detail,
            "defective": self.defective,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QCVerdict":
        try:
            return cls(
                grade=str(data["grade"]).strip().lower(),
                defect=(str(data["defect"]) if data.get("defect") else None),
                confidence=float(data.get("confidence") or 0.0),
                detail=str(data.get("detail") or ""),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ModeError(f"bad QC verdict {dict(data)!r}: {e}") from e


@dataclass(frozen=True)
class Routing:
    """Where the item goes, and why — the one thing the three modes disagree about."""

    mode: str
    bin: str  # the destination the arm will actually place into
    requested_bin: str | None  # what the order asked for (None in qc mode)
    reason: str  # narration, shown large on the dashboard
    diverted: bool = False  # destination != requested_bin because of QC
    qc: QCVerdict | None = None
    alert: dict[str, str] | None = None  # alert payload to raise, if any

    @property
    def zone(self) -> Zone | None:
        return zone_for(self.bin)

    def to_dict(self) -> dict[str, Any]:
        zone = self.zone
        return {
            "mode": self.mode,
            "bin": self.bin,
            "requested_bin": self.requested_bin,
            "reason": self.reason,
            "diverted": self.diverted,
            "qc": self.qc.to_dict() if self.qc else None,
            "alert": dict(self.alert) if self.alert else None,
            "zone": zone.to_dict() if zone else None,
        }


def needs_qc(mode: str) -> bool:
    """Does this mode require a QC verdict before it can route?"""
    return check_mode(mode) in ("qc", "fusion")


def names_bin(mode: str) -> bool:
    """Does the incoming order carry a destination bin in this mode?"""
    return check_mode(mode) != "qc"


def check_mode(mode: str) -> str:
    """Normalize + validate a mode name."""
    m = str(mode or "").strip().lower()
    if m not in MODES:
        raise ModeError(f"unknown mode {mode!r}; expected one of {MODES}")
    return m


def _divert_alert(item: str, requested: str, zone: Zone, qc: QCVerdict) -> dict[str, str]:
    """The alert that explains a diversion (same kwargs shape as agent_service.alerts)."""
    defect = qc.defect or "failed QC"
    severity = "error" if qc.grade == "reject" else "warning"
    return {
        "name": f"QC diversion: {item} → zone {zone.label} ({zone.color})",
        "description": (
            f"{item} was ordered to bin {requested} but the QC check graded it "
            f"{qc.grade.upper()} ({defect}, confidence {qc.confidence:.0%}). "
            f"Diverted to the {zone.color} zone {zone.label}: {zone.meaning}."
        ),
        "alert_type": "qc_diverted",
        "severity": severity,
        "category": "business",
    }


def route(
    mode: str,
    *,
    item: str = "item",
    order_bin: str | None = None,
    qc: QCVerdict | None = None,
) -> Routing:
    """Decide the destination for one item under ``mode`` — the whole mode difference.

    ``order`` needs ``order_bin``; ``qc`` needs ``qc`` and ignores ``order_bin``;
    ``fusion`` needs both. Raises :class:`ModeError` when a required input is missing,
    which is a programming error at the call site, never a runtime surprise mid-demo
    (the mock driver and the live bridge both supply the inputs the mode declares via
    :func:`needs_qc` / :func:`names_bin`).
    """
    m = check_mode(mode)
    requested = str(order_bin).strip().upper() if order_bin else None

    if m == "order":
        if not requested:
            raise ModeError("mode 'order' needs the order's destination bin")
        zone = zone_for(requested)
        where = f" ({zone.color} zone)" if zone else ""
        return Routing(
            mode=m,
            bin=requested,
            requested_bin=requested,
            reason=f"order names bin {requested}{where}",
        )

    if qc is None:
        raise ModeError(f"mode {m!r} needs a QC verdict")

    if m == "qc":
        label = ZONE_FOR_GRADE[qc.grade]
        zone = ZONES[label]
        return Routing(
            mode=m,
            bin=label,
            requested_bin=None,
            reason=(
                f"QC graded {qc.grade.upper()}"
                f"{f' ({qc.defect})' if qc.defect else ''} → {zone.color} zone {label}"
            ),
            qc=qc,
        )

    # fusion: the order names a bin, QC can overrule it.
    if not requested:
        raise ModeError("mode 'fusion' needs the order's destination bin")
    if not qc.defective:
        return Routing(
            mode=m,
            bin=requested,
            requested_bin=requested,
            reason=f"QC PASS → honouring the ordered bin {requested}",
            qc=qc,
        )
    label = REJECT_ZONE if qc.grade == "reject" else ZONE_FOR_GRADE[qc.grade]
    zone = ZONES[label]
    return Routing(
        mode=m,
        bin=label,
        requested_bin=requested,
        reason=(
            f"defect ({qc.defect or qc.grade}) → DIVERTED from bin {requested} "
            f"to the {zone.color} zone {label}"
        ),
        diverted=label != requested,
        qc=qc,
        alert=_divert_alert(item, requested, zone, qc) if label != requested else None,
    )


MODE_BLURB: dict[str, str] = {
    "order": "The order names the destination bin. Pick it, place it, verify it.",
    "qc": "No bin in the order — the VLM grades the item and the traffic light routes it.",
    "fusion": "The order names a bin, but a QC defect diverts the item to red and alerts.",
}
