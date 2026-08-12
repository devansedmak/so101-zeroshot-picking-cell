"""Closed-loop visual verification: **the project thesis**.

After a place action we re-capture the overhead frame and ask: *is the item actually
in the target bin?* That yes/no is what turns the old open-loop pick-and-place into a
Physical AI agent: ok means a "fulfilled" alert, not-ok means
retry once, else an ERROR alert.

HOW (deliberately the simplest thing that works, ship first): no new VLM task type.
We reuse the *same* ``detect_points`` call as the pick (point at the item) and
then do a pure geometric test: does that pixel fall inside the target bin's rectangle?
Bins don't move in a fixed overhead view, so their pixel boxes are **calibration data**
(``hardware/config/bin-regions.json``), not perception.

Split mirrors detect.py: ``evaluate_placement`` is pure (unit-tested offline, no
credits) and ``verify_placement`` is the thin network wrapper. The [y, x] / 0-1000
convention is NOT re-derived here; it comes from ``detect.parse_detections``.

TODO(hardware): the bin pixel rectangles must be measured on the real overhead frame
once the camera is mounted (same session as the homography calibration).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .detect import TASK_POINTS, VLM_SCALE, Detection, detect, pick_target

# Where the surveyed bin rectangles live (git-friendly, next to the homography).
DEFAULT_BIN_REGIONS_PATH = Path("hardware/config/bin-regions.json")

# Fallback frame size: the VLM's own 0..1000 grid. Using it means bin rectangles are
# expressed resolution-independently, which is what an uncalibrated caller wants.
DEFAULT_FRAME_SIZE = (int(VLM_SCALE), int(VLM_SCALE))


class BinRegionsMissing(FileNotFoundError):
    """No bin-region calibration file yet (needs the real overhead frame)."""


class UnknownBinRegion(KeyError):
    """No surveyed pixel rectangle for the requested bin."""


@dataclass(frozen=True)
class BinRegion:
    """A bin as an axis-aligned **pixel** rectangle in the fixed overhead view.

    ``x0,y0`` is the top-left and ``x1,y1`` the bottom-right corner (normalized on
    construction, so swapped inputs are tolerated). Static per camera mount, so this is
    calibration data, persisted as JSON.
    """

    label: str
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        # frozen dataclass, so normalize via object.__setattr__ (cheaper than a factory).
        # Compute all four first: assigning x0 before reading it again would corrupt x1.
        x0, x1 = sorted((float(self.x0), float(self.x1)))
        y0, y1 = sorted((float(self.y0), float(self.y1)))
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "y1", y1)

    def contains(self, px: float, py: float) -> bool:
        """Is pixel ``(px, py)`` inside the rectangle? Edges count as inside."""
        return self.x0 <= px <= self.x1 and self.y0 <= py <= self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BinRegion":
        try:
            return cls(
                label=str(data["label"]),
                x0=float(data["x0"]),
                y0=float(data["y0"]),
                x1=float(data["x1"]),
                y1=float(data["y1"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"bad bin region {data!r}: {e}") from e


@dataclass(frozen=True)
class VerifyResult:
    """Did the item end up in the bin? ``reason`` is demo-narration/alert text."""

    ok: bool
    item: str
    bin_label: str
    reason: str
    detection: Detection | None = None  # the point that decided it (None = not seen)

    @property
    def point(self) -> tuple[float, float] | None:
        """The deciding pixel, or ``None`` if the item was never detected."""
        d = self.detection
        return None if d is None else (d.x, d.y)

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe form for alert metadata and session logs."""
        return {
            "ok": self.ok,
            "item": self.item,
            "bin": self.bin_label,
            "reason": self.reason,
            "point": list(self.point) if self.point else None,
            "label": self.detection.label if self.detection else None,
        }


def evaluate_placement(
    detections: list[Detection],
    item: str,
    bin_region: BinRegion,
) -> VerifyResult:
    """Pure verdict: is ``item``'s detected point inside ``bin_region``?

    Three outcomes, all non-raising: never detected (the item is gone / occluded /
    still in the gripper), detected outside the bin (dropped on the table or in the
    wrong bin), detected inside (success). Target selection reuses ``pick_target`` so
    verification and picking agree on which detection "is" the item.
    """
    target = pick_target(detections, item)
    if target is None:
        return VerifyResult(
            ok=False,
            item=item,
            bin_label=bin_region.label,
            reason=f"{item!r} not detected in the frame after placing (dropped or occluded)",
        )
    if not bin_region.contains(target.x, target.y):
        return VerifyResult(
            ok=False,
            item=item,
            bin_label=bin_region.label,
            reason=(
                f"{item!r} is at px=({target.x:.0f}, {target.y:.0f}), outside bin "
                f"{bin_region.label} [{bin_region.x0:.0f},{bin_region.y0:.0f} → "
                f"{bin_region.x1:.0f},{bin_region.y1:.0f}]"
            ),
            detection=target,
        )
    return VerifyResult(
        ok=True,
        item=item,
        bin_label=bin_region.label,
        reason=f"{item!r} verified inside bin {bin_region.label} at px=({target.x:.0f}, {target.y:.0f})",
        detection=target,
    )


def verify_placement(
    client: Any,
    image: Any,
    item: str,
    bin_region: BinRegion,
    *,
    model: Any = None,
    image_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
    task: str = TASK_POINTS,
) -> VerifyResult:
    """One VLM call on the re-captured ``image`` -> geometric verdict.

    ``client`` only needs ``mlmodels.run(...)`` (duck-typed, so tests use a fake and
    burn no credits). ``image_size`` is the (w, h) the bin rectangles were surveyed on.
    Leave it at the 0..1000 default only if the regions are on that grid too.
    ``model`` defaults to the pinned VLM slug, resolved lazily from the env.
    """
    if model is None:
        import os  # lazy: keeps this module import-cheap and env-free for pure tests

        model = os.getenv("CYBERWAVE_VLM_MODEL", "gemini-robotics-er")
    dets = detect(client, image, item, model=model, image_size=image_size, task=task)
    return evaluate_placement(dets, item, bin_region)


# ------------------------------------------------------------- calibration I/O
def save_bin_regions(
    path: str | Path,
    regions: Iterable[BinRegion] | Mapping[str, BinRegion],
    *,
    frame_size: tuple[int, int] | None = None,
) -> Path:
    """Persist bin rectangles as JSON (``frame_size`` records what they were surveyed on)."""
    values = list(regions.values()) if isinstance(regions, Mapping) else list(regions)
    payload = {
        "type": "bin_regions",
        "units": "pixel",
        "frame_size": list(frame_size) if frame_size else None,
        "regions": [r.to_dict() for r in values],
    }
    p = Path(path)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p


def load_bin_regions(path: str | Path = DEFAULT_BIN_REGIONS_PATH) -> dict[str, BinRegion]:
    """Load bin rectangles -> ``{label: BinRegion}``.

    Raises :class:`BinRegionsMissing` (a ``FileNotFoundError``) with a clear message if
    the calibration hasn't been done yet; callers degrade to unverified rather than
    crashing the run.
    """
    p = Path(path)
    if not p.exists():
        raise BinRegionsMissing(
            f"no bin-region calibration at {p} — survey the bin pixel rectangles on a "
            f"real overhead frame first (see src/perception/verify.py TODO)"
        )
    data = json.loads(p.read_text())
    raw = data.get("regions", data) if isinstance(data, dict) else data
    regions = [BinRegion.from_dict(r) for r in raw]
    return {r.label: r for r in regions}


def bin_region_for(regions: Mapping[str, BinRegion], bin_label: str) -> BinRegion:
    """Look up a bin's rectangle, raising :class:`UnknownBinRegion` if unsurveyed."""
    region = regions.get(bin_label)
    if region is None:
        raise UnknownBinRegion(f"no surveyed region for bin {bin_label!r}; known: {sorted(regions)}")
    return region
