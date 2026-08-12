"""Zero-shot open-vocab detection via the Cyberwave-hosted VLM (Gemini Robotics-ER).

This is the perception half of the loop seam: given an overhead
frame + an item name from the order, ask the hosted VLM to *point* at the item, and
return the pixel location that will feed planar homography -> table coords -> IK.

Real SDK surface (verified against cyberwave.mlmodels in S5):
    result = cw.mlmodels.run(model, image=<path|bytes|PIL>, prompt="red marker",
                             structured_task="detect_points")
    result.output == [{"point": [y, x], "label": "..."}]   # NOTE: [y, x] order, 0-1000 scale
  detect_boxes -> [{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}]  (also 0-1000).

The network call and the parsing are split so the parser is unit-testable with mocked
VLM output (no credits, no network): the pipeline's #1 risk validated offline. The
[y, x] / 0-1000 convention is encoded **once**, here, so nothing downstream re-derives it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# Structured-task ids from the SDK catalog (cw.mlmodels.list_structured_actions()).
TASK_POINTS = "detect_points"
TASK_BOXES = "detect_boxes"

# Gemini emits spatial coords on a 0..1000 grid regardless of image size.
VLM_SCALE = 1000.0


@dataclass(frozen=True)
class Detection:
    """One localized item, in **pixel** coordinates of the source frame.

    ``x, y`` is the point to grasp at (a point detection, or a box centre). ``nx, ny``
    keep the resolution-independent 0..1 form; ``box`` (x0, y0, x1, y1 px) is set for
    box detections. Homography consumes ``(x, y)``.
    """

    label: str
    x: float
    y: float
    nx: float
    ny: float
    box: tuple[float, float, float, float] | None = None


class _MLModels(Protocol):
    def run(self, model: Any, **kwargs: Any) -> Any: ...


class _Client(Protocol):
    @property
    def mlmodels(self) -> _MLModels: ...


def _clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def _unwrap(output: Any) -> list[dict[str, Any]]:
    """Coerce a VLM ``output`` into a list of item dicts.

    The backend returns either a bare list or a single-key wrapper like
    ``{"points": [...]}`` / ``{"detections": [...]}``, so unwrap the latter.
    """
    if output is None:
        return []
    if isinstance(output, dict):
        lists = [v for v in output.values() if isinstance(v, list)]
        if len(lists) == 1:
            return [x for x in lists[0] if isinstance(x, dict)]
        return []
    if isinstance(output, list):
        return [x for x in output if isinstance(x, dict)]
    return []


def parse_detections(
    output: Any,
    img_w: int,
    img_h: int,
    *,
    default_label: str = "",
) -> list[Detection]:
    """Parse a ``detect_points`` / ``detect_boxes`` output into pixel Detections.

    Pure: no network, no SDK. Handles both point (``{"point": [y, x]}``) and box
    (``{"box_2d": [ymin, xmin, ymax, xmax]}``) items on the 0..1000 scale; malformed
    entries are skipped rather than raising.
    """
    dets: list[Detection] = []
    for item in _unwrap(output):
        label = str(item.get("label") or default_label)
        try:
            if "point" in item:
                y_raw, x_raw = item["point"][:2]
                nx, ny = _clamp01(float(x_raw) / VLM_SCALE), _clamp01(float(y_raw) / VLM_SCALE)
                dets.append(Detection(label, nx * img_w, ny * img_h, nx, ny))
            elif "box_2d" in item:
                ymin, xmin, ymax, xmax = (float(v) for v in item["box_2d"][:4])
                cnx = _clamp01((xmin + xmax) / 2 / VLM_SCALE)
                cny = _clamp01((ymin + ymax) / 2 / VLM_SCALE)
                box = (
                    xmin / VLM_SCALE * img_w,
                    ymin / VLM_SCALE * img_h,
                    xmax / VLM_SCALE * img_w,
                    ymax / VLM_SCALE * img_h,
                )
                dets.append(Detection(label, cnx * img_w, cny * img_h, cnx, cny, box))
        except (ValueError, TypeError, IndexError):
            continue  # skip malformed entries; a partial detection list is still useful
    return dets


def pick_target(
    detections: list[Detection],
    item: str | None = None,
) -> Detection | None:
    """Choose the single detection to grasp: first label-match, else first overall.

    The VLM returns most-confident-first, so "first" is a sane default. ``item`` (if
    given) prefers a case-insensitive label match before falling back.
    """
    if not detections:
        return None
    if item:
        needle = item.strip().lower()
        for d in detections:
            if needle in d.label.lower():
                return d
    return detections[0]


def detect(
    client: _Client,
    image: Any,
    item: str,
    *,
    model: Any,
    image_size: tuple[int, int],
    task: str = TASK_POINTS,
) -> list[Detection]:
    """Run the hosted VLM on ``image`` for ``item`` -> pixel Detections.

    ``client`` only needs ``mlmodels.run(...)`` (a fake satisfies it offline).
    ``image`` is anything the SDK accepts (path / bytes / PIL). ``image_size`` is
    (width, height) in px, required to convert the VLM's 0..1000 coords to pixels.
    """
    result = client.mlmodels.run(model, image=image, prompt=item, structured_task=task)
    if getattr(result, "is_queued", None) and result.is_queued():
        raise RuntimeError(
            f"VLM run was queued (async cloud-node); polling not implemented. "
            f"Poll {getattr(result, 'poll_url', '?')} — Gemini is expected to run sync."
        )
    w, h = image_size
    return parse_detections(result.output, w, h, default_label=item)
