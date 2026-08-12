"""Grasp axis: which way is the object lying on the table?

WHY THIS MODULE EXISTS. ``control.ik.solve_ik`` used to pin wrist_roll (joint "5") to
0°, so the jaws always closed along one fixed direction. A long thin object (a marker,
a pen, a screwdriver) lying across that direction gets grasped **end-on**: the fingers
close on its two flanks near a tip and it squirts out. The VLM tells us *where* the item
is (detect.py) but not *how it is turned*; this module answers that from the same
overhead frame, so IK can roll the wrist to close ACROSS the object's long axis.

Deliberately geometric, not learned: one local crop around the point the VLM already
returned is enough for an axis. No extra model call, no credits, no network, so the
whole thing is unit-testable on synthesised frames (tests/test_orient.py).

Figure/ground here is **colour distance from the crop's border colour**, not luminance.
The demo table carries a white calibration sheet AND coloured paper drop zones, so
"dark blob on a light table" is false half the time, while "different colour from
whatever surrounds it" holds on both. The distance is measured in Lab, where a
perceptual colour difference is a plain Euclidean norm.

OpenCV is imported lazily inside the functions (same rule as locate.py/verify.py), so
importing this module (and therefore the agent loop) never requires it.

Contract: every helper returns ``None`` rather than raising when the scene doesn't
support an answer (flat crop, blob not elongated, point on no blob). A missing axis must
degrade to the old fixed-axis grasp, never fail an order. See locate.locate_item.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .homography import Homography, Point

__all__ = ["MIN_ELONGATION", "axis_angle_table", "long_axis_pixels"]

#: A blob at least this much longer than it is wide has a meaningful long axis. Below it
#: the minAreaRect orientation is noise (a square/round item is grasp-angle agnostic
#: anyway), so we say "no axis" and let IK keep its default roll.
MIN_ELONGATION: float = 1.6

#: Peak Lab distance (0-255 channels) below which the crop is one flat colour; there is
#: no figure to segment and Otsu would happily threshold sensor noise into a blob.
_MIN_COLOUR_DISTANCE: float = 12.0

#: Smallest usable crop side in px (below this a window is all border and no object).
_MIN_CROP_PX: int = 16

_EPS = 1e-9


def _as_bgr(image: Any) -> np.ndarray | None:  # noqa: ANN401, path or decoded frame
    """Decoded BGR frame from a path or an in-memory array; ``None`` if unreadable."""
    import cv2  # noqa: PLC0415, optional/heavy: probed at call time

    if isinstance(image, np.ndarray):
        img = image
    elif isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:  # missing, empty or not an image; caller degrades to no axis
            return None
    else:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, :3]


def _figure_mask(crop: np.ndarray) -> np.ndarray | None:
    """Binary figure/ground mask of ``crop``, by Lab distance from its border colour.

    The border ring of the window is, by construction, table; the object we care about
    is at the window's centre. Its median is a robust background estimate even when a
    neighbouring object clips a corner.
    """
    import cv2  # noqa: PLC0415, optional/heavy: probed at call time

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]], axis=0)
    bg = np.median(border, axis=0)

    dist = np.linalg.norm(lab - bg, axis=2)
    peak = float(dist.max())
    if peak < _MIN_COLOUR_DISTANCE:
        return None

    d8 = np.clip(dist / peak * 255.0, 0.0, 255.0).astype(np.uint8)
    d8 = cv2.medianBlur(d8, 3)  # JPEG speckle would otherwise survive Otsu as specks
    _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _contour_containing(contours: Any, xy: tuple[float, float]) -> Any:  # noqa: ANN401
    """The contour the detected point falls inside, or ``None``.

    NOT "the biggest contour": items on the demo table sit close together, and a bigger
    neighbour inside the same window would otherwise steal the axis and rotate the wrist
    to grasp the wrong object's orientation.
    """
    import cv2  # noqa: PLC0415, optional/heavy: probed at call time

    best = None
    best_area = -1.0
    for c in contours:
        if cv2.pointPolygonTest(c, (float(xy[0]), float(xy[1])), False) < 0:
            continue
        area = float(cv2.contourArea(c))
        if area > best_area:
            best, best_area = c, area
    return best


def long_axis_pixels(
    image: Any,  # noqa: ANN401, a saved frame (path) or an already-decoded BGR array
    center_px: Point,
    win: int = 160,
) -> tuple[Point, Point] | None:
    """Long axis of the object at ``center_px``, as two endpoints in FRAME pixels.

    Crops a ``win``×``win`` window around the detected point (clipped at the frame
    borders), separates figure from ground by colour distance (:func:`_figure_mask`),
    keeps the contour that *contains* the point, and fits a minimum-area rectangle. The
    two returned points are the midpoints of that rectangle's **short** edges, so the
    segment between them runs along the object's length.

    Returns ``None`` (meaning "no meaningful axis, keep the default grasp") when the
    frame is unreadable, the window is degenerate, the crop is one flat colour, no blob
    contains the point, or the blob is rounder than :data:`MIN_ELONGATION`.
    """
    import cv2  # noqa: PLC0415, optional/heavy: probed at call time

    img = _as_bgr(image)
    if img is None:
        return None
    h, w = img.shape[:2]

    cx, cy = float(center_px[0]), float(center_px[1])
    half = max(_MIN_CROP_PX // 2, int(win) // 2)
    x0, y0 = int(max(0, math.floor(cx - half))), int(max(0, math.floor(cy - half)))
    x1, y1 = int(min(w, math.ceil(cx + half))), int(min(h, math.ceil(cy + half)))
    if x1 - x0 < _MIN_CROP_PX or y1 - y0 < _MIN_CROP_PX:
        return None

    mask = _figure_mask(img[y0:y1, x0:x1])
    if mask is None:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = _contour_containing(contours, (cx - x0, cy - y0))
    if cnt is None:
        return None

    rect = cv2.minAreaRect(cnt)
    side_a, side_b = float(rect[1][0]), float(rect[1][1])
    long_side, short_side = max(side_a, side_b), min(side_a, side_b)
    if short_side < _EPS or long_side / short_side < MIN_ELONGATION:
        return None

    # boxPoints returns the 4 corners in order, so corner i -> i+1 is an edge. The two
    # SHORT edges cap the object's ends; the segment joining their midpoints IS the axis.
    box = cv2.boxPoints(rect)
    edges = [float(np.linalg.norm(box[(i + 1) % 4] - box[i])) for i in range(4)]
    s = int(np.argmin(edges))
    m0 = (box[s] + box[(s + 1) % 4]) / 2.0
    m1 = (box[(s + 2) % 4] + box[(s + 3) % 4]) / 2.0

    return (
        (float(m0[0]) + x0, float(m0[1]) + y0),
        (float(m1[0]) + x0, float(m1[1]) + y0),
    )


def axis_angle_table(homography: Homography, p0: Point, p1: Point) -> float:
    """Heading of the segment ``p0`` -> ``p1`` on the TABLE plane, in degrees mod 180.

    Both endpoints are projected through the fitted pixel -> table homography *before* the
    angle is taken. Measuring the angle in table mm rather than in pixels is what makes
    this correct for free: the image y axis points down while the table's +Y points up
    (so an image-space angle has the wrong sign), and the overhead camera is never
    exactly perpendicular (so pixel angles are sheared by perspective). Projecting both
    ends absorbs the flip and the perspective; the residual error is the homography's
    own, which the calibration already bounds at < 2 cm.

    Mod 180 because an axis has no head or tail: ``p0`` -> ``p1`` and ``p1`` -> ``p0`` are the
    same axis, and a parallel-jaw gripper cannot tell them apart either.
    """
    x0, y0 = homography.project(p0)
    x1, y1 = homography.project(p1)
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
