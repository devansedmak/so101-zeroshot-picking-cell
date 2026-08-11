"""Unit tests for grasp-axis estimation (pure OpenCV maths, no camera, no VLM).

There is no hardware in the loop here, so the proof that rotation-invariant grasping
works is this file: frames are **synthesised** with a filled rotated rectangle whose
angle we chose, and the module must recover that angle — on a white calibration sheet
AND on a coloured paper zone, including the case where object and background have the
same *luminance* and differ only in colour (which is precisely why the segmentation is
colour-distance based and not a grey threshold).

Angles are compared modulo 180° throughout: an axis has no head or tail.
Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (see runbook).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.perception.homography import Homography
from src.perception.orient import axis_angle_table, long_axis_pixels

cv2 = pytest.importorskip("cv2")

IMG_W, IMG_H = 400, 400
CENTER = (200.0, 200.0)

WHITE_SHEET = (255, 255, 255)  # BGR — the calibration sheet
PAPER_ZONE = (200, 120, 60)    # BGR — a coloured drop-zone sheet
MARKER = (40, 40, 200)         # BGR — a red marker-ish object

# Same L* (128), opposite chroma: a luminance threshold sees NO edge between these two,
# a Lab colour-distance threshold sees a huge one. Asserted in the test below.
EQUAL_L_BG = (200, 120, 60)
EQUAL_L_FG = (0, 0, 240)

# Pixel→table map used for the table-space angle: a pure similarity with the y FLIP the
# real overhead rig has (image y grows downward, table +Y grows "up"). Uniform scale, so
# any angle error can only come from the code, not from an anisotropic calibration.
_MM_PER_PX = 0.5
_PIXELS = [(0, 0), (IMG_W, 0), (IMG_W, IMG_H), (0, IMG_H)]
_TABLE_MM = [
    (100.0 + px * _MM_PER_PX, 100.0 - py * _MM_PER_PX) for px, py in _PIXELS
]


def _homography() -> Homography:
    return Homography.fit(_PIXELS, _TABLE_MM)


def _corners(center, length: float, width: float, angle_deg: float) -> np.ndarray:
    """4 integer corners of a rectangle of ``length``×``width`` rotated by ``angle_deg``.

    ``angle_deg`` is measured in IMAGE convention (x right, y down) so the long axis
    direction is (cos θ, sin θ) — that is the ground truth the tests assert against.
    """
    t = math.radians(angle_deg)
    d = np.array([math.cos(t), math.sin(t)])
    n = np.array([-math.sin(t), math.cos(t)])
    c = np.array(center, dtype=float)
    pts = [
        c + d * length / 2 + n * width / 2,
        c + d * length / 2 - n * width / 2,
        c - d * length / 2 - n * width / 2,
        c - d * length / 2 + n * width / 2,
    ]
    return np.round(np.array(pts)).astype(np.int32)


def _scene(bg, shapes) -> np.ndarray:
    """A synthetic frame: flat ``bg`` plus a list of (corners, colour) filled polygons."""
    img = np.full((IMG_H, IMG_W, 3), bg, dtype=np.uint8)
    for corners, colour in shapes:
        cv2.fillPoly(img, [corners], colour)
    return img


def _rect_scene(angle_deg: float, bg=WHITE_SHEET, fg=MARKER, length=140, width=30):
    return _scene(bg, [(_corners(CENTER, length, width, angle_deg), fg)])


def _pixel_angle(p0, p1) -> float:
    """Image-space heading of the returned axis, degrees mod 180."""
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0


def _angle_error(a: float, b: float) -> float:
    """Smallest difference between two axis angles, respecting the 180° wrap."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


ANGLES = [0.0, 30.0, 45.0, 90.0, 135.0]
TOLERANCE_DEG = 3.0


# --- the axis is recovered, in pixels and on the table --------------------


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("bg", [WHITE_SHEET, PAPER_ZONE], ids=["white-sheet", "paper-zone"])
def test_long_axis_recovers_the_drawn_angle_in_pixels(angle, bg):
    ends = long_axis_pixels(_rect_scene(angle, bg=bg), CENTER)
    assert ends is not None
    assert _angle_error(_pixel_angle(*ends), angle) < TOLERANCE_DEG


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("bg", [WHITE_SHEET, PAPER_ZONE], ids=["white-sheet", "paper-zone"])
def test_axis_angle_table_recovers_the_drawn_angle_in_mm(angle, bg):
    # The rig flips y (image down vs table up), so an image angle θ is -θ on the table.
    # Getting this identity right is the whole point of angling in table space.
    ends = long_axis_pixels(_rect_scene(angle, bg=bg), CENTER)
    assert ends is not None
    table_deg = axis_angle_table(_homography(), *ends)
    assert _angle_error(table_deg, -angle) < TOLERANCE_DEG
    assert 0.0 <= table_deg < 180.0  # normalized: an axis has no head or tail


def test_axis_angle_table_is_the_same_whichever_end_comes_first():
    ends = long_axis_pixels(_rect_scene(30.0), CENTER)
    assert ends is not None
    h = _homography()
    assert axis_angle_table(h, *ends) == pytest.approx(axis_angle_table(h, ends[1], ends[0]))


def test_the_endpoints_straddle_the_detected_point():
    # Midpoints of the SHORT edges ⇒ the two ends of the object, not two corners:
    # their mean must land back on the blob's centre.
    ends = long_axis_pixels(_rect_scene(45.0), CENTER)
    assert ends is not None
    mid = ((ends[0][0] + ends[1][0]) / 2, (ends[0][1] + ends[1][1]) / 2)
    assert mid == pytest.approx(CENTER, abs=2.0)


# --- figure/ground is COLOUR distance, not luminance ---------------------


def test_object_and_background_with_equal_luminance_are_still_separated():
    lab = cv2.cvtColor(np.uint8([[list(EQUAL_L_BG), list(EQUAL_L_FG)]]), cv2.COLOR_BGR2LAB)
    bg_l, fg_l = int(lab[0, 0, 0]), int(lab[0, 1, 0])
    assert abs(bg_l - fg_l) <= 2  # a grey/Otsu-on-luminance threshold would see nothing

    ends = long_axis_pixels(_rect_scene(30.0, bg=EQUAL_L_BG, fg=EQUAL_L_FG), CENTER)
    assert ends is not None
    assert _angle_error(_pixel_angle(*ends), 30.0) < TOLERANCE_DEG


# --- the elongation gate -------------------------------------------------


def test_a_round_blob_has_no_meaningful_axis():
    img = np.full((IMG_H, IMG_W, 3), WHITE_SHEET, dtype=np.uint8)
    cv2.circle(img, (int(CENTER[0]), int(CENTER[1])), 40, MARKER, -1)
    assert long_axis_pixels(img, CENTER) is None


def test_a_barely_elongated_blob_is_rejected_too():
    # 1.2 : 1 is below MIN_ELONGATION (1.6) — its minAreaRect angle would be noise.
    assert long_axis_pixels(_rect_scene(30.0, length=60, width=50), CENTER) is None


# --- the contour must CONTAIN the point, not merely be the largest -------


def test_a_bigger_neighbour_in_the_window_does_not_steal_the_axis():
    """Two objects in one crop: the one under the VLM's point wins, even if smaller."""
    target = (_corners(CENTER, 60, 16, 20.0), MARKER)
    # Deliberately longer, thicker, and at a very different angle: "largest contour"
    # would pick this one and roll the wrist ~70° off.
    neighbour = (_corners((255.0, 145.0), 120, 30, 110.0), (60, 160, 60))
    ends = long_axis_pixels(_scene(WHITE_SHEET, [target, neighbour]), CENTER)

    assert ends is not None
    assert _angle_error(_pixel_angle(*ends), 20.0) < TOLERANCE_DEG
    assert _angle_error(_pixel_angle(*ends), 110.0) > 45.0  # definitely not the neighbour


def test_a_point_on_bare_table_has_no_axis():
    # The VLM pointed next to the object, not at it: no contour contains the point.
    assert long_axis_pixels(_rect_scene(30.0), (60.0, 60.0)) is None


# --- graceful degradation (locate.py relies on this) ---------------------


def test_a_flat_crop_has_no_axis():
    assert long_axis_pixels(np.full((IMG_H, IMG_W, 3), WHITE_SHEET, np.uint8), CENTER) is None


def test_a_saved_frame_is_read_from_disk(tmp_path):
    path = tmp_path / "overhead.png"  # PNG: no JPEG ringing around the drawn edges
    cv2.imwrite(str(path), _rect_scene(135.0))
    ends = long_axis_pixels(path, CENTER)
    assert ends is not None
    assert _angle_error(_pixel_angle(*ends), 135.0) < TOLERANCE_DEG


@pytest.mark.parametrize("bad", ["/nonexistent/frame.jpg", None, 42])
def test_an_unreadable_image_degrades_to_no_axis(bad):
    # locate_item hands whatever it has straight in; "no axis" must never be an exception.
    assert long_axis_pixels(bad, CENTER) is None


def test_a_window_clipped_at_the_frame_border_still_works():
    # Object hard against the top-left corner: the crop is clipped on two sides.
    corner = (70.0, 70.0)
    img = _scene(WHITE_SHEET, [(_corners(corner, 100, 24, 45.0), MARKER)])
    ends = long_axis_pixels(img, corner)
    assert ends is not None
    assert _angle_error(_pixel_angle(*ends), 45.0) < TOLERANCE_DEG


def test_the_window_size_bounds_what_is_looked_at():
    # A tiny window sees only the middle of the object: still the same axis, and the
    # endpoints are pulled inside the window (proving `win` is honoured).
    ends = long_axis_pixels(_rect_scene(0.0), CENTER, win=60)
    assert ends is not None
    assert _angle_error(_pixel_angle(*ends), 0.0) < TOLERANCE_DEG
    assert all(abs(p[0] - CENTER[0]) <= 31 for p in ends)
