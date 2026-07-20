"""End-to-end seam proof: perception → control, fully offline with synthetic data.

Proves the composed pipeline the loop was missing — VLM output → pixel → table
homography → IK → executable MotionPlan — without a camera, robot, or network.
Nothing here calls the SDK; the VLM output and the homography are hand-built.
Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (see runbook).
"""

from __future__ import annotations

import pytest

from src.agent_service.poses import pick_plan_from_table
from src.control import validate_plan
from src.perception import Homography, parse_detections, pick_target

# 640×480 overhead frame; an affine pixel→table calibration whose image maps onto
# a reachable patch of the table (X∈[100,200] mm, Y∈[-100,100] mm).
IMG_W, IMG_H = 640, 480
_PIXELS = [(0, 0), (640, 0), (640, 480), (0, 480)]
_TABLE_MM = [(100, -100), (200, -100), (200, 100), (100, 100)]


def test_detect_to_homography_to_ik_yields_a_valid_plan():
    homography = Homography.fit(_PIXELS, _TABLE_MM)

    # Synthetic VLM detection at the image centre: [y, x] on the 0..1000 grid.
    vlm_output = [{"point": [500, 500], "label": "red marker"}]
    detections = parse_detections(vlm_output, IMG_W, IMG_H, default_label="red marker")
    target = pick_target(detections, item="red marker")
    assert target is not None
    assert (target.x, target.y) == pytest.approx((320.0, 240.0))  # centre pixel

    # Pixel → table mm → IK-driven pick plan.
    table_x, table_y = homography.project((target.x, target.y))
    assert (table_x, table_y) == pytest.approx((150.0, 0.0), abs=1e-6)  # patch centre

    plan = pick_plan_from_table(table_x, table_y)
    assert validate_plan(plan) == []  # the composed pipeline is executable


def test_seam_holds_across_the_calibrated_patch():
    # Every corner detection of the calibrated image must also compose to a valid plan.
    homography = Homography.fit(_PIXELS, _TABLE_MM)
    for px, py in [(64, 48), (576, 48), (576, 432), (64, 432)]:
        det = parse_detections(
            [{"point": [py / IMG_H * 1000, px / IMG_W * 1000], "label": "eraser"}],
            IMG_W,
            IMG_H,
        )
        tx, ty = homography.project((det[0].x, det[0].y))
        assert validate_plan(pick_plan_from_table(tx, ty)) == []
