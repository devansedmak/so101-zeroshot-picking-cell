"""End-to-end seam proof: perception -> control, fully offline with synthetic data.

Proves the composed pipeline the loop was missing: VLM output -> pixel -> table
homography -> IK -> executable MotionPlan, without a camera, robot, or network.
Nothing here calls the SDK; the VLM output and the homography are hand-built.

The second half of this file drives the **real runnable entrypoint**
(``run_order.main``), because the first half passing while the loop still picked from
the hardcoded pose table is exactly the gap that made the headline claim unproven.
It asserts the ``--perceive`` resolution order: calibration present means the IK path,
calibration/detection missing means the hardcoded path **with a warning**, and an unknown
item means a recorded failed outcome rather than an exception.
Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import json
import re

import pytest

from src.agent_service import run_order
from src.agent_service.poses import PICK_POSES, pick_choice_from_table, pick_plan_from_table
from src.control import validate_plan
from src.perception import Homography, parse_detections, pick_target

# 640×480 overhead frame; an affine pixel -> table calibration whose image maps onto
# a reachable patch of the table (X∈[100,200] mm, Y∈[-100,100] mm).
IMG_W, IMG_H = 640, 480
_PIXELS = [(0, 0), (640, 0), (640, 480), (0, 480)]
_TABLE_MM = [(100, -100), (200, -100), (200, 100), (100, 100)]

# One canned VLM answer: [y, x] on the 0..1000 grid -> px (480, 120) -> table (175, -50) mm
# under the calibration above (see tests/test_locate.py for the independent arithmetic).
_CANNED = [{"point": [250, 750], "label": "red marker"}]
_PERCEIVED_SAY = "picking at table (175, -50) mm"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Ramping is instantaneous so the entrypoint tests don't wait on wall-clock time."""
    monkeypatch.setattr("src.control.motion.time.sleep", lambda _s: None)


def test_detect_to_homography_to_ik_yields_a_valid_plan():
    homography = Homography.fit(_PIXELS, _TABLE_MM)

    # Synthetic VLM detection at the image centre: [y, x] on the 0..1000 grid.
    vlm_output = [{"point": [500, 500], "label": "red marker"}]
    detections = parse_detections(vlm_output, IMG_W, IMG_H, default_label="red marker")
    target = pick_target(detections, item="red marker")
    assert target is not None
    assert (target.x, target.y) == pytest.approx((320.0, 240.0))  # centre pixel

    # Pixel -> table mm -> IK-driven pick plan.
    table_x, table_y = homography.project((target.x, target.y))
    assert (table_x, table_y) == pytest.approx((150.0, 0.0), abs=1e-6)  # patch centre

    plan = pick_plan_from_table(table_x, table_y)
    assert validate_plan(plan) == []  # the composed pipeline is executable


def test_a_measured_object_axis_reaches_the_wrist_roll_of_the_plan():
    """The orientation half of the seam: perception's axis must survive into the pose.

    Without this, the axis is measured and then quietly dropped, which looks identical
    to today's behaviour right up until a long object squirts out of the jaws.
    """
    pose_of = lambda plan: next(a.pose for a in plan.actions if a.type == "set_pose")  # noqa: E731

    assert pose_of(pick_plan_from_table(150.0, 0.0))["5"] == 0.0  # no axis means unchanged
    rolled = pose_of(pick_plan_from_table(150.0, 0.0, 0.0))
    assert rolled["5"] == pytest.approx(-90.0)  # object lying along the reach direction
    assert validate_plan(pick_plan_from_table(150.0, 0.0, 0.0)) == []

    choice = pick_choice_from_table(150.0, 0.0, axis_deg=0.0)
    assert choice.axis_deg == 0.0  # provenance: WHY the wrist is rolled
    assert choice.to_dict()["axis_deg"] == 0.0


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


# --- the seam through the REAL entrypoint: run_order --perceive ----------


def _offline_perceive_argv(
    tmp_path,
    *,
    calibrated: bool,
    detections=_CANNED,
    item="red marker",
    frame_image=None,
):
    """``--perceive`` argv that is guaranteed offline: canned VLM output, saved frame.

    No camera (``--frame``), no VLM call (``--detections`` selects the canned client), no
    connection (dry-run is the default), and no image decode (``--frame-size``).
    ``frame_image`` (a BGR array) writes a REAL frame instead of the stub, for the tests
    that need the grasp-axis estimator to actually see something.
    """
    if frame_image is None:
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"")  # never opened: --frame-size makes the size explicit
    else:
        import cv2  # noqa: PLC0415, only the axis tests need a decodable frame

        frame = tmp_path / "frame.png"  # PNG: no JPEG ringing on the drawn edges
        cv2.imwrite(str(frame), frame_image)
    dets = tmp_path / "dets.json"
    dets.write_text(json.dumps(detections))
    calib = tmp_path / "homography.json"
    if calibrated:
        Homography.fit(_PIXELS, _TABLE_MM).save(calib, units="mm")
    return [
        "--item", item, "--bin", "A", "--perceive",
        "--homography", str(calib),
        "--frame", str(frame),
        "--frame-size", f"{IMG_W}x{IMG_H}",
        "--detections", str(dets),
    ]


def test_perceive_picks_the_ik_path_when_a_calibration_is_present(tmp_path, capsys):
    rc = run_order.main(_offline_perceive_argv(tmp_path, calibrated=True))
    out = capsys.readouterr().out

    assert rc == 0
    assert "perceived picking enabled" in out
    assert "'red marker' at px=(480, 120) → table (175, -50) mm" in out
    assert _PERCEIVED_SAY in out  # the plan came from IK, not from PICK_POSES
    assert "picking red marker" not in out
    assert "✅ fulfilled: red marker → bin A" in out


def test_perceive_falls_back_to_hardcoded_when_uncalibrated(tmp_path, capsys):
    rc = run_order.main(_offline_perceive_argv(tmp_path, calibrated=False))
    out = capsys.readouterr().out

    assert rc == 0
    assert "⚠ perceived picking disabled" in out  # says WHY: no calibration file
    assert "homography.json" in out
    assert "picking red marker" in out  # the hardcoded plan ran instead
    assert _PERCEIVED_SAY not in out
    assert "✅ fulfilled: red marker → bin A" in out


def test_perceive_falls_back_when_the_item_is_not_detected(tmp_path, capsys):
    rc = run_order.main(_offline_perceive_argv(tmp_path, calibrated=True, detections=[]))
    out = capsys.readouterr().out

    assert rc == 0
    assert "⚠ perceived pick unavailable: ItemNotFound" in out
    assert "picking red marker" in out and _PERCEIVED_SAY not in out
    assert "✅ fulfilled" in out


def test_perceive_fulfils_an_item_that_has_no_hardcoded_pose(tmp_path, capsys):
    """The zero-shot payoff: an item absent from PICK_POSES is still pickable."""
    assert "stapler" not in PICK_POSES
    canned = [{"point": [250, 750], "label": "stapler"}]
    rc = run_order.main(
        _offline_perceive_argv(tmp_path, calibrated=True, detections=canned, item="stapler")
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert _PERCEIVED_SAY in out
    assert "✅ fulfilled: stapler → bin A" in out


def test_perceive_undetected_and_unhardcoded_fails_the_order_without_raising(tmp_path, capsys):
    assert "ghost" not in PICK_POSES  # nothing to fall back to
    rc = run_order.main(
        _offline_perceive_argv(tmp_path, calibrated=True, detections=[], item="ghost")
    )
    out = capsys.readouterr().out

    assert rc == 1  # a recorded failure, not a traceback
    assert "⚠ perceived pick unavailable: ItemNotFound" in out
    assert "❌ failed: ghost → bin A" in out and "UnknownItem" in out


def _frame_with_a_bar(angle_deg: float):
    """640×480 white frame with one red bar at pixel (480, 120), where _CANNED points."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import math

    img = np.full((IMG_H, IMG_W, 3), (255, 255, 255), np.uint8)
    t = math.radians(angle_deg)
    d, n = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    c = np.array([480.0, 120.0])
    box = np.round(
        np.array([c + d * 60 + n * 13, c + d * 60 - n * 13, c - d * 60 - n * 13, c - d * 60 + n * 13])
    ).astype(np.int32)
    cv2.fillPoly(img, [box], (40, 40, 200))
    return img


def test_perceive_rolls_the_wrist_to_the_seen_object_axis(tmp_path, capsys):
    """The headline of oriented grasping, through the real entrypoint: a bar lying at an
    angle produces a NON-zero wrist_roll, and the run log says which angle it saw."""
    rc = run_order.main(
        _offline_perceive_argv(tmp_path, calibrated=True, frame_image=_frame_with_a_bar(174.0))
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert _PERCEIVED_SAY in out and "axis 165°" in out  # ≈174° in px, sheared by the calibration
    # The bar lies along the reach direction, so the wrist must roll ~a quarter turn to
    # close across it; the old fixed 5=+0.0° would have grasped it end-on.
    rolls = [float(v) for v in re.findall(r"5=([-+][\d.]+)°", out)]
    assert any(abs(v) > 45.0 for v in rolls)
    assert "✅ fulfilled" in out


def test_perceive_fails_the_order_when_the_grasp_angle_is_out_of_range(tmp_path, capsys, monkeypatch):
    """An angle the wrist cannot reach must FAIL loudly, not fall back to a pose that
    would grasp a seen item at a knowingly wrong angle. Reproduced by restoring the
    pre-widening ±60° roll limit (hardware/config/joint-ranges.md)."""
    from src.control.motion import DEFAULT_JOINT_LIMITS

    monkeypatch.setitem(DEFAULT_JOINT_LIMITS, "5", (-60, 60))
    rc = run_order.main(
        _offline_perceive_argv(tmp_path, calibrated=True, frame_image=_frame_with_a_bar(174.0))
    )
    out = capsys.readouterr().out

    assert rc == 1
    assert "✋ grasp angle unreachable" in out and "rotate the object" in out
    assert "❌ failed: red marker → bin A" in out
    assert "picking red marker" not in out  # NOT silently degraded to the hardcoded pose


def test_fulfill_order_records_which_pick_path_it_took():
    """The loop's own contract: provenance is recorded, the default stays hardcoded."""
    from src.agent_service import fulfill_order, pick_choice_from_table
    from src.control import MotionExecutor
    from src.wms_mock import Order

    class _Null:
        class joints:  # noqa: N801, duck-type for MotionExecutor
            @staticmethod
            def set(*a, **k):
                return None

    order = Order("red marker", "A")
    ex = MotionExecutor(_Null(), dry_run=True)

    default = fulfill_order(order, ex, dry_run_alert=True)
    assert default.ok and default.pick_source == "hardcoded"
    assert default.pick_target_mm is None and not default.perceived

    perceived = fulfill_order(
        order,
        ex,
        dry_run_alert=True,
        resolve_pick=lambda o: pick_choice_from_table(175.0, -50.0),
    )
    assert perceived.ok and perceived.perceived
    assert perceived.pick_target_mm == (175.0, -50.0)


def test_default_run_is_untouched_by_the_new_seam(capsys):
    """No --perceive means byte-for-byte the old behaviour: hardcoded poses, no warnings."""
    rc = run_order.main(["--item", "red marker", "--bin", "A"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "picking red marker" in out
    assert "perceive" not in out and "⚠" not in out

