"""Unit tests for tools/check_reach.py: the zone-reachability gate (pure math, no hardware).

Two things are being protected here:
1. the reach BAND is derived from the repo's own `solve_ik`/`in_limits`, so it can never
   silently disagree with the solver the demo actually runs;
2. the CURRENT surveyed layout must keep failing, and the recommended slide must keep
   passing. This file is the regression that stops the geometry problem coming back
   unnoticed after someone re-surveys the mat.

Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.control.ik import PICK_Z, Unreachable, forward_kinematics, in_limits, solve_ik
from tools import check_reach as cr

REPO_ROOT = Path(__file__).resolve().parents[1]
HOMOGRAPHY = REPO_ROOT / "hardware" / "config" / "homography.json"
BIN_REGIONS = REPO_ROOT / "hardware" / "config" / "bin-regions.json"

# The recommended slide for the surveyed mat layout. If this constant has to change,
# the written instruction has to change with it; that is the point of asserting on it.
RECOMMENDED_SLIDE_MM = (-30.0, 0.0)


# --- the band comes out of ik.py, it is not re-derived -------------------


def test_coaxial_band_matches_the_documented_annulus():
    lo, hi = cr.band(model=cr.COAXIAL)
    assert lo == pytest.approx(171.1, abs=0.2)
    assert hi == pytest.approx(246.6, abs=0.2)


def test_band_edges_agree_with_solve_ik_and_in_limits():
    lo, hi = cr.band(model=cr.COAXIAL)
    for r, reachable in ((lo + 0.5, True), (hi - 0.5, True)):
        assert in_limits(solve_ik(r, 0.0, PICK_Z)) is reachable
    for r in (lo - 0.5, hi + 0.5):
        try:
            assert not in_limits(solve_ik(r, 0.0, PICK_Z))
        except Unreachable:
            pass  # past the 2-link envelope entirely, also unreachable


def test_inner_edge_is_the_elbow_and_the_outer_edge_is_the_wrist():
    lo, hi = cr.band(model=cr.COAXIAL)
    assert "elbow_flex" in cr.binding_joint(lo - 0.2, PICK_Z, cr.COAXIAL)
    assert "wrist_flex" in cr.binding_joint(hi + 0.2, PICK_Z, cr.COAXIAL)
    assert cr.binding_joint((lo + hi) / 2, PICK_Z, cr.COAXIAL) is None


def test_pan_offset_pushes_the_whole_band_outward_by_about_31_mm():
    lo_c, hi_c = cr.band(model=cr.COAXIAL)
    lo_o, hi_o = cr.band(model=cr.OFFSET)
    assert lo_o - lo_c == pytest.approx(31.2, abs=0.5)
    assert hi_o - hi_c == pytest.approx(31.0, abs=0.5)
    # ...which is exactly the closed form: d = hypot(r + a, b)
    assert lo_o == pytest.approx(math.hypot(lo_c + cr.SHOULDER_FORWARD_MM, cr.SHOULDER_LATERAL_MM), abs=0.1)


def test_robust_band_is_the_intersection_of_the_two_models():
    lo, hi = cr.robust_band()
    assert (lo, hi) == (max(cr.band(model=m)[0] for m in cr.MODELS),
                        min(cr.band(model=m)[1] for m in cr.MODELS))
    assert lo < hi, "the two models must still overlap, or no placement can satisfy both"


# --- the two IK models ---------------------------------------------------


def test_coaxial_model_is_exactly_what_ik_py_does():
    assert cr.joint_solution(210.0, 60.0, model=cr.COAXIAL) == pytest.approx(solve_ik(210.0, 60.0))


def test_offset_model_solves_the_shoulder_plane_radius_not_the_pan_radius():
    d = 240.0
    joints = cr.joint_solution(d, 0.0, model=cr.OFFSET)
    x, y, _ = forward_kinematics(joints | {"1": 0.0})  # tip radius in the shoulder plane
    expected = math.sqrt(d**2 - cr.SHOULDER_LATERAL_MM**2) - cr.SHOULDER_FORWARD_MM
    assert math.hypot(x, y) == pytest.approx(expected, abs=1e-6)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown IK model"):
        cr.joint_solution(200.0, 0.0, model="magic")


# --- the real surveyed layout -------------------------------------------


@pytest.fixture(scope="module")
def real_zones():
    from src.perception.homography import Homography
    from src.perception.verify import load_bin_regions

    return lambda slide: cr.zone_reaches(
        Homography.load(HOMOGRAPHY), load_bin_regions(BIN_REGIONS), slide=slide
    )


def test_current_layout_radii_match_the_measured_values(real_zones):
    got = {z.label: z.radius for z in real_zones((0.0, 0.0))}
    assert got["A"] == pytest.approx(259.0, abs=0.5)
    assert got["B"] == pytest.approx(262.1, abs=0.5)
    assert got["C"] == pytest.approx(246.4, abs=0.5)


def test_current_layout_fails_and_names_the_offending_zones(real_zones):
    bad = sorted(z.label for z in real_zones((0.0, 0.0)) if not z.ok)
    assert bad == ["A", "B"], "A and B are past the vertical-gripper limit as surveyed"


def test_recommended_slide_rescues_all_three_zones_under_both_models(real_zones):
    zones = real_zones(RECOMMENDED_SLIDE_MM)
    assert all(z.ok for z in zones)
    assert min(z.margin for z in zones) > 10.0, "the recommended slide promises >1 cm of slack"


def test_zone_c_is_flagged_as_extrapolated_beyond_the_calibration_sheet(real_zones):
    flagged = {z.label: z.outside_sheet > 0 for z in real_zones((0.0, 0.0))}
    assert flagged["C"], "zone C sits outside the printed 297x210 sheet — must be flagged"
    assert not flagged["A"]


# --- the slide search ----------------------------------------------------


def test_best_slide_beats_the_hand_picked_number_only_slightly(real_zones):
    lo, hi = cr.robust_band()
    pts = [z.table for z in real_zones((0.0, 0.0))]
    (dx, dy), value = cr.best_slide(pts, lo, hi)
    assert dx < 0 and value > 15.0
    (ax, ay), avalue = cr.best_slide(pts, lo, hi, toward_base_only=True)
    assert ay == 0.0
    assert -35.0 < ax < -25.0, "the pure toward-base optimum is the documented ~3 cm"
    assert avalue > 13.0


def test_best_slide_maximises_the_minimum_margin_not_merely_feasibility():
    lo, hi = cr.robust_band()
    pts = [(hi + 30.0, 0.0), (hi + 25.0, 50.0), (hi + 15.0, -50.0)]
    (dx, dy), value = cr.best_slide(pts, lo, hi)
    moved = [math.hypot(x + dx, y + dy) for x, y in pts]
    assert all(lo <= r <= hi for r in moved)
    # no nearby translation does better, i.e. this really is a local maximum
    for step in ((2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0)):
        nudged = min(
            min(math.hypot(x + dx + step[0], y + dy + step[1]) - lo,
                hi - math.hypot(x + dx + step[0], y + dy + step[1]))
            for x, y in pts
        )
        assert nudged <= value + 1e-6


# --- CLI contract --------------------------------------------------------


def test_main_exits_non_zero_on_the_current_layout(capsys):
    assert cr.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_exits_zero_for_the_recommended_slide(capsys):
    assert cr.main(["--slide=-30,0"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "no file is modified" in out


def test_main_reports_missing_calibration_as_exit_2(tmp_path, capsys):
    assert cr.main(["--homography", str(tmp_path / "nope.json")]) == 2
    assert "ERROR" in capsys.readouterr().err


def test_main_rejects_a_malformed_slide(capsys):
    assert cr.main(["--slide", "12"]) == 2


def test_main_with_an_empty_region_file_is_an_input_error(tmp_path, capsys):
    empty = tmp_path / "bins.json"
    empty.write_text(json.dumps({"type": "bin_regions", "units": "pixel", "regions": []}) + "\n")
    assert cr.main(["--bin-regions", str(empty)]) == 2
    assert "no zones" in capsys.readouterr().err


def test_selftest_passes_without_touching_any_config_file(capsys):
    assert cr.selftest() == 0
    assert "SELFTEST PASSED" in capsys.readouterr().out


def test_single_model_flag_narrows_the_verdict():
    from src.perception.homography import Homography
    from src.perception.verify import load_bin_regions

    zones = cr.zone_reaches(
        Homography.load(HOMOGRAPHY), load_bin_regions(BIN_REGIONS), models=(cr.OFFSET,)
    )
    # Under the (unimplemented) corrected model the surveyed zones already reach, which is
    # precisely why the default demands BOTH models agree before anyone moves the paper.
    assert all(z.ok for z in zones)
