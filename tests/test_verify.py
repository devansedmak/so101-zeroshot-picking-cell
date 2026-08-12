"""Unit tests for closed-loop placement verification (pure, no network/credits).

Covers the geometry (bin rectangles + point-in-bin), the three verdicts the loop's
retry policy branches on, the JSON calibration round-trip, and the fake-client wiring
for ``verify_placement``, including that the VLM's ``[y, x]`` / 0-1000 convention
still lands on the right pixels here (it is reused from detect.parse_detections, never
re-derived). Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import pytest

from src.perception import (
    BinRegion,
    BinRegionsMissing,
    Detection,
    UnknownBinRegion,
    VerifyResult,
    bin_region_for,
    evaluate_placement,
    load_bin_regions,
    save_bin_regions,
    verify_placement,
)

BIN_A = BinRegion("A", 100, 100, 300, 200)


# --- BinRegion geometry --------------------------------------------------


def test_contains_inside_outside_and_edges():
    assert BIN_A.contains(200, 150)  # centre
    assert BIN_A.contains(100, 100)  # top-left corner counts as inside
    assert BIN_A.contains(300, 200)  # bottom-right corner too
    assert not BIN_A.contains(99.9, 150)
    assert not BIN_A.contains(200, 200.1)
    assert BIN_A.center == (200.0, 150.0)


def test_swapped_corners_are_normalized():
    r = BinRegion("B", 300, 200, 100, 100)
    assert (r.x0, r.y0, r.x1, r.y1) == (100.0, 100.0, 300.0, 200.0)
    assert r.contains(200, 150)


# --- evaluate_placement: the three verdicts ------------------------------


def test_item_inside_bin_is_ok():
    dets = [Detection("red marker", 200, 150, 0.2, 0.15)]
    res = evaluate_placement(dets, "red marker", BIN_A)
    assert res.ok and res.detection is dets[0]
    assert res.point == (200, 150)
    assert "verified" in res.reason


def test_item_outside_bin_reason_names_where_it_is():
    dets = [Detection("red marker", 800, 600, 0.8, 0.6)]
    res = evaluate_placement(dets, "red marker", BIN_A)
    assert not res.ok
    assert "800" in res.reason and "600" in res.reason and "outside bin A" in res.reason
    assert res.detection is dets[0]  # we still know where it went


def test_nothing_detected_is_not_ok():
    res = evaluate_placement([], "red marker", BIN_A)
    assert not res.ok and res.detection is None and res.point is None
    assert "not detected" in res.reason


def test_label_mismatch_falls_back_like_the_pick_does():
    # pick_target's documented fallback: the VLM is prompted with the item name, so a
    # non-matching label is a labelling quirk, not a different object. Kept identical to
    # the pick so verification can't disagree with what was grasped.
    res = evaluate_placement([Detection("eraser", 200, 150, 0.2, 0.15)], "red marker", BIN_A)
    assert res.ok


def test_prefers_the_matching_label():
    dets = [
        Detection("eraser", 900, 900, 0.9, 0.9),
        Detection("red marker", 200, 150, 0.2, 0.15),
    ]
    assert evaluate_placement(dets, "red marker", BIN_A).ok


def test_result_to_dict_is_json_safe():
    res = evaluate_placement([Detection("red marker", 200, 150, 0.2, 0.15)], "red marker", BIN_A)
    d = res.to_dict()
    assert d["ok"] is True and d["bin"] == "A" and d["point"] == [200, 150]
    assert isinstance(VerifyResult(False, "x", "A", "no").to_dict()["point"], type(None))


# --- calibration JSON round-trip -----------------------------------------


def test_bin_regions_json_round_trip(tmp_path):
    path = tmp_path / "bin-regions.json"
    regions = [BIN_A, BinRegion("B", 400, 100, 600, 200)]
    save_bin_regions(path, regions, frame_size=(1280, 720))
    loaded = load_bin_regions(path)
    assert set(loaded) == {"A", "B"}
    assert loaded["A"] == BIN_A
    assert bin_region_for(loaded, "B").center == (500.0, 150.0)
    with pytest.raises(UnknownBinRegion):
        bin_region_for(loaded, "Z")


def test_save_accepts_a_mapping_too(tmp_path):
    path = tmp_path / "bins.json"
    save_bin_regions(path, {"A": BIN_A})
    assert load_bin_regions(path)["A"] == BIN_A


def test_missing_calibration_raises_clear_error(tmp_path):
    with pytest.raises(BinRegionsMissing, match="no bin-region calibration"):
        load_bin_regions(tmp_path / "nope.json")


def test_bad_region_dict_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"regions": [{"label": "A", "x0": 1}]}')
    with pytest.raises(ValueError):
        load_bin_regions(path)


# --- verify_placement wiring with a fake client --------------------------


class _FakeResult:
    def __init__(self, output):
        self.output = output

    def is_queued(self):
        return False


class FakeMLModels:
    def __init__(self, output):
        self._output = output
        self.calls: list[dict] = []

    def run(self, model, **kwargs):
        self.calls.append({"model": model, **kwargs})
        return _FakeResult(self._output)


class FakeClient:
    def __init__(self, output):
        self.mlmodels = FakeMLModels(output)


def test_verify_placement_ok_and_yx_convention():
    # VLM point [y=150, x=200] on the 0-1000 grid -> px (200, 150) on a 1000x1000 frame
    client = FakeClient([{"point": [150, 200], "label": "red marker"}])
    res = verify_placement(
        client, "frame.jpg", "red marker", BIN_A, model="m", image_size=(1000, 1000)
    )
    assert res.ok, res.reason
    assert res.point == pytest.approx((200.0, 150.0))
    call = client.mlmodels.calls[0]
    assert call["structured_task"] == "detect_points"
    assert call["prompt"] == "red marker" and call["image"] == "frame.jpg"


def test_verify_placement_yx_swap_would_land_outside():
    # same numbers in [x, y] order (the bug this convention guards against) -> outside
    client = FakeClient([{"point": [200, 150], "label": "red marker"}])
    res = verify_placement(
        client,
        "frame.jpg",
        "red marker",
        BinRegion("A", 190, 140, 210, 160),
        model="m",
        image_size=(1000, 1000),
    )
    assert not res.ok


def test_verify_placement_empty_output_is_not_ok():
    client = FakeClient([])
    res = verify_placement(client, "f.jpg", "red marker", BIN_A, model="m")
    assert not res.ok and res.detection is None


def test_verify_placement_scales_to_frame_size():
    # point [y=500, x=500] on a 1280x720 frame -> px (640, 360)
    client = FakeClient([{"point": [500, 500]}])
    res = verify_placement(
        client,
        "f.jpg",
        "red marker",
        BinRegion("A", 600, 340, 700, 400),
        model="m",
        image_size=(1280, 720),
    )
    assert res.ok
    assert res.point == pytest.approx((640.0, 360.0))
