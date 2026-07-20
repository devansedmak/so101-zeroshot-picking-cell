"""Unit tests for zero-shot VLM detection parsing (pure, no network/credits).

Validates the coordinate convention that the whole pick depends on — the VLM's
``[y, x]`` order on a 0-1000 scale → source pixels — plus the fake-client wiring for
``detect``. Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (see runbook).
"""

from __future__ import annotations

import pytest

from src.perception import (
    TASK_POINTS,
    Detection,
    detect,
    parse_detections,
    pick_target,
)


# --- coordinate convention (the critical bit) ----------------------------


def test_point_is_yx_order_scaled_to_pixels():
    # point [y=250, x=750] on 0-1000 → normalized (0.75, 0.25) → px on a 1000x400 frame
    dets = parse_detections([{"point": [250, 750], "label": "red marker"}], 1000, 400)
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "red marker"
    assert (d.nx, d.ny) == pytest.approx((0.75, 0.25))
    assert (d.x, d.y) == pytest.approx((750.0, 100.0))  # x=0.75*1000, y=0.25*400


def test_box_center_and_pixels():
    dets = parse_detections([{"box_2d": [100, 200, 300, 600], "label": "eraser"}], 1000, 1000)
    d = dets[0]
    # center = ((200+600)/2, (100+300)/2)/1000 = (0.4, 0.2)
    assert (d.nx, d.ny) == pytest.approx((0.4, 0.2))
    assert d.box == pytest.approx((200.0, 100.0, 600.0, 300.0))


def test_out_of_range_coords_are_clamped():
    dets = parse_detections([{"point": [1200, -50]}], 800, 600)
    d = dets[0]
    assert 0.0 <= d.nx <= 1.0 and 0.0 <= d.ny <= 1.0
    assert (d.nx, d.ny) == pytest.approx((0.0, 1.0))  # x=-50→0, y=1200→1000→1.0


# --- robustness ----------------------------------------------------------


def test_output_dict_wrapper_is_unwrapped():
    out = {"points": [{"point": [500, 500], "label": "pen"}]}
    dets = parse_detections(out, 100, 100)
    assert len(dets) == 1 and dets[0].label == "pen"


def test_default_label_used_when_missing():
    dets = parse_detections([{"point": [500, 500]}], 100, 100, default_label="red marker")
    assert dets[0].label == "red marker"


def test_malformed_entries_skipped_but_valid_kept():
    out = [
        {"point": [500]},                 # too short → skipped
        {"point": ["a", "b"]},            # non-numeric → skipped
        {"nonsense": 1},                  # no spatial key → skipped
        {"point": [500, 500], "label": "ok"},
    ]
    dets = parse_detections(out, 100, 100)
    assert len(dets) == 1 and dets[0].label == "ok"


def test_empty_and_none_output():
    assert parse_detections([], 100, 100) == []
    assert parse_detections(None, 100, 100) == []


# --- pick_target ---------------------------------------------------------


def test_pick_target_prefers_label_match():
    dets = [
        Detection("blue marker", 1, 1, 0.1, 0.1),
        Detection("red marker", 2, 2, 0.2, 0.2),
    ]
    assert pick_target(dets, "red marker").label == "red marker"


def test_pick_target_falls_back_to_first():
    dets = [Detection("pen", 1, 1, 0.1, 0.1), Detection("cup", 2, 2, 0.2, 0.2)]
    assert pick_target(dets, "stapler").label == "pen"
    assert pick_target([], "x") is None


# --- detect() wiring with a fake client ----------------------------------


class _FakeResult:
    def __init__(self, output):
        self.output = output
        self.status = "completed"

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


def test_detect_calls_sdk_and_parses():
    client = FakeClient([{"point": [500, 250], "label": "red marker"}])
    dets = detect(
        client,
        "frame.jpg",
        "red marker",
        model="gemini-robotics-er",
        image_size=(1000, 1000),
    )
    # SDK was called with the right structured task + prompt
    call = client.mlmodels.calls[0]
    assert call["structured_task"] == TASK_POINTS
    assert call["prompt"] == "red marker"
    assert call["image"] == "frame.jpg"
    # and the response parsed to pixels (x=0.25*1000, y=0.5*1000)
    assert (dets[0].x, dets[0].y) == pytest.approx((250.0, 500.0))


def test_detect_raises_on_queued_result():
    class _Queued(_FakeResult):
        def is_queued(self):
            return True

    class _QMLModels(FakeMLModels):
        def run(self, model, **kwargs):
            r = _Queued(None)
            r.poll_url = "http://poll"
            return r

    client = FakeClient(None)
    client.mlmodels = _QMLModels(None)
    with pytest.raises(RuntimeError, match="queued"):
        detect(client, "f.jpg", "x", model="m", image_size=(10, 10))
