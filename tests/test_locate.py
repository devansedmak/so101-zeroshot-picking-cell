"""Unit tests for the pick-target locator: capture → detect → homography → table mm.

Fully offline: a fake VLM client (duck-types ``cw.mlmodels``, zero credits), a homography
fitted from known correspondences, and no camera — capture is monkeypatched, or bypassed
entirely by handing in a saved frame. The pixel→mm numbers are asserted against the
affine map computed **by hand** from the correspondences (``_expected_table_mm``), not
against whatever the code returns, so a sign flip or a [y, x] swap fails here.
Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (see runbook).
"""

from __future__ import annotations

import pytest

from src.perception import Homography
from src.perception.locate import (
    FrameSizeUnknown,
    HomographyMissing,
    ItemNotFound,
    LocatedItem,
    frame_size,
    load_homography,
    locate_item,
)

# Same synthetic rig as tests/test_seam.py: a 640×480 overhead frame whose image maps
# onto a reachable patch of the table (X ∈ [100, 200] mm, Y ∈ [-100, 100] mm).
IMG_W, IMG_H = 640, 480
_PIXELS = [(0, 0), (640, 0), (640, 480), (0, 480)]
_TABLE_MM = [(100, -100), (200, -100), (200, 100), (100, 100)]


def _expected_table_mm(px: float, py: float) -> tuple[float, float]:
    """The map implied by the four correspondences above, derived independently.

    Corner-to-corner it is a plain affine scale+offset, so the ground truth is arithmetic
    we can write down without calling Homography at all.
    """
    return (100.0 + px / IMG_W * 100.0, -100.0 + py / IMG_H * 200.0)


def _homography() -> Homography:
    return Homography.fit(_PIXELS, _TABLE_MM)


class FakeVLM:
    """Offline stand-in for the hosted VLM: replays canned output, records the call.

    Satisfies the only surface ``detect`` uses (``mlmodels.run``) — no network, no credits.
    """

    def __init__(self, output=None) -> None:
        self.output = output
        self.calls: list[dict] = []

    @property
    def mlmodels(self) -> "FakeVLM":
        return self

    def run(self, model, **kwargs):
        self.calls.append({"model": model, **kwargs})
        return self


# --- the happy path: detection → pixel → table mm ------------------------


def test_locate_item_maps_the_detection_to_the_right_table_mm():
    # VLM answers [y, x] on a 0..1000 grid (D15) ⇒ px = (0.75·640, 0.25·480).
    client = FakeVLM([{"point": [250, 750], "label": "red marker"}])
    located = locate_item(
        "red marker",
        client=client,
        homography=_homography(),
        image="/nonexistent/frame.jpg",  # the fake never opens it
        image_size=(IMG_W, IMG_H),
    )

    assert isinstance(located, LocatedItem)
    assert located.pixel == pytest.approx((480.0, 120.0))
    assert located.table_mm == pytest.approx(_expected_table_mm(480.0, 120.0), abs=1e-6)
    assert (located.x_mm, located.y_mm) == pytest.approx((175.0, -50.0), abs=1e-6)
    assert located.item == "red marker" and located.label == "red marker"
    assert str(located.frame) == "/nonexistent/frame.jpg"


def test_locate_item_calls_the_vlm_with_the_item_as_the_open_vocab_prompt(monkeypatch):
    monkeypatch.setenv("CYBERWAVE_VLM_MODEL", "test-slug")
    client = FakeVLM([{"point": [500, 500]}])
    locate_item(
        "blue marker",
        client=client,
        homography=_homography(),
        image="frame.jpg",
        image_size=(IMG_W, IMG_H),
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "test-slug"  # pinned slug comes from the env (D15)
    assert call["prompt"] == "blue marker"  # zero-shot: the order text IS the prompt
    assert call["structured_task"] == "detect_points"


def test_locate_item_prefers_the_label_matching_the_ordered_item():
    client = FakeVLM(
        [
            {"point": [250, 250], "label": "eraser"},
            {"point": [250, 750], "label": "red marker"},
        ]
    )
    located = locate_item(
        "red marker",
        client=client,
        homography=_homography(),
        image="frame.jpg",
        image_size=(IMG_W, IMG_H),
    )
    assert located.label == "red marker"
    assert located.table_mm == pytest.approx(_expected_table_mm(480.0, 120.0), abs=1e-6)


def test_located_item_to_dict_is_json_safe():
    client = FakeVLM([{"point": [250, 750], "label": "red marker"}])
    located = locate_item(
        "red marker",
        client=client,
        homography=_homography(),
        image="frame.jpg",
        image_size=(IMG_W, IMG_H),
    )
    import json

    payload = located.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["frame"] == "frame.jpg"
    assert payload["table_mm"] == pytest.approx([175.0, -50.0], abs=1e-6)
    assert "→ table (175, -50) mm" in located.describe()


# --- the typed failure --------------------------------------------------


@pytest.mark.parametrize("output", [None, [], [{"label": "eraser"}], {"points": []}])
def test_locate_item_raises_item_not_found_when_nothing_usable_comes_back(output):
    with pytest.raises(ItemNotFound) as excinfo:
        locate_item(
            "red marker",
            client=FakeVLM(output),
            homography=_homography(),
            image="frame.jpg",
            image_size=(IMG_W, IMG_H),
        )
    assert "red marker" in str(excinfo.value)  # message names what was looked for


# --- capture: bypassed by a supplied frame, used otherwise ---------------


def test_supplied_image_bypasses_capture_entirely(monkeypatch):
    def boom(*a, **k):  # pragma: no cover — must never run
        raise AssertionError("capture_still must not be called when image= is given")

    monkeypatch.setattr("src.perception.capture.capture_still", boom)
    located = locate_item(
        "eraser",
        client=FakeVLM([{"point": [500, 500], "label": "eraser"}]),
        homography=_homography(),
        image="saved-frame.jpg",
        image_size=(IMG_W, IMG_H),
    )
    assert located.table_mm == pytest.approx((150.0, 0.0), abs=1e-6)  # patch centre


def test_without_an_image_it_captures_one_still(monkeypatch, tmp_path):
    grabbed = tmp_path / "overhead.jpg"
    calls: list[dict] = []

    def fake_capture(device=None, *a, **k):
        calls.append({"device": device})
        return grabbed

    monkeypatch.setattr("src.perception.capture.capture_still", fake_capture)
    located = locate_item(
        "eraser",
        client=FakeVLM([{"point": [500, 500], "label": "eraser"}]),
        homography=_homography(),
        camera="/dev/video4",
        image_size=(IMG_W, IMG_H),
    )
    assert calls == [{"device": "/dev/video4"}]  # exactly one still, on the given device
    assert located.frame == grabbed


# --- frame size ---------------------------------------------------------


def test_frame_size_is_read_from_a_real_frame_when_not_given(tmp_path):
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    frame = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame), numpy.zeros((IMG_H, IMG_W, 3), numpy.uint8))

    assert frame_size(frame) == (IMG_W, IMG_H)
    located = locate_item(  # no image_size ⇒ probed from the file
        "red marker",
        client=FakeVLM([{"point": [250, 750], "label": "red marker"}]),
        homography=_homography(),
        image=frame,
    )
    assert located.table_mm == pytest.approx(_expected_table_mm(480.0, 120.0), abs=1e-6)


def test_unreadable_frame_without_a_size_raises_a_typed_error(tmp_path):
    junk = tmp_path / "not-an-image.jpg"
    junk.write_bytes(b"not a jpeg")
    with pytest.raises(FrameSizeUnknown):
        locate_item(
            "red marker",
            client=FakeVLM([{"point": [250, 750]}]),
            homography=_homography(),
            image=junk,
        )


# --- the grasp axis rides along (best effort, never fatal) ---------------


def _frame_with_a_marker(tmp_path, angle_deg: float):
    """A frame with one filled rotated bar at the image centre, at a KNOWN image angle."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import math

    img = np.full((IMG_H, IMG_W, 3), (255, 255, 255), np.uint8)
    t = math.radians(angle_deg)
    d = np.array([math.cos(t), math.sin(t)])
    n = np.array([-math.sin(t), math.cos(t)])
    c = np.array([IMG_W / 2, IMG_H / 2])
    box = np.round(
        np.array([c + d * 70 + n * 15, c + d * 70 - n * 15, c - d * 70 - n * 15, c - d * 70 + n * 15])
    ).astype(np.int32)
    cv2.fillPoly(img, [box], (40, 40, 200))
    path = tmp_path / "overhead.png"  # PNG: no JPEG ringing on the drawn edges
    cv2.imwrite(str(path), img)
    return path


def test_locate_item_reports_the_object_axis_in_table_degrees(tmp_path):
    import math

    angle_px = 30.0
    frame = _frame_with_a_marker(tmp_path, angle_px)
    # The VLM points at the image centre, i.e. at the bar: [y, x] on the 0..1000 grid.
    located = locate_item(
        "red marker",
        client=FakeVLM([{"point": [500, 500], "label": "red marker"}]),
        homography=_homography(),
        image=frame,
    )

    # Ground truth derived from the correspondences, not from the code: this calibration
    # scales x and y differently (100 mm / 640 px vs 200 mm / 480 px), so a 30° bar in
    # pixels is NOT 30° on the table — which is exactly why the angle is taken after
    # projecting both endpoints.
    sx, sy = 100.0 / IMG_W, 200.0 / IMG_H
    expected = math.degrees(
        math.atan2(sy * math.sin(math.radians(angle_px)), sx * math.cos(math.radians(angle_px)))
    ) % 180.0
    assert located.axis_deg is not None
    assert min(abs(located.axis_deg - expected), 180 - abs(located.axis_deg - expected)) < 3.0
    assert f"axis {located.axis_deg:.0f}°" in located.describe()


def test_a_frame_with_no_measurable_axis_still_locates_the_item():
    # The stub frame here is never even decodable — orientation must degrade to None
    # instead of raising, leaving today's fixed-roll behaviour intact.
    located = locate_item(
        "red marker",
        client=FakeVLM([{"point": [250, 750], "label": "red marker"}]),
        homography=_homography(),
        image="/nonexistent/frame.jpg",
        image_size=(IMG_W, IMG_H),
    )
    assert located.axis_deg is None
    assert located.table_mm == pytest.approx((175.0, -50.0), abs=1e-6)
    assert located.to_dict()["axis_deg"] is None
    assert "axis" not in located.describe()


# --- calibration I/O ----------------------------------------------------


def test_load_homography_missing_says_how_to_fix_it(tmp_path):
    with pytest.raises(HomographyMissing) as excinfo:
        load_homography(tmp_path / "homography.json")
    message = str(excinfo.value)
    assert "calibrate_homography" in message  # the message IS the runbook step


def test_load_homography_round_trip_feeds_locate(tmp_path):
    path = _homography().save(tmp_path / "homography.json", units="mm")
    located = locate_item(
        "red marker",
        client=FakeVLM([{"point": [250, 750], "label": "red marker"}]),
        homography=load_homography(path),
        image="frame.jpg",
        image_size=(IMG_W, IMG_H),
    )
    assert located.table_mm == pytest.approx((175.0, -50.0), abs=1e-6)
