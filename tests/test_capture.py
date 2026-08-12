"""Offline tests for frame capture and the two bring-up tools (no camera, no network).

The hardware window is ~15 minutes long, so everything that can be checked without the
camera plugged in is checked here: which /dev/videoN gets picked (the built-in Acer lid
camera must NEVER win: a lid camera cannot be fixed relative to the table, so the
planar homography would drift), which backend is chosen, and that failures raise a
readable :class:`CaptureError` instead of returning a black frame.

Device enumeration is driven by a **fake /sys/class/video4linux tree** (tmp_path) so the
selection logic is testable with nothing plugged in; that injection point is why
``list_video_devices``/``select_device`` take a ``sys_dir``.

Also covers the non-interactive parsing and the PASS/FAIL-and-refuse behaviour of
tools/calibrate_homography.py and tools/pick_bin_regions.py, plus both tools'
``--selftest`` paths, so a green suite tonight means tomorrow is mechanical.
Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.perception.capture import (
    CaptureError,
    capture_still,
    capture_twin_frame,
    choose_backend,
    list_video_devices,
    select_device,
)
from src.perception.verify import BinRegion, load_bin_regions
from tools import calibrate_homography as calib
from tools import pick_bin_regions as bins

# Real names as enumerated on this host (hardware/config/cameras.md).
ACER_BUILTIN = "Integrated RGB Camera: Integrate"  # USB 5986:215d, /dev/video0-3
KIT_CAM = "USB2.0_CAM1: USB2.0 CAM1"  # USB 05a3:9230, the overhead cam


def make_sysfs(tmp_path: Path, nodes: dict[str, tuple[str, int]]) -> Path:
    """Build a fake /sys/class/video4linux: ``{"video4": (name, index)}``."""
    root = tmp_path / "video4linux"
    root.mkdir()
    for node, (name, index) in nodes.items():
        d = root / node
        d.mkdir()
        (d / "name").write_text(name + "\n")
        (d / "index").write_text(f"{index}\n")
    return root


# --- enumeration ---------------------------------------------------------


def test_list_video_devices_reads_names_sorted_numerically(tmp_path):
    root = make_sysfs(
        tmp_path,
        {
            "video10": (KIT_CAM, 0),
            "video4": (KIT_CAM, 0),
            "video0": (ACER_BUILTIN, 0),
            "not-a-video-node": ("junk", 0),
        },
    )
    assert list_video_devices(root) == [
        ("/dev/video0", ACER_BUILTIN),
        ("/dev/video4", KIT_CAM),
        ("/dev/video10", KIT_CAM),  # video10 sorts AFTER video4, not lexically before
    ]


def test_list_video_devices_missing_sysfs_is_empty(tmp_path):
    assert list_video_devices(tmp_path / "nope") == []


# --- device auto-selection (the built-in camera must never win) ----------


def test_skips_builtin_acer_camera_in_favour_of_kit_cam(tmp_path):
    root = make_sysfs(
        tmp_path,
        {
            "video0": (ACER_BUILTIN, 0),
            "video1": (ACER_BUILTIN, 1),
            "video2": ("Integrated IR Camera: Integrate", 0),
            "video3": ("Integrated IR Camera: Integrate", 1),
            "video4": (KIT_CAM, 0),
            "video5": (KIT_CAM, 1),
        },
    )
    assert select_device(sys_dir=root) == "/dev/video4"


def test_prefers_capture_node_over_metadata_node(tmp_path):
    # Same camera, two nodes; sysfs index 1 is the metadata node (yields no images).
    root = make_sysfs(tmp_path, {"video6": (KIT_CAM, 1), "video7": (KIT_CAM, 0)})
    assert select_device(sys_dir=root) == "/dev/video7"


def test_unknown_camera_name_is_still_eligible(tmp_path):
    # An unrecognized name must never block the demo; only the built-in is excluded.
    root = make_sysfs(
        tmp_path, {"video0": (ACER_BUILTIN, 0), "video4": ("Some Unbranded Cam", 0)}
    )
    assert select_device(sys_dir=root) == "/dev/video4"


def test_only_builtin_present_raises_with_the_listing(tmp_path):
    root = make_sysfs(tmp_path, {"video0": (ACER_BUILTIN, 0), "video1": (ACER_BUILTIN, 1)})
    with pytest.raises(CaptureError) as e:
        select_device(sys_dir=root)
    msg = str(e.value)
    assert "no external camera" in msg
    assert "/dev/video0" in msg and "05a3:9230" in msg  # tells you what to plug in


def test_no_devices_at_all_raises(tmp_path):
    with pytest.raises(CaptureError, match="none"):
        select_device(sys_dir=tmp_path / "empty")


def test_explicit_device_list_bypasses_sysfs():
    assert select_device([("/dev/video0", ACER_BUILTIN), ("/dev/video9", KIT_CAM)]) == "/dev/video9"


# --- backend selection ---------------------------------------------------


def test_choose_backend_prefers_cv2_when_importable(monkeypatch):
    monkeypatch.setattr("src.perception.capture._cv2_or_none", lambda: object())
    assert choose_backend() == "cv2"


def test_choose_backend_falls_back_to_ffmpeg_without_cv2(monkeypatch):
    monkeypatch.setattr("src.perception.capture._cv2_or_none", lambda: None)
    monkeypatch.setattr("src.perception.capture.shutil.which", lambda _: "/usr/bin/ffmpeg")
    assert choose_backend() == "ffmpeg"


def test_choose_backend_no_backend_at_all_raises(monkeypatch):
    monkeypatch.setattr("src.perception.capture._cv2_or_none", lambda: None)
    monkeypatch.setattr("src.perception.capture.shutil.which", lambda _: None)
    with pytest.raises(CaptureError, match="no capture backend"):
        choose_backend()


def test_choose_backend_forced_but_unavailable_raises(monkeypatch):
    monkeypatch.setattr("src.perception.capture._cv2_or_none", lambda: None)
    with pytest.raises(CaptureError, match="opencv-python is not installed"):
        choose_backend("cv2")
    monkeypatch.setattr("src.perception.capture.shutil.which", lambda _: None)
    with pytest.raises(CaptureError, match="ffmpeg binary"):
        choose_backend("ffmpeg")


def test_choose_backend_unknown_name_raises():
    with pytest.raises(CaptureError, match="unknown backend"):
        choose_backend("gstreamer")


# --- capture_still dispatch ----------------------------------------------


def _fake_writer(calls: list[tuple[str, Path, int]], content: bytes = b"jpeg"):
    def _write(device: str, out_path: Path, warmup_frames: int) -> None:
        calls.append((device, out_path, warmup_frames))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content)

    return _write


def test_capture_still_int_device_becomes_dev_node(tmp_path, monkeypatch):
    calls: list[tuple[str, Path, int]] = []
    monkeypatch.setattr("src.perception.capture._capture_ffmpeg", _fake_writer(calls))
    monkeypatch.setattr("src.perception.capture.shutil.which", lambda _: "/usr/bin/ffmpeg")
    out = capture_still(4, tmp_path / "frame.jpg", 3, backend="ffmpeg")
    assert out.exists() and calls == [("/dev/video4", tmp_path / "frame.jpg", 3)]


def test_capture_still_autoselects_when_device_is_none(tmp_path, monkeypatch):
    calls: list[tuple[str, Path, int]] = []
    monkeypatch.setattr("src.perception.capture.select_device", lambda: "/dev/video7")
    monkeypatch.setattr("src.perception.capture._capture_cv2", _fake_writer(calls))
    monkeypatch.setattr("src.perception.capture._cv2_or_none", lambda: object())
    capture_still(None, tmp_path / "auto.jpg")
    assert calls[0][0] == "/dev/video7"


def test_capture_still_empty_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.perception.capture._capture_ffmpeg", _fake_writer([], content=b"")
    )
    monkeypatch.setattr("src.perception.capture.shutil.which", lambda _: "/usr/bin/ffmpeg")
    with pytest.raises(CaptureError, match="missing/empty"):
        capture_still("/dev/video4", tmp_path / "empty.jpg", backend="ffmpeg")


def test_capture_still_ffmpeg_nonzero_exit_is_an_error(tmp_path, monkeypatch):
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "/dev/video99: No such file or directory"

    monkeypatch.setattr("src.perception.capture.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("src.perception.capture.subprocess.run", lambda *a, **k: _Proc())
    with pytest.raises(CaptureError, match="No such file"):
        capture_still("/dev/video99", tmp_path / "x.jpg", backend="ffmpeg")


def test_capture_still_bad_device_raises_captureerror(tmp_path):
    """Real backend, absent device: the tomorrow-morning 'wrong node' failure."""
    backend = "cv2" if _cv2_available() else "ffmpeg"
    with pytest.raises(CaptureError):
        capture_still("/dev/video-definitely-not-a-camera", tmp_path / "n.jpg", 0,
                      backend=backend)


def _cv2_available() -> bool:
    try:
        import cv2  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


# --- platform (twin) path: duck-typed, no network ------------------------


class _Twin:
    def __init__(self, result):
        self.result = result
        self.kwargs: dict = {}

    def get_frame(self, fmt, **kwargs):
        self.kwargs = {"format": fmt, **kwargs}
        return self.result


class _Client:
    def __init__(self, twin):
        self._twin = twin

    def twin(self, twin_id=None):
        return self._twin


def test_capture_twin_frame_uses_cloud_source_and_returns_path(tmp_path):
    dest = tmp_path / "twin.jpg"
    twin = _Twin(str(dest))
    client = _Client(twin)

    def _fake_get_frame(fmt, **kwargs):
        twin.kwargs = {"format": fmt, **kwargs}
        Path(kwargs["path"]).write_bytes(b"jpeg")
        return kwargs["path"]

    twin.get_frame = _fake_get_frame  # type: ignore[method-assign]
    out = capture_twin_frame(client, "twin-uuid", dest)
    assert out == dest and out.read_bytes() == b"jpeg"
    assert twin.kwargs["source"] == "cloud"  # the one-shot REST call, not the stream


def test_capture_twin_frame_none_means_not_streaming(tmp_path):
    with pytest.raises(CaptureError, match="not streaming"):
        capture_twin_frame(_Client(_Twin(None)), "twin-uuid", tmp_path / "t.jpg")


def test_capture_twin_frame_unknown_sdk_lists_what_it_found(tmp_path):
    class _Bare:
        def hello(self):  # pragma: no cover, only its name is asserted on
            return None

    with pytest.raises(NotImplementedError, match="hello"):
        capture_twin_frame(_Bare(), "twin-uuid", tmp_path / "t.jpg")


# --- tools/calibrate_homography.py: non-interactive parsing --------------


def test_parse_point_spec_accepts_the_documented_forms():
    assert calib.parse_point_spec("812,455=0,0") == ((812.0, 455.0), (0.0, 0.0))
    assert calib.parse_point_spec(" 1180 , 470 -> 297 , 0 ") == ((1180.0, 470.0), (297.0, 0.0))
    assert calib.parse_point_spec("10,20:-5.5,3.25") == ((10.0, 20.0), (-5.5, 3.25))


@pytest.mark.parametrize("spec", ["812,455", "812=0,0", "a,b=0,0", "1,2=3,4,5"])
def test_parse_point_spec_rejects_junk(spec):
    with pytest.raises(ValueError):
        calib.parse_point_spec(spec)


def test_parse_points_file_csv_and_json_agree(tmp_path):
    csv_path = tmp_path / "p.csv"
    csv_path.write_text("px,py,X,Y\n0,0,0,0\n100,0,297,0\n100,80,297,210\n0,80,0,210\n# tail\n")
    json_path = tmp_path / "p.json"
    json_path.write_text(
        json.dumps(
            {
                "points": [
                    {"pixel": [0, 0], "world": [0, 0]},
                    [100, 0, 297, 0],
                    [100, 80, 297, 210],
                    [0, 80, 0, 210],
                ]
            }
        )
    )
    pairs = calib.parse_points_file(csv_path)
    assert len(pairs) == 4
    assert pairs == calib.parse_points_file(json_path)


def test_parse_points_file_rejects_short_rows(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("px,py,X,Y\n0,0,0\n")
    with pytest.raises(ValueError, match="4 numbers"):
        calib.parse_points_file(bad)


def test_sheet_world_points_is_a4_in_click_order():
    assert calib.sheet_world_points(297, 210) == [
        (0.0, 0.0),  # origin corner, goes at the arm base
        (297.0, 0.0),
        (297.0, 210.0),
        (0.0, 210.0),
    ]
    with pytest.raises(ValueError):
        calib.sheet_world_points(0, 210)


# --- tools/calibrate_homography.py: the PASS/FAIL bar --------------------


def _clean_pairs() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """world = 2*pixel mm: a clean, exactly-fittable calibration."""
    pix = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0), (100.0, 100.0)]
    return [(p, (2 * p[0], 2 * p[1])) for p in pix]


def test_fit_passes_and_saves_a_loadable_homography(tmp_path):
    out = tmp_path / "homography.json"
    passed, h, saved = calib.fit_and_report(_clean_pairs(), out_path=out)
    assert passed and saved == out and h is not None
    meta = json.loads(out.read_text())
    assert meta["units"] == "mm" and meta["passed"] is True and meta["n_points"] == 5
    assert meta["max_error_mm"] < calib.SHIP_BAR_MM


def test_fit_refuses_to_save_when_over_the_2cm_bar(tmp_path):
    pairs = _clean_pairs()
    pairs[4] = (pairs[4][0], (pairs[4][1][0] + 60.0, pairs[4][1][1]))  # 60 mm blunder
    out = tmp_path / "refused.json"
    passed, h, saved = calib.fit_and_report(pairs, out_path=out)
    assert passed is False and saved is None and not out.exists()
    assert h is not None  # the fit itself succeeded; only the ship bar failed


def test_force_saves_a_failing_fit_but_marks_it(tmp_path):
    pairs = _clean_pairs()
    pairs[4] = (pairs[4][0], (pairs[4][1][0] + 60.0, pairs[4][1][1]))
    out = tmp_path / "forced.json"
    passed, _, saved = calib.fit_and_report(pairs, out_path=out, force=True)
    assert passed is False and saved == out
    assert json.loads(out.read_text())["passed"] is False


def test_tolerance_is_tunable(tmp_path):
    pairs = _clean_pairs()
    pairs[4] = (pairs[4][0], (pairs[4][1][0] + 60.0, pairs[4][1][1]))
    passed, _, _ = calib.fit_and_report(pairs, tolerance=100.0, out_path=tmp_path / "loose.json")
    assert passed is True


def test_degenerate_fit_is_reported_not_raised(tmp_path):
    collinear = [((float(i), float(i)), (2.0 * i, 2.0 * i)) for i in range(4)]
    passed, h, saved = calib.fit_and_report(collinear, out_path=tmp_path / "never.json")
    assert (passed, h, saved) == (False, None, None)
    assert not (tmp_path / "never.json").exists()


def test_calibrate_cli_non_interactive_round_trip(tmp_path):
    out = tmp_path / "cli.json"
    argv = [
        "--points", "0,0=0,0", "200,0=400,0", "200,200=400,400", "0,200=0,400",
        "--out", str(out),
    ]
    assert calib.main(argv) == 0 and out.exists()


def test_calibrate_cli_pixels_plus_sheet(tmp_path):
    out = tmp_path / "sheet.json"
    argv = [
        "--pixels", "0,0", "200,0", "200,140", "0,140",
        "--sheet", "297,210", "--out", str(out),
    ]
    assert calib.main(argv) == 0
    assert json.loads(out.read_text())["n_points"] == 4


def test_calibrate_cli_exits_nonzero_on_a_failing_bar(tmp_path):
    out = tmp_path / "fail.json"
    argv = [
        "--points", "0,0=0,0", "200,0=400,0", "200,200=400,400", "0,200=0,400",
        "100,100=260,200", "--out", str(out),
    ]
    assert calib.main(argv) == 1 and not out.exists()


def test_calibrate_cli_rejects_too_few_points(tmp_path):
    assert calib.main(["--points", "0,0=0,0", "1,1=2,2", "--out", str(tmp_path / "x.json")]) == 2


# --- the GUI pre-flight (a dead display must not core-dump the tool) -----


def test_gui_probe_detects_headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    reason = calib.gui_unavailable_reason()
    assert reason is not None and "DISPLAY" in reason


def test_gui_probe_reports_a_qt_abort_instead_of_dying(monkeypatch):
    # Qt SIGABRTs the process when it cannot reach the display, which is why the probe
    # runs in a subprocess at all; here we fake that child's failure.
    class _Proc:
        returncode = -6
        stdout = ""
        stderr = "qt.qpa.xcb: could not connect to display\nAborted"

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(calib.subprocess, "run", lambda *a, **k: _Proc())
    reason = calib.gui_unavailable_reason()
    assert reason is not None and "cannot open a window" in reason


def test_gui_probe_ok_returns_none(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(calib.subprocess, "run", lambda *a, **k: _Proc())
    assert calib.gui_unavailable_reason() is None


def test_click_points_headless_points_at_the_keyboard_path(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError) as e:
        calib.click_points(tmp_path / "frame.jpg")
    assert "--points" in str(e.value)  # the fallback command, not a traceback


def test_calibrate_cli_headless_exits_2_with_advice(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    out = tmp_path / "never.json"
    assert calib.main(["--image", str(tmp_path / "frame.jpg"), "--out", str(out)]) == 2
    assert "--points" in capsys.readouterr().err
    assert not out.exists()


def test_calibrate_selftest_passes_offline(capsys):
    assert calib.selftest() == 0
    assert "SELFTEST PASSED" in capsys.readouterr().out


# --- tools/pick_bin_regions.py -------------------------------------------


def test_parse_region_spec_normalizes_corners():
    r = bins.parse_region_spec("A=300,240,100,100")
    assert (r.label, r.x0, r.y0, r.x1, r.y1) == ("A", 100.0, 100.0, 300.0, 240.0)
    assert bins.parse_region_spec("B: 1 2 3 4").center == (2.0, 3.0)


@pytest.mark.parametrize("spec", ["A=1,2,3", "=1,2,3,4", "A=1,2,1,4", "A 1 2 3 4", "A=x,2,3,4"])
def test_parse_region_spec_rejects_junk(spec):
    with pytest.raises(ValueError):
        bins.parse_region_spec(spec)


def test_default_bin_labels_match_the_order_source():
    # src/wms_mock/orders.py posts {"bin": "A"} and run_order looks the region up by
    # that exact string; "Bin A" would silently disable verification.
    assert bins.DEFAULT_BINS == ("A", "B", "C")


def test_regions_from_clicks_is_click_order_agnostic():
    clicks = [(300.0, 240.0), (100.0, 100.0), (340.0, 100.0), (540.0, 240.0)]
    got = bins.regions_from_clicks(("A", "B"), clicks)
    assert [(r.label, r.x0, r.y0, r.x1, r.y1) for r in got] == [
        ("A", 100.0, 100.0, 300.0, 240.0),
        ("B", 340.0, 100.0, 540.0, 240.0),
    ]


def test_regions_from_clicks_needs_two_clicks_per_bin():
    with pytest.raises(ValueError, match="2 clicks per bin"):
        bins.regions_from_clicks(("A", "B"), [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])


def test_regions_from_clicks_rejects_zero_area():
    with pytest.raises(ValueError, match="zero-area"):
        bins.regions_from_clicks(("A",), [(10.0, 10.0), (10.0, 50.0)])


def test_write_and_verify_round_trips_through_load_bin_regions(tmp_path):
    out = tmp_path / "bin-regions.json"
    regions = [bins.parse_region_spec(s) for s in
               ("A=100,100,300,240", "B=340,100,540,240", "C=580,100,780,240")]
    ok, saved = bins.write_and_verify(regions, out_path=out, frame_size=(1280, 720))
    assert ok and saved == out

    payload = json.loads(out.read_text())
    assert payload["type"] == "bin_regions" and payload["units"] == "pixel"
    assert payload["frame_size"] == [1280, 720]

    loaded = load_bin_regions(out)  # the exact call run_order makes
    assert sorted(loaded) == ["A", "B", "C"]
    assert isinstance(loaded["A"], BinRegion)
    assert loaded["A"].contains(*loaded["A"].center)


def test_write_and_verify_refuses_duplicate_labels(tmp_path):
    out = tmp_path / "dup.json"
    dup = [BinRegion("A", 0, 0, 10, 10), BinRegion("A", 20, 20, 30, 30)]
    assert bins.write_and_verify(dup, out_path=out) == (False, None)
    assert not out.exists()


def test_write_and_verify_warns_on_overlap_but_still_saves(tmp_path, capsys):
    out = tmp_path / "overlap.json"
    ok, saved = bins.write_and_verify(
        [BinRegion("A", 0, 0, 100, 100), BinRegion("B", 50, 50, 150, 150)], out_path=out
    )
    assert ok and saved == out
    assert "overlap" in capsys.readouterr().out


def test_bins_cli_non_interactive(tmp_path):
    out = tmp_path / "cli-bins.json"
    argv = ["--regions", "A=100,100,300,240", "B=340,100,540,240",
            "--frame-size", "1280,720", "--out", str(out)]
    assert bins.main(argv) == 0
    assert sorted(load_bin_regions(out)) == ["A", "B"]


def test_bins_cli_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "never.json"
    assert bins.main(["--regions", "A=1,1,2,2", "--dry-run", "--out", str(out)]) == 0
    assert not out.exists()


def test_bins_headless_error_names_its_own_flag(tmp_path, monkeypatch, capsys):
    # Each tool must advise ITS OWN keyboard-only flag, not the other tool's.
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    out = tmp_path / "never.json"
    assert bins.main(["--image", str(tmp_path / "frame.jpg"), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "--regions" in err and "--points" not in err
    assert not out.exists()


def test_bins_selftest_passes_offline(capsys):
    assert bins.selftest() == 0
    assert "SELFTEST PASSED" in capsys.readouterr().out


# --- the two tools never touch the repo's real calibration files ---------


def test_tools_default_paths_point_at_hardware_config():
    assert calib.DEFAULT_OUT_PATH.name == "homography.json"
    assert bins.DEFAULT_OUT_PATH.name == "bin-regions.json"
    assert calib.DEFAULT_OUT_PATH.parent == bins.DEFAULT_OUT_PATH.parent
    assert calib.DEFAULT_OUT_PATH.parent.is_dir()  # hardware/config/ exists


def test_ffmpeg_binary_is_present_for_the_fallback_backend():
    # Not a unit test of our code: a bring-up precondition worth failing loudly on,
    # since ffmpeg is the backend that works when cv2 cannot open the node.
    assert shutil.which("ffmpeg"), "install ffmpeg: sudo apt-get install ffmpeg"
