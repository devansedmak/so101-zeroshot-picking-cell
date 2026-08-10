"""Calibrate the pixel→table homography — the highest-risk hardware step.

WHY a dedicated tool: `src/perception/homography.py` already has the math (DLT +
Hartley, unit-tested offline), but tomorrow morning there is exactly ONE short
hardware window. Everything that could go wrong then — a display that won't open, a
mistyped millimetre, a silently bad fit that only shows up as the gripper missing by
5 cm — has to be impossible or loud. So:

* **Three ways in, one way out.** Interactive clicking, `--points` on the command
  line, and `--points-file` all converge on :func:`fit_and_report`. If the GUI
  misbehaves, retype four numbers and keep going; nothing is display-gated.
* **The ship bar is enforced, not printed.** `max` reprojection error must be
  ≤ 20 mm (2 cm, cameras.md checkpoint). A failing fit is REFUSED, not saved —
  a saved-but-wrong homography is worse than no homography, because the loop
  would trust it. `--force` exists for "I know, I'll fix it later".
* **`--selftest` proves the whole path with no camera and no display**, so the
  tool is known-good the night before.

PHYSICAL TARGET (do this, don't improvise): print/tape a sheet with 4 marked
corners of known spacing on the mat — A4 landscape = 297 × 210 mm, A3 = 420 × 297 mm.
No checkerboard detection code exists: you just click the corners. Put the sheet's
**origin corner at the arm base**, +X pointing away from the arm, +Y to the arm's
left, so the homography world frame IS the IK base frame (see
hardware/config/link-lengths.md §Frame convention — `src/control/ik.py` assumes it).
Then `--sheet 297,210` assigns the mm coordinates for you and you type nothing.

CLI:
    .venv/bin/python tools/calibrate_homography.py --selftest              # offline proof
    .venv/bin/python tools/calibrate_homography.py --sheet 297,210         # capture + click 4 corners
    .venv/bin/python tools/calibrate_homography.py --image frame.jpg --sheet 297,210
    .venv/bin/python tools/calibrate_homography.py --points "812,455=0,0" "1180,470=297,0" ...
    .venv/bin/python tools/calibrate_homography.py --points-file points.csv   # px,py,X,Y per line
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `.venv/bin/python tools/calibrate_homography.py`
    sys.path.insert(0, str(REPO_ROOT))

from src.perception.homography import Homography, ReprojectionError  # noqa: E402

__all__ = [
    "SHIP_BAR_MM",
    "Pair",
    "click_points",
    "fit_and_report",
    "gui_unavailable_reason",
    "parse_point_spec",
    "parse_points_file",
    "selftest",
    "sheet_world_points",
]

# One (pixel, world_mm) correspondence.
Pair = tuple[tuple[float, float], tuple[float, float]]

DEFAULT_OUT_PATH = REPO_ROOT / "hardware" / "config" / "homography.json"

# cameras.md checkpoint criterion: every calibration point must reproject within 2 cm.
SHIP_BAR_MM = 20.0

# Clicking happens on a downscaled copy when the frame is bigger than this, so the
# whole table fits on screen; clicks are divided back out to true image pixels.
_MAX_DISPLAY = (1280, 720)


# ------------------------------------------------------------------ parsing
def _parse_xy(text: str, *, what: str, context: str) -> tuple[float, float]:
    """Parse ``"12,34"`` / ``"12 34"`` → ``(12.0, 34.0)``."""
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != 2:
        raise ValueError(f"{what} needs exactly 2 numbers, got {text.strip()!r} in {context!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as e:
        raise ValueError(f"{what} in {context!r} is not numeric: {e}") from e


def parse_point_spec(spec: str) -> Pair:
    """Parse one correspondence ``"px,py=X,Y"`` → ``((px, py), (X, Y))``.

    Left of the separator is the **pixel** in the overhead frame, right is the
    **table coordinate in mm**. ``=``, ``->`` and ``:`` all work as separators.
    """
    text = spec.strip()
    for sep in ("=", "->", ":"):
        if sep in text:
            left, right = text.split(sep, 1)
            break
    else:
        raise ValueError(
            f"bad point spec {spec!r}: expected 'px,py=X,Y' "
            "(pixels on the left, table mm on the right)"
        )
    return (
        _parse_xy(left, what="pixel", context=spec),
        _parse_xy(right, what="world mm", context=spec),
    )


def _pair_from_row(row: Sequence[float], context: str) -> Pair:
    if len(row) != 4:
        raise ValueError(f"{context}: expected 4 numbers (px,py,X,Y), got {list(row)!r}")
    px, py, x, y = (float(v) for v in row)
    return ((px, py), (x, y))


def parse_points_file(path: str | Path) -> list[Pair]:
    """Read correspondences from JSON or CSV — the keyboard-only path.

    JSON: ``{"points": [...]}`` or a bare list, whose items are either
    ``{"pixel": [px, py], "world": [X, Y]}`` or ``[px, py, X, Y]``.
    CSV/TXT: one ``px,py,X,Y`` per line; blank lines and ``#`` comments ignored,
    and a non-numeric header row is skipped.
    """
    p = Path(path)
    text = p.read_text()
    pairs: list[Pair] = []

    if p.suffix.lower() == ".json":
        data = json.loads(text)
        items = data.get("points", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError(f"{p}: expected a list of points (or a 'points' key)")
        for i, item in enumerate(items, start=1):
            ctx = f"{p} point #{i}"
            if isinstance(item, dict):
                try:
                    pix, world = item["pixel"], item["world"]
                except KeyError as e:
                    raise ValueError(f"{ctx}: missing key {e}") from e
                pairs.append(_pair_from_row([*pix, *world], ctx))
            else:
                pairs.append(_pair_from_row(list(item), ctx))
        return pairs

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            row = [float(v) for v in line.replace(",", " ").split()]
        except ValueError:
            if lineno == 1:  # tolerate a header line
                continue
            raise ValueError(f"{p}:{lineno}: not 4 numbers: {raw.strip()!r}") from None
        pairs.append(_pair_from_row(row, f"{p}:{lineno}"))
    if not pairs:
        raise ValueError(f"{p}: no points found")
    return pairs


def sheet_world_points(width_mm: float, height_mm: float) -> list[tuple[float, float]]:
    """Table mm for the 4 corners of a ``width × height`` sheet, in click order.

    Order is origin → +X → diagonal → +Y, i.e. ``(0,0), (W,0), (W,H), (0,H)``.
    Origin corner goes at the arm base (link-lengths.md §Frame convention).
    """
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError(f"sheet size must be positive, got {width_mm}x{height_mm}")
    return [(0.0, 0.0), (float(width_mm), 0.0), (float(width_mm), float(height_mm)),
            (0.0, float(height_mm))]


# ------------------------------------------------------------------ the fit
def fit_and_report(
    pairs: Sequence[Pair],
    *,
    tolerance: float = SHIP_BAR_MM,
    out_path: str | Path | None = DEFAULT_OUT_PATH,
    force: bool = False,
    note: str = "",
) -> tuple[bool, Homography | None, Path | None]:
    """Fit → print per-point error → PASS/FAIL vs ``tolerance`` → save (or refuse).

    THE one code path all three input modes share, and the one `--selftest` exercises.
    Returns ``(passed, homography, saved_path)``; ``saved_path`` is ``None`` when
    nothing was written (fit failed, bar not met without ``--force``, or
    ``out_path=None``). Never raises on a bad fit — prints and returns
    ``(False, None, None)`` — because a stack trace at 8 a.m. helps nobody.
    """
    print(f"[calib] {len(pairs)} correspondences (pixel → table mm):")
    for i, (pix, world) in enumerate(pairs, start=1):
        print(f"[calib]   #{i} px=({pix[0]:8.1f},{pix[1]:8.1f})  ->  "
              f"table=({world[0]:8.1f},{world[1]:8.1f}) mm")

    pixel_pts = [p for p, _ in pairs]
    world_pts = [w for _, w in pairs]
    try:
        h = Homography.fit(pixel_pts, world_pts)
    except ValueError as e:
        print(f"[calib] FIT FAILED: {e}")
        print("[calib]   fix: use ≥4 points, spread out, NOT on a line and not all "
              "bunched in one corner of the frame.")
        return False, None, None

    err: ReprojectionError = h.reprojection_error(pixel_pts, world_pts)
    print("[calib] reprojection error per point (mm): "
          + ", ".join(f"{e:.1f}" for e in err.per_point))
    print(f"[calib] rms={err.rms:.1f} mm   max={err.max:.1f} mm   "
          f"ship bar: max <= {tolerance:.1f} mm")

    passed = err.within(tolerance)
    if passed:
        print(f"[calib] PASS — calibration is within {tolerance / 10:.1f} cm.")
    else:
        print(f"[calib] *** FAIL *** max error {err.max:.1f} mm > {tolerance:.1f} mm.")
        print("[calib]   likely causes, in order: (a) a mistyped mm coordinate, "
              "(b) a click on the wrong corner / wrong click order, (c) the sheet or "
              "camera moved between clicking and measuring, (d) the mat is not flat.")
        print("[calib]   fix: re-run and re-click. Do NOT ship a failing calibration — "
              "the loop trusts this file silently.")

    if out_path is None:
        print("[calib] (no output path; nothing written)")
        return passed, h, None
    out = Path(out_path)
    if not passed and not force:
        print(f"[calib] REFUSED to write {out} (use --force to override).")
        return passed, h, None

    out.parent.mkdir(parents=True, exist_ok=True)
    saved = h.save(
        out,
        units="mm",
        n_points=len(pairs),
        rms_error_mm=round(err.rms, 3),
        max_error_mm=round(err.max, 3),
        ship_bar_mm=tolerance,
        passed=passed,
        note=note or "calibrated with tools/calibrate_homography.py",
    )
    flag = "" if passed else "  (FORCED — FAILING CALIBRATION)"
    print(f"[calib] saved → {saved}{flag}")
    print("[calib] copy-paste to re-fit without re-clicking (one --points, many specs):")
    print("           --points " + " ".join(
        f'"{p[0]:.1f},{p[1]:.1f}={w[0]:.1f},{w[1]:.1f}"' for p, w in pairs))
    return passed, h, saved


# ------------------------------------------------------------ interactive
_NO_GUI_HINT = (
    "Use the keyboard-only path instead (same code path, same result): read the pixel "
    'coordinates off any image viewer and pass --points "px,py=X,Y" ... , or '
    "--pixels 'px,py' ... together with --sheet W,H, or --points-file FILE."
)

# Cheapest possible GUI smoke test, run in a throwaway subprocess.
_GUI_PROBE_CODE = (
    "import cv2, numpy;"
    "cv2.namedWindow('probe', cv2.WINDOW_AUTOSIZE);"
    "cv2.imshow('probe', numpy.zeros((8, 8, 3), 'uint8'));"
    "cv2.waitKey(1);"
    "cv2.destroyAllWindows();"
    "cv2.waitKey(1)"
)


def gui_unavailable_reason(*, timeout_s: float = 20.0) -> str | None:
    """``None`` if an OpenCV window can actually open, else why it cannot.

    WHY a subprocess: when the Qt platform plugin cannot reach the display, OpenCV
    **aborts the whole process** (SIGABRT) — a Python ``try/except`` cannot save us, so
    the interactive path would die with a core dump instead of telling Devan to use
    ``--points``. Probing in a child process makes that failure survivable and loud.
    """
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no DISPLAY / WAYLAND_DISPLAY (headless shell or ssh without X forwarding)"
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [sys.executable, "-c", _GUI_PROBE_CODE],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"the GUI probe did not finish ({type(e).__name__}: {e})"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        return f"OpenCV cannot open a window: {detail}"
    return None


def click_points(
    image_path: str | Path,
    *,
    min_points: int = 4,
    window: str = "click calibration points  [ENTER=done  u=undo  ESC=abort]",
    max_display: tuple[int, int] = _MAX_DISPLAY,
    probe_gui: bool = True,
    fallback_hint: str = _NO_GUI_HINT,
) -> list[tuple[float, float]]:
    """Show ``image_path`` and return the clicked points in TRUE image pixels.

    Shared with tools/pick_bin_regions.py.
    ``fallback_hint`` is the "do this instead" text every no-GUI error ends with, so
    each tool can name its own keyboard-only flags. The frame is downscaled to fit
    ``max_display`` and clicks are divided back out, because an OpenCV
    ``WINDOW_NORMAL`` reports *window* coordinates — resizing a window would
    silently corrupt a calibration. Every click is echoed to the terminal so the
    numbers survive even if the window is unreadable.

    Raises:
        RuntimeError: OpenCV missing, no usable GUI/display, or aborted with ESC.
    """
    try:
        import cv2  # noqa: PLC0415 — optional GUI backend, probed at call time
    except ImportError as e:  # pragma: no cover — opencv is a project dependency
        raise RuntimeError(
            "opencv-python is not installed, so there is no GUI. "
            f"{fallback_hint} To get the GUI: .venv/bin/pip install opencv-python"
        ) from e

    if probe_gui:
        reason = gui_unavailable_reason()
        if reason is not None:
            raise RuntimeError(f"cannot open a click window — {reason}. {fallback_hint}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"could not read image {image_path} (bad path or not an image?)")
    h, w = img.shape[:2]
    scale = min(1.0, max_display[0] / w, max_display[1] / h)
    view = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img.copy()
    print(f"[click] {image_path}  {w}x{h} px  (display scale {scale:.3f})")
    print(f"[click] click at least {min_points} points, then press ENTER. "
          "u=undo last, ESC=abort.")

    picked: list[tuple[float, float]] = []

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        true_pt = (x / scale, y / scale)
        picked.append(true_pt)
        print(f"[click]   #{len(picked)}  px=({true_pt[0]:.1f}, {true_pt[1]:.1f})")

    def _redraw() -> None:
        canvas = view.copy()
        for i, (tx, ty) in enumerate(picked, start=1):
            sx, sy = int(tx * scale), int(ty * scale)
            cv2.drawMarker(canvas, (sx, sy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(canvas, str(i), (sx + 8, sy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
        cv2.imshow(window, canvas)

    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)  # AUTOSIZE: mouse coords == view px
        cv2.setMouseCallback(window, _on_mouse)
        while True:
            _redraw()
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):  # ENTER
                if len(picked) < min_points:
                    print(f"[click] need {min_points - len(picked)} more point(s).")
                    continue
                break
            if key == 27:  # ESC
                raise RuntimeError("aborted at the click window (ESC)")
            if key in (ord("u"), 8) and picked:  # u / BACKSPACE
                dropped = picked.pop()
                print(f"[click]   undo ({dropped[0]:.1f}, {dropped[1]:.1f})")
    except Exception as e:  # noqa: BLE001 — cv2.error has no useful common base
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(
            f"the OpenCV window failed ({type(e).__name__}: {e}). {fallback_hint}"
        ) from e
    finally:
        try:
            cv2.destroyWindow(window)
            cv2.waitKey(1)
        except Exception:  # noqa: BLE001, S110 — teardown must never mask the result
            pass
    return picked


def _prompt_world_points(pixels: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Ask, on the terminal, for the table mm of each clicked pixel (retry on typos)."""
    world: list[tuple[float, float]] = []
    print("\n[calib] now type each point's TABLE coordinate in mm "
          "(origin = arm base, +X away from the arm, +Y to its left):")
    for i, (px, py) in enumerate(pixels, start=1):
        while True:
            raw = input(f"    #{i} px=({px:.1f}, {py:.1f})  ->  X Y [mm]: ")
            try:
                world.append(_parse_xy(raw, what="world mm", context=f"point #{i}"))
                break
            except ValueError as e:
                print(f"    ! {e} — try again, e.g. '297 0'")
    return world


def _grab_frame(device: str | None, warmup: int) -> Path:
    from src.perception.capture import CaptureError, capture_still  # noqa: PLC0415

    try:
        path = capture_still(device, None, warmup)
    except CaptureError as e:
        raise RuntimeError(
            f"could not grab a frame: {e}\n"
            "  (or pass --image FILE with a frame you already captured via "
            "`.venv/bin/python -m src.perception.capture --save frame.jpg`)"
        ) from e
    print(f"[calib] captured {path}")
    return path


# ------------------------------------------------------------------ selftest
def _synthetic_pairs(noise_mm: float = 1.5) -> list[Pair]:
    """A genuinely projective ground truth + deterministic sub-bar noise."""
    import numpy as np  # noqa: PLC0415 — only the selftest needs it directly

    h_true = np.array([[1.9, 0.06, -55.0], [-0.04, 1.85, -40.0], [0.0004, -0.0002, 1.0]])
    pixels = [(120.0, 90.0), (1160.0, 105.0), (1180.0, 640.0), (140.0, 620.0),
              (650.0, 360.0), (300.0, 500.0)]
    wobble = [(noise_mm, 0.0), (-noise_mm, noise_mm), (0.0, -noise_mm),
              (noise_mm, noise_mm), (-noise_mm, 0.0), (0.0, noise_mm)]
    pairs: list[Pair] = []
    for (px, py), (dx, dy) in zip(pixels, wobble):
        v = h_true @ np.array([px, py, 1.0])
        pairs.append(((px, py), (v[0] / v[2] + dx, v[1] / v[2] + dy)))
    return pairs


def selftest() -> int:
    """Run fit → error → PASS/FAIL → save on synthetic points. No camera, no display."""
    print("=" * 72)
    print("SELFTEST tools/calibrate_homography.py — synthetic points, no camera/display")
    print("=" * 72)
    failures: list[str] = []

    def check(cond: bool, what: str) -> None:
        print(f"  [{'ok' if cond else 'FAIL'}] {what}")
        if not cond:
            failures.append(what)

    with tempfile.TemporaryDirectory(prefix="calib_selftest_") as tmp:
        tmpdir = Path(tmp)

        print("\n-- 1. good calibration (noise well under the 2 cm bar) → PASS + saves")
        good = _synthetic_pairs()
        out_good = tmpdir / "homography.json"
        passed, h, saved = fit_and_report(good, out_path=out_good, note="selftest")
        check(passed, "PASS verdict on clean synthetic points")
        check(saved == out_good and out_good.exists(), f"wrote {out_good.name}")
        if h is not None and out_good.exists():
            reloaded = Homography.load(out_good)
            same = reloaded.project((640, 360)) == h.project((640, 360))
            check(same, "saved file reloads and projects identically")
            meta = json.loads(out_good.read_text())
            check(meta["units"] == "mm" and meta["n_points"] == len(good)
                  and meta["passed"] is True, "metadata records units/n_points/passed")

        print("\n-- 2. bad calibration (one 60 mm outlier) → FAIL + REFUSES to save")
        bad = list(good)
        bx, by = bad[4][1]
        bad[4] = (bad[4][0], (bx + 60.0, by))
        out_bad = tmpdir / "refused.json"
        passed_bad, _, saved_bad = fit_and_report(bad, out_path=out_bad)
        check(not passed_bad, "FAIL verdict on the outlier set")
        check(saved_bad is None and not out_bad.exists(), "nothing written without --force")

        print("\n-- 3. same bad set with --force → saves, flagged passed=false")
        out_forced = tmpdir / "forced.json"
        _, _, saved_forced = fit_and_report(bad, out_path=out_forced, force=True)
        check(saved_forced is not None and out_forced.exists(), "--force writes the file")
        if out_forced.exists():
            check(json.loads(out_forced.read_text())["passed"] is False,
                  "forced file is marked passed=false")

        print("\n-- 4. degenerate input (4 collinear points) → FIT FAILED, no crash")
        collinear = [((0.0, 0.0), (0.0, 0.0)), ((10.0, 10.0), (20.0, 20.0)),
                     ((20.0, 20.0), (40.0, 40.0)), ((30.0, 30.0), (60.0, 60.0))]
        ok_deg, h_deg, saved_deg = fit_and_report(collinear, out_path=tmpdir / "never.json")
        check(not ok_deg and h_deg is None and saved_deg is None,
              "degenerate set reported, not saved, not raised")

        print("\n-- 5. non-interactive parsers reach the same numbers")
        specs = [f"{p[0]:.1f},{p[1]:.1f}={w[0]:.4f},{w[1]:.4f}" for p, w in good]
        roundtrip = [parse_point_spec(s) for s in specs]
        check(
            all(
                abs(a[0][0] - b[0][0]) < 1e-6 and abs(a[0][1] - b[0][1]) < 1e-6
                and abs(a[1][0] - b[1][0]) < 1e-3 and abs(a[1][1] - b[1][1]) < 1e-3
                for a, b in zip(roundtrip, good)
            ),
            "--points specs parse back to the same correspondences",
        )
        check(parse_point_spec("10,20 -> 0,0") == ((10.0, 20.0), (0.0, 0.0)),
              "'->' separator and whitespace tolerated")
        csv_path = tmpdir / "points.csv"
        csv_path.write_text("# px,py,X,Y\n" + "\n".join(
            f"{p[0]},{p[1]},{w[0]},{w[1]}" for p, w in good) + "\n")
        check(len(parse_points_file(csv_path)) == len(good), "--points-file CSV parses")
        json_path = tmpdir / "points.json"
        json_path.write_text(json.dumps({"points": [
            {"pixel": list(p), "world": list(w)} for p, w in good]}))
        check(parse_points_file(json_path) == parse_points_file(csv_path),
              "--points-file JSON == CSV")
        check(sheet_world_points(297, 210) == [(0.0, 0.0), (297.0, 0.0), (297.0, 210.0),
                                               (0.0, 210.0)], "A4 --sheet world points")

        print("\n-- 6. GUI pre-flight (a dead display must NOT kill the process)")
        saved_env = {k: os.environ.pop(k) for k in ("DISPLAY", "WAYLAND_DISPLAY")
                     if k in os.environ}
        try:
            reason = gui_unavailable_reason()
            check(reason is not None and "DISPLAY" in reason,
                  f"headless is detected before touching Qt ({reason})")
            try:
                click_points(tmpdir / "no-such-image.jpg")
                check(False, "click_points raises a readable error when headless")
            except RuntimeError as e:
                check("--points" in str(e), "click_points points at the keyboard-only path")
        finally:
            os.environ.update(saved_env)
        live = gui_unavailable_reason()
        print(f"  [--] this shell's GUI: {'usable' if live is None else 'NOT usable: ' + live}")

    print("\n" + "=" * 72)
    if failures:
        print(f"SELFTEST FAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    print("SELFTEST PASSED — fit → error → PASS/FAIL → save/refuse all work offline.")
    return 0


# ------------------------------------------------------------------ CLI
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=".venv/bin/python tools/calibrate_homography.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Fit and save the pixel→table homography (hardware/config/homography.json).",
        epilog="""\
PHYSICAL TARGET (5 min prep, do it before the hardware window)
  Tape a stiff sheet with 4 clearly marked corners flat on the mat. Known sizes:
  A4 landscape = 297 x 210 mm, A3 landscape = 420 x 297 mm. No checkerboard
  detection is involved — you click the corners yourself.

  Put the sheet's ORIGIN corner AT THE ARM BASE, with the long edge pointing away
  from the arm. Then the homography's world frame == the IK base frame (origin under
  the base pivot, +X away from the arm, +Y to the arm's left) and no offset maths is
  needed. See hardware/config/link-lengths.md, section "Frame convention".

CLICK ORDER for --sheet W,H  (4 clicks, and you type nothing):
     1 origin (0,0) at the arm base
     2 (W,0)   — W mm along +X, away from the arm
     3 (W,H)   — the far/diagonal corner
     4 (0,H)   — H mm along +Y, to the arm's left

EXAMPLES
  .venv/bin/python tools/calibrate_homography.py --selftest
  .venv/bin/python tools/calibrate_homography.py --sheet 297,210
  .venv/bin/python tools/calibrate_homography.py --image frame.jpg --sheet 297,210
  .venv/bin/python tools/calibrate_homography.py --image frame.jpg          # type mm per click
  .venv/bin/python tools/calibrate_homography.py \\
      --points "812,455=0,0" "1180,470=297,0" "1195,690=297,210" "800,672=0,210"
  .venv/bin/python tools/calibrate_homography.py --points-file points.csv   # px,py,X,Y per line

The fit is REFUSED unless every point reprojects within --tolerance (default 20 mm
= the 2 cm ship bar in hardware/config/cameras.md). --force overrides; don't.
""",
    )
    p.add_argument("--image", help="Use this existing overhead frame instead of capturing one.")
    p.add_argument("-d", "--device", help="Camera /dev/videoN or index (default: auto-select).")
    p.add_argument("-w", "--warmup", type=int, default=5, help="Frames to discard when capturing.")
    p.add_argument(
        "--points",
        nargs="+",
        metavar="px,py=X,Y",
        help="Non-interactive correspondences (≥4). Pixels left, table mm right.",
    )
    p.add_argument("--points-file", help="Non-interactive JSON or CSV of px,py,X,Y rows.")
    p.add_argument(
        "--pixels",
        nargs="+",
        metavar="px,py",
        help="Non-interactive pixels only; requires --sheet to supply the mm.",
    )
    p.add_argument(
        "--sheet",
        metavar="W,H",
        help="Sheet size in mm (e.g. 297,210): assigns (0,0),(W,0),(W,H),(0,H) to the "
        "4 points in click order, so no mm typing.",
    )
    p.add_argument("-o", "--out", default=str(DEFAULT_OUT_PATH), help="Where to save the JSON.")
    p.add_argument("--tolerance", type=float, default=SHIP_BAR_MM,
                   help=f"Max allowed reprojection error in mm (default {SHIP_BAR_MM:g}).")
    p.add_argument("--force", action="store_true",
                   help="Save even if the fit misses the ship bar (logs passed=false).")
    p.add_argument("--dry-run", action="store_true", help="Fit and report, write nothing.")
    p.add_argument("--selftest", action="store_true",
                   help="Prove the fit→error→PASS/FAIL→save path offline (no camera/display).")
    return p


def _pairs_from_args(args: argparse.Namespace) -> list[Pair]:
    """Resolve whichever input mode was given into correspondences."""
    sheet = _parse_xy(args.sheet, what="sheet size", context="--sheet") if args.sheet else None

    if args.points:
        return [parse_point_spec(s) for s in args.points]
    if args.points_file:
        return parse_points_file(args.points_file)
    if args.pixels:
        if sheet is None:
            raise RuntimeError("--pixels needs --sheet W,H (otherwise use --points px,py=X,Y)")
        pixels = [_parse_xy(s, what="pixel", context="--pixels") for s in args.pixels]
        world = sheet_world_points(*sheet)
        if len(pixels) != 4:
            raise RuntimeError(f"--pixels with --sheet needs exactly 4 points, got {len(pixels)}")
        return list(zip(pixels, world))

    # interactive
    image = Path(args.image) if args.image else _grab_frame(args.device, args.warmup)
    pixels = click_points(image, min_points=4)
    if sheet is not None:
        if len(pixels) != 4:
            raise RuntimeError(
                f"--sheet assigns exactly 4 corners but {len(pixels)} points were clicked; "
                "re-run and click 4, or drop --sheet and type the mm yourself"
            )
        world = sheet_world_points(*sheet)
        print("[calib] --sheet assignment (click order):")
        for i, (pix, wpt) in enumerate(zip(pixels, world), start=1):
            print(f"[calib]   #{i} px=({pix[0]:.1f},{pix[1]:.1f}) -> ({wpt[0]:.0f},{wpt[1]:.0f}) mm")
    else:
        world = _prompt_world_points(pixels)
    return list(zip(pixels, world))


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    try:
        pairs = _pairs_from_args(args)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if len(pairs) < 4:
        print(f"ERROR: need ≥4 correspondences, got {len(pairs)}", file=sys.stderr)
        return 2

    passed, _, _ = fit_and_report(
        pairs,
        tolerance=args.tolerance,
        out_path=None if args.dry_run else Path(args.out),
        force=args.force,
        note=f"image={args.image or 'live capture'}",
    )
    if passed:
        print("[calib] next: .venv/bin/python tools/pick_bin_regions.py  (see hardware/config/cameras.md)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
