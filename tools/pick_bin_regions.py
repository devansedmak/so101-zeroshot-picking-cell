"""Survey the bin pixel rectangles for closed-loop verification (D9 thesis).

WHY: `src/perception/verify.py` decides "did the item land in bin A?" by testing
whether the VLM's detected pixel falls inside bin A's rectangle. Bins don't move in a
fixed overhead view, so those rectangles are **calibration data**, not perception,
and `hardware/config/bin-regions.json` is the last missing input to the closed loop
(`run_order._live_verifier` silently disables verification without it, which would
quietly gut the demo's thesis).

WHY a tool instead of hand-writing the JSON: the labels must match the order source
exactly (`src/wms_mock/orders.py` uses ``"A"``/``"B"``/``"C"``, not ``"Bin A"``), the
rectangles must be in the *captured frame's* pixels (`_live_verifier` passes the real
image size to the VLM), and the schema must be the one `load_bin_regions` reads. All
three are easy to get subtly wrong at 8 a.m., so this tool writes via
`verify.save_bin_regions` / `BinRegion` and round-trips through `load_bin_regions`
before declaring success.

Same shape as tools/calibrate_homography.py: interactive clicking, a non-interactive
`--regions` path for when the display misbehaves, and `--selftest` that proves the
whole write -> reload -> verdict path with no camera and no display.

CLI:
    .venv/bin/python tools/pick_bin_regions.py --selftest                  # offline proof
    .venv/bin/python tools/pick_bin_regions.py --image frame.jpg           # click 2 corners per bin
    .venv/bin/python tools/pick_bin_regions.py --regions "A=100,100,300,240" "B=340,100,540,240"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `.venv/bin/python tools/pick_bin_regions.py`
    sys.path.insert(0, str(REPO_ROOT))

from src.perception.verify import (  # noqa: E402
    BinRegion,
    Detection,
    evaluate_placement,
    load_bin_regions,
    save_bin_regions,
)
from tools.calibrate_homography import click_points  # noqa: E402  (shared click widget)

__all__ = [
    "DEFAULT_BINS",
    "parse_region_spec",
    "regions_from_clicks",
    "selftest",
    "write_and_verify",
]

DEFAULT_OUT_PATH = REPO_ROOT / "hardware" / "config" / "bin-regions.json"

# Labels MUST match the order source (src/wms_mock/orders.py posts {"bin": "A"}):
# run_order looks the region up by that exact string, so "Bin A" would not resolve.
DEFAULT_BINS: tuple[str, ...] = ("A", "B", "C")

# What to do instead when no window can open (click_points appends this to its error).
_NO_GUI_HINT = (
    "Use the keyboard-only path instead (same code path, same result): read each bin's "
    'corner pixels off any image viewer and pass --regions "A=x0,y0,x1,y1" "B=..." '
    '"C=..." [--frame-size W,H].'
)


# ------------------------------------------------------------------ parsing
def parse_region_spec(spec: str) -> BinRegion:
    """Parse ``"A=x0,y0,x1,y1"`` -> :class:`BinRegion` (corner order is normalized)."""
    text = spec.strip()
    for sep in ("=", ":"):
        if sep in text:
            label, rect = text.split(sep, 1)
            break
    else:
        raise ValueError(f"bad region spec {spec!r}: expected 'LABEL=x0,y0,x1,y1' in pixels")
    label = label.strip()
    if not label:
        raise ValueError(f"bad region spec {spec!r}: empty bin label")
    nums = [n for n in rect.replace(",", " ").split() if n]
    if len(nums) != 4:
        raise ValueError(f"bad region spec {spec!r}: need 4 pixel numbers x0,y0,x1,y1")
    try:
        x0, y0, x1, y1 = (float(n) for n in nums)
    except ValueError as e:
        raise ValueError(f"bad region spec {spec!r}: not numeric ({e})") from e
    if x0 == x1 or y0 == y1:
        raise ValueError(f"bad region spec {spec!r}: zero-area rectangle")
    return BinRegion(label=label, x0=x0, y0=y0, x1=x1, y1=y1)


def regions_from_clicks(
    labels: Sequence[str], points: Sequence[tuple[float, float]]
) -> list[BinRegion]:
    """Pair up 2 clicks per bin (any two opposite corners) -> one rectangle per label."""
    if len(points) != 2 * len(labels):
        raise ValueError(
            f"need exactly 2 clicks per bin: {len(labels)} bins → {2 * len(labels)} points, "
            f"got {len(points)}"
        )
    regions: list[BinRegion] = []
    for i, label in enumerate(labels):
        (x0, y0), (x1, y1) = points[2 * i], points[2 * i + 1]
        if x0 == x1 or y0 == y1:
            raise ValueError(f"bin {label!r}: the two clicks form a zero-area rectangle")
        regions.append(BinRegion(label=label, x0=x0, y0=y0, x1=x1, y1=y1))
    return regions


# ------------------------------------------------------------- write + prove
def write_and_verify(
    regions: Sequence[BinRegion],
    *,
    out_path: str | Path | None = DEFAULT_OUT_PATH,
    frame_size: tuple[int, int] | None = None,
) -> tuple[bool, Path | None]:
    """Report the rectangles, save via ``save_bin_regions``, and prove the round-trip.

    THE one code path every input mode shares (and the one `--selftest` exercises).
    "Prove" = reload with ``load_bin_regions`` and run ``evaluate_placement`` on a
    synthetic detection at each bin's centre: if a centre point does not verify as
    inside, the file is useless to the loop and this returns ``False``.
    Overlapping rectangles are reported as a WARNING (an item in the overlap would
    verify for either bin) but do not block; a slightly loose box still ships.
    """
    print(f"[bins] {len(regions)} bin rectangles (pixels"
          + (f", frame {frame_size[0]}x{frame_size[1]}" if frame_size else "") + "):")
    for r in regions:
        cx, cy = r.center
        print(f"[bins]   {r.label:<4} [{r.x0:7.1f},{r.y0:7.1f} → {r.x1:7.1f},{r.y1:7.1f}]  "
              f"centre=({cx:.1f},{cy:.1f})  {r.x1 - r.x0:.0f}x{r.y1 - r.y0:.0f} px")

    labels = [r.label for r in regions]
    if len(set(labels)) != len(labels):
        print(f"[bins] ERROR: duplicate labels {labels} — one rectangle per bin, please.")
        return False, None
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            if a.x0 <= b.x1 and b.x0 <= a.x1 and a.y0 <= b.y1 and b.y0 <= a.y1:
                print(f"[bins] WARNING: {a.label} and {b.label} overlap — an item in the "
                      "overlap would verify for both. Shrink the boxes if the bins are "
                      "physically separate.")
    if frame_size:
        w, h = frame_size
        for r in regions:
            if r.x0 < 0 or r.y0 < 0 or r.x1 > w or r.y1 > h:
                print(f"[bins] WARNING: {r.label} sticks out of the {w}x{h} frame — "
                      "clicks may have come from a different image.")

    if out_path is None:
        print("[bins] (no output path; nothing written)")
        return True, None
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = save_bin_regions(out, regions, frame_size=frame_size)
    print(f"[bins] saved → {saved}")

    # Prove the schema is the one the loop reads, not merely valid JSON.
    reloaded = load_bin_regions(saved)
    if sorted(reloaded) != sorted(labels):
        print(f"[bins] ERROR: round-trip lost labels: wrote {sorted(labels)}, "
              f"read {sorted(reloaded)}")
        return False, saved
    for r in regions:
        back = reloaded[r.label]
        if (back.x0, back.y0, back.x1, back.y1) != (r.x0, r.y0, r.x1, r.y1):
            print(f"[bins] ERROR: round-trip changed bin {r.label}")
            return False, saved
        cx, cy = back.center
        verdict = evaluate_placement(
            [Detection(label="probe", x=cx, y=cy, nx=0.0, ny=0.0)], "probe", back
        )
        if not verdict.ok:
            print(f"[bins] ERROR: bin {r.label} does not verify its own centre: {verdict.reason}")
            return False, saved
    print(f"[bins] round-trip OK via load_bin_regions() — bins {sorted(reloaded)} are "
          "usable by run_order's verifier.")
    return True, saved


# ------------------------------------------------------------- interactive
def _interactive_regions(
    image: Path, labels: Sequence[str]
) -> tuple[list[BinRegion], tuple[int, int] | None]:
    print(f"[bins] click TWO opposite corners for each bin, in this order: "
          f"{', '.join(labels)}  ({2 * len(labels)} clicks total)")
    print("[bins] click just INSIDE each bin's rim: a box slightly smaller than the bin "
          "is safer than one that spills onto the table.")
    points = click_points(
        image,
        min_points=2 * len(labels),
        window="click 2 corners per bin  [ENTER=done  u=undo  ESC=abort]",
        fallback_hint=_NO_GUI_HINT,
    )
    if len(points) > 2 * len(labels):
        print(f"[bins] {len(points)} clicks for {len(labels)} bins — keeping the "
              f"first {2 * len(labels)} (use u to undo next time).")
        points = points[: 2 * len(labels)]
    return regions_from_clicks(labels, points), _image_size(image)


def _image_size(image: str | Path) -> tuple[int, int] | None:
    """(w, h) of the frame the rectangles were surveyed on, recorded in the JSON."""
    try:
        import cv2  # noqa: PLC0415, optional; the size is metadata, not load-bearing

        img = cv2.imread(str(image))
        if img is None:
            return None
        h, w = img.shape[:2]
        return (int(w), int(h))
    except ImportError:  # pragma: no cover, opencv is a project dependency
        return None


def _grab_frame(device: str | None, warmup: int) -> Path:
    from src.perception.capture import CaptureError, capture_still  # noqa: PLC0415

    try:
        path = capture_still(device, None, warmup)
    except CaptureError as e:
        raise RuntimeError(
            f"could not grab a frame: {e}\n"
            "  (or pass --image FILE — reuse the SAME frame the homography was "
            "calibrated on, so pixels agree)"
        ) from e
    print(f"[bins] captured {path}")
    return path


# ------------------------------------------------------------------ selftest
def selftest() -> int:
    """Prove parse -> write -> reload -> verdict offline, on synthetic rectangles."""
    print("=" * 72)
    print("SELFTEST tools/pick_bin_regions.py — synthetic rectangles, no camera/display")
    print("=" * 72)
    failures: list[str] = []

    def check(cond: bool, what: str) -> None:
        print(f"  [{'ok' if cond else 'FAIL'}] {what}")
        if not cond:
            failures.append(what)

    with tempfile.TemporaryDirectory(prefix="bins_selftest_") as tmp:
        tmpdir = Path(tmp)

        print("\n-- 1. --regions specs parse (and corner order is normalized)")
        specs = ["A=100,100,300,240", "B=340,100,540,240", "C=580,240,780,100"]
        regions = [parse_region_spec(s) for s in specs]
        check([r.label for r in regions] == ["A", "B", "C"], "labels A/B/C parsed")
        check((regions[2].x0, regions[2].y0, regions[2].x1, regions[2].y1)
              == (580.0, 100.0, 780.0, 240.0), "swapped corners normalized to TL/BR")
        check(parse_region_spec("A: 1 2 3 4").center == (2.0, 3.0),
              "':' separator and whitespace tolerated")

        print("\n-- 2. clicks → rectangles (2 clicks per bin, any two opposite corners)")
        clicks = [(300.0, 240.0), (100.0, 100.0),  # A, clicked bottom-right first
                  (340.0, 100.0), (540.0, 240.0),  # B
                  (580.0, 100.0), (780.0, 240.0)]  # C
        from_clicks = regions_from_clicks(DEFAULT_BINS, clicks)
        check([(r.label, r.x0, r.y0, r.x1, r.y1) for r in from_clicks]
              == [(r.label, r.x0, r.y0, r.x1, r.y1) for r in regions],
              "click order-independent: matches the --regions result")
        try:
            regions_from_clicks(("A", "B"), clicks[:3])
            check(False, "odd click count rejected")
        except ValueError:
            check(True, "odd click count rejected")

        print("\n-- 3. write → reload → per-bin centre verdict (the loop's own path)")
        out = tmpdir / "bin-regions.json"
        ok, saved = write_and_verify(regions, out_path=out, frame_size=(1280, 720))
        check(ok and saved == out and out.exists(), "wrote and round-tripped the file")
        payload = json.loads(out.read_text())
        check(payload["type"] == "bin_regions" and payload["units"] == "pixel"
              and payload["frame_size"] == [1280, 720],
              "schema is verify.save_bin_regions' (type/units/frame_size)")
        check(sorted(load_bin_regions(out)) == ["A", "B", "C"],
              "load_bin_regions() returns {label: BinRegion} for A/B/C")

        print("\n-- 4. the geometry actually discriminates bins")
        loaded = load_bin_regions(out)
        ax, ay = loaded["A"].center
        at_a = Detection(label="eraser", x=ax, y=ay, nx=0.0, ny=0.0)
        in_a = evaluate_placement([at_a], "eraser", loaded["A"])
        wrong_bin = evaluate_placement([at_a], "eraser", loaded["B"])
        missing = evaluate_placement([], "eraser", loaded["A"])
        check(in_a.ok, "item at bin A's centre verifies for A")
        check(not wrong_bin.ok and "outside bin B" in wrong_bin.reason,
              "same item does NOT verify for B (and says why)")
        check(not missing.ok and "not detected" in missing.reason,
              "no detection → not-ok with the 'not detected' reason")

        print("\n-- 5. guards")
        dup = write_and_verify([BinRegion("A", 0, 0, 10, 10), BinRegion("A", 20, 20, 30, 30)],
                               out_path=tmpdir / "dup.json")
        check(dup == (False, None) and not (tmpdir / "dup.json").exists(),
              "duplicate labels refused, nothing written")
        for bad in ("A=1,2,3", "=1,2,3,4", "A=1,2,1,4", "A 1 2 3 4"):
            try:
                parse_region_spec(bad)
                check(False, f"rejects bad spec {bad!r}")
            except ValueError:
                check(True, f"rejects bad spec {bad!r}")

        print("\n-- 6. no display → this tool's own keyboard-only advice, not a crash")
        saved_env = {k: os.environ.pop(k) for k in ("DISPLAY", "WAYLAND_DISPLAY")
                     if k in os.environ}
        try:
            _interactive_regions(tmpdir / "no-such-image.jpg", list(DEFAULT_BINS))
            check(False, "headless clicking raises a readable error")
        except RuntimeError as e:
            check("--regions" in str(e), "headless error names --regions")
        finally:
            os.environ.update(saved_env)

    print("\n" + "=" * 72)
    if failures:
        print(f"SELFTEST FAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    print("SELFTEST PASSED — parse → save_bin_regions → load_bin_regions → verdict all "
          "work offline.")
    return 0


# ------------------------------------------------------------------ CLI
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=".venv/bin/python tools/pick_bin_regions.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Survey bin pixel rectangles → hardware/config/bin-regions.json.",
        epilog="""\
WHAT TO CLICK
  Two opposite corners per bin, just INSIDE the rim, bins in the order given by
  --bins (default A B C — the labels src/wms_mock/orders.py posts). A box slightly
  SMALLER than the bin mouth is safer: an over-wide box would call a near-miss drop
  on the table a success, which is exactly the failure the verification exists to catch.

  Use the SAME frame you calibrated the homography on (--image frame.jpg), so the
  pixel coordinates in both calibration files refer to one camera pose. If the camera
  is bumped afterwards, both files must be redone.

EXAMPLES
  .venv/bin/python tools/pick_bin_regions.py --selftest
  .venv/bin/python tools/pick_bin_regions.py --image frame.jpg
  .venv/bin/python tools/pick_bin_regions.py --image frame.jpg --bins A B
  .venv/bin/python tools/pick_bin_regions.py --regions "A=100,100,300,240" "B=340,100,540,240" \\
      "C=580,100,780,240" --frame-size 1280,720

The file is reloaded with verify.load_bin_regions() and each bin must verify a
detection at its own centre before this exits 0.
""",
    )
    p.add_argument("--image", help="Existing overhead frame to click on (else one is captured).")
    p.add_argument("-d", "--device", help="Camera /dev/videoN or index (default: auto-select).")
    p.add_argument("-w", "--warmup", type=int, default=5, help="Frames to discard when capturing.")
    p.add_argument("--bins", nargs="+", default=list(DEFAULT_BINS), metavar="LABEL",
                   help="Bin labels in click order (default: A B C).")
    p.add_argument("--regions", nargs="+", metavar="LABEL=x0,y0,x1,y1",
                   help="Non-interactive rectangles in pixels — no display needed.")
    p.add_argument("--frame-size", metavar="W,H",
                   help="Frame the pixels refer to (recorded as metadata; auto with --image).")
    p.add_argument("-o", "--out", default=str(DEFAULT_OUT_PATH), help="Where to save the JSON.")
    p.add_argument("--dry-run", action="store_true", help="Report only, write nothing.")
    p.add_argument("--selftest", action="store_true",
                   help="Prove parse→save→load→verdict offline (no camera/display).")
    return p


def _regions_from_args(
    args: argparse.Namespace,
) -> tuple[list[BinRegion], tuple[int, int] | None]:
    frame_size: tuple[int, int] | None = None
    if args.frame_size:
        nums = [n for n in args.frame_size.replace(",", " ").split() if n]
        if len(nums) != 2:
            raise ValueError(f"--frame-size wants W,H, got {args.frame_size!r}")
        frame_size = (int(float(nums[0])), int(float(nums[1])))

    if args.regions:
        return [parse_region_spec(s) for s in args.regions], frame_size

    image = Path(args.image) if args.image else _grab_frame(args.device, args.warmup)
    regions, detected = _interactive_regions(image, list(args.bins))
    return regions, frame_size or detected


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    try:
        regions, frame_size = _regions_from_args(args)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    ok, _ = write_and_verify(
        regions,
        out_path=None if args.dry_run else Path(args.out),
        frame_size=frame_size,
    )
    if ok and not args.dry_run:
        print("[bins] closed loop is now calibrated: run_order --verify can use it.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
