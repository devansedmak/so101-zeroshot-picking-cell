"""Zero-shot detect on a still frame — perception entrypoint (VLM optional).

Runs the hosted VLM's ``detect_points`` on an image and prints where an item is,
in pixels. **Default is dry-run** (parses a bundled sample response — no network, no
credits), so the parsing/coordinate math is demoable offline. ``--run`` makes the real
VLM call; ``--save out.png`` writes an annotated overlay.

This is the offline-provable half of the loop seam (decisions.md D10/D13): the pixel
it prints is what planar homography will turn into a table coordinate for the pick.

Usage:
    python -m src.perception.detect_frame                         # dry-run sample
    python -m src.perception.detect_frame --run -i frame.jpg -q "red marker"
    python -m src.perception.detect_frame --run -i frame.jpg -q "red marker" --save out.png
"""

from __future__ import annotations

import argparse
import os
import sys

from .detect import TASK_POINTS, parse_detections, pick_target

# A canned detect_points response (Gemini shape: [{"point": [y, x], "label": ...}],
# 0-1000 scale) so --dry-run exercises the real parser with zero network/credits.
SAMPLE_OUTPUT = [
    {"point": [512, 300], "label": "red marker"},
    {"point": [640, 700], "label": "blue marker"},
    {"point": [480, 500], "label": "eraser"},
]
SAMPLE_SIZE = (1280, 720)  # a typical overhead frame (w, h)


def _dry_run(item: str) -> int:
    print("[detect_frame] DRY RUN — bundled sample, no network/credits.\n")
    w, h = SAMPLE_SIZE
    dets = parse_detections(SAMPLE_OUTPUT, w, h)
    for d in dets:
        print(f"  • {d.label:<12} px=({d.x:6.1f}, {d.y:6.1f})  norm=({d.nx:.3f}, {d.ny:.3f})")
    target = pick_target(dets, item)
    print()
    if target:
        print(f"[detect_frame] target for {item!r}: px=({target.x:.1f}, {target.y:.1f}) → feeds homography")
        return 0
    print(f"[detect_frame] no detection matched {item!r}", file=sys.stderr)
    return 1


def _image_size(path: str) -> tuple[int, int]:
    from PIL import Image  # lazy: only the real run needs Pillow

    with Image.open(path) as im:
        return im.size  # (w, h)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zero-shot detect on a still frame.")
    parser.add_argument("-i", "--image", help="Path to the frame (required with --run).")
    parser.add_argument("-q", "--item", default="red marker", help="Open-vocab item to point at.")
    parser.add_argument("--run", action="store_true", help="Make the real VLM call (default: dry-run sample).")
    parser.add_argument("--save", help="Write an annotated overlay PNG here (--run only).")
    args = parser.parse_args(argv)

    if not args.run:
        return _dry_run(args.item)

    if not args.image:
        print("ERROR: --run needs -i/--image.", file=sys.stderr)
        return 2

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    model = os.getenv("CYBERWAVE_VLM_MODEL", "gemini-robotics-er")

    from cyberwave import Cyberwave

    cw = Cyberwave()
    print(f"[detect_frame] model={model!r}  image={args.image!r}  item={args.item!r}")
    size = _image_size(args.image)
    # Call the SDK directly (not the detect() helper) so we keep the raw ``result``
    # object for save_annotated_image below.
    result = cw.mlmodels.run(model, image=args.image, prompt=args.item, structured_task=TASK_POINTS)
    dets = parse_detections(result.output, *size, default_label=args.item)

    for d in dets:
        print(f"  • {d.label:<12} px=({d.x:6.1f}, {d.y:6.1f})")
    target = pick_target(dets, args.item)
    if target:
        print(f"\n[detect_frame] target for {args.item!r}: px=({target.x:.1f}, {target.y:.1f})")
    else:
        print(f"\n[detect_frame] no detection matched {args.item!r}", file=sys.stderr)

    if args.save:
        result.save_annotated_image(args.image, args.save)
        print(f"[detect_frame] annotated overlay saved → {args.save}")

    return 0 if target else 1


if __name__ == "__main__":
    raise SystemExit(main())
