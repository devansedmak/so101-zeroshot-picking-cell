#!/usr/bin/env python3
"""Generate the A4 homography calibration sheet as a print-exact PDF.

Why a generator and not a downloaded checkerboard: `tools/calibrate_homography.py`
does no marker detection. You CLICK 4 marks and `--sheet W,H` assigns their mm
coordinates in click order -> (0,0), (W,0), (W,H), (0,H). So the sheet must carry
exactly 4 unambiguous marks, labelled in that order, at a spacing you can verify
with a ruler. This writes a vector PDF (pure stdlib, no reportlab) so the geometry
is exact at 100% scale.

    .venv/bin/python tools/make_calibration_sheet.py            # -> hardware/config/calibration-sheet-a4.pdf
    .venv/bin/python tools/make_calibration_sheet.py --rect 250,160 --out /tmp/sheet.pdf

Print at **100% / "Actual size"** (never "Fit to page"), then measure the printed
100 mm verification bar. If it is not 100 mm, do not re-print blindly: measure the
real cross-to-cross spacing and pass that to `--sheet` instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PT_PER_MM = 72.0 / 25.4
A4_LONG_MM, A4_SHORT_MM = 297.0, 210.0

# The 4 corner marks, in the click order calibrate_homography.py expects.
CORNER_LABELS = ("1 ORIGIN (arm base)", "2 +X (away from arm)", "3 DIAGONAL", "4 +Y (arm's LEFT)")
# Optional extras: typed by hand into --points only if a 4-point fit is marginal.
EXTRA_FRACTIONS = ((0.24, 0.75), (0.76, 0.25))


def _esc(s: str) -> str:
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


class Canvas:
    """Minimal PDF content-stream builder working in millimetres."""

    def __init__(self) -> None:
        self.ops: list[str] = ["0 0 0 RG", "0 0 0 rg"]

    def width(self, mm: float) -> None:
        self.ops.append(f"{mm * PT_PER_MM:.3f} w")

    def line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.ops.append(
            f"{x0 * PT_PER_MM:.3f} {y0 * PT_PER_MM:.3f} m "
            f"{x1 * PT_PER_MM:.3f} {y1 * PT_PER_MM:.3f} l S"
        )

    def circle(self, cx: float, cy: float, r: float) -> None:
        k = 0.5522847498 * r
        x, y = cx, cy
        pts = [
            ((x + r, y), (x + r, y + k), (x + k, y + r), (x, y + r)),
            ((x, y + r), (x - k, y + r), (x - r, y + k), (x - r, y)),
            ((x - r, y), (x - r, y - k), (x - k, y - r), (x, y - r)),
            ((x, y - r), (x + k, y - r), (x + r, y - k), (x + r, y)),
        ]
        (sx, sy) = pts[0][0]
        self.ops.append(f"{sx * PT_PER_MM:.3f} {sy * PT_PER_MM:.3f} m")
        for _, c1, c2, end in pts:
            self.ops.append(
                " ".join(
                    f"{v * PT_PER_MM:.3f}"
                    for v in (c1[0], c1[1], c2[0], c2[1], end[0], end[1])
                )
                + " c"
            )
        self.ops.append("S")

    def text(self, x: float, y: float, s: str, size: float = 8.0, bold: bool = False) -> None:
        font = "/F2" if bold else "/F1"
        self.ops.append(
            f"BT {font} {size:.1f} Tf 1 0 0 1 {x * PT_PER_MM:.3f} {y * PT_PER_MM:.3f} Tm "
            f"({_esc(s)}) Tj ET"
        )

    def stream(self) -> bytes:
        return "\n".join(self.ops).encode("latin-1")


def build_pdf(canvas: Canvas, page_w_mm: float, page_h_mm: float) -> bytes:
    """Assemble a one-page PDF with Helvetica / Helvetica-Bold available."""
    content = canvas.stream()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        (
            "<</Type/Page/Parent 2 0 R/MediaBox[0 0 "
            f"{page_w_mm * PT_PER_MM:.2f} {page_h_mm * PT_PER_MM:.2f}]"
            "/Resources<</Font<</F1 5 0 R/F2 6 0 R>>>>/Contents 4 0 R>>"
        ).encode("latin-1"),
        b"<</Length " + str(len(content)).encode() + b">>\nstream\n" + content + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica/Encoding/WinAnsiEncoding>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica-Bold/Encoding/WinAnsiEncoding>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    )
    return bytes(out)


def draw_sheet(rect_w: float, rect_h: float, page_w: float, page_h: float) -> Canvas:
    c = Canvas()
    x0 = (page_w - rect_w) / 2.0
    y0 = (page_h - rect_h) / 2.0
    corners = [(x0, y0), (x0 + rect_w, y0), (x0 + rect_w, y0 + rect_h), (x0, y0 + rect_h)]
    world = [(0.0, 0.0), (rect_w, 0.0), (rect_w, rect_h), (0.0, rect_h)]

    # Faint rectangle joining the marks: makes a wrong-corner click obvious by eye.
    c.width(0.15)
    for i in range(4):
        c.line(*corners[i], *corners[(i + 1) % 4])

    # The 4 primary marks: 24 mm crosshair + 5 mm ring, centre left clean to click.
    for (cx, cy), (wx, wy), label in zip(corners, world, CORNER_LABELS):
        c.width(0.7)
        c.line(cx - 12, cy, cx - 1.5, cy)
        c.line(cx + 1.5, cy, cx + 12, cy)
        c.line(cx, cy - 12, cx, cy - 1.5)
        c.line(cx, cy + 1.5, cx, cy + 12)
        c.width(0.4)
        c.circle(cx, cy, 5.0)
        inward_x = 1 if cx < page_w / 2 else -1
        inward_y = 1 if cy < page_h / 2 else -1
        tx = cx + (14 if inward_x > 0 else -14 - 2.1 * len(label))
        ty = cy + inward_y * 15
        c.text(tx, ty, label, size=9.0, bold=True)
        c.text(tx, ty - 4.2, f"= ({wx:g}, {wy:g}) mm", size=7.5)

    # Optional extra points, for the documented 6-point re-fit if a 4-point fit is marginal.
    c.width(0.4)
    for fx, fy in EXTRA_FRACTIONS:
        ex, ey = x0 + fx * rect_w, y0 + fy * rect_h
        c.line(ex - 6, ey, ex + 6, ey)
        c.line(ex, ey - 6, ex, ey + 6)
        c.text(ex + 7.5, ey - 1.2, f"({fx * rect_w:.0f}, {fy * rect_h:.0f}) mm", size=6.5)

    # Centre block: what this sheet is, and the command it feeds.
    cx0, cy0 = x0 + rect_w * 0.5 - 62, y0 + rect_h * 0.60
    c.text(cx0, cy0, "SO-101 homography calibration sheet", size=11.0, bold=True)
    c.text(cx0, cy0 - 6.0, f"cross-to-cross spacing: {rect_w:g} x {rect_h:g} mm", size=8.5)
    c.text(cx0, cy0 - 11.0, "PRINT AT 100% / ACTUAL SIZE - never 'fit to page'", size=8.5, bold=True)
    c.text(cx0, cy0 - 16.0, f"then: tools/calibrate_homography.py --image /tmp/calib.jpg --sheet {rect_w:g},{rect_h:g}", size=7.5)
    c.text(cx0, cy0 - 21.0, "click marks 1 -> 2 -> 3 -> 4, then ENTER", size=7.5)
    c.text(cx0, cy0 - 26.0, "mark 1 goes AT THE ARM BASE; mark 2 points away from the arm;", size=7.5)
    c.text(cx0, cy0 - 31.0, "mark 4 must end up on the arm's LEFT (else the sheet is flipped)", size=7.5)

    # 100 mm verification bar with 10 mm ticks: catches any printer scaling.
    bx, by = x0 + rect_w * 0.5 - 50, y0 + rect_h * 0.34
    c.width(0.5)
    c.line(bx, by, bx + 100, by)
    for i in range(11):
        h = 3.5 if i % 5 else 5.5
        c.line(bx + i * 10, by, bx + i * 10, by + h)
    c.text(bx, by - 5.0, "VERIFY WITH A RULER: this bar is exactly 100 mm (ticks 10 mm)", size=7.5, bold=True)
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rect", default="250,160", help="cross-to-cross spacing W,H in mm (default 250,160)")
    ap.add_argument("--page", default=f"{A4_LONG_MM:g},{A4_SHORT_MM:g}", help="page size mm (default A4 landscape)")
    ap.add_argument("--out", default="hardware/config/calibration-sheet-a4.pdf")
    args = ap.parse_args()

    rect_w, rect_h = (float(v) for v in args.rect.split(","))
    page_w, page_h = (float(v) for v in args.page.split(","))
    margin_x, margin_y = (page_w - rect_w) / 2.0, (page_h - rect_h) / 2.0
    if rect_w <= 0 or rect_h <= 0:
        raise SystemExit("--rect must be positive")
    if margin_x < 8 or margin_y < 8:
        raise SystemExit(
            f"margins {margin_x:.1f}/{margin_y:.1f} mm are inside typical printer no-print borders; "
            "shrink --rect (250,160 is the tested default)"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas = draw_sheet(rect_w, rect_h, page_w, page_h)
    out.write_bytes(build_pdf(canvas, page_w, page_h))
    print(f"[sheet] wrote {out} ({out.stat().st_size} bytes)")
    print(f"[sheet] page {page_w:g}x{page_h:g} mm, marks {rect_w:g}x{rect_h:g} mm, margins {margin_x:.1f}/{margin_y:.1f} mm")
    print(f"[sheet] print at 100%, verify the 100 mm bar, then use: --sheet {rect_w:g},{rect_h:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
