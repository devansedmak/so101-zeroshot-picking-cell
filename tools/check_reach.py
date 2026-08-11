#!/usr/bin/env python3
"""Are the three paper zones inside the arm's reachable band? One command, one verdict.

WHY: the zone rectangles are surveyed in PIXELS (`hardware/config/bin-regions.json`) and
the arm thinks in MILLIMETRES from its own pan axis. Nothing in the loop ever compared the
two, so a layout that simply cannot be reached looks perfectly healthy right up to the
moment `poses.pick_plan_from_table` raises `Unreachable` mid-demo. This tool closes that
gap: pixels → table mm → radius → reachable yes/no, with the *binding joint named*, and a
non-zero exit if any zone fails. Run it after sliding the mat; nothing here touches
hardware, the network, or any config file.

THE BAND. With the gripper held vertical (`ik.solve_ik`) reach is a thin annulus, not a
disc. Outside it the wrist cannot stay pointing down; inside it the elbow folds past its
limit. Both ends come from `motion.DEFAULT_JOINT_LIMITS`, so this tool bisects the real
`solve_ik` + `in_limits` rather than re-deriving the trigonometry — if the limits or the
link lengths change, the band here follows automatically.

TWO IK MODELS, ON PURPOSE. `ik.py` assumes the shoulder-lift pivot is coaxial with the
base-pan axis. It is not: it sits ~30.4 mm forward and ~18.3 mm lateral of it
(progress.md 2026-08-11, hardware/config/link-lengths.md). The correction is derived but
deliberately not implemented before the demo, and it shifts the whole band OUTWARD by
~31 mm. So a layout tuned to the shipped model can be un-reachable under the corrected one
and vice versa. This tool therefore evaluates **both** and by default demands a zone pass
under both — the only placement that survives implementing (or not implementing) the fix.

EXTRAPOLATION WARNING. The homography was fitted from marks on one printed 297x210 mm
sheet. Outside that rectangle it extrapolates, and extrapolation of a projective map is
not gentle: an independently fitted affine agrees to sub-mm on the sheet and diverges by
tens of mm ~10 cm outside it. Any zone centre that lands outside the printed rectangle is
flagged, because its radius carries an error bar the reprojection RMS does not describe.
When the tape measure and this tool disagree out there, believe the tape measure.

CLI:
    .venv/bin/python tools/check_reach.py                    # verdict for the current files
    .venv/bin/python tools/check_reach.py --slide=-30,0      # preview a 3 cm slide, files untouched
    .venv/bin/python tools/check_reach.py --suggest          # best slide + its margin
    .venv/bin/python tools/check_reach.py --selftest         # prove the maths, no config needed
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `.venv/bin/python tools/check_reach.py`
    sys.path.insert(0, str(REPO_ROOT))

from src.control.ik import PICK_Z, Unreachable, in_limits, solve_ik  # noqa: E402
from src.control.motion import DEFAULT_JOINT_LIMITS, JOINT_LABELS  # noqa: E402
from src.perception.homography import Homography  # noqa: E402
from src.perception.verify import (  # noqa: E402
    DEFAULT_BIN_REGIONS_PATH,
    BinRegion,
    BinRegionsMissing,
    load_bin_regions,
)

__all__ = [
    "COAXIAL",
    "OFFSET",
    "SHEET_MM",
    "ZoneReach",
    "band",
    "best_slide",
    "joint_solution",
    "out_of_limits",
    "robust_band",
    "selftest",
    "zone_reaches",
]

DEFAULT_HOMOGRAPHY_PATH = REPO_ROOT / "hardware" / "config" / "homography.json"

# Offset of the shoulder-lift pivot from the base-pan axis (mm), measured off the official
# SO-101 URDF. `ik.py` ignores both; see the module docstring.
SHOULDER_FORWARD_MM: float = 30.4
SHOULDER_LATERAL_MM: float = 18.3

# The printed calibration sheet: A4 landscape, origin corner AT THE ARM BASE, +X away from
# the arm, +Y to its left (hardware/config/cameras.md step 0 / step 5 `--sheet 297,210`).
# This rectangle is the homography's interpolation region — outside it, radii extrapolate.
SHEET_MM: tuple[float, float] = (297.0, 210.0)

COAXIAL = "coaxial"  # exactly what src/control/ik.py does today
OFFSET = "offset"    # with the derived-but-unimplemented pan→shoulder offset applied
MODELS: tuple[str, ...] = (COAXIAL, OFFSET)

_BISECT_TOL_MM = 1e-4


# --------------------------------------------------------------------- geometry
def joint_solution(x: float, y: float, z: float = PICK_Z, *, model: str = COAXIAL) -> dict[str, float]:
    """Joint angles (deg) for a table point, under ``COAXIAL`` or ``OFFSET``.

    Both models delegate the planar 2-link + vertical-wrist solve to the repo's own
    :func:`src.control.ik.solve_ik`; they differ only in the radius handed to it and in
    the base-pan angle. Raises :class:`Unreachable` exactly as ``solve_ik`` does.
    """
    d = math.hypot(x, y)
    if model == OFFSET:
        if d <= SHOULDER_LATERAL_MM:
            raise Unreachable(
                f"target ({x:.1f}, {y:.1f}) is {d:.1f} mm from the pan axis, inside the "
                f"{SHOULDER_LATERAL_MM:.1f} mm lateral shoulder offset"
            )
        # The shoulder plane is tangent to a circle of radius b about the pan axis, so the
        # in-plane radius is √(d²−b²) − a and the pan angle picks up −atan2(b, √(d²−b²)).
        s = math.sqrt(d * d - SHOULDER_LATERAL_MM**2)
        radius = s - SHOULDER_FORWARD_MM
        pan = math.degrees(math.atan2(y, x) - math.atan2(SHOULDER_LATERAL_MM, s))
    elif model == COAXIAL:
        radius = d
        pan = math.degrees(math.atan2(y, x))
    else:
        raise ValueError(f"unknown IK model {model!r}: expected one of {MODELS}")

    joints = solve_ik(radius, 0.0, z)  # solve in the shoulder plane…
    joints["1"] = pan                  # …then restore the true base pan
    return joints


def out_of_limits(joints: dict[str, float]) -> list[str]:
    """Servo IDs of every joint outside ``DEFAULT_JOINT_LIMITS`` (empty ⇒ all good)."""
    return [
        j
        for j, angle in joints.items()
        if j in DEFAULT_JOINT_LIMITS and not (DEFAULT_JOINT_LIMITS[j][0] <= angle <= DEFAULT_JOINT_LIMITS[j][1])
    ]


def _reachable_at(d: float, z: float, model: str) -> bool:
    """Is a point ``d`` mm from the pan axis (bearing irrelevant) reachable?"""
    try:
        joints = joint_solution(d, 0.0, z, model=model)
    except Unreachable:
        return False
    return in_limits(joints)


def _bisect(bad: float, good: float, z: float, model: str) -> float:
    """Boundary radius between a known-unreachable ``bad`` and a known-reachable ``good``.

    Returns the value on the REACHABLE side, so the band this builds is never optimistic.
    Works in either direction (``bad`` may be the inner or the outer probe).
    """
    while abs(good - bad) > _BISECT_TOL_MM:
        mid = (bad + good) / 2
        if _reachable_at(mid, z, model):
            good = mid
        else:
            bad = mid
    return good


def band(z: float = PICK_Z, *, model: str = COAXIAL) -> tuple[float, float]:
    """Reachable radius band (mm from the PAN axis) with the gripper held vertical.

    Found by bisecting the repo's own solver, so it can never drift out of step with
    ``ik.py``'s link lengths or ``DEFAULT_JOINT_LIMITS``.
    """
    mid = 200.0 if model == COAXIAL else 230.0
    if not _reachable_at(mid, z, model):
        raise ValueError(f"nothing is reachable at Z={z:.1f} mm under model {model!r}")
    return _bisect(0.0, mid, z, model), _bisect(600.0, mid, z, model)


def robust_band(z: float = PICK_Z, *, models: Iterable[str] = MODELS) -> tuple[float, float]:
    """Intersection of the per-model bands — radii that work whichever model is right."""
    bands = [band(z, model=m) for m in models]
    return max(b[0] for b in bands), min(b[1] for b in bands)


def binding_joint(d: float, z: float, model: str) -> str | None:
    """Human name of the joint that stops radius ``d``, or ``None`` if it is reachable."""
    try:
        joints = joint_solution(d, 0.0, z, model=model)
    except Unreachable:
        return "reach envelope (L1+L2)"
    bad = out_of_limits(joints)
    if not bad:
        return None
    return ", ".join(f"{JOINT_LABELS.get(j, j)} {joints[j]:+.1f}°" for j in sorted(bad))


# ------------------------------------------------------------------------ zones
@dataclass(frozen=True)
class ZoneReach:
    """One zone's centre carried all the way from pixels to a reachability verdict."""

    label: str
    pixel: tuple[float, float]
    table: tuple[float, float]
    radius: float
    verdicts: dict[str, str | None]  # model → binding joint (None ⇒ reachable)
    margin: float                    # mm to the nearest edge of the robust band (<0 ⇒ out)
    outside_sheet: float             # mm outside the printed calibration rectangle (≤0 ⇒ in)

    @property
    def ok(self) -> bool:
        return all(v is None for v in self.verdicts.values())


def _outside_sheet(table: tuple[float, float], sheet: tuple[float, float] = SHEET_MM) -> float:
    """How far outside the printed rectangle a table point is (≤0 ⇒ inside)."""
    x, y = table
    return max(-x, x - sheet[0], -y, y - sheet[1])


def zone_reaches(
    homography: Homography,
    regions: dict[str, BinRegion],
    *,
    slide: tuple[float, float] = (0.0, 0.0),
    z: float = PICK_Z,
    models: Sequence[str] = MODELS,
) -> list[ZoneReach]:
    """Project every zone centre and judge it. ``slide`` is a mat translation in table mm."""
    lo, hi = robust_band(z, models=models)
    out: list[ZoneReach] = []
    for label in sorted(regions):
        pixel = regions[label].center
        px, py = homography.project(pixel)
        table = (px + slide[0], py + slide[1])
        radius = math.hypot(*table)
        out.append(
            ZoneReach(
                label=label,
                pixel=pixel,
                table=table,
                radius=radius,
                verdicts={m: binding_joint(radius, z, m) for m in models},
                margin=min(radius - lo, hi - radius),
                outside_sheet=_outside_sheet(table),
            )
        )
    return out


def best_slide(
    points: Sequence[tuple[float, float]],
    lo: float,
    hi: float,
    *,
    toward_base_only: bool = False,
) -> tuple[tuple[float, float], float]:
    """Mat translation (mm) maximising the SMALLEST distance from either band edge.

    Maximising the minimum margin (rather than just "get them all inside") is what makes
    the instruction survive a few mm of ruler error. ``toward_base_only`` restricts the
    search to −X, i.e. the one-number "slide it N cm straight toward the base" move that a
    human can execute in a single attempt without rotating the sheet.

    Coarse grid then a shrinking pattern search: the objective is piecewise smooth with
    kinks where the binding zone changes, so a plain gradient step is not reliable.
    """
    pts = [(float(x), float(y)) for x, y in points]

    def margin(dx: float, dy: float) -> float:
        return min(
            min(math.hypot(x + dx, y + dy) - lo, hi - math.hypot(x + dx, y + dy)) for x, y in pts
        )

    span = max(math.hypot(x, y) for x, y in pts) + hi
    ys = [0.0] if toward_base_only else [i * 5.0 for i in range(int(-span // 5), int(span // 5) + 1)]
    best = max(
        ((margin(dx, dy), (dx, dy)) for dy in ys for dx in (i * 5.0 for i in range(int(-span // 5), 1))),
        key=lambda t: t[0],
    )
    value, (dx, dy) = best
    step = 2.5
    while step > 1e-3:
        moves = [(step, 0.0), (-step, 0.0)] if toward_base_only else [
            (step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step),
            (step, step), (step, -step), (-step, step), (-step, -step),
        ]
        for mx, my in moves:
            cand = margin(dx + mx, dy + my)
            if cand > value:
                value, dx, dy = cand, dx + mx, dy + my
                break
        else:
            step /= 2
    return (dx, dy), value


# ----------------------------------------------------------------------- report
def _parse_slide(text: str) -> tuple[float, float]:
    nums = [n for n in text.replace(",", " ").split() if n]
    if len(nums) != 2:
        raise ValueError(f"--slide wants DX,DY in table mm, got {text!r}")
    return float(nums[0]), float(nums[1])


def report(
    zones: Sequence[ZoneReach],
    *,
    z: float,
    models: Sequence[str],
    slide: tuple[float, float],
    suggest: bool,
) -> bool:
    """Print the table + verdict. Returns True iff every zone passes under every model."""
    for model in models:
        lo, hi = band(z, model=model)
        note = "as shipped in ik.py" if model == COAXIAL else "with the 30.4/18.3 mm pan offset"
        print(
            f"[reach] {model:8s} band at tip Z={z:+.1f} mm: {lo:7.1f} .. {hi:7.1f} mm "
            f"from the pan axis   ({note})"
        )
    lo, hi = robust_band(z, models=models)
    if len(models) > 1:
        print(f"[reach] {'robust':8s} band (intersection, target the middle): "
              f"{lo:7.1f} .. {hi:7.1f} mm   → aim for r ≈ {(lo + hi) / 2:.0f} mm")
    if slide != (0.0, 0.0):
        print(f"[reach] previewing a mat slide of ({slide[0]:+.1f}, {slide[1]:+.1f}) mm "
              f"— no file is modified")
    print()

    head = f"  {'zone':4s} {'pixel centre':>19s} {'table (X, Y) mm':>19s} {'r mm':>8s} {'margin':>8s}  verdict"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for zr in zones:
        verdict = "OK" if zr.ok else "UNREACHABLE"
        print(
            f"  {zr.label:4s} ({zr.pixel[0]:8.1f},{zr.pixel[1]:8.1f}) "
            f"({zr.table[0]:8.1f},{zr.table[1]:8.1f}) {zr.radius:8.1f} {zr.margin:+8.1f}  {verdict}"
        )
        for model in models:
            binding = zr.verdicts[model]
            if binding is not None:
                lo_m, hi_m = band(z, model=model)
                side = "too far out" if zr.radius > hi_m else "too close in"
                print(f"       ↳ {model} model: {side} — blocked by {binding}")
        if zr.outside_sheet > 0:
            print(f"       ⚠ {zr.outside_sheet:.0f} mm OUTSIDE the printed {SHEET_MM[0]:.0f}x"
                  f"{SHEET_MM[1]:.0f} mm calibration sheet — this radius is EXTRAPOLATED; "
                  f"cross-check it with a tape measure from the base column.")
    print()

    bad = [zr.label for zr in zones if not zr.ok]
    if bad:
        print(f"[reach] ✗ FAIL — zone(s) {', '.join(bad)} cannot be reached with the gripper vertical.")
    else:
        worst = min(zones, key=lambda zr: zr.margin)
        print(f"[reach] ✓ PASS — all {len(zones)} zones reachable; smallest margin "
              f"{worst.margin:+.1f} mm (zone {worst.label}).")

    if suggest or bad:
        pts = [zr.table for zr in zones]
        (dx, dy), value = best_slide(pts, lo, hi)
        (ax, _), avalue = best_slide(pts, lo, hi, toward_base_only=True)
        print()
        print("[reach] suggested FURTHER slide of the mat (from where it is now):")
        print(f"[reach]   simplest — {abs(ax) / 10:.1f} cm straight TOWARD THE BASE "
              f"(−X), no rotation → margin {avalue:+.1f} mm")
        print(f"[reach]   best 2-D — ({dx:+.1f}, {dy:+.1f}) mm i.e. {abs(dx) / 10:.1f} cm toward the base "
              f"and {abs(dy) / 10:.1f} cm to the arm's {'right' if dy < 0 else 'left'} "
              f"→ margin {value:+.1f} mm")
        print("[reach]   then RE-SURVEY: .venv/bin/python tools/pick_bin_regions.py --image <new frame>")
    return not bad


# --------------------------------------------------------------------- selftest
def selftest() -> int:
    """Prove the band maths and the slide search with no config files and no hardware."""
    failures: list[str] = []

    def check(label: str, ok: bool) -> None:
        print(f"  [{'ok ' if ok else 'FAIL'}] {label}")
        if not ok:
            failures.append(label)

    print("-- 1. bands come out of ik.py + DEFAULT_JOINT_LIMITS")
    lo_c, hi_c = band(model=COAXIAL)
    lo_o, hi_o = band(model=OFFSET)
    print(f"     coaxial [{lo_c:.1f}, {hi_c:.1f}]   offset [{lo_o:.1f}, {hi_o:.1f}]")
    check("coaxial band is a thin annulus, not a disc", 100.0 < lo_c < hi_c < 300.0)
    check("the ignored pan offset pushes the band outward", lo_o > lo_c and hi_o > hi_c)
    check("inner edge is the elbow", "elbow_flex" in (binding_joint(lo_c - 0.2, PICK_Z, COAXIAL) or ""))
    # Just outside it is the wrist that gives up first; the 2-link envelope only takes over
    # ~0.7 mm further out, so probe close to the edge or the wrong cause is reported.
    check("outer edge is the wrist", "wrist_flex" in (binding_joint(hi_c + 0.2, PICK_Z, COAXIAL) or ""))
    check("well past the outer edge it is the envelope",
          "envelope" in (binding_joint(hi_c + 20.0, PICK_Z, COAXIAL) or ""))

    print("-- 2. a mid-band point is reachable under BOTH models")
    lo, hi = robust_band()
    mid = (lo + hi) / 2
    check(f"r={mid:.1f} mm passes coaxial and offset",
          all(binding_joint(mid, PICK_Z, m) is None for m in MODELS))

    print("-- 3. the slide search rescues a deliberately too-far layout")
    far = [(hi + 40.0, 0.0), (hi + 30.0, 60.0), (hi + 20.0, -60.0)]
    (dx, dy), value = best_slide(far, lo, hi)
    print(f"     suggested ({dx:+.1f}, {dy:+.1f}) mm → margin {value:+.1f} mm")
    check("a positive-margin slide is found", value > 0)
    check("the slide is toward the base (−X)", dx < 0)
    moved = [math.hypot(x + dx, y + dy) for x, y in far]
    check("every moved point is inside the robust band", all(lo <= r <= hi for r in moved))

    print("-- 4. the sheet-validity flag fires outside the printed rectangle")
    check("inside the sheet is not flagged", _outside_sheet((150.0, 100.0)) <= 0)
    check("50 mm past the near edge is flagged", _outside_sheet((150.0, -50.0)) == 50.0)

    if failures:
        print(f"\n*** SELFTEST FAILED *** {len(failures)} check(s): {failures}")
        return 1
    print("\nSELFTEST PASSED")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="check_reach.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Can the arm reach the surveyed paper zones? Exits non-zero if not.",
        epilog="""\
WHAT IT DOES
  Reads hardware/config/homography.json + bin-regions.json, projects each zone's pixel
  centre to table mm, and checks the radius from the base-pan axis against the vertical-
  gripper reach band derived from src/control/ik.py and DEFAULT_JOINT_LIMITS. Read-only:
  no hardware, no network, no file is written.

EXAMPLES
  .venv/bin/python tools/check_reach.py                 # after sliding the mat + re-surveying
  .venv/bin/python tools/check_reach.py --slide=-30,0   # what a 3 cm slide toward the base would do
  .venv/bin/python tools/check_reach.py --suggest       # print the recommended slide too
  .venv/bin/python tools/check_reach.py --model coaxial # judge with the shipped IK only
  .venv/bin/python tools/check_reach.py --selftest      # no config files needed

EXIT CODES
  0 = every zone reachable · 1 = at least one zone unreachable · 2 = missing/bad input
""",
    )
    p.add_argument("--homography", default=str(DEFAULT_HOMOGRAPHY_PATH),
                   help="Pixel→table homography JSON (default: hardware/config/homography.json).")
    p.add_argument("--bin-regions", default=str(DEFAULT_BIN_REGIONS_PATH),
                   help="Surveyed zone rectangles JSON (default: hardware/config/bin-regions.json).")
    p.add_argument("--slide", metavar="DX,DY",
                   help="Hypothetical mat translation in table mm (+X away from the arm, "
                        "+Y to its left). Reports only — nothing is written.")
    p.add_argument("--z", type=float, default=PICK_Z,
                   help=f"Gripper-tip height above the table in mm (default {PICK_Z:g} = touching).")
    p.add_argument("--model", choices=(*MODELS, "both"), default="both",
                   help="Which IK model must accept the zones (default: both — the placement "
                        "then survives implementing the pan-offset fix, or not).")
    p.add_argument("--suggest", action="store_true",
                   help="Always print the margin-maximising slide, even when the zones pass.")
    p.add_argument("--selftest", action="store_true",
                   help="Prove the band maths offline; needs no config files.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.selftest:
        return selftest()

    models = MODELS if args.model == "both" else (args.model,)
    try:
        slide = _parse_slide(args.slide) if args.slide else (0.0, 0.0)
        homography = Homography.load(args.homography)
        regions = load_bin_regions(args.bin_regions)
        zones = zone_reaches(homography, regions, slide=slide, z=args.z, models=models)
    except (BinRegionsMissing, FileNotFoundError) as e:
        print(f"ERROR: missing calibration input: {e}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not zones:
        print("ERROR: no zones in the bin-regions file — nothing to check.", file=sys.stderr)
        return 2
    return 0 if report(zones, z=args.z, models=models, slide=slide, suggest=args.suggest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
