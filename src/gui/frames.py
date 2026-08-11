"""The overhead frame the dashboard overlays sit on.

Two jobs, both of which must work with the network cable pulled out:

1. **Serve** whatever frame the run is currently looking at (:class:`FrameStore` holds
   JPEG *bytes* — never a path — so a real capture in ``/tmp`` that gets cleaned up
   mid-demo cannot blank the page).
2. **Synthesize** a convincing overhead scene when there is no real capture
   (:func:`synth_scene`), drawing the *real* calibrated zone rectangles from
   ``hardware/config/bin-regions.json``. The mock frame is therefore not a cartoon: the
   pixel rectangles the page draws its overlays from are the same numbers
   ``perception.verify`` would use on a real frame.

**View window.** The calibrated zones occupy a ~140×320 px patch of a 1920×1080 frame
(≈0.91 mm/px), which is unreadable on video. So a frame is served *cropped* to a view
window and the page maps overlay coordinates through the same window. Mock crops to the
zones by default; live defaults to the whole frame (safe: nothing can fall outside it)
and is overridable with ``--view``.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .modes import ZONES

#: Where the surveyed zone rectangles live (same file perception.verify reads).
BIN_REGIONS_PATH = Path("hardware/config/bin-regions.json")

#: Fallback frame size when nothing tells us otherwise (matches the calibrated capture).
DEFAULT_FRAME_SIZE = (1920, 1080)

#: 16:9 crop served to the page when auto-framing the zones.
VIEW_W, VIEW_H = 960, 540

JPEG_QUALITY = 86


@dataclass(frozen=True)
class View:
    """The crop of the full frame that is actually served, in full-frame pixels."""

    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def clamp_to(self, size: Sequence[int]) -> "View":
        fw, fh = int(size[0]), int(size[1])
        w = max(1, min(int(self.w), fw))
        h = max(1, min(int(self.h), fh))
        x = max(0, min(int(self.x), fw - w))
        y = max(0, min(int(self.y), fh - h))
        return View(x, y, w, h)


def full_view(size: Sequence[int] = DEFAULT_FRAME_SIZE) -> View:
    return View(0, 0, int(size[0]), int(size[1]))


def parse_view(spec: str) -> View:
    """``"430,296,960,540"`` → :class:`View`."""
    parts = [p for p in str(spec).replace("x", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"bad view {spec!r}: expected x,y,w,h")
    return View(*(int(float(p)) for p in parts))


# ------------------------------------------------------------------ zone geometry
def load_zone_rects(path: str | Path = BIN_REGIONS_PATH) -> dict[str, tuple[float, ...]]:
    """``{label: (x0, y0, x1, y1)}`` in full-frame pixels, from the calibration file.

    Falls back to a plausible synthetic layout if the file is missing, so the dashboard
    still runs on a laptop that never did the survey (it says so on the page).
    """
    import json

    p = Path(path)
    if not p.exists():
        return _fallback_rects()
    try:
        data = json.loads(p.read_text())
        rects: dict[str, tuple[float, ...]] = {}
        for r in data.get("regions", []) if isinstance(data, dict) else data:
            label = str(r["label"]).strip().upper()
            x0, x1 = sorted((float(r["x0"]), float(r["x1"])))
            y0, y1 = sorted((float(r["y0"]), float(r["y1"])))
            rects[label] = (x0, y0, x1, y1)
        return rects or _fallback_rects()
    except (OSError, ValueError, KeyError, TypeError):
        return _fallback_rects()


def load_frame_size(path: str | Path = BIN_REGIONS_PATH) -> tuple[int, int]:
    """The pixel size the zone rectangles were surveyed on."""
    import json

    p = Path(path)
    if not p.exists():
        return DEFAULT_FRAME_SIZE
    try:
        data = json.loads(p.read_text())
        size = data.get("frame_size") if isinstance(data, dict) else None
        if size and len(size) >= 2:
            return int(size[0]), int(size[1])
    except (OSError, ValueError, TypeError):
        pass
    return DEFAULT_FRAME_SIZE


def _fallback_rects() -> dict[str, tuple[float, ...]]:
    return {
        "A": (841.5, 664.5, 961.5, 724.5),
        "B": (843.0, 502.5, 916.5, 612.0),
        "C": (855.0, 408.0, 979.5, 447.0),
    }


def auto_view(
    rects: dict[str, tuple[float, ...]],
    size: Sequence[int],
    *,
    extra: Sequence[Sequence[float]] = (),
) -> View:
    """A 960×540 window centred on the action — the crop that makes the zones readable.

    ``extra`` are additional points that must be framed (the mock passes the props'
    resting positions). Without them the window centres on the zones alone, which pushes
    the pick area into a corner and wastes a third of the shot on empty table — on video
    that reads as "the interesting part is off-screen".
    """
    if not rects:
        return full_view(size)
    xs = [v for r in rects.values() for v in (r[0], r[2])]
    ys = [v for r in rects.values() for v in (r[1], r[3])]
    for point in extra:
        xs.append(float(point[0]))
        ys.append(float(point[1]))
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return View(int(cx - VIEW_W / 2), int(cy - VIEW_H / 2), VIEW_W, VIEW_H).clamp_to(size)


def zone_centre(rect: Sequence[float]) -> tuple[float, float]:
    return ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)


# ------------------------------------------------------------------ synthesis
def _hex_bgr(css: str) -> tuple[int, int, int]:
    h = css.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


#: The props on the table. ``(name, colour_css, w, h)`` in full-frame px (≈0.91 mm/px).
PROPS: tuple[tuple[str, str, int, int], ...] = (
    ("red marker", "#d62828", 90, 22),
    ("blue marker", "#2b5cd6", 90, 22),
    ("eraser", "#e8e2d4", 40, 26),
    ("green sharpener", "#3f9e5a", 34, 26),
)

#: Where each prop rests on the table before the pick (full-frame px).
PROP_HOME: dict[str, tuple[int, int]] = {
    "red marker": (640, 470),
    "blue marker": (655, 560),
    "eraser": (600, 650),
    "green sharpener": (690, 680),
}


def synth_scene(
    *,
    rects: dict[str, tuple[float, ...]] | None = None,
    size: Sequence[int] = DEFAULT_FRAME_SIZE,
    held: str | None = None,
    placed: tuple[str, str] | None = None,
    dropped: tuple[str, tuple[float, float]] | None = None,
    label: str = "",
) -> Any:
    """Render a plausible overhead frame as a BGR numpy array.

    ``held`` removes a prop from the table (it is in the gripper). ``placed=(item, zone)``
    draws it inside a zone rectangle; ``dropped=(item, (x, y))`` draws it at an arbitrary
    pixel (the "missed the zone" frame the verifier must catch).
    """
    import cv2
    import numpy as np

    w, h = int(size[0]), int(size[1])
    rects = rects or load_zone_rects()
    rng = np.random.default_rng(7)

    # Table: a warm grey desk with fine grain + a soft light falloff.
    img = np.full((h, w, 3), (122, 128, 136), dtype=np.uint8)
    grain = rng.normal(0, 5.5, (h, w, 1)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + grain, 0, 255)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    falloff = 1.06 - 0.30 * (((xx - w * 0.45) / w) ** 2 + ((yy - h * 0.5) / h) ** 2) * 4
    img = np.clip(img * falloff[..., None], 0, 255).astype(np.uint8)

    # The printed calibration sheet (it is really on the table, and it sells the shot).
    sx, sy, sw, sh = 505, 330, 300, 210
    cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (238, 240, 243), -1)
    cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (176, 179, 184), 2)
    for i, (mx, my) in enumerate(
        ((sx + 26, sy + 26), (sx + sw - 26, sy + 26), (sx + sw - 26, sy + sh - 26), (sx + 26, sy + sh - 26))
    ):
        cv2.circle(img, (mx, my), 7, (60, 62, 66), -1)
        cv2.circle(img, (mx, my), 12, (60, 62, 66), 1)
        cv2.putText(img, str(i + 1), (mx + 15, my + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 62, 66), 1)

    # The three paper zones, at their surveyed rectangles.
    for lbl, rect in sorted(rects.items()):
        zone = ZONES.get(lbl)
        colour = _hex_bgr(zone.css) if zone else (200, 200, 200)
        x0, y0, x1, y1 = (int(round(v)) for v in rect)
        patch = img[y0:y1, x0:x1]
        if patch.size:
            tint = np.full_like(patch, colour, dtype=np.uint8)
            img[y0:y1, x0:x1] = cv2.addWeighted(patch, 0.22, tint, 0.78, 0)
        cv2.rectangle(img, (x0, y0), (x1, y1), tuple(int(c * 0.65) for c in colour), 2)
        cv2.putText(img, lbl, (x0 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    def draw_prop(name: str, cx: float, cy: float, angle: float = 0.0) -> None:
        spec = next((p for p in PROPS if p[0] == name), None)
        if spec is None:
            spec = (name, "#8a8f98", 50, 24)
        _, css, pw, ph = spec
        colour = _hex_bgr(css)
        box = cv2.boxPoints(((float(cx), float(cy)), (float(pw), float(ph)), float(angle)))
        pts = box.astype(np.int32)
        shadow = (pts + np.array([4, 5], dtype=np.int32)).astype(np.int32)
        cv2.fillConvexPoly(img, shadow, (86, 90, 96))
        cv2.fillConvexPoly(img, pts, colour)
        cv2.polylines(img, [pts], True, tuple(int(c * 0.6) for c in colour), 1)

    for name, (px, py) in PROP_HOME.items():
        if name == held or (placed and name == placed[0]) or (dropped and name == dropped[0]):
            continue
        draw_prop(name, px, py, angle=(hash(name) % 40) - 20)

    if placed:
        item, lbl = placed
        rect = rects.get(str(lbl).upper())
        if rect:
            cx, cy = zone_centre(rect)
            draw_prop(item, cx, cy, angle=12)
    if dropped:
        item, (dx, dy) = dropped
        draw_prop(item, dx, dy, angle=-8)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    stamp = label or "CAM0 · overhead · 1920x1080"
    cv2.putText(img, stamp, (24, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (18, 18, 18), 4)
    cv2.putText(img, stamp, (24, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 1)
    return img


def encode_jpeg(img: Any, view: View | None = None) -> bytes:
    """Crop to ``view`` (if any) and JPEG-encode."""
    import cv2

    if view is not None:
        h, w = img.shape[:2]
        v = view.clamp_to((w, h))
        img = img[v.y : v.y + v.h, v.x : v.x + v.w]
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:  # pragma: no cover — cv2 only fails here on a corrupt array
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf.tobytes())


def find_real_frame(*dirs: str | Path) -> Path | None:
    """The newest real capture we can use as the backdrop, if one exists."""
    candidates: list[Path] = []
    for d in dirs or (Path("hardware/config"), Path("hardware"), Path("docs")):
        p = Path(d)
        if p.is_dir():
            candidates += [f for f in p.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.stat().st_mtime)


def read_frame_bytes(path: str | Path, view: View | None = None) -> bytes:
    """Load a real capture and re-encode it through the same crop as the mock frames."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read frame {path}")
    return encode_jpeg(img, view)


# ------------------------------------------------------------------ store
class FrameStore:
    """Thread-safe "what the camera is showing right now", as JPEG bytes.

    The dashboard is a viewer: it never opens a camera. Bytes arrive either from the
    mock synthesizer or from a real run pushing ``{"kind": "frame", "path": ...}``.
    """

    def __init__(self, *, view: View | None = None, size: Sequence[int] = DEFAULT_FRAME_SIZE) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes = b""
        self._token = "0"
        self.view = view or full_view(size)
        self.size = (int(size[0]), int(size[1]))

    @property
    def token(self) -> str:
        with self._lock:
            return self._token

    def get(self) -> tuple[bytes, str]:
        with self._lock:
            return self._jpeg, self._token

    def set_jpeg(self, jpeg: bytes, token: str | None = None) -> str:
        with self._lock:
            self._jpeg = jpeg
            self._token = token or str(int(_int(self._token) + 1))
            return self._token

    def set_image(self, img: Any, token: str | None = None) -> str:
        return self.set_jpeg(encode_jpeg(img, self.view), token)

    def set_path(self, path: str | Path, token: str | None = None) -> str:
        return self.set_jpeg(read_frame_bytes(path, self.view), token)

    def ensure(self) -> None:
        """Guarantee there is *something* to show (synthesize the idle table)."""
        with self._lock:
            empty = not self._jpeg
        if empty:
            self.set_image(synth_scene(size=self.size), token="idle")


def _int(v: Any) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def pixel_to_view_fraction(px: float, py: float, view: View) -> tuple[float, float]:
    """Full-frame pixel → 0..1 position inside the served crop (what the page needs)."""
    fx = (float(px) - view.x) / max(1.0, float(view.w))
    fy = (float(py) - view.y) / max(1.0, float(view.h))
    return fx, fy


def jitter(x: float, y: float, radius: float, seed: int = 0) -> tuple[float, float]:
    """A deterministic small offset — makes canned detections look measured, not typed."""
    a = (seed * 2.399963) % (2 * math.pi)
    return x + radius * math.cos(a), y + radius * math.sin(a)
