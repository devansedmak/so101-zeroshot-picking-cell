"""Planar homography: camera pixels -> flat table coordinates (mm).

The second half of the loop seam: the VLM points at an item in
pixels (src/perception/detect.py), and this maps that pixel to a real (X, Y) on the
table plane, which IK then turns into a pick pose. Valid because the scene is planar
(fixed overhead camera, no depth sensor).

Math is a Direct Linear Transform with Hartley isotropic normalization (numerically
stable on real, noisy calibration clicks). Pure numpy (no camera, no SDK), so it's
fully unit-testable offline; only the actual 4+ point calibration needs the mounted
camera. Ship bar: reprojection error < 2 cm (cameras.md checkpoint criterion).

Convention: ``H`` maps homogeneous pixel [x, y, 1] -> homogeneous table [X, Y, 1] up to
scale; ``project`` does the perspective divide. Calibrate in whatever world unit you
measure (we use **mm**); the map is unit-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

Point = Sequence[float]
_EPS = 1e-9


@dataclass(frozen=True)
class ReprojectionError:
    """How well a fitted homography reproduces its calibration points (world units)."""

    per_point: list[float]
    rms: float
    max: float

    def within(self, tolerance: float) -> bool:
        """True iff every point reprojects within ``tolerance`` (e.g. 20.0 mm = 2 cm)."""
        return self.max <= tolerance


def _as_array(pts: Iterable[Point], name: str) -> np.ndarray:
    arr = np.asarray(list(pts), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be an (N, 2) array of (x, y); got shape {arr.shape}")
    return arr


def _normalization_matrix(pts: np.ndarray) -> np.ndarray:
    """Hartley normalization: translate to centroid, scale so mean distance = √2."""
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = float(np.sqrt((shifted**2).sum(axis=1)).mean())
    if mean_dist < _EPS:
        raise ValueError("degenerate points: all correspondences are coincident")
    s = np.sqrt(2.0) / mean_dist
    return np.array(
        [[s, 0.0, -s * centroid[0]], [0.0, s, -s * centroid[1]], [0.0, 0.0, 1.0]]
    )


@dataclass(frozen=True)
class Homography:
    """A fitted pixel -> table planar homography (3×3, homogeneous)."""

    H: np.ndarray

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, pixel_pts: Iterable[Point], world_pts: Iterable[Point]) -> "Homography":
        """Solve for the homography from ≥4 (pixel -> world) correspondences.

        Raises ``ValueError`` on too few points, mismatched counts, or a degenerate
        (e.g. collinear) configuration that has no unique solution.
        """
        src = _as_array(pixel_pts, "pixel_pts")
        dst = _as_array(world_pts, "world_pts")
        if src.shape[0] != dst.shape[0]:
            raise ValueError(f"point count mismatch: {src.shape[0]} pixel vs {dst.shape[0]} world")
        if src.shape[0] < 4:
            raise ValueError(f"need ≥4 correspondences to fit a homography, got {src.shape[0]}")

        t_src = _normalization_matrix(src)
        t_dst = _normalization_matrix(dst)
        src_n = (t_src @ np.column_stack([src, np.ones(len(src))]).T).T
        dst_n = (t_dst @ np.column_stack([dst, np.ones(len(dst))]).T).T

        rows = []
        for (u, v, _), (X, Y, _) in zip(src_n, dst_n):
            rows.append([u, v, 1, 0, 0, 0, -X * u, -X * v, -X])
            rows.append([0, 0, 0, u, v, 1, -Y * u, -Y * v, -Y])
        A = np.asarray(rows, dtype=float)

        # Solution = right-singular vector of the smallest singular value.
        _, sv, vh = np.linalg.svd(A)
        if sv[-1] > _EPS and sv[-2] < _EPS:
            raise ValueError("degenerate configuration (e.g. collinear points): no unique homography")
        h_norm = vh[-1].reshape(3, 3)

        # Undo normalization: H = inv(T_dst) * H_norm * T_src.
        H = np.linalg.inv(t_dst) @ h_norm @ t_src
        if abs(H[2, 2]) < _EPS:
            raise ValueError("degenerate homography (H[2,2] ≈ 0)")
        return cls(H / H[2, 2])

    # -------------------------------------------------------------- project
    def project(self, xy: Point) -> tuple[float, float]:
        """Map one pixel (x, y) -> table (X, Y) with the perspective divide."""
        x, y = float(xy[0]), float(xy[1])
        v = self.H @ np.array([x, y, 1.0])
        if abs(v[2]) < _EPS:
            raise ValueError(f"point {xy} maps to infinity under this homography")
        return float(v[0] / v[2]), float(v[1] / v[2])

    def project_many(self, pts: Iterable[Point]) -> np.ndarray:
        """Vectorized :meth:`project` -> (N, 2) array of table coords."""
        arr = _as_array(pts, "pts")
        hom = np.column_stack([arr, np.ones(len(arr))]) @ self.H.T
        w = hom[:, 2:3]
        if np.any(np.abs(w) < _EPS):
            raise ValueError("one or more points map to infinity under this homography")
        return hom[:, :2] / w

    def inverse(self) -> "Homography":
        """The reverse map (table -> pixel), handy for verification overlays."""
        inv = np.linalg.inv(self.H)
        return Homography(inv / inv[2, 2])

    # ---------------------------------------------------------------- error
    def reprojection_error(
        self, pixel_pts: Iterable[Point], world_pts: Iterable[Point]
    ) -> ReprojectionError:
        """Per-point / RMS / max distance between projected pixels and true world pts."""
        dst = _as_array(world_pts, "world_pts")
        proj = self.project_many(pixel_pts)
        d = np.sqrt(((proj - dst) ** 2).sum(axis=1))
        return ReprojectionError(
            per_point=[float(x) for x in d],
            rms=float(np.sqrt((d**2).mean())),
            max=float(d.max()),
        )

    # ----------------------------------------------------------- serialize
    def save(self, path: str | Path, *, units: str = "mm", **meta: Any) -> Path:
        """Write H + metadata as JSON (git-friendly; see hardware/config/)."""
        p = Path(path)
        payload = {
            "type": "planar_homography",
            "src": "pixel",
            "dst": f"table_{units}",
            "units": units,
            "H": self.H.tolist(),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **meta,
        }
        p.write_text(json.dumps(payload, indent=2) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Homography":
        data = json.loads(Path(path).read_text())
        return cls(np.asarray(data["H"], dtype=float))
