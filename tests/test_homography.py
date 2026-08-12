"""Unit tests for the pixel -> table planar homography (pure numpy, no camera).

Covers exact recovery (affine + projective), the <2 cm ship bar via reprojection
error, round-trip inverse, degenerate-input guards, and JSON save/load.
Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.perception import Homography, ReprojectionError


# --- exact recovery ------------------------------------------------------


def test_recovers_affine_scale_and_offset():
    # world = 0.5 * pixel + (10, 20): a clean, hand-computable mapping.
    def truth(px, py):
        return (0.5 * px + 10.0, 0.5 * py + 20.0)

    pix = [(0, 0), (100, 0), (100, 80), (0, 80)]
    world = [truth(*p) for p in pix]
    h = Homography.fit(pix, world)

    # a held-out point projects to the exact formula
    assert h.project((40, 60)) == pytest.approx(truth(40, 60), abs=1e-6)
    # calibration points reproject with ~zero error
    err = h.reprojection_error(pix, world)
    assert err.max < 1e-6


def test_recovers_a_true_projective_map():
    # A genuinely projective H (non-zero bottom row).
    H_true = np.array(
        [
            [1.2, 0.1, 15.0],
            [-0.05, 1.1, -8.0],
            [0.0008, -0.0003, 1.0],
        ]
    )

    def project_with(H, p):
        v = H @ np.array([p[0], p[1], 1.0])
        return (v[0] / v[2], v[1] / v[2])

    pix = [(10, 20), (300, 25), (290, 210), (15, 200), (150, 100), (60, 180)]
    world = [project_with(H_true, p) for p in pix]
    h = Homography.fit(pix, world)

    for p in [(120, 90), (250, 150), (33, 44)]:
        assert h.project(p) == pytest.approx(project_with(H_true, p), abs=1e-6)


# --- reprojection error / ship bar --------------------------------------


def test_reprojection_error_flags_2cm_tolerance():
    # world in mm; perfect fit -> within 2 cm; injected 30 mm outlier -> not within.
    pix = [(0, 0), (200, 0), (200, 200), (0, 200), (100, 100)]
    world = [(0, 0), (400, 0), (400, 400), (0, 400), (200, 200)]  # world = 2*pixel
    h = Homography.fit(pix, world)

    clean = h.reprojection_error(pix, world)
    assert clean.within(20.0)  # 2 cm

    noisy_world = list(world)
    noisy_world[4] = (230, 200)  # 30 mm off in X
    err = h.reprojection_error(pix, noisy_world)
    assert isinstance(err, ReprojectionError)
    assert not err.within(20.0)
    assert err.max == pytest.approx(30.0, abs=1.0)


# --- inverse round-trip --------------------------------------------------


def test_inverse_is_round_trip():
    pix = [(5, 5), (250, 10), (240, 190), (12, 205), (130, 95)]
    world = [(1.3 * x + 3, 0.9 * y - 4) for (x, y) in pix]
    h = Homography.fit(pix, world)
    inv = h.inverse()
    for p in pix:
        w = h.project(p)
        assert inv.project(w) == pytest.approx(p, abs=1e-6)


# --- guards --------------------------------------------------------------


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="≥4"):
        Homography.fit([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])


def test_mismatched_counts_raise():
    with pytest.raises(ValueError, match="mismatch"):
        Homography.fit([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (1, 0), (1, 1)])


def test_collinear_points_raise():
    pix = [(0, 0), (1, 1), (2, 2), (3, 3)]  # all on a line
    world = [(0, 0), (10, 0), (20, 0), (30, 0)]
    with pytest.raises(ValueError):
        Homography.fit(pix, world)


def test_bad_shape_raises():
    with pytest.raises(ValueError, match="N, 2"):
        Homography.fit([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [(0, 0)] * 4)


# --- serialization -------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    pix = [(0, 0), (100, 0), (100, 100), (0, 100)]
    world = [(0, 0), (250, 0), (250, 250), (0, 250)]
    h = Homography.fit(pix, world)

    path = h.save(tmp_path / "homography.json", units="mm", n_points=len(pix))
    loaded = Homography.load(path)

    assert np.allclose(loaded.H, h.H)
    assert loaded.project((50, 50)) == pytest.approx(h.project((50, 50)), abs=1e-9)
    import json

    meta = json.loads(path.read_text())
    assert meta["units"] == "mm" and meta["dst"] == "table_mm" and meta["n_points"] == 4
