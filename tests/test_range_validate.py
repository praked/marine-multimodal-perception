import cv2
import numpy as np

from scripts.eval.range_validate import _undistort_point, evaluate


def test_undistort_point_returns_finite():
    K = np.array([[418.5, 0, 444.1], [0, 418.6, 300.5], [0, 0, 1.0]])
    D = np.array([-0.029, 0.008, -0.022, 0.010])
    u, v = _undistort_point(432, 600, K, D)
    assert np.isfinite(u) and np.isfinite(v)


def test_evaluate_runs_and_ranges_a_below_centre_point(tmp_path):
    # A blank frame: horizon is low-confidence so range falls back to level.
    img = np.full((648, 864, 3), 80, np.uint8)
    p = tmp_path / "f.jpg"
    cv2.imwrite(str(p), img)
    # A point below the principal point (undistorted space) -> a finite range.
    samples = [{"image": str(p), "pixel": [432, 520],
                "pixel_space": "undistorted", "true_range_m": 5.0}]
    rows = evaluate(samples, intrinsics_path=None, use_horizon=False)
    assert len(rows) == 1
    _img, true_r, rng, _conf = rows[0]
    assert true_r == 5.0
    assert rng is not None and rng > 0


def test_evaluate_point_above_horizon_is_none(tmp_path):
    img = np.full((648, 864, 3), 80, np.uint8)
    p = tmp_path / "f.jpg"
    cv2.imwrite(str(p), img)
    # Well above the principal point with level reference -> no water intersection.
    samples = [{"image": str(p), "pixel": [432, 50],
                "pixel_space": "undistorted", "true_range_m": 5.0}]
    rows = evaluate(samples, intrinsics_path=None, use_horizon=False)
    assert rows[0][2] is None
