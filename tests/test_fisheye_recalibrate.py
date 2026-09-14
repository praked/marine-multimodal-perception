import numpy as np
import pytest

from scripts.sensor_processing.fisheye_recalibrate import (
    _coverage_report,
    _save_coverage_png,
    _yaml_block,
    calibrate,
)

SIZE = (864, 648)


def _pts(rng_seed, lo, hi, n=54):
    r = np.random.default_rng(rng_seed)
    return r.uniform(lo, hi, size=(n, 1, 2)).astype(np.float32)


def test_coverage_full_is_ok():
    pts = [_pts(i, [0, 0], [864, 648]) for i in range(8)]
    rep = _coverage_report(pts, SIZE)
    assert "OK: all regions covered" in rep


def test_coverage_centre_only_warns_bottom():
    pts = [_pts(i, [320, 230], [540, 410]) for i in range(8)]
    rep = _coverage_report(pts, SIZE)
    assert "WARNING" in rep
    # the near-field-specific warning must fire (bottom row empty)
    assert "NEAR-FIELD RANGE" in rep


def test_yaml_block_handles_both_D_shapes():
    K = np.array([[418.5, 0, 444.1], [0, 418.6, 300.5], [0, 0, 1.0]])
    for D in (np.array([[-0.029], [0.008], [-0.022], [0.010]]),
              np.array([-0.029, 0.008, -0.022, 0.010])):
        block = _yaml_block(K, D, SIZE)
        assert "image_size: [864, 648]" in block
        assert block.count("- [") == 7   # 3 K rows + 4 D rows
        assert "model: fisheye" in block


def test_save_coverage_png_writes(tmp_path):
    out = tmp_path / "cov.png"
    _save_coverage_png([_pts(0, [0, 0], [864, 648])], SIZE, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_calibrate_too_few_images_exits(tmp_path):
    import cv2
    # one blank image -> no corners -> < 5 usable -> SystemExit
    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), np.zeros((648, 864, 3), np.uint8))
    with pytest.raises(SystemExit):
        calibrate([str(p)], cols=9, rows=6, square_mm=25.0)
