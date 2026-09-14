"""CLI + corner-detection tests for scripts/sensor_processing/fisheye_recalibrate.

Real synthetic checkerboards drive cv2.findChessboardCorners (covers the
detection path); cv2.fisheye.calibrate itself is monkeypatched: a frontal-only
synthetic board set is numerically degenerate for the real solver, and the CLI
logic (reporting, YAML output, coverage PNG) is what we're testing.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from scripts.sensor_processing import fisheye_recalibrate as fr

COLS, ROWS = 4, 3          # INNER corners
SQ = 32                    # square px


def _board_image(offset=(0, 0), size=(400, 300)):
    """Draw a (COLS+1)x(ROWS+1)-square checkerboard with a white margin."""
    w, h = size
    img = np.full((h, w), 255, np.uint8)
    ox, oy = offset
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                y0, x0 = oy + 40 + r * SQ, ox + 40 + c * SQ
                img[y0:y0 + SQ, x0:x0 + SQ] = 0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _write_boards(tmp_path, n=6):
    for i in range(n):
        cv2.imwrite(str(tmp_path / f"b{i}.jpg"), _board_image(offset=(i * 6, i * 4)))
    # an unreadable "image" exercises the imread-None skip branch
    (tmp_path / "z_corrupt.jpg").write_text("not a jpeg")


def test_fisheye_calib_flags_fallback(monkeypatch):
    """Newer opencv-python wheels drop the cv2.fisheye.CALIB_* enum constants
    while keeping the functions (2026-07-06 CI failure). The helper must fall
    back to the ABI-stable numeric values (1<<1) + (1<<3) = 10."""
    monkeypatch.delattr(cv2.fisheye, "CALIB_RECOMPUTE_EXTRINSIC", raising=False)
    monkeypatch.delattr(cv2.fisheye, "CALIB_FIX_SKEW", raising=False)
    assert fr._fisheye_calib_flags() == 10


def test_fisheye_calib_flags_uses_cv2_when_present():
    """Whatever this cv2 build exposes, the helper returns an int == 10:
    the enum values are fixed in the C++ header."""
    assert fr._fisheye_calib_flags() == 10


def _fake_calibrate(objpoints, imgpoints, size, K, D, flags=None, criteria=None):
    n = len(objpoints)
    K = np.array([[300.0, 0, size[0] / 2], [0, 300.0, size[1] / 2], [0, 0, 1.0]])
    D = np.zeros((4, 1))
    rvecs = [np.zeros((3, 1)) for _ in range(n)]
    tvecs = [np.array([[0.0], [0.0], [5.0]]) for _ in range(n)]
    return 0.31, K, D, rvecs, tvecs


def test_find_corners_on_synthetic_boards(tmp_path):
    _write_boards(tmp_path)
    images = sorted(str(p) for p in tmp_path.glob("*.jpg"))
    obj, img, size, used, skipped = fr._find_corners(images, COLS, ROWS)
    assert len(used) == 6
    assert len(obj) == len(img) == 6
    assert size == (400, 300)
    assert any("z_corrupt" in s for s in skipped)


def test_main_end_to_end_with_stubbed_solver(tmp_path, monkeypatch, capsys):
    _write_boards(tmp_path)
    monkeypatch.setattr(cv2.fisheye, "calibrate", _fake_calibrate)
    out_yaml = tmp_path / "calib.yaml"
    cov_png = tmp_path / "cov.png"
    rc = fr.main(["--images", str(tmp_path / "*.jpg"),
                  "--cols", str(COLS), "--rows", str(ROWS),
                  "--out", str(out_yaml), "--coverage-png", str(cov_png)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "used 6/7 images" in out
    assert "RMS): 0.310 px (GOOD" in out
    assert cov_png.exists() and cov_png.stat().st_size > 0
    text = out_yaml.read_text()
    assert "model: fisheye" in text and "image_size: [400, 300]" in text


def test_main_solver_failure_becomes_actionable_exit(tmp_path, monkeypatch):
    _write_boards(tmp_path)

    def _boom(*a, **kw):
        raise cv2.error("ill-conditioned")

    monkeypatch.setattr(cv2.fisheye, "calibrate", _boom)
    with pytest.raises(SystemExit, match="TILTED"):
        fr.main(["--images", str(tmp_path / "*.jpg"),
                 "--cols", str(COLS), "--rows", str(ROWS)])


def test_main_no_images_match(tmp_path):
    with pytest.raises(SystemExit, match="no images match"):
        fr.main(["--images", str(tmp_path / "*.jpg")])
