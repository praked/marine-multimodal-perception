"""Tests for scripts/eval/snapshot.py (Agg backend, no GUI)."""

import sys
from unittest import mock

import numpy as np
import pytest

import scripts.eval.snapshot as snap


def test_overlay_camera_none_returns_placeholder():
    out = snap._overlay_camera(None, None, None)
    assert out.shape == (10, 10, 3)


def test_overlay_camera_draws(monkeypatch):
    """A minimal fake sensor result exercises the horizon line, radar dot,
    and detection-circle branches."""
    class FakeSensor:
        undistorted = np.full((120, 160, 3), 40, dtype=np.uint8)
        horizon_line = (0.0, 60.0, 0.9)   # slope, intercept, confidence
        coords = [(80, 70)]

    projected = np.array([[80.0, 60.0], [np.nan, 5.0], [10000.0, 10.0]])
    ranges = np.array([2.0, 3.0, 4.0])
    out = snap._overlay_camera(FakeSensor(), projected, ranges)
    assert out.shape == (120, 160, 3)
    assert not np.array_equal(out, FakeSensor.undistorted)


def test_main_renders_png(tmp_path, synthetic_triplet, monkeypatch):
    out_png = tmp_path / "snap.png"
    argv = ["snap", "--triplet", str(synthetic_triplet.fisheye.parent /
                                     synthetic_triplet.timestamp),
            "--frame", "0", "--out", str(out_png)]
    with mock.patch.object(sys, "argv", argv):
        snap.main()
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_main_frame_out_of_range_exits(tmp_path, synthetic_triplet):
    out_png = tmp_path / "snap.png"
    argv = ["snap", "--triplet", str(synthetic_triplet.fisheye.parent /
                                     synthetic_triplet.timestamp),
            "--frame", "9999", "--out", str(out_png)]
    with mock.patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit):
            snap.main()
