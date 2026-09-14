import sys
from collections import deque
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.eval.radar_video import _draw_frame, _fig_to_bgr, main
from scripts.sensor_processing.pipeline import make_bins


def test_fig_to_bgr_returns_uint8_3channel():
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    ax.plot([0, 1], [0, 1])
    out = _fig_to_bgr(fig)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    plt.close(fig)


def test_draw_frame_with_points():
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    edges, _ = make_bins({"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10})
    pts = np.array([[0.0, 1.5], [0.5, 2.0]])
    trail = deque(maxlen=5)
    trail.append(np.array([[0.0, 1.0]]))
    _draw_frame(ax, pts, trail, edges, y_max=9.0)
    assert ax.get_xlim() == (-5, 5)
    assert ax.get_ylim() == (0, 9.0)
    plt.close(fig)


def test_draw_frame_no_current_points():
    fig, ax = plt.subplots()
    edges, _ = make_bins({"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10})
    _draw_frame(ax, np.empty((0, 2)), deque(maxlen=5), edges, y_max=9.0)
    plt.close(fig)


def test_draw_frame_skips_empty_trail_entries():
    fig, ax = plt.subplots()
    edges, _ = make_bins({"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10})
    trail = deque(maxlen=3)
    trail.append(np.empty((0, 2)))
    trail.append(np.array([[1.0, 2.0]]))
    _draw_frame(ax, np.array([[0.0, 1.0]]), trail, edges, y_max=9.0)
    plt.close(fig)


def test_fig_to_bgr_older_matplotlib_fallback():
    """Force the buffer_rgba path to raise AttributeError so the legacy
    tostring_rgb branch runs (lines 66-70)."""
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
    ax.plot([0, 1], [0, 1])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgb_bytes = np.zeros((h * w * 3,), dtype=np.uint8).tobytes()

    class FakeCanvas:
        def draw(self):
            pass

        def buffer_rgba(self):
            raise AttributeError("no buffer_rgba on old matplotlib")

        def tostring_rgb(self):
            return rgb_bytes

        def get_width_height(self):
            return (w, h)

    fig.canvas = FakeCanvas()
    out = _fig_to_bgr(fig)
    assert out.dtype == np.uint8
    assert out.shape == (h, w, 3)
    plt.close(fig)


def test_main_writes_mp4(tmp_path, boats_triplet):
    out = tmp_path / "out.mp4"
    with mock.patch.object(sys, "argv", [
        "rv", "--triplet",
        str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--out", str(out), "--fps", "12", "--trail", "5",
    ]):
        main()
    assert out.exists()
    assert out.stat().st_size > 1000


def test_main_with_filter_flag(tmp_path, boats_triplet):
    out = tmp_path / "out.mp4"
    with mock.patch.object(sys, "argv", [
        "rv", "--triplet",
        str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--out", str(out), "--filter",
    ]):
        main()
    assert out.exists()
