"""Tests for scripts/eval/range_vs_radar.py: camera-vs-radar range validation.

The unit helpers (radar_bearings_ranges / associate / summarise) are tested
directly; main() runs against the synthetic triplet with the pipeline swapped
for a deterministic stub (detector behaviour is covered by test_pipeline)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import scripts.eval.range_vs_radar as rvr


def test_radar_bearings_ranges_filters_window():
    pts = np.array([[0.0, 2.0, 0.0],      # bearing 0, range 2 -> kept
                    [1.0, 0.2, 0.0],      # below y_min -> dropped
                    [0.0, 12.0, 0.0]])    # beyond y_max -> dropped
    b, r = rvr.radar_bearings_ranges(pts, y_min=0.5, y_max=9.0)
    assert len(b) == 1 and abs(b[0]) < 1e-9 and r[0] == 2.0


def test_radar_bearings_ranges_empty():
    b, r = rvr.radar_bearings_ranges(np.empty((0, 3)), 0.5, 9.0)
    assert len(b) == 0 and len(r) == 0


def test_associate_picks_closest_in_window():
    bearings = np.array([0.0, 3.0, 40.0])
    ranges = np.array([5.0, 2.0, 1.0])
    rng, n = rvr.associate(1.0, bearings, ranges, tol_deg=5.0)
    assert rng == 2.0 and n == 2
    assert rvr.associate(1.0, bearings, ranges, tol_deg=0.5) == (None, 0)
    assert rvr.associate(0.0, np.empty(0), np.empty(0), 5.0) == (None, 0)


def test_summarise_groups():
    rows = [{"clip": "a", "rel_err": 0.1, "within_20pct": True},
            {"clip": "a", "rel_err": -0.5, "within_20pct": False},
            {"clip": "b", "rel_err": 0.0, "within_20pct": True}]
    s = rvr.summarise(rows, key=lambda r: r["clip"])
    assert s["a"]["n"] == 2 and s["b"]["within_20pct"] == 1.0
    assert abs(s["a"]["median_abs_rel_err"] - 0.3) < 1e-9


class _StubPipeline:
    """Deterministic stand-in: one ranged fisheye detection dead ahead (close
    to the synthetic triplet's radar return at ~3.8 deg / 1.5 m), one unranged,
    one out of the association window; thermal far off the radar range."""

    def __init__(self, intrinsics, detection, attitude_provider=None):
        pass

    def process_fisheye(self, frame, frame_id=None):
        return SimpleNamespace(angles=[3.0, 3.0, 30.0], ranges=[1.6, None, 2.0])

    def process_thermal(self, frame):
        return SimpleNamespace(angles=[3.0], ranges=[8.0])


def test_main_writes_csv_and_summary(tmp_path, monkeypatch, synthetic_triplet, capsys):
    monkeypatch.setattr(rvr, "ObstacleDetectionPipeline", _StubPipeline)
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    out = tmp_path / "rvr"
    rc = rvr.main(["--triplet", str(prefix), "--out", str(out),
                   "--max-frames", "3"])
    assert rc == 0
    csv_text = (out / "range_vs_radar.csv").read_text()
    assert "fisheye" in csv_text and "thermal" in csv_text
    printed = capsys.readouterr().out
    assert "per clip" in printed and "overall" in printed
    # fisheye 1.6 m vs radar 1.5 m -> within 20%; thermal 8 m -> not
    assert "within_20pct" in csv_text


def test_main_no_associations_returns_1(tmp_path, monkeypatch, synthetic_triplet, capsys):
    monkeypatch.setattr(rvr, "ObstacleDetectionPipeline", _StubPipeline)
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    # radar window that excludes the synthetic returns entirely
    rc = rvr.main(["--triplet", str(prefix), "--out", str(tmp_path / "o"),
                   "--y-min", "5.0", "--y-max", "6.0"])
    assert rc == 1
    assert "No camera-radar associations" in capsys.readouterr().out


def test_main_scene_lookup(tmp_path, monkeypatch, synthetic_triplet, capsys):
    import scripts.utils.datasets as ds
    monkeypatch.setattr(rvr, "ObstacleDetectionPipeline", _StubPipeline)
    # point the datasets module at the synthetic tree so --scene finds "Synth"
    monkeypatch.setattr(ds, "DATA_DIR", synthetic_triplet.fisheye.parent.parent)
    rc = rvr.main(["--scene", "Synth", "--out", str(tmp_path / "s"),
                   "--sensors", "fisheye"])
    assert rc == 0
    assert "Synth/" in capsys.readouterr().out


def test_main_requires_clips():
    with pytest.raises(SystemExit):
        rvr.main(["--out", "x"])
