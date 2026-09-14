"""Tests for scripts/eval/thermal_gt_eval.py: thermal vs fisheye-GT scorer.

The unit helpers (GT interval loading / static marking / metrics / matching)
are exercised with synthetic geometry so every branch is deterministic;
main() + collect_run run end-to-end against the synthetic triplet fixture
with a synthetic GroundingDINO-style labels JSONL (no data/ needed)."""

from __future__ import annotations

import json
import math
import sys
from unittest import mock

import numpy as np
import pytest

import scripts.eval.thermal_gt_eval as tge
from scripts.eval.thermal_gt_eval import (
    GTInterval,
    ThermalGTMetrics,
    _interval_distance,
    fisheye_bearing_from_norm_x,
    load_gt_intervals,
    mark_static_intervals,
    score_frames,
)

# Simple synthetic pinhole: fx=100, cx=50 -> bearing(px) = atan2(px - 50, 100).
K_SYNTH = np.array([[100.0, 0.0, 50.0],
                    [0.0, 100.0, 50.0],
                    [0.0, 0.0, 1.0]])
CLIP_ID = "Synth/2099-01-01_00-00-00"


def _bearing(px: float) -> float:
    return float(np.degrees(np.arctan2(px - 50.0, 100.0)))


# ---------------------------------------------------------------------------
# fisheye_bearing_from_norm_x / _interval_distance
# ---------------------------------------------------------------------------

def test_fisheye_bearing_from_norm_x_center_and_edge():
    # px = 0.5 * 100 = 50 = cx -> dead ahead.
    assert fisheye_bearing_from_norm_x(0.5, 100, K_SYNTH) == pytest.approx(0.0)
    # px = 100 -> atan2(50, 100) ~ 26.565 deg, right positive.
    assert fisheye_bearing_from_norm_x(1.0, 100, K_SYNTH) == pytest.approx(
        26.565, abs=1e-3)
    assert fisheye_bearing_from_norm_x(0.0, 100, K_SYNTH) < 0


def test_interval_distance_inside_and_outside():
    iv = GTInterval("boat", -5.0, 5.0)
    assert _interval_distance(0.0, iv) == 0.0
    assert _interval_distance(-5.0, iv) == 0.0
    assert _interval_distance(8.0, iv) == pytest.approx(3.0)
    assert _interval_distance(-12.0, iv) == pytest.approx(7.0)


def test_gtinterval_widened():
    iv = GTInterval("boat", -5.0, 5.0)
    assert iv.widened(6.0) == (-11.0, 11.0)


# ---------------------------------------------------------------------------
# load_gt_intervals
# ---------------------------------------------------------------------------

def _labels_records():
    """One frame with in-FOV / out-of-FOV / partial / degenerate-wide boxes,
    one bbox-free frame, and one record for a different clip."""
    return [
        {"frame_id": f"{CLIP_ID}/000000", "frame_ts": "00:00:00.0",
         "width": 100,
         "fisheye_bboxes": [
             # in FOV: [-5.71, +5.71] deg
             {"cls": "boat", "xyxy": [0.4, 0.1, 0.6, 0.9]},
             # entirely right of fov_hi=20: [21.8, 26.57] -> dropped
             {"cls": "buoy", "xyxy": [0.9, 0.1, 1.0, 0.9]},
             # partial: [-26.57, -5.71] -> clipped to [-20, -5.71]
             {"cls": "person", "xyxy": [0.4, 0.1, 0.0, 0.9]},  # swapped x
             # full-width: clipped to [-20, 20] = 40 deg > 35 -> degenerate
             {"cls": "boat", "xyxy": [0.0, 0.1, 1.0, 0.9]},
         ]},
        {"frame_id": f"{CLIP_ID}/000001", "frame_ts": "00:00:01.0",
         "width": 100},   # no fisheye_bboxes key
        {"frame_id": "Other/2020-01-01_00-00-00/000000",
         "frame_ts": "00:00:00.0", "width": 100, "fisheye_bboxes": []},
    ]


def _write_labels(path, records):
    # trailing blank line exercises the empty-line skip
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n\n")


def test_load_gt_intervals_filters_and_clips(tmp_path, capsys):
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, _labels_records())
    gt = load_gt_intervals(labels, CLIP_ID, K_SYNTH, -20.0, 20.0)
    assert set(gt.keys()) == {"00:00:00.0", "00:00:01.0"}
    assert gt["00:00:01.0"] == []
    ivs = gt["00:00:00.0"]
    assert [iv.cls for iv in ivs] == ["boat", "person"]
    boat, person = ivs
    assert boat.lo_deg == pytest.approx(-5.71, abs=0.01)
    assert boat.hi_deg == pytest.approx(5.71, abs=0.01)
    # partial interval clipped to the FOV edge
    assert person.lo_deg == pytest.approx(-20.0)
    assert person.hi_deg == pytest.approx(-5.71, abs=0.01)
    assert "dropped 1 degenerate" in capsys.readouterr().out


def test_load_gt_intervals_class_filter(tmp_path):
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, _labels_records())
    gt = load_gt_intervals(labels, CLIP_ID, K_SYNTH, -20.0, 20.0,
                           classes=("person",))
    assert [iv.cls for iv in gt["00:00:00.0"]] == ["person"]


# ---------------------------------------------------------------------------
# mark_static_intervals
# ---------------------------------------------------------------------------

def test_mark_static_intervals_empty_noop():
    gt: dict = {}
    mark_static_intervals(gt, -20.0, 20.0)   # early return, no crash
    assert gt == {}


def test_mark_static_intervals_persistent_vs_transient():
    gt = {f"00:00:{i:02d}.0": [GTInterval("boat", 0.0, 4.0)]
          for i in range(10)}
    gt["00:00:00.0"].append(GTInterval("person", 15.0, 16.0))
    mark_static_intervals(gt, -20.0, 20.0)
    # boat occupies its cells in 10/10 frames >= 0.15 -> static everywhere
    for ts in gt:
        assert gt[ts][0].static is True
    # person occupies its cells in 1/10 frames < 0.15 -> moving
    assert gt["00:00:00.0"][1].static is False


# ---------------------------------------------------------------------------
# ThermalGTMetrics derived properties
# ---------------------------------------------------------------------------

def test_metrics_zero_division_all_nan():
    m = ThermalGTMetrics()
    for v in (m.recall(), m.precision_strict(), m.precision_lenient(),
              m.f1(), m.bearing_mae(), m.fp_per_frame(), m.bin_precision(),
              m.bin_recall(), m.bin_f1(), m.static_recall(),
              m.moving_recall()):
        assert math.isnan(v)
    row = m.as_row()
    assert row["n_frames"] == 0 and row["per_class"] == {}


def test_metrics_f1_zero_sum_is_nan():
    # precision 0 (one FP, no TP) and recall 0 (one missed GT) -> p+r == 0
    m = ThermalGTMetrics(n_frames=1, n_gt=1, n_gt_recalled=0,
                         n_det=1, n_det_fp=1)
    assert m.precision_lenient() == 0.0 and m.recall() == 0.0
    assert math.isnan(m.f1())
    m.bin_fp, m.bin_fn = 1, 1
    assert m.bin_precision() == 0.0 and m.bin_recall() == 0.0
    assert math.isnan(m.bin_f1())


def test_metrics_populated_values():
    m = ThermalGTMetrics(
        n_frames=4, n_gt=4, n_gt_recalled=3, n_det=6, n_det_tp=3,
        n_det_radar=2, n_det_fp=1, sum_abs_bearing_err=1.5, n_bearing_err=3,
        bin_tp=2, bin_fp=1, bin_fn=2)
    m.per_class_gt["boat"] = [4, 3]
    m.static_gt[:] = [2, 2]
    m.moving_gt[:] = [2, 1]
    assert m.recall() == pytest.approx(0.75)
    assert m.precision_strict() == pytest.approx(0.5)     # 3 / 6
    assert m.precision_lenient() == pytest.approx(0.75)   # 3 / 4
    assert m.f1() == pytest.approx(0.75)
    assert m.bearing_mae() == pytest.approx(0.5)
    assert m.fp_per_frame() == pytest.approx(0.25)
    assert m.bin_precision() == pytest.approx(2 / 3)
    assert m.bin_recall() == pytest.approx(0.5)
    assert m.bin_f1() == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert m.static_recall() == 1.0 and m.moving_recall() == 0.5
    row = m.as_row()
    assert row["recall"] == 0.75
    assert row["per_class"] == {"boat": {"n": 4, "recalled": 3}}
    assert row["n_det_radar"] == 2


# ---------------------------------------------------------------------------
# score_frames
# ---------------------------------------------------------------------------

def test_score_frames_matching_and_bins():
    gt = {"00:00:00.0": [GTInterval("boat", -5.0, 5.0, static=True),
                         GTInterval("person", 20.0, 25.0)]}
    thermal = {"00:00:00.0": [0.0, 40.0, 60.0],
               "00:00:09.0": [0.0]}          # no GT record -> not scored
    radar = {"00:00:00.0": [39.0]}
    edges = np.array([-30.0, -15.0, 0.0, 15.0, 30.0])
    m = score_frames(gt, thermal, radar, tol_deg=6.0, bin_edges=edges,
                     fov=(-27.0, 29.0))
    assert m.n_frames == 1
    # boat recalled by det 0.0; person not (40 outside widened [14, 31])
    assert m.n_gt == 2 and m.n_gt_recalled == 1
    assert m.per_class_gt == {"boat": [1, 1], "person": [1, 0]}
    assert m.static_gt == [1, 1] and m.moving_gt == [1, 0]
    # det 0.0 TP (err 0); det 40 radar-corroborated (|40-39| <= 6); det 60 FP
    assert (m.n_det, m.n_det_tp, m.n_det_radar, m.n_det_fp) == (3, 1, 1, 1)
    assert m.bearing_mae() == 0.0
    # gt bins {1, 2, 3}; det bins {2} (40/60 outside edges -> None discarded)
    assert (m.bin_tp, m.bin_fp, m.bin_fn) == (1, 0, 2)


def test_score_frames_no_bin_edges():
    gt = {"00:00:00.0": [GTInterval("boat", -5.0, 5.0)]}
    m = score_frames(gt, {"00:00:00.0": []}, {}, tol_deg=6.0)
    assert m.n_gt == 1 and m.n_gt_recalled == 0
    assert (m.bin_tp, m.bin_fp, m.bin_fn) == (0, 0, 0)


def test_score_frames_fov_restricts_bins():
    # GT + detection both inside the edges but outside the thermal FOV:
    # both bin sets are clipped empty -> no bin counts at all.
    gt = {"00:00:00.0": [GTInterval("boat", 20.0, 25.0)]}
    thermal = {"00:00:00.0": [22.0]}
    edges = np.array([-30.0, -15.0, 0.0, 15.0, 30.0])
    m = score_frames(gt, thermal, {}, tol_deg=1.0, bin_edges=edges,
                     fov=(-15.0, 15.0))
    assert (m.bin_tp, m.bin_fp, m.bin_fn) == (0, 0, 0)
    assert m.n_det_tp == 1   # object grain unaffected by the bin FOV clip


def test_score_frames_temporal_dilation_precision_only():
    gt = {"00:00:00.0": [GTInterval("boat", -5.0, 5.0)],
          "00:00:01.0": []}
    thermal = {"00:00:01.0": [0.0]}
    # Without dilation the persistent detection is charged FP...
    m0 = score_frames(gt, thermal, {}, tol_deg=6.0, dilate_frames=0)
    assert m0.n_det_fp == 1 and m0.n_det_tp == 0
    # ...with +-1 frame dilation it matches the neighbour frame's GT.
    m1 = score_frames(gt, thermal, {}, tol_deg=6.0, dilate_frames=1)
    assert m1.n_det_tp == 1 and m1.n_det_fp == 0
    # Recall side stays per-frame (frame 1 has no GT).
    assert m1.n_gt == 0


# ---------------------------------------------------------------------------
# thermal_fov_deg (real intrinsics)
# ---------------------------------------------------------------------------

def test_thermal_fov_deg_linear_matches_datasheet(intrinsics_real):
    lo, hi, P, wh = tge.thermal_fov_deg(intrinsics_real, "linear")
    assert lo < 0 < hi
    assert 50 < hi - lo < 62          # ~57 deg HFOV (Lepton 3 datasheet)
    assert P.shape == (3, 3)
    assert wh[0] > 0 and wh[1] > 0


def test_thermal_fov_deg_pinhole_wider(intrinsics_real):
    lo_l, hi_l, _, _ = tge.thermal_fov_deg(intrinsics_real, "linear")
    lo_p, hi_p, _, _ = tge.thermal_fov_deg(intrinsics_real, "pinhole")
    assert (hi_p - lo_p) > (hi_l - lo_l)   # calibrated fx=63.8 -> ~103 deg


# ---------------------------------------------------------------------------
# collect_run + main (end-to-end on the synthetic triplet)
# ---------------------------------------------------------------------------

def _write_e2e_labels(path, n_frames: int):
    """Labels matching the synthetic triplet's radar timestamps, with a
    centred boat box (~[-4, +4.3] deg under the real fisheye K), one
    degenerate full-width box, and one record for another clip."""
    recs = []
    for i in range(n_frames):
        recs.append({
            "frame_id": f"{CLIP_ID}/{i:06d}", "frame_ts": f"00:00:{i:02d}.0",
            "width": 864, "height": 648,
            "fisheye_bboxes": [
                {"cls": "boat", "xyxy": [0.48, 0.55, 0.55, 0.80]}],
        })
    recs[0]["fisheye_bboxes"].append(
        {"cls": "person", "xyxy": [0.0, 0.5, 1.0, 0.9]})   # degenerate wide
    recs.append({"frame_id": "Other/2020-01-01_00-00-00/000000",
                 "frame_ts": "00:00:00.0", "width": 864,
                 "fisheye_bboxes": []})
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_collect_run_skip_frames(synthetic_triplet, intrinsics_real,
                                 detection_real):
    prefix = str(synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp)
    thermal_by_ts, radar_by_ts, stats = tge.collect_run(
        prefix, detection_real, intrinsics_real, skip_frames=2)
    assert len(thermal_by_ts) == 3          # 5 frames, first 2 skipped
    assert set(thermal_by_ts) == set(radar_by_ts) == set(stats)
    for s in stats.values():
        assert set(s) == {"std", "dyn"}


def test_main_end_to_end(tmp_path, synthetic_triplet, repo_root, capsys):
    prefix = str(synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp)
    labels = tmp_path / "labels.jsonl"
    _write_e2e_labels(labels, 5)
    json_out = tmp_path / "out" / "row.json"
    argv = ["thermal_gt_eval",
            "--triplet", prefix,
            "--labels", str(labels),
            "--detection", str(repo_root / "configs" / "detection.yaml"),
            "--skip-frames", "0",
            "--dilate", "1",
            "--json-out", str(json_out)]
    with mock.patch.object(sys, "argv", argv):
        tge.main()
    out = capsys.readouterr().out
    assert "thermal usable FOV (linear)" in out
    assert "5 labelled frames" in out
    assert f"wrote {json_out}" in out
    row = json.loads(json_out.read_text())
    assert row["n_frames"] == 5 and row["n_gt"] == 5
    assert "bin_f1" in row and "per_class" in row
