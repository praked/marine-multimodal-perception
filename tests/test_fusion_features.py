"""Phase-0 fusion-scorer feature plumbing + exporter tests.

Covers: SNR/NOISE threading (iterate_triplet -> process_mmwave), fisheye
luminance stats, range_reference provenance, the feature-table exporter
(scripts/fusion_model/build_features.py), sun/time helpers, and the
incumbent-score integrity check.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts.fusion_model.build_features import (
    DEFAULT_LAT,
    DEFAULT_LON,
    bin_column_bands,
    build_for_triplet,
    check_score_reproduction,
    frame_datetime_utc,
    seg_bin_features,
    write_tables,
)
from scripts.sensor_processing.gps_boat1 import sun_position
from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    iterate_triplet,
)
from scripts.utils.datasets import Triplet
from scripts.utils.segmentation import OBSTACLE, SKY, WATER, SegDetectParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_quad_triplet(tmp_path):
    """Synthetic triplet whose radar CSV carries V + SNR + NOISE columns
    (the post-2026-07-09 capture format), incl. a NaN SNR cell and a
    sub-y_min point that the pipeline must filter out."""
    scene_dir = tmp_path / "Synth"
    scene_dir.mkdir()
    ts = "2099-07-01_12-00-00"
    fish_path = scene_dir / f"fisheye_{ts}.mp4"
    therm_path = scene_dir / f"thermal_{ts}.mp4"
    mm_path = scene_dir / f"mmwave_{ts}.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(fish_path), fourcc, 3.0, (160, 120))
    tw = cv2.VideoWriter(str(therm_path), fourcc, 3.0, (160, 120))
    for i in range(4):
        f = np.full((120, 160, 3), 60 + i * 5, dtype=np.uint8)
        f[40:80, 60:100] = 200
        fw.write(f)
        tw.write(f)
    fw.release()
    tw.release()

    rows = ["Date,Time,X,Y,Z,V,SNR,NOISE"]
    for i in range(4):
        t = f"12:00:{i:02d}.0"
        rows.append(f"2099-07-01,{t},0.1,1.5,0.0,0.2,14.5,8.0")
        rows.append(f"2099-07-01,{t},0.5,2.0,0.0,-0.1,,7.5")     # NaN SNR
        rows.append(f"2099-07-01,{t},0.0,0.2,0.0,0.0,20.0,6.0")  # < y_min
    mm_path.write_text("\n".join(rows) + "\n")

    return Triplet(scene="Synth", timestamp=ts,
                   fisheye=fish_path, thermal=therm_path, mmwave=mm_path)


# ---------------------------------------------------------------------------
# SNR/NOISE threading
# ---------------------------------------------------------------------------

def test_iterate_triplet_yields_side_info_columns(synthetic_quad_triplet,
                                                  detection_real):
    frames = list(iterate_triplet(synthetic_quad_triplet, detection_real,
                                  clip_overrides={}))
    assert len(frames) == 4
    _, _, _, pts = frames[0]
    assert pts.shape[1] == 6          # X, Y, Z, V, SNR, NOISE


def test_iterate_triplet_without_v_stays_nx3(synthetic_triplet,
                                             detection_real):
    _, _, _, pts = next(iter(iterate_triplet(synthetic_triplet,
                                             detection_real,
                                             clip_overrides={})))
    assert pts.shape[1] == 3          # legacy CSVs unchanged


def test_process_mmwave_side_info_aligned(intrinsics_real, detection_real):
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    pts = np.array([
        [0.1, 1.5, 0.0, 0.2, 14.5, 8.0],
        [0.5, 2.0, 0.0, -0.1, np.nan, 7.5],
        [0.0, 0.2, 0.0, 0.0, 20.0, 6.0],   # below y_min -> filtered
    ])
    out = pl.process_mmwave(pts)
    assert len(out.points_xyz) == 2
    assert out.snr_db == [14.5, None]      # NaN cell -> None, filter aligned
    assert out.noise_db == [8.0, 7.5]


def test_process_mmwave_nx3_has_empty_side_info(intrinsics_real,
                                                detection_real,
                                                sample_radar_points):
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    out = pl.process_mmwave(sample_radar_points)
    assert out.snr_db == [] and out.noise_db == []


# ---------------------------------------------------------------------------
# Luminance + range_reference provenance
# ---------------------------------------------------------------------------

def test_fisheye_luminance_dark_vs_bright(intrinsics_real, detection_real):
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    dark = np.full((648, 864, 3), 5, dtype=np.uint8)
    out = pl.process_fisheye(dark)
    assert out.is_dark and out.luminance_mean < 25.0
    bright = np.full((648, 864, 3), 120, dtype=np.uint8)
    out2 = pl.process_fisheye(bright)
    assert not out2.is_dark
    assert out2.luminance_p95 >= out2.luminance_p05


def test_range_reference_recorded(intrinsics_real, detection_real,
                                  small_fisheye_frame):
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    img = cv2.resize(small_fisheye_frame, (864, 648))
    out = pl.process_fisheye(img)
    # No IMU, no seg mask -> the walk lands on horizon or level.
    assert out.range_reference in ("horizon", "level")


def test_range_reference_none_when_range_disabled(intrinsics_real,
                                                  detection_real,
                                                  small_fisheye_frame):
    det = {**detection_real,
           "range": {**detection_real.get("range", {}), "enabled": False}}
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    img = cv2.resize(small_fisheye_frame, (864, 648))
    out = pl.process_fisheye(img)
    assert out.range_reference is None


# ---------------------------------------------------------------------------
# Sun / time helpers
# ---------------------------------------------------------------------------

def test_frame_datetime_utc_summer_offset():
    utc = frame_datetime_utc("2026-07-08", "14:00:00.0", "CET")
    assert (utc.hour, utc.minute) == (12, 0)      # CEST = UTC+2


def test_frame_datetime_utc_midnight_rollover():
    utc = frame_datetime_utc("2026-07-08", "00:00:05.0", "CET",
                             chunk_start_hms="23-59-50")
    assert utc.day == 8 and utc.hour == 22        # local 07-09 00:00 CEST


def test_sun_position_institutionone_noon_is_high():
    utc = frame_datetime_utc("2026-07-08", "13:30:00.0", "CET")
    elev, az = sun_position(DEFAULT_LAT, DEFAULT_LON, utc)
    assert elev > 50.0
    assert 100.0 < az < 260.0


# ---------------------------------------------------------------------------
# Seg-derived e_fisheye evidence
# ---------------------------------------------------------------------------

def _synthetic_seg_mask():
    """80x100: sky top, water bottom, one obstacle block straddling the
    water edge on the left."""
    seg = np.full((80, 100), SKY, dtype=np.uint8)
    seg[40:, :] = WATER
    seg[30:55, 10:30] = OBSTACLE
    return seg


def test_seg_bin_features_synthetic_mask():
    seg = _synthetic_seg_mask()
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    edges = np.array([-30.0, -15.0, 0.0, 15.0, 30.0])
    feats = seg_bin_features(seg, edges, K, SegDetectParams(min_area=10))
    fracs = feats["seg_obstacle_frac"]
    assert len(fracs) == 4
    # The obstacle sits left of centre -> negative-bearing bins carry it.
    assert np.nansum(fracs[:2]) > 0.0
    assert fracs[2] == 0.0 or np.isnan(fracs[2]) or fracs[2] < fracs[1]
    assert sum(feats["seg_comp_count"]) >= 1.0


def test_seg_bin_features_no_mask_is_nan():
    K = np.eye(3)
    edges = np.array([-30.0, 0.0, 30.0])
    feats = seg_bin_features(None, edges, K, SegDetectParams())
    assert all(np.isnan(v) for v in feats["seg_obstacle_frac"])


def test_bin_column_bands_clip_to_image():
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    edges = np.array([-80.0, 0.0, 80.0])   # tan(80 deg) -> far off-image
    bands = bin_column_bands(edges, K, width=100)
    assert bands[0] == (0, 50) and bands[1] == (50, 100)


# ---------------------------------------------------------------------------
# Exporter end-to-end on the synthetic quad triplet
# ---------------------------------------------------------------------------

def test_build_for_triplet_and_integrity(synthetic_quad_triplet,
                                         intrinsics_real, detection_real,
                                         tmp_path):
    bins_df, frames_df, meta = build_for_triplet(
        synthetic_quad_triplet, intrinsics_real, detection_real,
        use_imu=False, limit=3)
    n_bins = len(meta["bin_edges_deg"]) - 1
    assert len(frames_df) == 3
    assert len(bins_df) == 3 * n_bins

    # Integrity: incumbent score recomputable (also ran inside the builder).
    check_score_reproduction(bins_df)
    broken = bins_df.copy()
    broken.loc[0, "score_legacy"] = 0.999
    with pytest.raises(AssertionError):
        check_score_reproduction(broken)

    # Context columns present + sane.
    fr = frames_df.iloc[0]
    assert np.isfinite(fr["sun_elevation_deg"])
    assert fr["radar_has_snr"] and fr["radar_has_doppler"]
    assert np.isfinite(fr["luminance_mean"])
    assert fr["split"] in ("train", "val", "test")
    # Reserved YOLO columns exist and are inert.
    assert not bins_df["yolo_available"].any()
    assert bins_df["yolo_max_conf"].isna().all()
    # No seg masks for a synthetic clip -> NaN evidence + flag False.
    assert not bins_df["seg_available"].any()
    assert bins_df["seg_obstacle_frac"].isna().all()

    out_dir = write_tables(bins_df, frames_df, meta, tmp_path, fmt="csv")
    assert (out_dir / "bins.csv").exists()
    assert (out_dir / "frames.csv").exists()
    saved = json.loads((out_dir / "meta.json").read_text())
    assert saved["schema_version"] == 4   # v4: per-frame exposure context
    # Reserved target columns exist and are inert with the tracker off.
    assert not bins_df["target_present"].any()
    assert bins_df["target_cpa_m"].isna().all()
    assert saved["targets_enabled"] is False
    reread = pd.read_csv(out_dir / "bins.csv")
    assert len(reread) == len(bins_df)


# ---------------------------------------------------------------------------
# metrics.py default label paths
# ---------------------------------------------------------------------------

def test_metrics_default_labels_glob_qwen_dir(repo_root):
    from scripts.eval.metrics import DEFAULT_LABELS
    qwen_dir = repo_root / "labels" / "qwen"
    if not qwen_dir.is_dir() or not list(qwen_dir.glob("*.jsonl")):
        pytest.skip("labels/qwen/*.jsonl not present")
    globbed = [p for p in DEFAULT_LABELS if p.parent.name == "qwen"]
    assert globbed, "labels/qwen/*.jsonl missing from DEFAULT_LABELS"
    assert all(p.suffix == ".jsonl" for p in globbed)


# --- Radar mount yaw ---------------------------------------------------------
#
# Measured -6.0 deg on 2026-08-19. Applied by rotating the point cloud, so the
# y-window, self-clutter zone, tracker and projection all see one corrected
# frame rather than a bearing patched after the fact.

def test_mount_yaw_rotates_bearings_and_is_off_by_default():
    import numpy as np
    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
    from scripts.utils.calibration import load_detection, load_intrinsics

    det = load_detection()
    pts = np.array([[0.0, 3.0, 0.0]])          # dead ahead: bearing 0

    det_off = {**det, "mmwave": {**det["mmwave"], "mount_yaw_deg": 0.0}}
    intr = load_intrinsics()
    p_off = ObstacleDetectionPipeline(intr, det_off)
    assert abs(p_off.process_mmwave(pts).angles[0]) < 1e-6

    # A -6 deg offset is corrected by rotating +6, so a point the radar reports
    # dead ahead is really at +6 deg.
    det_on = {**det, "mmwave": {**det["mmwave"], "mount_yaw_deg": -6.0}}
    p_on = ObstacleDetectionPipeline(intr, det_on)
    assert abs(p_on.process_mmwave(pts).angles[0] - 6.0) < 0.05


def test_mount_yaw_preserves_radial_distance():
    """A rotation about the vertical cannot change how far anything is.

    NB `MMWaveResult.ranges` is the Y (FORWARD) component, not radial distance,
    so it legitimately shifts for off-axis points under the correction: at 3 m
    and 18 deg off-axis a 6 deg rotation moves Y by ~0.1 m. What must be
    invariant is the radial distance, so that is what this asserts.
    """
    import numpy as np
    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
    from scripts.utils.calibration import load_detection, load_intrinsics

    det = load_detection(); intr = load_intrinsics()
    pts = np.array([[1.0, 3.0, 0.2]])
    r_in = float(np.hypot(pts[0, 0], pts[0, 1]))
    res = ObstacleDetectionPipeline(
        intr, {**det, "mmwave": {**det["mmwave"], "mount_yaw_deg": -6.0}}
    ).process_mmwave(pts)
    x, y = res.points_xyz[0, 0], res.points_xyz[0, 1]
    assert abs(float(np.hypot(x, y)) - r_in) < 1e-6
    # ...and the bearing moved by exactly the correction.
    before = np.degrees(np.arctan2(1.0, 3.0))
    assert abs(res.angles[0] - (before + 6.0)) < 0.05


def test_seg_darkness_gate(intrinsics_real, detection_real):
    """segmentation.darkness_gate (2026-09-12): a supplied water mask is
    dropped on a frame below fisheye.darkness_thresh, exactly as the live
    SegWorker refuses such frames; kept on a lit frame; kept on a dark frame
    when the gate is off (pre-2026-09-12 behaviour)."""
    import copy
    det = copy.deepcopy(detection_real)
    det.setdefault("segmentation", {})["enabled"] = True
    det["segmentation"]["darkness_gate"] = True
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    h, w = 648, 864
    mask = np.full((h, w), 1, dtype=np.uint8)        # water everywhere
    mask[: h // 2] = 2                                # sky above
    dark = np.full((h, w, 3), 5, dtype=np.uint8)
    bright = np.full((h, w, 3), 120, dtype=np.uint8)
    assert pl.process_fisheye(dark, seg_mask=mask).seg_mask is None
    assert pl.process_fisheye(bright, seg_mask=mask).seg_mask is not None
    det["segmentation"]["darkness_gate"] = False
    pl_off = ObstacleDetectionPipeline(intrinsics_real, det)
    assert pl_off.process_fisheye(dark, seg_mask=mask).seg_mask is not None
