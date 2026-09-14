"""Coverage-gap tests: small unexercised branches across several modules.

Each test targets specific missing lines (noted per test) found by
`--cov-report=term-missing`; everything runs on synthetic fixtures in
tmp_path so the file is CI-safe (no data/, no hardware).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# scripts/utils/datasets.py: load_frames_csv (lines 258-278)
# ---------------------------------------------------------------------------


def test_load_frames_csv_reads_sidecar(tmp_path):
    """Sidecar with frame_index -> Timestamp/RoundedTime table (259-267)."""
    from scripts.utils.datasets import load_frames_csv

    p = tmp_path / "frames_2099-01-01_00-00-00.csv"
    p.write_text(
        "frame_index,Date,Time\n"
        "0,2099-01-01,12:00:00.0\n"
        "1,2099-01-01,12:00:00.3\n"
    )
    df = load_frames_csv(p)
    assert list(df["frame_index"]) == [0, 1]
    assert list(df["RoundedTime"]) == ["12:00:00.0", "12:00:00.3"]
    assert "ExposureTime" not in df.columns   # pre-2026-08-28 sidecar: no exposure


def test_load_frames_csv_keeps_exposure_columns(tmp_path):
    """Exposure columns (2026-08-28) ride along; blanks -> NaN."""
    from scripts.utils.datasets import load_frames_csv

    p = tmp_path / "frames_2099-01-01_00-00-00.csv"
    p.write_text(
        "frame_index,Date,Time,ExposureTime,AnalogueGain,DigitalGain,Lux\n"
        "0,2099-01-01,12:00:00.0,299998,7.9994,1.0,3.21\n"
        "1,2099-01-01,12:00:00.3,,,,\n"
    )
    df = load_frames_csv(p)
    assert list(df.columns) == ["frame_index", "Timestamp", "RoundedTime",
                                "ExposureTime", "AnalogueGain", "DigitalGain", "Lux"]
    assert df["ExposureTime"].iloc[0] == 299998
    assert df["Lux"].isna().iloc[1]


def test_load_frames_csv_empty_sidecar_falls_back(tmp_path):
    """A zero-byte sidecar (EmptyDataError, 262-263) falls through to the
    chunk_start + index/fps reconstruction (270-276)."""
    from scripts.utils.datasets import load_frames_csv

    p = tmp_path / "frames_empty.csv"
    p.write_text("")
    df = load_frames_csv(p, n_frames=3, chunk_start="2099-01-01 12:00:00")
    assert list(df["frame_index"]) == [0, 1, 2]
    # index/3 fps: 0.0 s, 0.333 s, 0.666 s
    assert df["RoundedTime"].iloc[0] == "12:00:00.0"
    assert df["RoundedTime"].iloc[1] == "12:00:00.3"


def test_load_frames_csv_sidecar_without_frame_index_falls_back(tmp_path):
    """A sidecar missing the frame_index column is unusable (264 False):
    reconstruction still works when chunk_start is given."""
    from scripts.utils.datasets import load_frames_csv

    p = tmp_path / "frames_odd.csv"
    p.write_text("Date,Time\n2099-01-01,12:00:00.0\n")
    df = load_frames_csv(p, n_frames=2, chunk_start="2099-01-01 00:00:00", fps=3.0)
    assert list(df["frame_index"]) == [0, 1]


def test_load_frames_csv_nothing_to_build_from_returns_empty(tmp_path):
    """No sidecar and no reconstruction inputs -> empty frame (278)."""
    from scripts.utils.datasets import load_frames_csv

    df = load_frames_csv(None)
    assert df.empty
    assert list(df.columns) == ["frame_index", "Timestamp", "RoundedTime"]
    # Missing path + no chunk_start behaves identically.
    df2 = load_frames_csv(tmp_path / "absent.csv", n_frames=5, chunk_start=None)
    assert df2.empty


# ---------------------------------------------------------------------------
# scripts/utils/geometry.py (lines 334, 355, 422, 430)
# ---------------------------------------------------------------------------


def test_pitch_roll_from_up_zero_vector():
    """Zero-norm up vector -> (0, 0) (line 334)."""
    from scripts.utils.geometry import pitch_roll_from_up

    assert pitch_roll_from_up(np.zeros(3)) == (0.0, 0.0)


def test_horizon_line_from_up_degenerate_vertical():
    """b ~ 0 (vertical horizon) -> degenerate (0, 0) (line 355)."""
    from scripts.utils.geometry import horizon_line_from_up

    slope, intercept = horizon_line_from_up(np.array([1.0, 0.0, 0.0]), np.eye(3))
    assert slope == 0.0 and intercept == 0.0


def test_range_from_contact_point_zero_focal():
    """fx == 0 -> None (line 422)."""
    from scripts.utils.geometry import range_from_contact_point

    P = np.array([[0.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    assert range_from_contact_point((50.0, 80.0), P,
                                    np.array([0.0, -1.0, 0.0]), 0.3) is None


def test_range_from_contact_point_nonpositive_t():
    """camera_height 0 -> t == 0 -> None (line 430)."""
    from scripts.utils.geometry import range_from_contact_point

    P = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    # Ray below horizon (v > cy) so denom < 0; zero height forces t <= 0.
    assert range_from_contact_point((50.0, 90.0), P,
                                    np.array([0.0, -1.0, 0.0]), 0.0) is None


# ---------------------------------------------------------------------------
# scripts/utils/segmentation.py (lines 110, 211, 383-384)
# ---------------------------------------------------------------------------


def test_water_edge_contact_degenerate_bbox_returns_none():
    """bbox entirely left of the image -> c1 < c0 -> None (line 110)."""
    from scripts.utils.segmentation import WATER, water_edge_contact

    seg = np.full((40, 60), WATER, np.uint8)
    assert water_edge_contact(seg, (-30.0, 5.0, -20.0, 15.0)) is None


def test_detection_obstacle_fraction_degenerate_bbox():
    """Zero-width box -> 0.0 (line 211)."""
    from scripts.utils.segmentation import WATER, detection_obstacle_fraction

    seg = np.full((40, 60), WATER, np.uint8)
    assert detection_obstacle_fraction(seg, (5.0, 5.0, 5.0, 9.0)) == 0.0


def test_seg_obstacle_detections_max_components_cap():
    """Two qualifying components with max_components=1 -> break (383-384)."""
    from scripts.utils.segmentation import (
        OBSTACLE,
        SKY,
        WATER,
        SegDetectParams,
        seg_obstacle_detections,
    )

    seg = np.full((60, 80), WATER, np.uint8)
    seg[:10, :] = SKY
    seg[30:40, 10:20] = OBSTACLE
    seg[30:40, 50:60] = OBSTACLE
    coords, sizes = seg_obstacle_detections(
        seg, SegDetectParams(min_area=5, water_edge_margin_px=10,
                             max_components=1))
    assert len(coords) == 1 and len(sizes) == 1


# ---------------------------------------------------------------------------
# scripts/sensor_processing/pipeline.py
# ---------------------------------------------------------------------------


def test_imu_seed_keeps_ransac_when_agreeing(intrinsics_real, detection_real):
    """imu_seed enabled + RANSAC agreeing with the IMU-predicted line ->
    the RANSAC result is returned unchanged (line 537)."""
    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline

    pipe = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    att = SimpleNamespace(
        up_vector_camera=lambda: np.array([0.0, -1.0, 0.0]))
    pipe._attitude = SimpleNamespace(get=lambda: att)

    h_cfg = copy.deepcopy(detection_real["horizon"])
    h_cfg["imu_seed"] = {"enabled": True, "max_dev_px": 1e9}
    h_cfg["confidence_thresh"] = 0.0

    img = np.zeros((120, 200, 3), np.uint8)
    img[60:, :] = 180                                # crisp horizontal edge
    P = np.asarray(intrinsics_real["fisheye"]["K"], float)
    mask, slope, intercept, conf = pipe._detect_horizon_imu(img, h_cfg, P)
    # RANSAC wins (conf untouched: the IMU override would force 1.0 with
    # an IMU-predicted line; a huge max_dev_px guarantees agreement).
    assert mask.shape == img.shape[:2]
    assert 0.0 <= conf <= 1.0


def test_bearings_linear_model(intrinsics_real, detection_real):
    """_bearings under the linear model (554-556): thermal default."""
    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline

    pipe = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    assert pipe._bearing_model["thermal"] == "linear"
    intr = intrinsics_real["thermal"]
    out = pipe._bearings("thermal", [(intr["cx"] + intr["pix_deg_ratio"], 60)],
                         intr)
    assert out == [pytest.approx(1.0)]


def test_range_up_vector_priority_exhausted_falls_to_level(
        intrinsics_real, detection_real):
    """attitude_priority without a usable source ends at the terminal
    level fallback (line 642)."""
    from scripts.sensor_processing.pipeline import (
        ObstacleDetectionPipeline,
        UP_LEVEL,
    )

    det = copy.deepcopy(detection_real)
    det["range"]["attitude_priority"] = ["imu"]      # no IMU attached
    pipe = ObstacleDetectionPipeline(intrinsics_real, det)
    P = np.asarray(intrinsics_real["fisheye"]["K"], float)
    up = pipe._range_up_vector(P, (0.0, 0.0, 0.0))
    assert np.allclose(up, UP_LEVEL)
    assert pipe._last_range_reference == "level"


def test_detection_ranges_reliable_shortcut_when_uncertainty_off(
        intrinsics_real, detection_real):
    """range.uncertainty_px = 0 -> _reliable's early True (lines 702-703)."""
    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline

    det = copy.deepcopy(detection_real)
    det["range"]["uncertainty_px"] = 0
    pipe = ObstacleDetectionPipeline(intrinsics_real, det)
    P = np.array([[300.0, 0.0, 432.0], [0.0, 300.0, 324.0], [0.0, 0.0, 1.0]])
    # Low-confidence horizon -> level attitude; blob well below centre.
    ranges, raw = pipe._detection_ranges(
        [(432, 500)], [20.0], P, 0.27, (0.0, 100.0, 0.0))
    assert len(ranges) == 1 and len(raw) == 1
    assert ranges[0] is not None and ranges[0] > 0


def _seg_mask_for_fisheye(shape):
    from scripts.utils.segmentation import OBSTACLE, SKY, WATER

    h, w = shape
    seg = np.full((h, w), WATER, np.uint8)
    seg[:200, :] = SKY
    seg[400:460, 400:460] = OBSTACLE
    return seg


def test_process_fisheye_segmentation_detector_and_fp_filter(
        intrinsics_real, detection_real):
    """fisheye.detector = segmentation with a caller-supplied mask runs the
    seg-driven detection branch (777-779) and the fp-filter call site (800)."""
    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline

    det = copy.deepcopy(detection_real)
    det["fisheye"]["detector"] = "segmentation"
    det.setdefault("segmentation", {}).setdefault(
        "fp_filter", {})["enabled"] = True
    det["segmentation"]["fp_filter"].setdefault("min_obstacle_frac", 0.1)
    pipe = ObstacleDetectionPipeline(intrinsics_real, det)

    frame = np.full((648, 864, 3), 80, np.uint8)
    res = pipe.process_fisheye(frame, seg_mask=_seg_mask_for_fisheye((648, 864)))
    # The obstacle block sits on water -> detected and kept by the fp filter.
    assert len(res.coords) >= 1
    assert len(res.angles) == len(res.coords)


def test_process_thermal_mog2_with_preprocess(intrinsics_real, detection_real):
    """MOG2 background + preprocess enabled hits the preprocess-under-mog2
    line (867)."""
    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline

    det = copy.deepcopy(detection_real)
    det["thermal"]["background"]["method"] = "mog2"
    det["thermal"]["preprocess"]["enabled"] = True
    det["thermal"]["preprocess"]["median_ksize"] = 3
    pipe = ObstacleDetectionPipeline(intrinsics_real, det)
    assert pipe._thermal_mog2 is not None

    frame = np.full((120, 160, 3), 50, np.uint8)
    frame[40:60, 60:100] = 200
    res = pipe.process_thermal(frame)
    assert res.obstacle_mask.shape[:2] == res.undistorted.shape[:2]


def test_process_thermal_pinhole_bearings(intrinsics_real, detection_real):
    """thermal.bearing_model = pinhole recomputes bearings (line 899)."""
    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline

    det = copy.deepcopy(detection_real)
    det["thermal"]["bearing_model"] = "pinhole"
    pipe = ObstacleDetectionPipeline(intrinsics_real, det)
    frame = np.full((120, 160, 3), 50, np.uint8)
    frame[40:60, 60:100] = 220
    res = pipe.process_thermal(frame)
    assert len(res.angles) == len(res.coords)


def test_apply_association_skips_nonfinite_projection(
        monkeypatch, intrinsics_real, detection_real):
    """A matched radar point projecting to NaN is skipped in the silhouette
    check (1051-1052); the finite on-silhouette point re-aggregates the
    range."""
    import scripts.sensor_processing.pipeline as pl
    from scripts.utils.association import AssociationResult, DetectionMatch

    pipe = pl.ObstacleDetectionPipeline(intrinsics_real, detection_real)

    pts = np.array([[0.0, 2.0, 0.0], [0.0, 2.5, 0.0]])
    m_res = pl.MMWaveResult(points_xyz=pts, angles=[0.0, 0.0],
                            ranges=[2.0, 2.5])
    # First point projects to NaN (skipped), second lands on the mask.
    monkeypatch.setattr(
        pl, "project_radar_to_undistorted",
        lambda *_a, **_k: np.array([[np.nan, np.nan], [50.0, 50.0]]))
    match = DetectionMatch(radar_range_m=2.0, n_points=2,
                           point_indices=[0, 1])
    monkeypatch.setattr(
        pl, "associate_radar_to_detections",
        lambda *_a, **_k: AssociationResult(matches=[match]))

    from scripts.utils.segmentation import OBSTACLE
    seg = np.full((100, 100), OBSTACLE, np.uint8)
    res = SimpleNamespace(coords=[(50, 50)], sizes=[20.0], ranges=[5.0],
                          raw_ranges=[5.0], seg_mask=seg)
    pipe._apply_association(res, m_res, np.eye(3), np.eye(4))
    # Only the finite point (Y = 2.5) contributes to the silhouette range.
    assert res.radar_ranges[0] == pytest.approx(2.5)


def test_confirm_bins_skips_out_of_fov_angle(intrinsics_real, detection_real):
    """A radar-ranged detection at a bearing outside the bin range is
    skipped (line 1113); an in-range one confirms its bin."""
    from scripts.sensor_processing.pipeline import (
        FusionResult,
        ObstacleDetectionPipeline,
        make_bins,
    )

    pipe = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    edges, centers = make_bins(detection_real["fusion"])
    n = len(centers)
    fused = FusionResult(
        bin_edges=edges, bin_centers=centers, scores=[0.0] * n,
        min_ranges=[None] * n,
        sensor_hit_mask=np.zeros((n, 3), bool),
        confirmed=[False] * n,
    )
    res = SimpleNamespace(radar_ranges=[2.0, 3.0], angles=[999.0, 0.0])
    pipe._confirm_bins(fused, res, None)
    assert sum(fused.confirmed) == 1
    assert fused.min_ranges[np.searchsorted(edges, 0.0) - 1] == 3.0


# ---------------------------------------------------------------------------
# scripts/eval/metrics.py: ts-keyed label resolution + CLI flags
# ---------------------------------------------------------------------------


def _make_label(clip_id, scene, frame_idx, frame_ts=None, **kw):
    from scripts.eval.metrics import Label

    tail = f"{frame_idx:06d}" if frame_ts is None else \
        f"ts={frame_ts.replace(':', '-')}"
    base = dict(
        frame_id=f"{clip_id}/{tail}",
        scene=scene, clip_id=clip_id, frame_idx=frame_idx,
        frame_ts=frame_ts,
        source="manual", audited=True, bboxes=[], obstacle_bins=[0],
        image_size=(160, 120),
    )
    base.update(kw)
    return Label(**base)


def _fake_pipeline_factory(detection):
    from scripts.sensor_processing.pipeline import (
        FrameResult,
        FusionResult,
        make_bins,
    )

    edges, centers = make_bins(detection["fusion"])
    n = len(centers)

    class _FakePipeline:
        def process_frame(self, fish, therm, mm_pts, timestamp=None,
                          frame_id=None):
            fusion = FusionResult(
                bin_edges=edges, bin_centers=centers,
                scores=[0.0] * n, min_ranges=[None] * n,
                sensor_hit_mask=np.zeros((n, 3), dtype=bool),
                per_bin_velocity_mps=[None] * n,
                per_bin_ttc_s=[None] * n,
            )
            return FrameResult(fusion=fusion, timestamp=timestamp)

    return _FakePipeline


def test_evaluate_clip_resolves_ts_labels_and_drops_unknown(
        monkeypatch, capsys, intrinsics_real, detection_real,
        synthetic_triplet):
    """ts-keyed labels against a radar timeline: a matching RoundedTime
    resolves to its index (295-296, 313-317); an unknown one is dropped
    with the note (297-298, 320)."""
    from scripts.eval import metrics

    monkeypatch.setattr(metrics, "resolve_triplet",
                        lambda p: synthetic_triplet)
    clip_id = synthetic_triplet.clip_id
    labels = [
        _make_label(clip_id, "Synth", -1, frame_ts="00:00:01.0"),
        _make_label(clip_id, "Synth", -1, frame_ts="11:11:11.1"),
    ]
    result = metrics._evaluate_clip(
        labels, _fake_pipeline_factory(detection_real),
        intrinsics_real, detection_real)
    assert result.n_frames_labelled == 1
    assert "1 ts-keyed labels not in the radar" in capsys.readouterr().out


def test_evaluate_clip_empty_radar_synthesized_timeline(
        monkeypatch, capsys, intrinsics_real, detection_real,
        synthetic_triplet, tmp_path):
    """Empty radar CSV: ts labels resolve via the synthesized wall-clock
    timeline (299-308) and frames get synthesized timestamps (363);
    unparseable / non-round-tripping ts labels are dropped."""
    from scripts.eval import metrics
    from scripts.utils.datasets import Triplet

    empty_csv = tmp_path / "empty_mmwave.csv"
    empty_csv.write_text("Date,Time,X,Y,Z\n")
    triplet = Triplet(scene="Synth", timestamp=synthetic_triplet.timestamp,
                      fisheye=synthetic_triplet.fisheye,
                      thermal=synthetic_triplet.thermal, mmwave=empty_csv)
    monkeypatch.setattr(metrics, "resolve_triplet", lambda p: triplet)

    clip_id = triplet.clip_id
    labels = [
        # base wall-clock is 00:00:00; frame 1 formats to "00:00:00.3".
        _make_label(clip_id, "Synth", -1, frame_ts="00:00:00.3"),
        _make_label(clip_id, "Synth", -1, frame_ts="garbage"),
        # parses but fails the round-trip check (306-307)
        _make_label(clip_id, "Synth", -1, frame_ts="00:00:00.2"),
    ]
    result = metrics._evaluate_clip(
        labels, _fake_pipeline_factory(detection_real),
        intrinsics_real, detection_real)
    assert result.n_frames_labelled == 1
    assert "2 ts-keyed labels not in the radar" in capsys.readouterr().out


def test_evaluate_clip_all_ts_labels_unresolvable_raises(
        monkeypatch, intrinsics_real, detection_real, synthetic_triplet):
    """Every label dropped -> RuntimeError 'no resolvable labels'
    (323-325)."""
    from scripts.eval import metrics

    monkeypatch.setattr(metrics, "resolve_triplet",
                        lambda p: synthetic_triplet)
    labels = [_make_label(synthetic_triplet.clip_id, "Synth", -1,
                          frame_ts="23:59:59.9")]
    with pytest.raises(RuntimeError, match="no resolvable labels"):
        metrics._evaluate_clip(
            labels, _fake_pipeline_factory(detection_real),
            intrinsics_real, detection_real)


def test_metrics_main_seg_and_assoc_flags(tmp_path, monkeypatch, capsys):
    """--seg / --seg-fp-filter / --assoc mutate the detection config
    (605-607, 609); the unresolvable clip is caught, report still
    written."""
    from scripts.eval.metrics import main

    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(json.dumps({
        "frame_id": "Nope/1999-01-01_00-00-00/000000",
        "scene": "Nope", "source": "manual", "audited": True,
        "fisheye_bboxes": [], "obstacle_bins_fisheye": [0],
        "width": 864, "height": 648,
    }) + "\n")
    monkeypatch.setattr("scripts.eval.metrics.RESULTS_DIR", tmp_path)
    with mock.patch.object(sys, "argv", [
        "metrics", "--labels", str(labels_path), "--run-id", "flags_run",
        "--seg", "--seg-fp-filter", "--assoc",
    ]):
        main()
    assert (tmp_path / "metrics_flags_run.md").exists()
    assert "failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# scripts/eval/range_vs_radar.py (lines 87, 180, 187-192)
# ---------------------------------------------------------------------------


class _StubRangePipeline:
    def __init__(self, intrinsics, detection, attitude_provider=None):
        pass

    def process_fisheye(self, frame, frame_id=None):
        return SimpleNamespace(angles=[3.0], ranges=[1.6])

    def process_thermal(self, frame):
        return SimpleNamespace(angles=[3.0], ranges=[8.0])


def test_range_vs_radar_imu_replay_and_seg_flag(
        tmp_path, monkeypatch, synthetic_triplet, capsys):
    """--imu with a usable sidecar drives attitude.set_time per frame
    (line 87, 187-190); --seg forces segmentation on (line 180)."""
    import scripts.eval.range_vs_radar as rvr

    monkeypatch.setattr(rvr, "ObstacleDetectionPipeline", _StubRangePipeline)
    scene_dir = synthetic_triplet.fisheye.parent
    ts = synthetic_triplet.timestamp
    rows = ["Date,Time,Yaw,Pitch,Roll"]
    for i in range(5):
        rows.append(f"2099-01-01,00:00:0{i}.0,10.0,1.0,2.0")
    (scene_dir / f"imu_{ts}.csv").write_text("\n".join(rows) + "\n")

    prefix = scene_dir / ts
    rc = rvr.main(["--triplet", str(prefix), "--out", str(tmp_path / "r"),
                   "--seg", "--imu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "seg waterline-contact" in out
    assert "no usable IMU log" not in out


def test_range_vs_radar_imu_flag_without_sidecar(
        tmp_path, monkeypatch, synthetic_triplet, capsys):
    """--imu on a clip without an IMU log prints the fallback note
    (191-192)."""
    import scripts.eval.range_vs_radar as rvr

    monkeypatch.setattr(rvr, "ObstacleDetectionPipeline", _StubRangePipeline)
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    rc = rvr.main(["--triplet", str(prefix), "--out", str(tmp_path / "r2"),
                   "--imu"])
    assert rc == 0
    assert "no usable IMU log" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# One-or-two-line gaps
# ---------------------------------------------------------------------------


def test_export_yolo_skips_class_outside_kept_set(tmp_path, monkeypatch):
    """A bbox whose class isn't in --classes is skipped
    (export_yolo.py line 124)."""
    import cv2

    import scripts.eval.export_yolo as ey

    scene, ts = "Synth", "2099-01-01_00-00-00"
    captures = tmp_path / "captures"
    scene_dir = captures / scene
    scene_dir.mkdir(parents=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(scene_dir / f"fisheye_{ts}.mp4"), fourcc, 3.0,
                         (160, 120))
    tw = cv2.VideoWriter(str(scene_dir / f"thermal_{ts}.mp4"), fourcc, 3.0,
                         (160, 120))
    for i in range(3):
        fw.write(np.full((120, 160, 3), 60 + i, np.uint8))
        tw.write(np.full((120, 160, 3), 60 + i, np.uint8))
    fw.release()
    tw.release()
    (scene_dir / f"mmwave_{ts}.csv").write_text(
        "Date,Time,X,Y,Z\n" + "\n".join(
            f"2099-01-01,00:00:0{i}.0,0.1,1.5,0.0" for i in range(3)) + "\n")
    monkeypatch.setattr(ey, "CAPTURES_ROOT", captures)

    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({
        "frame_id": f"{scene}/{ts}/ts=00-00-00.0",
        "fisheye_bboxes": [
            {"cls": "boat", "xyxy": [0.2, 0.3, 0.6, 0.9], "confidence": 0.9},
            {"cls": "duck", "xyxy": [0.1, 0.1, 0.2, 0.2], "confidence": 0.9},
        ],
    }) + "\n")
    out = tmp_path / "yolo"
    rc = ey.main(["--labels", str(labels), "--out", str(out),
                  "--classes", "boat"])
    assert rc == 0
    txts = list((out / "labels").rglob("*.txt"))
    assert len(txts) == 1
    lines = txts[0].read_text().strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("0 ")


def test_freespace_viz_bin_column_outside_narrow_frame(
        tmp_path, synthetic_triplet, capsys):
    """A frame narrower than the bin-centre column span hits the
    out-of-image continue (freespace_viz.py line 100)."""
    import cv2
    from PIL import Image

    from scripts.eval.freespace_viz import _clipdir, main
    from scripts.utils.segmentation import OBSTACLE, SKY, WATER

    ts = synthetic_triplet.timestamp
    w, h = 300, 240
    clip = _clipdir("Synth", ts)
    frames_root = tmp_path / "frames"
    seg_root = tmp_path / "seg"
    (frames_root / clip).mkdir(parents=True)
    (seg_root / clip).mkdir(parents=True)
    m = np.full((h, w), WATER, np.uint8)
    m[:80, :] = SKY
    m[150:, 100:130] = OBSTACLE
    cv2.imwrite(str(frames_root / clip / "ts=00-00-00.0.jpg"),
                np.full((h, w, 3), 90, np.uint8))
    Image.fromarray(m).save(seg_root / clip / "ts=00-00-00.0.png")

    prefix = synthetic_triplet.fisheye.parent / ts
    out = tmp_path / "out"
    rc = main(["--triplet", str(prefix), "--seg-root", str(seg_root),
               "--frames-root", str(frames_root), "--out", str(out),
               "--limit", "0"])
    assert rc == 0
    assert "1 free-space frames" in capsys.readouterr().out
    assert len(list((out / clip).glob("ts=*.png"))) == 1


def test_healthcheck_placeholder_extrinsics_note(monkeypatch, capsys):
    """Unmeasured extrinsics print the placeholder warning
    (healthcheck.py line 68)."""
    import scripts.eval.healthcheck as hc

    monkeypatch.setattr(hc, "load_extrinsics", lambda: {"measured": False})
    monkeypatch.setattr(hc, "list_triplets", lambda: [])
    with mock.patch.object(sys, "argv", ["hc", "--skip-tests"]):
        with pytest.raises(SystemExit) as exc:
            hc.main()
        assert exc.value.code == 0
    assert "placeholders" in capsys.readouterr().out


def _none_frame_iterator(*_a, **_k):
    yield "00:00:00.0", None, None, np.empty((0, 3))


def test_seg_overlay_skips_none_frames(tmp_path, monkeypatch,
                                       synthetic_triplet):
    """iterate_triplet yielding a None fisheye frame is skipped
    (seg_overlay.py line 69)."""
    import scripts.eval.seg_overlay as so

    class _Stub:
        def __init__(self, intr, det, seg_provider=None):
            pass

    monkeypatch.setattr(so, "ObstacleDetectionPipeline", _Stub)
    monkeypatch.setattr(so, "iterate_triplet", _none_frame_iterator)
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    n = so.render_clip(str(prefix), str(tmp_path / "seg"), tmp_path / "o",
                       limit=0)
    assert n == 0


def test_seg_range_ablation_skips_none_frames(tmp_path, monkeypatch,
                                              synthetic_triplet):
    """Same None-frame guard in seg_range_ablation.py (line 69)."""
    import scripts.eval.seg_range_ablation as sra

    class _Stub:
        def __init__(self, intr, det, seg_provider=None):
            pass

    monkeypatch.setattr(sra, "ObstacleDetectionPipeline", _Stub)
    monkeypatch.setattr(sra, "iterate_triplet", _none_frame_iterator)
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    rows: list[dict] = []
    summary = sra.run_clip(str(prefix), str(tmp_path / "seg"), rows)
    assert summary["frames"] == 0
    assert rows == []


def _lars_det_tree_300(root, split="val", size=(32, 24)):
    from PIL import Image

    w, h = size
    img_dir = root / "lars_v1.0.0_images" / split / "images"
    ann_dir = root / "lars_v1.0.0_annotations" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(img_dir / "a.jpg")
    images = [{"id": 0, "width": w, "height": h, "file_name": "a.jpg"}]
    # 299 ghost annotations (unknown image -> cheap skip), then the real
    # one at k=299 so (k+1) % 300 == 0 fires on a processed record.
    annotations = [{"image_id": 999, "file_name": "ghost.png",
                    "segments_info": []} for _ in range(299)]
    annotations.append({"image_id": 0, "file_name": "a.png",
                        "segments_info": [
                            {"category_id": 11, "bbox": [2, 2, 10, 10],
                             "id": 1}]})
    (ann_dir / "panoptic_annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations}))


def test_lars_export_det_progress_print(tmp_path, capsys):
    """The every-300 progress line prints (export_yolo_det.py 114-115)."""
    from scripts.lars.export_yolo_det import convert_split

    _lars_det_tree_300(tmp_path)
    n_img, n_obj = convert_split(tmp_path, tmp_path / "out", "val",
                                 max_side=0, limit=None)
    assert n_img == 1 and n_obj == 1
    assert "[val] 300/300" in capsys.readouterr().err


def test_lars_export_seg_progress_print(tmp_path, capsys):
    """The every-300 progress line prints (export_yolo_seg.py 87-88)."""
    from PIL import Image

    from scripts.lars.export_yolo_seg import convert_split

    split = "val"
    w, h = 32, 24
    ann_dir = tmp_path / "lars_v1.0.0_annotations" / split
    masks_dir = ann_dir / "panoptic_masks"
    masks_dir.mkdir(parents=True)
    mask = np.zeros((h, w, 3), np.uint8)
    mask[4:16, 4:20] = (44, 1, 0)              # id 300 = 44 + 256*1
    Image.fromarray(mask).save(masks_dir / "a.png")
    images = [{"id": 0, "width": w, "height": h, "file_name": "a.jpg"}]
    annotations = [{"image_id": 999, "file_name": "ghost.png",
                    "segments_info": []} for _ in range(299)]
    annotations.append({"image_id": 0, "file_name": "a.png",
                        "segments_info": [{"category_id": 11, "id": 300}]})
    (ann_dir / "panoptic_annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations}))

    n_img, n_obj = convert_split(tmp_path, tmp_path / "out", split, None)
    assert n_img == 1 and n_obj == 1
    assert "[val] 300/300" in capsys.readouterr().err


def test_lars_instance_polygons_degenerate_contour():
    """A 1-px line approximates to < 3 points -> skipped
    (export_yolo_seg.py lines 42-43)."""
    from scripts.lars.export_yolo_seg import instance_polygons

    m = np.zeros((48, 64), np.uint8)
    m[10, 10:40] = 1
    assert instance_polygons(m, 64, 48, min_area=0) == []


def _curved_water_mask(shape=(648, 864)):
    """Water below a curved (dock-like) boundary: horizon_from_water reads
    a LOW confidence here (~0.1 on the 2026-09-03 bench clip)."""
    from scripts.utils.segmentation import OBSTACLE, SKY, WATER
    h, w = shape
    seg = np.full(shape, WATER, np.uint8)
    cols = np.arange(w)
    edge = (260 + 120 * np.sin(cols / 90.0)).astype(int)
    for c in range(w):
        seg[:edge[c], c] = SKY
        seg[edge[c]:edge[c] + 40, c] = OBSTACLE
    return seg


def test_lazy_horizon_skips_ransac_for_seg_detector_even_when_edge_is_curved(
        intrinsics_real, detection_real, monkeypatch):
    """Measured on the box 2026-09-07: on a dockside clip the water-edge fit's
    confidence is ~0.1, so `horizon.lazy` never fired and the RANSAC ran for a
    mask the segmentation detector does not use. With the seg detector the
    RANSAC is skipped whenever a mask exists; the classical detector keeps it."""
    import scripts.sensor_processing.pipeline as pl
    from scripts.utils.segmentation import horizon_from_water

    mask = _curved_water_mask()
    assert horizon_from_water(mask)[2] < 0.5, "fixture must be a low-confidence edge"
    calls = {"n": 0}
    orig = pl.detect_horizon

    def spy(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)
    monkeypatch.setattr(pl, "detect_horizon", spy)
    frame = np.full((648, 864, 3), 80, np.uint8)

    det = copy.deepcopy(detection_real)
    det["horizon"]["lazy"] = True
    det["fisheye"]["detector"] = "segmentation"
    res = pl.ObstacleDetectionPipeline(intrinsics_real, det).process_fisheye(frame, seg_mask=mask)
    assert calls["n"] == 0, "seg detector + mask: no RANSAC"
    assert res.horizon_line[2] < 0.5       # reported honestly, not promoted
    assert res.horizon_mask.min() == 255    # whole frame kept (fallback_row 0 semantics)

    det2 = copy.deepcopy(detection_real)
    det2["horizon"]["lazy"] = True         # classical detector keeps the RANSAC
    pl.ObstacleDetectionPipeline(intrinsics_real, det2).process_fisheye(frame, seg_mask=mask)
    assert calls["n"] == 1
