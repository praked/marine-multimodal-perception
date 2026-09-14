import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from scripts.eval.metrics import (
    ClipResult,
    Counters,
    HeadingCounters,
    _heading_table,
    _parse_frame_id,
    load_labels,
    main,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_parse_frame_id_ok():
    scene, ts, idx, fts = _parse_frame_id("Boats/2025-06-23_16-21-07/000123")
    assert scene == "Boats" and ts == "2025-06-23_16-21-07" and idx == 123
    assert fts is None


def test_parse_frame_id_zero_padded_zero():
    _, _, idx, _ = _parse_frame_id("Boats/ts/000000")
    assert idx == 0


def test_parse_frame_id_ts_scheme():
    scene, ts, idx, fts = _parse_frame_id(
        "2026-06-17_institutionone_day1/2026-06-17_12-41-23/ts=12-41-26.3")
    assert scene == "2026-06-17_institutionone_day1"
    assert idx == -1 and fts == "12:41:26.3"


def test_parse_frame_id_invalid_raises():
    with pytest.raises(ValueError):
        _parse_frame_id("just_a_string")


def test_counters_precision_recall_f1():
    c = Counters()
    c.add(True, True)    # TP
    c.add(True, True)    # TP
    c.add(True, False)   # FP
    c.add(False, True)   # FN
    c.add(False, False)  # TN
    assert c.precision() == pytest.approx(2 / 3)
    assert c.recall() == pytest.approx(2 / 3)
    assert c.f1() == pytest.approx(2 / 3)


def test_counters_no_positives_nan():
    c = Counters()
    c.add(False, False)
    import math
    assert math.isnan(c.precision())
    assert math.isnan(c.recall())
    assert math.isnan(c.f1())


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def _write_label(path: Path, **kwargs):
    obj = {
        "frame_id": "Boats/ts/0",
        "scene": "Boats",
        "source": "manual",
        "audited": True,
        "fisheye_bboxes": [],
        "obstacle_bins_fisheye": [],
        "width": 864, "height": 648,
    }
    obj.update(kwargs)
    path.write_text(json.dumps(obj) + "\n")


def test_load_labels_skips_unaudited_by_default(tmp_path):
    p = tmp_path / "x.jsonl"
    _write_label(p, audited=False)
    labels = load_labels([p])
    assert labels == []


def test_load_labels_includes_unaudited_when_flag_set(tmp_path):
    p = tmp_path / "x.jsonl"
    _write_label(p, audited=False)
    labels = load_labels([p], include_unaudited=True)
    assert len(labels) == 1


def test_load_labels_missing_file_skipped(tmp_path):
    labels = load_labels([tmp_path / "nope.jsonl"])
    assert labels == []


def test_load_labels_picks_up_safe_heading(tmp_path):
    """A frame with `safe_heading_deg` present should round-trip
    through load_labels with safe_heading_labelled=True. Missing key
    must stay False (unlabelled-for-heading)."""
    p = tmp_path / "x.jsonl"
    p.write_text(
        json.dumps({
            "frame_id": "Boats/ts/1", "scene": "Boats", "source": "manual",
            "audited": True, "fisheye_bboxes": [], "obstacle_bins_fisheye": [],
            "width": 864, "height": 648,
        }) + "\n" +
        json.dumps({
            "frame_id": "Boats/ts/2", "scene": "Boats", "source": "manual",
            "audited": True, "fisheye_bboxes": [], "obstacle_bins_fisheye": [],
            "width": 864, "height": 648,
            "safe_heading_deg": -22.0,
        }) + "\n" +
        json.dumps({
            "frame_id": "Boats/ts/3", "scene": "Boats", "source": "manual",
            "audited": True, "fisheye_bboxes": [], "obstacle_bins_fisheye": [],
            "width": 864, "height": 648,
            "safe_heading_deg": None,
        }) + "\n"
    )
    labels = load_labels([p])
    assert len(labels) == 3
    assert labels[0].safe_heading_labelled is False
    assert labels[1].safe_heading_labelled is True
    assert labels[1].safe_heading_deg == -22.0
    assert labels[2].safe_heading_labelled is True
    assert labels[2].safe_heading_deg is None


# ---------------------------------------------------------------------------
# HeadingCounters
# ---------------------------------------------------------------------------

def test_heading_counters_within_15():
    hc = HeadingCounters()
    hc.add_frame(label_deg=-20.0, raw_deg=-12.0, smoothed_deg=-15.0)
    hc.add_frame(label_deg=10.0, raw_deg=30.0, smoothed_deg=18.0)
    assert hc.n_compared_raw == 2
    assert hc.n_within_15_raw == 1            # only the -20 vs -12 (err 8)
    assert hc.n_within_15_smoothed == 2       # |-15-(-20)|=5, |18-10|=8


def test_heading_counters_abstention_paths():
    hc = HeadingCounters()
    # Label says no safe direction; raw abstains (correct), smoothed picks (false rec).
    hc.add_frame(label_deg=None, raw_deg=None, smoothed_deg=20.0)
    # Label says go right; raw abstains (missed), smoothed says +18 (within 15° of +20).
    hc.add_frame(label_deg=20.0, raw_deg=None, smoothed_deg=18.0)
    assert hc.n_correct_abstain_raw == 1
    assert hc.n_false_recommend_smoothed == 1
    assert hc.n_missed_recommend_raw == 1
    assert hc.n_compared_smoothed == 1
    assert hc.n_within_15_smoothed == 1


def test_heading_counters_smoothed_missed_recommend():
    """Label is a real heading but the smoothed prediction abstains ->
    n_missed_recommend_smoothed increments (line 192)."""
    hc = HeadingCounters()
    hc.add_frame(label_deg=20.0, raw_deg=18.0, smoothed_deg=None)
    assert hc.n_missed_recommend_smoothed == 1
    assert hc.n_compared_raw == 1
    assert hc.n_compared_smoothed == 0


def test_heading_counters_nan_when_no_comparisons():
    import math
    hc = HeadingCounters()
    assert math.isnan(hc.mae_raw())
    assert math.isnan(hc.within15_raw())


# ---------------------------------------------------------------------------
# Heading table rendering
# ---------------------------------------------------------------------------

def test_heading_table_empty_when_no_labels():
    cr = ClipResult(clip_id="Boats/ts", scene="Boats")
    assert _heading_table("Boats", [cr]) == ""


def test_heading_table_renders_mae_and_within15():
    cr = ClipResult(clip_id="Boats/ts", scene="Boats")
    cr.heading.add_frame(label_deg=10.0, raw_deg=12.0, smoothed_deg=11.0)
    cr.heading.add_frame(label_deg=-20.0, raw_deg=-50.0, smoothed_deg=-25.0)
    out = _heading_table("Boats", [cr])
    assert "Boats heading" in out
    assert "raw" in out
    assert "smoothed" in out
    # 1/2 raw within 15° → 50.0%, 2/2 smoothed within 15° → 100.0%.
    assert "50.0%" in out
    assert "100.0%" in out


def test_heading_table_renders_abstain_section():
    cr = ClipResult(clip_id="Ducks/ts", scene="Ducks")
    cr.heading.add_frame(label_deg=None, raw_deg=None, smoothed_deg=None)
    cr.heading.add_frame(label_deg=None, raw_deg=15.0, smoothed_deg=15.0)
    out = _heading_table("Ducks", [cr])
    assert "Abstain calibration" in out
    # Two label-null frames: 1 correct abstain (raw), 1 false recommend (raw).
    assert out.count("| 1 | 1 |") >= 1


def test_load_labels_multiple_files(tmp_path):
    a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"
    _write_label(a, frame_id="Boats/t1/0")
    _write_label(b, frame_id="Boats/t1/1")
    labels = load_labels([a, b])
    assert len(labels) == 2


def test_load_labels_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n\n" + json.dumps({
        "frame_id": "A/b/0", "scene": "A", "audited": True,
        "obstacle_bins_fisheye": [],
    }) + "\n\n")
    assert len(load_labels([p])) == 1


def test_load_labels_parses_clip_id(tmp_path):
    p = tmp_path / "x.jsonl"
    _write_label(p, frame_id="Ducks/2025-07-15_05-36-18/000042")
    labels = load_labels([p])
    assert labels[0].clip_id == "Ducks/2025-07-15_05-36-18"
    assert labels[0].frame_idx == 42


# ---------------------------------------------------------------------------
# Integration on a real clip
# ---------------------------------------------------------------------------

def test_main_runs_against_real_label(tmp_path, monkeypatch, boats_triplet):
    """Write a single label that points into the real boats clip and run main()."""
    labels_path = tmp_path / "labels.jsonl"
    _write_label(labels_path,
                 frame_id=f"{boats_triplet.clip_id}/000000",
                 scene=boats_triplet.scene,
                 obstacle_bins_fisheye=[0],
                 audited=True)
    monkeypatch.setattr("scripts.eval.metrics.RESULTS_DIR", tmp_path)
    with mock.patch.object(sys, "argv", [
        "metrics", "--labels", str(labels_path), "--run-id", "test_run",
    ]):
        main()
    out = tmp_path / "metrics_test_run.md"
    assert out.exists()
    text = out.read_text()
    assert "Boats" in text
    assert "fused" in text


def test_main_with_heading_labels(tmp_path, monkeypatch, boats_triplet):
    """A label carrying safe_heading_deg drives the heading accumulation
    (line 354) and renders the heading section of the report (lines 564-568)."""
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps({
            "frame_id": f"{boats_triplet.clip_id}/000000",
            "scene": boats_triplet.scene,
            "source": "manual", "audited": True,
            "fisheye_bboxes": [], "obstacle_bins_fisheye": [0],
            "width": 864, "height": 648,
            "safe_heading_deg": 0.0,
        }) + "\n"
    )
    monkeypatch.setattr("scripts.eval.metrics.RESULTS_DIR", tmp_path)
    with mock.patch.object(sys, "argv", [
        "metrics", "--labels", str(labels_path), "--run-id", "heading_run",
    ]):
        main()
    text = (tmp_path / "metrics_heading_run.md").read_text()
    assert "Heading evaluation" in text


def test_main_clip_failure_is_caught(tmp_path, monkeypatch, capsys):
    """A label whose clip cannot be resolved is caught in main()'s
    try/except (lines 531-532) and reported, not raised."""
    labels_path = tmp_path / "labels.jsonl"
    _write_label(labels_path,
                 frame_id="Nope/1999-01-01_00-00-00/000000",
                 scene="Nope", obstacle_bins_fisheye=[0], audited=True)
    monkeypatch.setattr("scripts.eval.metrics.RESULTS_DIR", tmp_path)
    with mock.patch.object(sys, "argv", [
        "metrics", "--labels", str(labels_path), "--run-id", "fail_run",
    ]):
        main()
    captured = capsys.readouterr()
    assert "failed" in captured.out
    # Report still written even though the only clip failed.
    assert (tmp_path / "metrics_fail_run.md").exists()


# ---------------------------------------------------------------------------
# _evaluate_clip internals
# ---------------------------------------------------------------------------

def _make_label(clip_id, scene, frame_idx, **kw):
    from scripts.eval.metrics import Label
    base = dict(
        frame_id=f"{clip_id}/{frame_idx:06d}",
        scene=scene, clip_id=clip_id, frame_idx=frame_idx,
        source="manual", audited=True, bboxes=[], obstacle_bins=[0],
        image_size=(160, 120),
    )
    base.update(kw)
    return Label(**base)


def test_evaluate_clip_raises_on_unopenable_video(tmp_path, monkeypatch,
                                                  intrinsics_real, detection_real):
    """When neither cap opens, _evaluate_clip raises RuntimeError
    (lines 249-252)."""
    from scripts.eval import metrics
    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
    from scripts.utils.datasets import Triplet

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope")
    csv = tmp_path / "m.csv"
    csv.write_text("Date,Time,X,Y,Z\n2099-01-01,00:00:00.0,0,1,0\n")
    fake = Triplet(scene="Boats", timestamp="ts", fisheye=bad, thermal=bad,
                   mmwave=csv)
    monkeypatch.setattr(metrics, "resolve_triplet", lambda p: fake)

    labels = [_make_label("Boats/ts", "Boats", 0)]
    with pytest.raises(RuntimeError):
        metrics._evaluate_clip(
            labels,
            lambda: ObstacleDetectionPipeline(intrinsics_real, detection_real),
            intrinsics_real, detection_real,
        )


def test_evaluate_clip_label_beyond_radar_and_video_end(
        tmp_path, monkeypatch, intrinsics_real, detection_real,
        synthetic_triplet):
    """Covers: radar fallback when frame_idx >= len(timestamps) (line 283),
    the video-exhausted break (line 277), and per-bin list-length guards
    (lines 302, 304) via a pipeline whose fusion returns wrong-length lists.
    """
    import numpy as np

    from scripts.eval import metrics
    from scripts.sensor_processing.pipeline import (
        FrameResult,
        FusionResult,
        make_bins,
    )

    # Rewrite the radar CSV to have FEWER timestamps (2) than the 5 video
    # frames, so for frame_idx >= 2 the radar-empty fallback (line 283)
    # fires while the video keeps reading.
    short_csv = tmp_path / "short_mmwave.csv"
    short_csv.write_text(
        "Date,Time,X,Y,Z\n"
        "2099-01-01,00:00:00.0,0.1,1.5,0.0\n"
        "2099-01-01,00:00:01.0,0.5,2.0,0.0\n"
    )
    from scripts.utils.datasets import Triplet
    triplet = Triplet(
        scene=synthetic_triplet.scene, timestamp=synthetic_triplet.timestamp,
        fisheye=synthetic_triplet.fisheye, thermal=synthetic_triplet.thermal,
        mmwave=short_csv,
    )
    monkeypatch.setattr(metrics, "resolve_triplet", lambda p: triplet)

    edges, centers = make_bins(detection_real["fusion"])
    n_bins = len(centers)

    class _FakePipeline:
        def process_frame(self, fish, therm, mm_pts, timestamp=None,
                          frame_id=None):
            fusion = FusionResult(
                bin_edges=edges,
                bin_centers=centers,
                scores=[0.0] * n_bins,
                min_ranges=[None] * n_bins,
                sensor_hit_mask=np.zeros((n_bins, 3), dtype=bool),
                # Deliberately wrong lengths -> triggers the reset guards.
                per_bin_velocity_mps=[None],
                per_bin_ttc_s=[None, None],
            )
            return FrameResult(fusion=fusion, timestamp=timestamp)

    # Label index 6 is beyond the 5 radar timestamps (line 283) and beyond
    # the 5 video frames (so the while loop breaks at line 277).
    labels = [
        _make_label(synthetic_triplet.clip_id, synthetic_triplet.scene, 0),
        _make_label(synthetic_triplet.clip_id, synthetic_triplet.scene, 6),
    ]
    result = metrics._evaluate_clip(
        labels, _FakePipeline, intrinsics_real, detection_real,
    )
    # Only frame 0 was reachable; the loop broke before frame 6.
    assert result.n_frames_labelled == 1


def test_main_no_labels_exits(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.eval.metrics.DEFAULT_LABELS",
                        [tmp_path / "absent.jsonl"])
    with mock.patch.object(sys, "argv", ["metrics"]):
        with pytest.raises(SystemExit):
            main()


def test_main_with_tracker(tmp_path, monkeypatch, boats_triplet):
    """--track should pass through and label the report as tracked=on."""
    labels_path = tmp_path / "labels.jsonl"
    _write_label(labels_path,
                 frame_id=f"{boats_triplet.clip_id}/000000",
                 scene=boats_triplet.scene,
                 obstacle_bins_fisheye=[0], audited=True)
    monkeypatch.setattr("scripts.eval.metrics.RESULTS_DIR", tmp_path)
    with mock.patch.object(sys, "argv", [
        "metrics", "--labels", str(labels_path),
        "--run-id", "tracked_run", "--track",
        "--track-min-hits", "2", "--track-max-age", "3",
    ]):
        main()
    text = (tmp_path / "metrics_tracked_run.md").read_text()
    assert "tracking: on" in text
