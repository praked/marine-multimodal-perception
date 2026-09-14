"""Phase-1 fusion-scorer tests: target builder + stratified eval harness.

Fixtures (conftest): synthetic_feature_root = two clips exported for real
by build_features + fabricated audited labels.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.eval.metrics import load_labels
from scripts.fusion_model.build_targets import (
    build_targets_for_clip,
    label_mono_range,
)
from scripts.fusion_model.evaluate import (
    average_precision,
    daypart,
    evaluate_scores,
    load_dataset,
    pr_at_threshold,
    reliability,
)


def _feature_dir(root, clip_id):
    return next(p for p in root.iterdir()
                if clip_id.replace("/", "__") == p.name)


# ---------------------------------------------------------------------------
# Target builder
# ---------------------------------------------------------------------------

def test_targets_positive_centre_bin_with_tolerance(synthetic_feature_root):
    root, labels, clip_ids = synthetic_feature_root
    df = build_targets_for_clip(_feature_dir(root, clip_ids[0]), labels)
    assert not df.empty
    centre = df[(df.frame_index == 2) & (df.bin_center_deg.abs() < 6)]
    assert (centre.y_obstacle == 1).all()
    far = df[(df.frame_index == 2) & (df.bin_center_deg.abs() > 30)]
    assert (far.y_obstacle == 0).all()
    assert centre.label_classes.str.contains("person").any()


def test_targets_rows_only_for_labeled_frames(synthetic_feature_root):
    root, labels, clip_ids = synthetic_feature_root
    fdir = _feature_dir(root, clip_ids[0])
    df = build_targets_for_clip(fdir, labels, dilation_frames=7)
    assert set(df.frame_index.unique()) == {1, 2, 3}
    # Same-frame support everywhere here -> dilation adds nothing and
    # nothing is flagged as dilation-only.
    df0 = build_targets_for_clip(fdir, labels, dilation_frames=0)
    sel = df.bin_center_deg.abs() < 6
    assert df[sel].y_obstacle.sum() == df0[df0.bin_center_deg.abs() < 6].y_obstacle.sum()
    assert (df.from_dilation == 0).all()


def test_targets_relevance_gate_far_label(tmp_path, synthetic_feature_root):
    root, labels, clip_ids = synthetic_feature_root
    scene, ts = clip_ids[0].split("/")
    far_path = tmp_path / "far.jsonl"
    far_path.write_text(json.dumps({
        "frame_id": f"{scene}/{ts}/000002", "scene": scene, "audited": True,
        "source": "manual", "width": 864, "height": 648,
        # bbox bottom ABOVE the principal point -> above the level horizon
        # -> mono range None -> far bank -> perception-only, not nav GT.
        "fisheye_bboxes": [{"cls": "structure",
                            "xyxy": [0.45, 0.10, 0.55, 0.25]}],
        "obstacle_bins_fisheye": [0]}) + "\n")
    labels = load_labels([far_path])
    # With the explicit 30 m gate the far-bank label is perception-only.
    df = build_targets_for_clip(_feature_dir(root, clip_ids[0]), labels,
                                max_relevant_range_m=30.0)
    pos = df[df.y_obstacle == 1]
    assert len(pos) and (pos.y_relevant == 0).all() and (pos.y_nav == 0).all()
    # DEFAULT is unbounded (D.2 ruling, AuthorOne 2026-08-24): the same label
    # IS nav-relevant.
    df2 = build_targets_for_clip(_feature_dir(root, clip_ids[0]), labels)
    pos2 = df2[df2.y_obstacle == 1]
    assert len(pos2) and (pos2.y_relevant == 1).all() and (pos2.y_nav == 1).all()


def test_label_mono_range_near_vs_far(intrinsics_real):
    K = np.asarray(intrinsics_real["fisheye"]["K"], float)
    near = label_mono_range((0.45, 0.55, 0.55, 0.85), K, (864, 648),
                            None, 0.27)
    far = label_mono_range((0.45, 0.10, 0.55, 0.25), K, (864, 648),
                           None, 0.27)
    assert near is not None and near < 30.0
    assert far is None


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def test_pr_and_ap_basics():
    y = np.array([1, 0, 1, 0, 0], dtype=float)
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    r = pr_at_threshold(y, s, 0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 0)
    assert average_precision(y, s) == pytest.approx((1.0 + 2 / 3) / 2)


def test_reliability_perfect_and_bad():
    y = np.array([1.0] * 50 + [0.0] * 50)
    s_good = np.array([0.95] * 50 + [0.05] * 50)
    s_bad = np.array([0.55] * 50 + [0.45] * 50)
    assert reliability(y, s_good)["ece"] == pytest.approx(0.05, abs=0.01)
    assert reliability(y, s_bad)["ece"] > 0.4


def test_daypart_civil_twilight():
    e = pd.Series([45.0, 3.0, -3.0, -20.0])
    assert list(daypart(e)) == ["day", "dusk", "dusk", "night"]


# ---------------------------------------------------------------------------
# Dataset load + incumbent evaluation
# ---------------------------------------------------------------------------

def test_load_dataset_and_evaluate_incumbent(synthetic_feature_root):
    root, labels, clip_ids = synthetic_feature_root
    for d in root.iterdir():
        df_t = build_targets_for_clip(d, labels)
        if not df_t.empty:
            df_t.to_csv(d / "targets.csv", index=False)
    df = load_dataset(root)
    assert not df.empty and "y_nav" in df.columns
    res = evaluate_scores(df, "score_legacy", "y_nav")
    assert res["n_rows"] == len(df)
    assert "scene" in res["strata"] and "daypart" in res["strata"]
    assert res["person_rows"] > 0
