"""Phase-2 fusion-scorer tests: models, calibration, threat, bake-off
driver, and the fusion.py --scorer shadow emit.

Fixtures (conftest): synthetic_feature_root, make_fusion_triplet.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.fusion_model.build_targets import build_targets_for_clip
from scripts.fusion_model.evaluate import average_precision, reliability
from scripts.fusion_model.models import (
    CONTEXT_COLUMNS,
    GatedMixtureScorer,
    IsotonicCalibrator,
    load_fusion_model_config,
    severity_for_classes,
    threat_from_score,
    urgency,
)


def _synthetic_gate_dataset(n=6000, seed=0):
    """Ground truth: thermal evidence only counts at night, fisheye only by
    day; radar always counts. The gate must recover that structure."""
    rng = np.random.default_rng(seed)
    e = rng.uniform(0, 1, (n, 3))
    c = np.zeros((n, len(CONTEXT_COLUMNS)))
    night = rng.uniform(size=n) < 0.5
    c[:, 0] = np.where(night, -0.2, 0.6)          # sun_elevation_norm
    c[:, 1] = np.where(night, 0.05, 0.5)          # luminance_norm
    c[:, 2] = night.astype(float)                 # is_dark
    c[:, 3] = 1.0                                 # thermal_quality_ok
    c[:, 4] = 1.0 - night                         # seg_available
    c[:, 5] = 1.0                                 # imu_available
    c[:, 6] = rng.uniform(0, 1, n)                # abs_bearing_norm
    c[:, 7] = rng.uniform(0, 1, n)                # range_norm
    signal = np.where(night, e[:, 1], e[:, 0]) * 3.0 + e[:, 2] * 2.0 - 2.0
    p = 1 / (1 + np.exp(-signal))
    y = (rng.uniform(size=n) < p).astype(float)
    return e, c, y, night


# ---------------------------------------------------------------------------
# Gated mixture (v1a)
# ---------------------------------------------------------------------------

def test_gated_mixture_learns_context_gating():
    e, c, y, night = _synthetic_gate_dataset()
    m = GatedMixtureScorer.fit(e, c, y, seed=0, epochs=400, lr=0.05)
    w = m.gate_weights(c)
    # The interpretability contract: thermal weight up at night, fisheye
    # weight up by day: recovered from data, not hand-coded.
    assert w[night, 1].mean() > 1.5 * w[~night, 1].mean()
    assert w[~night, 0].mean() > 1.5 * w[night, 0].mean()
    p = m.predict_proba(e, c)
    assert average_precision(y, p) > y.mean() + 0.2


def test_gated_mixture_weights_nonnegative_and_roundtrip(tmp_path):
    e, c, y, _ = _synthetic_gate_dataset(n=800)
    m = GatedMixtureScorer.fit(e, c, y, seed=1, epochs=100)
    assert (m.gate_weights(c) >= 0).all()          # softplus construction
    path = tmp_path / "m.json"
    m.save(path)
    m2 = GatedMixtureScorer.load(path)
    np.testing.assert_allclose(m.predict_proba(e, c),
                               m2.predict_proba(e, c), rtol=1e-12)


def test_gated_mixture_seed_determinism():
    e, c, y, _ = _synthetic_gate_dataset(n=500)
    a = GatedMixtureScorer.fit(e, c, y, seed=7, epochs=50)
    b = GatedMixtureScorer.fit(e, c, y, seed=7, epochs=50)
    np.testing.assert_array_equal(a.u, b.u)


# ---------------------------------------------------------------------------
# Calibration + threat
# ---------------------------------------------------------------------------

def test_isotonic_calibrator_monotone_and_improves_ece():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 500)
    y = (rng.uniform(size=500) < s ** 2).astype(float)   # miscalibrated
    cal = IsotonicCalibrator.fit(s, y)
    out = cal.transform(np.linspace(0, 1, 50))
    assert (np.diff(out) >= -1e-9).all()
    assert reliability(y, cal.transform(s))["ece"] < reliability(y, s)["ece"]


def test_urgency_and_severity():
    cfg = load_fusion_model_config()
    assert urgency(1.0, None, 1.0, cfg) > urgency(10.0, None, 1.0, cfg)
    assert urgency(None, 5.0, 1.0, cfg) > urgency(None, 50.0, 1.0, cfg)
    # unknown range/ttc floors, never zero (unknown != safe)
    assert urgency(None, None, 1.0, cfg) == pytest.approx(
        cfg["urgency"]["unknown_floor"])
    assert severity_for_classes("person|boat", cfg) == 1.0
    assert severity_for_classes("buoy", cfg) < severity_for_classes("person", cfg)
    assert severity_for_classes("", cfg) == 1.0   # unknown class = neutral
    t = threat_from_score(np.array([0.8]), np.array([2.0]),
                          np.array([float("nan")]), 1.0, cfg)
    assert 0.0 < t[0] <= 0.8


# ---------------------------------------------------------------------------
# Bake-off driver end-to-end
# ---------------------------------------------------------------------------

def test_train_bakeoff_end_to_end(synthetic_feature_root, tmp_path,
                                  monkeypatch):
    from scripts.fusion_model import train as train_mod
    root, labels, clip_ids = synthetic_feature_root
    for d in root.iterdir():
        df_t = build_targets_for_clip(d, labels)
        if not df_t.empty:
            df_t.to_csv(d / "targets.csv", index=False)
    monkeypatch.setattr(train_mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path / "models")
    rc = train_mod.main(["--features", str(root), "--tag", "test",
                         "--seeds", "0", "1"])
    assert rc == 0
    assert (tmp_path / "results" / "bakeoff_test.md").exists()
    # tagged runs write a tagged artefact; only an untagged/v1a run may
    # touch the LIVE fusion_scorer_v1a.json
    assert (tmp_path / "models" / "fusion_scorer_test.json").exists()
    assert not (tmp_path / "models" / "fusion_scorer_v1a.json").exists()
    gc = pd.read_csv(tmp_path / "results" / "gate_curves_test.csv")
    assert {"w_fisheye", "w_thermal", "w_radar"} <= set(gc.columns)
    report = (tmp_path / "results" / "bakeoff_test.md").read_text()
    assert "incumbent (score_legacy)" in report
    assert "v1a gated mixture" in report


# ---------------------------------------------------------------------------
# fusion.py --scorer shadow emit
# ---------------------------------------------------------------------------

def test_fusion_shadow_scorer_emits_fields(tmp_path, make_fusion_triplet):
    from scripts.sensor_processing import fusion as fusion_mod
    e, c, y, _ = _synthetic_gate_dataset(n=400)
    m = GatedMixtureScorer.fit(e, c, y, seed=0, epochs=30)
    model_path = tmp_path / "scorer.json"
    m.save(model_path)

    trip = make_fusion_triplet(tmp_path / "clip", "2099-07-03_12-00-00", n=3)
    out = tmp_path / "sectors.jsonl"
    import sys as _sys
    argv = _sys.argv
    _sys.argv = ["fusion", "--triplet",
                 str(trip.fisheye.parent / trip.timestamp),
                 "--out", str(out), "--scorer", str(model_path),
                 "--print-every", "0"]
    try:
        fusion_mod.main()
    finally:
        _sys.argv = argv
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert recs, "no JSONL records written"
    n_bins = len(recs[0]["bin_centers_deg"])
    for r in recs:
        assert len(r["p_obstacle"]) == n_bins
        assert len(r["threat"]) == n_bins
        assert all(0.0 <= v <= 1.0 for v in r["p_obstacle"])
        assert all(0.0 <= v <= 1.0 for v in r["threat"])
        assert "scores" in r   # legacy fields untouched


def test_calibrator_roundtrip(tmp_path):
    """The artefact must carry the isotonic calibrator: without it the
    deployed p_obstacle floor sits at the class-balanced prior (~0.5) and
    open water reads as obstacle (2026-08-22)."""
    import numpy as np
    from scripts.fusion_model.models import (
        GatedMixtureScorer, IsotonicCalibrator, load_any_scorer)

    rng = np.random.default_rng(0)
    e = rng.uniform(0, 1, (200, 3))
    c = rng.uniform(0, 1, (200, 8))
    y = (e.sum(axis=1) > 1.8).astype(float)
    m = GatedMixtureScorer.fit(e, c, y, seed=0, epochs=50)
    m.calibrator = IsotonicCalibrator.fit(m.predict_proba(e, c), y)
    path = tmp_path / "m.json"
    m.save(path)
    loaded = load_any_scorer(path)
    assert loaded.calibrator is not None
    raw = loaded.predict_proba(e, c)
    cal = loaded.predict_proba_calibrated(e, c)
    assert not np.allclose(raw, cal)
    # calibrated no-evidence floor must sit near the true base rate,
    # far below the class-balanced raw floor
    e0 = np.zeros((50, 3))
    c0 = c[:50]
    assert cal.mean() == np.mean(loaded.calibrator.transform(raw))
    assert loaded.predict_proba_calibrated(e0, c0).mean() \
        <= loaded.predict_proba(e0, c0).mean() + 1e-9


def test_motion_context_columns():
    """Motion-aware context is opt-in and artefact-declared: an 8-column
    artefact and an 11-column one build their own matrices; absent
    target_* columns fall back to safe defaults (no target, CPA far)."""
    import numpy as np
    import pandas as pd
    from scripts.fusion_model.models import (
        CONTEXT_COLUMNS, MOTION_CONTEXT_COLUMNS, context_matrix)

    df = pd.DataFrame({
        "sun_elevation_deg": [30.0, 30.0],
        "luminance_mean": [120.0, 120.0],
        "is_dark": [False, False],
        "thermal_quality_ok": [True, True],
        "seg_available": [True, True],
        "imu_available": [True, True],
        "bin_center_deg": [0.0, 10.0],
        "radar_min_range_m": [3.0, float("nan")],
        "target_present": [True, False],
        "target_closing_mps": [0.5, float("nan")],
        "target_cpa_m": [1.0, float("nan")],
    })
    base = context_matrix(df, cfg={}, columns=tuple(CONTEXT_COLUMNS))
    assert base.shape == (2, len(CONTEXT_COLUMNS))
    cols = tuple(CONTEXT_COLUMNS) + MOTION_CONTEXT_COLUMNS
    c = context_matrix(df, cfg={}, columns=cols)
    assert c.shape == (2, len(cols))
    assert not np.isnan(c).any()
    i = len(CONTEXT_COLUMNS)
    assert c[0, i] == 1.0 and c[1, i] == 0.0          # present
    assert c[0, i + 1] == 0.5 and c[1, i + 1] == 0.0  # closing norm, NaN->0
    assert c[0, i + 2] < 1.0 and c[1, i + 2] == 1.0   # cpa norm, NaN->far


def test_typed_evidence_channel():
    """4th evidence channel is artefact-declared: 3- and 4-sensor models
    build their own matrices; missing yolo columns fall back to zero."""
    import numpy as np
    import pandas as pd
    from scripts.fusion_model.models import SENSORS, evidence_matrix

    df = pd.DataFrame({
        "seg_obstacle_frac": [0.1, 0.0],
        "thermal_det_count": [1.0, 0.0],
        "radar_n_points": [4.0, 0.0],
        "yolo_max_conf": [0.83, float("nan")],
    })
    e3 = evidence_matrix(df, cfg={}, sensors=SENSORS)
    assert e3.shape == (2, 3)
    e4 = evidence_matrix(df, cfg={}, sensors=SENSORS + ("yolo",))
    assert e4.shape == (2, 4)
    assert e4[0, 3] == 0.83 and e4[1, 3] == 0.0
    # without the column at all, the channel is zero (not an error)
    e4b = evidence_matrix(df.drop(columns=["yolo_max_conf"]), cfg={},
                          sensors=SENSORS + ("yolo",))
    assert (e4b[:, 3] == 0.0).all()


def test_yolo_bin_features_bearing_mapping():
    import numpy as np
    from scripts.fusion_model.build_features import yolo_bin_features

    K = np.array([[418.5, 0, 432.0], [0, 418.5, 324.0], [0, 0, 1.0]])
    edges = np.arange(-55.0, 65.0, 10.0)
    # a box centred on the image centre -> bearing 0 -> the [-5,5) bin
    boxes = [{"cls": "boat", "xyxy": [0.45, 0.4, 0.55, 0.6],
              "confidence": 0.9}]
    f = yolo_bin_features(boxes, edges, K)
    centre_bin = int(np.searchsorted(edges, 0.0) - 1)
    assert f["yolo_available"] == [True] * (len(edges) - 1)
    assert f["yolo_max_conf"][centre_bin] == 0.9
    assert f["yolo_top_cls"][centre_bin] == "boat"
    # stream absent
    f0 = yolo_bin_features(None, edges, K)
    assert f0["yolo_available"] == [False] * (len(edges) - 1)
