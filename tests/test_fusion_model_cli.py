"""Fusion-model CLI entrypoints + guard/edge branches (coverage push).

Drives the argparse main() of build_features / build_targets / evaluate /
train end-to-end against synthetic inputs in tmp_path, and exercises the
scattered guard branches (off-FOV bins, missing tables, parquet fallback,
sun failure, IMU sidecar replay, sklearn present/absent via sys.modules
injection; scikit-learn is an optional dep that is absent in CI).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts.eval.metrics import Label, load_labels
from scripts.fusion_model import build_features as bf
from scripts.fusion_model import build_targets as bt
from scripts.fusion_model import evaluate as ev
from scripts.fusion_model import models as fm_models
from scripts.fusion_model import train as tr
from scripts.fusion_model.build_features import (
    build_for_triplet,
    detection_bin_features,
    frame_datetime_utc,
    free_space_bin_features,
    radar_bin_features,
    seg_bin_features,
    write_tables,
    _agg,
)
from scripts.fusion_model.build_targets import (
    build_targets_for_clip,
    label_bearings_px,
    label_mono_range,
)
from scripts.fusion_model.evaluate import (
    average_precision,
    evaluate_scores,
    load_dataset,
    render_report,
    _fmt,
)
from scripts.fusion_model.models import (
    GatedMixtureScorer,
    fit_gbt,
    gbt_predict,
)
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import REPO_ROOT, Triplet, resolve_triplet
from scripts.utils.geometry import UP_LEVEL
from scripts.utils.segmentation import OBSTACLE, SKY, WATER, SegDetectParams


# ---------------------------------------------------------------------------
# Local synthetic-corpus helpers (module-scoped so the pipeline runs once)
# ---------------------------------------------------------------------------

def _make_triplet(base_dir: Path, ts: str, n: int = 6) -> Triplet:
    """Synthetic triplet with a V+SNR+NOISE radar CSV (current format)."""
    scene_dir = base_dir / "Synth"
    scene_dir.mkdir(exist_ok=True, parents=True)
    fish = scene_dir / f"fisheye_{ts}.mp4"
    therm = scene_dir / f"thermal_{ts}.mp4"
    mm = scene_dir / f"mmwave_{ts}.csv"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(fish), fourcc, 3.0, (160, 120))
    tw = cv2.VideoWriter(str(therm), fourcc, 3.0, (160, 120))
    for _ in range(n):
        f = np.full((120, 160, 3), 70, dtype=np.uint8)
        f[40:80, 60:100] = 200
        fw.write(f)
        tw.write(f)
    fw.release()
    tw.release()
    hms = ts.split("_")[1].split("-")
    base_s = int(hms[0]) * 3600 + int(hms[1]) * 60 + int(hms[2])
    rows = ["Date,Time,X,Y,Z,V,SNR,NOISE"]
    for i in range(n):
        t = base_s + i
        tstr = f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}.0"
        rows.append(f"{ts[:10]},{tstr},0.1,1.5,0.0,0.2,14.0,8.0")
    mm.write_text("\n".join(rows) + "\n")
    return Triplet(scene="Synth", timestamp=ts, fisheye=fish,
                   thermal=therm, mmwave=mm)


def _label_recs(ts: str, frames=(1, 2, 3), person_on: int | None = 2) -> list[dict]:
    recs = []
    for idx in frames:
        bboxes = [{"cls": "boat", "xyxy": [0.45, 0.55, 0.55, 0.80],
                   "confidence": 0.9}]
        if idx == person_on:
            bboxes.append({"cls": "person", "xyxy": [0.48, 0.60, 0.52, 0.78]})
        recs.append({"frame_id": f"Synth/{ts}/{idx:06d}", "scene": "Synth",
                     "audited": True, "source": "manual",
                     "width": 864, "height": 648,
                     "fisheye_bboxes": bboxes, "obstacle_bins_fisheye": [0]})
    return recs


def _force_split(root: Path, value: str = "train") -> None:
    for d in root.iterdir():
        f = d / "frames.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["split"] = value
            df.to_csv(f, index=False)


CORPUS_TS = ("2098-07-01_12-00-00", "2098-07-02_12-00-00")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Two feature-exported clips + targets + label files, built once.

    Returns (features_root, labels_all_path, labels_clip1_path, clip_ids).
    Splits are forced to 'train' so train.py's fallback holdout logic is
    deterministic in every environment.
    """
    base = tmp_path_factory.mktemp("fusion_cli_corpus")
    root = base / "features"
    intr, det = load_intrinsics(), load_detection()
    clip_ids = []
    for ts in CORPUS_TS:
        trip = _make_triplet(base, ts)
        bins_df, frames_df, meta = build_for_triplet(
            trip, intr, det, use_imu=False)
        write_tables(bins_df, frames_df, meta, root, fmt="csv")
        clip_ids.append(meta["clip_id"])
    _force_split(root)

    lab_all = base / "labels_all.jsonl"
    recs = [r for ts in CORPUS_TS for r in _label_recs(ts)]
    lab_all.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    lab_one = base / "labels_clip1.jsonl"
    lab_one.write_text("\n".join(json.dumps(r)
                                 for r in _label_recs(CORPUS_TS[0])) + "\n")

    labels = load_labels([lab_all])
    for d in root.iterdir():
        df_t = build_targets_for_clip(d, labels)
        if not df_t.empty:
            df_t.to_csv(d / "targets.csv", index=False)
    return root, lab_all, lab_one, clip_ids


def _feature_dir(root: Path, clip_id: str) -> Path:
    return next(p for p in root.iterdir()
                if p.name == clip_id.replace("/", "__"))


class _FakeHGB:
    # module-level so the v1b bundle's joblib serialization can pickle it
    # (train.py now SHIPS the ensemble; a <locals> class would fail there)
    def __init__(self, random_state=0, class_weight=None, max_depth=None,
                 max_iter=None, learning_rate=None):
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self._p = float(y.mean()) if len(y) else 0.5
        return self

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        p = np.full(n, self._p)
        return np.column_stack([1.0 - p, p])


def _fake_sklearn(monkeypatch) -> None:
    """Inject a minimal fake sklearn so the v1b GBT paths run without the
    real (optional, CI-absent) dependency."""
    fake = types.ModuleType("sklearn")
    ens = types.ModuleType("sklearn.ensemble")
    ens.HistGradientBoostingClassifier = _FakeHGB
    fake.ensemble = ens
    monkeypatch.setitem(sys.modules, "sklearn", fake)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", ens)


def _block_sklearn(monkeypatch) -> None:
    """Force `import sklearn` to raise ImportError even when installed."""
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)


# ---------------------------------------------------------------------------
# build_features: guard/edge branches
# ---------------------------------------------------------------------------

def test_frame_datetime_utc_bad_chunk_start_ignored():
    utc = frame_datetime_utc("2026-07-08", "14:00:00.0", "CET",
                             chunk_start_hms="not-a-time")
    assert (utc.hour, utc.minute) == (12, 0)   # parse error -> no rollover


def test_agg_unknown_aggregation_raises():
    with pytest.raises(ValueError):
        _agg([1.0, 2.0], "bogus")


def test_seg_bin_features_offimage_bands_and_out_of_fov_component():
    seg = np.full((80, 100), SKY, dtype=np.uint8)
    seg[40:, :] = WATER
    seg[30:55, 10:30] = OBSTACLE          # centroid bearing ~ -17 deg
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    edges = np.array([60.0, 70.0, 80.0])  # entirely off-image + off-FOV
    feats = seg_bin_features(seg, edges, K, SegDetectParams(min_area=10))
    assert all(np.isnan(v) for v in feats["seg_obstacle_frac"])
    assert sum(feats["seg_comp_count"]) == 0.0
    assert all(np.isnan(v) for v in feats["seg_comp_max_size_px"])


def test_detection_bin_features_none_and_out_of_fov():
    edges = np.array([-10.0, 0.0, 10.0])
    out = detection_bin_features(None, edges, "fisheye")
    assert out["fisheye_det_count"] == [0.0, 0.0]
    res = SimpleNamespace(angles=[80.0], sizes=[5.0], ranges=[2.0],
                          raw_ranges=[2.1], votes=[True],
                          radar_hits=[1], radar_ranges=[1.5])
    out2 = detection_bin_features(res, edges, "fisheye")
    assert sum(out2["fisheye_det_count"]) == 0.0     # off-FOV -> no bin
    assert sum(out2["fisheye_det_votes"]) == 0.0


def test_radar_bin_features_out_of_fov_point():
    edges = np.array([-10.0, 0.0, 10.0])
    m = SimpleNamespace(angles=[80.0], ranges=[2.0], snr_db=[10.0],
                        noise_db=[5.0], velocities_xy=[(0.0, 0.1)])
    out = radar_bin_features(m, edges)
    assert out["radar_n_points"] == [0.0, 0.0]
    assert all(np.isnan(v) for v in out["radar_min_range_m"])


def test_free_space_bin_features_with_mask_both_attitude_paths(
        intrinsics_real, detection_real):
    seg = np.full((648, 864), SKY, dtype=np.uint8)
    seg[300:, :] = WATER
    seg[280:340, 400:460] = OBSTACLE
    result = SimpleNamespace(fisheye=SimpleNamespace(seg_mask=seg))
    edges = np.arange(-55.0, 56.0, 10.0)

    out = free_space_bin_features(result, intrinsics_real, detection_real,
                                  0.27, edges, attitude=None)
    assert len(out["free_space_m"]) == len(edges) - 1
    assert len(out["free_space_blocked"]) == len(edges) - 1

    class _Att:
        def up_vector_camera(self):
            return UP_LEVEL

    out2 = free_space_bin_features(result, intrinsics_real, detection_real,
                                   0.27, edges, attitude=_Att())
    assert len(out2["free_space_m"]) == len(edges) - 1


def test_build_for_triplet_sun_failure_writes_nan(tmp_path, intrinsics_real,
                                                  detection_real, monkeypatch):
    trip = _make_triplet(tmp_path, "2099-07-06_12-00-00", n=3)

    def boom(*a, **k):
        raise ValueError("no sun today")

    monkeypatch.setattr(bf, "sun_position", boom)
    _, frames_df, _ = build_for_triplet(trip, intrinsics_real, detection_real,
                                        use_imu=False, limit=2)
    assert frames_df["sun_elevation_deg"].isna().all()
    assert frames_df["sun_azimuth_deg"].isna().all()


def test_build_for_triplet_with_imu_sidecar(tmp_path, intrinsics_real,
                                            detection_real):
    ts = "2099-07-07_12-00-00"
    _make_triplet(tmp_path, ts, n=3)
    scene_dir = tmp_path / "Synth"
    rows = ["Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az"]
    for i in range(30):
        rows.append(f"2099-07-07,12:00:{i * 0.2:04.1f},"
                    "10.0,1.5,-2.0,0.0,0.0,9.8")
    (scene_dir / f"imu_{ts}.csv").write_text("\n".join(rows) + "\n")

    trip = resolve_triplet(scene_dir / ts)
    assert trip.imu is not None
    _, frames_df, meta = build_for_triplet(trip, intrinsics_real,
                                           detection_real, use_imu=True,
                                           limit=2)
    assert meta["imu_replayed"]
    assert frames_df["imu_available"].all()
    assert np.isfinite(frames_df["attitude_roll_deg"]).all()


# ---------------------------------------------------------------------------
# build_features: write_tables parquet paths + CLI
# ---------------------------------------------------------------------------

def _tiny_tables():
    bins = pd.DataFrame({"a": [1.0]})
    frames = pd.DataFrame({"b": [2.0]})
    meta = {"clip_id": "Synth/2099-01-01_00-00-00"}
    return bins, frames, meta


def test_write_tables_parquet_success(tmp_path, monkeypatch):
    def fake_to_parquet(self, path, index=False):
        Path(path).write_bytes(b"PAR1")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    bins, frames, meta = _tiny_tables()
    out = write_tables(bins, frames, meta, tmp_path / "r", fmt="parquet")
    assert (out / "bins.parquet").exists()
    assert (out / "frames.parquet").exists()
    assert (out / "meta.json").exists()


def test_write_tables_parquet_importerror_and_auto_fallback(tmp_path,
                                                            monkeypatch):
    def boom(self, *a, **k):
        raise ImportError("no parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    bins, frames, meta = _tiny_tables()
    with pytest.raises(ImportError):
        write_tables(bins, frames, meta, tmp_path / "r2", fmt="parquet")
    out = write_tables(bins, frames, meta, tmp_path / "r3", fmt="auto")
    assert (out / "bins.csv").exists()          # csv fallback
    assert not (out / "bins.parquet").exists()


def test_build_features_main_ok_with_skip(tmp_path):
    ts = "2099-07-04_12-00-00"
    _make_triplet(tmp_path, ts, n=4)
    out_root = tmp_path / "featout"
    rc = bf.main([
        "--triplet", str(tmp_path / "Synth" / ts),
        "--triplet", str(tmp_path / "Synth" / "2099-01-01_00-00-00"),  # skip
        "--out", str(out_root), "--format", "csv", "--no-imu",
        "--limit", "2",
        "--intrinsics", str(REPO_ROOT / "configs" / "intrinsics.yaml"),
        "--detection", str(REPO_ROOT / "configs" / "detection.yaml"),
        "--lat", "46.0", "--lon", "9.0", "--tz", "CET",
    ])
    assert rc == 0
    out_dir = out_root / f"Synth__{ts}"
    assert (out_dir / "bins.csv").exists()
    assert (out_dir / "frames.csv").exists()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["n_frames"] == 2


def test_build_features_main_requires_targets(monkeypatch):
    monkeypatch.setattr(bf, "list_triplets", lambda: [])
    with pytest.raises(SystemExit):
        bf.main(["--all"])                      # no resolvable triplets


def test_build_features_main_all_skipped_returns_1(tmp_path):
    rc = bf.main(["--triplet", str(tmp_path / "nope" / "2099"),
                  "--out", str(tmp_path / "o")])
    assert rc == 1


# ---------------------------------------------------------------------------
# build_targets: guard branches
# ---------------------------------------------------------------------------

def _mklabel(frame_id: str, bboxes: list[dict], clip_id: str | None = None,
             frame_idx: int = 0, frame_ts: str | None = None) -> Label:
    scene = frame_id.split("/")[0]
    return Label(frame_id=frame_id, scene=scene,
                 clip_id=clip_id or "/".join(frame_id.split("/")[:2]),
                 frame_idx=frame_idx, source="manual", audited=True,
                 bboxes=bboxes, obstacle_bins=[], frame_ts=frame_ts)


def test_label_bearings_px_skips_malformed_bboxes(intrinsics_real):
    K = np.asarray(intrinsics_real["fisheye"]["K"], float)
    lab = _mklabel("Synth/2099/000001",
                   [{"cls": "a"},                             # no xyxy
                    {"cls": "b", "xyxy": [0.1, 0.2, 0.3]},     # len 3
                    {"cls": "c", "xyxy": [0.4, 0.4, 0.6, 0.8]}])
    out = label_bearings_px(lab, 864, K)
    assert len(out) == 1 and out[0][2] == "c"


def test_label_mono_range_uses_confident_horizon(intrinsics_real):
    K = np.asarray(intrinsics_real["fisheye"]["K"], float)
    rng = label_mono_range((0.45, 0.55, 0.55, 0.85), K, (864, 648),
                           (0.0, 324.0, 0.9), 0.27)
    assert rng is not None and rng > 0.0


def test_build_targets_missing_frames_table_raises(tmp_path):
    d = tmp_path / "X__1"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps(
        {"clip_id": "X/1", "bin_edges_deg": [-5.0, 5.0]}))
    with pytest.raises(FileNotFoundError):
        build_targets_for_clip(d, [])


def test_build_targets_no_matching_labels_empty(corpus):
    root, _, _, clip_ids = corpus
    other = _mklabel("Other/1/000001",
                     [{"cls": "boat", "xyxy": [0.4, 0.5, 0.6, 0.8]}],
                     clip_id="Other/1", frame_idx=1)
    df = build_targets_for_clip(_feature_dir(root, clip_ids[0]), [other])
    assert df.empty


def test_build_targets_ts_keyed_out_of_range_and_bboxless(corpus, tmp_path):
    root, _, _, clip_ids = corpus
    ts = CORPUS_TS[0]
    recs = [
        # ts-keyed label resolving to frame 1 via the frames table
        {"frame_id": f"Synth/{ts}/ts=12-00-01.0", "scene": "Synth",
         "audited": True, "source": "manual", "width": 864, "height": 648,
         "fisheye_bboxes": [{"cls": "boat",
                             "xyxy": [0.45, 0.55, 0.55, 0.80]}],
         "obstacle_bins_fisheye": [0]},
        # ts-keyed label with a timestamp this export never saw -> dropped
        {"frame_id": f"Synth/{ts}/ts=23-59-59.9", "scene": "Synth",
         "audited": True, "source": "manual", "width": 864, "height": 648,
         "fisheye_bboxes": [{"cls": "boat",
                             "xyxy": [0.45, 0.55, 0.55, 0.80]}],
         "obstacle_bins_fisheye": [0]},
        # index beyond the export's frame range -> dropped
        {"frame_id": f"Synth/{ts}/000999", "scene": "Synth",
         "audited": True, "source": "manual", "width": 864, "height": 648,
         "fisheye_bboxes": [{"cls": "boat",
                             "xyxy": [0.45, 0.55, 0.55, 0.80]}],
         "obstacle_bins_fisheye": [0]},
        # bbox-less explicit-empty frame -> kept as all-negative bins
        {"frame_id": f"Synth/{ts}/000000", "scene": "Synth",
         "audited": True, "source": "manual", "width": 864, "height": 648,
         "fisheye_bboxes": [], "obstacle_bins_fisheye": []},
    ]
    lab_path = tmp_path / "edge_labels.jsonl"
    lab_path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    df = build_targets_for_clip(_feature_dir(root, clip_ids[0]),
                                load_labels([lab_path]))
    assert set(df.frame_index.unique()) == {0, 1}
    centre1 = df[(df.frame_index == 1) & (df.bin_center_deg.abs() < 6)]
    assert (centre1.y_obstacle == 1).all()
    frame0 = df[df.frame_index == 0]
    assert (frame0.n_labels_frame == 0).all()
    # frame 0's centre-bin positive can only come from dilation (frame 1)
    centre0 = frame0[frame0.bin_center_deg.abs() < 6]
    assert (centre0.from_dilation == 1).all()


def test_build_targets_main_full_and_filters(corpus):
    root, _, lab_one, clip_ids = corpus
    rc = bt.main(["--features", str(root), "--labels", str(lab_one),
                  "--dilation-frames", "7", "--tolerance-deg", "5.0",
                  "--max-relevant-range", "30"])
    # clip 1 written, clip 2 has no labels in this file -> skipped
    assert rc == 0
    assert (_feature_dir(root, clip_ids[0]) / "targets.csv").exists()

    rc2 = bt.main(["--features", str(root), "--labels", str(lab_one),
                   "--clip", clip_ids[0]])
    assert rc2 == 0


def test_build_targets_main_no_labels_returns_1(corpus, tmp_path):
    root, _, _, _ = corpus
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert bt.main(["--features", str(root), "--labels", str(empty)]) == 1


def test_build_targets_main_default_labels_no_features(tmp_path):
    (tmp_path / "nofeat").mkdir()
    # Without --labels the metrics defaults load (may be empty in CI);
    # either way nothing matches an empty features root -> rc 1.
    assert bt.main(["--features", str(tmp_path / "nofeat")]) == 1


# ---------------------------------------------------------------------------
# evaluate: loader edges, metrics edges, report, CLI
# ---------------------------------------------------------------------------

def test_load_dataset_skips_incomplete_dirs(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "stray.txt").write_text("x")
    (root / "nometa").mkdir()
    d1 = root / "A__1"
    d1.mkdir()
    (d1 / "meta.json").write_text(json.dumps({"clip_id": "A/1"}))
    d2 = root / "B__2"
    d2.mkdir()
    (d2 / "meta.json").write_text(json.dumps({"clip_id": "B/2"}))
    pd.DataFrame({"clip_id": ["B/2"], "frame_index": [0]}
                 ).to_csv(d2 / "bins.csv", index=False)
    d3 = root / "C__3"
    d3.mkdir()
    (d3 / "meta.json").write_text(json.dumps({"clip_id": "C/3"}))
    pd.DataFrame({"clip_id": ["C/3"], "frame_index": [0], "bin_index": [0]}
                 ).to_csv(d3 / "bins.csv", index=False)
    pd.DataFrame({"clip_id": ["C/3"], "frame_index": [0], "scene": ["C"]}
                 ).to_csv(d3 / "frames.csv", index=False)

    assert load_dataset(root).empty                    # no targets anywhere
    assert load_dataset(root, require_targets=False,
                        clips=["Z/9"]).empty           # clip filter miss


def test_average_precision_no_positives_is_nan():
    assert np.isnan(average_precision(np.zeros(4),
                                      np.array([0.1, 0.2, 0.3, 0.4])))


def test_evaluate_scores_openwater_fp_and_person_report():
    df = pd.DataFrame({
        "clip_id": ["OpenWater/c1"] * 4 + ["Boats/c2"] * 4,
        "frame_index": [0, 0, 1, 1] * 2,
        "scene": ["OpenWater"] * 4 + ["Boats"] * 4,
        "sun_elevation_deg": [30.0, 30.0, -10.0, -10.0] * 2,
        "radar_min_range_m": [2.0, np.nan, 8.0, np.nan] * 2,
        "min_range_m": [2.0, np.nan, 8.0, 20.0] * 2,
        "score_x": [0.9, 0.1, 0.7, 0.2] * 2,
        "y_nav": [1, 0, 0, 0, 1, 0, 1, 0],
        "label_classes": ["person", "", "boat", ""] * 2,
        "audited": [1, 0, 1, 0] * 2,
    })
    res = evaluate_scores(df, "score_x", "y_nav")
    assert "openwater_fp_per_frame" in res
    assert res["person_rows"] == 2
    assert 0.0 <= res["person_recall@0.5"] <= 1.0
    report = render_report({"score_x": res}, None, "synthetic")
    assert "person-class recall" in report
    assert "open-water FP bins/frame" in report


def test_fmt_non_float_passthrough():
    assert _fmt(3) == "3"
    assert _fmt("x") == "x"
    assert _fmt(float("nan")) == "—"


def test_evaluate_main_end_to_end(corpus, tmp_path):
    root, _, _, _ = corpus
    out = tmp_path / "rep.md"
    rc = ev.main(["--features", str(root), "--out", str(out)])
    assert rc == 0
    assert out.exists() and out.with_suffix(".json").exists()
    text = out.read_text()
    assert "score_legacy" in text
    assert "Blocked-bin rate per frame" in text


def test_evaluate_main_empty_features_returns_1(tmp_path):
    (tmp_path / "emptyroot").mkdir()
    assert ev.main(["--features", str(tmp_path / "emptyroot")]) == 1


# ---------------------------------------------------------------------------
# models: sklearn present/absent + serialization guard
# ---------------------------------------------------------------------------

def test_fit_gbt_returns_none_without_sklearn(monkeypatch):
    _block_sklearn(monkeypatch)
    assert fit_gbt(pd.DataFrame({"radar_n_points": [1.0]}),
                   np.array([1.0])) is None


def test_fit_gbt_and_predict_with_fake_sklearn(monkeypatch):
    _fake_sklearn(monkeypatch)
    df = pd.DataFrame({
        "radar_n_points": [1.0, 2.0, 3.0, 0.0],
        "seg_obstacle_frac": [np.nan] * 4,     # all-NaN -> neutralized
        "bin_center_deg": [-10.0, 0.0, 10.0, 20.0],
    })
    y = np.array([0.0, 1.0, 1.0, 0.0])
    clf = fit_gbt(df, y, seed=3)
    assert clf is not None
    assert "seg_obstacle_frac" in clf._fusion_allnan_cols
    p = gbt_predict(clf, df)
    assert p.shape == (4,)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_gated_mixture_load_rejects_wrong_kind(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"kind": "something_else"}))
    with pytest.raises(ValueError):
        GatedMixtureScorer.load(p)


# ---------------------------------------------------------------------------
# train: CLI branches
# ---------------------------------------------------------------------------

def test_train_main_empty_features_returns_1(tmp_path):
    (tmp_path / "emptyroot").mkdir()
    assert tr.main(["--features", str(tmp_path / "emptyroot")]) == 1


def test_train_clip_holdout_fallback_without_sklearn(corpus, tmp_path,
                                                     monkeypatch):
    root, _, _, _ = corpus
    _block_sklearn(monkeypatch)
    monkeypatch.setattr(tr, "RESULTS_DIR", tmp_path / "res")
    monkeypatch.setattr(tr, "MODELS_DIR", tmp_path / "mod")
    rc = tr.main(["--features", str(root), "--tag", "hold", "--seeds", "0"])
    assert rc == 0
    report = (tmp_path / "res" / "bakeoff_hold.md").read_text()
    assert "clip-holdout" in report            # all-train split fell back
    assert "SKIPPED" in report                 # v1b skipped without sklearn
    # tagged runs write tagged artefacts; the live v1a name is reserved
    assert (tmp_path / "mod" / "fusion_scorer_hold.json").exists()
    assert not (tmp_path / "mod" / "fusion_scorer_v1a.json").exists()


def test_train_v1b_with_fake_sklearn(corpus, tmp_path, monkeypatch):
    # sklearn is faked below, but saving the bundle needs the real joblib
    # (optional dependency, absent on CI): skip rather than fail there.
    pytest.importorskip("joblib")
    root, _, _, _ = corpus
    _fake_sklearn(monkeypatch)
    monkeypatch.setattr(tr, "RESULTS_DIR", tmp_path / "res")
    monkeypatch.setattr(tr, "MODELS_DIR", tmp_path / "mod")
    rc = tr.main(["--features", str(root), "--tag", "gbt", "--seeds", "0"])
    assert rc == 0
    report = (tmp_path / "res" / "bakeoff_gbt.md").read_text()
    assert "v1b GBT (mean of seeds)" in report
    assert "v1b trained" in report


def test_train_single_clip_frame_tail_holdout(tmp_path, intrinsics_real,
                                              detection_real, monkeypatch):
    ts = "2099-07-05_12-00-00"
    trip = _make_triplet(tmp_path, ts, n=6)
    bins_df, frames_df, meta = build_for_triplet(
        trip, intrinsics_real, detection_real, use_imu=False)
    root = tmp_path / "feat1"
    write_tables(bins_df, frames_df, meta, root, fmt="csv")
    _force_split(root)

    lab = tmp_path / "lab.jsonl"
    lab.write_text("\n".join(json.dumps(r) for r in
                             _label_recs(ts, frames=range(6))) + "\n")
    d = next(p for p in root.iterdir() if p.is_dir())
    build_targets_for_clip(d, load_labels([lab])).to_csv(
        d / "targets.csv", index=False)

    _block_sklearn(monkeypatch)
    monkeypatch.setattr(tr, "RESULTS_DIR", tmp_path / "res")
    monkeypatch.setattr(tr, "MODELS_DIR", tmp_path / "mod")
    rc = tr.main(["--features", str(root), "--tag", "single",
                  "--seeds", "0"])
    assert rc == 0
    report = (tmp_path / "res" / "bakeoff_single.md").read_text()
    assert "frame-tail" in report and "LEAKY" in report
