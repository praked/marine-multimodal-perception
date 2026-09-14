"""Fusion scorer models: the v1a/v1b bake-off pair + threat composition.

v1a `GatedMixtureScorer` (AuthorTwo's formulation, the default ship candidate):

    p(bin) = sigmoid( sum_s w_s(context) * e_s  +  b(context) )
    w_s(context) = softplus(u_s + v_s . context)   >= 0 by construction

Pure numpy (trains in seconds, runs on the Pi in microseconds, serializes
to JSON). The learned w_s(context) curves ARE the deliverable: "how much
does each sensor contribute at night" is read straight off the gate.

v1b `GBTScorer`: unconstrained HistGradientBoosting on the raw tabular
columns: the upper bound on what non-additive structure (agreement
bonuses, vetoes, cross-sensor gating) buys at the same features. Needs
scikit-learn (optional dep, laptop-only); absent -> None with a message.

Shared: `prepare_matrix` (bins+frames+targets -> X/e/c arrays with the
retired-blob columns excluded), `IsotonicCalibrator` (PAV), and
`threat_from_score` (urgency composition; severity table is interim policy
policy placeholder; see configs/fusion_model.yaml).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scripts.utils.datasets import REPO_ROOT

FUSION_MODEL_CONFIG = REPO_ROOT / "configs" / "fusion_model.yaml"

SENSORS = ("fisheye", "thermal", "radar")

# Context features for the gate (v1a): all computable on the Pi offline.
# Order matters (serialized with the model). abs_bearing lets the gate
# learn FOV-edge falloff; range_norm puts distance INSIDE the gate
# (per-sensor reliability envelopes: plan §1.2 mechanism 1).
CONTEXT_COLUMNS = (
    "sun_elevation_norm",     # sun_elevation_deg / 90
    "luminance_norm",         # luminance_mean / 255
    "is_dark",                # bool
    "thermal_quality_ok",     # bool
    "seg_available",          # bool
    "imu_available",          # bool
    "abs_bearing_norm",       # |bin_center| / 55
    "range_norm",             # min(bin range evidence, cap)/cap; 1 = far/unknown
)

# Optional per-bin target-motion context (bins schema v3, motion.targets
# tracker). Selected via fusion_model.yaml `context.motion: true` or
# train.py --motion-context; the artefact records its own column list, so
# 8-column and 11-column models coexist (context_matrix builds whatever
# the caller asks for).
MOTION_CONTEXT_COLUMNS = (
    "target_present",         # bool: a confirmed track occupies the bin
    "target_closing_norm",    # closing speed / 1 m/s (Doppler window), NaN->0
    "target_cpa_norm",        # CPA / cpa_cap clipped 0..1; 1 = far/none/unknown
)

# Raw tabular columns the GBT sees (NaN-native). The retired legacy blob
# channel (hit_fisheye + fisheye_det_* under detector=blob) is EXCLUDED
# from every model feature set: baseline reproduction only.
GBT_COLUMNS = (
    "seg_obstacle_frac", "seg_comp_count", "seg_comp_max_size_px",
    "free_space_m",
    "thermal_det_count", "thermal_det_max_size_px", "thermal_det_votes",
    "radar_n_points", "radar_min_range_m", "radar_median_range_m",
    "radar_max_snr_db", "radar_mean_snr_db", "radar_n_velocity",
    "per_bin_velocity_mps", "per_bin_ttc_s", "confirmed",
    "sun_elevation_deg", "luminance_mean", "is_dark",
    "thermal_quality_std", "thermal_quality_dyn_range", "thermal_quality_ok",
    "seg_available", "imu_available", "mmwave_available",
    "bin_center_deg",
)


def load_fusion_model_config(path: str | Path = FUSION_MODEL_CONFIG) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------

def merge_tables(bins_df: pd.DataFrame, frames_df: pd.DataFrame,
                 targets_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """bins x frames (x targets) -> one flat frame keyed (clip, frame, bin)."""
    frame_cols = [c for c in frames_df.columns
                  if c not in ("scene", "timestamp") or c == "scene"]
    df = bins_df.merge(frames_df[frame_cols], on=["clip_id", "frame_index"],
                       how="left", suffixes=("", "_frame"))
    if targets_df is not None and not targets_df.empty:
        df = df.merge(
            targets_df.drop(columns=["bin_center_deg"], errors="ignore"),
            on=["clip_id", "frame_index", "bin_index"], how="inner")
    return df


def evidence_matrix(df: pd.DataFrame, cfg: dict | None = None,
                    sensors: tuple[str, ...] | None = None) -> np.ndarray:
    """Per-sensor scalar evidence e_s (N x len(sensors)), the v1a squashings.

    Absent modality -> 0 evidence (the gate sees the availability flags in
    context, so "absent" and "present but empty" stay distinguishable).
    `sensors` defaults to the base trio; a loaded artefact passes its own
    serialised list, so 3- and 4-channel (typed-YOLO) models coexist. The
    "yolo" channel is the per-bin max typed-detection confidence (already
    0..1 -> no squashing; NaN/absent -> 0)."""
    sensors = tuple(sensors) if sensors is not None else SENSORS
    ev = (cfg or load_fusion_model_config()).get("evidence", {})
    seg_scale = float(ev.get("fisheye_seg_frac_scale", 0.05))
    th_scale = float(ev.get("thermal_count_scale", 2.0))
    ra_scale = float(ev.get("radar_points_scale", 5.0))
    chans = {
        "fisheye": lambda: np.tanh(
            df["seg_obstacle_frac"].fillna(0.0).to_numpy() / seg_scale),
        "thermal": lambda: np.tanh(
            df["thermal_det_count"].fillna(0.0).to_numpy() / th_scale),
        "radar": lambda: np.tanh(
            df["radar_n_points"].fillna(0.0).to_numpy() / ra_scale),
        "yolo": lambda: (df["yolo_max_conf"] if "yolo_max_conf" in df.columns
                         else pd.Series(0.0, index=df.index)
                         ).fillna(0.0).clip(0, 1).to_numpy(),
    }
    return np.column_stack([chans[s]() for s in sensors])


def context_matrix(df: pd.DataFrame, cfg: dict | None = None,
                   columns: tuple[str, ...] | None = None) -> np.ndarray:
    """Gate context c (N x len(columns)), NaN-free by construction.

    `columns` defaults to the module CONTEXT_COLUMNS (+ motion columns when
    cfg `context.motion` is truthy); a loaded artefact passes its own
    serialised list so old and motion-aware models coexist."""
    cfg = cfg or load_fusion_model_config()
    if columns is None:
        columns = tuple(CONTEXT_COLUMNS)
        if (cfg.get("context") or {}).get("motion"):
            columns = columns + MOTION_CONTEXT_COLUMNS
    ur = cfg.get("urgency", {})
    range_cap = float(ur.get("range_cap_m", 15.0))
    cpa_cap = float((ur.get("motion") or {}).get("cpa_cap_m", 5.0))
    sun = (df["sun_elevation_deg"].fillna(0.0) / 90.0).clip(-1, 1)
    lum = (df["luminance_mean"].fillna(127.5) / 255.0).clip(0, 1)
    rng = df["radar_min_range_m"].copy()
    if "min_range_m" in df.columns:
        rng = rng.fillna(df["min_range_m"])
    range_norm = (rng.fillna(range_cap).clip(0, range_cap) / range_cap)
    cols = {
        "sun_elevation_norm": sun,
        "luminance_norm": lum,
        "is_dark": df["is_dark"].fillna(False).astype(float),
        "thermal_quality_ok": df["thermal_quality_ok"].fillna(False).astype(float),
        "seg_available": df["seg_available"].fillna(False).astype(float),
        "imu_available": df["imu_available"].fillna(False).astype(float),
        "abs_bearing_norm": (df["bin_center_deg"].abs() / 55.0).clip(0, 2),
        "range_norm": range_norm,
    }
    if any(c in columns for c in MOTION_CONTEXT_COLUMNS):
        present = (df["target_present"] if "target_present" in df.columns
                   else pd.Series(False, index=df.index))
        closing = (df["target_closing_mps"] if "target_closing_mps" in df.columns
                   else pd.Series(float("nan"), index=df.index))
        cpa = (df["target_cpa_m"] if "target_cpa_m" in df.columns
               else pd.Series(float("nan"), index=df.index))
        cols["target_present"] = present.fillna(False).astype(float)
        cols["target_closing_norm"] = (closing.fillna(0.0) / 1.0).clip(-1, 1)
        cols["target_cpa_norm"] = (cpa.fillna(cpa_cap).clip(0, cpa_cap)
                                   / cpa_cap)
    return np.column_stack([np.asarray(cols[c], dtype=np.float64)
                            for c in columns])


def prepare_matrix(df: pd.DataFrame, cfg: dict | None = None,
                   columns: tuple[str, ...] | None = None,
                   sensors: tuple[str, ...] | None = None,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(e, c, y) arrays from a merged table; y None without targets."""
    cfg = cfg or load_fusion_model_config()
    e = evidence_matrix(df, cfg, sensors=sensors)
    c = context_matrix(df, cfg, columns=columns)
    y = (df["y_nav"].to_numpy(dtype=np.float64)
         if "y_nav" in df.columns else None)
    return e, c, y


# ---------------------------------------------------------------------------
# v1a: gated mixture (pure numpy)
# ---------------------------------------------------------------------------

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


@dataclass
class GatedMixtureScorer:
    """p = sigmoid(sum_s softplus(u_s + v_s.c) * e_s + b0 + b.c)."""

    u: np.ndarray          # (S,)
    v: np.ndarray          # (S, C)
    b0: float
    b: np.ndarray          # (C,)
    sensors: tuple[str, ...] = SENSORS
    context_columns: tuple[str, ...] = CONTEXT_COLUMNS
    train_info: dict = field(default_factory=dict)
    # Fitted on TRAIN predictions (train.py): maps the class-balanced raw
    # score back to a true-base-rate probability. Without it the deployed
    # p_obstacle floor sits near the balanced prior (~0.5), not the ~15%
    # real positive rate — the 2026-08-22 open-water misread.
    calibrator: "IsotonicCalibrator | None" = None

    # -- inference ---------------------------------------------------------

    def gate_weights(self, c: np.ndarray) -> np.ndarray:
        """w_s(context), (N x S): the interpretability deliverable."""
        return _softplus(self.u[None, :] + c @ self.v.T)

    def predict_proba(self, e: np.ndarray, c: np.ndarray) -> np.ndarray:
        w = self.gate_weights(c)
        logit = (w * e).sum(axis=1) + self.b0 + c @ self.b
        return _sigmoid(logit)

    def predict_proba_calibrated(self, e: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Calibrated probability when the artefact carries a calibrator;
        raw class-balanced score otherwise (pre-2026-08-22 artefacts)."""
        p = self.predict_proba(e, c)
        return self.calibrator.transform(p) if self.calibrator is not None else p

    # -- training ----------------------------------------------------------

    @classmethod
    def fit(cls, e: np.ndarray, c: np.ndarray, y: np.ndarray, *,
            seed: int = 0, epochs: int = 300, lr: float = 0.05,
            l2: float = 1e-3, class_balance: bool = True,
            ) -> "GatedMixtureScorer":
        """Full-batch Adam on BCE. Data is small (<=1e6 rows) and the
        parameter count tiny (S*(C+1) + C + 1), so this is seconds."""
        rng = np.random.default_rng(seed)
        n, S, C = len(y), e.shape[1], c.shape[1]
        u = rng.normal(0.0, 0.1, S)
        v = rng.normal(0.0, 0.1, (S, C))
        b0 = float(np.log(max(y.mean(), 1e-3) / max(1 - y.mean(), 1e-3)))
        b = np.zeros(C)
        # Positive-class weighting: bin-level positives are sparse.
        pos_w = ((n - y.sum()) / max(y.sum(), 1.0)) if class_balance else 1.0
        sample_w = np.where(y > 0.5, pos_w, 1.0)
        sample_w = sample_w / sample_w.mean()

        params = [u, v, np.array([b0]), b]
        m = [np.zeros_like(p) for p in params]
        s_ = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for t in range(1, epochs + 1):
            z = u[None, :] + c @ v.T                 # (N, S)
            w = _softplus(z)
            logit = (w * e).sum(axis=1) + params[2][0] + c @ b
            p = _sigmoid(logit)
            g_logit = (p - y) * sample_w / n         # dBCE/dlogit
            gw = g_logit[:, None] * e                # (N, S)
            gz = gw * _sigmoid(z)                    # softplus' = sigmoid
            grads = [
                gz.sum(axis=0) + l2 * u,
                gz.T @ c + l2 * v,
                np.array([g_logit.sum()]),
                c.T @ g_logit + l2 * b,
            ]
            for p_, g_, m_, v_ in zip(params, grads, m, s_):
                m_ *= beta1
                m_ += (1 - beta1) * g_
                v_ *= beta2
                v_ += (1 - beta2) * g_ * g_
                mh = m_ / (1 - beta1 ** t)
                vh = v_ / (1 - beta2 ** t)
                p_ -= lr * mh / (np.sqrt(vh) + eps)
        return cls(u=params[0], v=params[1], b0=float(params[2][0]),
                   b=params[3],
                   train_info={"seed": seed, "epochs": epochs, "lr": lr,
                               "l2": l2, "n": int(n),
                               "pos_rate": float(np.mean(y))})

    # -- serialization -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        obj = {"kind": "gated_mixture_v1a",
               "sensors": list(self.sensors),
               "context_columns": list(self.context_columns),
               "u": self.u.tolist(), "v": self.v.tolist(),
               "b0": self.b0, "b": self.b.tolist(),
               "train_info": self.train_info}
        if self.calibrator is not None:
            # PAV keeps one knot per training row; thin to quantile knots
            # (monotone map is piecewise-constant-ish: 512 knots reproduce
            # it to <1e-3 while keeping the artefact small).
            x, yh = self.calibrator.x, self.calibrator.yhat
            if len(x) > 512:
                idx = np.unique(np.linspace(0, len(x) - 1, 512).astype(int))
                x, yh = x[idx], yh[idx]
            obj["calibrator"] = {"x": x.tolist(), "yhat": yh.tolist()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GatedMixtureScorer":
        with open(path) as fh:
            obj = json.load(fh)
        if obj.get("kind") != "gated_mixture_v1a":
            raise ValueError(f"not a gated-mixture model: {path}")
        cal = obj.get("calibrator")
        return cls(u=np.asarray(obj["u"], float),
                   v=np.asarray(obj["v"], float),
                   b0=float(obj["b0"]), b=np.asarray(obj["b"], float),
                   sensors=tuple(obj["sensors"]),
                   context_columns=tuple(obj["context_columns"]),
                   train_info=obj.get("train_info", {}),
                   calibrator=(IsotonicCalibrator(
                       x=np.asarray(cal["x"], float),
                       yhat=np.asarray(cal["yhat"], float))
                       if cal else None))


# ---------------------------------------------------------------------------
# v1b: GBT (optional scikit-learn)
# ---------------------------------------------------------------------------

def fit_gbt(df: pd.DataFrame, y: np.ndarray, seed: int = 0):
    """HistGradientBoostingClassifier on GBT_COLUMNS (NaN-native), or None
    when scikit-learn isn't installed (optional laptop-only dep)."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        return None
    X = df.reindex(columns=list(GBT_COLUMNS)).astype(float)
    # Columns that are ENTIRELY NaN in training (e.g. radar SNR on a
    # pre-2026-07-09 corpus) crash sklearn's binner and carry no signal
    # for this model instance: neutralize to 0 on both fit and predict
    # (partial NaN stays NaN: HistGradientBoosting handles it natively).
    allnan = [c for c in X.columns if X[c].isna().all()]
    if allnan:
        X[allnan] = 0.0
    clf = HistGradientBoostingClassifier(
        random_state=seed, class_weight="balanced", max_depth=4,
        max_iter=200, learning_rate=0.1)
    clf.fit(X.to_numpy(), y)
    clf._fusion_allnan_cols = allnan
    return clf


def gbt_predict(clf, df: pd.DataFrame, columns=None) -> np.ndarray:
    """`columns` defaults to the module GBT_COLUMNS; a bundle passes its own
    recorded column set so subset models (e.g. radar-only) predict on the
    columns they were fit on (2026-09-03)."""
    cols = list(columns) if columns is not None else list(GBT_COLUMNS)
    X = df.reindex(columns=cols).astype(float)
    for c in getattr(clf, "_fusion_allnan_cols", []):
        X[c] = 0.0
    return clf.predict_proba(X.to_numpy())[:, 1]


@dataclass
class GBTScorerBundle:
    """Deployable v1b: the seed ENSEMBLE (predict = mean over seeds, exactly
    what the bake-off evaluates) + isotonic calibrator, serialized with
    joblib. Earned its deployment story 2026-09-02: AP 0.883 vs incumbent
    0.731 on the audited-night holdout (teacher_comparison.md §4e ff).

    Interface parity with GatedMixtureScorer where fusion.py needs it
    (`sensors`, `context_columns`) plus the df-native predict path — the
    pipeline's `sdf` (bin_rows_for_frame) carries GBT_COLUMNS natively, so
    no e/c factoring happens for this scorer.
    """

    clfs: list = field(default_factory=list)
    columns: tuple[str, ...] = GBT_COLUMNS
    sensors: tuple[str, ...] = SENSORS
    context_columns: tuple[str, ...] = CONTEXT_COLUMNS
    train_info: dict = field(default_factory=dict)
    calibrator: "IsotonicCalibrator | None" = None
    kind: str = "gbt_v1b"

    def predict_proba_df(self, df: pd.DataFrame) -> np.ndarray:
        return np.mean(np.stack([gbt_predict(c, df, self.columns)
                                 for c in self.clfs]), axis=0)

    def predict_proba_calibrated_df(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba_df(df)
        return self.calibrator.transform(p) if self.calibrator is not None else p

    def save(self, path: str | Path) -> None:
        import joblib
        import sklearn
        obj = {"kind": "gbt_v1b", "clfs": self.clfs,
               "columns": list(self.columns), "sensors": list(self.sensors),
               "context_columns": list(self.context_columns),
               "train_info": {**self.train_info,
                              "sklearn_version": getattr(sklearn, "__version__", "unknown")}}
        if self.calibrator is not None:
            x, yh = self.calibrator.x, self.calibrator.yhat
            if len(x) > 512:
                idx = np.unique(np.linspace(0, len(x) - 1, 512).astype(int))
                x, yh = x[idx], yh[idx]
            obj["calibrator"] = {"x": np.asarray(x), "yhat": np.asarray(yh)}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "GBTScorerBundle":
        import warnings
        import joblib
        import sklearn
        obj = joblib.load(path)
        if obj.get("kind") != "gbt_v1b":
            raise ValueError(f"not a v1b GBT bundle: {path}")
        saved_ver = obj.get("train_info", {}).get("sklearn_version")
        if saved_ver and saved_ver != sklearn.__version__:
            warnings.warn(
                f"v1b bundle trained under scikit-learn {saved_ver}, "
                f"running {sklearn.__version__} — re-train or pin on drift",
                stacklevel=2)
        cal = obj.get("calibrator")
        return cls(clfs=list(obj["clfs"]), columns=tuple(obj["columns"]),
                   sensors=tuple(obj["sensors"]),
                   context_columns=tuple(obj["context_columns"]),
                   train_info=obj.get("train_info", {}),
                   calibrator=(IsotonicCalibrator(
                       x=np.asarray(cal["x"], float),
                       yhat=np.asarray(cal["yhat"], float)) if cal else None))


# ---------------------------------------------------------------------------
# Probability calibration (isotonic / PAV, pure numpy)
# ---------------------------------------------------------------------------

@dataclass
class IsotonicCalibrator:
    """Monotone score->probability map fit with pool-adjacent-violators."""

    x: np.ndarray
    yhat: np.ndarray

    @classmethod
    def fit(cls, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        order = np.argsort(scores, kind="stable")
        xs, ys = np.asarray(scores)[order], np.asarray(y, float)[order]
        vals = ys.copy()
        wts = np.ones_like(vals)
        # classic PAV: merge adjacent violating blocks
        i = 0
        idx_start = list(range(len(vals)))
        v, w, starts = list(vals), list(wts), list(idx_start)
        i = 0
        while i < len(v) - 1:
            if v[i] <= v[i + 1] + 1e-12:
                i += 1
                continue
            merged_w = w[i] + w[i + 1]
            merged_v = (v[i] * w[i] + v[i + 1] * w[i + 1]) / merged_w
            v[i:i + 2] = [merged_v]
            w[i:i + 2] = [merged_w]
            starts[i + 1:i + 2] = []
            i = max(i - 1, 0)
        fitted = np.empty_like(ys)
        bounds = starts + [len(ys)]
        for bi in range(len(v)):
            fitted[bounds[bi]:bounds[bi + 1]] = v[bi]
        return cls(x=xs, yhat=fitted)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.interp(scores, self.x, self.yhat)


# ---------------------------------------------------------------------------
# Threat composition
# ---------------------------------------------------------------------------

def _finite(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def urgency(range_m: float | None, ttc_s: float | None,
            severity: float = 1.0, cfg: dict | None = None,
            cpa_m: float | None = None,
            t_cpa_s: float | None = None) -> float:
    """max(range term, ttc term[, CPA term]) scaled by severity, floored
    for unknowns.

    Severity values are UNREVIEWED policy placeholders (fusion_model.yaml
    urgency.severity); interim policy per AuthorTwo 2026-08-22 (humans > living
    beings > damage-limiting order) — AuthorOne sanity pass at D.2.

    The CPA term (fusion_model.yaml urgency.motion, default OFF) is the
    geometry channel for crossing targets: the range and TTC terms are both
    radial, so a crossing boat that will pass close reads as low-urgency
    until it is nearly upon us. When enabled and the caller supplies a
    per-target CPA prediction (motion.targets tracker), the term is
    (1 - cpa/cpa_cap) * (1 - t_cpa/t_cpa_cap): high only when the predicted
    miss distance is small AND soon. It joins the max(), so it can only
    raise urgency, never lower it: callers not passing CPA are unchanged.
    """
    ur = (cfg or load_fusion_model_config()).get("urgency", {})
    r_cap = float(ur.get("range_cap_m", 15.0))
    t_cap = float(ur.get("ttc_cap_s", 60.0))
    floor = float(ur.get("unknown_floor", 0.3))
    terms = []
    if _finite(range_m):
        terms.append(max(0.0, min(1.0, 1.0 - float(range_m) / r_cap)))
    if _finite(ttc_s):
        terms.append(max(0.0, min(1.0, 1.0 - float(ttc_s) / t_cap)))
    mo = ur.get("motion", {}) or {}
    if mo.get("enabled", False) and _finite(cpa_m) and _finite(t_cpa_s):
        cpa_cap = float(mo.get("cpa_cap_m", 5.0))
        tc_cap = float(mo.get("t_cpa_cap_s", 60.0))
        near = max(0.0, min(1.0, 1.0 - float(cpa_m) / cpa_cap))
        soon = max(0.0, min(1.0, 1.0 - float(t_cpa_s) / tc_cap))
        terms.append(near * soon)
    base = max(terms) if terms else floor
    return float(severity) * base


def severity_for_classes(classes: str | None, cfg: dict | None = None) -> float:
    """Max severity over a '|'-joined class string; unknown/empty -> 1.0
    (neutral, never DOWN-weight for lack of class evidence)."""
    table = ((cfg or load_fusion_model_config()).get("urgency", {})
             .get("severity", {}) or {})
    if not classes:
        return float(table.get("unknown", 1.0))
    sevs = [float(table.get(c, table.get("unknown", 1.0)))
            for c in str(classes).split("|") if c]
    return max(sevs) if sevs else float(table.get("unknown", 1.0))


def threat_from_score(p: np.ndarray, range_m, ttc_s,
                      severity: np.ndarray | float = 1.0,
                      cfg: dict | None = None,
                      cpa_m=None, t_cpa_s=None) -> np.ndarray:
    """threat = p * urgency, elementwise; range_m/ttc_s arrays may hold NaN.

    `cpa_m`/`t_cpa_s` are optional per-bin CPA arrays (NaN = no tracked
    target in the bin) from the motion.targets tracker; omitted -> the
    output is unchanged (and the urgency motion term is additionally
    config-gated, see `urgency`).
    """
    cfg = cfg or load_fusion_model_config()
    p = np.asarray(p, dtype=np.float64)
    r = np.asarray(range_m, dtype=np.float64)
    t = np.asarray(ttc_s, dtype=np.float64)
    nan = np.full(p.shape, np.nan)
    c = np.asarray(cpa_m if cpa_m is not None else nan, dtype=np.float64)
    tc = np.asarray(t_cpa_s if t_cpa_s is not None else nan, dtype=np.float64)
    sev = np.broadcast_to(np.asarray(severity, dtype=np.float64), p.shape)
    out = np.empty_like(p)
    for i in range(len(p)):
        out[i] = p[i] * urgency(
            None if math.isnan(r[i]) else float(r[i]),
            None if math.isnan(t[i]) else float(t[i]),
            float(sev[i]), cfg,
            cpa_m=None if math.isnan(c[i]) else float(c[i]),
            t_cpa_s=None if math.isnan(tc[i]) else float(tc[i]))
    return out


def load_any_scorer(path: str | Path) -> Any:
    """Load a serialized scorer by format: .joblib -> v1b GBT bundle
    (deployable since 2026-09-02: beats the incumbent by +0.15 AP on the
    audited-night holdout), .json -> v1a gated mixture."""
    if str(path).endswith((".joblib", ".pkl")):
        return GBTScorerBundle.load(path)
    return GatedMixtureScorer.load(path)
