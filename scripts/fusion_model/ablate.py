"""Ablations of the v1b GBT scorer on one feature table (paper Table III).

Same split rule as `train.py` (frozen split, else hash-order clip holdout),
same seeds, same evaluator; each variant re-trains the GBT on a modified
column set or evaluates the full model on modified INPUTS:

  full                 GBT_COLUMNS, train-fit calibration (the first bundle)
  (all column-subset variants below are OOF-calibrated like full_oofcal)
  no_context           evidence only: sun/luminance/dark/thermal-quality/
                       availability/bin-centre columns dropped
  rgb_only             segmentation + free-space evidence (+ its context)
  thermal_only         thermal-blob evidence (+ thermal quality)
  radar_only           radar evidence only (trained that way)
  full_typed           GBT_COLUMNS + per-bin typed-detector confidence
                       (features built with --det-root <detector output>)
  full_motion          GBT_COLUMNS + the per-bin target-motion columns
                       (features built with motion.targets on)
  full@radar_input     the FULL model fed radar+IMU rows only — camera
                       columns NaN, seg/thermal availability 0 — i.e. what the
                       deployed tail-mode shadow actually gives it
  full_uncalibrated    full model without the isotonic map
  full_oofcal          full model with an isotonic map fit on out-of-fold
                       (clip-fold) predictions instead of train predictions
  incumbent            score_legacy (n_hits/3)
  v1a_gate             the non-negative gated mixture (context columns)

Metrics per variant: AP, best F1, ECE, Brier, person recall@0.5, and the same
by daypart (sun elevation: day / dusk = |elev| < twilight / night).

    python -m scripts.fusion_model.ablate --features data/features_audited --tag audited
    python -m scripts.fusion_model.ablate --features data/features_ablation_noassoc \
        --tag noassoc --variants full incumbent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fusion_model import models as M  # noqa: E402
from scripts.fusion_model.evaluate import (  # noqa: E402
    daypart,
    evaluate_scores,
    load_dataset,
)
from scripts.fusion_model.models import (  # noqa: E402
    CONTEXT_COLUMNS,
    GBT_COLUMNS,
    SENSORS,
    GatedMixtureScorer,
    IsotonicCalibrator,
    fit_gbt,
    gbt_predict,
    load_fusion_model_config,
    prepare_matrix,
)

RESULTS_DIR = REPO_ROOT / "results" / "fusion_model"

CONTEXT_LIKE = ("sun_elevation_deg", "luminance_mean", "is_dark",
                "thermal_quality_std", "thermal_quality_dyn_range",
                "thermal_quality_ok", "seg_available", "imu_available",
                "mmwave_available", "bin_center_deg")
RGB_EVIDENCE = ("seg_obstacle_frac", "seg_comp_count", "seg_comp_max_size_px",
                "free_space_m")
THERMAL_EVIDENCE = ("thermal_det_count", "thermal_det_max_size_px",
                    "thermal_det_votes")
RADAR_EVIDENCE = ("radar_n_points", "radar_min_range_m", "radar_median_range_m",
                  "radar_max_snr_db", "radar_mean_snr_db", "radar_n_velocity",
                  "per_bin_velocity_mps", "per_bin_ttc_s", "confirmed")

VARIANT_COLUMNS = {
    "full": tuple(GBT_COLUMNS),
    "no_context": tuple(c for c in GBT_COLUMNS if c not in CONTEXT_LIKE),
    "rgb_only": RGB_EVIDENCE + ("sun_elevation_deg", "luminance_mean",
                                "is_dark", "seg_available", "bin_center_deg"),
    "thermal_only": THERMAL_EVIDENCE + ("thermal_quality_std",
                                        "thermal_quality_dyn_range",
                                        "thermal_quality_ok", "bin_center_deg"),
    "radar_only": RADAR_EVIDENCE + ("mmwave_available", "bin_center_deg"),
    # Tail-mode candidate: `confirmed` needs a camera (radar<->camera
    # association) and is always False beside capture, so a model trained
    # with it sees a train/serve mismatch on that column.
    "radar_only_noconfirm": tuple(c for c in RADAR_EVIDENCE if c != "confirmed")
    + ("mmwave_available", "bin_center_deg"),
    # needs features built with a typed detector's boxes in --det-root
    "full_typed": tuple(GBT_COLUMNS) + ("yolo_available", "yolo_max_conf"),
    # needs features built with motion.targets enabled (else all-NaN -> 0)
    "full_motion": tuple(GBT_COLUMNS) + (
        "target_present", "target_range_m", "target_closing_mps",
        "target_v_tangential_mps", "target_speed_mps", "target_cpa_m",
        "target_t_cpa_s"),
}
ALL_VARIANTS = ("incumbent", "v1a_gate", "full", "full_uncalibrated",
                "full_oofcal", "no_context", "rgb_only", "thermal_only",
                "radar_only", "radar_only_noconfirm", "full@radar_input")


def split(df: pd.DataFrame, holdout: str | None):
    """train.py's rule, verbatim in effect. `holdout` may also be
    "clips:<substr>,<substr>" for an explicit clip_id-substring holdout."""
    name = holdout or "val"
    if name.startswith("clips:"):
        subs = [x for x in name[6:].split(",") if x]
        mask = df["clip_id"].apply(lambda c: any(x in str(c) for x in subs))
        return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True), f"clips({','.join(subs)})"
    mask = df["split"] == name
    if not mask.any() or mask.all():
        clips = sorted(df["clip_id"].unique())
        n_hold = max(1, len(clips) // 5)
        held = set(clips[-n_hold:])
        mask = df["clip_id"].isin(held)
        name = f"clip-holdout({len(held)} clips: {sorted(held)[0]} .. {sorted(held)[-1]})"
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True), name


def brier(y: np.ndarray, s: np.ndarray) -> float:
    m = np.isfinite(s)
    return float(np.mean((s[m] - y[m]) ** 2)) if m.any() else float("nan")


def metrics(df: pd.DataFrame, col: str, target: str, twl: float) -> dict:
    r = evaluate_scores(df, col, target, twilight_deg=twl)
    y = df[target].to_numpy(dtype=float)
    s = df[col].to_numpy(dtype=float)
    out = {"ap": r["average_precision"], "ece": r["calibration"]["ece"],
           "brier": brier(y, s),
           "best_f1": r["best_f1"]["f1"] if r["best_f1"] else float("nan"),
           "f1_at": r["best_f1"]["threshold"] if r["best_f1"] else None,
           "person_recall": r.get("person_recall@0.5"),
           "n": int(len(df)), "n_pos": int(y.sum()), "by_daypart": {}}
    dp = daypart(df["sun_elevation_deg"], twl)
    for name in ("day", "dusk", "night"):
        sub = df[dp == name]
        if len(sub) < 20 or sub[target].sum() == 0:
            continue
        rr = evaluate_scores(sub, col, target, twilight_deg=twl)
        out["by_daypart"][name] = {
            "n": int(len(sub)), "n_pos": int(sub[target].sum()),
            "ap": rr["average_precision"], "ece": rr["calibration"]["ece"],
            "brier": brier(sub[target].to_numpy(dtype=float),
                           sub[col].to_numpy(dtype=float)),
            "best_f1": rr["best_f1"]["f1"] if rr["best_f1"] else float("nan"),
            "person_recall": rr.get("person_recall@0.5")}
    return out


def fit_predict_gbt(train_df, eval_df, y_tr, cols, seeds):
    """Train/predict the GBT on a column subset (module constant swapped for
    the call: fit_gbt/gbt_predict read it at call time)."""
    saved = M.GBT_COLUMNS
    M.GBT_COLUMNS = tuple(cols)
    try:
        clfs = [fit_gbt(train_df, y_tr, seed=s) for s in seeds]
        if any(c is None for c in clfs):
            raise SystemExit("scikit-learn missing")
        p_tr = np.mean([gbt_predict(c, train_df) for c in clfs], axis=0)
        p_ev = np.mean([gbt_predict(c, eval_df) for c in clfs], axis=0)
    finally:
        M.GBT_COLUMNS = saved
    cal = IsotonicCalibrator.fit(p_tr, y_tr)
    return p_ev, cal.transform(p_ev)


def oof_calibrator(train_df, y_tr, cols, seeds, n_folds: int = 5):
    """Isotonic map fit on OUT-OF-FOLD predictions (folds = whole clips), so
    the map sees the model's honest confidence rather than its fit to its
    own training rows (train-fit maps were measured to HURT: ECE 0.227 ->
    0.321 on the audited holdout)."""
    clips = sorted(train_df["clip_id"].unique())
    folds = [set(clips[i::n_folds]) for i in range(n_folds)]
    oof = np.full(len(train_df), np.nan)
    saved = M.GBT_COLUMNS
    M.GBT_COLUMNS = tuple(cols)
    try:
        for held in folds:
            m = train_df["clip_id"].isin(held).to_numpy()
            if m.sum() == 0 or (~m).sum() == 0:
                continue
            clfs = [fit_gbt(train_df[~m], y_tr[~m], seed=s) for s in seeds]
            oof[m] = np.mean([gbt_predict(c, train_df[m]) for c in clfs], axis=0)
    finally:
        M.GBT_COLUMNS = saved
    ok = np.isfinite(oof)
    return IsotonicCalibrator.fit(oof[ok], y_tr[ok])


def radar_input_only(df: pd.DataFrame) -> pd.DataFrame:
    """What tail-mode shadow feeds the scorer: no camera pixels at all."""
    d = df.copy()
    for c in RGB_EVIDENCE + THERMAL_EVIDENCE + (
            "fisheye_det_count", "fisheye_det_max_size_px", "fisheye_det_votes",
            "luminance_mean", "thermal_quality_std", "thermal_quality_dyn_range"):
        if c in d.columns:
            d[c] = np.nan
    for c in ("seg_available", "thermal_quality_ok", "is_dark"):
        if c in d.columns:
            d[c] = 0
    d["hit_fisheye"] = 0
    d["hit_thermal"] = 0
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--features", default="data/features_audited")
    ap.add_argument("--target", default="y_nav", choices=["y_nav", "y_obstacle"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="*", default=list(ALL_VARIANTS))
    args = ap.parse_args(argv)

    cfg = load_fusion_model_config()
    twl = float(cfg.get("context", {}).get("twilight_deg", 6.0))
    df = load_dataset(args.features)
    if df.empty:
        print("no feature+target tables", file=sys.stderr)
        return 1
    train_df, eval_df, hold = split(df, args.holdout)
    y_tr = train_df[args.target].to_numpy(dtype=float)
    print(f"[data] train {len(train_df)} rows, eval {len(eval_df)} rows "
          f"(pos {int(eval_df[args.target].sum())}), holdout {hold}",
          file=sys.stderr)

    results: dict[str, dict] = {}
    ev = eval_df.copy()
    for v in args.variants:
        print(f"[variant] {v}", file=sys.stderr)
        if v == "incumbent":
            results[v] = metrics(ev, "score_legacy", args.target, twl)
        elif v == "v1a_gate":
            e_tr, c_tr, _ = prepare_matrix(train_df, cfg, columns=CONTEXT_COLUMNS,
                                           sensors=SENSORS)
            e_ev, c_ev, _ = prepare_matrix(ev, cfg, columns=CONTEXT_COLUMNS,
                                           sensors=SENSORS)
            tr_cfg = cfg.get("train", {})
            ms = [GatedMixtureScorer.fit(
                e_tr, c_tr, y_tr, seed=s, epochs=int(tr_cfg.get("epochs", 300)),
                lr=float(tr_cfg.get("learning_rate", 0.05)),
                l2=float(tr_cfg.get("l2", 1e-3))) for s in args.seeds]
            p = np.mean([m.predict_proba(e_ev, c_ev) for m in ms], axis=0)
            cal = IsotonicCalibrator.fit(
                np.mean([m.predict_proba(e_tr, c_tr) for m in ms], axis=0), y_tr)
            ev["_p"] = cal.transform(p)
            results[v] = metrics(ev, "_p", args.target, twl)
        elif v in ("full", "full_uncalibrated", "full_oofcal", "full@radar_input"):
            raw, cal = fit_predict_gbt(train_df, ev, y_tr, VARIANT_COLUMNS["full"],
                                       args.seeds)
            if v == "full":
                ev["_p"] = cal
            elif v == "full_uncalibrated":
                ev["_p"] = raw
            elif v == "full_oofcal":
                ev["_p"] = oof_calibrator(train_df, y_tr, VARIANT_COLUMNS["full"],
                                          args.seeds).transform(raw)
            else:
                raw2, cal2 = fit_predict_gbt(train_df, radar_input_only(ev), y_tr,
                                             VARIANT_COLUMNS["full"], args.seeds)
                ev["_p"] = cal2
            results[v] = metrics(ev, "_p", args.target, twl)
        elif v in VARIANT_COLUMNS:
            # every column-subset variant gets the same OOF calibration as the
            # reference row, so rows differ only in their columns
            raw, _ = fit_predict_gbt(train_df, ev, y_tr, VARIANT_COLUMNS[v],
                                     args.seeds)
            ev["_p"] = oof_calibrator(train_df, y_tr, VARIANT_COLUMNS[v],
                                      args.seeds).transform(raw)
            results[v] = metrics(ev, "_p", args.target, twl)
        else:
            print(f"unknown variant {v}", file=sys.stderr)
        r = results.get(v)
        if r:
            print(f"  AP {r['ap']:.3f}  F1 {r['best_f1']:.3f}  ECE {r['ece']:.3f}  "
                  f"Brier {r['brier']:.3f}  person R {r['person_recall']}",
                  file=sys.stderr)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"ablation_{args.tag}"
    out.with_suffix(".json").write_text(json.dumps(
        {"features": str(args.features), "target": args.target, "holdout": hold,
         "seeds": args.seeds, "n_train": len(train_df), "n_eval": len(eval_df),
         "results": results}, indent=1, default=float))
    lines = [f"# Scorer ablation ({args.tag}): target {args.target}, "
             f"holdout {hold}, seeds {args.seeds}", "",
             f"train rows {len(train_df)}, eval rows {len(eval_df)} "
             f"(pos {int(eval_df[args.target].sum())})", "",
             "| variant | AP | best F1 | ECE | Brier | person R@0.5 | "
             "day AP/F1/ECE | dusk AP/F1/ECE | night AP/F1/ECE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for v, r in results.items():
        def dp(name):
            d = r["by_daypart"].get(name)
            return (f"{d['ap']:.3f}/{d['best_f1']:.3f}/{d['ece']:.3f} (n={d['n']})"
                    if d else "—")
        pr = r["person_recall"]
        lines.append(f"| {v} | {r['ap']:.3f} | {r['best_f1']:.3f} | {r['ece']:.3f} | "
                     f"{r['brier']:.3f} | {pr if pr is None else round(pr, 3)} | "
                     f"{dp('day')} | {dp('dusk')} | {dp('night')} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}.md / .json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
