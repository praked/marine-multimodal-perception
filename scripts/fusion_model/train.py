"""Multi-seed bake-off: gated mixture (v1a) vs GBT (v1b) vs incumbent.

Assembles every clip with feature + target tables, splits by the frozen
`scripts.data.splits` assignment already recorded in the frames table
(outing-level group_hash, no re-rolling), trains:

  - v1a GatedMixtureScorer × train.seeds (pure numpy; the ship candidate)
  - v1b HistGradientBoosting × seeds (optional scikit-learn; skipped with
    a message when absent: the delta v1b−v1a measures how much
    non-additive structure matters, plan §1.1)

then evaluates v1a / v1b / the rule-based incumbent on the held-out split
with the Phase-1 harness, isotonic-calibrates on val, composes `threat`,
and writes:

  results/fusion_model/bakeoff_<tag>.md / .json     comparison report
  results/fusion_model/gate_curves_<tag>.csv        w_s(context) grids
  models/fusion_scorer_v1a.json                     best-seed v1a (gitignored)

Bounded expectations pre-data: the current corpus is few-scene and
day-dominated; this run proves the loop and the error bars, not final
numbers (plan Phase 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fusion_model.build_features import FEATURES_ROOT
from scripts.fusion_model.evaluate import (
    RESULTS_DIR,
    blocked_rate_by_scene,
    evaluate_scores,
    load_dataset,
    render_report,
)
from scripts.fusion_model.models import (
    CONTEXT_COLUMNS,
    GatedMixtureScorer,
    IsotonicCalibrator,
    fit_gbt,
    gbt_predict,
    load_fusion_model_config,
    prepare_matrix,
    CONTEXT_COLUMNS,
    MOTION_CONTEXT_COLUMNS,
    SENSORS,
    threat_from_score,
)
from scripts.utils.datasets import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"


def gate_curves(model: GatedMixtureScorer, cfg: dict) -> pd.DataFrame:
    """w_s(context) on interpretation grids: the paper figure data.

    Two sweeps around a neutral context: sun elevation (dark tracks it)
    and range_norm (the distance-reliability envelope inside the gate)."""
    neutral = {
        "sun_elevation_norm": 0.5, "luminance_norm": 0.5, "is_dark": 0.0,
        "thermal_quality_ok": 1.0, "seg_available": 1.0,
        "imu_available": 1.0, "abs_bearing_norm": 0.0, "range_norm": 0.5,
    }
    rows = []
    for sweep_name, col, values, coupled in (
        ("sun_elevation", "sun_elevation_norm",
         np.linspace(-0.3, 1.0, 27), True),
        ("range", "range_norm", np.linspace(0.0, 1.0, 21), False),
    ):
        for val in values:
            ctx = dict(neutral)
            ctx[col] = float(val)
            if coupled:   # below-horizon sun ⇒ dark frame, low luminance
                ctx["is_dark"] = 1.0 if val < 0.0 else 0.0
                ctx["luminance_norm"] = 0.05 if val < 0.0 else 0.5
            c = np.array([[ctx[k] for k in CONTEXT_COLUMNS]])
            w = model.gate_weights(c)[0]
            rows.append({"sweep": sweep_name, "x": float(val),
                         **{f"w_{s}": float(w[i])
                            for i, s in enumerate(model.sensors)}})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(FEATURES_ROOT))
    ap.add_argument("--target", default="y_nav",
                    choices=["y_nav", "y_obstacle"])
    ap.add_argument("--tag", default="v1",
                    help="suffix for report/model artefacts")
    ap.add_argument("--holdout-clips", default="",
                    help="comma-separated clip_id substrings to hold out (overrides --holdout)")
    ap.add_argument("--holdout", default=None, choices=[None, "val", "test"],
                    help="evaluation split (default: val, falling back to a "
                         "20%% clip holdout when every clip landed in train)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--typed-evidence", action="store_true",
                    help="add the typed-YOLO confidence as a 4th evidence "
                         "channel (needs yolo_* columns filled: features "
                         "built with data/det present)")
    ap.add_argument("--motion-context", action="store_true",
                    help="add the per-bin target-motion columns "
                         "(MOTION_CONTEXT_COLUMNS; needs features built "
                         "with motion.targets enabled) to the gate context")
    ap.add_argument("--gbt-columns", default="full",
                    help="GBT column subset for v1b: full | no_context | "
                         "rgb_only | thermal_only | radar_only | radar_only_noconfirm (ablate.py's "
                         "definitions). radar_only = the tail-mode shadow's "
                         "input (no camera pixels).")
    ap.add_argument("--oof-calibration", action="store_true",
                    help="fit the v1b bundle's isotonic map on out-of-fold "
                         "(clip-fold) predictions instead of the model's own "
                         "training predictions. Measured 2026-09-03 on the "
                         "audited holdout: train-fit map ECE 0.321 (worse than "
                         "raw 0.227), OOF map 0.138.")
    args = ap.parse_args(argv)

    cfg = load_fusion_model_config()
    seeds = args.seeds if args.seeds else list(
        cfg.get("train", {}).get("seeds", [0, 1, 2]))
    tr_cfg = cfg.get("train", {})

    df = load_dataset(args.features)
    if df.empty:
        print("no feature+target tables; run build_features + build_targets",
              file=sys.stderr)
        return 1

    # Split rows by the frames table's frozen assignment. Small-corpus
    # fallback: if the labelled clips all hashed into one split, hold out
    # whole clips (never frames: outing leakage) by hash order.
    holdout_name = args.holdout or "val"
    eval_mask = df["split"] == holdout_name
    if args.holdout_clips:
        # Explicit clip holdout by clip_id substring (e.g. "2026-09-08_20-4" holds
        # out the 20:4x chunks of the lake session): the way to keep a new
        # outing IN training while evaluating on part of it (2026-09-12).
        subs = [x for x in args.holdout_clips.split(",") if x]
        eval_mask = df["clip_id"].apply(lambda c: any(x in str(c) for x in subs))
        holdout_name = f"clips({','.join(subs)})"
    if not eval_mask.any() or eval_mask.all():
        clips = sorted(df["clip_id"].unique())
        if len(clips) > 1:
            n_hold = max(1, len(clips) // 5)
            held = set(clips[-n_hold:])
            eval_mask = df["clip_id"].isin(held)
            holdout_name = f"clip-holdout({', '.join(sorted(held))})"
            print(f"[split] frozen split gave no usable holdout on this "
                  f"corpus -> holding out clips: {sorted(held)}",
                  file=sys.stderr)
        else:
            # Single labelled clip: hold out the last 25% of frames. Same-
            # outing leakage: smoke-run quality only, never a real number.
            cut = df["frame_index"].quantile(0.75)
            eval_mask = df["frame_index"] > cut
            holdout_name = "frame-tail-25% (LEAKY: single-clip smoke only)"
            print("[split] single labelled clip -> frame-tail holdout; "
                  "numbers are leaky, smoke only", file=sys.stderr)
    train_df = df[~eval_mask].reset_index(drop=True)
    eval_df = df[eval_mask].reset_index(drop=True)
    print(f"[data] train rows {len(train_df)} "
          f"(pos {int(train_df[args.target].sum())}), "
          f"eval[{holdout_name}] rows {len(eval_df)} "
          f"(pos {int(eval_df[args.target].sum())})", file=sys.stderr)

    ctx_cols = tuple(CONTEXT_COLUMNS)
    if args.motion_context:
        ctx_cols = ctx_cols + MOTION_CONTEXT_COLUMNS
    sensors = tuple(SENSORS) + (("yolo",) if args.typed_evidence else ())
    e_tr, c_tr, _ = prepare_matrix(train_df, cfg, columns=ctx_cols,
                                   sensors=sensors)
    e_ev, c_ev, _ = prepare_matrix(eval_df, cfg, columns=ctx_cols,
                                   sensors=sensors)
    y_tr = train_df[args.target].to_numpy(dtype=float)

    # --- v1a × seeds --------------------------------------------------------
    v1a_models, v1a_scores = [], []
    for seed in seeds:
        m = GatedMixtureScorer.fit(
            e_tr, c_tr, y_tr, seed=seed,
            epochs=int(tr_cfg.get("epochs", 300)),
            lr=float(tr_cfg.get("learning_rate", 0.05)),
            l2=float(tr_cfg.get("l2", 1e-3)))
        m.context_columns = ctx_cols  # artefact records its own column list
        m.sensors = sensors
        v1a_models.append(m)
        v1a_scores.append(m.predict_proba(e_ev, c_ev))
    v1a_mean = np.mean(np.stack(v1a_scores), axis=0)
    eval_df["p_v1a"] = v1a_mean
    # Isotonic calibration fit on the same eval predictions would leak;
    # fit on TRAIN predictions instead (calibration is monotone, cheap).
    cal = IsotonicCalibrator.fit(
        np.mean(np.stack([m.predict_proba(e_tr, c_tr)
                          for m in v1a_models]), axis=0), y_tr)
    eval_df["p_v1a_cal"] = cal.transform(v1a_mean)

    # --- v1b × seeds (optional sklearn) --------------------------------------
    gbt_note = ""
    gbts = []
    from scripts.fusion_model import models as _models
    from scripts.fusion_model.ablate import VARIANT_COLUMNS, oof_calibrator
    gbt_cols = tuple(VARIANT_COLUMNS[args.gbt_columns])
    _models.GBT_COLUMNS = gbt_cols        # fit/predict read it at call time
    for seed in seeds:
        clf = fit_gbt(train_df, y_tr, seed=seed)
        if clf is None:
            gbt_note = ("v1b SKIPPED: scikit-learn not installed "
                        "(optional dep: pip install scikit-learn)")
            break
        gbts.append(clf)
    if gbts:
        eval_df["p_v1b"] = np.mean(
            np.stack([gbt_predict(clf, eval_df) for clf in gbts]), axis=0)

    # --- threat composition (severity table is UNREVIEWED placeholder) -------
    eval_df["threat_v1a"] = threat_from_score(
        eval_df["p_v1a_cal"].to_numpy(),
        eval_df["min_range_m"].to_numpy(dtype=float),
        eval_df["per_bin_ttc_s"].to_numpy(dtype=float),
        severity=1.0, cfg=cfg)

    # --- evaluation -----------------------------------------------------------
    twl = float(cfg.get("context", {}).get("twilight_deg", 6.0))
    results = {"incumbent (score_legacy)": evaluate_scores(
        eval_df, "score_legacy", args.target, twilight_deg=twl)}
    results["v1a gated mixture (mean of seeds)"] = evaluate_scores(
        eval_df, "p_v1a", args.target, twilight_deg=twl)
    results["v1a calibrated"] = evaluate_scores(
        eval_df, "p_v1a_cal", args.target, twilight_deg=twl)
    if "p_v1b" in eval_df.columns:
        results["v1b GBT (mean of seeds)"] = evaluate_scores(
            eval_df, "p_v1b", args.target, twilight_deg=twl)

    # per-seed AP spread: the honest error bar
    seed_aps = []
    for i, s in enumerate(v1a_scores):
        eval_df["_seed_tmp"] = s
        seed_aps.append(evaluate_scores(
            eval_df, "_seed_tmp", args.target,
            twilight_deg=twl)["average_precision"])
    eval_df.drop(columns=["_seed_tmp"], inplace=True)

    blocked = blocked_rate_by_scene(
        load_dataset(args.features, require_targets=False), "score_legacy")

    # --- artefacts -------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    title = (f"Fusion-scorer bake-off ({tag}): target {args.target}, "
             f"holdout {holdout_name}, seeds {seeds}")
    header_notes = [
        f"v1a AP per seed: {[round(a, 3) for a in seed_aps]} "
        f"(mean {np.nanmean(seed_aps):.3f} ± {np.nanstd(seed_aps):.3f})",
        gbt_note or "v1b trained",
        "PRE-DATA RUN: few-scene, day-dominated corpus; proves the loop "
        "and error bars, not final numbers (plan Phase 2/3).",
        "threat_v1a uses the UNREVIEWED severity/urgency placeholders "
        "(configs/fusion_model.yaml): set with AuthorOne before Phase 3.",
    ]
    body = render_report(results, blocked, title)
    first_nl = body.index("\n")
    report = (body[:first_nl] + "\n\n"
              + "\n".join(f"> {n}" for n in header_notes if n)
              + body[first_nl:])
    out_md = RESULTS_DIR / f"bakeoff_{tag}.md"
    out_md.write_text(report)
    with open(RESULTS_DIR / f"bakeoff_{tag}.json", "w") as fh:
        json.dump({k: v for k, v in results.items()}, fh, indent=2,
                  default=float)

    # Gate curves for the best seed (highest eval AP) + model artefact.
    best_i = int(np.nanargmax(seed_aps))
    gate_curves(v1a_models[best_i], cfg).to_csv(
        RESULTS_DIR / f"gate_curves_{tag}.csv", index=False)
    # Tagged runs write a TAGGED artefact: models/fusion_scorer_v1a.json is
    # the LIVE scorer (fusion.py --scorer default) and must only change on a
    # deliberate promotion, never as a side effect of an experiment
    # (2026-09-02: an audited-retrain probe silently clobbered the live
    # d2_winner; restored from the cluster_d2 archive).
    model_path = MODELS_DIR / (
        f"fusion_scorer_{tag}.json" if tag and tag != "v1a" else "fusion_scorer_v1a.json")
    # Ship the calibrator INSIDE the artefact: fit on the best seed's own
    # TRAIN predictions (monotone, no eval leak) so the deployed shadow
    # p_obstacle is a probability, not a class-balanced ranking score.
    best = v1a_models[best_i]
    best.calibrator = IsotonicCalibrator.fit(
        best.predict_proba(e_tr, c_tr), y_tr)
    best.save(model_path)

    # v1b ensemble bundle (deployable since 2026-09-02): mean-over-seeds —
    # exactly what the bake-off evaluates — with a train-fit calibrator.
    # Same tagged-artefact discipline as v1a.
    if gbts:
        from scripts.fusion_model.models import GBTScorerBundle
        bundle = GBTScorerBundle(
            clfs=gbts, columns=gbt_cols,
            train_info={"tag": tag, "seeds": list(seeds),
                        "target": args.target, "n_train": int(len(y_tr)),
                        "gbt_columns": args.gbt_columns,
                        "calibration": "oof" if args.oof_calibration else "train"})
        if args.oof_calibration:
            bundle.calibrator = oof_calibrator(train_df, y_tr, gbt_cols, seeds)
        else:
            p_tr = bundle.predict_proba_df(train_df)
            bundle.calibrator = IsotonicCalibrator.fit(p_tr, y_tr)
        v1b_path = MODELS_DIR / f"fusion_scorer_{tag or 'v1b'}_v1b.joblib"
        bundle.save(v1b_path)
        print(f"wrote {v1b_path} (v1b ensemble, {len(gbts)} seeds)")

    print(f"wrote {out_md}")
    print(f"wrote {RESULTS_DIR / f'gate_curves_{tag}.csv'}")
    print(f"wrote {model_path} (best seed {seeds[best_i]})")
    if gbt_note:
        print(gbt_note, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
