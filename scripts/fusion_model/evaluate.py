"""Stratified evaluation harness for fusion scorers (Phase 1, plan §6).

Evaluates ANY per-bin score column against the bearing-space targets:
the rule-based incumbent (`score_legacy`) first; that run is the
baseline table every learned scorer must beat, then model prediction
columns added by train.py.

Metrics:
  - precision / recall / F1 across a threshold sweep + average precision
  - probability calibration: 10-bin reliability table + ECE (a score the
    nav layer thresholds must be calibrated, not just ranked)
  - strata: scene, day/dusk/night (sun elevation, civil-twilight ±6°),
    range band (near ≤5 m / mid 5–15 / far-or-unknown), audited-only
  - person-class recall (the swimmer metric, runbook §3.3)
  - open-water FP: FP bins/frame on OpenWater-scene labelled rows, plus a
    label-free blocked-rate table per scene (the §10.6 property view)

Targets default to `y_nav` (nav-relevant); `--target y_obstacle` gives the
perception view. Rows exist only for labelled frames (build_targets).

Usage:
    python -m scripts.fusion_model.evaluate                    # incumbent
    python -m scripts.fusion_model.evaluate --score p_v1a --pred-csv ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fusion_model.build_features import FEATURES_ROOT
from scripts.fusion_model.models import load_fusion_model_config, merge_tables
from scripts.utils.datasets import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results" / "fusion_model"
DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.66, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------

def _read_table(path_base: Path) -> pd.DataFrame | None:
    for ext in ("parquet", "csv"):
        p = path_base.with_suffix(f".{ext}")
        if p.exists():
            return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    return None


def load_dataset(features_root: str | Path = FEATURES_ROOT,
                 require_targets: bool = True,
                 clips: list[str] | None = None) -> pd.DataFrame:
    """Concatenate merged (bins x frames x targets) tables across clips."""
    parts = []
    for d in sorted(Path(features_root).iterdir()):
        if not d.is_dir() or not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        if clips and meta["clip_id"] not in clips:
            continue
        bins_df = _read_table(d / "bins")
        frames_df = _read_table(d / "frames")
        targets_df = _read_table(d / "targets")
        if bins_df is None or frames_df is None:
            continue
        if require_targets and (targets_df is None or targets_df.empty):
            continue
        parts.append(merge_tables(bins_df, frames_df, targets_df))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def daypart(sun_elevation_deg: pd.Series, twilight_deg: float = 6.0
            ) -> pd.Series:
    """day / dusk / night via civil-twilight bounds on sun elevation."""
    e = sun_elevation_deg
    return pd.Series(
        np.where(e > twilight_deg, "day",
                 np.where(e < -twilight_deg, "night", "dusk")),
        index=e.index)


def range_band(df: pd.DataFrame) -> pd.Series:
    """near ≤5 m / mid 5–15 m / far-or-unknown, from the bin's best range
    evidence (radar first, then the fused min_range)."""
    r = df["radar_min_range_m"]
    if "min_range_m" in df.columns:
        r = r.fillna(df["min_range_m"])
    return pd.Series(np.where(r <= 5.0, "near<=5m",
                              np.where(r <= 15.0, "mid5-15m", "far/unknown")),
                     index=df.index)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def pr_at_threshold(y: np.ndarray, s: np.ndarray, thresh: float) -> dict:
    pred = s >= thresh
    tp = int(np.sum(pred & (y > 0.5)))
    fp = int(np.sum(pred & (y < 0.5)))
    fn = int(np.sum(~pred & (y > 0.5)))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if tp and prec + rec else (0.0 if tp + fp + fn else float("nan")))
    return {"threshold": thresh, "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """AP = sum over positives of precision at each positive's rank."""
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    cum_tp = np.cumsum(ys)
    ranks = np.arange(1, len(ys) + 1)
    prec_at = cum_tp / ranks
    return float((prec_at * ys).sum() / ys.sum())


def reliability(y: np.ndarray, s: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability table + expected calibration error."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece = [], 0.0
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi if hi < 1.0 else s <= hi)
        if not m.any():
            continue
        conf, acc = float(s[m].mean()), float(y[m].mean())
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
                     "mean_score": conf, "frac_positive": acc})
        ece += (m.sum() / n) * abs(conf - acc)
    return {"bins": rows, "ece": float(ece)}


def evaluate_scores(df: pd.DataFrame, score_col: str, target_col: str = "y_nav",
                    thresholds: list[float] | None = None,
                    twilight_deg: float = 6.0) -> dict:
    """Full metric block for one scorer on one (sub)table."""
    y = df[target_col].to_numpy(dtype=float)
    s = df[score_col].fillna(0.0).to_numpy(dtype=float)
    thresholds = thresholds or DEFAULT_THRESHOLDS
    sweep = [pr_at_threshold(y, s, t) for t in thresholds]
    best = max((r for r in sweep if not np.isnan(r["f1"])),
               key=lambda r: r["f1"], default=None)

    out = {
        "score": score_col, "target": target_col,
        "n_rows": int(len(df)), "n_pos": int(y.sum()),
        "average_precision": average_precision(y, s),
        "sweep": sweep, "best_f1": best,
        "calibration": reliability(y, s),
        "strata": {},
    }

    dp = daypart(df["sun_elevation_deg"], twilight_deg)
    strata: dict[str, pd.Series] = {
        "scene": df["scene"],
        "daypart": dp,
        "range_band": range_band(df),
    }
    if "audited" in df.columns:
        strata["audited"] = df["audited"].map({1: "audited", 0: "unaudited"})
    for name, key in strata.items():
        blocks = {}
        for val, sub in df.groupby(key):
            ys = sub[target_col].to_numpy(dtype=float)
            ss = sub[score_col].fillna(0.0).to_numpy(dtype=float)
            blocks[str(val)] = {
                "n": int(len(sub)), "n_pos": int(ys.sum()),
                "ap": average_precision(ys, ss),
                **{k: v for k, v in pr_at_threshold(ys, ss, 0.5).items()
                   if k in ("precision", "recall", "f1")},
            }
        out["strata"][name] = blocks

    # Swimmer/person recall: rows whose same-frame labels include person.
    if "label_classes" in df.columns:
        person = df[df["label_classes"].fillna("").str.contains("person")]
        if len(person):
            ys = person[target_col].to_numpy(dtype=float)
            ss = person[score_col].fillna(0.0).to_numpy(dtype=float)
            out["person_recall@0.5"] = pr_at_threshold(ys, ss, 0.5)["recall"]
            out["person_rows"] = int(len(person))

    # Open-water FP bins per labelled frame at each threshold.
    ow = df[df["scene"].astype(str).str.startswith("OpenWater")]
    if len(ow):
        n_frames = ow.groupby(["clip_id", "frame_index"]).ngroups
        out["openwater_fp_per_frame"] = {
            str(t): float(((ow[score_col].fillna(0.0) >= t)
                           & (ow[target_col] < 0.5)).sum() / n_frames)
            for t in (0.33, 0.5, 0.66)}
    return out


def blocked_rate_by_scene(bins_frames: pd.DataFrame, score_col: str,
                          thresholds=(0.33, 0.5, 0.66)) -> pd.DataFrame:
    """Label-free view: mean bins >= threshold per frame, per scene (the
    §10.6 open-water-suppression property, comparable across scorers)."""
    rows = []
    for scene, sub in bins_frames.groupby("scene"):
        n_frames = sub.groupby(["clip_id", "frame_index"]).ngroups
        row = {"scene": scene, "n_frames": n_frames}
        for t in thresholds:
            row[f"bins>={t}/frame"] = float(
                (sub[score_col].fillna(0.0) >= t).sum() / max(n_frames, 1))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if isinstance(v, float):
        return "—" if np.isnan(v) else f"{v:.3f}"
    return str(v)


def render_report(results: dict[str, dict], blocked: pd.DataFrame | None,
                  title: str) -> str:
    lines = [f"# {title}", ""]
    for name, res in results.items():
        lines += [f"## {name}  (target: {res['target']})", "",
                  f"- rows {res['n_rows']}, positives {res['n_pos']}"
                  f" ({res['n_pos'] / max(res['n_rows'], 1):.1%})",
                  f"- **average precision {_fmt(res['average_precision'])}**"
                  f", ECE {_fmt(res['calibration']['ece'])}"]
        if res.get("best_f1"):
            b = res["best_f1"]
            lines.append(f"- best F1 {_fmt(b['f1'])} @ {b['threshold']}"
                         f" (P {_fmt(b['precision'])} / R {_fmt(b['recall'])})")
        if "person_recall@0.5" in res:
            lines.append(f"- person-class recall@0.5 "
                         f"{_fmt(res['person_recall@0.5'])} "
                         f"({res['person_rows']} rows)")
        if "openwater_fp_per_frame" in res:
            lines.append("- open-water FP bins/frame: "
                         + ", ".join(f"{k}: {_fmt(v)}" for k, v in
                                     res["openwater_fp_per_frame"].items()))
        lines += ["", "| threshold | P | R | F1 |", "|---|---|---|---|"]
        lines += [f"| {r['threshold']} | {_fmt(r['precision'])} | "
                  f"{_fmt(r['recall'])} | {_fmt(r['f1'])} |"
                  for r in res["sweep"]]
        for stratum, blocks in res["strata"].items():
            lines += ["", f"### by {stratum}", "",
                      "| value | n | pos | AP | P@0.5 | R@0.5 | F1@0.5 |",
                      "|---|---|---|---|---|---|---|"]
            lines += [f"| {k} | {v['n']} | {v['n_pos']} | {_fmt(v['ap'])} | "
                      f"{_fmt(v['precision'])} | {_fmt(v['recall'])} | "
                      f"{_fmt(v['f1'])} |" for k, v in blocks.items()]
        lines.append("")
    if blocked is not None and not blocked.empty:
        try:
            table = blocked.to_markdown(index=False)   # needs tabulate
        except ImportError:
            table = "```\n" + blocked.to_string(index=False) + "\n```"
        lines += ["## Blocked-bin rate per frame (label-free, all frames)",
                  "", table, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI (incumbent evaluation)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(FEATURES_ROOT))
    ap.add_argument("--score", default="score_legacy",
                    help="score column to evaluate (default: incumbent)")
    ap.add_argument("--target", default="y_nav",
                    choices=["y_nav", "y_obstacle"])
    ap.add_argument("--out", default=None,
                    help="report path (default results/fusion_model/"
                         "eval_<score>.md)")
    args = ap.parse_args(argv)

    cfg = load_fusion_model_config()
    df = load_dataset(args.features)
    if df.empty:
        print("no feature+target tables found; run build_features and "
              "build_targets first", file=sys.stderr)
        return 1
    res = evaluate_scores(df, args.score, args.target,
                          twilight_deg=float(cfg.get("context", {})
                                             .get("twilight_deg", 6.0)))
    all_bins = load_dataset(args.features, require_targets=False)
    blocked = blocked_rate_by_scene(all_bins, args.score)

    out = Path(args.out) if args.out else RESULTS_DIR / f"eval_{args.score}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report({args.score: res}, blocked,
                                 f"Fusion-scorer evaluation: {args.score}"))
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
