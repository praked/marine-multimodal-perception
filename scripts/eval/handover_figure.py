"""Modality-handover figure: one continuous dusk -> night recording.

Rows (time on x, wall clock):
  1. sun elevation (RTC + position) and the camera's mean luminance -- the
     auto-exposure holds luminance while the sun sets, which is why exposure
     time/gain are logged since 2026-08-28 (absent on this recording);
  2. per-modality evidence: fraction of sectors per frame with a fisheye,
     thermal or radar hit (rolling median);
  3. fused obstacle probability on the sectors that the audit labelled
     positive (rolling median) for the rule-based vote and the GBT scorer,
     with the decision threshold; audited frames marked.

    python -m scripts.eval.handover_figure --features data/features_full \
        --scene 2026-08-26_afloat --start 19:22 --end 20:45 \
        --scorer models/fusion_scorer_live_v1b.joblib --out images/handover_2026-08-26.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fusion_model.evaluate import load_dataset  # noqa: E402
from scripts.fusion_model.models import load_any_scorer  # noqa: E402

FISHEYE, THERMAL, RADAR = "#1487B8", "#B8741A", "#7A6BB5"
NAVY, MUTED, GOOD, WARN, LINE = "#0A2A3A", "#4C7B94", "#2E8B57", "#D97E0C", "#DCE9F0"


def seconds_of_day(ts: pd.Series) -> np.ndarray:
    parts = ts.str.split(":", expand=True).astype(float)
    return (parts[0] * 3600 + parts[1] * 60 + parts[2]).to_numpy()


def rolling(x: np.ndarray, y: np.ndarray, win_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Median of y in win_s-wide bins of x (x in seconds)."""
    if len(x) == 0:
        return x, y
    edges = np.arange(x.min(), x.max() + win_s, win_s)
    idx = np.digitize(x, edges)
    xs, ys = [], []
    for i in np.unique(idx):
        m = idx == i
        if m.sum() >= 3:
            xs.append(np.median(x[m])); ys.append(np.nanmedian(y[m]))
    return np.array(xs), np.array(ys)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features_full")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", default="00:00")
    ap.add_argument("--end", default="23:59")
    ap.add_argument("--scorer", default="models/fusion_scorer_live_v1b.joblib")
    ap.add_argument("--scorer2", default=None, help="optional second GBT bundle")
    ap.add_argument("--win", type=float, default=60.0, help="rolling window (s)")
    ap.add_argument("--out", default="images/handover.png")
    args = ap.parse_args(argv)

    df = load_dataset(args.features, require_targets=False)
    df = df[df["clip_id"].str.startswith(args.scene + "/")].copy()
    sod = seconds_of_day(df["timestamp"])
    t0 = sum(float(v) * k for v, k in zip(args.start.split(":"), (3600, 60)))
    t1 = sum(float(v) * k for v, k in zip(args.end.split(":"), (3600, 60)))
    df = df[(sod >= t0) & (sod <= t1)].copy()
    df["sod"] = seconds_of_day(df["timestamp"])
    if df.empty:
        print("no rows in window", file=sys.stderr)
        return 1
    print(f"{len(df)} sector rows, {df['clip_id'].nunique()} chunks, "
          f"{df['timestamp'].nunique()} frames", file=sys.stderr)

    # scorer(s)
    scorer = load_any_scorer(args.scorer)
    df["p_gbt"] = scorer.predict_proba_calibrated_df(df)
    if args.scorer2:
        df["p_gbt2"] = load_any_scorer(args.scorer2).predict_proba_calibrated_df(df)

    # per-frame aggregates
    fr = df.groupby("timestamp").agg(
        sod=("sod", "first"), sun=("sun_elevation_deg", "first"),
        lum=("luminance_mean", "first"),
        fish=("hit_fisheye", "mean"), therm=("hit_thermal", "mean"),
        radar=("hit_mmwave", "mean"), conf=("confirmed", "mean"),
        seg=("seg_available", "first"), tq=("thermal_quality_ok", "first"),
        audited=("audited", "max") if "audited" in df.columns else ("sod", "size"),
    ).reset_index().sort_values("sod")
    has_gt = "y_nav" in df.columns and df["y_nav"].notna().any()
    if has_gt:
        pos = df[df["y_nav"] == 1]
        ppos = pos.groupby("timestamp").agg(sod=("sod", "first"),
                                             vote=("score_legacy", "mean"),
                                             gbt=("p_gbt", "mean")).sort_values("sod")
        if args.scorer2:
            ppos = ppos.join(pos.groupby("timestamp")["p_gbt2"].mean())

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.15], "hspace": 0.12,
                                          "left": 0.08, "right": 0.9, "top": 0.95, "bottom": 0.09})
    x = (fr["sod"].to_numpy() - t0) / 60.0
    ax = axes[0]
    ax.plot(x, fr["sun"], color=WARN, lw=2, label="sun elevation (°)")
    ax.axhline(0, color=WARN, lw=0.8, ls=":")
    ax.axhline(-6, color=WARN, lw=0.8, ls="--")
    ax.text(x.max(), -6.4, "civil twilight (−6°)", color=WARN, fontsize=9, ha="right", va="top")
    ax.set_ylabel("sun elev. (°)", color=WARN)
    ax2 = ax.twinx()
    xs, ys = rolling(fr["sod"].to_numpy(), fr["lum"].to_numpy(), args.win)
    ax2.plot((xs - t0) / 60, ys, color=NAVY, lw=1.6, label="mean luminance (auto-exposed)")
    ax2.set_ylabel("luminance", color=NAVY); ax2.set_ylim(0, 255)
    ax.set_title(f"{args.scene}: continuous dusk → night, per-frame context, evidence and fused probability", fontsize=11)

    ax = axes[1]
    for col, c, lab in (("fish", FISHEYE, "fisheye"), ("therm", THERMAL, "thermal"), ("radar", RADAR, "radar")):
        xs, ys = rolling(fr["sod"].to_numpy(), fr[col].to_numpy(), args.win)
        ax.plot((xs - t0) / 60, ys, color=c, lw=1.8, label=f"{lab} hit fraction")
    xs, ys = rolling(fr["sod"].to_numpy(), fr["conf"].to_numpy(), args.win)
    ax.plot((xs - t0) / 60, ys, color=GOOD, lw=1.2, ls="--", label="radar-confirmed fraction")
    ax.set_ylabel("sectors with evidence"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5, ncol=4, loc="upper left", frameon=False)

    ax = axes[2]
    if has_gt:
        xs, ys = rolling(ppos["sod"].to_numpy(), ppos["vote"].to_numpy(), args.win)
        ax.plot((xs - t0) / 60, ys, color=MUTED, lw=1.8, label="rule-based vote on labelled-positive sectors")
        xs, ys = rolling(ppos["sod"].to_numpy(), ppos["gbt"].to_numpy(), args.win)
        ax.plot((xs - t0) / 60, ys, color=NAVY, lw=2.2, label="GBT p(obstacle) on labelled-positive sectors")
        if args.scorer2:
            xs, ys = rolling(ppos["sod"].to_numpy(), ppos["p_gbt2"].to_numpy(), args.win)
            ax.plot((xs - t0) / 60, ys, color=GOOD, lw=1.6, label="GBT (re-calibrated)")
        aud = fr[fr["audited"] > 0] if "audited" in fr.columns else fr.iloc[0:0]
        if len(aud):
            ax.scatter((aud["sod"] - t0) / 60, np.full(len(aud), 0.02), s=6, color="#C0392B",
                       label="human-audited frames", zorder=3)
    ax.axhline(0.5, color=WARN, lw=1, ls="--"); ax.text(0.2, 0.52, "threshold 0.5", color=WARN, fontsize=8.5)
    ax.set_ylim(0, 1); ax.set_ylabel("p on positive sectors")
    ax.set_xlabel(f"minutes after {args.start}")
    ax.legend(fontsize=8.5, loc="lower left", frameon=False)
    # sunset + the audited span, on every row
    sun = fr["sun"].to_numpy()
    if (sun > 0).any() and (sun < 0).any():
        xs_set = x[np.argmax(sun < 0)]
        for a in axes:
            a.axvline(xs_set, color=WARN, lw=1, ls=":")
        axes[0].text(xs_set + 0.4, axes[0].get_ylim()[1] * 0.85, "sunset", color=WARN, fontsize=9)
    if "audited" in fr.columns and (fr["audited"] > 0).any():
        aud_x = x[fr["audited"].to_numpy() > 0]
        for a in axes:
            a.axvspan(aud_x.min(), aud_x.max(), color="#C0392B", alpha=0.05, lw=0)
        axes[2].text(aud_x.min() + 0.4, 0.93, "human-audited span (positives = audit); "
                     "outside: teacher pseudo-labels", color="#C0392B", fontsize=8.5)
    for a in axes:
        a.grid(color=LINE, lw=0.6); a.spines[["top", "right"]].set_visible(False)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250); print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
