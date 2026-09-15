"""Modality-handover figure for the 2026-09-08 lake session (17:11 -> 21:36 CEST).

Column-width companion of scripts/eval/handover_figure.py, built from protocol
sector records instead of feature tables (the session has no labels; the
learned p field itself is the object of interest).

Rows (wall clock on x):
  1. illuminance: sun elevation (RTC + launch position) and the fisheye's mean
     luminance (log axis) with the exposure-saturation point and the darkness
     gate (fisheye.darkness_thresh = 25) marked;
  2. per-sensor evidence rate: fraction of the 7 sectors with a fisheye,
     thermal or radar hit per frame (rolling median);
  3. learned output: mean and highest-sector p(obstacle) of the live scorer
     bundle and the fused n/3 vote (rolling medians), with the per-band
     marginal sensor weights (least squares of p on the three hit flags);
  4. availability on the box: fraction of the vessel's own shadow ticks per
     minute that ran the camera path, had a fresh segmentation mask, and had a
     fresh YOLO result (tail-mode ticks count as no camera path).

Sector source for rows 2-3 defaults to the offline replay of the live bundle
under the box's config with the darkness gate (every frame); the live shadow
records feed row 4 only. The night-trial window (20:27-21:10) is shaded.

    python -m scripts.eval.handover_figure_0908 \
        --sectors /Volumes/ROS2_SSD/asvproject/aime_2026-09-12/sectors_ctrl_live_gatedcfg \
        --features /Volumes/ROS2_SSD/asvproject/aime_2026-09-12/features_gated_0908 \
        --live results/sectors_night \
        --out results/closed_loop/2026-09-08/handover_2026-09-08.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sensor_processing.gps_boat1 import sun_position  # noqa: E402

FISHEYE, THERMAL, RADAR = "#1487B8", "#B8741A", "#7A6BB5"
NAVY, MUTED, GOOD, WARN, LINE = "#0A2A3A", "#4C7B94", "#2E8B57", "#D97E0C", "#DCE9F0"
TRIAL, DARK = "#0A2A3A", "#6B6B6B"

CEST = timezone(timedelta(hours=2))
LAT, LON = 46.0, 9.0
SCENE = "2026-09-08_afloat_2026-09-08_17-10-55"
NIGHT_TRIALS = ("20:27:00", "21:10:00")
SKIP_CHUNKS = ("2026-09-08_21-36-23",)  # unreadable
DARK_THRESH = 25.0  # fisheye.darkness_thresh
DATE = (2026, 9, 8)


def sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def hhmm(s: float) -> str:
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}"


def sun_elev(sod: float) -> float:
    t = datetime(*DATE, tzinfo=CEST) + timedelta(seconds=float(sod))
    return sun_position(LAT, LON, t.astimezone(timezone.utc))[0]


def load_sectors(d: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(d, f"{SCENE}__*.jsonl"))):
        if any(f.endswith(f"__{c}.jsonl") for c in SKIP_CHUNKS):
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not (r.get("timestamp") and r.get("p_obstacle") and r.get("sensor_hits")):
                    continue
                hits = np.asarray(r["sensor_hits"], float)
                p = np.asarray(r["p_obstacle"], float)
                sc = np.asarray(r.get("scores") or [np.nan] * len(p), float)
                sh = r.get("shadow")
                rows.append({
                    "sod": sec(r["timestamp"]), "chunk": os.path.basename(f),
                    "fish": hits[:, 0].mean(), "therm": hits[:, 1].mean(), "radar": hits[:, 2].mean(),
                    "p_mean": p.mean(), "p_max": p.max(), "vote_mean": np.nanmean(sc),
                    "hits": hits, "p": p,
                    "cam": (bool(sh.get("camera")) if sh else np.nan),
                    "seg": (bool(sh.get("seg_fresh")) if sh else np.nan),
                    "yolo": (bool(sh.get("yolo_fresh")) if sh else np.nan),
                    "has_shadow": sh is not None,
                })
    df = pd.DataFrame(rows).sort_values("sod").reset_index(drop=True)
    return df


def load_frames(d: str) -> pd.DataFrame:
    fr = []
    for sub in sorted(glob.glob(os.path.join(d, f"{SCENE}__*"))):
        if any(sub.endswith(f"__{c}") for c in SKIP_CHUNKS):
            continue
        f = os.path.join(sub, "frames.csv")
        if os.path.exists(f):
            fr.append(pd.read_csv(f))
    df = pd.concat(fr, ignore_index=True)
    df["sod"] = df["timestamp"].map(sec)
    return df.sort_values("sod").reset_index(drop=True)


def rolling(x: np.ndarray, y: np.ndarray, win_s: float, min_n: int = 3):
    """Median of y in win_s-wide bins of x (x in seconds)."""
    if len(x) == 0:
        return x, y
    edges = np.arange(x.min(), x.max() + win_s, win_s)
    idx = np.digitize(x, edges)
    xs, ys = [], []
    for i in np.unique(idx):
        m = idx == i
        if m.sum() >= min_n:
            xs.append(np.median(x[m])); ys.append(np.nanmedian(y[m]))
    return np.array(xs), np.array(ys)


def per_minute(x: np.ndarray, y: np.ndarray):
    """Mean of y per wall-clock minute; returns (minute_start_s, mean, n)."""
    m = (x // 60).astype(int)
    out = pd.DataFrame({"m": m, "y": y}).groupby("m")["y"].agg(["mean", "size"])
    return out.index.to_numpy() * 60.0, out["mean"].to_numpy(), out["size"].to_numpy()


def band_stats(df: pd.DataFrame) -> dict:
    """Per-band vote rate per sensor, mean p, highest-sector p and marginal weights."""
    bands = {
        "day (sun > +6)": df["sun"] > 6,
        "dusk (-6..+6)": (df["sun"] <= 6) & (df["sun"] >= -6),
        "night (sun < -6)": df["sun"] < -6,
        "night trial window 20:27-21:10": (df["sod"] >= sec(NIGHT_TRIALS[0])) & (df["sod"] <= sec(NIGHT_TRIALS[1])),
    }
    out = {}
    for name, m in bands.items():
        sub = df[m]
        if len(sub) == 0:
            continue
        H = np.concatenate(sub["hits"].to_list())  # (n_bins_total, 3)
        P = np.concatenate(sub["p"].to_list())
        X = np.column_stack([np.ones(len(H)), H])
        w, *_ = np.linalg.lstsq(X, P, rcond=None)
        out[name] = {
            "frames": int(len(sub)), "sectors": int(len(P)),
            "from": hhmm(sub["sod"].min()), "to": hhmm(sub["sod"].max()),
            "vote_rate_fisheye": float(H[:, 0].mean()),
            "vote_rate_thermal": float(H[:, 1].mean()),
            "vote_rate_radar": float(H[:, 2].mean()),
            "p_mean": float(P.mean()),
            "p_max_bin_mean": float(sub["p_max"].mean()),
            "vote_mean": float(sub["vote_mean"].mean()),
            "weights_bias_fisheye_thermal_radar": [float(v) for v in w],
        }
    return out


def draw_compact(sec_df, fr, live, stats, landmarks, t0, t1, w, out, height):
    """Three-panel, <= 3.4 in tall variant: availability folded into the evidence row."""
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 1, figsize=(3.5, height), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0, 1.05], "hspace": 0.12,
                                          "left": 0.13, "right": 0.87, "top": 0.985, "bottom": 0.105})
    ax_sun, ax_ev, ax_p = axes
    tn0, tn1 = sec(NIGHT_TRIALS[0]), sec(NIGHT_TRIALS[1])
    sun_set, dark_on, exp_sat, pontoon = (landmarks[k] for k in
                                          ("sunset", "dark_gate_on", "exposure_saturated", "lit_pontoon"))
    for a in axes:
        if np.isfinite(dark_on):
            a.axvspan(dark_on, t1, color=DARK, alpha=0.07, lw=0)
            a.axvline(dark_on, color=DARK, lw=0.6, ls="--")
        a.axvspan(tn0, tn1, color=TRIAL, alpha=0.10, lw=0)
        if np.isfinite(sun_set):
            a.axvline(sun_set, color=WARN, lw=0.6, ls=":")

    # row 1
    xs, ys = rolling(fr["sod"].to_numpy(), fr["sun"].to_numpy(), w)
    ax_sun.plot(xs, ys, color=WARN, lw=1.3)
    ax_sun.axhline(0, color=WARN, lw=0.4, ls=":"); ax_sun.axhline(-6, color=WARN, lw=0.4, ls="--")
    ax_sun.set_ylim(-22, 38); ax_sun.set_yticks([-10, 0, 10, 20, 30])
    ax_sun.set_ylabel("sun elev. (deg)", color=WARN, labelpad=1); ax_sun.tick_params(axis="y", colors=WARN)
    ax_sun.text(t0 + 90, -7.5, "civil twilight", color=WARN, fontsize=5.5, ha="left", va="top")
    ax_sun.text((tn0 + tn1) / 2, 37, "night trials", color=TRIAL, fontsize=5.8, ha="center", va="top")
    if np.isfinite(sun_set):
        ax_sun.text(sun_set - 60, 37, "sunset", color=WARN, fontsize=5.8, ha="right", va="top")
    ax_lum = ax_sun.twinx()
    xs, ys = rolling(fr["sod"].to_numpy(), fr["luminance_mean"].to_numpy(), w)
    ax_lum.plot(xs, np.maximum(ys, 0.04), color=NAVY, lw=1.2)
    ax_lum.set_yscale("log"); ax_lum.set_ylim(0.03, 1500)
    ax_lum.set_yticks([0.1, 1, 10, 100]); ax_lum.set_yticklabels(["0.1", "1", "10", "100"])
    ax_lum.axhline(DARK_THRESH, color=DARK, lw=0.5, ls="--")
    ax_lum.text(sec("19:05:00"), DARK_THRESH * 1.2, f"dark gate (lum. < {DARK_THRESH:g})", color=DARK,
                fontsize=5.5, va="bottom", ha="center")
    ax_lum.set_ylabel("luminance (log)", color=NAVY, labelpad=1); ax_lum.tick_params(axis="y", colors=NAVY)
    if np.isfinite(exp_sat):
        y_at = np.interp(exp_sat, xs, ys)
        ax_lum.plot(exp_sat, y_at, "v", color=NAVY, ms=3.5, mec="white", mew=0.4, zorder=4)
        ax_lum.annotate("exposure saturates", (exp_sat, y_at), (sec("19:22:00"), 0.55), fontsize=5.5,
                        color=NAVY, ha="center", va="top", arrowprops=dict(arrowstyle="-", lw=0.4, color=NAVY))
    if np.isfinite(pontoon):
        ax_lum.annotate("lit pontoon", (pontoon + 20, 60), (sec("20:52:00"), 6), fontsize=5.5, color=NAVY,
                        ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.4, color=NAVY))
    ax_lum.spines[["top"]].set_visible(False); ax_lum.spines["right"].set_color(NAVY)

    # row 2: evidence + vessel mask-fresh fraction
    for col, c, lab in (("fish", FISHEYE, "fisheye"), ("therm", THERMAL, "thermal"), ("radar", RADAR, "radar")):
        xs, ys = rolling(sec_df["sod"].to_numpy(), sec_df[col].to_numpy(), w)
        ax_ev.plot(xs, ys, color=c, lw=1.0, label=lab)
    mx, my, _ = per_minute(live["sod"].to_numpy(), live["seg"].to_numpy())
    ax_ev.step(mx, my, where="post", color=GOOD, lw=0.7, ls="--", label="mask fresh (vessel)")
    ax_ev.set_ylim(0, 1.32); ax_ev.set_yticks([0, 0.5, 1])
    ax_ev.set_ylabel("sectors with a hit", labelpad=1)
    ax_ev.legend(loc="upper left", ncol=4, frameon=False, handlelength=1.1, columnspacing=0.6,
                 bbox_to_anchor=(0.0, 1.04))
    if np.isfinite(dark_on):
        ax_ev.text(dark_on + 45, 0.96, "dark", color=DARK, fontsize=5.5, ha="left", va="top")

    # row 3
    xs, ys = rolling(sec_df["sod"].to_numpy(), sec_df["vote_mean"].to_numpy(), w)
    ax_p.plot(xs, ys, color=MUTED, lw=0.9, ls="--", label="n/3 vote, mean")
    xs, ys = rolling(sec_df["sod"].to_numpy(), sec_df["p_max"].to_numpy(), w)
    ax_p.plot(xs, ys, color=NAVY, lw=0.8, ls=":", label="p, highest sector")
    xs, ys = rolling(sec_df["sod"].to_numpy(), sec_df["p_mean"].to_numpy(), w)
    ax_p.plot(xs, ys, color=NAVY, lw=1.3, label="p, mean")
    ax_p.axhline(0.5, color=WARN, lw=0.5, ls="--")
    ax_p.text(dark_on + 45 if np.isfinite(dark_on) else t1 - 60, 0.52, "0.5", color=WARN, fontsize=5.5,
              ha="left", va="bottom")
    ax_p.set_ylim(0, 1.42); ax_p.set_yticks([0, 0.5, 1])
    ax_p.set_ylabel("learned p(obstacle)", labelpad=1)
    ax_p.legend(loc="upper left", ncol=3, frameon=False, handlelength=1.3, columnspacing=0.7,
                bbox_to_anchor=(0.0, 1.04))
    for name, key, xc in (("day (sun > +6)", "day", "18:10"), ("dusk (-6..+6)", "dusk", "19:40"),
                          ("night trial window 20:27-21:10", "trials", "20:55")):
        if name not in stats:
            continue
        _, wf, wt, wr = stats[name]["weights_bias_fisheye_thermal_radar"]
        line = f"F{wf:+.2f} T{wt:+.2f} R{wr:+.2f}".replace("+0.", "+.").replace("-0.", "-.")
        ax_p.text(sec(xc + ":00"), 1.13, f"{key} weights\n{line}", fontsize=5.3, ha="center", va="top",
                  color=NAVY, linespacing=1.1)

    ticks = np.arange(sec("17:30:00"), t1 + 1, 1800)
    ax_p.set_xticks(ticks); ax_p.set_xticklabels([hhmm(v) for v in ticks])
    ax_p.set_xlim(t0, t1); ax_p.set_xlabel("wall clock, CEST (2026-09-08)", labelpad=1)
    for a in axes:
        a.grid(color=LINE, lw=0.4); a.spines[["top", "right"]].set_visible(False)
        a.tick_params(length=2, pad=1.2)
    ax_sun.spines["right"].set_visible(True); ax_sun.spines["right"].set_color(NAVY)
    fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out, "and", out.with_suffix(".pdf"), f"({fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in)",
          file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sectors", required=True, help="dir of <scene>__<chunk>.jsonl (offline replay)")
    ap.add_argument("--features", required=True, help="dir of <scene>__<chunk>/frames.csv")
    ap.add_argument("--live", default="results/sectors_night", help="dir of the live shadow records (row 4)")
    ap.add_argument("--start", default="17:11")
    ap.add_argument("--end", default="21:36")
    ap.add_argument("--win", type=float, default=60.0, help="rolling window (s)")
    ap.add_argument("--out", default="results/closed_loop/2026-09-08/handover_2026-09-08.png")
    ap.add_argument("--compact", action="store_true",
                    help="three-panel variant (availability folded into the evidence row)")
    ap.add_argument("--height", type=float, default=3.35, help="figure height in inches (compact only)")
    args = ap.parse_args(argv)

    t0 = sec(args.start + ":00"); t1 = sec(args.end + ":00")
    sec_df = load_sectors(args.sectors)
    sec_df = sec_df[(sec_df["sod"] >= t0) & (sec_df["sod"] <= t1)].copy()
    fr = load_frames(args.features)
    fr = fr[(fr["sod"] >= t0) & (fr["sod"] <= t1)].copy()
    live = load_sectors(args.live)
    live = live[(live["sod"] >= t0) & (live["sod"] <= t1)].copy()
    # tail-mode ticks (no shadow camera flag) = no camera path on the box
    for c in ("cam", "seg", "yolo"):
        live[c] = live[c].fillna(False).astype(float)
    print(f"{len(sec_df)} replay records ({sec_df['chunk'].nunique()} chunks), "
          f"{len(fr)} frames, {len(live)} live ticks", file=sys.stderr)

    # sun elevation, cached per 10 s
    cache = {}
    def sun_of(s):
        k = int(s // 10)
        if k not in cache:
            cache[k] = sun_elev(k * 10)
        return cache[k]
    sec_df["sun"] = sec_df["sod"].map(sun_of)
    fr["sun"] = fr["sod"].map(sun_of)

    stats = band_stats(sec_df)

    # landmark times
    sun_set = fr.loc[fr["sun"] < 0, "sod"].min()
    civil = fr.loc[fr["sun"] < -6, "sod"].min()
    exp_max = fr["exposure_time_us"].max()
    m_x, m_exp, _ = per_minute(fr["sod"].to_numpy(), (fr["exposure_time_us"] >= exp_max * 0.999).astype(float).to_numpy())
    exp_sat = m_x[np.argmax(m_exp >= 0.5)] if (m_exp >= 0.5).any() else np.nan
    m_x, m_dark, _ = per_minute(fr["sod"].to_numpy(), fr["is_dark"].astype(float).to_numpy())
    dark_on = m_x[np.argmax(m_dark >= 0.5)] if (m_dark >= 0.5).any() else np.nan
    m_x, m_lum, _ = per_minute(fr["sod"].to_numpy(), fr["luminance_mean"].to_numpy())
    lum_below2 = m_x[np.argmax((m_lum < 2) & (m_x > (dark_on if np.isfinite(dark_on) else 0)))]
    pontoon = m_x[(m_x > sec("21:20:00")) & (m_lum > 50)]
    pontoon = pontoon.min() if len(pontoon) else np.nan
    landmarks = {"sunset": sun_set, "civil_twilight": civil, "exposure_saturated": exp_sat,
                 "dark_gate_on": dark_on, "luminance_below_2": lum_below2, "lit_pontoon": pontoon}
    print({k: (hhmm(v) if np.isfinite(v) else None) for k, v in landmarks.items()}, file=sys.stderr)

    if args.compact:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        draw_compact(sec_df, fr, live, stats, landmarks, t0, t1, args.win, out, args.height)
        return 0

    # ---------- figure ----------
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6.5, "axes.titlesize": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(4, 1, figsize=(3.5, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0, 1.05, 0.5], "hspace": 0.1,
                                          "left": 0.135, "right": 0.865, "top": 0.985, "bottom": 0.07})
    ax_sun, ax_ev, ax_p, ax_av = axes
    w = args.win
    tn0, tn1 = sec(NIGHT_TRIALS[0]), sec(NIGHT_TRIALS[1])

    # shared shading: dark (from the darkness gate) + night-trial window
    for a in axes:
        if np.isfinite(dark_on):
            a.axvspan(dark_on, t1, color=DARK, alpha=0.07, lw=0)
        a.axvspan(tn0, tn1, color=TRIAL, alpha=0.10, lw=0)
        if np.isfinite(sun_set):
            a.axvline(sun_set, color=WARN, lw=0.7, ls=":")
        if np.isfinite(dark_on):
            a.axvline(dark_on, color=DARK, lw=0.7, ls="--")

    # row 1: sun + luminance
    xs, ys = rolling(fr["sod"].to_numpy(), fr["sun"].to_numpy(), w)
    ax_sun.plot(xs, ys, color=WARN, lw=1.4)
    ax_sun.axhline(0, color=WARN, lw=0.5, ls=":")
    ax_sun.axhline(-6, color=WARN, lw=0.5, ls="--")
    ax_sun.set_ylim(-22, 38); ax_sun.set_yticks([-10, 0, 10, 20, 30])
    ax_sun.set_ylabel("sun elev. (deg)", color=WARN, labelpad=2)
    ax_sun.tick_params(axis="y", colors=WARN)
    ax_sun.text(sec("17:25:00"), 8.5, "sun elevation", color=WARN, fontsize=6.5, va="bottom", ha="left")
    ax_sun.text(t0 + 90, -7.0, "civil twilight (-6 deg)", color=WARN, fontsize=6, ha="left", va="top")
    ax_lum = ax_sun.twinx()
    xs, ys = rolling(fr["sod"].to_numpy(), fr["luminance_mean"].to_numpy(), w)
    ax_lum.plot(xs, np.maximum(ys, 0.04), color=NAVY, lw=1.3)
    ax_lum.set_yscale("log"); ax_lum.set_ylim(0.03, 1500)
    ax_lum.set_yticks([0.1, 1, 10, 100]); ax_lum.set_yticklabels(["0.1", "1", "10", "100"])
    ax_lum.axhline(DARK_THRESH, color=DARK, lw=0.6, ls="--")
    ax_lum.text(sec("19:12:00"), DARK_THRESH * 1.2, f"dark gate (lum. < {DARK_THRESH:g})", color=DARK,
                fontsize=6, va="bottom", ha="center")
    ax_lum.set_ylabel("mean luminance (log)", color=NAVY, labelpad=2)
    ax_lum.tick_params(axis="y", colors=NAVY)
    ax_lum.text(sec("18:10:00"), 175, "fisheye luminance", color=NAVY, fontsize=6.5, va="bottom", ha="center")
    if np.isfinite(exp_sat):
        y_at = np.interp(exp_sat, xs, ys)
        ax_lum.plot(exp_sat, y_at, "v", color=NAVY, ms=4, mec="white", mew=0.5, zorder=4)
        ax_lum.annotate(f"exposure saturates\n({exp_max/1000:.1f} ms, gain {fr['analogue_gain'].max():.1f})",
                        (exp_sat, y_at), (sec("19:32:00"), 1.6), fontsize=5.8, color=NAVY, ha="center",
                        va="top", arrowprops=dict(arrowstyle="-", lw=0.5, color=NAVY))
    if np.isfinite(pontoon):
        ax_lum.annotate("lit pontoon", (pontoon + 20, 60), (sec("20:50:00"), 6), fontsize=5.8, color=NAVY,
                        ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=NAVY))
    for sp in ("top",):
        ax_lum.spines[sp].set_visible(False)

    # row 2: evidence rates
    for col, c, lab in (("fish", FISHEYE, "fisheye"), ("therm", THERMAL, "thermal"), ("radar", RADAR, "radar")):
        xs, ys = rolling(sec_df["sod"].to_numpy(), sec_df[col].to_numpy(), w)
        ax_ev.plot(xs, ys, color=c, lw=1.2, label=lab)
    ax_ev.set_ylim(0, 1.3); ax_ev.set_yticks([0, 0.5, 1])
    ax_ev.set_ylabel("sectors with a hit", labelpad=2)
    ax_ev.legend(loc="upper left", ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8,
                 bbox_to_anchor=(0.0, 1.03))

    # row 3: learned p + fused vote
    xs, ys = rolling(sec_df["sod"].to_numpy(), sec_df["vote_mean"].to_numpy(), w)
    ax_p.plot(xs, ys, color=MUTED, lw=1.0, ls="--", label="n/3 vote, mean")
    xs, ys = rolling(sec_df["sod"].to_numpy(), sec_df["p_max"].to_numpy(), w)
    ax_p.plot(xs, ys, color=NAVY, lw=0.9, ls=":", label="p, highest sector")
    xs, ys = rolling(sec_df["sod"].to_numpy(), sec_df["p_mean"].to_numpy(), w)
    ax_p.plot(xs, ys, color=NAVY, lw=1.5, label="p, mean")
    ax_p.axhline(0.5, color=WARN, lw=0.6, ls="--")
    ax_p.text(dark_on + 45 if np.isfinite(dark_on) else t1 - 60, 0.52, "threshold 0.5", color=WARN, fontsize=6,
              ha="left", va="bottom")
    ax_p.set_ylim(0, 1.35); ax_p.set_yticks([0, 0.5, 1])
    ax_p.set_ylabel("learned p(obstacle)", labelpad=2)
    ax_p.legend(loc="upper left", ncol=3, frameon=False, handlelength=1.4, bbox_to_anchor=(0.0, 1.03),
                columnspacing=0.8)
    # per-band marginal sensor weights (least squares of p on the hit flags)
    for name, key, xc in (("day (sun > +6)", "day", "18:10"), ("dusk (-6..+6)", "dusk", "19:33"),
                          ("night trial window 20:27-21:10", "night trials", "20:58")):
        if name not in stats:
            continue
        _, wf, wt, wr = stats[name]["weights_bias_fisheye_thermal_radar"]
        ax_p.text(sec(xc + ":00"), 1.13, f"{key}\nF {wf:+.2f}  T {wt:+.2f}\nR {wr:+.2f}", fontsize=5.6,
                  ha="center", va="top", color=NAVY, linespacing=1.1)

    # row 4: availability on the box (live shadow records)
    for col, c, lw_, lab in (("cam", NAVY, 1.8, "camera path"), ("seg", GOOD, 1.1, "seg mask fresh"),
                             ("yolo", WARN, 0.8, "YOLO fresh")):
        mx, my, _ = per_minute(live["sod"].to_numpy(), live[col].to_numpy())
        ax_av.step(mx, my, where="post", color=c, lw=lw_, label=lab)
    ax_av.set_ylim(-0.08, 2.1); ax_av.set_yticks([0, 1])
    ax_av.set_ylabel("box\navail.", labelpad=2)
    ax_av.legend(loc="upper left", ncol=3, frameon=False, handlelength=1.2, columnspacing=0.7,
                 bbox_to_anchor=(0.0, 1.06))

    # labels for the shaded regions (top row)
    ax_sun.text((tn0 + tn1) / 2, 37.5, "night trials", color=TRIAL, fontsize=6, ha="center", va="top")
    if np.isfinite(sun_set):
        ax_sun.text(sun_set - 60, 37.5, "sunset", color=WARN, fontsize=6, ha="right", va="top")
    if np.isfinite(dark_on):
        ax_ev.text(dark_on + 45, 1.27, "dark", color=DARK, fontsize=6, ha="left", va="top")

    # x axis
    ticks = np.arange(sec("17:30:00"), t1 + 1, 1800)
    ax_av.set_xticks(ticks); ax_av.set_xticklabels([hhmm(v) for v in ticks])
    ax_av.set_xlim(t0, t1)
    ax_av.set_xlabel("wall clock, CEST (2026-09-08)", labelpad=2)
    for a in axes:
        a.grid(color=LINE, lw=0.5); a.spines[["top", "right"]].set_visible(False)
        a.tick_params(length=2, pad=1.5)
    ax_sun.spines["right"].set_visible(True); ax_sun.spines["right"].set_color(NAVY)
    ax_lum.spines[["top"]].set_visible(False)
    ax_lum.spines["right"].set_color(NAVY)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out, "and", out.with_suffix(".pdf"), file=sys.stderr)

    # numbers next to the figure
    summary = {"command": "python -m scripts.eval.handover_figure_0908 " + " ".join(argv or sys.argv[1:]),
               "landmarks": {k: (hhmm(v) if np.isfinite(v) else None) for k, v in landmarks.items()},
               "bands": stats,
               "n_replay_records": int(len(sec_df)), "n_frames": int(len(fr)), "n_live_ticks": int(len(live))}
    # availability per band from the live records
    live["sun"] = live["sod"].map(sun_of)
    av = {}
    for name, m in (("day (sun > +6)", live["sun"] > 6), ("dusk (-6..+6)", (live["sun"] <= 6) & (live["sun"] >= -6)),
                    ("night (sun < -6)", live["sun"] < -6),
                    ("night trial window 20:27-21:10", (live["sod"] >= tn0) & (live["sod"] <= tn1))):
        sub = live[m]
        if len(sub):
            av[name] = {"ticks": int(len(sub)), "camera_path": float(sub["cam"].mean()),
                        "seg_fresh": float(sub["seg"].mean()), "yolo_fresh": float(sub["yolo"].mean())}
    summary["availability_live"] = av
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
