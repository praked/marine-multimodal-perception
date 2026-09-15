#!/usr/bin/env python3
"""Paper figures for the closed-loop avoidance trials (Table IV companion).

Inputs: the boat's ``sector_experiment_*.jsonl`` + ``sector_rx_*.jsonl`` logs
(pulled to the SSD by the water runbook §6) and, optionally, the box's capture
session for camera frames at the decision instant.

Outputs (``--out``):
  summary.png/.pdf        four panels: every leg in a bow-up frame coloured by
                          sun elevation, the evening timeline with light level
                          and outcomes, chosen heading vs obstacle bearing,
                          decision statistics
  trial_<HHMMSS>.png/.pdf per trial: camera at the decision (fisheye; thermal
                          added after sunset), sector field + track (bow-up,
                          metres), per-sector evidence over time with the
                          decision and the leg marked
  trials.csv              one row per launch (Table IV inputs)

    python -m scripts.eval.closed_loop_figures \\
        --logs /Volumes/ROS2_SSD/asvproject/captures/2026-09-08_afloat/boat_logs/boat-a \\
        --capture /Volumes/ROS2_SSD/asvproject/captures/2026-09-08_afloat/2026-09-08_17-10-55 \\
        --out results/closed_loop/2026-09-08 --select 183046 184316 192145 192931 203435 210313 175534
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

CEST = timezone(timedelta(hours=2))
BINS = [-45, -30, -15, 0, 15, 30, 45]
LAT0 = 46.0
END = {"interrupted": "handed back by the crew (RC switch)", "checkpoint reached": "checkpoint reached",
       "station keeping done": "station keeping done", "running": "still running when the log was pulled"}


# ----------------------------------------------------------------- loading
def load_jsonl(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def wall(r):
    return datetime.fromisoformat(r["wall"]).replace(tzinfo=CEST)


def enu(lat, lon, lat0, lon0):
    return ((lon - lon0) * 111320.0 * math.cos(math.radians(lat0)),
            (lat - lat0) * 111320.0)


def rot_bow_up(e, n, heading_deg):
    h = math.radians(heading_deg)
    return e * math.cos(h) - n * math.sin(h), e * math.sin(h) + n * math.cos(h)


def sun_elev(lat, lon, when):
    try:
        from scripts.sensor_processing.gps_boat1 import sun_position
        return sun_position(lat, lon, when.astimezone(timezone.utc))[0]
    except Exception:
        return float("nan")


class Trial:
    def __init__(self, path):
        self.path = path
        self.stamp = os.path.basename(path)[18:33]          # YYYYmmdd_HHMMSS
        self.rows = load_jsonl(path)
        k = lambda kind: [r for r in self.rows if r["kind"] == kind]
        self.start = (k("start") or [{}])[0]
        self.hw = (k("hardware") or [{}])[0]
        self.decides = k("decide")
        self.accepted = [r for r in self.decides if r["result"] in ("ok", "fixed")]
        self.checkpoint = (k("checkpoint") or [None])[-1]
        self.cycles = k("cycle")
        self.station = k("station")
        self.phases = k("phase")
        done = [r for r in self.phases if r["phase"] == "DONE"]
        self.end_why = done[-1].get("why", "") if done else "running"
        self.armed = not self.hw.get("dry_run", True)
        self.evidence = self.hw.get("evidence", "learned")
        self.t_start = wall(self.start) if self.start else None
        self.t_end = wall(self.rows[-1]) if self.rows else None
        d = self.accepted[-1] if self.accepted else None
        self.decision = d
        self.t_decide = wall(d) if d else None
        self.lat0 = d.get("lat") if d else None
        self.lon0 = d.get("lon") if d else None
        if (self.lat0 is None or self.lon0 is None) and self.checkpoint:
            self.lat0, self.lon0 = self.checkpoint["from_lat"], self.checkpoint["from_lon"]
        self.compass = self.checkpoint["compass_deg"] if self.checkpoint else (d.get("heading") if d else None)
        self.rel = d.get("rel_heading_deg") if d else None
        self.p = d.get("p") if d else None
        self.min_range = d.get("min_range_m") if d else None
        self.sun = sun_elev(self.lat0 or LAT0, self.lon0 or 9.0, self.t_decide or self.t_start) if (self.t_decide or self.t_start) else float("nan")
        self.leg_s = len(self.cycles) / 10.0
        self.moved = 0.0
        pts = [(r["lat"], r["lon"]) for r in self.cycles if r.get("lat") is not None]
        if len(pts) > 1:
            e, n = enu(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1])
            self.moved = math.hypot(e, n)

    @property
    def obstacle_bin(self):
        if not self.p:
            return None
        i = int(np.argmax(self.p))
        return BINS[i] if self.p[i] >= 0.6 else None

    def track_bow_up(self, rows=None):
        rows = self.cycles if rows is None else rows
        out = []
        for r in rows:
            if r.get("lat") is None or self.lat0 is None:
                continue
            e, n = enu(r["lat"], r["lon"], self.lat0, self.lon0)
            out.append(rot_bow_up(e, n, self.compass or 0.0) + ((wall(r) - self.t_decide).total_seconds() if self.t_decide else 0.0,))
        return out

    def light(self):
        return "day" if self.sun > 6 else ("dusk" if self.sun > -6 else "night")


# ------------------------------------------------------------- receiver log
def rx_records(logs_dir, t0, t1):
    out = []
    for f in sorted(glob.glob(os.path.join(logs_dir, "sector_rx_*.jsonl"))):
        for r in load_jsonl(f):
            ra = r.get("received_at")
            if not ra:
                continue
            t = datetime.fromisoformat(ra).replace(tzinfo=CEST)
            if t0 <= t <= t1:
                out.append((t, r))
    return out


def evidence_p(r, mode):
    learned, fused = r.get("p_obstacle"), r.get("scores")
    if mode == "max" and learned is not None and fused is not None:
        return [max(a, b) for a, b in zip(learned, fused)]
    return learned if learned is not None else fused


# ------------------------------------------------------------------ frames
def frame_at(capture_dir, when, stream="fisheye"):
    """Frame of the box capture nearest to ``when`` (CEST), or None."""
    import cv2
    best = None
    for f in sorted(glob.glob(os.path.join(capture_dir, "frames_*.csv"))):
        ts = os.path.basename(f)[7:26]
        t_chunk = datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=CEST)
        if t_chunk <= when <= t_chunk + timedelta(minutes=6):
            best = f
    if best is None:
        return None
    ts = os.path.basename(best)[7:26]
    idx = None
    with open(best) as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            t = datetime.strptime(row["Date"] + " " + row["Time"], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=CEST)
            if t >= when:
                idx = int(row["frame_index"]); break
            idx = int(row["frame_index"])
    vid = os.path.join(capture_dir, f"{stream}_{ts}.mp4")
    rec = os.path.join(capture_dir, "recovered", f"{stream}_{ts}.mp4")
    if os.path.exists(rec):
        vid = rec
    if not os.path.exists(vid) or idx is None:
        return None
    cap = cv2.VideoCapture(vid)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, n - 1)))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    if stream == "thermal":
        # Lepton stuck row (sensor row 21 -> row 98 of the rotated frame, since 2026-07-14)
        lo, hi = 96, 100
        if hi < fr.shape[0]:
            for r in range(lo + 1, hi):
                w = (r - lo) / (hi - lo)
                fr[r] = ((1 - w) * fr[lo].astype(float) + w * fr[hi].astype(float)).astype(fr.dtype)
    elif fr.mean() < 40:
        # night frame: display gamma so the scene is visible (the detector saw the raw frame)
        fr = (255.0 * (fr / 255.0) ** 0.4).astype(np.uint8)
    return fr


# ----------------------------------------------------------------- drawing
def draw_field(ax, tr, radius=9.0, label_bins=True):
    from matplotlib.patches import Wedge
    import matplotlib.cm as cm
    if tr.p:
        for b, p, mr in zip(BINS, tr.p, tr.min_range or [None] * 7):
            # bow up: bearing b (clockwise from bow) -> matplotlib angle 90 - b
            w = Wedge((0, 0), radius, 90 - b - 7.5, 90 - b + 7.5, facecolor=cm.Reds(0.15 + 0.85 * p),
                      edgecolor="white", linewidth=0.6, alpha=0.9, zorder=1)
            ax.add_patch(w)
            if mr is not None:
                a = math.radians(90 - b)
                ax.plot(mr * math.cos(a), mr * math.sin(a), "k|", ms=8, mew=1.5, zorder=3)
            if label_bins:
                a = math.radians(90 - b)
                rr = radius + (2.4 if (BINS.index(b) % 2 == 0) else 4.4)
                ax.text(rr * math.cos(a), rr * math.sin(a), f"{p:.2f}",
                        ha="center", va="center", fontsize=6.5, color="#333")
    ax.plot([0, 0], [0, radius], color="#888", lw=0.8, ls=":", zorder=2)


def draw_trial_axes(ax, tr, arrow_len=20.0):
    draw_field(ax, tr)
    if tr.rel is not None:
        a = math.radians(90 - tr.rel)
        ax.annotate("", xy=(arrow_len * math.cos(a), arrow_len * math.sin(a)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#1487B8"), zorder=4)
        ax.text(arrow_len * 1.06 * math.cos(a), arrow_len * 1.06 * math.sin(a), f"{tr.rel:+.0f}°",
                color="#1487B8", fontsize=9, ha="center", va="center", fontweight="bold")
    trk = tr.track_bow_up()
    if trk:
        xs, ys, ts = zip(*trk)
        sc = ax.scatter(xs, ys, c=ts, cmap="viridis", s=9, zorder=5)
        ax.plot(xs, ys, color="#2a2a2a", lw=0.8, alpha=0.6, zorder=4)
        ax.plot(xs[-1], ys[-1], "s", color="#2a2a2a", ms=5, zorder=6)
    stn = tr.track_bow_up(tr.station)
    if stn:
        xs, ys, _ = zip(*stn)
        ax.plot(xs, ys, color="#B8741A", lw=1.2, ls="--", alpha=0.9, zorder=4, label="station keeping")
    ax.plot(0, 0, "k^", ms=8, zorder=7, label="boat at the decision (bow up)")
    ax.plot([], [], color="#1487B8", lw=2, label="chosen heading")
    ax.plot([], [], color="#2a2a2a", lw=1, label="leg (colour = time)")
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.85)
    ax.set_aspect("equal")
    lim = 24
    ax.set_xlim(-lim, lim); ax.set_ylim(-12, lim + 8)
    ax.set_xlabel("starboard  →  (m)"); ax.set_ylabel("ahead at the decision  →  (m)")
    ax.grid(alpha=0.25)


def draw_evidence(ax, tr, logs_dir):
    hold0 = [r for r in tr.phases if r["phase"] == "HOLD"]
    t_a = (wall(hold0[0]) if hold0 else tr.t_start) - timedelta(seconds=5)
    last = (tr.station or tr.cycles or [None])[-1]
    t_b = (wall(last) if last else tr.t_end) + timedelta(seconds=5)
    recs = rx_records(logs_dir, t_a, min(t_b, tr.t_end))
    if not recs:
        ax.text(0.5, 0.5, "no receiver records", transform=ax.transAxes, ha="center"); return
    t0 = tr.t_decide or tr.t_start
    ts = np.array([(t - t0).total_seconds() for t, _ in recs])
    P = np.array([evidence_p(r, tr.evidence) or [np.nan] * 7 for _, r in recs], dtype=float).T
    im = ax.pcolormesh(np.append(ts, ts[-1] + 1), np.arange(8) - 0.5, P, cmap="Reds", vmin=0, vmax=1, shading="flat")
    ax.set_yticks(range(7)); ax.set_yticklabels([f"{b:+d}°" for b in BINS]); ax.invert_yaxis()
    ax.axvline(0, color="#1487B8", lw=2)
    ax.text(0.6, 6.9, "decision", color="#1487B8", fontsize=8, va="top")
    ax.set_xlim(ts[0], ts[-1] + 1)
    hold = [r for r in tr.phases if r["phase"] == "HOLD"]
    if hold:
        ax.axvline((wall(hold[0]) - t0).total_seconds(), color="#444", lw=1, ls=":")
    if tr.cycles:
        te = (wall(tr.cycles[-1]) - t0).total_seconds()
        ax.axvspan(0, te, color="#1487B8", alpha=0.13)
        ax.text(te / 2, -0.35, "leg", color="#0D5E80", fontsize=8, ha="center", va="top")
    if tr.station:
        ts0, ts1 = (wall(tr.station[0]) - t0).total_seconds(), (wall(tr.station[-1]) - t0).total_seconds()
        ax.axvspan(ts0, ts1, color="#B8741A", alpha=0.16)
        ax.text((ts0 + ts1) / 2, -0.35, "station keeping", color="#8a5512", fontsize=8, ha="center", va="top")
    if tr.rel is not None:
        y = BINS.index(min(BINS, key=lambda b: abs(b - tr.rel)))
        ax.plot(0, y, "o", color="#1487B8", ms=7, mfc="white", mew=2, zorder=5)
    ax.set_xlabel("time from the decision (s)")
    ax.set_ylabel("sector (bearing from the bow)")
    return im


def per_trial_figure(tr, logs_dir, capture_dir, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    night = tr.sun < -6
    ncam = 2 if (capture_dir and night) else 1
    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(1, 3 + (1 if ncam == 2 else 0), width_ratios=[1.3] * ncam + [1.25, 1.6], wspace=0.28)
    col = 0
    if capture_dir:
        for stream in (["fisheye", "thermal"] if night else ["fisheye"]):
            ax = fig.add_subplot(gs[0, col]); col += 1
            fr = frame_at(capture_dir, tr.t_decide or tr.t_start, stream)
            if fr is not None:
                ax.imshow(fr, interpolation="nearest")
            else:
                ax.text(0.5, 0.5, f"no {stream} frame", ha="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{stream} at the decision", fontsize=10)
    ax = fig.add_subplot(gs[0, col]); col += 1
    draw_trial_axes(ax, tr)
    ax.set_title("sector field and track (bow up)", fontsize=10)
    ax = fig.add_subplot(gs[0, col]); col += 1
    im = draw_evidence(ax, tr, logs_dir)
    ax.set_title(f"evidence per sector ({tr.evidence})", fontsize=10)
    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("p(obstacle)")
    when = (tr.t_decide or tr.t_start).strftime("%H:%M:%S")
    fig.suptitle(f"{when} CEST · sun {tr.sun:+.1f}° ({tr.light()}) · chosen heading {tr.rel:+.0f}° relative "
                 f"(compass {tr.compass:.0f}° → {(tr.compass + tr.rel) % 360:.0f}°) · leg {tr.leg_s:.0f} s, {tr.moved:.0f} m · "
                 f"end: {END.get(tr.end_why, tr.end_why)}", fontsize=10, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"trial_{tr.stamp[9:]}.{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def summary_figure(trials, out, legs_only=False):
    """legs_only: show driven legs only (no refused-launch markers, bars or counts)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    acc = [t for t in trials if t.decision and t.armed and t.cycles]
    norm = Normalize(vmin=-12, vmax=25)
    cmap = cm.get_cmap("plasma_r") if hasattr(cm, "get_cmap") else plt.get_cmap("plasma_r")
    fig, axs = plt.subplots(2, 3, figsize=(19, 10.5))
    # E: absolute positions of every driven leg (local metres about the pontoon)
    axE = axs[0, 1]
    ref = [t for t in trials if t.t_start and t.t_start.hour == 17 and t.lat0]
    lat_r, lon_r = (ref[0].lat0, ref[0].lon0) if ref else (acc[0].lat0, acc[0].lon0)
    for t in acc:
        pts = [(r["lat"], r["lon"]) for r in t.cycles + t.station if r.get("lat") is not None]
        if not pts:
            continue
        xy = [enu(la, lo, lat_r, lon_r) for la, lo in pts]
        xs, ys = zip(*xy)
        c = cmap(norm(t.sun))
        axE.plot(xs, ys, color=c, lw=1.4, alpha=0.9)
        axE.plot(xs[0], ys[0], "^", color=c, ms=6, mec="k", mew=0.4)
    axE.plot(0, 0, "ks", ms=7); axE.text(1.5, 1.5, "pontoon (17:21 fix)", fontsize=8)
    axE.set_aspect("equal"); axE.grid(alpha=0.25)
    axE.set_xlabel("east of the pontoon (m)"); axE.set_ylabel("north of the pontoon (m)")
    axE.set_title("E  Where the legs were run (▲ = decision point), coloured by sun elevation", fontsize=10, loc="left")
    # F: numbers card
    axF = axs[1, 2]; axF.axis("off")
    armed = [t for t in trials if t.armed]
    refused = [t for t in armed if t.decides and not t.cycles]
    windows = sum(len(t.decides) for t in armed)
    ok_w = sum(len(t.accepted) for t in armed)
    mx = [t for t in acc if t.evidence == "max"]
    lines = [
        "Trials 2026-09-08, boat-a, the lake",
        *([f"legs driven: {len(acc)}"] if legs_only else
          [f"armed launches: {len(armed)}   hold windows: {windows}   accepted: {ok_w}",
           f"legs driven: {len(acc)}   refused launches: {len(refused)}"]),
        f"legs by light: day {sum(t.light()=='day' for t in acc)}, dusk {sum(t.light()=='dusk' for t in acc)}, night {sum(t.light()=='night' for t in acc)}",
        f"distance under autonomy: {sum(t.moved for t in acc):.0f} m in {sum(t.leg_s for t in acc):.0f} s",
        f"station keeping after the leg: {sum(1 for t in acc if t.station)} legs, {sum(len(t.station) for t in acc)} s",
        "",
        "evidence = max (from 18:20): " + f"{len(mx)} legs, median |heading| {np.median([abs(t.rel) for t in mx]):.0f}°,",
        f"   {sum(1 for t in mx if t.obstacle_bin is not None and abs(t.rel - t.obstacle_bin) >= 15)} of {sum(1 for t in mx if t.obstacle_bin is not None)} with an obstacle sector turned ≥ 15° away from it",
        "learned evidence only (before 18:20): " + f"{sum(1 for t in acc if t.evidence != 'max')} legs, {sum(1 for t in acc if t.evidence != 'max' and t.rel == 0)} straight ahead",
        "   (radar returned nothing beyond 1 m; the scorer floors camera-only evidence)",
        "",
        "hold 10–20 s, throttle 1600–1900 µs, checkpoint 15–20 m, abort rule off,",
        "heading fallback level, box seg-primary / typed, 15° sectors",
    ]
    axF.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9.5, family="monospace", transform=axF.transAxes)
    axF.set_title("F  Numbers", fontsize=10, loc="left")
    # A: every leg, bow up
    ax = axs[0, 0]
    for t in acc:
        c = cmap(norm(t.sun))
        trk = t.track_bow_up()
        if trk:
            xs, ys, _ = zip(*trk)
            ax.plot(xs, ys, color=c, lw=1.6, alpha=0.9, ls=("-" if t.evidence == "max" else (0, (2, 2))))
            ax.plot(xs[-1], ys[-1], "o", color=c, ms=4)
        if t.rel is not None:
            a = math.radians(90 - t.rel)
            ax.annotate("", xy=(6 * math.cos(a), 6 * math.sin(a)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", lw=1, color=c, alpha=0.8))
        ob = t.obstacle_bin
        if ob is not None:
            a = math.radians(90 - ob)
            ax.plot([9.6 * math.cos(a), 10.8 * math.cos(a)], [9.6 * math.sin(a), 10.8 * math.sin(a)], color="#B3392B", lw=2.5, alpha=0.8)
    for b in BINS:
        a = math.radians(90 - b)
        ax.plot([0, 9 * math.cos(a)], [0, 9 * math.sin(a)], color="#ccc", lw=0.6, ls=":")
    th = np.linspace(math.radians(90 - 52.5), math.radians(90 + 52.5), 60)
    ax.plot(9 * np.cos(th), 9 * np.sin(th), color="#999", lw=0.8)
    ax.plot(0, 0, "k^", ms=9)
    ax.plot([], [], color="#555", lw=1.6, label="evidence = max (from 18:20)")
    ax.plot([], [], color="#555", lw=1.6, ls=(0, (2, 2)), label="evidence = learned only (radar-blind: 6 of 8 went straight)")
    ax.plot([], [], color="#B3392B", lw=2.5, label="obstacle sector at the decision (p ≥ 0.6)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_aspect("equal"); ax.set_xlim(-30, 30); ax.set_ylim(-8, 42)
    ax.set_xlabel("starboard → (m)"); ax.set_ylabel("ahead at the decision → (m)"); ax.grid(alpha=0.25)
    ax.set_title(f"A  Every driven leg, bow-up frame (n={len(acc)})", fontsize=10, loc="left")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02); cb.set_label("sun elevation at the decision (°)")
    # B: timeline
    ax = axs[0, 2]
    lat, lon = LAT0, 9.0
    ts = [datetime(2026, 9, 8, 17, 0, tzinfo=CEST) + timedelta(minutes=m) for m in range(0, 260, 2)]
    el = [sun_elev(lat, lon, t) for t in ts]
    hours = [t.hour + t.minute / 60 for t in ts]
    ax.plot(hours, el, color="#B8741A", lw=1.5, label="sun elevation")
    ax.axhline(0, color="#999", lw=0.6); ax.axhline(-6, color="#999", lw=0.6, ls="--")
    ax.axvspan(17, 21.2, ymin=0, ymax=1, color="none")
    for t in trials:
        if not t.armed or t.t_start is None:
            continue
        h = t.t_start.hour + t.t_start.minute / 60 + t.t_start.second / 3600
        if t.decision and t.cycles:
            ax.plot(h, t.sun, "o", color=cmap(norm(t.sun)), ms=5 + 0.35 * t.moved, mec="k", mew=0.5, zorder=4)
        elif t.decides and not legs_only:
            ax.plot(h, t.sun, "o", color="white", mec="#B3392B", mew=1.2, ms=6, zorder=4)
    for h, lab in ((18 + 20 / 60, "evidence = max"), (18 + 43 / 60, "station keeping"), (19 + 25 / 60, "max-spread 0.5")):
        ax.axvline(h, color="#1487B8", lw=1, ls=":")
        ax.text(h + 0.03, -14.5, lab, color="#1487B8", fontsize=7.5, rotation=90, va="bottom", ha="left")
    ax.set_xlim(17.2, 21.2); ax.set_ylim(-15, 28)
    ax.set_xlabel("time of day (CEST)"); ax.set_ylabel("sun elevation (°)")
    ax.text(21.1, 0.5, "sunset", ha="right", fontsize=8, color="#666"); ax.text(21.1, -5.5, "civil dusk", ha="right", fontsize=8, color="#666")
    ax.set_title("B  Legs over the evening (marker size ∝ distance)" if legs_only else "B  Launches over the evening: filled = leg driven (size ∝ distance), hollow = every window refused", fontsize=10, loc="left")
    ax.grid(alpha=0.25)
    # C: chosen heading vs obstacle bearing
    ax = axs[1, 0]
    for t in acc:
        ob = t.obstacle_bin
        if ob is None:
            ax.plot(0, t.rel, "x", color="#999", ms=7)
        elif t.evidence == "max":
            ax.plot(ob, t.rel, "o", color=cmap(norm(t.sun)), ms=8, mec="k", mew=0.5)
        else:
            ax.plot(ob, t.rel, "o", color="white", ms=8, mec=cmap(norm(t.sun)), mew=1.8)
    xs = np.linspace(-50, 50, 3)
    ax.plot(xs, xs, color="#B3392B", lw=1, ls="--", label="toward the obstacle")
    ax.axhline(0, color="#999", lw=0.6); ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlim(-55, 55); ax.set_ylim(-55, 55); ax.set_aspect("equal")
    ax.set_xlabel("bearing of the strongest sector, p ≥ 0.6 (°; × = no sector above 0.6)")
    ax.set_ylabel("chosen heading, relative to the bow (°)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.25)
    ax.set_title("C  Where it went vs where the obstacle was (hollow = learned evidence only)", fontsize=10, loc="left")
    # D: statistics
    ax = axs[1, 1]
    by = {"day": [], "dusk": [], "night": []}
    for t in acc:
        by[t.light()].append(t.rel)
    labels = list(by)
    vals = [len(by[k]) for k in labels]
    ref = {"day": 0, "dusk": 0, "night": 0}
    for t in trials:
        if t.armed and t.decides and not t.cycles:
            ref[t.light()] += 1
    x = np.arange(3)
    xoff = 0.0 if legs_only else -0.18
    ax.bar(x + xoff, vals, 0.5 if legs_only else 0.36, color="#1487B8", label="legs driven")
    if not legs_only:
        ax.bar(x + 0.18, [ref[k] for k in labels], 0.36, color="#B3392B", alpha=0.7, label="launches refused")
    for i, k in enumerate(labels):
        if by[k]:
            hs = [f"{v:+.0f}" for v in sorted(by[k])]
            lines = [", ".join(hs[j:j + 6]) for j in range(0, len(hs), 6)]
            ax.text(i + xoff, vals[i] + 0.15, "\n".join(lines), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{k}\n(sun > 6°)" if k == "day" else (f"{k}\n(6° … −6°)" if k == "dusk" else f"{k}\n(sun < −6°)") for k in labels])
    ax.set_ylabel("legs" if legs_only else "launches"); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, max(vals + ([] if legs_only else [ref[k] for k in labels])) * 1.35 + 1)
    med = np.median([abs(t.rel) for t in acc if t.rel is not None]) if acc else float("nan")
    ax.set_title(f"D  Outcome by light level · median |heading| {med:.0f}° · legs {sum(t.moved for t in acc):.0f} m total", fontsize=10, loc="left")
    fig.suptitle("Closed-loop avoidance trials, the lake, 2026-09-08 (boat-a, box in full/typed mode)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"summary.{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_csv(trials, out):
    with open(os.path.join(out, "trials.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start", "name", "armed", "throttle_us", "evidence", "sun_elev_deg", "light", "windows", "accepted",
                    "rel_heading_deg", "obstacle_bin_deg", "p_at_bins", "leg_s", "moved_m", "station_s", "end"])
        for t in trials:
            w.writerow([t.t_start.strftime("%H:%M:%S") if t.t_start else "", t.start.get("name", ""), int(t.armed),
                        t.hw.get("throttle_us"), t.evidence, f"{t.sun:.1f}", t.light(), len(t.decides), len(t.accepted),
                        t.rel, t.obstacle_bin, " ".join(f"{p:.2f}" for p in (t.p or [])), f"{t.leg_s:.0f}", f"{t.moved:.1f}",
                        len(t.station), t.end_why])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--capture", default=None, help="box capture session dir (frames at the decision)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", default="17:25", help="ignore launches before this time (HH:MM)")
    ap.add_argument("--select", nargs="*", default=[], help="HHMMSS stamps for per-trial figures (default: every driven leg)")
    ap.add_argument("--legs-only", action="store_true", help="summary shows driven legs only (no refused-launch markers, bars or counts)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    trials = [Trial(p) for p in sorted(glob.glob(os.path.join(a.logs, "sector_experiment_*.jsonl")))]
    hh, mm = map(int, a.since.split(":"))
    trials = [t for t in trials if t.t_start and (t.t_start.hour, t.t_start.minute) >= (hh, mm)]
    write_csv(trials, a.out)
    summary_figure(trials, a.out, legs_only=a.legs_only)
    sel = set(a.select)
    for t in trials:
        if (sel and t.stamp[9:] in sel) or (not sel and t.decision and t.cycles):
            if t.decision:
                per_trial_figure(t, a.logs, a.capture, a.out)
                print("trial figure", t.stamp[9:], t.light(), f"rel {t.rel:+.0f}", f"{t.moved:.0f} m")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
