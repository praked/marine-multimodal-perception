"""Score the per-target motion classifier against ground-truth clips.

The `motion.targets` tracker (scripts/sensor_processing/target_motion.py)
claims closing / crossing / diverging / static per tracked object. This tool
measures that claim on clips with known motion, radar-only (no video decode:
target motion is radar-driven), and prints the measured noise floors that the
detection.yaml thresholds must trace to.

Ground truth, two modes:

  - `--gt <file.json>`: explicit windows, for clips with known geometry
    (the 2026-08-19 canoe crossing). Format:
        {"windows": [{"t0": 12.0, "t1": 34.5, "state": "crossing",
                      "bearing_min": -40, "bearing_max": 40}, ...]}
    t0/t1 are seconds from the first radar timestamp (floats) or absolute
    "HH:MM:SS.f" strings; bearing_min/max (optional) restrict which target
    is scored inside the window (nearest-range target in the span).

  - `--auto-radial-gt`: for pure-radial clips (the 2026-07-06 pacing tests:
    alternating approach/recede blocks). Per-frame GT is derived from the
    POSITION channel: an LS slope of the median raw range over a sliding
    window (closing if < -thresh, diverging if > +thresh, else static).
    Position-derived, so it does not consume the Doppler values the
    classifier's radial channel uses (shared physics, independent
    measurement path).

Every GT frame also contributes noise diagnostics: radial and crossing GT
windows should carry ~zero and ~nonzero tangential speed respectively, so
the static/radial-window |v_tangential| p95 IS the crossing threshold's
noise floor. The summary prints a recommended `crossing_min_tangential_mps`.
When the clip has an IMU sidecar the whole run is repeated with the yaw
correction disabled: on a turning boat the corrected variant must show the
LOWER static-window tangential noise (arbitrates `imu_yaw_sign`).

Usage:
    python -m scripts.eval.target_motion_eval \\
        --triplet data/captures/2026-08-19_afloat/canoe_fx/2026-08-19_15-58-20 \\
        --gt labels/motion_gt/canoe_2026-08-19_15-58-20.json \\
        --out results/target_motion

    python -m scripts.eval.target_motion_eval \\
        --triplet data/captures/<pacing-clip> --auto-radial-gt \\
        --no-self-clutter --out results/target_motion
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

from scripts.sensor_processing.imu_bno085 import load_imu_config
from scripts.sensor_processing.imu_replay import ImuLogAttitudeProvider
from scripts.sensor_processing.motion import parse_radar_timestamp
from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import load_mmwave_csv, resolve_triplet

STATES = ["closing", "crossing", "diverging", "static", "unknown"]


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_gt_windows(path: Path, t0_clip: float) -> list[dict]:
    """Parse a GT file into [{t0_s, t1_s, state, bearing_min, bearing_max}]
    with times in absolute seconds-of-day."""
    spec = json.loads(Path(path).read_text())
    out = []
    for w in spec.get("windows", []):
        def _abs(v):
            if isinstance(v, str):
                t = parse_radar_timestamp(v)
                if t is None:
                    raise ValueError(f"unparseable GT time {v!r}")
                return t
            return t0_clip + float(v)
        out.append({
            "t0_s": _abs(w["t0"]),
            "t1_s": _abs(w["t1"]),
            "state": str(w["state"]),
            "bearing_min": w.get("bearing_min"),
            "bearing_max": w.get("bearing_max"),
        })
    return out


def auto_radial_gt(frame_times: list[float], median_ranges: list[float],
                   window_s: float = 2.0, slope_thresh: float = 0.1,
                   transition_guard_s: float = 1.0
                   ) -> list[str | None]:
    """Per-frame closing/diverging/static GT from the raw median-range slope.

    `slope_thresh` (m/s) mirrors closing_min_mps territory but is applied to
    a POSITION-derived slope; the 2026-07-06 pacing blocks moved at
    0.24-0.97 m/s so 0.1 separates them cleanly from the static holds.
    None where the window has too few frames to fit.

    Frames within `transition_guard_s` of a GT state change are blanked to
    None: at a block flip both the GT slope window and the tracker's rate
    window straddle two motion regimes, so neither label is meaningful
    there. Measured on the pacing clip: 95/117 GT frames sat within 1 s of
    a transition (the subject turns every few seconds) and carried 55 of
    the 64 apparent misclassifications: scoring them would measure the
    smearing of two low-pass filters against each other, not the
    classifier. 0 disables the guard.
    """
    t = np.asarray(frame_times)
    r = np.asarray(median_ranges, dtype=float)
    out: list[str | None] = []
    for i in range(len(t)):
        sel = (t >= t[i] - window_s / 2) & (t <= t[i] + window_s / 2) \
            & np.isfinite(r)
        if sel.sum() < 5:
            out.append(None)
            continue
        tt, rr = t[sel] - t[sel].mean(), r[sel]
        denom = float(np.dot(tt, tt))
        if denom <= 0:
            out.append(None)
            continue
        slope = float(np.dot(tt, rr - rr.mean()) / denom)
        if slope <= -slope_thresh:
            out.append("closing")
        elif slope >= slope_thresh:
            out.append("diverging")
        else:
            out.append("static")
    if transition_guard_s > 0:
        guarded = list(out)
        for i in range(len(t)):
            if out[i] is None:
                continue
            near = np.abs(t - t[i]) <= transition_guard_s
            if any(out[j] is not None and out[j] != out[i]
                   for j in np.where(near)[0]):
                guarded[i] = None
        out = guarded
    return out


# ---------------------------------------------------------------------------
# Radar-only clip iteration
# ---------------------------------------------------------------------------

def radar_frames(triplet):
    """Yield (timestamp, points) per radar frame, same column contract as
    iterate_triplet (Nx3 / Nx4 +V / Nx6 +V,SNR,NOISE)."""
    df = load_mmwave_csv(triplet.mmwave)
    if df is None or df.empty:
        return
    point_cols = ["X", "Y", "Z"]
    if "V" in df.columns:
        point_cols.append("V")
        if "SNR" in df.columns and "NOISE" in df.columns:
            point_cols += ["SNR", "NOISE"]
    grouped = df.groupby("RoundedTime")
    for ts in sorted(grouped.groups.keys()):
        sub = grouped.get_group(ts)[point_cols].dropna(subset=["X", "Y", "Z"])
        pts = (sub.values.astype(np.float64) if len(sub)
               else np.empty((0, len(point_cols))))
        yield ts, pts


def run_variant(triplet, detection, intrinsics, attitude, label: str
                ) -> list[dict]:
    """One full tracker pass; returns per-frame rows for every reported
    target plus a row (target_id None) for target-free frames."""
    pipeline = ObstacleDetectionPipeline(intrinsics, detection,
                                         attitude_provider=attitude)
    rows: list[dict] = []
    for ts, pts in radar_frames(triplet):
        if attitude is not None:
            attitude.set_time(ts)
        res = pipeline.process_frame(None, None, pts, timestamp=ts)
        t_s = parse_radar_timestamp(ts)
        m = res.mmwave
        med_r = (float(np.median(m.ranges)) if m is not None and m.ranges
                 else float("nan"))
        base = {"variant": label, "timestamp": ts, "t_s": t_s,
                "n_points": len(m.points_xyz) if m is not None else 0,
                "median_range_m": med_r,
                "ego_yaw_deg": (res.targets.ego_yaw_deg
                                if res.targets is not None else None)}
        tgts = res.targets.targets if res.targets is not None else []
        if not tgts:
            rows.append({**base, "target_id": None})
            continue
        for t in tgts:
            rows.append({**base, "target_id": t.track_id,
                         "bearing_deg": t.bearing_deg, "range_m": t.range_m,
                         "closing_mps": t.closing_mps,
                         "v_tangential_mps": t.v_tangential_mps,
                         "speed_mps": t.speed_mps, "cpa_m": t.cpa_m,
                         "t_cpa_s": t.t_cpa_s, "state": t.motion_state,
                         "ego_corrected": t.ego_corrected,
                         "age_frames": t.age_frames,
                         "n_cluster_points": t.n_points})
    return rows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def pick_target(frame_rows: list[dict], window: dict | None) -> dict | None:
    """The reported target a GT statement is about.

    Windowed GT (known geometry): nearest-range target inside the window's
    bearing span. Whole-scene radial GT (auto mode, window None): the
    DOMINANT target by cluster point count: the auto GT derives from the
    whole-frame median range, which the biggest cluster drives; on the
    dense 2026-07-06 bench scene nearest-range instead picks foreground
    clutter over the pacer (measured: this halved apparent closing recall).
    """
    cands = [r for r in frame_rows if r.get("target_id") is not None]
    if window is not None:
        lo, hi = window.get("bearing_min"), window.get("bearing_max")
        if lo is not None:
            cands = [r for r in cands if r["bearing_deg"] >= float(lo)]
        if hi is not None:
            cands = [r for r in cands if r["bearing_deg"] <= float(hi)]
        return min(cands, key=lambda r: r["range_m"]) if cands else None
    return (max(cands, key=lambda r: r["n_cluster_points"])
            if cands else None)


def score_variant(rows: list[dict], gt_of_frame) -> dict:
    """Confusion matrix + tangential-noise stats for one variant.

    `gt_of_frame(t_s, frame_rows)` -> (state | None, window | None).
    """
    by_frame: dict[float, list[dict]] = {}
    for r in rows:
        by_frame.setdefault(r["t_s"], []).append(r)

    confusion = {g: {p: 0 for p in STATES} for g in STATES}
    missed = 0
    n_gt = 0
    vt_radial: list[float] = []   # |v_t| where GT says no tangential motion
    vt_crossing: list[float] = []
    # (t, bearing, yaw) on crossing-GT frames: the section-5.3 aliasing
    # diagnostic. On a non-turning boat the subject's DETRENDED boat-frame
    # bearing should be uncorrelated with the reported yaw; a genuine turn
    # correlates them (bearings sweep at the yaw rate).
    yaw_diag: list[tuple[float, float, float]] = []
    for t_s in sorted(by_frame):
        gt, window = gt_of_frame(t_s, by_frame[t_s])
        if gt is None:
            continue
        n_gt += 1
        tgt = pick_target(by_frame[t_s], window)
        if tgt is None:
            missed += 1
            continue
        confusion[gt][tgt.get("state", "unknown")] += 1
        vt = tgt.get("v_tangential_mps")
        if vt is not None:
            if gt in ("static", "closing", "diverging"):
                vt_radial.append(abs(vt))
            elif gt == "crossing":
                vt_crossing.append(abs(vt))
        yaw = tgt.get("ego_yaw_deg")
        if gt == "crossing" and yaw is not None:
            yaw_diag.append((t_s, float(tgt["bearing_deg"]), float(yaw)))

    def _pct(vals, q):
        return float(np.percentile(vals, q)) if vals else None

    yaw_bearing_r = None
    if len(yaw_diag) >= 8:
        t = np.array([d[0] for d in yaw_diag])
        b = np.array([d[1] for d in yaw_diag])
        y = np.array([d[2] for d in yaw_diag])
        y = np.degrees(np.unwrap(np.radians(y)))

        def _detrend(v):
            coef = np.polyfit(t, v, 1)
            return v - np.polyval(coef, t)

        db, dy = _detrend(b), _detrend(y)
        if float(np.std(db)) > 0 and float(np.std(dy)) > 0:
            yaw_bearing_r = float(np.corrcoef(db, dy)[0, 1])

    return {
        "n_gt_frames": n_gt,
        "n_missed": missed,
        "confusion": confusion,
        "vt_radial_p50": _pct(vt_radial, 50),
        "vt_radial_p95": _pct(vt_radial, 95),
        "vt_crossing_p50": _pct(vt_crossing, 50),
        "n_vt_radial": len(vt_radial),
        "n_vt_crossing": len(vt_crossing),
        "yaw_bearing_r": yaw_bearing_r,
        "n_yaw_diag": len(yaw_diag),
    }


def per_state_pr(confusion: dict) -> dict[str, tuple[float | None, float | None]]:
    out = {}
    for s in STATES:
        tp = confusion[s][s]
        fn = sum(confusion[s].values()) - tp
        fp = sum(confusion[g][s] for g in STATES if g != s)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        out[s] = (prec, rec)
    return out


def render_summary(clip_id: str, variant_stats: dict[str, dict]) -> str:
    lines = [f"# target_motion_eval: {clip_id}", ""]
    for label, st in variant_stats.items():
        lines.append(f"## variant: {label}")
        lines.append(f"- GT frames: {st['n_gt_frames']}  "
                     f"(no target reported on {st['n_missed']})")
        lines.append("- confusion (rows = GT, cols = predicted):")
        header = "  | GT\\pred | " + " | ".join(STATES) + " |"
        lines.append(header)
        lines.append("  |" + "---|" * (len(STATES) + 1))
        for g in STATES:
            row = st["confusion"][g]
            if not any(row.values()):
                continue
            lines.append(f"  | {g} | " +
                         " | ".join(str(row[p]) for p in STATES) + " |")
        for s, (p, r) in per_state_pr(st["confusion"]).items():
            if p is None and r is None:
                continue
            lines.append(
                f"- {s}: precision "
                f"{'n/a' if p is None else f'{p:.3f}'} / recall "
                f"{'n/a' if r is None else f'{r:.3f}'}")
        if st["vt_radial_p95"] is not None:
            lines.append(
                f"- |v_tangential| on radial/static GT frames "
                f"(n={st['n_vt_radial']}): p50 {st['vt_radial_p50']:.3f} / "
                f"p95 {st['vt_radial_p95']:.3f} m/s")
            lines.append(
                f"  -> recommended crossing_min_tangential_mps ~ "
                f"{1.5 * st['vt_radial_p95']:.2f} (1.5x the p95 noise floor)")
        if st["vt_crossing_p50"] is not None:
            lines.append(
                f"- |v_tangential| on crossing GT frames "
                f"(n={st['n_vt_crossing']}): p50 {st['vt_crossing_p50']:.3f} m/s "
                f"(must sit clearly above the threshold)")
        if st.get("yaw_bearing_r") is not None:
            lines.append(
                f"- detrended bearing vs yaw on crossing GT frames "
                f"(n={st['n_yaw_diag']}): r = {st['yaw_bearing_r']:+.2f} "
                f"(~0 on a non-turning boat; a real turn correlates them)")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", required=True,
                    help="Clip prefix (radar CSV required; video unused).")
    ap.add_argument("--gt", default=None,
                    help="GT windows JSON (see module docstring).")
    ap.add_argument("--auto-radial-gt", action="store_true",
                    help="Derive closing/diverging/static GT from the raw "
                         "median-range slope (pure-radial clips: pacing).")
    ap.add_argument("--out", default="results/target_motion")
    ap.add_argument("--detection", default=None)
    ap.add_argument("--intrinsics", default=None)
    ap.add_argument("--no-imu", action="store_true",
                    help="Skip the IMU sidecar (single uncorrected variant).")
    ap.add_argument("--yaw-source", choices=("raw", "rotation"), default=None,
                    help="Override the replay yaw source for the imu_yaw "
                         "variant: 'raw' = the RVC sensor yaw (gimbal-aliased "
                         "on the box mount; reproduces the 2026-08-24 "
                         "baseline), 'rotation' = camera-frame heading by "
                         "rotation composition. Default: configs/imu.yaml.")
    ap.add_argument("--imu-yaw-sign", type=float, default=None,
                    help="Override motion.targets.imu_yaw_sign (+1/-1) for "
                         "the imu_yaw variant; the sign that LOWERS the "
                         "static-window tangential noise is the right one.")
    ap.add_argument("--tag", default=None,
                    help="Suffix for the output file stem, so variant runs "
                         "(yaw source / sign) do not overwrite each other.")
    ap.add_argument("--no-self-clutter", action="store_true",
                    help="Disable mmwave.self_clutter (lab/bench clips: the "
                         "zone is boat-mount-specific, CLAUDE.md §18).")
    ap.add_argument("--y-max", type=float, default=None, metavar="M",
                    help="Override mmwave.y_max for this run. The default "
                         "5.0 m detection window drops the radar's 5-9 m "
                         "early-warning band, which is where a crossing "
                         "target matters most (the 2026-08-19 crossing GT "
                         "subject transits at 6-8.7 m and is INVISIBLE "
                         "under the default): pass 9.0 to evaluate over "
                         "the full envelope.")
    args = ap.parse_args()
    if not args.gt and not args.auto_radial_gt:
        ap.error("need --gt or --auto-radial-gt")

    triplet = resolve_triplet(args.triplet, require=("mmwave",))
    detection = (load_detection(args.detection) if args.detection
                 else load_detection())
    intrinsics = (load_intrinsics(args.intrinsics) if args.intrinsics
                  else load_intrinsics())
    detection.setdefault("motion", {}).setdefault(
        "targets", {})["enabled"] = True
    if args.no_self_clutter:
        detection["mmwave"].setdefault("self_clutter", {})["enabled"] = False
    if args.y_max is not None:
        detection["mmwave"]["y_max"] = float(args.y_max)

    have_imu = triplet.imu is not None and not args.no_imu
    variants: dict[str, list[dict]] = {}
    for label in (["imu_yaw", "no_yaw"] if have_imu else ["no_yaw"]):
        det = copy.deepcopy(detection)
        attitude = None
        if label == "imu_yaw":
            replay_cfg = dict(load_imu_config().get("replay", {}) or {})
            if args.yaw_source is not None:
                replay_cfg["yaw_source"] = args.yaw_source
            attitude = ImuLogAttitudeProvider.for_triplet(triplet, replay_cfg)
            if attitude is None:
                print("[warn] IMU sidecar unreadable; skipping imu_yaw "
                      "variant", file=sys.stderr)
                continue
            # Force the correction on for this variant regardless of the
            # shipped default (off since 2026-08-24): the pair exists to
            # keep re-measuring it as the yaw signal improves.
            det["motion"]["targets"]["use_imu_yaw"] = True
            if args.imu_yaw_sign is not None:
                det["motion"]["targets"]["imu_yaw_sign"] = args.imu_yaw_sign
        else:
            det["motion"]["targets"]["use_imu_yaw"] = False
        variants[label] = run_variant(triplet, det, intrinsics, attitude,
                                      label)

    # Frame-time axis + GT resolution (shared across variants: GT is a
    # property of the clip, not the tracker).
    any_rows = next(iter(variants.values()))
    frame_ts = sorted({r["t_s"] for r in any_rows})
    if args.auto_radial_gt:
        med_by_t = {}
        for r in any_rows:
            med_by_t[r["t_s"]] = r["median_range_m"]
        auto = auto_radial_gt(frame_ts, [med_by_t[t] for t in frame_ts])
        gt_map = dict(zip(frame_ts, auto))

        def gt_of_frame(t_s, _rows):
            return gt_map.get(t_s), None
    else:
        windows = load_gt_windows(Path(args.gt), frame_ts[0])

        def gt_of_frame(t_s, _rows):
            for w in windows:
                if w["t0_s"] <= t_s <= w["t1_s"]:
                    return w["state"], w
            return None, None

    stats = {label: score_variant(rows, gt_of_frame)
             for label, rows in variants.items()}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = triplet.clip_id.replace("/", "__")
    if args.tag:
        stem = f"{stem}_{args.tag}"
    csv_path = out_dir / f"{stem}_frames.csv"
    fields = ["variant", "timestamp", "t_s", "n_points", "median_range_m",
              "ego_yaw_deg",
              "target_id", "bearing_deg", "range_m", "closing_mps",
              "v_tangential_mps", "speed_mps", "cpa_m", "t_cpa_s", "state",
              "ego_corrected", "age_frames", "n_cluster_points"]
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for rows in variants.values():
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})

    summary = render_summary(triplet.clip_id, stats)
    (out_dir / f"{stem}_summary.md").write_text(summary)
    print(summary)
    print(f"[out] {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
