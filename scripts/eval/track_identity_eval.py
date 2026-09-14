"""Score the cross-sensor identity layer on the motion GT clips.

Three variants over one clip (full pipeline, fisheye + radar; thermal is
deliberately not decoded -- identity is fisheye-only):

  - `baseline`  : motion.targets only. Must reproduce the
                  target_motion_eval numbers (crossing recall 0.806 bar on
                  the 2026-08-19 canoe clip at the current defaults).
  - `identity`  : + motion.targets.identity. The motion metrics must be
                  IDENTICAL to baseline (the layer only annotates); what it
                  adds is the pairing honesty report: pairs formed/broken,
                  id switches, frames paired, GT-window pairing coverage.
  - `cam_rate`  : + motion.targets.camera_bearing_rate. The dropout-fill
                  measurement: GT frames where baseline reported no target
                  but the paired camera track sustains one (n_points == 0
                  rows), and the crossing recall including those frames.

Acceptance bar (docs/history/2026-08-24_target_motion.md section 5.1):
crossing recall must not regress from 0.806; the radar's 4 s mid-transit
dropout (rel 27-31 s) is the window camera sustain exists to fill.

Usage:
    python -m scripts.eval.track_identity_eval \\
        --triplet data/captures/2026-08-19_afloat/canoe_fx/2026-08-19_15-58-20 \\
        --gt labels/motion_gt/canoe_2026-08-19_15-58-20.json \\
        --out results/track_identity

    python -m scripts.eval.track_identity_eval \\
        --triplet data/captures/2026-08-19_afloat/imu_rock/2026-08-19_15-39-40 \\
        --gt labels/motion_gt/imu_rock_2026-08-19_15-39-40.json \\
        --out results/track_identity
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

from scripts.eval.target_motion_eval import (
    load_gt_windows,
    per_state_pr,
    pick_target,
    score_variant,
)
from scripts.sensor_processing.fusion import _frame_id_for
from scripts.sensor_processing.imu_bno085 import load_imu_config
from scripts.sensor_processing.imu_replay import ImuLogAttitudeProvider
from scripts.sensor_processing.motion import parse_radar_timestamp
from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    iterate_triplet,
)
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import resolve_triplet

VARIANTS = ("baseline", "identity", "cam_rate")


def variant_detection(detection: dict, label: str) -> dict:
    det = copy.deepcopy(detection)
    tgt = det.setdefault("motion", {}).setdefault("targets", {})
    tgt["enabled"] = True
    tgt.setdefault("identity", {})["enabled"] = label in ("identity",
                                                          "cam_rate")
    tgt["camera_bearing_rate"] = label == "cam_rate"
    return det


def run_variant(triplet, detection, intrinsics, attitude, label: str
                ) -> tuple[list[dict], dict]:
    """One full-pipeline pass; per-frame rows (target_motion_eval schema +
    identity columns) and the identity manager's honesty counters."""
    pipeline = ObstacleDetectionPipeline(intrinsics, detection,
                                         attitude_provider=attitude)
    rows: list[dict] = []
    for ts, fish, _therm, pts in iterate_triplet(triplet, detection):
        if attitude is not None:
            attitude.set_time(ts)
        fid = _frame_id_for(triplet.scene, triplet.timestamp, ts)
        res = pipeline.process_frame(fish, None, pts, timestamp=ts,
                                     frame_id=fid)
        t_s = parse_radar_timestamp(ts)
        base = {"variant": label, "timestamp": ts, "t_s": t_s,
                "n_points": (len(res.mmwave.points_xyz)
                             if res.mmwave is not None else 0),
                "n_detections": (len(res.fisheye.coords)
                                 if res.fisheye is not None else 0)}
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
                         "age_frames": t.age_frames,
                         "n_cluster_points": t.n_points,
                         "camera_track_id": t.camera_track_id,
                         "rate_source": t.rate_source,
                         "sustained": t.n_points == 0})
    stats = (pipeline._identity.stats()
             if pipeline._identity is not None else {})
    return rows, stats


def identity_report(rows: list[dict], windows: list[dict]) -> dict:
    """GT-focused pairing honesty: coverage + switches on the GT subject."""
    by_frame: dict[float, list[dict]] = {}
    for r in rows:
        by_frame.setdefault(r["t_s"], []).append(r)
    n_gt = paired = sustained = missed = 0
    cam_ids: list[int] = []
    rad_ids: list[int] = []
    for t_s in sorted(by_frame):
        window = next((w for w in windows
                       if w["t0_s"] <= t_s <= w["t1_s"]), None)
        if window is None:
            continue
        n_gt += 1
        tgt = pick_target(by_frame[t_s], window)
        if tgt is None:
            missed += 1
            continue
        rad_ids.append(tgt["target_id"])
        if tgt.get("sustained"):
            sustained += 1
        cid = tgt.get("camera_track_id")
        if cid is not None:
            paired += 1
            cam_ids.append(cid)

    def _switches(ids: list[int]) -> int:
        return sum(1 for a, b in zip(ids, ids[1:]) if a != b)

    return {
        "gt_frames": n_gt,
        "gt_missed": missed,
        "gt_paired": paired,
        "gt_sustained": sustained,
        "gt_camera_ids": sorted(set(cam_ids)),
        "gt_camera_id_switches": _switches(cam_ids),
        "gt_radar_id_switches": _switches(rad_ids),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", required=True,
                    help="Clip prefix (fisheye video + radar CSV required).")
    ap.add_argument("--gt", required=True,
                    help="GT windows JSON (target_motion_eval format).")
    ap.add_argument("--out", default="results/track_identity")
    ap.add_argument("--detection", default=None)
    ap.add_argument("--intrinsics", default=None)
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--no-self-clutter", action="store_true",
                    help="Disable mmwave.self_clutter (bench clips only).")
    ap.add_argument("--y-max", type=float, default=None, metavar="M",
                    help="Override mmwave.y_max (default detection.yaml "
                         "9.0 since D.2; the flag remains for A/Bs).")
    ap.add_argument("--tag", default=None,
                    help="Suffix for the output file stem.")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=VARIANTS)
    ap.add_argument("--pair-min-hits", type=int, default=None,
                    help="Override identity.pair_min_hits (the hysteresis "
                         "sweep: run 2/3, 3/5, 5/8 with --tag).")
    ap.add_argument("--pair-window", type=int, default=None,
                    help="Override identity.pair_window.")
    args = ap.parse_args()

    triplet = resolve_triplet(args.triplet, require=("fisheye", "mmwave"))
    detection = (load_detection(args.detection) if args.detection
                 else load_detection())
    intrinsics = (load_intrinsics(args.intrinsics) if args.intrinsics
                  else load_intrinsics())
    if args.no_self_clutter:
        detection["mmwave"].setdefault("self_clutter", {})["enabled"] = False
    if args.y_max is not None:
        detection["mmwave"]["y_max"] = float(args.y_max)
    id_over = detection.setdefault("motion", {}).setdefault(
        "targets", {}).setdefault("identity", {})
    if args.pair_min_hits is not None:
        id_over["pair_min_hits"] = args.pair_min_hits
    if args.pair_window is not None:
        id_over["pair_window"] = args.pair_window

    variants: dict[str, tuple[list[dict], dict]] = {}
    for label in args.variants:
        det = variant_detection(detection, label)
        attitude = None
        if not args.no_imu:
            attitude = ImuLogAttitudeProvider.for_triplet(
                triplet, (load_imu_config().get("replay", {}) or {}))
        print(f"[run] variant {label} ...", file=sys.stderr)
        variants[label] = run_variant(triplet, det, intrinsics, attitude,
                                      label)

    any_rows = next(iter(variants.values()))[0]
    frame_ts = sorted({r["t_s"] for r in any_rows})
    windows = load_gt_windows(Path(args.gt), frame_ts[0])

    def gt_of_frame(t_s, _rows):
        for w in windows:
            if w["t0_s"] <= t_s <= w["t1_s"]:
                return w["state"], w
        return None, None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = triplet.clip_id.replace("/", "__")
    if args.tag:
        stem = f"{stem}_{args.tag}"
    fields = ["variant", "timestamp", "t_s", "n_points", "n_detections",
              "target_id", "bearing_deg", "range_m", "closing_mps",
              "v_tangential_mps", "speed_mps", "cpa_m", "t_cpa_s", "state",
              "age_frames", "n_cluster_points", "camera_track_id",
              "rate_source", "sustained"]
    csv_path = out_dir / f"{stem}_frames.csv"
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for rows, _stats in variants.values():
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})

    lines = [f"# track_identity_eval: {triplet.clip_id}", ""]
    for label, (rows, mgr_stats) in variants.items():
        st = score_variant(rows, gt_of_frame)
        lines.append(f"## variant: {label}")
        lines.append(f"- GT frames: {st['n_gt_frames']}  "
                     f"(no target reported on {st['n_missed']})")
        for s, (p, r) in per_state_pr(st["confusion"]).items():
            if p is None and r is None:
                continue
            lines.append(
                f"- {s}: precision "
                f"{'n/a' if p is None else f'{p:.3f}'} / recall "
                f"{'n/a' if r is None else f'{r:.3f}'}")
        if mgr_stats:
            lines.append(f"- identity: {mgr_stats}")
            rep = identity_report(rows, windows)
            lines.append(
                f"- GT-window pairing: {rep['gt_paired']}/{rep['gt_frames']}"
                f" frames paired (missed {rep['gt_missed']}, sustained "
                f"{rep['gt_sustained']}); camera ids {rep['gt_camera_ids']}"
                f" (switches {rep['gt_camera_id_switches']}), radar id "
                f"switches {rep['gt_radar_id_switches']}")
        lines.append("")
    if {"baseline", "cam_rate"} <= set(variants):
        b = score_variant(variants["baseline"][0], gt_of_frame)
        c = score_variant(variants["cam_rate"][0], gt_of_frame)
        lines.append("## dropout fill (baseline -> cam_rate)")
        lines.append(f"- GT frames without a reported target: "
                     f"{b['n_missed']} -> {c['n_missed']}")
        lines.append("")
    summary = "\n".join(lines)
    (out_dir / f"{stem}_summary.md").write_text(summary)
    print(summary)
    print(f"[out] {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
