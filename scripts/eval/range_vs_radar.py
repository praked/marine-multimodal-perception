"""Validate monocular camera range against radar range, by bearing.

Branch A.1's exit criterion is "range_from_water_plane agrees with radar Y
within ±20% on labelled boat detections". This tool measures that agreement
on any triplet, no labels needed: every camera detection that carries a
monocular range is associated with the radar returns in a bearing window
around it, and the camera range is compared against the closest associated
radar return (min Y: the same "nearest obstacle at this bearing" reading
fusion's min_ranges uses).

Radar filtering here is deliberately independent of detection.yaml's
`mmwave:` block: for validation we want the radar's full usable envelope
(default 0.5–9 m; the low cut drops the known 0.29–0.46 m mount-clutter
cluster, CLAUDE.md §10.5), not the fusion-tuned window.

Caveats (by design, documented rather than hidden):
  - Association is bearing-only. A camera detection and a radar return at
    the same bearing can be different objects at different ranges; use the
    per-row output to spot-check, and prefer clips with a single dominant
    obstacle (Boats) for headline numbers.
  - The camera range method follows the active config: bbox-bottom by
    default, seg waterline-contact when segmentation is enabled and masks
    exist for the clip. The summary records which was active.

Usage:
    # every Boats triplet, both cameras
    python -m scripts.eval.range_vs_radar --scene Boats

    # a specific clip (e.g. tomorrow's new capture)
    python -m scripts.eval.range_vs_radar \\
        --triplet data/captures/<mission>/<ts> --out results/range_vs_radar
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    iterate_triplet,
)
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import list_triplets, resolve_triplet


def radar_bearings_ranges(pts: np.ndarray, y_min: float, y_max: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """(bearings_deg, ranges_m) for the validation envelope.

    Accepts Nx3 or Nx4 (Doppler column ignored here; this tool compares
    positions).
    """
    if len(pts) == 0:
        return np.empty(0), np.empty(0)
    x, y = pts[:, 0], pts[:, 1]
    keep = (y >= y_min) & (y <= y_max)
    x, y = x[keep], y[keep]
    return np.degrees(np.arctan2(x, y)), y


def associate(cam_angle: float, radar_bearings: np.ndarray,
              radar_ranges: np.ndarray, tol_deg: float
              ) -> tuple[float | None, int]:
    """(closest associated radar range, number of associated returns)."""
    if len(radar_bearings) == 0:
        return None, 0
    in_window = np.abs(radar_bearings - cam_angle) <= tol_deg
    n = int(in_window.sum())
    if n == 0:
        return None, 0
    return float(radar_ranges[in_window].min()), n


def run_clip(triplet, pipeline, detection_cfg, args, attitude=None) -> list[dict]:
    rows: list[dict] = []
    n_frames = 0
    for ts, fish, therm, pts in iterate_triplet(triplet, detection_cfg):
        n_frames += 1
        if args.max_frames and n_frames > args.max_frames:
            break
        if attitude is not None:
            attitude.set_time(ts)
        bearings, ranges = radar_bearings_ranges(
            np.asarray(pts, dtype=np.float64), args.y_min, args.y_max)
        if len(bearings) == 0:
            continue   # no radar this frame -> nothing to validate against

        per_sensor = {}
        if "fisheye" in args.sensors:
            per_sensor["fisheye"] = pipeline.process_fisheye(
                fish, frame_id=f"{triplet.clip_id}/ts={ts.replace(':', '-')}")
        if "thermal" in args.sensors:
            per_sensor["thermal"] = pipeline.process_thermal(therm)

        for sensor, res in per_sensor.items():
            for angle, cam_range in zip(res.angles, res.ranges):
                if cam_range is None:
                    continue
                radar_range, n_assoc = associate(
                    angle, bearings, ranges, args.tol_deg)
                if radar_range is None:
                    continue
                rel_err = (cam_range - radar_range) / radar_range
                rows.append({
                    "clip": triplet.clip_id,
                    "ts": ts,
                    "sensor": sensor,
                    "bearing_deg": round(float(angle), 1),
                    "cam_range_m": round(float(cam_range), 2),
                    "radar_range_m": round(radar_range, 2),
                    "rel_err": round(float(rel_err), 3),
                    "within_20pct": abs(rel_err) <= 0.20,
                    "n_radar_assoc": n_assoc,
                })
    return rows


def summarise(rows: list[dict], key) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for name, rs in sorted(groups.items()):
        errs = np.array([r["rel_err"] for r in rs])
        out[name] = {
            "n": len(rs),
            "median_abs_rel_err": float(np.median(np.abs(errs))),
            "mean_signed_rel_err": float(np.mean(errs)),
            "within_20pct": float(np.mean([r["within_20pct"] for r in rs])),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triplet", action="append", default=[],
                    help="triplet prefix (repeatable)")
    ap.add_argument("--scene", default=None,
                    help="run every triplet in this scene (e.g. Boats)")
    ap.add_argument("--sensors", default="fisheye,thermal",
                    help="comma list: fisheye,thermal")
    ap.add_argument("--tol-deg", type=float, default=5.0,
                    help="bearing association window (± deg; default 5 = "
                         "half a fusion bin)")
    ap.add_argument("--y-min", type=float, default=0.5,
                    help="radar validation range floor (m); default 0.5 "
                         "drops the mount-clutter cluster")
    ap.add_argument("--y-max", type=float, default=9.0,
                    help="radar validation range cap (m); default 9 = the "
                         "profile's max unambiguous range")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="per-clip frame cap (0 = whole clip)")
    ap.add_argument("--out", type=Path, default=Path("results/range_vs_radar"))
    ap.add_argument("--seg", action="store_true",
                    help="Force segmentation on (waterline-contact range "
                         "where masks exist under data/seg/).")
    ap.add_argument("--imu", action="store_true",
                    help="Replay a clip's imu_<ts>.csv sidecar as the "
                         "attitude source for the camera range (falls back "
                         "to horizon/level where the log has gaps).")
    args = ap.parse_args(argv)
    args.sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]

    triplets = [resolve_triplet(t) for t in args.triplet]
    if args.scene:
        triplets.extend(list_triplets(args.scene))
    if not triplets:
        ap.error("give --triplet and/or --scene")

    intrinsics = load_intrinsics()
    detection = load_detection()
    if args.seg:
        detection.setdefault("segmentation", {})["enabled"] = True
    seg_on = bool((detection.get("segmentation", {}) or {}).get("enabled"))

    all_rows: list[dict] = []
    for t in triplets:
        attitude = None
        if args.imu:
            from scripts.sensor_processing.imu_bno085 import load_imu_config
            from scripts.sensor_processing.imu_replay import ImuLogAttitudeProvider
            attitude = ImuLogAttitudeProvider.for_triplet(
                t, (load_imu_config().get("replay", {}) or {}))
            if attitude is None:
                print(f"{t.clip_id}: no usable IMU log: horizon/level attitude")
        pipeline = ObstacleDetectionPipeline(intrinsics, detection,
                                             attitude_provider=attitude)
        rows = run_clip(t, pipeline, detection, args, attitude=attitude)
        print(f"{t.clip_id}: {len(rows)} associated detections")
        all_rows.extend(rows)

    if not all_rows:
        print("No camera-radar associations found (radar empty, or no "
              "camera detections carried a range).")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "range_vs_radar.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f"\ncamera range method: "
          f"{'seg waterline-contact (where masks exist)' if seg_on else 'bbox-bottom water-plane'}")
    print(f"association: ±{args.tol_deg}° bearing window, radar "
          f"{args.y_min}–{args.y_max} m\n")
    header = f"{'group':<42} {'n':>6} {'med|err|':>9} {'mean err':>9} {'<=20%':>7}"
    for title, key in (("per clip", lambda r: r["clip"]),
                       ("per sensor", lambda r: r["sensor"]),
                       ("overall", lambda r: "all")):
        print(f"-- {title}")
        print(header)
        for name, s in summarise(all_rows, key).items():
            print(f"{name:<42} {s['n']:>6} {s['median_abs_rel_err']:>8.1%} "
                  f"{s['mean_signed_rel_err']:>+8.1%} {s['within_20pct']:>6.0%}")
        print()
    print(f"-> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
