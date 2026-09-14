"""Pixel-level association report: arbitrates bearings + measures A.1 range.

Runs the pipeline with `fusion.association` forced ON over one or more clips
and reports, per camera and per bearing model (linear vs pinhole):

  - match rate: fraction of detections whose padded bbox captured >= 1
    projected radar return (and how many bins per frame get `confirmed`);
  - bearing residual: camera bearing minus the median radar bearing of the
    MATCHED points: the empirical arbiter between the legacy magic-number
    bearings (cx=472 / 7.2) and the calibrated pinhole ones (cx_K=444.1 / fx).
    Association gating is pixel-based, so the residual is meaningful for both
    models;
  - range agreement: monocular water-plane vs radar range for matched
    detections: the honest A.1 range check (object-gated, unlike
    range_vs_radar's bearing-window association).

    python -m scripts.eval.association_report \
        --triplet data/captures/2026-07-08/2026-07-08_16-37-01 --imu --seg
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    iterate_triplet,
)
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import resolve_triplet


def _stats(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    a = np.asarray(vals)
    return (f"n={len(a)}  median {np.median(a):+.2f}  mean {a.mean():+.2f}  "
            f"p5 {np.percentile(a, 5):+.2f}  p95 {np.percentile(a, 95):+.2f}")


def run(triplet, detection, intrinsics, attitude, max_frames: int = 0) -> dict:
    pipeline = ObstacleDetectionPipeline(intrinsics, detection,
                                         attitude_provider=attitude)
    out = {s: {"n_det": 0, "n_matched": 0, "bearing_resid": [],
               "mono": [], "radar": []} for s in ("fisheye", "thermal")}
    n_frames = 0
    confirmed_bins = 0
    for ts, fish, therm, pts in iterate_triplet(triplet, detection):
        n_frames += 1
        if max_frames and n_frames > max_frames:
            break
        if attitude is not None:
            attitude.set_time(ts)
        fid = f"{triplet.clip_id}/ts={ts.replace(':', '-')}"
        res = pipeline.process_frame(fish, therm, pts, timestamp=ts,
                                     frame_id=fid)
        confirmed_bins += sum(res.fusion.confirmed)
        for name, r in (("fisheye", res.fisheye), ("thermal", res.thermal)):
            if r is None:
                continue
            o = out[name]
            o["n_det"] += len(r.coords)
            if not r.radar_ranges:
                continue
            # radar bearings of matched points, re-derived per detection via
            # the pipeline's DetectionMatch (stored range + implicit bearing
            # through angles): we recompute residuals from what the result
            # carries: camera angle vs matched radar range/bearing pair.
            from scripts.utils.association import (
                AssociationParams,
                associate_radar_to_detections,
            )
            from scripts.utils.cv_common import pinhole_new_K
            from scripts.utils.geometry import project_radar_to_undistorted
            if name == "fisheye":
                P = np.asarray(intrinsics["fisheye"]["K"], float)
                T = pipeline.extrinsics["T_radar_to_fisheye"]
            else:
                h_px, w_px = r.undistorted.shape[:2]
                P = pinhole_new_K(intrinsics["thermal"]["K"],
                                  intrinsics["thermal"]["D"], (w_px, h_px))
                T = pipeline.extrinsics["T_radar_to_thermal"]
            px = project_radar_to_undistorted(res.mmwave.points_xyz, P, T)
            assoc = associate_radar_to_detections(
                res.mmwave.points_xyz, px, r.coords, r.sizes,
                AssociationParams.from_config(
                    (detection["fusion"].get("association", {}) or {})))
            for angle, mono, m in zip(r.angles,
                                      (r.mono_ranges or r.ranges), assoc.matches):
                if m.radar_range_m is None:
                    continue
                o["n_matched"] += 1
                if m.radar_bearing_deg is not None:
                    o["bearing_resid"].append(float(angle) - m.radar_bearing_deg)
                if mono is not None:
                    o["mono"].append(float(mono))
                    o["radar"].append(m.radar_range_m)
    out["n_frames"] = n_frames
    out["confirmed_bins_per_frame"] = confirmed_bins / max(n_frames, 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", required=True)
    ap.add_argument("--imu", action="store_true")
    ap.add_argument("--seg", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args(argv)

    triplet = resolve_triplet(args.triplet)
    intrinsics = load_intrinsics()

    attitude = None
    if args.imu:
        from scripts.sensor_processing.imu_bno085 import load_imu_config
        from scripts.sensor_processing.imu_replay import ImuLogAttitudeProvider
        attitude = ImuLogAttitudeProvider.for_triplet(
            triplet, (load_imu_config().get("replay", {}) or {}))

    for model in ("linear", "pinhole"):
        detection = load_detection()
        detection["fusion"].setdefault("association", {})["enabled"] = True
        detection["fisheye"]["bearing_model"] = model
        detection["thermal"]["bearing_model"] = model
        if args.seg:
            detection.setdefault("segmentation", {})["enabled"] = True
        r = run(triplet, detection, intrinsics, attitude, args.max_frames)
        print(f"\n=== bearing_model = {model} "
              f"({r['n_frames']} frames, "
              f"{r['confirmed_bins_per_frame']:.2f} confirmed bins/frame)")
        for cam in ("fisheye", "thermal"):
            o = r[cam]
            rate = o["n_matched"] / o["n_det"] if o["n_det"] else 0.0
            print(f"  {cam}: {o['n_det']} detections, "
                  f"{o['n_matched']} radar-matched ({rate:.1%})")
            print(f"    bearing residual (cam - radar, deg): "
                  f"{_stats(o['bearing_resid'])}")
            if o["mono"]:
                rel = (np.asarray(o["mono"]) - np.asarray(o["radar"])) \
                    / np.asarray(o["radar"])
                within = float(np.mean(np.abs(rel) <= 0.20))
                print(f"    mono vs radar range: {_stats(list(rel))} "
                      f"(rel err; <=20%: {within:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
