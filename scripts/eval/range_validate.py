"""Validate monocular water-plane range against KNOWN, measured distances.

Isolates how good the range really is, and how much a recalibration helps:
place a target (the checkerboard works) at the waterline at measured distances,
note the pixel of its waterline contact, and this reports computed vs true
range. Run it once with the current intrinsics and again with a recalibrated
set (`--intrinsics configs/intrinsics_fisheye_recalib.yaml`) to see the change.

Uses the exact pipeline geometry (undistort -> back-project bbox bottom ->
intersect the water plane at the measured camera height). Level reference by
default; `--use-horizon` uses the detected-horizon attitude like the dashboard.

Samples YAML:
  samples:
    - image: data/captures/2026-06-17_institutionone_day1/<frame>.jpg
      pixel: [432, 600]      # waterline-contact pixel of the object
      pixel_space: raw       # raw (default) | undistorted
      true_range_m: 5.0

Usage:
  python -m scripts.eval.range_validate --samples calib/range_samples.yaml
  python -m scripts.eval.range_validate --samples calib/range_samples.yaml \
      --intrinsics configs/intrinsics_fisheye_recalib.yaml --use-horizon
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
from scripts.utils.geometry import UP_LEVEL, range_from_water_plane_up


def _undistort_point(u, v, K, D):
    pt = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv2.fisheye.undistortPoints(pt, K, np.asarray(D, dtype=np.float64).reshape(-1, 1), P=K)
    return float(und[0, 0, 0]), float(und[0, 0, 1])


def evaluate(samples, intrinsics_path, use_horizon):
    intr = (load_intrinsics(intrinsics_path) if intrinsics_path else load_intrinsics())
    det = load_detection()
    pl = ObstacleDetectionPipeline(intr, det)
    K = intr["fisheye"]["K"]
    D = intr["fisheye"]["D"]
    height = pl._camera_height["fisheye"]

    rows = []
    for s in samples:
        img = cv2.imread(s["image"])
        if img is None:
            print(f"  ! cannot read {s['image']}")
            continue
        res = pl.process_fisheye(img)                      # raw frame in
        up = (pl._range_up_vector(K, res.horizon_line) if use_horizon else UP_LEVEL)
        u, v = s["pixel"]
        if s.get("pixel_space", "raw") == "raw":
            u, v = _undistort_point(u, v, K, D)
        rng = range_from_water_plane_up((u, v, u, v), K, up, height, max_range_m=None)
        rows.append((s.get("image", "?"), float(s["true_range_m"]), rng,
                     res.horizon_line[2]))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="YAML file (see module docstring)")
    ap.add_argument("--intrinsics", default=None,
                    help="override intrinsics YAML (e.g. a recalibration result)")
    ap.add_argument("--use-horizon", action="store_true",
                    help="use detected-horizon attitude (default: level)")
    args = ap.parse_args(argv)

    doc = yaml.safe_load(Path(args.samples).read_text()) or {}
    samples = doc.get("samples", [])
    if not samples:
        raise SystemExit("no 'samples' in the YAML")

    rows = evaluate(samples, args.intrinsics, args.use_horizon)
    print(f"\nintrinsics: {args.intrinsics or 'configs/intrinsics.yaml'} | "
          f"attitude: {'detected-horizon' if args.use_horizon else 'level'}\n")
    print(f"{'true(m)':>8} {'computed(m)':>12} {'error':>9}  {'horizon_conf':>12}  image")
    errs = []
    for img, true_r, rng, conf in rows:
        if rng is None:
            print(f"{true_r:8.1f} {'None':>12} {'(above horizon)':>9}  {conf:12.2f}  {Path(img).name}")
            continue
        e = (rng - true_r) / true_r
        errs.append(abs(e))
        print(f"{true_r:8.1f} {rng:12.1f} {e:+8.0%}  {conf:12.2f}  {Path(img).name}")
    if errs:
        print(f"\nmean |error| = {np.mean(errs):.0%}  (n={len(errs)} ranged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
