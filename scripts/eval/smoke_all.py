"""Smoke-run the canonical pipeline against every triplet in data/.

Useful pre-trip validation: confirms that no clip crashes the pipeline,
and prints a one-line summary per clip (n frames, mean fused score,
mean max-bin score, any-hit rate). Run as:

    python -m scripts.eval.smoke_all
    python -m scripts.eval.smoke_all --strict   # exit nonzero on any failure
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    iterate_triplet,
)
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import list_triplets


def run_one(triplet, intrinsics, detection) -> dict:
    pipeline = ObstacleDetectionPipeline(intrinsics, detection)
    n_frames = 0
    score_sum = 0.0
    max_sum = 0.0
    hits = 0
    threshold = float(detection["fusion"].get("hit_threshold", 0.33))
    t0 = time.time()
    for ts, fish, therm, mm_pts in iterate_triplet(triplet, detection):
        res = pipeline.process_frame(fish, therm, mm_pts, timestamp=ts)
        max_score = max(res.fusion.scores)
        n_frames += 1
        score_sum += sum(res.fusion.scores) / len(res.fusion.scores)
        max_sum += max_score
        hits += int(max_score >= threshold)
    return {
        "n_frames": n_frames,
        "mean_score": score_sum / n_frames if n_frames else 0.0,
        "mean_max": max_sum / n_frames if n_frames else 0.0,
        "hit_rate": hits / n_frames if n_frames else 0.0,
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="Exit with nonzero status on any clip failure.")
    ap.add_argument("--scene", default=None,
                    help="Restrict to one scene (Boats/Ducks/OpenWater/Rain).")
    args = ap.parse_args()

    triplets = list_triplets(args.scene)
    if not triplets:
        print("no triplets found in data/")
        sys.exit(1)

    intrinsics = load_intrinsics()
    detection = load_detection()

    fmt = "{:32s}  {:5s}  {:>5s}  {:>7s}  {:>7s}  {:>7s}  {:>6s}"
    print(fmt.format("clip", "scene", "N", "meanS", "meanMx", "hitR", "sec"))
    failed = 0
    for tri in triplets:
        try:
            r = run_one(tri, intrinsics, detection)
            print(f"{tri.clip_id:32s}  {tri.scene:5s}  {r['n_frames']:5d}  "
                  f"{r['mean_score']:7.3f}  {r['mean_max']:7.3f}  "
                  f"{r['hit_rate']:7.3f}  {r['seconds']:6.1f}")
        except Exception as e:
            failed += 1
            print(f"{tri.clip_id:32s}  FAIL: {e}")
            if args.strict:
                traceback.print_exc()
                sys.exit(2)

    if failed:
        print(f"\n{failed} clips failed")
        sys.exit(2)
    print(f"\nall {len(triplets)} clips OK")


if __name__ == "__main__":
    main()
