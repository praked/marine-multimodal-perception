"""Hard-negative miner: Branch B.1.

Runs the current detection pipeline against scenes that *should* have
no obstacles (OpenWater by default), records every frame where any
bin fires above the hit threshold, and saves those frames as
labelling candidates. These are the most valuable training examples
for reducing the OpenWater false-positive rate.

Usage:
    python -m scripts.eval.hard_negatives
    python -m scripts.eval.hard_negatives --scene OpenWater --max-per-clip 30

Output:
    labels/hard_negatives/<scene>/<timestamp>/<frame_idx>.jpg
    labels/hard_negatives/candidates.jsonl  (frame_id + reason)

Audit and class-label these via label_tool.py to add them to
labels/master.jsonl as confirmed negatives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    iterate_triplet,
)
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import REPO_ROOT, list_triplets

HARD_NEG_DIR = REPO_ROOT / "labels" / "hard_negatives"
CANDIDATES_PATH = HARD_NEG_DIR / "candidates.jsonl"


def mine_clip(triplet, intrinsics, detection, max_per_clip: int, save_images: bool) -> int:
    pipeline = ObstacleDetectionPipeline(intrinsics, detection)
    threshold = float(detection["fusion"].get("hit_threshold", 0.33))
    n_found = 0
    out_dir = HARD_NEG_DIR / triplet.scene / triplet.timestamp
    with open(CANDIDATES_PATH, "a") as cand_fp:
        for fidx, (ts, fish, therm, mm_pts) in enumerate(iterate_triplet(triplet, detection)):
            if n_found >= max_per_clip:
                break
            res = pipeline.process_frame(fish, therm, mm_pts, timestamp=ts)
            max_score = max(res.fusion.scores)
            if max_score < threshold:
                continue
            frame_id = f"{triplet.clip_id}/{fidx:06d}"
            sensor_hits = res.fusion.sensor_hit_mask.sum(axis=0).tolist()
            cand_fp.write(json.dumps({
                "frame_id": frame_id,
                "scene": triplet.scene,
                "max_score": max_score,
                "sensor_hits_total": {
                    "fisheye": int(sensor_hits[0]),
                    "thermal": int(sensor_hits[1]),
                    "mmwave":  int(sensor_hits[2]),
                },
                "expected_label": "none",
            }) + "\n")
            if save_images:
                out_dir.mkdir(parents=True, exist_ok=True)
                # Save the undistorted fisheye (what the labeller will see).
                und = res.fisheye.undistorted if res.fisheye is not None else fish
                cv2.imwrite(str(out_dir / f"{fidx:06d}.jpg"), und,
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
            n_found += 1
    return n_found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="OpenWater",
                    help="Scene folder to mine (default OpenWater).")
    ap.add_argument("--max-per-clip", type=int, default=30,
                    help="Stop after this many candidates per clip.")
    ap.add_argument("--no-images", action="store_true",
                    help="Skip saving JPEG candidates (just write the JSONL).")
    args = ap.parse_args()

    HARD_NEG_DIR.mkdir(parents=True, exist_ok=True)
    # Truncate the candidates file at the start of a fresh mine.
    CANDIDATES_PATH.write_text("")

    intrinsics = load_intrinsics()
    detection = load_detection()
    triplets = list_triplets(args.scene)
    if not triplets:
        raise SystemExit(f"no triplets in data/{args.scene}/")
    total = 0
    for tri in tqdm(triplets, desc=args.scene):
        total += mine_clip(tri, intrinsics, detection,
                           max_per_clip=args.max_per_clip,
                           save_images=not args.no_images)
    print(f"\nfound {total} hard-negative candidates -> {CANDIDATES_PATH}")
    if not args.no_images:
        print(f"images saved under {HARD_NEG_DIR}")


if __name__ == "__main__":
    main()
