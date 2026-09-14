"""Parameter-sweep harness: Phase I.4.4 of PLAN.md.

Reads a grid YAML, runs the canonical pipeline against every eval clip
for every parameter combination, writes per-(combo, clip, frame) records
plus a per-(combo, clip) summary.

Usage:
    python -m scripts.eval.sweep --grid configs/sweep_default.yaml

Output:
    results/sweeps/<run_id>/
        config.yaml          # grid + eval_clips snapshot
        per_frame.csv        # one row per (combo_id, clip, frame_idx)
        summary.csv          # one row per (combo_id, clip)
        ranked.csv           # combos ranked by proxy score across clips
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    iterate_triplet,
)
from scripts.utils.calibration import load_detection, load_intrinsics, set_dotted
from scripts.utils.datasets import resolve_triplet

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO_ROOT / "results" / "sweeps"


def _enumerate_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    out: list[dict] = []
    for combo in itertools.product(*values):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def _combo_id(combo: dict) -> str:
    parts = [f"{k.split('.')[-1]}={v}" for k, v in combo.items()]
    return "_".join(parts)


def run(grid_path: str, run_id: str | None = None) -> Path:
    with open(grid_path, "r") as f:
        grid_cfg = yaml.safe_load(f)
    eval_clips: dict[str, str] = grid_cfg["eval_clips"]
    scene_weights: dict[str, float] = grid_cfg.get("scene_weights", {})
    grid: dict[str, list] = grid_cfg["grid"]

    combos = _enumerate_grid(grid)
    print(f"sweeping {len(combos)} combos x {len(eval_clips)} clips")

    intrinsics = load_intrinsics()
    base_detection = load_detection()

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SWEEP_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(grid_path, out_dir / "config.yaml")

    per_frame_fp = open(out_dir / "per_frame.csv", "w", newline="")
    pf_writer = csv.writer(per_frame_fp)
    pf_writer.writerow(["combo_id", "scene", "clip_id", "frame_idx", "timestamp",
                        "max_score", "any_hit", "fisheye_hits", "thermal_hits",
                        "mmwave_hits", "any_min_range"])

    summary_fp = open(out_dir / "summary.csv", "w", newline="")
    sm_writer = csv.writer(summary_fp)
    sm_writer.writerow(["combo_id", "scene", "clip_id", "n_frames",
                        "hit_rate", "mean_max_score", "fp_proxy",
                        "mean_min_range"])

    proxy_rows: list[tuple[str, dict, float]] = []

    for combo in tqdm(combos, desc="combos"):
        det = deepcopy(base_detection)
        for k, v in combo.items():
            set_dotted(det, k, v)
        threshold = det["fusion"].get("hit_threshold", 0.33)

        per_clip_proxy: dict[str, float] = {}

        for scene, prefix in eval_clips.items():
            triplet = resolve_triplet(prefix)
            pipeline = ObstacleDetectionPipeline(intrinsics, det)
            n = 0
            hits = 0
            score_sum = 0.0
            range_sum = 0.0
            range_n = 0

            for fidx, (ts, fish, therm, mm_pts) in enumerate(iterate_triplet(triplet, det)):
                res = pipeline.process_frame(fish, therm, mm_pts, timestamp=ts)
                n += 1
                max_score = max(res.fusion.scores)
                score_sum += max_score
                any_hit = max_score >= threshold
                hits += int(any_hit)

                # Track min radar range across all bins for this frame.
                mins = [r for r in res.fusion.min_ranges if r is not None]
                frame_min_range = min(mins) if mins else None
                if frame_min_range is not None:
                    range_sum += frame_min_range
                    range_n += 1

                pf_writer.writerow([
                    _combo_id(combo), scene, triplet.clip_id, fidx, ts,
                    f"{max_score:.3f}", int(any_hit),
                    int(res.fusion.sensor_hit_mask[:, 0].any()),
                    int(res.fusion.sensor_hit_mask[:, 1].any()),
                    int(res.fusion.sensor_hit_mask[:, 2].any()),
                    f"{frame_min_range:.2f}" if frame_min_range is not None else "",
                ])

            if n == 0:
                continue
            hit_rate = hits / n
            mean_max_score = score_sum / n
            mean_min_range = range_sum / range_n if range_n else float("nan")
            # fp_proxy = hit_rate on OpenWater, else 0 (only OpenWater hit_rate
            # measures false positives without labels).
            fp_proxy = hit_rate if scene == "OpenWater" else 0.0

            sm_writer.writerow([
                _combo_id(combo), scene, triplet.clip_id, n,
                f"{hit_rate:.4f}", f"{mean_max_score:.4f}",
                f"{fp_proxy:.4f}", f"{mean_min_range:.2f}",
            ])
            per_clip_proxy[scene] = hit_rate

        # Compute scene-weighted proxy score for this combo.
        # PLAN.md §I.4.4 weighting: 0.4 Boats + 0.3 Ducks + 0.2 (1 - FPR OW) + 0.1 Rain.
        # In pre-label mode we use hit_rate as a coarse stand-in for recall.
        def w(s):
            return float(scene_weights.get(s, 0.0))
        score = (
            w("Boats") * per_clip_proxy.get("Boats", 0.0)
            + w("Ducks") * per_clip_proxy.get("Ducks", 0.0)
            + w("OpenWater") * (1.0 - per_clip_proxy.get("OpenWater", 0.0))
            + w("Rain") * per_clip_proxy.get("Rain", 0.0)
        )
        proxy_rows.append((_combo_id(combo), combo, score))

    per_frame_fp.close()
    summary_fp.close()

    # Ranked output.
    proxy_rows.sort(key=lambda r: r[2], reverse=True)
    with open(out_dir / "ranked.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["combo_id", "proxy_score", "combo_json"])
        for combo_id, combo, score in proxy_rows:
            w.writerow([combo_id, f"{score:.4f}", json.dumps(combo)])

    print(f"sweep complete -> {out_dir}")
    if proxy_rows:
        top = proxy_rows[0]
        print(f"top proxy score: {top[2]:.4f}  combo: {top[1]}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default=str(REPO_ROOT / "configs" / "sweep_default.yaml"))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    run(args.grid, args.run_id)


if __name__ == "__main__":
    main()
