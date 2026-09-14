"""Thermal-only parameter sweep scored against fisheye pseudo-GT.

The 2026-07-09 tuning harness (docs/history/2026-07-09_thermal_tuning.md). Unlike
scripts/eval/sweep.py (whole-pipeline, label-free proxy), this sweeps ONLY
`thermal.*` keys and scores each combo with scripts/eval/thermal_gt_eval
(fisheye GroundingDINO GT in bearing space + radar corroboration).

Fast because the invariants are computed once and cached:
  * raw thermal frames (after clip overrides) are decoded once and held in
    memory (~33 MB for 590 frames at 160x120);
  * radar bearings per frame are computed once with the current radar
    config (the radar is not swept);
  * GT intervals are loaded once.
Each combo then re-runs just the thermal path (undistort -> horizon ->
background -> blob), ~1 s per combo, parallelised across processes.

Usage:
    python -m scripts.eval.thermal_sweep --grid configs/thermal_sweep_coarse.yaml
    python -m scripts.eval.thermal_sweep --grid ... --workers 8 --run-id coarse
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from scripts.utils.datasets import REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "thermal_tuning"

# Worker globals, set once per process by _init_worker (macOS spawns fresh
# interpreters, so everything heavy is loaded from the cache file).
_G: dict = {}


def _enumerate_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        out.append(dict(zip(keys, combo)))
    return out


def _combo_id(combo: dict) -> str:
    return "_".join(f"{k.split('.')[-1]}={v}" for k, v in combo.items())


def _set_dotted_create(cfg: dict, dotted_key: str, value) -> None:
    """set_dotted, creating intermediate dicts (thermal.horizon is absent
    from detection.yaml by default; sweeping it must not KeyError)."""
    parts = dotted_key.split(".")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if nxt is None:
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def build_cache(triplet_prefix: str, detection: dict, intrinsics: dict,
                cache_path: Path, fp_clips: dict[str, str] | None = None) -> dict:
    """Decode the clip once: raw thermal frames + radar bearings per ts.

    `fp_clips` maps a short name -> triplet prefix of a false-positive
    reference clip (e.g. 2025 open water, no obstacles): only its thermal
    frames are cached; the sweep reports det/frame per combo on each.
    """
    from scripts.sensor_processing.pipeline import (
        ObstacleDetectionPipeline,
        iterate_triplet,
    )
    from scripts.utils.datasets import resolve_triplet

    triplet = resolve_triplet(triplet_prefix)
    pipeline = ObstacleDetectionPipeline(intrinsics, detection)
    frames: list[tuple[str, np.ndarray]] = []
    radar_by_ts: dict[str, list[float]] = {}
    for ts, _fish, therm, mm_pts in iterate_triplet(triplet, detection):
        m_res = pipeline.process_mmwave(mm_pts, timestamp=ts)
        frames.append((ts, therm))
        radar_by_ts[ts] = list(m_res.angles)
    fp_frames: dict[str, list[np.ndarray]] = {}
    for name, prefix in (fp_clips or {}).items():
        t = resolve_triplet(prefix)
        fp_frames[name] = [therm for _ts, _f, therm, _mm
                           in iterate_triplet(t, detection)]
    cache = {"frames": frames, "radar_by_ts": radar_by_ts,
             "triplet_prefix": triplet_prefix, "fp_frames": fp_frames}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def _init_worker(cache_path: str, labels_path: str, clip_id: str,
                 tol_deg: float, skip_frames: int, dilate_frames: int):
    """Load shared state once per worker process."""
    from scripts.eval.thermal_gt_eval import (
        load_gt_intervals,
        mark_static_intervals,
        thermal_fov_deg,
    )
    from scripts.sensor_processing.pipeline import make_bins
    from scripts.utils.calibration import load_detection, load_intrinsics

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    intrinsics = load_intrinsics()
    detection = load_detection()
    # FOV clip follows the DEFAULT config's bearing model. Sweeping
    # thermal.bearing_model itself would silently mis-clip GT: don't.
    fov_lo, fov_hi, _P, _wh = thermal_fov_deg(
        intrinsics, str(detection["thermal"].get("bearing_model", "linear")))
    gt = load_gt_intervals(Path(labels_path), clip_id,
                           intrinsics["fisheye"]["K"], fov_lo, fov_hi)
    mark_static_intervals(gt, fov_lo, fov_hi)
    _G.update(
        frames=cache["frames"],
        fp_frames=cache.get("fp_frames", {}),
        radar_by_ts=cache["radar_by_ts"],
        gt=gt,
        intrinsics=intrinsics,
        base_detection=detection,
        fov=(fov_lo, fov_hi),
        edges=make_bins(detection["fusion"])[0],
        tol_deg=tol_deg,
        skip_frames=skip_frames,
        dilate_frames=dilate_frames,
    )


def _run_combo(combo: dict) -> dict:
    """Run the thermal path for one parameter combo and score it."""
    from scripts.eval.thermal_gt_eval import score_frames
    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline

    det = deepcopy(_G["base_detection"])
    for k, v in combo.items():
        _set_dotted_create(det, k, v)
    pipeline = ObstacleDetectionPipeline(_G["intrinsics"], det)
    thermal_by_ts: dict[str, list[float]] = {}
    for fidx, (ts, therm) in enumerate(_G["frames"]):
        t_res = pipeline.process_thermal(therm)
        if fidx < _G["skip_frames"]:
            continue
        thermal_by_ts[ts] = list(t_res.angles)
    m = score_frames(_G["gt"], thermal_by_ts, _G["radar_by_ts"],
                     tol_deg=_G["tol_deg"], bin_edges=_G["edges"],
                     fov=_G["fov"], dilate_frames=_G["dilate_frames"])
    row = m.as_row()
    # FP-reference clips (target-free scenes): detections/frame under this
    # combo. A FRESH pipeline per clip: the running-mean/MOG2 background is
    # per-clip state. First 10 frames are warm-up (background settling from
    # initial_average) and are excluded from the rate.
    for name, frames in _G["fp_frames"].items():
        fp_pipeline = ObstacleDetectionPipeline(_G["intrinsics"], det)
        n_det = n_fr = 0
        for i, therm in enumerate(frames):
            t_res = fp_pipeline.process_thermal(therm)
            if i < 10:
                continue
            n_fr += 1
            n_det += len(t_res.angles)
        row[f"fpclip_{name}"] = round(n_det / n_fr, 3) if n_fr else float("nan")
    row["combo_id"] = _combo_id(combo)
    row["combo_json"] = json.dumps(combo)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", required=True,
                    help="YAML with `grid:` (dotted thermal.* keys -> value "
                         "lists) and optional triplet/labels overrides.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tol", type=float, default=None)
    args = ap.parse_args()

    from scripts.utils.calibration import load_detection, load_intrinsics
    from scripts.eval.thermal_gt_eval import DEFAULT_SKIP_FRAMES, DEFAULT_TOL_DEG

    with open(args.grid) as f:
        grid_cfg = yaml.safe_load(f)
    grid: dict[str, list] = grid_cfg["grid"]
    bad = [k for k in grid if not k.startswith("thermal.")]
    if bad:
        raise SystemExit(f"non-thermal keys in grid (unsupported here): {bad}")
    triplet_prefix = grid_cfg.get(
        "triplet", "data/captures/2026-07-08/2026-07-08_16-37-01")
    labels_path = str(REPO_ROOT / grid_cfg.get(
        "labels", "labels/qwen/det_2026-07-08.jsonl"))
    p = Path(triplet_prefix)
    clip_id = f"{p.parent.name}/{p.name}"
    tol = args.tol if args.tol is not None else float(
        grid_cfg.get("tol_deg", DEFAULT_TOL_DEG))
    skip = int(grid_cfg.get("skip_frames", DEFAULT_SKIP_FRAMES))
    dilate = int(grid_cfg.get("gt_dilate_frames", 0))

    combos = _enumerate_grid(grid)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_DIR / f"sweep_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.grid, out_dir / "grid.yaml")
    print(f"{len(combos)} combos -> {out_dir}")

    fp_clips = grid_cfg.get("fp_clips", {}) or {}
    cache_path = out_dir / "frame_cache.pkl"
    print("building frame/radar cache …")
    build_cache(triplet_prefix, load_detection(), load_intrinsics(), cache_path,
                fp_clips=fp_clips)

    init_args = (str(cache_path), labels_path, clip_id, tol, skip, dilate)
    rows: list[dict] = []
    if args.workers <= 1:
        _init_worker(*init_args)
        for i, combo in enumerate(combos):
            rows.append(_run_combo(combo))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(combos)}")
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=init_args) as pool:
            for i, row in enumerate(pool.imap_unordered(_run_combo, combos)):
                rows.append(row)
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{len(combos)}")

    # Rank by object-grain F1 (lenient precision: radar-corroborated
    # detections are not chargeable), tie-break bin F1. per_class is kept
    # as JSON so the CSV stays flat.
    for r in rows:
        r["per_class"] = json.dumps(r["per_class"])
    rows.sort(key=lambda r: (-(r["f1"] if r["f1"] == r["f1"] else -1),
                             -(r["bin_f1"] if r["bin_f1"] == r["bin_f1"] else -1)))
    cols = ["combo_id", "f1", "recall", "recall_static", "recall_moving",
            "precision_lenient", "precision_strict", "bearing_mae",
            "fp_per_frame", "bin_p", "bin_r", "bin_f1", "n_det", "n_det_tp",
            "n_det_radar", "n_det_fp", "n_gt", "n_frames", "per_class",
            "combo_json"]
    cols[9:9] = [f"fpclip_{name}" for name in fp_clips]
    with open(out_dir / "ranked.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c) for c in cols} for r in rows)
    cache_path.unlink()   # 30+ MB, reproducible: don't leave it around

    print(f"wrote {out_dir / 'ranked.csv'}")
    for r in rows[:5]:
        print(f"  f1={r['f1']:.3f} R={r['recall']:.3f} "
              f"(S {r['recall_static']:.2f}/M {r['recall_moving']:.2f}) "
              f"P={r['precision_lenient']:.3f} fp/f={r['fp_per_frame']:.2f} "
              f"binF1={r['bin_f1']:.3f}  {r['combo_id']}")


if __name__ == "__main__":
    main()
