"""Quantify the segmentation range fix: bbox-bottom vs waterline-contact.

For each clip (default: the four frozen val clips) this runs the SAME fisheye
detector twice: once with segmentation OFF (range from bbox bottom-centre +
RANSAC/level horizon) and once ON (range from the water-edge contact point +
water-edge up-vector), and reports, per detection, how the range estimate
moves. The detector is deterministic, so detections align by index and only the
range differs.

The expected signal: where the box over-captures water (tilted horizon, loose
box), the contact point sits higher than the box bottom, so the segmentation
range is LARGER (the object was being reported as too close).

Needs masks under --seg-root (data/seg, produced by scripts/gpu_seg). Frames
without a mask are skipped. Writes a per-detection CSV and prints a summary.

    python -m scripts.eval.seg_range_ablation \
        --triplet data/Boats/2025-06-23_16-21-07 \
        --seg-root data/seg --out results/seg_range_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from scripts.eval.dashboard import _frame_id_for
from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline, iterate_triplet
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import resolve_triplet
from scripts.utils.segmentation import SegProvider

DEFAULT_CLIPS = [
    "data/Boats/2025-06-23_16-21-07",
    "data/Ducks/2025-07-15_05-36-18",
    "data/OpenWater/2025-06-23_23-53-27",
    "data/Rain/2025-07-15_02-59-02",
]


def _mk_pipelines(seg_root: str):
    intr = load_intrinsics()
    det = load_detection()
    base_det = {**det, "segmentation": {"enabled": False}}
    seg_det = {**det, "segmentation": {
        "enabled": True, "seg_root": seg_root,
        "use_for_range": True, "use_for_horizon": True,
        "horizon_min_confidence": det.get("segmentation", {}).get(
            "horizon_min_confidence", 0.4),
        "contact": det.get("segmentation", {}).get("contact", {}),
    }}
    provider = SegProvider(seg_root)
    base = ObstacleDetectionPipeline(intr, base_det)
    seg = ObstacleDetectionPipeline(intr, seg_det, seg_provider=provider)
    return base, seg, provider


def run_clip(triplet_prefix: str, seg_root: str, rows: list[dict]) -> dict:
    triplet = resolve_triplet(triplet_prefix)
    base, seg, provider = _mk_pipelines(seg_root)
    det = load_detection()

    n_frames = n_masked = n_det = n_contact = 0
    deltas: list[float] = []
    for ts, fisheye_frame, _thermal, _pts in iterate_triplet(triplet, det):
        if fisheye_frame is None:
            continue
        n_frames += 1
        fid = _frame_id_for(triplet.scene, triplet.timestamp, ts)
        mask = provider.get(fid)
        if mask is None:
            continue
        n_masked += 1
        rb = base.process_fisheye(fisheye_frame)
        rs = seg.process_fisheye(fisheye_frame, frame_id=fid)
        for i, (coord, size) in enumerate(zip(rb.coords, rb.sizes)):
            n_det += 1
            base_r = rb.ranges[i] if i < len(rb.ranges) else None
            seg_r = rs.ranges[i] if i < len(rs.ranges) else None
            changed = base_r is not None and seg_r is not None and abs(seg_r - base_r) > 1e-6
            if changed:
                n_contact += 1
                deltas.append(seg_r - base_r)
            rows.append({
                "clip": triplet.clip_id,
                "frame_id": fid,
                "u": coord[0], "v": coord[1], "size": round(size, 1),
                "base_range_m": None if base_r is None else round(base_r, 3),
                "seg_range_m": None if seg_r is None else round(seg_r, 3),
                "delta_m": None if not changed else round(seg_r - base_r, 3),
            })

    summary = {
        "clip": triplet.clip_id,
        "frames": n_frames,
        "masked_frames": n_masked,
        "detections": n_det,
        "range_changed": n_contact,
        "mean_delta_m": round(statistics.fmean(deltas), 3) if deltas else None,
        "median_delta_m": round(statistics.median(deltas), 3) if deltas else None,
        "frac_farther": round(sum(d > 0 for d in deltas) / len(deltas), 3) if deltas else None,
    }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--triplet", action="append", default=[],
                    help="triplet prefix (repeatable); default = 4 val clips")
    ap.add_argument("--seg-root", default="data/seg")
    ap.add_argument("--out", type=Path, default=Path("results/seg_range_ablation.csv"))
    args = ap.parse_args(argv)

    if not Path(args.seg_root).is_dir():
        print(f"No masks at {args.seg_root}; run scripts/gpu_seg first.")
        return 1

    clips = args.triplet or DEFAULT_CLIPS
    rows: list[dict] = []
    summaries = []
    for c in clips:
        if not Path(c).parent.exists() and not Path(c + ".mp4").exists():
            # resolve_triplet handles prefixes; just attempt and report failures
            pass
        try:
            summaries.append(run_clip(c, args.seg_root, rows))
        except Exception as exc:  # noqa: BLE001
            print(f"  {c}: skipped ({exc})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\nPer-detection CSV: {args.out}  ({len(rows)} rows)")
    print("\n=== Summary (segmentation range vs bbox-bottom) ===")
    hdr = f"{'clip':<32}{'frames':>7}{'masked':>7}{'dets':>6}{'changed':>8}{'mean Δm':>9}{'med Δm':>8}{'%farther':>9}"
    print(hdr)
    for s in summaries:
        print(f"{s['clip']:<32}{s['frames']:>7}{s['masked_frames']:>7}"
              f"{s['detections']:>6}{s['range_changed']:>8}"
              f"{str(s['mean_delta_m']):>9}{str(s['median_delta_m']):>8}"
              f"{str(s['frac_farther']):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
