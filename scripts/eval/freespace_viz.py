"""Visualise the segmentation-driven navigable free-space profile (Phase 1).

For each frame of a clip: tint the navigable water (contiguous water from the
bottom), draw the free-space boundary curve (nearest non-navigable edge per
column), and annotate the per-azimuth-bin free distance (metres / open /
BLOCKED). This is the "how far can I go in each direction" nav primitive derived
purely from the water mask, no RANSAC horizon.

    python -m scripts.eval.freespace_viz \
        --triplet data/captures/2026-06-17_institutionone_day1/2026-06-17_13-16-25 \
        --seg-root data/seg --out results/phase1_freespace --limit 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.utils.calibration import load_detection, load_intrinsics  # noqa: E402
from scripts.utils.datasets import resolve_triplet  # noqa: E402
from scripts.utils.geometry import UP_LEVEL, load_extrinsics, up_from_horizon_line  # noqa: E402
from scripts.utils.segmentation import (  # noqa: E402
    WATER,
    SegProvider,
    free_space_profile,
    horizon_from_water,
)


def _clipdir(scene, ts):
    return f"{scene}__{ts}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triplet", required=True)
    ap.add_argument("--seg-root", default="data/seg")
    ap.add_argument("--frames-root", type=Path, default=Path("data/seg_frames"))
    ap.add_argument("--out", type=Path, default=Path("results/phase1_freespace"))
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args(argv)

    triplet = resolve_triplet(args.triplet)
    intr = load_intrinsics()["fisheye"]
    det = load_detection()
    K = np.asarray(intr["K"], float)
    cx, pix_deg = float(intr["cx"]), float(intr["pix_deg_ratio"])
    height = float((load_extrinsics().get("camera_height_m", {}) or {}).get("fisheye", 0.27))
    fcfg = det["fusion"]
    edges = np.arange(fcfg["bin_min_deg"], fcfg["bin_max_deg"] + fcfg["bin_step_deg"],
                      fcfg["bin_step_deg"])
    max_range = float((det.get("range", {}) or {}).get("max_range_m", 15.0))

    seg_provider = SegProvider(args.seg_root)
    clip_frames = args.frames_root / _clipdir(triplet.scene, triplet.timestamp)
    out_dir = args.out / _clipdir(triplet.scene, triplet.timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(clip_frames.glob("ts=*.jpg"))
    n = 0
    for jpg in jpgs:
        # jpg.stem is "ts=<HH-MM-SS.f>", exactly the frame_id tail.
        fid = f"{triplet.scene}/{triplet.timestamp}/{jpg.stem}"
        mask = seg_provider.get(fid)
        if mask is None:
            continue
        import cv2
        img = cv2.imread(str(jpg))[:, :, ::-1]              # RGB
        h, w = mask.shape[:2]

        # up-vector from the seg water edge (else level)
        s, b, conf = horizon_from_water(mask)
        up = up_from_horizon_line(s, b, K) if conf > 0.3 else UP_LEVEL

        prof = free_space_profile(mask, K, up, height, cx, pix_deg, edges,
                                  max_range_m=max_range)

        # navigable region = contiguous water from the bottom
        water = mask == WATER
        contig = np.cumprod(water[::-1, :], axis=0).astype(bool)[::-1, :]

        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.imshow(img)
        nav = np.zeros((h, w, 4))
        nav[contig] = [0.1, 0.9, 0.3, 0.28]                 # green navigable tint
        ax.imshow(nav)
        cols = np.arange(w)
        ax.plot(cols, prof.boundary_rows.clip(0, h - 1), color="lime", lw=1.5)

        # per-bin free distance annotation at the bin-centre column
        for center, dist, blk in zip(prof.bin_centers, prof.free_dist_m, prof.blocked):
            col = int(cx + center * pix_deg)
            if not (0 <= col < w):
                continue
            txt = "BLOCKED" if blk else ("open" if dist is None else f"{dist:.1f} m")
            colour = "red" if blk else ("cyan" if dist is None else "yellow")
            ax.text(col, h - 8, txt, color=colour, fontsize=7, ha="center",
                    rotation=90, va="bottom",
                    bbox=dict(fc="black", alpha=0.5, pad=0.5, edgecolor="none"))
            ax.axvline(col, color="white", alpha=0.15, lw=0.5)
        ax.set_title(f"{triplet.clip_id}  {jpg.stem}   navigable free-space "
                     f"(water-mask, no RANSAC)", fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{jpg.stem}.png", dpi=95)
        plt.close(fig)
        n += 1
        if args.limit and n >= args.limit:
            break

    print(f"{triplet.clip_id}: {n} free-space frames -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
