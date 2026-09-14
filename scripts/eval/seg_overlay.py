"""Render water-segmentation range QA overlays (non-interactive, Agg).

For each masked frame of a clip, draws on the undistorted fisheye image:
  * a translucent water tint + the water-edge line (segmentation horizon),
  * each detection box,
  * the bbox-bottom point (red) and the segmentation waterline-contact point
    (green) with both range estimates,
so the range improvement is visible at a glance. Writes PNGs to
results/seg_overlay/<clip>/.

Needs masks under --seg-root (data/seg). Run after scripts/gpu_seg/06_extract.

    python -m scripts.eval.seg_overlay --triplet data/Boats/2025-06-23_16-21-07 \
        --seg-root data/seg --limit 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.eval.dashboard import _frame_id_for  # noqa: E402
from scripts.sensor_processing.pipeline import (  # noqa: E402
    ObstacleDetectionPipeline,
    iterate_triplet,
)
from scripts.utils.calibration import load_detection, load_intrinsics  # noqa: E402
from scripts.utils.datasets import resolve_triplet  # noqa: E402
from scripts.utils.segmentation import (  # noqa: E402
    WATER,
    ContactParams,
    SegProvider,
    horizon_from_water,
    water_edge_contact,
)


def _pipelines(seg_root):
    intr = load_intrinsics()
    det = load_detection()
    base = ObstacleDetectionPipeline(intr, {**det, "segmentation": {"enabled": False}})
    seg_cfg = {**det.get("segmentation", {}), "enabled": True, "seg_root": seg_root,
               "use_for_range": True, "use_for_horizon": True}
    seg = ObstacleDetectionPipeline(intr, {**det, "segmentation": seg_cfg},
                                    seg_provider=SegProvider(seg_root))
    return base, seg


def render_clip(triplet_prefix, seg_root, out_root, limit):
    triplet = resolve_triplet(triplet_prefix)
    base, seg = _pipelines(seg_root)
    det = load_detection()
    provider = SegProvider(seg_root)
    cp = ContactParams(**{**{"search_pad_px": 40, "min_columns": 5,
                             "water_below_px": 3, "percentile": 80},
                          **(det.get("segmentation", {}).get("contact", {}) or {})})
    out_dir = out_root / f"{triplet.scene}__{triplet.timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for ts, fisheye_frame, _t, _p in iterate_triplet(triplet, det):
        if fisheye_frame is None:
            continue
        fid = _frame_id_for(triplet.scene, triplet.timestamp, ts)
        mask = provider.get(fid)
        if mask is None:
            continue
        rb = base.process_fisheye(fisheye_frame)
        rs = seg.process_fisheye(fisheye_frame, frame_id=fid)
        img = rs.undistorted[:, :, ::-1]  # BGR->RGB

        fig, ax = plt.subplots(figsize=(9, 6.8))
        ax.imshow(img)
        # water tint
        water = (mask == WATER)
        tint = np.zeros((*mask.shape, 4))
        tint[water] = [0.16, 0.65, 0.88, 0.25]
        ax.imshow(tint)
        # water-edge line
        s, b, conf = horizon_from_water(mask)
        if conf > 0:
            xs = np.array([0, mask.shape[1] - 1])
            ax.plot(xs, s * xs + b, "c-", lw=1.2, alpha=0.8,
                    label=f"water edge (conf {conf:.2f})")
        # detections
        for i, ((px, py), size) in enumerate(zip(rs.coords, rs.sizes)):
            r = size / 2.0
            ax.add_patch(plt.Rectangle((px - r, py - r), size, size,
                                       fill=False, ec="yellow", lw=1.0))
            rb_r = rb.ranges[i] if i < len(rb.ranges) else None
            rs_r = rs.ranges[i] if i < len(rs.ranges) else None
            ax.plot(px, py + r, "r.", ms=8)            # bbox bottom
            contact = water_edge_contact(mask, (px - r, py - r, px + r, py + r), cp)
            if contact is not None:
                ax.plot(contact[0], contact[1], "g.", ms=9)  # seg contact
            txt = f"box {rb_r:.1f}m" if rb_r else "box -"
            txt += f"\nseg {rs_r:.1f}m" if rs_r else "\nseg -"
            ax.text(px + r + 3, py, txt, color="w", fontsize=7,
                    bbox=dict(fc="black", alpha=0.5, pad=1))
        ax.set_title(f"{triplet.clip_id}  {ts}", fontsize=9)
        ax.axis("off")
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"ts={ts.replace(':', '-')}.png", dpi=90)
        plt.close(fig)
        n += 1
        if limit and n >= limit:
            break
    print(f"{triplet.clip_id}: {n} overlays -> {out_dir}")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--triplet", action="append", default=[])
    ap.add_argument("--seg-root", default="data/seg")
    ap.add_argument("--out", type=Path, default=Path("results/seg_overlay"))
    ap.add_argument("--limit", type=int, default=20, help="max frames per clip (0=all)")
    args = ap.parse_args(argv)
    if not Path(args.seg_root).is_dir():
        print(f"No masks at {args.seg_root}; run scripts/gpu_seg first.")
        return 1
    clips = args.triplet or [
        "data/Boats/2025-06-23_16-21-07", "data/Ducks/2025-07-15_05-36-18",
        "data/OpenWater/2025-06-23_23-53-27", "data/Rain/2025-07-15_02-59-02",
    ]
    total = 0
    for c in clips:
        try:
            total += render_clip(c, args.seg_root, args.out, args.limit)
        except Exception as exc:  # noqa: BLE001
            print(f"  {c}: skipped ({exc})")
    print(f"Total {total} overlays -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
