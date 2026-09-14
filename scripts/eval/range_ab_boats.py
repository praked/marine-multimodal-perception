"""Range A/B on the YOLO boat detections: bbox-bottom vs seg waterline-contact.

Cleaner than seg_overlay (which ranges the blob detector's ~30 firings/frame):
here we range the typed YOLO **boat** boxes (data/det) two ways:
  * bbox-bottom (the old method), and
  * the segmentation waterline-contact point inside the box (the new method),
and show how the distance number moves. Picks the closest boats across the
mission so the improvement is visible, writes a CSV + a contact sheet.

    python -m scripts.eval.range_ab_boats --out results/phase1_range_ab_boats --top 20
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.utils.calibration import load_detection, load_intrinsics  # noqa: E402
from scripts.utils.geometry import (  # noqa: E402
    UP_LEVEL,
    load_extrinsics,
    range_from_contact_point,
    range_from_water_plane_up,
    up_from_horizon_line,
)
from scripts.utils.segmentation import (  # noqa: E402
    ContactParams,
    horizon_from_water,
    load_seg_mask,
    water_edge_contact,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det-root", default="data/det")
    ap.add_argument("--seg-root", default="data/seg")
    ap.add_argument("--frames-root", type=Path, default=Path("data/seg_frames"))
    ap.add_argument("--classes", default="boat_ship,row_boats")
    ap.add_argument("--min-conf", type=float, default=0.35)
    ap.add_argument("--out", type=Path, default=Path("results/phase1_range_ab_boats"))
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args(argv)

    intr = load_intrinsics()["fisheye"]
    det = load_detection()
    K = np.asarray(intr["K"], float)
    height = float((load_extrinsics().get("camera_height_m", {}) or {}).get("fisheye", 0.27))
    max_range = float((det.get("range", {}) or {}).get("max_range_m", 15.0))
    cp = ContactParams(**{**{"search_pad_px": 40, "min_columns": 5,
                            "water_below_px": 3, "percentile": 80},
                          **(det.get("segmentation", {}).get("contact", {}) or {})})
    keep = set(args.classes.split(","))

    rows = []
    for jf in sorted(glob.glob(f"{args.det_root}/*.jsonl")):
        clipdir = Path(jf).stem
        for line in open(jf):
            rec = json.loads(line)
            boxes = [b for b in rec.get("fisheye_bboxes", [])
                     if b.get("cls") in keep and b.get("confidence", 0) >= args.min_conf]
            if not boxes:
                continue
            ts_token = rec["frame_id"].split("/")[-1]
            jpg = args.frames_root / clipdir / f"{ts_token}.jpg"
            mask_p = Path(args.seg_root) / clipdir / f"{ts_token}.png"
            if not (jpg.exists() and mask_p.exists()):
                continue
            mask = load_seg_mask(mask_p)
            h, w = mask.shape[:2]
            s, b, conf = horizon_from_water(mask)
            up = up_from_horizon_line(s, b, K) if conf > 0.3 else UP_LEVEL
            for bx in boxes:
                x0, y0, x1, y1 = bx["xyxy"]
                px = (x0 * w, y0 * h, x1 * w, y1 * h)
                r_box = range_from_water_plane_up(px, K, up, height, max_range_m=max_range)
                contact = water_edge_contact(mask, px, cp)
                r_seg = (range_from_contact_point(contact, K, up, height, max_range_m=max_range)
                         if contact is not None else None)
                if r_box is None and r_seg is None:
                    continue
                rows.append({"clip": clipdir, "frame": ts_token, "cls": bx["cls"],
                             "conf": round(bx["confidence"], 3), "px": px,
                             "bbox_range_m": r_box, "seg_range_m": r_seg,
                             "delta_m": (None if (r_box is None or r_seg is None)
                                         else round(r_seg - r_box, 2))})

    # summary
    both = [r for r in rows if r["bbox_range_m"] is not None and r["seg_range_m"] is not None]
    deltas = [r["seg_range_m"] - r["bbox_range_m"] for r in both]
    print(f"{len(rows)} boat detections ranged; {len(both)} have both estimates.")
    if deltas:
        farther = sum(d > 0 for d in deltas)
        print(f"  mean delta (seg - bbox): {np.mean(deltas):+.2f} m   "
              f"median {np.median(deltas):+.2f} m   "
              f"% farther: {100*farther/len(deltas):.0f}%")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "range_ab_boats.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["clip", "frame", "cls", "conf",
                                             "bbox_range_m", "seg_range_m", "delta_m"])
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({k: r[k] for k in wcsv.fieldnames})

    # contact sheet: closest boats by seg range (the operationally relevant ones)
    top = sorted(both, key=lambda r: r["seg_range_m"])[:args.top]
    if top:
        cols = 5
        rws = (len(top) + cols - 1) // cols
        fig, axes = plt.subplots(rws, cols, figsize=(cols * 3.2, rws * 2.6))
        axes = np.array(axes).ravel()
        for ax, r in zip(axes, top):
            jpg = args.frames_root / r["clip"] / f"{r['frame']}.jpg"
            im = cv2.imread(str(jpg))[:, :, ::-1].copy()
            x0, y0, x1, y1 = [int(v) for v in r["px"]]
            cv2.rectangle(im, (x0, y0), (x1, y1), (255, 220, 0), 2)
            cv2.circle(im, ((x0 + x1) // 2, y1), 5, (255, 0, 0), -1)     # bbox bottom (red)
            pad = 50
            crop = im[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
            ax.imshow(crop)
            ax.set_title(f"bbox {r['bbox_range_m']:.1f}m -> seg {r['seg_range_m']:.1f}m "
                         f"({r['delta_m']:+.1f})", fontsize=7)
            ax.axis("off")
        for j in range(len(top), len(axes)):
            axes[j].axis("off")
        fig.suptitle("Closest boats: bbox-bottom vs seg waterline-contact range", fontsize=12)
        fig.tight_layout()
        fig.savefig(args.out / "closest_boats.png", dpi=100)
        plt.close(fig)
    print(f"-> {args.out}/closest_boats.png + range_ab_boats.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
