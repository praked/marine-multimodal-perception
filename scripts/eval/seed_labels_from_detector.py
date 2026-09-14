"""Seed the dashboard label store from a detector run.

Pre-fills labels/manual.jsonl with a detector run's boxes (remapped onto the
eval-clip frame_ids) as dashboard-manual records, so you open the eval clip
already populated and only add/delete/fix a few. Idempotent: existing records
for the eval mission are replaced, not duplicated.

    python -m scripts.eval.seed_labels_from_detector \\
        --pred results/grid/e10_se1200_bt02_tt018/labels.jsonl \\
        --mapping labels/eval_smoke36_mapping.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.sensor_processing.pipeline import angle_to_bin, make_bins
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import REPO_ROOT

MANUAL_PATH = REPO_ROOT / "labels" / "manual.jsonl"


def _frame_meta(fake_fid: str):
    """(scene, triplet_ts, frame_ts) from <scene>/<ts>/ts=HH-MM-SS.f."""
    scene, triplet_ts, sel = fake_fid.split("/")
    frame_ts = sel[3:].replace("-", ":", 2) if sel.startswith("ts=") else None
    return scene, triplet_ts, frame_ts


def _derive_bins(bboxes, w, h, cx, pix_deg, fusion):
    edges, centers = make_bins(fusion)
    bins = set()
    for bb in bboxes:
        x0, _y0, x1, _y1 = bb["xyxy"]
        angle = ((x0 + x1) / 2.0 * w - cx) / pix_deg
        idx = angle_to_bin(angle, edges)
        if idx is not None:
            bins.add(int(centers[idx]))
    return sorted(bins)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="Detector labels JSONL.")
    ap.add_argument("--mapping", required=True,
                    help="eval_smoke36_mapping.json (original<->fake ids).")
    ap.add_argument("--manual", default=str(MANUAL_PATH),
                    help="Label store to seed (default labels/manual.jsonl).")
    args = ap.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    o2f = mapping["original_to_fake"]
    eval_scene = next(iter(mapping["fake_to_original"])).split("/")[0]

    intr = load_intrinsics()["fisheye"]
    fusion = load_detection()["fusion"]

    # Build seeded records (dashboard-manual schema) for the eval-clip frames.
    seeded = {}
    for line in Path(args.pred).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        fake = o2f.get(r["frame_id"])
        if fake is None:
            continue
        scene, triplet_ts, frame_ts = _frame_meta(fake)
        w = r.get("width", 864)
        h = r.get("height", 648)
        bboxes = [{"cls": b["cls"], "xyxy": [float(v) for v in b["xyxy"]]}
                  for b in r.get("fisheye_bboxes", [])]
        seeded[fake] = {
            "frame_id": fake, "scene": scene, "triplet_ts": triplet_ts,
            "frame_ts": frame_ts, "source": "dashboard-manual", "audited": True,
            "fisheye_bboxes": bboxes,
            "obstacle_bins_fisheye": _derive_bins(
                bboxes, w, h, intr["cx"], intr["pix_deg_ratio"], fusion),
            "width": w, "height": h,
        }

    # Merge: keep all non-eval records, drop old eval records, add seeded ones.
    manual = Path(args.manual)
    kept = []
    if manual.exists():
        for line in manual.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("scene") == eval_scene:
                continue  # replace any prior eval-clip records
            kept.append(rec)
    kept.extend(seeded.values())

    manual.parent.mkdir(parents=True, exist_ok=True)
    tmp = manual.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in kept))
    os.replace(tmp, manual)

    n_box = sum(len(r["fisheye_bboxes"]) for r in seeded.values())
    print(f"seeded {len(seeded)} eval-clip frames ({n_box} boxes) into {manual}")
    print(f"  scene: {eval_scene}  (other records preserved: {len(kept) - len(seeded)})")
    print("Open the clip to refine:")
    print(f"  python -m scripts.eval.dashboard --triplet "
          f"data/captures/{eval_scene}/{next(iter(seeded)).split('/')[1]}")


if __name__ == "__main__":
    main()
