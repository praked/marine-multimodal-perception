"""Water-reflection post-filter for teacher label JSONL (prompt-v1 companion).

DART labels reflections of people/boats/pillars as the object itself
(teacher_comparison.md §4b #2). A reflection is geometrically recognisable
without any model: its box sits (almost) entirely on WATER pixels of the
frame's segmentation mask AND a same-class box stands above it sharing a
horizontal column (the mirror partner). Both conditions are required — a
swimmer or duck legitimately sits on water but has no partner above, and a
moored boat above the waterline fails the water-fraction test.

    python -m scripts.eval.filter_reflections \
        --labels labels/dart --seg-dir data/seg --out labels/dart_v1 \
        [--report-only] [--water-min 0.85] [--overlap-min 0.5] [--flag-only]

--report-only: print per-file drop counts, write nothing.
--flag-only:   keep every box, add "reflection": true instead of dropping.
Frames without a mask pass through untouched (honest no-op, mirrors the
segmentation gating convention).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.segmentation import (  # noqa: E402
    WATER,
    load_seg_mask,
    mask_path_for_frame_id,
)


def water_fraction(seg: np.ndarray, xyxy: list[float]) -> float:
    """Fraction of the (normalised-coords) box covered by WATER pixels."""
    h, w = seg.shape[:2]
    x0 = max(0, int(round(xyxy[0] * w)))
    y0 = max(0, int(round(xyxy[1] * h)))
    x1 = min(w, int(round(xyxy[2] * w)))
    y1 = min(h, int(round(xyxy[3] * h)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = seg[y0:y1, x0:x1]
    return float((patch == WATER).mean())


def _h_overlap(a: list[float], b: list[float]) -> float:
    """Horizontal overlap as a fraction of the NARROWER box's width."""
    inter = min(a[2], b[2]) - max(a[0], b[0])
    narrow = min(a[2] - a[0], b[2] - b[0])
    return inter / narrow if narrow > 0 and inter > 0 else 0.0


def reflection_indices(
    boxes: list[dict],
    seg: np.ndarray | None,
    water_min: float = 0.85,
    overlap_min: float = 0.5,
) -> set[int]:
    """Indices of boxes judged to be water reflections.

    Box i is a reflection iff water_fraction >= water_min AND some
    same-class box j sits above it (j's bottom above i's centre) with
    horizontal overlap >= overlap_min of the narrower box. The partner
    itself must NOT be mostly on water (else two stacked reflections, or a
    wave pattern, would suppress each other arbitrarily).
    """
    if seg is None:
        return set()
    wf = [water_fraction(seg, b["xyxy"]) for b in boxes]
    out: set[int] = set()
    for i, b in enumerate(boxes):
        if wf[i] < water_min:
            continue
        cy = (b["xyxy"][1] + b["xyxy"][3]) / 2
        for j, p in enumerate(boxes):
            if j == i or p["cls"] != b["cls"]:
                continue
            if p["xyxy"][3] <= cy and wf[j] < water_min and _h_overlap(b["xyxy"], p["xyxy"]) >= overlap_min:
                out.add(i)
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True, help="dir of det_*.jsonl")
    ap.add_argument("--seg-dir", default="data/seg")
    ap.add_argument("--out", help="output dir for filtered JSONL")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--flag-only", action="store_true")
    ap.add_argument("--water-min", type=float, default=0.85)
    ap.add_argument("--overlap-min", type=float, default=0.5)
    args = ap.parse_args()
    if not args.report_only and not args.out:
        ap.error("--out is required unless --report-only")

    seg_root = Path(args.seg_dir)
    files = sorted(Path(args.labels).glob("det_*.jsonl"))
    total_boxes = total_drop = total_nomask = 0
    for f in files:
        out_records = []
        dropped = nomask = nboxes = 0
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            boxes = rec.get("fisheye_bboxes", [])
            nboxes += len(boxes)
            path = mask_path_for_frame_id(seg_root, rec["frame_id"])
            seg = load_seg_mask(path) if path and path.exists() else None
            if seg is None and boxes:
                nomask += 1
            refl = reflection_indices(boxes, seg, args.water_min, args.overlap_min)
            dropped += len(refl)
            if args.flag_only:
                rec["fisheye_bboxes"] = [
                    ({**b, "reflection": True} if i in refl else b)
                    for i, b in enumerate(boxes)
                ]
            else:
                rec["fisheye_bboxes"] = [b for i, b in enumerate(boxes) if i not in refl]
            out_records.append(rec)
        total_boxes += nboxes
        total_drop += dropped
        total_nomask += nomask
        print(f"{f.name}: {nboxes} boxes, {dropped} reflections"
              + (f", {nomask} frames without mask" if nomask else ""))
        if not args.report_only:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f.name).write_text(
                "\n".join(json.dumps(r) for r in out_records) + "\n")
    pct = 100 * total_drop / total_boxes if total_boxes else 0
    print(f"TOTAL: {total_drop}/{total_boxes} boxes ({pct:.1f}%) judged reflections"
          + (f"; {total_nomask} frames without mask (untouched)" if total_nomask else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
