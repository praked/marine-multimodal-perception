"""Render a labels JSONL as bbox overlays + a contact-sheet montage.

Used both for ad-hoc audits and by the prompt-iteration harness
(scripts/gpu_labeling/iterate_prompt.sh) to eyeball a prompt's output.

Frames are pulled through the same ``iter_clip_frames`` the labeller uses
(undistorted, dashboard frame_ids), so overlays land in the exact coordinate
space the boxes were written in.

    python -m scripts.eval.render_label_overlays \\
        --labels labels/qwen/qwen_2026-06-17_institutionone_day1.jsonl \\
        --captures-dir data/captures/2026-06-17_institutionone_day1 \\
        --out-dir results/qwen_audit/latest
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import cv2

from scripts.eval.qwen_batch_labeler import iter_clip_frames
from scripts.utils.calibration import load_detection, load_intrinsics

CLASS_COLORS = {
    "boat": (0, 200, 0), "buoy": (0, 0, 230), "person": (255, 80, 0),
    "other": (160, 0, 160), "duck": (0, 200, 255), "structure": (0, 165, 255),
}


def _draw(und, rec, banner):
    img = und.copy()
    h, w = img.shape[:2]
    for b in rec["fisheye_bboxes"]:
        x0, y0, x1, y1 = b["xyxy"]
        p0 = (int(x0 * w), int(y0 * h))
        p1 = (int(x1 * w), int(y1 * h))
        c = CLASS_COLORS.get(b["cls"], (255, 255, 255))
        cv2.rectangle(img, p0, p1, c, 2)
        cv2.putText(img, b["cls"][:5], (p0[0], max(11, p0[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    for col, th in ((((0, 0, 0)), 3), ((255, 255, 255), 1)):
        cv2.putText(img, banner, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th,
                    cv2.LINE_AA)
    return img


def _montage(thumbs, out_dir: Path, prefix: str, cols: int, per_page: int):
    paths = []
    pages = max(1, math.ceil(len(thumbs) / per_page))
    blank = thumbs[0] * 0 if thumbs else None
    for pg in range(pages):
        chunk = thumbs[pg * per_page:(pg + 1) * per_page]
        if not chunk:
            break
        rows = []
        for ri in range(math.ceil(len(chunk) / cols)):
            row = chunk[ri * cols:ri * cols + cols]
            row = row + [blank] * (cols - len(row))
            rows.append(cv2.hconcat(row))
        out = out_dir / f"{prefix}_{pg + 1:02d}.png"
        cv2.imwrite(str(out), cv2.vconcat(rows))
        paths.append(out)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="Labels JSONL to render.")
    ap.add_argument("--captures-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--detections-only", action="store_true",
                    help="Only render frames that have >=1 bbox.")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--per-page", type=int, default=36)
    ap.add_argument("--thumb-w", type=int, default=288)
    ap.add_argument("--thumb-h", type=int, default=216)
    ap.add_argument("--exposure", type=float, default=1.0,
                    help="Brighten frames before drawing (match the detector run).")
    args = ap.parse_args()

    captures_dir = Path(args.captures_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.labels) if l.strip()]
    if args.detections_only:
        rows = [r for r in rows if r["fisheye_bboxes"]]
    by_clip = collections.defaultdict(dict)
    for r in rows:
        by_clip[r["triplet_ts"]][r["frame_ts"]] = r

    intr = load_intrinsics()
    det = load_detection()
    thumbs = []
    cls_count = collections.Counter()
    for clip_ts in sorted(by_clip):
        want = dict(by_clip[clip_ts])
        for fid, scene, tts, fts, und in iter_clip_frames(
                captures_dir, clip_ts, 1, det, intr):
            if fts not in want:
                continue
            rec = want.pop(fts)
            if args.exposure != 1.0:
                und = cv2.convertScaleAbs(und, alpha=args.exposure, beta=0)
            for b in rec["fisheye_bboxes"]:
                cls_count[b["cls"]] += 1
            n = len(rec["fisheye_bboxes"])
            img = _draw(und, rec, f"{clip_ts[-8:]} {fts} [{n}]")
            thumbs.append(cv2.resize(img, (args.thumb_w, args.thumb_h)))
            if not want:
                break

    sheets = _montage(thumbs, out_dir, "contact", args.cols, args.per_page)
    print(f"rendered {len(thumbs)} frames -> {len(sheets)} contact sheet(s) in {out_dir}")
    print(f"bbox classes: {dict(cls_count)}")
    for s in sheets:
        print(f"  {s}")


if __name__ == "__main__":
    main()
