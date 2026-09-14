"""Union two teachers' pseudo-labels into one teacher-blind suggestion set.

Audits seeded from a single teacher inherit its misses (the auditor only
draws what the suggestions did not show). For the GroundingDINO vs DART
comparison (docs/plans/teacher_comparison.md) every audited frame must see
BOTH teachers' boxes, so the auditor arbitrates both. This tool merges
det_<clip>.jsonl files from two label dirs, frame by frame:

  - boxes from both teachers are kept, each tagged with its `source`;
  - a box that both teachers propose (same canonical class, IoU >= --iou)
    is emitted once (the higher-confidence copy) with `source: "both"`
    and `agree: true`, so the audit also yields an agreement statistic;
  - frames present in only one file are passed through as-is.

Output records keep the labeller contract (frame_id, fisheye_bboxes,
obstacle_bins_fisheye, ...) with `source: "union"` at record level, so the
dashboard `--pseudo` reader and the baker consume them unchanged. Frames are
paired by (scene, frame_ts) — NOT by raw frame_id — because the two teachers
attribute chunks differently (Track-A DINO wrote activity-level frame_ids,
DART per-chunk ones): pairing on frame_id silently split the same physical
frame into an only-DINO and an only-DART record (caught 2026-09-01 when the
outing's baked labels nearly doubled). The emitted frame_id/triplet_ts is
teacher A's when both exist (the bundle-side convention).

    python -m scripts.eval.merge_teacher_suggestions \
        --a labels/qwen --b /path/to/labels_dart --out labels/union \
        [--only 2026-08-26_afloat] [--iou 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def frame_key(r: dict) -> str | None:
    """Physical-frame identity: (scene, frame_ts). Chunk attribution differs
    between teachers, so frame_id must not take part in pairing."""
    scene = r.get("scene") or (r.get("frame_id", "").split("/") or [None])[0]
    fts = r.get("frame_ts")
    return f"{scene}|{fts}" if scene and fts else r.get("frame_id")


def load_dir(path: Path, only: str | None) -> dict[str, dict]:
    """(scene, frame_ts) key -> record (last one wins)."""
    out: dict[str, dict] = {}
    for f in sorted(path.glob("det_*.jsonl")):
        if only and only not in f.name:
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = frame_key(r)
            if k:
                out[k] = r
    return out


def merge_boxes(boxes_a: list[dict], boxes_b: list[dict], src_a: str, src_b: str,
                iou_thr: float) -> list[dict]:
    """Union with per-class IoU matching; matched pairs collapse to one box."""
    a = [dict(b, source=b.get("source", src_a)) for b in boxes_a]
    b = [dict(x, source=x.get("source", src_b)) for x in boxes_b]
    used_b: set[int] = set()
    out: list[dict] = []
    for ba in a:
        best, best_i = 0.0, -1
        for i, bb in enumerate(b):
            if i in used_b or bb.get("cls") != ba.get("cls"):
                continue
            v = iou(ba["xyxy"], bb["xyxy"])
            if v > best:
                best, best_i = v, i
        if best_i >= 0 and best >= iou_thr:
            bb = b[best_i]
            used_b.add(best_i)
            keep = ba if float(ba.get("confidence", 0)) >= float(bb.get("confidence", 0)) else bb
            merged = dict(keep, source="both", agree=True, iou_between=round(best, 3),
                          confidence_other=float((bb if keep is ba else ba).get("confidence", 0)))
            out.append(merged)
        else:
            out.append(dict(ba, agree=False))
    for i, bb in enumerate(b):
        if i not in used_b:
            out.append(dict(bb, agree=False))
    return out


def merge(records_a: dict[str, dict], records_b: dict[str, dict], src_a: str, src_b: str,
          iou_thr: float) -> tuple[dict[str, dict], dict]:
    fids = sorted(set(records_a) | set(records_b))
    merged: dict[str, dict] = {}
    stats = {"frames": len(fids), "only_a": 0, "only_b": 0, "both_frames": 0,
             "boxes_a": 0, "boxes_b": 0, "boxes_out": 0, "agreed": 0}
    for fid in fids:
        ra, rb = records_a.get(fid), records_b.get(fid)  # fid = (scene, frame_ts) key
        base = dict(ra or rb)
        ba = (ra or {}).get("fisheye_bboxes", []) if ra else []
        bb = (rb or {}).get("fisheye_bboxes", []) if rb else []
        stats["boxes_a"] += len(ba)
        stats["boxes_b"] += len(bb)
        if ra and rb:
            stats["both_frames"] += 1
        elif ra:
            stats["only_a"] += 1
        else:
            stats["only_b"] += 1
        boxes = merge_boxes(ba, bb, (ra or {}).get("source", src_a), (rb or {}).get("source", src_b), iou_thr)
        stats["boxes_out"] += len(boxes)
        stats["agreed"] += sum(1 for x in boxes if x.get("agree"))
        base["fisheye_bboxes"] = boxes
        base["source"] = "union"
        base["teachers"] = sorted({(ra or {}).get("source", src_a) if ra else None,
                                   (rb or {}).get("source", src_b) if rb else None} - {None})
        # bins are recomputed downstream from boxes where needed; drop the stale cache
        base.pop("obstacle_bins_fisheye", None)
        merged[fid] = base
    return merged, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True, help="label dir of teacher A (e.g. labels/qwen)")
    ap.add_argument("--b", required=True, help="label dir of teacher B (e.g. labels_dart)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="substring filter on file names")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--name-a", default="grounding-dino")
    ap.add_argument("--name-b", default="dart-sam3")
    args = ap.parse_args(argv)

    ra = load_dir(Path(args.a), args.only)
    rb = load_dir(Path(args.b), args.only)
    merged, stats = merge(ra, rb, args.name_a, args.name_b, args.iou)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for rec in merged.values():
        scene, chunk = rec["frame_id"].split("/")[:2]
        by_clip[f"{scene}__{chunk}"].append(rec)
    for clip, recs in sorted(by_clip.items()):
        with open(out / f"det_{clip}.jsonl", "w") as fp:
            for rec in sorted(recs, key=lambda r: r["frame_id"]):
                fp.write(json.dumps(rec) + "\n")
    print(json.dumps(stats), file=sys.stderr)
    print(f"wrote {len(by_clip)} files / {len(merged)} frames -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
