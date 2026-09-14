"""Score detector output against the hand-annotated eval clip.

Closes the loop started by make_eval_clip.py: you annotate the eval clip once
in the dashboard (ground truth), then this scores any detector run against it,
no per-config visual check needed. It remaps the detector's ORIGINAL frame_ids
onto the eval-clip frame_ids via the mapping, IoU-matches boxes per frame, and
reports precision / recall / F1, both per class and class-agnostic (any box on
any obstacle: robust to the boat-vs-structure labelling nuance).

    # score one detector run:
    python -m scripts.eval.score_detections \\
        --truth labels/manual.jsonl \\
        --pred results/grid/e12_se1000_bt02_tt018/labels.jsonl \\
        --mapping labels/eval_smoke36_mapping.json

    # rank a whole grid directory by class-agnostic F1:
    python -m scripts.eval.score_detections \\
        --truth labels/manual.jsonl \\
        --grid-dir results/grid \\
        --mapping labels/eval_smoke36_mapping.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter)


def _load(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["frame_id"]] = r.get("fisheye_bboxes", [])
    return out


def _match(preds, truths, iou_thr, agnostic):
    """Greedy IoU match within one frame. Returns (tp, fp, fn) per class."""
    from collections import Counter
    tp, fp, fn = Counter(), Counter(), Counter()
    classes = set(b["cls"] for b in preds) | set(b["cls"] for b in truths)
    if agnostic:
        classes = {"obstacle"}

        def cof(b):
            return "obstacle"
    else:
        def cof(b):
            return b["cls"]
    for cls in classes:
        P = sorted([b for b in preds if cof(b) == cls],
                   key=lambda b: -b.get("confidence", 0.0))
        T = [b for b in truths if cof(b) == cls]
        used = [False] * len(T)
        for p in P:
            best, bj = iou_thr, -1
            for j, t in enumerate(T):
                if used[j]:
                    continue
                v = _iou(p["xyxy"], t["xyxy"])
                if v >= best:
                    best, bj = v, j
            if bj >= 0:
                used[bj] = True
                tp[cls] += 1
            else:
                fp[cls] += 1
        fn[cls] += used.count(False)
    return tp, fp, fn


def score(truth, pred, iou_thr=0.5):
    """truth/pred: {frame_id: [bbox,...]} on the SAME id space. Returns dict."""
    from collections import Counter
    frames = set(truth)  # score only annotated frames
    res = {}
    for agnostic, key in ((False, "per_class"), (True, "agnostic")):
        TP, FP, FN = Counter(), Counter(), Counter()
        for fid in frames:
            tp, fp, fn = _match(pred.get(fid, []), truth[fid], iou_thr, agnostic)
            TP.update(tp); FP.update(fp); FN.update(fn)
        rows = {}
        for cls in set(TP) | set(FP) | set(FN):
            t, f, m = TP[cls], FP[cls], FN[cls]
            prec = t / (t + f) if t + f else 0.0
            rec = t / (t + m) if t + m else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows[cls] = {"tp": t, "fp": f, "fn": m, "precision": round(prec, 3),
                         "recall": round(rec, 3), "f1": round(f1, 3)}
        res[key] = rows
    return res


def _remap_pred(pred, original_to_fake):
    """Detector frame_ids are ORIGINAL; map them onto the eval-clip ids."""
    out = {}
    for fid, boxes in pred.items():
        out[original_to_fake.get(fid, fid)] = boxes
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="Ground-truth labels JSONL "
                    "(eval-clip frame_ids), e.g. labels/manual.jsonl.")
    ap.add_argument("--pred", help="One detector output JSONL.")
    ap.add_argument("--grid-dir", help="Directory of <tag>/labels.jsonl to rank.")
    ap.add_argument("--mapping", required=True,
                    help="eval_smoke36_mapping.json (remaps pred ids).")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--scene", default=None,
                    help="Restrict truth to this scene (eval mission name).")
    args = ap.parse_args()

    mapping = json.loads(Path(args.mapping).read_text())
    o2f = mapping["original_to_fake"]
    eval_ids = set(mapping["fake_to_original"])

    truth = _load(args.truth)
    # Keep only the annotated eval-clip frames.
    truth = {k: v for k, v in truth.items()
             if k in eval_ids and (args.scene is None or k.startswith(args.scene + "/"))}
    if not truth:
        raise SystemExit("No ground-truth labels on the eval clip yet: annotate "
                         "data/captures/<mission> in the dashboard first.")
    print(f"ground truth: {len(truth)} annotated frames, "
          f"{sum(len(v) for v in truth.values())} boxes", flush=True)

    def one(pred_path):
        pred = _remap_pred(_load(pred_path), o2f)
        return score(truth, pred, args.iou)

    if args.grid_dir:
        rows = []
        for d in sorted(Path(args.grid_dir).iterdir()):
            lp = d / "labels.jsonl"
            if not lp.exists():
                continue
            r = one(lp)
            ag = r["agnostic"].get("obstacle", {})
            rows.append((d.name, ag.get("precision", 0), ag.get("recall", 0),
                         ag.get("f1", 0)))
        rows.sort(key=lambda x: -x[3])
        print(f"\nranked by class-agnostic F1 @IoU{args.iou}:")
        print(f"  {'config':32s} {'prec':>6s} {'rec':>6s} {'F1':>6s}")
        for name, p, r, f in rows:
            print(f"  {name:32s} {p:6.3f} {r:6.3f} {f:6.3f}")
    elif args.pred:
        r = one(args.pred)
        print(f"\n@IoU{args.iou}")
        for key in ("agnostic", "per_class"):
            print(f"\n[{key}]")
            for cls, m in sorted(r[key].items()):
                print(f"  {cls:10s} P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"F1={m['f1']:.3f}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    else:
        raise SystemExit("pass --pred or --grid-dir")


if __name__ == "__main__":
    main()
