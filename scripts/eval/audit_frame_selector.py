"""Select the best N frames to annotate: an active-learning audit planner.

The plain disagreement queue (fusion_audit_queue) ranks frames one by one;
a good audit SET also needs class balance (rare classes weigh more),
coverage of external conditions (daypart / weather / luminance — a duck
audited in the rain teaches nothing about ducks in the sun), and temporal
diversity (frames within the ±7-frame target dilation are near-duplicate
training signal). This selector greedily maximises a submodular-style
objective:

    gain(f) = w_u * uncertainty(f)              # scorer/label conflict + p~0.5
            + w_c * sum_cls quota_gain(f, cls)  # diminishing per-class returns
            + w_e * condition_gain(f)           # diminishing per condition cell
            - crowding penalty                  # same clip, nearby frames

Inputs are existing artefacts: scored sector JSONLs, labels/qwen, and the
dashboard bundle's enrichment (clip-level weather/daypart/luminance).
Unlabelled frames are candidates too (true-negative confirmation is cheap
and the queue alone can never pick them). Output: a ranked CSV plan plus a
coverage report (per class / condition), ready to drive audit sessions.

    python -m scripts.eval.audit_frame_selector \
        --sectors results/sectors_night \
        --bundle /Volumes/ROS2_SSD/asvproject/dashboard_bundle \
        --n 800 --out results/audit_plan.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.utils.datasets import REPO_ROOT

# Rarity-aware class weights: quota gain is scaled by how far the corpus is
# from "enough of this class" — person/duck/animal dominate by design
# (severity policy) and because they are the scarce classes.
CLASS_WEIGHTS = {"person": 3.0, "duck": 3.0, "animal": 3.0, "buoy": 2.0,
                 "float": 2.0, "other": 1.5, "structure": 1.0, "boat": 1.0}
CLASS_CANON = {"boat_ship": "boat", "row_boats": "boat", "swimmer": "person",
               "paddle_board": "float"}
MIN_SPACING_FRAMES = 8  # ±7-frame dilation => closer frames are near-dupes


def load_labels(paths) -> dict[str, list[str]]:
    """frame_id -> list of canonical classes present."""
    out: dict[str, list[str]] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = r.get("frame_id")
            if not fid:
                continue
            out.setdefault(fid, []).extend(
                CLASS_CANON.get(b.get("cls", ""), b.get("cls", ""))
                for b in r.get("fisheye_bboxes", []))
    return out


def clip_conditions(bundle: Path) -> dict[str, tuple[str, ...]]:
    """clip key -> condition cell labels from the bundle enrichment."""
    out: dict[str, tuple[str, ...]] = {}
    for meta_p in sorted(bundle.glob("*/meta.json")):
        try:
            m = json.loads(meta_p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        e = m.get("enrichment") or {}
        cells = []
        parts = e.get("dayparts") or {}
        if parts:
            cells.append("part:" + max(parts, key=lambda k: parts[k]))
        w = e.get("weather") or {}
        if w.get("cloud_cover_pct") is not None:
            cells.append(f"cloud:{int(w['cloud_cover_pct'] // 20)}")
        if w.get("wind_speed_kmh") is not None:
            cells.append(f"wind:{int(w['wind_speed_kmh'] // 10)}")
        if w.get("precipitation_mm") is not None:
            cells.append("precip:wet" if w["precipitation_mm"] >= 0.1
                         else "precip:dry")
        lum = (e.get("luminance") or {}).get("mean")
        if lum is not None:
            cells.append(f"luma:{int(lum // 85)}")
        # per-CHUNK attribution: activity key covers all its chunks
        for chunk in m.get("chunks", [m.get("triplet_ts")]):
            out[f"{m['scene']}__{chunk}"] = tuple(cells)
    return out


def uncertainty(p: list[float], label_bins: set[int] | None,
                centers: list[float]) -> float:
    """Scorer/label conflict where labels exist; scorer entropy either way."""
    ent = sum(1.0 - abs(2 * pi - 1.0) for pi in p) / len(p)  # peak at p=0.5
    if label_bins is None:
        return 0.6 * ent  # unlabelled: entropy only, slightly discounted
    dis = sum(abs(pi - (1.0 if c in label_bins else 0.0))
              for pi, c in zip(p, centers)) / len(p)
    return dis + 0.4 * ent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sectors", required=True)
    ap.add_argument("--bundle", required=True,
                    help="dashboard bundle root (enrichment source)")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-uncertainty", type=float, default=1.0)
    ap.add_argument("--w-class", type=float, default=1.0)
    ap.add_argument("--w-condition", type=float, default=0.8)
    ap.add_argument("--exclude-scene", action="append",
                    default=["eval_smoke36"])
    ap.add_argument("--ignore-curation", action="store_true",
                    help="plan over deleted sets and cut frames too "
                         "(configs/curation.yaml is honoured by default)")
    args = ap.parse_args(argv)
    from scripts.utils.curation import Curation, load_curation
    curation = Curation.empty() if args.ignore_curation else load_curation()

    label_paths = (args.labels if args.labels
                   else sorted((REPO_ROOT / "labels" / "qwen").glob("*.jsonl")))
    labels = load_labels(label_paths)
    label_bins_by_fid: dict[str, set[int]] = {}
    for p in label_paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("frame_id"):
                label_bins_by_fid.setdefault(r["frame_id"], set()).update(
                    int(b) for b in r.get("obstacle_bins_fisheye", []))
    conditions = clip_conditions(Path(args.bundle))

    # ---- candidates ------------------------------------------------------
    cands = []
    for jf in sorted(Path(args.sectors).glob("*.jsonl")):
        clip = jf.stem
        scene, _, tts = clip.partition("__")
        if any(x in scene for x in args.exclude_scene):
            continue
        if curation.is_deleted(clip):
            continue   # deleted in the dashboard: never planned
        for idx, line in enumerate(jf.read_text().splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            p = r.get("p_obstacle")
            centers = r.get("bin_centers_deg")
            if not p or not centers:
                continue
            ts = r["timestamp"]
            if not curation.keep_frame(clip, ts):
                continue   # inside a cut (idx stays the raw line index)
            fid = f"{scene}/{tts}/ts={ts.replace(':', '-')}"
            lb = label_bins_by_fid.get(fid)
            cands.append({
                "clip": clip, "idx": idx, "frame_ts": ts, "frame_id": fid,
                "u": uncertainty(p, lb, centers),
                "classes": [c for c in labels.get(fid, []) if c],
                "cells": conditions.get(clip, ()),
                "labelled": lb is not None,
            })
    print(f"candidates: {len(cands)} frames "
          f"({sum(c['labelled'] for c in cands)} labelled)")

    # ---- greedy selection with diminishing returns -----------------------
    cls_count: dict[str, int] = defaultdict(int)
    cell_count: dict[str, int] = defaultdict(int)
    picked_idx: dict[str, list[int]] = defaultdict(list)
    chosen = []
    # coarse pre-sort so greedy scans a bounded pool per pick
    cands.sort(key=lambda c: -c["u"])
    pool = cands[: max(args.n * 20, 5000)]
    remaining = pool.copy()
    for _ in range(min(args.n, len(pool))):
        best, best_gain = None, -1e9
        for c in remaining:
            if any(abs(c["idx"] - i) < MIN_SPACING_FRAMES
                   for i in picked_idx[c["clip"]]):
                continue
            g = args.w_uncertainty * c["u"]
            for cls in set(c["classes"]):
                w = CLASS_WEIGHTS.get(cls, 1.0)
                g += args.w_class * w / math.sqrt(1 + cls_count[cls])
            for cell in c["cells"]:
                g += (args.w_condition / len(c["cells"] or (1,))
                      / math.sqrt(1 + cell_count[cell]))
            if not c["labelled"]:
                g += 0.15  # small standing bonus: true negatives are cheap
            if g > best_gain:
                best, best_gain = c, g
        if best is None:
            break
        chosen.append((best, best_gain))
        remaining.remove(best)
        picked_idx[best["clip"]].append(best["idx"])
        for cls in set(best["classes"]):
            cls_count[cls] += 1
        for cell in best["cells"]:
            cell_count[cell] += 1

    if not cands:
        print("no candidates — are the sectors/bundle paths mounted? "
              "(results/ and the bundle live on the SSD)")
        return 1
    out = Path(args.out)
    if not out.parent.is_dir():
        out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "clip", "frame_ts", "frame_id", "gain",
                    "uncertainty", "labelled", "classes", "conditions"])
        for i, (c, g) in enumerate(chosen, 1):
            w.writerow([i, c["clip"], c["frame_ts"], c["frame_id"],
                        round(g, 3), round(c["u"], 3), int(c["labelled"]),
                        "|".join(sorted(set(c["classes"]))),
                        "|".join(c["cells"])])
    print(f"wrote {len(chosen)} frames -> {out}")
    print("class coverage:", dict(sorted(cls_count.items())))
    print("condition cells:", dict(sorted(cell_count.items())))
    per_clip: dict[str, int] = defaultdict(int)
    for c, _ in chosen:
        per_clip[c["clip"]] += 1
    for k, v in sorted(per_clip.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
