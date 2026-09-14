"""Fine grid search over GroundingDINO detector parameters.

Runs on the GPU node. Sweeps the objects threshold x structure threshold x
boat_gate (configs/detector_gridsearch.yaml) over the fixed smoke set;
resolution + exposure are fixed at the known-best. For each config it writes a
labels JSONL + a contact-sheet PNG + a summary row. The model forward is reused
across boat_gate values (gating is post-process), so the gate sweep is free.

There is no ground truth on the node, so judge by eye from the contact sheets;
the summary gives per-class counts. Pull with pull_gridsearch.sh, then score
locally with scripts/eval/score_detections.py --grid-dir.

    python -m scripts.eval.detector_gridsearch \\
        --captures-dir /scratch0/$USER/asvproject/data/captures/2026-06-17_institutionone_day1 \\
        --frames-file labels/qwen/smoke_set.txt \\
        --grid configs/detector_gridsearch.yaml \\
        --out-dir /scratch0/$USER/asvproject/out/grid
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import math
import time
from pathlib import Path

import cv2

from scripts.eval.detector_labeler import (
    GroundingDINO,
    apply_exposure,
    detections_to_record,
    label_batch,
    suppress_structure_under_boat,
)
from scripts.eval.qwen_batch_labeler import iter_clip_frames, list_clip_timestamps
from scripts.utils.calibration import load_detection, load_intrinsics

CLASS_COLORS = {
    "boat": (0, 200, 0), "duck": (0, 200, 255), "buoy": (0, 0, 230),
    "person": (255, 80, 0), "structure": (0, 165, 255), "other": (160, 0, 160),
}


def _norm(d):
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in (d or {}).items()}


def _draw(img, rec):
    out = img.copy()
    h, w = out.shape[:2]
    for b in rec["fisheye_bboxes"]:
        x0, y0, x1, y1 = b["xyxy"]
        cv2.rectangle(out, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                      CLASS_COLORS.get(b["cls"], (255, 255, 255)), 2)
    return out


def _contact_sheet(thumbs, path, cols=6):
    if not thumbs:
        return
    blank = thumbs[0] * 0
    rows = []
    for ri in range(math.ceil(len(thumbs) / cols)):
        row = thumbs[ri * cols:ri * cols + cols]
        row = row + [blank] * (cols - len(row))
        rows.append(cv2.hconcat(row))
    cv2.imwrite(str(path), cv2.vconcat(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures-dir", required=True)
    ap.add_argument("--frames-file", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=6)
    args = ap.parse_args()

    import yaml
    spec = yaml.safe_load(Path(args.grid).read_text())
    model_id = spec.get("model", "IDEA-Research/grounding-dino-base")
    se = spec.get("image_shortest_edge")
    exposure = float(spec.get("exposure", 1.0))
    obj_q = _norm(spec["objects_queries"])
    obj_tt = float(spec.get("objects_text_threshold", 0.30))
    struct_q = _norm(spec["structure_queries"])
    large_q = _norm(spec.get("structures_large_queries", {}))
    g = spec["grid"]

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    captures_dir = Path(args.captures_dir).expanduser().resolve()
    frames_filter = {ln.split("#", 1)[0].strip()
                     for ln in Path(args.frames_file).read_text().splitlines()
                     if ln.split("#", 1)[0].strip()}
    intrinsics = load_intrinsics()
    detection = load_detection()
    fusion_params = detection["fusion"]
    intr_fish = intrinsics["fisheye"]

    # Load the smoke-set frames once, brightened to the fixed exposure.
    want_clips = {fid.split("/")[1] for fid in frames_filter if "/" in fid}
    clips = [c for c in list_clip_timestamps(captures_dir, None, None) if c in want_clips]
    frames = []  # (frame_id, scene, tts, fts, img)
    for clip_ts in clips:
        try:
            for fid, scene, tts, fts, img in iter_clip_frames(
                    captures_dir, clip_ts, 1, detection, intrinsics):
                if fid in frames_filter:
                    frames.append((fid, scene, tts, fts, apply_exposure(img, exposure)))
        except Exception as e:  # noqa: BLE001
            print(f"!! clip {clip_ts}: {e}", flush=True)
    print(f"loaded {len(frames)} smoke frames (exposure {exposure}x)", flush=True)

    summary_path = out_dir / "summary.csv"
    if not summary_path.exists():
        with open(summary_path, "w", newline="") as sf:
            csv.writer(sf).writerow(
                ["tag", "objects_bt", "struct_bt", "struct_tt", "boat_gate",
                 "total", "boat", "structure", "person", "buoy", "duck",
                 "frames_with_det", "seconds"])

    backend = GroundingDINO(model_id, se)
    gates = g["boat_gate"]
    t_all = time.time()
    combos = list(itertools.product(
        g["objects_box_threshold"], g["struct_box_threshold"], g["struct_text_threshold"]))
    n_total = len(combos) * len(gates)
    i = 0
    for obj_bt, struct_bt, struct_tt in combos:
        passes = [
            {"name": "objects", "box_threshold": obj_bt, "text_threshold": obj_tt,
             "queries": obj_q},
            {"name": "structure", "box_threshold": struct_bt,
             "text_threshold": struct_tt, "queries": struct_q},
        ]
        if large_q:
            passes.append({"name": "structures_large", "box_threshold": struct_bt,
                           "text_threshold": struct_tt, "queries": large_q})
        # One forward set for this (obj_bt, struct_bt); reused for every gate.
        t0 = time.time()
        merged = []
        for b0 in range(0, len(frames), args.batch_size):
            batch = frames[b0:b0 + args.batch_size]
            merged.extend(label_batch(backend, [f[4] for f in batch], passes))
        fwd_dt = time.time() - t0

        for gate in gates:
            i += 1
            tag = (f"obt{obj_bt}_sbt{struct_bt}_gate{gate}").replace(".", "")
            cfg_dir = out_dir / tag
            if (cfg_dir / "contact.png").exists():
                print(f"[{i}/{n_total}] {tag}: skip (done)", flush=True)
                continue
            cfg_dir.mkdir(parents=True, exist_ok=True)
            counts = collections.Counter()
            thumbs = []
            n_det = 0
            with open(cfg_dir / "labels.jsonl", "w") as lf:
                for (fid, scene, tts, fts, img), dets in zip(frames, merged):
                    kept = suppress_structure_under_boat(dets, None if gate >= 1.0 else gate)
                    rec = detections_to_record(fid, scene, tts, fts, img.shape,
                                               kept, model_id, fusion_params, intr_fish)
                    lf.write(json.dumps(rec) + "\n")
                    for b in rec["fisheye_bboxes"]:
                        counts[b["cls"]] += 1
                    n_det += 1 if rec["fisheye_bboxes"] else 0
                    th = _draw(img, rec)
                    cv2.putText(th, f"{tag} [{len(rec['fisheye_bboxes'])}]", (5, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    thumbs.append(cv2.resize(th, (288, 216)))
            _contact_sheet(thumbs, cfg_dir / "contact.png")
            with open(summary_path, "a", newline="") as sf:
                csv.writer(sf).writerow(
                    [tag, obj_bt, struct_bt, struct_tt, gate, sum(counts.values()),
                     counts["boat"], counts["structure"], counts["person"],
                     counts["buoy"], counts["duck"], n_det, round(fwd_dt, 1)])
            print(f"[{i}/{n_total}] {tag}: total={sum(counts.values())} "
                  f"struct={counts['structure']} boat={counts['boat']} "
                  f"person={counts['person']}", flush=True)

    print(f"\ngrid done in {time.time() - t_all:.0f}s -> {out_dir}\n"
          f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
