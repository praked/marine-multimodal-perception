"""GroundingDINO pseudo-labelling over the dashboard bundle tree (GPU node).

Reuses the tuned detector recipe (configs/detector_labeler.yaml, F1 0.702:
multi-pass prompts + NMS + structure-under-boat suppression) from
scripts.eval.detector_labeler, but iterates bundle frames
(<scene>__<triplet_ts>/frames/ts=*.jpg, undistorted 432x324 upscaled to
864x648) instead of raw mp4 triplets, and attributes each record to its REAL
capture chunk via meta.json. Output: one labels JSONL per activity in the
labels/qwen schema (det_<scene>__<ts>.jsonl), resumable per frame via the
same cache-dir mechanism.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.detector_labeler import (  # noqa: E402
    GroundingDINO,
    detections_to_record,
    label_batch,
    load_config,
    suppress_structure_under_boat,
)
from scripts.utils.calibration import load_detection, load_intrinsics  # noqa: E402

NATIVE = (864, 648)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    backend = GroundingDINO(cfg["model"], cfg.get("image_shortest_edge"),
                            cfg.get("image_longest_edge"))
    passes = cfg["passes"]
    detection = load_detection()
    fusion_params = detection["fusion"]
    intr_fish = load_intrinsics()["fisheye"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(d for d in Path(args.bundle_root).iterdir()
                   if d.is_dir() and ((d / "meta.json").exists()
                                      or any(d.glob("ts=*.jpg"))))
    if args.only:
        clips = [c for c in clips if args.only in c.name]

    t0 = time.time()
    for ci, clip in enumerate(clips):
        if (clip / "meta.json").exists():
            meta = json.loads((clip / "meta.json").read_text())
            scene = meta["scene"]
            first_chunk = meta["chunks"][0]
            chunk_of = {f["ts"]: f.get("chunk", first_chunk)
                        for f in meta["frames"]}
            frame_glob = clip / "frames"
        else:  # frames-root mode: <scene>__<triplet_ts>/ts=*.jpg
            scene, _, first_chunk = clip.name.partition("__")
            chunk_of = {}
            frame_glob = clip
        out_path = out_dir / f"det_{clip.name}.jsonl"
        done: set[str] = set()
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                try:
                    done.add(json.loads(line)["frame_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

        frames = sorted(frame_glob.glob("ts=*.jpg"))[::args.every]
        todo = []
        for p in frames:
            ts = p.stem[len("ts="):]
            chunk = chunk_of.get(ts.replace("-", ":", 2), first_chunk)
            frame_id = f"{scene}/{chunk}/{p.stem}"
            if frame_id not in done:
                todo.append((p, chunk, frame_id, ts))
        if not todo:
            print(f"[{ci + 1}/{len(clips)}] {clip.name}: all {len(frames)} done",
                  flush=True)
            continue

        n = 0
        with open(out_path, "a") as fp:
            for start in range(0, len(todo), args.batch_size):
                batch = todo[start:start + args.batch_size]
                imgs = []
                keep = []
                for p, chunk, frame_id, ts in batch:
                    im = cv2.imread(str(p))
                    if im is None:
                        continue
                    imgs.append(cv2.resize(im, NATIVE,
                                           interpolation=cv2.INTER_CUBIC))
                    keep.append((chunk, frame_id, ts))
                if not imgs:
                    continue
                results = label_batch(backend, imgs, passes)
                for (chunk, frame_id, ts), dets, img in zip(keep, results, imgs):
                    fts = ts.replace("-", ":", 2)
                    rec = detections_to_record(
                        frame_id, scene, chunk, fts, img.shape,
                        suppress_structure_under_boat(dets, cfg.get("boat_gate")),
                        cfg["model"], fusion_params, intr_fish)
                    fp.write(json.dumps(rec) + "\n")
                    n += 1
                fp.flush()
        print(f"[{ci + 1}/{len(clips)}] {clip.name}: +{n} labelled "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"DONE in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
