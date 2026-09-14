"""Convert dashboard label JSONL(s) into a YOLO detection dataset.

Extracts the UNDISTORTED fisheye frames (the coordinate space the dashboard
labels live in) for each labelled frame_id, writes YOLO-format labels
(class cx cy w h, normalised), a train/val split, and data.yaml. Reusable as
the training set grows: re-run after each labelling session.

Optional semi-supervised mix: blend GroundingDINO pseudo-labels (--pseudo) with
the trusted human labels, DOWN-WEIGHTING the pseudo by oversampling the human
frames (--human-repeat K, so each human frame is seen K x per epoch). The
pseudo pool excludes the human frames and (via --exclude) the held-out eval
frames, so there is no leakage. Val is always the 1-in-N human split only.

    # human-only
    python -m scripts.eval.export_yolo --labels labels/training_frames.jsonl --out data/yolo_finetune
    # mixed: 50 human (weight 5) + 450 pseudo (weight 1)
    python -m scripts.eval.export_yolo --labels labels/training_frames.jsonl \
        --pseudo labels/qwen/det_2026-06-17_institutionone_day1.jsonl --pseudo-count 450 \
        --exclude labels/eval_smoke36_mapping.json --human-repeat 5 --out data/yolo_mixed
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

import cv2

from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.cv_common import undistort_fisheye
from scripts.sensor_processing.pipeline import iterate_triplet
from scripts.utils.datasets import resolve_triplet
from scripts.eval.dashboard import _frame_id_for

CAPTURES_ROOT = Path("data/captures")


def _xyxy_to_cxcywh(b):
    x0, y0, x1, y1 = b
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0), (y1 - y0))


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="labels/training_frames.jsonl",
                    help="trusted human labels")
    ap.add_argument("--out", default="data/yolo_finetune")
    ap.add_argument("--val-every", type=int, default=5,
                    help="every Nth human frame -> val (default 5). Spread, deterministic.")
    ap.add_argument("--classes", default=None,
                    help="comma-separated class order; default = classes present in human, sorted")
    ap.add_argument("--pseudo", default=None, help="GroundingDINO pseudo-label JSONL to mix in")
    ap.add_argument("--pseudo-count", type=int, default=0, help="how many pseudo frames to add")
    ap.add_argument("--human-repeat", type=int, default=1,
                    help="oversample each human TRAIN frame this many times (down-weights pseudo)")
    ap.add_argument("--exclude", default=None,
                    help="eval mapping JSON; its original frame_ids are kept out of the pseudo pool")
    args = ap.parse_args(argv)

    human = [r for r in _load(args.labels) if r.get("fisheye_bboxes")]
    if not human:
        raise SystemExit(f"no labelled frames in {args.labels}")
    present = collections.Counter(b["cls"] for r in human for b in r["fisheye_bboxes"])
    classes = args.classes.split(",") if args.classes else sorted(present)
    cidx = {c: i for i, c in enumerate(classes)}

    # Pseudo pool: exclude human frames + (optionally) the held-out eval frames.
    exclude = {r["frame_id"] for r in human}
    if args.exclude:
        m = json.load(open(args.exclude))
        exclude |= set(m.get("fake_to_original", {}).values())
    pseudo = []
    if args.pseudo and args.pseudo_count > 0:
        cand = [r for r in _load(args.pseudo)
                if r.get("fisheye_bboxes") and r["frame_id"] not in exclude
                and any(b["cls"] in cidx for b in r["fisheye_bboxes"])]
        stride = max(1, len(cand) // args.pseudo_count)   # spread across the mission
        pseudo = cand[::stride][:args.pseudo_count]

    print(f"human {len(human)} (classes {classes}, boxes {dict(present)}) | "
          f"pseudo {len(pseudo)} (repeat human x{args.human_repeat})")

    # Role per record: (split, n_copies). Human 1-in-N val; human-train oversampled; pseudo train x1.
    stride_v = max(2, args.val_every)
    items = []
    for i, r in enumerate(human):
        items.append((r, "val", 1) if i % stride_v == 0 else (r, "train", args.human_repeat))
    items += [(r, "train", 1) for r in pseudo]

    out = Path(args.out)
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    by_clip: dict[tuple, dict] = collections.defaultdict(dict)
    for rec, split, rep in items:
        scene, tts, _ = rec["frame_id"].split("/", 2)
        by_clip[(scene, tts)][rec["frame_id"]] = (split, rep, rec)

    intr, det = load_intrinsics(), load_detection()
    K, D = intr["fisheye"]["K"], intr["fisheye"]["D"]
    n_imgs = collections.Counter()
    for (scene, tts), wanted in by_clip.items():
        # nested missions flatten their scene name to <mission>_<sub>
        # (2026-08-19_afloat_imu_rock -> 2026-08-19_afloat/imu_rock): find
        # the underscore boundary that names a real capture directory.
        base = CAPTURES_ROOT / scene
        if not base.is_dir():
            parts = scene.split("_")
            for i in range(len(parts) - 1, 0, -1):
                cand = CAPTURES_ROOT / "_".join(parts[:i]) / "_".join(parts[i:])
                if cand.is_dir():
                    base = cand
                    break
        # only the fisheye is read: dead-thermal chunks (08-19) must not abort
        trip = resolve_triplet(str(base / tts), require=("fisheye",))
        for ts, fish, _therm, _mm in iterate_triplet(trip, det):
            fid = _frame_id_for(scene, tts, ts)
            if fid not in wanted:
                continue
            split, rep, rec = wanted[fid]
            und = undistort_fisheye(fish, K, D)   # identical to the dashboard's undistort
            h, w = und.shape[:2]
            lines = []
            for b in rec["fisheye_bboxes"]:
                if b["cls"] not in cidx:
                    continue
                cx, cy, bw, bh = _xyxy_to_cxcywh(b["xyxy"])
                lines.append(f"{cidx[b['cls']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stem = f"{tts}_{ts.replace(':', '-')}"
            for k in range(rep):
                s = stem if rep == 1 else f"{stem}_r{k}"
                cv2.imwrite(str(out / f"images/{split}/{s}.jpg"), und)
                (out / f"labels/{split}/{s}.txt").write_text("\n".join(lines) + "\n")
                n_imgs[split] += 1

    (out / "data.yaml").write_text(
        f"train: images/train\nval: images/val\nnc: {len(classes)}\nnames: {classes}\n")
    print(f"wrote train={n_imgs['train']} (human x{args.human_repeat} + pseudo) / val={n_imgs['val']} images")
    print(f"dataset -> {out}/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
