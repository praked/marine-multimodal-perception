"""Run a (fine-tuned) YOLO model on a triplet's UNDISTORTED fisheye frames and
write detections in the dashboard schema, for scoring with score_detections.

The model was trained on undistorted frames, so we undistort here too. With
--mapping (the eval_smoke36 mapping), frame_ids are emitted as the ORIGINAL
institutionone ids, which is what score_detections expects to remap onto the eval clip.

    python -m scripts.eval.yolo_infer \
        --model results/yolo_ft50/weights/best.pt \
        --triplet data/captures/eval_smoke36/2026-06-17_18-00-00 \
        --mapping labels/eval_smoke36_mapping.json \
        --out results/yolo_ft50/eval_pred.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline, iterate_triplet
from scripts.utils.datasets import resolve_triplet
from scripts.eval.dashboard import _frame_id_for


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--triplet", required=True, help="data/<scene>/<timestamp>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mapping", default=None,
                    help="eval mapping JSON; emits ORIGINAL frame_ids for scoring")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=864)
    args = ap.parse_args(argv)

    model = YOLO(args.model)
    names = model.names  # {idx: class_name}
    intr, det = load_intrinsics(), load_detection()
    pl = ObstacleDetectionPipeline(intr, det)
    fake_to_original = (json.load(open(args.mapping)).get("fake_to_original", {})
                        if args.mapping else {})

    trip = resolve_triplet(args.triplet)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_frames = n_det = 0
    with open(args.out, "w") as fp:
        for ts, fish, _therm, _mm in iterate_triplet(trip, det):
            und = pl.process_fisheye(fish).undistorted
            h, w = und.shape[:2]
            res = model.predict(und, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            fid = _frame_id_for(trip.scene, trip.timestamp, ts)
            fid = fake_to_original.get(fid, fid)   # fake -> original for the harness
            bboxes = []
            for b in res.boxes:
                x0, y0, x1, y1 = b.xyxy[0].tolist()
                bboxes.append({
                    "cls": names[int(b.cls)],
                    "xyxy": [x0 / w, y0 / h, x1 / w, y1 / h],
                    "confidence": round(float(b.conf), 4),
                })
            fp.write(json.dumps({"frame_id": fid, "fisheye_bboxes": bboxes}) + "\n")
            n_frames += 1
            n_det += len(bboxes)
    print(f"wrote {n_det} detections over {n_frames} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
