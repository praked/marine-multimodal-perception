"""Run a trained YOLO detector over pre-exported undistorted frame dirs and
write per-clip dashboard-schema detection JSONLs (runs ON THE GPU NODE).

Reuses the undistorted frames already on scratch from the segmentation pass
($SEG/pred_frames/<scene>__<triplet_ts>/ts=<HH-MM-SS.f>.jpg), so no mp4s need
re-uploading and no re-undistortion happens. Output, one JSONL per clip:

    <out>/<scene>__<triplet_ts>.jsonl
    {"frame_id": "<scene>/<triplet_ts>/ts=<...>",
     "fisheye_bboxes": [{"cls": "boat_ship", "xyxy": [x0,y0,x1,y1], "confidence": 0.83}]}

`xyxy` is normalised to [0,1] (matching scripts/eval/yolo_infer.py), which the
dashboard DetProvider consumes.

    python yolo_predict_frames.py --model .../best.pt \
        --frames-root /scratch0/$USER/asvproject_seg/pred_frames \
        --out /scratch0/$USER/asvproject_yolo/det --conf 0.25 --imgsz 960
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png")


def _frame_id_from(clip_dir_name: str, stem: str) -> str:
    # "<scene>__<triplet_ts>" + stem "ts=..." -> "<scene>/<triplet_ts>/ts=..."
    scene, _, triplet_ts = clip_dir_name.partition("__")
    return f"{scene}/{triplet_ts}/{stem}"


def predict_clip(model, clip_dir: Path, out_dir: Path, conf: float,
                 imgsz: int, batch: int) -> tuple[int, int]:
    frames = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not frames:
        return 0, 0
    names = model.names
    out_path = out_dir / f"{clip_dir.name}.jsonl"
    n_frames = n_det = 0
    with open(out_path, "w") as fp:
        for start in range(0, len(frames), batch):
            chunk = frames[start:start + batch]
            results = model.predict([str(p) for p in chunk], conf=conf,
                                    imgsz=imgsz, verbose=False)
            for p, res in zip(chunk, results):
                hh, ww = res.orig_shape
                bboxes = []
                for b in res.boxes:
                    x0, y0, x1, y1 = b.xyxy[0].tolist()
                    bboxes.append({
                        "cls": names[int(b.cls)],
                        "xyxy": [x0 / ww, y0 / hh, x1 / ww, y1 / hh],
                        "confidence": round(float(b.conf), 4),
                    })
                fp.write(json.dumps({
                    "frame_id": _frame_id_from(clip_dir.name, p.stem),
                    "fisheye_bboxes": bboxes,
                }) + "\n")
                n_frames += 1
                n_det += len(bboxes)
    print(f"  {clip_dir.name}: {n_det} dets over {n_frames} frames -> {out_path}")
    return n_frames, n_det


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames-root", required=True, help="dir of <clip>/ frame dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args(argv)

    model = YOLO(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_dirs = sorted(d for d in Path(args.frames_root).iterdir() if d.is_dir())
    print(f"{len(clip_dirs)} clips -> {out_dir}")
    tot_f = tot_d = 0
    for d in clip_dirs:
        nf, nd = predict_clip(model, d, out_dir, args.conf, args.imgsz, args.batch)
        tot_f += nf
        tot_d += nd
    print(f"Done. {tot_d} detections over {tot_f} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
