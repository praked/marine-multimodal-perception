"""Run a trained YOLOv8-seg model over pre-exported undistorted frame dirs and
write per-clip instance-mask JSONLs (runs ON THE GPU NODE).

Mirrors yolo_predict_frames.py but emits per-object POLYGONS (the panoptic-style
overlay the dashboard's InstanceSegProvider consumes). One JSONL per clip:

    <out>/<scene>__<triplet_ts>.jsonl
    {"frame_id": "<scene>/<triplet_ts>/ts=<...>",
     "instances": [{"cls": "boat_ship", "confidence": 0.8,
                    "polygon": [x1,y1,x2,y2,...]}]}   # normalised [0,1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png")


def _frame_id_from(clip_dir_name: str, stem: str) -> str:
    scene, _, triplet_ts = clip_dir_name.partition("__")
    return f"{scene}/{triplet_ts}/{stem}"


def predict_clip(model, clip_dir: Path, out_dir: Path, conf, imgsz, batch):
    frames = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not frames:
        return 0, 0
    names = model.names
    out_path = out_dir / f"{clip_dir.name}.jsonl"
    n_frames = n_obj = 0
    with open(out_path, "w") as fp:
        for start in range(0, len(frames), batch):
            chunk = frames[start:start + batch]
            results = model.predict([str(p) for p in chunk], conf=conf,
                                    imgsz=imgsz, verbose=False)
            for p, res in zip(chunk, results):
                instances = []
                if res.masks is not None:
                    polys = res.masks.xyn  # list of (N,2) normalised arrays
                    cls = res.boxes.cls.tolist()
                    cfs = res.boxes.conf.tolist()
                    for poly, c, cf in zip(polys, cls, cfs):
                        flat = [round(float(v), 5) for v in poly.reshape(-1)]
                        if len(flat) < 6:
                            continue
                        instances.append({
                            "cls": names[int(c)],
                            "confidence": round(float(cf), 4),
                            "polygon": flat,
                        })
                fp.write(json.dumps({
                    "frame_id": _frame_id_from(clip_dir.name, p.stem),
                    "instances": instances,
                }) + "\n")
                n_frames += 1
                n_obj += len(instances)
    print(f"  {clip_dir.name}: {n_obj} instances over {n_frames} frames -> {out_path}")
    return n_frames, n_obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames-root", required=True)
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
    tf = to = 0
    for d in clip_dirs:
        nf, no = predict_clip(model, d, out_dir, args.conf, args.imgsz, args.batch)
        tf += nf
        to += no
    print(f"Done. {to} instances over {tf} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
