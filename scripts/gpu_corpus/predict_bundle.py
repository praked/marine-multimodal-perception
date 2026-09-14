"""Full-corpus vision inference over the dashboard bundle tree (on the GPU node).

Consumes the R2 bundle layout (``<scene>__<triplet_ts>/frames/ts=*.jpg``,
undistorted fisheye at half res 432x324, plus ``meta.json`` whose per-frame
``chunk`` field attributes concatenated-activity frames to their real capture
chunk) and emits repo-format artefacts keyed by the REAL chunk triplet:

    <out>/seg/<scene>__<chunk_ts>/ts=<...>.png     class maps (obstacle0/water1/sky2), 864x648
    <out>/det/<scene>__<chunk_ts>.jsonl            {"frame_id", "fisheye_bboxes":[{cls,xyxy,confidence}]}
    <out>/det_seg/<scene>__<chunk_ts>.jsonl        {"frame_id", "instances":[{cls,confidence,polygon}]}

Frames are upscaled 2x back to 864x648 before inference so every model sees its
native training scale; masks/coords are emitted at canonical geometry. This is
the dashboard-grade pass -- rerun from raw mp4s (export_undistorted_frames) for
the archival full-resolution pass.

Resumable: clips whose outputs exist are skipped (--force to redo).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

NATIVE = (864, 648)


def load_student(pth: str, device):
    import torch
    from scripts.gpu_seg.train_student import build_student

    model = build_student(pretrained=False)
    state = torch.load(pth, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.to(device).eval()
    return model


def student_masks(model, imgs_bgr: list[np.ndarray], device) -> list[np.ndarray]:
    import torch

    batch = np.stack([
        cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for im in imgs_bgr
    ])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = (batch - mean) / std
    t = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
    with torch.no_grad():
        out = model(t)["out"]
    return [m.astype(np.uint8) for m in out.argmax(1).cpu().numpy()]


def frame_chunk_map(meta: dict) -> dict[str, str]:
    """ts -> real chunk triplet_ts (first chunk's frames carry no 'chunk')."""
    first = meta["chunks"][0]
    return {f["ts"]: f.get("chunk", first) for f in meta["frames"]}


def process_clip(clip_dir: Path, out_root: Path, student, yolo_det, yolo_seg,
                 device, batch: int, force: bool) -> dict:
    if (clip_dir / "meta.json").exists():
        meta = json.loads((clip_dir / "meta.json").read_text())
        scene = meta["scene"]
        chunk_of = frame_chunk_map(meta)
        frames = sorted((clip_dir / "frames").glob("ts=*.jpg"))
    else:
        # frames-root mode: <scene>__<triplet_ts>/ts=*.jpg (native export)
        scene, _, tts = clip_dir.name.partition("__")
        meta = {"chunks": [tts]}
        chunk_of = {}
        frames = sorted(clip_dir.glob("ts=*.jpg"))
    stats = {"frames": 0, "det": 0, "inst": 0, "seg": 0, "skipped": 0}

    det_fp: dict[str, object] = {}
    seg_fp: dict[str, object] = {}

    def sink(kind: str, chunk: str):
        cache = det_fp if kind == "det" else seg_fp
        if chunk not in cache:
            path = out_root / kind / f"{scene}__{chunk}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            cache[chunk] = open(path, "w")  # a chunk belongs to exactly one activity
        return cache[chunk]

    done_marker = out_root / "done" / f"{clip_dir.name}.json"
    if done_marker.exists() and not force:
        stats["skipped"] = len(frames)
        return stats

    for start in range(0, len(frames), batch):
        chunk_paths = frames[start:start + batch]
        imgs = []
        keep = []
        for p in chunk_paths:
            im = cv2.imread(str(p))
            if im is None:
                continue
            if (im.shape[1], im.shape[0]) != NATIVE:
                im = cv2.resize(im, NATIVE, interpolation=cv2.INTER_CUBIC)
            imgs.append(im)
            keep.append(p)
        if not imgs:
            continue

        masks = student_masks(student, imgs, device) if student else [None] * len(imgs)
        det_res = yolo_det.predict(imgs, conf=0.25, imgsz=960, verbose=False) if yolo_det else [None] * len(imgs)
        seg_res = yolo_seg.predict(imgs, conf=0.25, imgsz=960, verbose=False) if yolo_seg else [None] * len(imgs)

        for p, mask, dres, sres in zip(keep, masks, det_res, seg_res):
            ts = p.stem[len("ts="):]
            # meta ts uses colons, filenames dashes -- normalise for the lookup
            chunk = chunk_of.get(ts.replace("-", ":", 2), meta["chunks"][0])
            frame_id = f"{scene}/{chunk}/{p.stem}"
            stats["frames"] += 1

            if mask is not None:
                seg_dir = out_root / "seg" / f"{scene}__{chunk}"
                seg_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(seg_dir / f"{p.stem}.png"), mask)
                stats["seg"] += 1

            if dres is not None:
                hh, ww = dres.orig_shape
                bboxes = []
                for b in dres.boxes:
                    x0, y0, x1, y1 = b.xyxy[0].tolist()
                    bboxes.append({
                        "cls": yolo_det.names[int(b.cls)],
                        "xyxy": [round(x0 / ww, 4), round(y0 / hh, 4),
                                 round(x1 / ww, 4), round(y1 / hh, 4)],
                        "confidence": round(float(b.conf), 4),
                    })
                sink("det", chunk).write(json.dumps(
                    {"frame_id": frame_id, "fisheye_bboxes": bboxes}) + "\n")
                stats["det"] += len(bboxes)

            if sres is not None:
                hh, ww = sres.orig_shape
                instances = []
                if sres.masks is not None:
                    for c, cf, poly in zip(sres.boxes.cls, sres.boxes.conf,
                                           sres.masks.xy):
                        flat = [round(float(v) / (ww if i % 2 == 0 else hh), 4)
                                for i, v in enumerate(np.asarray(poly).reshape(-1))]
                        if len(flat) < 6:
                            continue
                        instances.append({
                            "cls": yolo_seg.names[int(c)],
                            "confidence": round(float(cf), 4),
                            "polygon": flat,
                        })
                sink("det_seg", chunk).write(json.dumps(
                    {"frame_id": frame_id, "instances": instances}) + "\n")
                stats["inst"] += len(instances)

    for fp in list(det_fp.values()) + list(seg_fp.values()):
        fp.close()
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(json.dumps(stats))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--student-pth", default=None)
    ap.add_argument("--yolo-det", default=None)
    ap.add_argument("--yolo-seg", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--only", default=None, help="substring filter on clip dirs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    student = load_student(args.student_pth, device) if args.student_pth else None
    yolo_det = yolo_seg = None
    if args.yolo_det or args.yolo_seg:
        from ultralytics import YOLO
        if args.yolo_det:
            yolo_det = YOLO(args.yolo_det)
        if args.yolo_seg:
            yolo_seg = YOLO(args.yolo_seg)

    out_root = Path(args.out_root)
    clips = sorted(d for d in Path(args.bundle_root).iterdir()
                   if d.is_dir() and ((d / "meta.json").exists()
                                      or any(d.glob("ts=*.jpg"))))
    if args.only:
        clips = [c for c in clips if args.only in c.name]
    t0 = time.time()
    total = {"frames": 0, "det": 0, "inst": 0, "seg": 0, "skipped": 0}
    for i, clip in enumerate(clips):
        st = process_clip(clip, out_root, student, yolo_det, yolo_seg,
                          device, args.batch, args.force)
        for k in total:
            total[k] += st[k]
        el = time.time() - t0
        print(f"[{i + 1}/{len(clips)}] {clip.name}: {st}  ({el:.0f}s elapsed)",
              flush=True)
    print(f"TOTAL {total} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
