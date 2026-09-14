"""GPU-node labeller: DART (SAM3 multi-class, github.com/mkturkcan/DART) over a
frames-root (<scene>__<triplet_ts>/ts=*.jpg) or a dashboard-bundle tree.

The open-vocabulary TEACHER CANDIDATE to compare against the GroundingDINO
recipe (label_bundle_dino.py). Same output contract — one
det_<clip>.jsonl per clip, records built by
scripts.eval.detector_labeler.detections_to_record — so metrics.py, the audit
planner and the dashboard read both teachers alike; `source` tells them apart.

    python scripts/gpu_corpus/label_bundle_dart.py --bundle-root $S/native_frames \
        --out-dir $S/corpus_out/labels_dart --only 2026-08-26_afloat --every 1

Config: configs/dart_labeler.yaml (prompt -> canonical class, thresholds,
imgsz). Needs the DART repo importable (pip install -e .) and its SAM3
checkpoint (auto-downloaded via huggingface_hub; the repo is gated — accept
the SAM licence on HF and have a token in HF_TOKEN / hf auth login).
Resumable per clip (records already in the output file are skipped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.detector_labeler import (  # noqa: E402
    detections_to_record,
    suppress_structure_under_boat,
)
from scripts.utils.calibration import load_detection, load_intrinsics  # noqa: E402

NATIVE = (864, 648)
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dart_labeler.yaml"


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    return yaml.safe_load(p.read_text())


class DartBackend:
    """Thin wrapper over DART's fast multi-class predictor."""

    def __init__(self, cfg: dict, device: str = "cuda", compile_mode: str | None = None,
                 text_cache: str | None = None, checkpoint: str | None = None):
        from sam3.model_builder import (build_pruned_sam3_image_model,
                                        build_sam3_image_model, load_pruned_config)
        from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
        self.imgsz = int(cfg.get("imgsz", 1008))
        if self.imgsz % 14:
            raise ValueError("imgsz must be a multiple of 14")
        # checkpoint: None -> the full SAM3 from the gated facebook/sam3 repo;
        # a DART pruned checkpoint (mehmetkeremturkcan/DART, public) is detected
        # by its embedded pruning config and built without any gated download.
        pruned = load_pruned_config(checkpoint) if checkpoint else None
        if pruned is not None:
            model = build_pruned_sam3_image_model(checkpoint_path=checkpoint,
                                                  pruning_config=pruned, device=device,
                                                  eval_mode=True)
        else:
            model = build_sam3_image_model(device=device, checkpoint_path=checkpoint,
                                           eval_mode=True)
        self.checkpoint_name = Path(checkpoint).name if checkpoint else "facebook/sam3:sam3.pt"
        if self.imgsz != 1008:
            model.backbone.vision_backbone.position_encoding.precompute_for_resolution(self.imgsz)
        self.predictor = Sam3MultiClassPredictorFast(
            model, resolution=self.imgsz, device=device, compile_mode=compile_mode,
            use_fp16=True)
        self.prompts = list(cfg["queries"].keys())
        self.canon = dict(cfg["queries"])
        # v1: suppressor prompts are detected but never emitted — each kills
        # overlapping detections of its target classes (motors-as-person,
        # reflections-as-object; teacher_comparison.md §4b/4c). Absent key
        # -> exact v0 behaviour.
        self.suppressors = {k: list(v) for k, v in (cfg.get("suppressors") or {}).items()}
        self.prompts += [k for k in self.suppressors if k not in self.prompts]
        self.predictor.set_classes(self.prompts, text_cache=text_cache) if text_cache \
            else self.predictor.set_classes(self.prompts)
        self.conf = float(cfg.get("confidence_threshold", 0.30))
        self.nms = float(cfg.get("nms_threshold", 0.50))

    def detect(self, img_bgr: np.ndarray, want_masks: bool = False) -> list[dict]:
        """-> [{cls (canonical), xyxy (normalised 0-1), score, prompt[, polygon]}].

        want_masks: attach the largest-contour polygon of each detection's SAM3
        mask as a flat normalised [x0,y0,x1,y1,...] list (the data/det_seg
        convention). The fast predictor computes masks regardless (its
        detection_only defaults to False), so this costs only the contouring."""
        h, w = img_bgr.shape[:2]
        image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        state = self.predictor.set_image(image)
        res = self.predictor.predict(state, confidence_threshold=self.conf,
                                     nms_threshold=self.nms)
        out = []
        boxes = res.get("boxes")
        masks = res.get("masks") if want_masks else None
        n = 0 if boxes is None else len(boxes)
        for i in range(n):
            prompt = res["class_names"][i]
            cls = self.canon.get(prompt)
            if cls is None and prompt not in self.suppressors:
                continue
            x0, y0, x1, y1 = [float(v) for v in boxes[i].tolist()]
            # DART returns pixel xyxy in the input image frame; guard for a
            # normalised variant (all coords <= 1) so a library change is loud.
            if max(x0, y0, x1, y1) <= 1.0001:
                raise RuntimeError("DART returned normalised boxes; adjust the driver")
            d = {"cls": cls if cls is not None else "_suppressor", "prompt": prompt,
                 "xyxy": [x0 / w, y0 / h, x1 / w, y1 / h],
                 "score": float(res["scores"][i])}
            if cls is None:
                d["targets"] = self.suppressors[prompt]
            if masks is not None and i < len(masks) and masks[i] is not None:
                poly = _mask_polygon(np.asarray(masks[i].squeeze().float().cpu() > 0.5,
                                               dtype=np.uint8))
                if poly is not None:
                    mh, mw = masks[i].squeeze().shape[-2:]
                    d["polygon"] = [round(v / (mw if k % 2 == 0 else mh), 4)
                                    for k, v in enumerate(poly)]
            out.append(d)
        return out


def _mask_polygon(mask_u8: np.ndarray, max_pts: int = 64):
    """Largest external contour as a flat [x0,y0,x1,y1,...] pixel list, or None."""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    eps = 0.002 * cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, eps, True)
    while len(c) > max_pts:
        eps *= 1.5
        c = cv2.approxPolyDP(c, eps, True)
    return [float(v) for pt in c.reshape(-1, 2) for v in pt]


def apply_suppressors(dets: list[dict], ios_thr: float = 0.5) -> list[dict]:
    """Drop detections overlapped by a suppressor prompt's box (targets
    listed per suppressor). Overlap = intersection over the SMALLER area:
    a motor box is small relative to the person box it explains, so IoU
    would under-trigger. Suppressor pseudo-detections are never emitted."""
    sups = [d for d in dets if d["cls"] == "_suppressor"]
    if not sups:
        return [d for d in dets if d["cls"] != "_suppressor"]

    def ios(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return inter / small if small > 0 else 0.0

    out = []
    for d in dets:
        if d["cls"] == "_suppressor":
            continue
        if any(d["cls"] in sp["targets"] and ios(d["xyxy"], sp["xyxy"]) >= ios_thr
               for sp in sups):
            continue
        out.append(d)
    return out


def merge_per_class_nms(dets: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """Several prompts map to one canonical class: keep the top-scoring box per
    overlapping group within a class (mirrors the DINO recipe's per-class NMS)."""
    keep: list[dict] = []
    for cls in sorted({d["cls"] for d in dets}):
        cand = sorted((d for d in dets if d["cls"] == cls), key=lambda d: -d["score"])
        chosen: list[dict] = []
        for d in cand:
            if all(_iou(d["xyxy"], c["xyxy"]) < iou_thr for c in chosen):
                chosen.append(d)
        keep.extend(chosen)
    return keep


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--only", default=None, help="substring filter on clip dirs")
    ap.add_argument("--frame-list", default=None,
                    help="optional file of frame_ids to label (one per line); "
                         "others are skipped")
    ap.add_argument("--checkpoint", default=None,
                    help="SAM3 .pt (full, or a DART pruned checkpoint); default = gated HF download")
    ap.add_argument("--compile", default=None, help="torch.compile mode or unset")
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--max-frames", type=int, default=0, help="smoke: stop after N")
    ap.add_argument("--masks", action="store_true",
                    help="attach SAM3 mask polygons (per-box `polygon`, normalised)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    backend = DartBackend(cfg, compile_mode=args.compile, text_cache=args.text_cache,
                          checkpoint=args.checkpoint)
    detection = load_detection()
    fusion_params = detection["fusion"]
    intr_fish = load_intrinsics()["fisheye"]
    model_version = f"dart/{backend.checkpoint_name}@{cfg.get('prompt_set', 'v0')}"
    wanted = None
    if args.frame_list:
        wanted = {l.strip() for l in Path(args.frame_list).read_text().splitlines() if l.strip()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(d for d in Path(args.bundle_root).iterdir()
                   if d.is_dir() and ((d / "meta.json").exists() or any(d.glob("ts=*.jpg"))))
    if args.only:
        clips = [c for c in clips if args.only in c.name]

    t0 = time.time()
    total = 0
    for ci, clip in enumerate(clips):
        if (clip / "meta.json").exists():
            meta = json.loads((clip / "meta.json").read_text())
            scene = meta["scene"]
            first_chunk = meta["chunks"][0]
            chunk_of = {f["ts"]: f.get("chunk", first_chunk) for f in meta["frames"]}
            frame_glob = clip / "frames"
        else:
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
            if frame_id in done or (wanted is not None and frame_id not in wanted):
                continue
            todo.append((p, chunk, frame_id, ts))
        if not todo:
            print(f"[{ci + 1}/{len(clips)}] {clip.name}: nothing to do ({len(frames)} frames)",
                  flush=True)
            continue
        n = 0
        with open(out_path, "a") as fp:
            for p, chunk, frame_id, ts in todo:
                im = cv2.imread(str(p))
                if im is None:
                    continue
                im = cv2.resize(im, NATIVE, interpolation=cv2.INTER_CUBIC)
                dets = apply_suppressors(backend.detect(im, want_masks=args.masks),
                                         float(cfg.get("suppressor_ios", 0.5)))
                dets = merge_per_class_nms(dets,
                                           cfg.get("nms_threshold", 0.5))
                dets = suppress_structure_under_boat(dets, cfg.get("boat_gate"))
                rec = detections_to_record(frame_id, scene, chunk, ts.replace("-", ":", 2),
                                           im.shape, dets, model_version, fusion_params,
                                           intr_fish)
                rec["source"] = "dart-sam3"
                # keep the winning prompt (prompt-tuning signal) + the mask polygon per box
                for bb, d in zip(rec["fisheye_bboxes"], dets):
                    bb["prompt"] = d["prompt"]
                    if "polygon" in d:
                        bb["polygon"] = d["polygon"]
                fp.write(json.dumps(rec) + "\n")
                n += 1
                total += 1
                if n % 200 == 0:
                    fp.flush()
                    print(f"  {clip.name}: {n}/{len(todo)}  ({(time.time() - t0):.0f}s, "
                          f"{total / max(time.time() - t0, 1e-6):.2f} rec/s)", flush=True)
                if args.max_frames and total >= args.max_frames:
                    break
        print(f"[{ci + 1}/{len(clips)}] {clip.name}: +{n} labelled "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
        if args.max_frames and total >= args.max_frames:
            break
    print(f"DONE {total} records in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
