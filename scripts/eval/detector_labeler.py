"""Open-vocabulary detector labeller (GroundingDINO): Phase I.4.5 Track B.

A purpose-built zero-shot object detector gives much tighter boxes than the
VLM on small water objects (boats, people), which the 8B model localises only
to ~5-10%. This module is the detector-based sibling of
``scripts/eval/qwen_batch_labeler.py``:

  * it reuses the dashboard's ``iter_clip_frames`` (undistorted frames, the
    ``ts=`` frame_id scheme), so its output shares frame_ids with the Qwen /
    manual / dashboard labels;
  * it writes the **same dashboard-compatible record schema**, so the output
    is interchangeable with the Qwen output and auditable with
    ``label_tool.py --audit`` and consumed by ``metrics.py``;
  * it is resumable (skip-done + per-frame response cache).

The model, query phrases, and thresholds are edited in
``configs/detector_labeler.yaml``. GroundingDINO is text-prompted: each query
phrase is matched against the image; ``queries`` maps the phrase onto one of
our canonical classes (boat/duck/buoy/person/structure/other).

Typical GPU-node invocation (see scripts/gpu_labeling/iterate_detector.sh)::

    python -m scripts.eval.detector_labeler \\
        --captures-dir /scratch0/$USER/asvproject/data/captures/2026-06-17_institutionone_day1 \\
        --frames-file labels/qwen/smoke_set.txt \\
        --out /scratch0/$USER/asvproject/out/det.jsonl \\
        --cache-dir /scratch0/$USER/asvproject/out/det_cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from scripts.sensor_processing.pipeline import angle_to_bin, make_bins
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import REPO_ROOT
from scripts.eval.qwen_batch_labeler import (
    _frame_to_jpeg_bytes,
    _load_done_frame_ids,
    iter_clip_frames,
    list_clip_timestamps,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "detector_labeler.yaml"

_FALLBACK_CFG = {
    "model": "IDEA-Research/grounding-dino-base",
    "passes": [{"name": "objects", "box_threshold": 0.35, "text_threshold": 0.30,
                "queries": {"boat": "boat", "duck": "duck", "buoy": "buoy",
                            "person": "person"}},
               {"name": "structure", "box_threshold": 0.28, "text_threshold": 0.22,
                "queries": {"post": "structure", "ladder": "structure",
                            "platform": "structure"}}],
}


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    try:
        import yaml
        cfg = yaml.safe_load(p.read_text())
    except FileNotFoundError:
        cfg = dict(_FALLBACK_CFG)
    cfg.setdefault("model", _FALLBACK_CFG["model"])
    cfg.setdefault("image_shortest_edge", None)
    cfg.setdefault("image_longest_edge", None)
    cfg.setdefault("boat_gate", None)  # drop structure boxes mostly on a boat
    # Back-compat: a flat queries+thresholds config is wrapped into one pass.
    if "passes" not in cfg:
        cfg["passes"] = [{"name": "all",
                          "box_threshold": cfg.get("box_threshold", 0.30),
                          "text_threshold": cfg.get("text_threshold", 0.25),
                          "queries": cfg.get("queries", {})}]
    for ps in cfg["passes"]:
        ps.setdefault("box_threshold", 0.30)
        ps.setdefault("text_threshold", 0.25)
        ps["queries"] = {str(k).strip().lower(): str(v).strip().lower()
                         for k, v in (ps.get("queries") or {}).items()}
    return cfg


def _fingerprint(cfg: dict) -> str:
    return json.dumps({"model": cfg["model"], "passes": cfg["passes"],
                       "se": cfg.get("image_shortest_edge")}, sort_keys=True)


def _iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(dets: list[dict], iou_thr: float = 0.6) -> list[dict]:
    """Greedy per-class NMS over merged passes (dets have cls/xyxy/score)."""
    kept: list[dict] = []
    for cls in {d["cls"] for d in dets}:
        cd = sorted([d for d in dets if d["cls"] == cls], key=lambda d: -d["score"])
        keep: list[dict] = []
        for d in cd:
            if all(_iou(d["xyxy"], k["xyxy"]) < iou_thr for k in keep):
                keep.append(d)
        kept.extend(keep)
    return kept


def _inter(a: list[float], b: list[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def suppress_structure_under_boat(dets: list[dict], contain_thr) -> list[dict]:
    """Drop `structure` boxes mostly covered by a `boat` box.

    A vessel sometimes gets both a boat box and a structure box (its mast/rig
    reads as a post). When `contain_thr` is set (0..1], a structure whose area
    is >= that fraction inside any boat box is dropped. None / >1 disables it.
    """
    if not contain_thr or contain_thr > 1.0:
        return dets
    boats = [d for d in dets if d["cls"] == "boat"]
    if not boats:
        return dets
    out = []
    for d in dets:
        if d["cls"] == "structure":
            sa = max(1e-9, (d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1]))
            covered = max((_inter(d["xyxy"], b["xyxy"]) for b in boats), default=0.0) / sa
            if covered >= contain_thr:
                continue
        out.append(d)
    return out


def _map_class(label: str, queries: dict[str, str]) -> str | None:
    """Map a GroundingDINO phrase onto a canonical class, or None to drop it.

    At low thresholds GroundingDINO emits fuzzy/garbled labels that match no
    query phrase; those are junk and returning None drops them (rather than
    flooding an `other` bucket of false positives)."""
    s = str(label).strip().lower()
    if s in queries:
        return queries[s]
    # GroundingDINO may return a sub-/super-string of the query phrase.
    for phrase, cls in queries.items():
        if phrase in s or s in phrase:
            return cls
    return None


# ---------------------------------------------------------------------------
# Detector backend
# ---------------------------------------------------------------------------

class GroundingDINO:
    def __init__(self, model_id: str, shortest_edge: int | None = None,
                 longest_edge: int | None = None):
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[gdino] loading {model_id} on {self.device} "
              f"(shortest_edge={shortest_edge or 'default'}) ...", flush=True)
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Bump the detector's input resolution: GroundingDINO defaults to an
        # 800px shortest edge, which loses thin poles. A larger edge gives the
        # backbone a finer feature map (slower, more recall on small objects).
        if shortest_edge:
            ip = getattr(self.processor, "image_processor", self.processor)
            ip.size = {"shortest_edge": int(shortest_edge),
                       "longest_edge": int(longest_edge or 4 * shortest_edge)}
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(self.device).eval()
        print(f"[gdino] loaded in {time.time() - t0:.0f}s", flush=True)

    def detect(self, imgs_bgr: list[np.ndarray], text: str,
               box_threshold: float, text_threshold: float) -> list[list[dict]]:
        """Return per-image list of {label, xyxy_norm, score}."""
        from PIL import Image
        pils = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in imgs_bgr]
        sizes = [(im.shape[0], im.shape[1]) for im in imgs_bgr]  # (H, W)
        inputs = self.processor(images=pils, text=[text] * len(pils),
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        # transformers renamed args across versions; try the modern signature
        # then fall back to the older `box_threshold` kwarg.
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=box_threshold,
                text_threshold=text_threshold, target_sizes=sizes)
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=box_threshold,
                text_threshold=text_threshold, target_sizes=sizes)

        out: list[list[dict]] = []
        for res, (h, w) in zip(results, sizes):
            labels = res.get("text_labels", res.get("labels", []))
            dets = []
            for box, score, label in zip(res["boxes"], res["scores"], labels):
                x0, y0, x1, y1 = (float(v) for v in box.tolist())
                dets.append({
                    "label": str(label),
                    "score": float(score),
                    "xyxy": [x0 / w, y0 / h, x1 / w, y1 / h],
                })
            out.append(dets)
        return out


def label_batch(backend, imgs_bgr: list[np.ndarray], passes: list[dict]
                ) -> list[list[dict]]:
    """Run every pass and merge per image into class-tagged dets (cls/xyxy/score)."""
    per_img: list[list[dict]] = [[] for _ in imgs_bgr]
    for ps in passes:
        text = " . ".join(ps["queries"].keys()) + " ."
        res = backend.detect(imgs_bgr, text, ps["box_threshold"], ps["text_threshold"])
        for i, dets in enumerate(res):
            for d in dets:
                cls = _map_class(d["label"], ps["queries"])
                if cls is None:
                    continue  # unmatched / junk label; drop it
                per_img[i].append({"cls": cls, "xyxy": d["xyxy"],
                                   "score": d["score"]})
    return [_nms(d) for d in per_img]


# ---------------------------------------------------------------------------
# Record building (dashboard-compatible, identical bins to label_tool)
# ---------------------------------------------------------------------------

def detections_to_record(frame_id, scene, triplet_ts, frame_ts, img_shape,
                         dets, model, fusion_params, intr_fish) -> dict:
    h, w = img_shape[:2]
    bboxes = []
    for d in dets:
        x0, y0, x1, y1 = d["xyxy"]
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in (x0, y0, x1, y1))
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        bboxes.append({"cls": d["cls"],
                       "xyxy": [x0, y0, x1, y1],
                       "confidence": round(float(d["score"]), 4)})

    edges, centers = make_bins(fusion_params)
    bins: set[int] = set()
    for bb in bboxes:
        x0, _y0, x1, _y1 = bb["xyxy"]
        cx_px = (x0 + x1) / 2.0 * w
        angle = (cx_px - intr_fish["cx"]) / intr_fish["pix_deg_ratio"]
        idx = angle_to_bin(angle, edges)
        if idx is not None:
            bins.add(int(centers[idx]))

    return {
        "frame_id": frame_id, "scene": scene, "triplet_ts": triplet_ts,
        "frame_ts": frame_ts, "source": "grounding-dino", "model_version": model,
        "audited": False, "fisheye_bboxes": bboxes,
        "obstacle_bins_fisheye": sorted(bins), "width": w, "height": h,
    }


def _frame_hash(jpeg: bytes, fp: str) -> str:
    h = hashlib.sha256()
    h.update(fp.encode())
    h.update(jpeg)
    return h.hexdigest()[:16]


def apply_exposure(img_bgr: np.ndarray, factor: float) -> np.ndarray:
    """Brighten by a linear gain (approx. exposure bump). factor 1.0 = no-op."""
    if factor == 1.0:
        return img_bgr
    return cv2.convertScaleAbs(img_bgr, alpha=factor, beta=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures-dir", required=True)
    ap.add_argument("--from", dest="clip_from", default=None)
    ap.add_argument("--to", dest="clip_to", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--config", default=None, help="detector_labeler.yaml path.")
    ap.add_argument("--model", default=None, help="Override config model id.")
    ap.add_argument("--box-threshold", type=float, default=None)
    ap.add_argument("--text-threshold", type=float, default=None)
    ap.add_argument("--image-shortest-edge", type=int, default=None,
                    help="Detector input resolution (default 800). Larger = "
                         "finer feature map, better on thin poles, slower.")
    ap.add_argument("--frames-file", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--exposure", type=float, default=1.0,
                    help="Linear brightness gain applied to every frame before "
                         "detection (1.0 = off; ~1.4 brightens dark water scenes).")
    ap.add_argument("--boat-gate", type=float, default=None,
                    help="Drop a structure box if >= this fraction of its area is "
                         "inside a boat box (0..1; off by default).")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.image_shortest_edge is not None:
        cfg["image_shortest_edge"] = args.image_shortest_edge
    if args.boat_gate is not None:
        cfg["boat_gate"] = args.boat_gate
    # --box-threshold/--text-threshold override EVERY pass (quick experiments).
    for ps in cfg["passes"]:
        if args.box_threshold is not None:
            ps["box_threshold"] = args.box_threshold
        if args.text_threshold is not None:
            ps["text_threshold"] = args.text_threshold
    print(f"detector: {cfg['model']}", flush=True)
    for ps in cfg["passes"]:
        print(f"  pass '{ps['name']}' bt={ps['box_threshold']} "
              f"tt={ps['text_threshold']}: {' . '.join(ps['queries'].keys())}",
              flush=True)

    captures_dir = Path(args.captures_dir).expanduser().resolve()
    if not captures_dir.is_dir():
        raise SystemExit(f"--captures-dir not found: {captures_dir}")
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = (Path(args.cache_dir).expanduser().resolve()
                 if args.cache_dir else out_path.parent / "det_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames_filter = None
    if args.frames_file:
        frames_filter = {ln.split("#", 1)[0].strip()
                         for ln in Path(args.frames_file).read_text().splitlines()
                         if ln.split("#", 1)[0].strip()}

    intrinsics = load_intrinsics()
    detection = load_detection()
    fusion_params = detection["fusion"]
    intr_fish = intrinsics["fisheye"]
    fp = _fingerprint(cfg) + f"|exp{args.exposure}"
    if args.exposure != 1.0:
        print(f"exposure gain: {args.exposure}x", flush=True)

    clips = list_clip_timestamps(captures_dir, args.clip_from, args.clip_to)
    if frames_filter is not None:
        want = {fid.split("/")[1] for fid in frames_filter if "/" in fid}
        clips = [c for c in clips if c in want]
    if not clips:
        raise SystemExit("no clips for the given range / frames-file")
    print(f"clips: {len(clips)} ({clips[0]} .. {clips[-1]})", flush=True)

    done = set() if args.no_resume else _load_done_frame_ids(out_path)
    if done:
        print(f"resume: skipping {len(done)} already-labelled frames", flush=True)

    backend = None if args.dry_run else GroundingDINO(
        cfg["model"], cfg.get("image_shortest_edge"), cfg.get("image_longest_edge"))
    out = open(out_path, "a")
    n_seen = n_written = n_calls = n_cache = n_bad = 0
    t0 = time.time()
    pending: list[tuple] = []  # (frame_id, scene, tts, fts, img, jpeg, hash)

    def flush():
        nonlocal n_written, n_calls
        if not pending:
            return
        imgs = [p[4] for p in pending]
        if args.dry_run:
            batch_dets = [[] for _ in pending]
        else:
            batch_dets = label_batch(backend, imgs, cfg["passes"])
        for (frame_id, scene, tts, fts, img, _jpeg, fhash), dets in zip(pending, batch_dets):
            if not args.dry_run:
                (cache_dir / f"{fhash}.json").write_text(json.dumps(
                    {"frame_id": frame_id, "model": cfg["model"], "dets": dets}))
                n_calls += 1
            rec = detections_to_record(
                frame_id, scene, tts, fts, img.shape,
                suppress_structure_under_boat(dets, cfg.get("boat_gate")),
                cfg["model"], fusion_params, intr_fish)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            done.add(frame_id)
            n_written += 1
            print(f"  [{frame_id}] n={len(rec['fisheye_bboxes'])} "
                  f"bins={rec['obstacle_bins_fisheye']}", flush=True)
        pending.clear()

    try:
        for clip_ts in clips:
            print(f"\n=== clip {clip_ts} ===", flush=True)
            try:
                for frame_id, scene, tts, fts, img in iter_clip_frames(
                        captures_dir, clip_ts, args.every, detection, intrinsics):
                    if frames_filter is not None and frame_id not in frames_filter:
                        continue
                    if args.max_frames is not None and n_seen >= args.max_frames:
                        break
                    n_seen += 1
                    if frame_id in done:
                        continue
                    img = apply_exposure(img, args.exposure)
                    jpeg = _frame_to_jpeg_bytes(img)
                    fhash = _frame_hash(jpeg, fp)
                    cache_path = cache_dir / f"{fhash}.json"
                    if cache_path.exists():
                        dets = json.loads(cache_path.read_text())["dets"]
                        rec = detections_to_record(
                            frame_id, scene, tts, fts, img.shape,
                            suppress_structure_under_boat(dets, cfg.get("boat_gate")),
                            cfg["model"], fusion_params, intr_fish)
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
                        done.add(frame_id)
                        n_written += 1
                        n_cache += 1
                        continue
                    pending.append((frame_id, scene, tts, fts, img, jpeg, fhash))
                    if len(pending) >= args.batch_size:
                        flush()
            except Exception as e:  # noqa: BLE001 - skip unreadable clips
                flush()
                print(f"!! clip {clip_ts}: SKIPPED ({type(e).__name__}: {e})", flush=True)
                n_bad += 1
                continue
            if args.max_frames is not None and n_seen >= args.max_frames:
                break
        flush()
    finally:
        flush()
        out.close()

    dt = time.time() - t0
    print(f"\ndone: seen={n_seen} written={n_written} (calls={n_calls} "
          f"cache={n_cache} bad_clips={n_bad}) in {dt:.0f}s "
          f"({n_written / dt if dt else 0:.2f} f/s)", flush=True)
    print(f"output -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
