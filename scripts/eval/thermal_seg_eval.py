#!/usr/bin/env python3
"""Thermal segmentation student as an obstacle DETECTOR, scored in bearing
space against human-audited boxes (2026-09-12, gate 1 of the MassMIND track).

The classical thermal path is scored by `thermal_gt_eval`; this tool scores a
segmentation ONNX (LRASPP student, 160x120 gray x3, ImageNet-normalised, the
`train_student` export) the same way: obstacle-class connected components on
the undistorted thermal frame become bearing intervals under the thermal
LINEAR model (cx 77 / 2.82 px/deg, the pipeline's `bearing_model: linear`),
and are matched against the audited boxes reduced to bearing intervals
(fisheye boxes via the pinhole K, pure-thermal boxes via the linear model),
clipped to the thermal FOV.

Frames come from a dashboard bundle (thermal JPEG per frame ts); the same
repair_dead_rows + undistort_thermal as the pipeline are applied first.

    python -m scripts.eval.thermal_seg_eval \\
        --onnx models/onnx/thermal_night_v4b_s0.onnx \\
        --labels labels/audited/det_2026-09-08_afloat_2026-09-08_17-10-55.jsonl \\
        --bundle-root /Volumes/ROS2_SSD/asvproject/dashboard_bundle \\
        --band 20:16:00-21:25:00 --json-out results/thermal_seg_eval/v4b_night.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.cv_common import repair_dead_rows, undistort_thermal

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
SEG_OBSTACLE = 0
FX_F, CX_F, W_F = 418.51, 444.11, 864          # fisheye pinhole at 864 px
T_CX, T_PXDEG, T_W = 77.0, 2.82, 160           # thermal linear model
FOV = ((0 - T_CX) / T_PXDEG, (T_W - 1 - T_CX) / T_PXDEG)


def sec(ts: str) -> float:
    h, m, s = ts.replace("-", ":").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def gt_intervals(rec: dict, classes: set | None, max_width: float = 35.0):
    out = []
    w = int(rec.get("width", W_F))
    for bb in rec.get("fisheye_bboxes", []):
        if classes and bb["cls"] not in classes:
            continue
        x0, _, x1, _ = bb["xyxy"]
        lo = math.degrees(math.atan2(min(x0, x1) * w - CX_F * w / W_F, FX_F * w / W_F))
        hi = math.degrees(math.atan2(max(x0, x1) * w - CX_F * w / W_F, FX_F * w / W_F))
        if hi < FOV[0] or lo > FOV[1]:
            continue
        lo, hi = max(lo, FOV[0]), min(hi, FOV[1])
        if hi - lo > max_width:
            continue
        out.append((bb["cls"], lo, hi))
    tw = int(rec.get("thermal_width", T_W))
    for bb in rec.get("thermal_bboxes", []):
        if classes and bb["cls"] not in classes:
            continue
        x0, _, x1, _ = bb["xyxy"]
        out.append((bb["cls"], (min(x0, x1) * tw - T_CX) / T_PXDEG,
                    (max(x0, x1) * tw - T_CX) / T_PXDEG))
    return out


def find_thermal_jpeg(bundle_root: str, scene: str, ts: str) -> str | None:
    key = "ts=" + ts.replace(":", "-") + ".jpg"
    for d in glob.glob(os.path.join(bundle_root, f"{scene}__*")):
        p = os.path.join(d, "thermal", key)
        if os.path.exists(p):
            return p
    return None


class SegOnnx:
    def __init__(self, path: str):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape
        self.h, self.w = int(shp[2]), int(shp[3])

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        img = cv2.resize(gray, (self.w, self.h))
        x = np.stack([img, img, img], -1).astype(np.float32) / 255.0
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
        logits = self.sess.run(None, {self.inp: x})[0][0]
        return logits.argmax(0).astype(np.uint8)


def protrusions_to_intervals(mask: np.ndarray, w_native: int, min_depth: int = 2,
                             min_width: int = 2, margin: int = 1):
    """Column runs where the OBSTACLE class intrudes into the water region
    below the segmented water edge (things floating in or rising out of the
    water), ignoring the shore/sky band above it. Robust to the shore band
    merging with the boats, which defeats connected components."""
    from scripts.utils.segmentation import horizon_from_water
    h, w = mask.shape
    slope, intercept, conf = horizon_from_water(mask)
    ys = np.arange(h)[:, None]
    if conf > 0.0:
        edge = slope * np.arange(w)[None, :] + intercept + margin
    else:
        edge = np.full((1, w), h * 0.5)
    below = (mask == SEG_OBSTACLE) & (ys >= edge)
    depth = below.sum(0)
    active = depth >= min_depth
    sx = w_native / w
    out = []
    x = 0
    while x < w:
        if not active[x]:
            x += 1
            continue
        x0 = x
        while x < w and (active[x] or (x + 1 < w and active[x + 1])):
            x += 1
        if x - x0 >= min_width:
            lo = (x0 * sx - T_CX) / T_PXDEG
            hi = (x * sx - T_CX) / T_PXDEG
            out.append((lo, hi, int(depth[x0:x].sum()), 0))
    return out


def components_to_intervals(mask: np.ndarray, min_area: int, w_native: int):
    obst = (mask == SEG_OBSTACLE).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(obst, 8)
    out = []
    sx = w_native / mask.shape[1]
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        # ignore components that are the whole sky/shore band: wider than the FOV
        lo = ((x) * sx - T_CX) / T_PXDEG
        hi = ((x + w) * sx - T_CX) / T_PXDEG
        out.append((lo, hi, int(a), int(y + h)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--bundle-root", required=True)
    ap.add_argument("--band", default=None, help="HH:MM:SS-HH:MM:SS frame_ts window")
    ap.add_argument("--tol", type=float, default=6.0)
    ap.add_argument("--min-area", type=int, default=8, help="component area in student px")
    ap.add_argument("--max-width", type=float, default=35.0, help="drop components wider than this (deg): sky/shore bands")
    ap.add_argument("--classes", default=None, help="comma list; default all")
    ap.add_argument("--reduce", default="protrusion", choices=["protrusion", "components"],
                    help="mask -> bearing intervals: obstacle protrusions below the water edge (default) or connected components")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)

    intr = load_intrinsics()
    det = load_detection()
    dead = det["thermal"].get("dead_rows") or []
    K_t, D_t = intr["thermal"]["K"], intr["thermal"]["D"]
    classes = set(a.classes.split(",")) if a.classes else None
    band = None
    if a.band:
        lo, hi = a.band.split("-")
        band = (sec(lo), sec(hi))
    model = SegOnnx(a.onnx)

    n_frames = n_missing = 0
    gt_n = defaultdict(int); gt_rec = defaultdict(int)
    det_n = det_tp = 0
    obst_frac = []
    for lf in a.labels:
        for line in open(lf):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = rec["frame_ts"]
            if band and not (band[0] <= sec(ts) <= band[1]):
                continue
            jp = find_thermal_jpeg(a.bundle_root, rec["scene"], ts)
            if jp is None:
                n_missing += 1
                continue
            frame = cv2.imread(jp)
            if frame is None:
                n_missing += 1
                continue
            frame = repair_dead_rows(frame, dead)
            und = undistort_thermal(frame, K_t, D_t)
            gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY) if und.ndim == 3 else und
            mask = model(gray)
            obst_frac.append(float((mask == SEG_OBSTACLE).mean()))
            if a.reduce == "protrusion":
                preds = protrusions_to_intervals(mask, und.shape[1])
            else:
                preds = components_to_intervals(mask, a.min_area, und.shape[1])
            preds = [p for p in preds if p[1] - p[0] <= a.max_width]
            gts = gt_intervals(rec, classes)
            n_frames += 1
            for cls, lo, hi in gts:
                gt_n[cls] += 1
                if any(p[0] <= hi + a.tol and p[1] >= lo - a.tol for p in preds):
                    gt_rec[cls] += 1
            for p in preds:
                det_n += 1
                if any(p[0] <= hi + a.tol and p[1] >= lo - a.tol for _, lo, hi in gts):
                    det_tp += 1
    tot = sum(gt_n.values()); rec_tot = sum(gt_rec.values())
    out = {
        "onnx": a.onnx, "band": a.band, "n_frames": n_frames, "n_missing": n_missing,
        "n_gt": tot, "recall": rec_tot / tot if tot else float("nan"),
        "per_class": {c: {"n": gt_n[c], "recalled": gt_rec[c]} for c in sorted(gt_n)},
        "n_det": det_n, "n_det_tp": det_tp,
        "precision": det_tp / det_n if det_n else float("nan"),
        "fp_per_frame": (det_n - det_tp) / n_frames if n_frames else float("nan"),
        "obstacle_frac_mean": float(np.mean(obst_frac)) if obst_frac else float("nan"),
    }
    print(json.dumps(out, indent=1))
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
