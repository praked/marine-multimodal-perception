"""Cross-modal thermal pseudo-labels: warp fisheye water-seg masks to thermal.

First stage of the thermal-student experiment (2026-07-09,
docs/history/2026-07-09_thermal_tuning.md). The fisheye LRASPP student's
water/sky/obstacle masks (SegProvider layout under data/seg/) are warped
into the (undistorted) thermal frame and dumped as (image, label) PNG pairs
that scripts/gpu_seg/train_student.py consumes unchanged (--size 160x120).

Geometry: a rotation-only mapping, justified twice over:
  * the measured fisheye<->thermal baseline is ~6 cm, so parallax is < 1 deg
    for anything beyond ~2.5 m (label classes at nearer range are big
    blobs anyway);
  * horizontally, thermal bearings under the LINEAR model were arbitrated
    against fisheye GT at 0.47 deg MAE (thermal_tuning log, follow-up
    section): the mapping is bearing = bearing.
Per-frame attitude is NOT needed: both cameras ride the same rigid box, so
the mapping is a fixed angular transform. The one unknown is the vertical:
we assume square thermal pixels (pix/deg vertical = horizontal 2.82) and
fit a single row offset from horizon correspondence (fisheye water-edge
line vs thermal RANSAC horizon) across every confident frame.

Output tree (train_student.py's <parent>/<stem> pair matching):
    <out>/images/<scene>__<ts>/ts=<HH-MM-SS.f>.png   (undistorted gray x3)
    <out>/labels/<scene>__<ts>/ts=<HH-MM-SS.f>.png   (0/1/2, 255=out-of-view)
    <out>/overlays/<scene>__<ts>__ts=*.jpg           (eyeball gate)
    <out>/fit_report.json

Typical (GPU node or laptop, no torch needed):
    python -m scripts.gpu_seg.thermal_labels \
        --triplet data/captures/2026-07-08/2026-07-08_16-04-11 \
        --triplet data/captures/2026-07-08/2026-07-08_16-18-12 \
        --triplet data/captures/2026-07-08/2026-07-08_16-37-01 \
        --triplet data/captures/2026-07-08/2026-07-08_16-44-56 \
        --out data/thermal_train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from scripts.sensor_processing.pipeline import iterate_triplet
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.cv_common import (detect_horizon, horizon_kwargs,
                                     repair_dead_rows, undistort_thermal)
from scripts.utils.datasets import REPO_ROOT, resolve_triplet
from scripts.utils.segmentation import horizon_from_water

IGNORE = 255                 # out-of-fisheye-view; train_student maps >=3 -> ignore
OVERLAY_EVERY = 150          # one gate overlay every N labelled frames
# Label colours for overlays (BGR): obstacle red, water blue, sky cyan-ish.
_COLORS = {0: (0, 0, 255), 1: (255, 80, 0), 2: (200, 200, 0)}



def _frames(triplet, detection):
    """iterate_triplet, but an unreadable clip (power-cut chunk without a moov
    atom, 2026-09-08 21-36-23) is skipped with a warning instead of aborting
    the whole export."""
    try:
        yield from iterate_triplet(triplet, detection)
    except RuntimeError as exc:
        print(f"WARNING: skipping {triplet.scene}__{triplet.timestamp}: {exc}",
              file=sys.stderr)

def _mask_path(seg_root: Path, clip_key: str, ts: str) -> Path:
    return seg_root / clip_key / f"ts={ts.replace(':', '-')}.png"


def fit_vertical_offset(
    triplets: list[str],
    seg_root: Path,
    intrinsics: dict,
    detection: dict,
    pdr_v: float,
    min_conf: float = 0.5,
) -> dict:
    """Median thermal row offset b such that v_t = b + pdr_v * elevation_deg.

    Elevation reference: the fisheye water-edge line (from the seg mask,
    robust) and the thermal RANSAC horizon, both sampled at their camera's
    principal column, matched per frame; b_i = r_thermal - pdr_v * phi_fisheye.
    """
    K_f = intrinsics["fisheye"]["K"]
    fy_f, cy_f, cx_f = float(K_f[1, 1]), float(K_f[1, 2]), float(K_f[0, 2])
    cx_t = float(intrinsics["thermal"]["cx"])
    t_params = detection["thermal"]
    dead_rows = t_params.get("dead_rows") or []
    h_params = {**detection["horizon"], **(t_params.get("horizon") or {})}
    K_t, D_t = intrinsics["thermal"]["K"], intrinsics["thermal"]["D"]

    samples: list[float] = []
    for prefix in triplets:
        triplet = resolve_triplet(prefix)
        clip_key = f"{triplet.scene}__{triplet.timestamp}"
        for ts, _fish, therm, _mm in _frames(triplet, detection):
            mp = _mask_path(seg_root, clip_key, ts)
            if not mp.exists():
                continue
            seg = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if seg is None:
                continue
            s_f, b_f, c_f = horizon_from_water(seg)
            if c_f < min_conf:
                continue
            therm = repair_dead_rows(therm, dead_rows)
            und_t = undistort_thermal(therm, K_t, D_t)
            _mask, s_t, b_t, c_t = detect_horizon(und_t, **horizon_kwargs(h_params))
            if c_t < min_conf:
                continue
            r_f = s_f * cx_f + b_f
            phi = float(np.degrees(np.arctan2(r_f - cy_f, fy_f)))
            r_t = s_t * cx_t + b_t
            samples.append(r_t - pdr_v * phi)
    if not samples:
        raise SystemExit("no confident horizon pairs: cannot fit the "
                         "vertical offset (check seg masks / clips)")
    arr = np.asarray(samples)
    return {
        "b_row": float(np.median(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "n_frames": int(len(arr)),
    }


def build_remap(intrinsics: dict, b_row: float, pdr_v: float,
                size_wh: tuple[int, int] = (160, 120),
                ) -> tuple[np.ndarray, np.ndarray]:
    """(map_x, map_y): thermal pixel -> fisheye undistorted pixel."""
    K_f = intrinsics["fisheye"]["K"]
    fx_f, fy_f = float(K_f[0, 0]), float(K_f[1, 1])
    cx_f, cy_f = float(K_f[0, 2]), float(K_f[1, 2])
    cx_t = float(intrinsics["thermal"]["cx"])
    pdr = float(intrinsics["thermal"]["pix_deg_ratio"])
    w, h = size_wh
    u_t = np.arange(w, dtype=np.float32)
    v_t = np.arange(h, dtype=np.float32)
    theta = np.radians((u_t - cx_t) / pdr)                # bearing per column
    phi = np.radians((v_t - b_row) / pdr_v)               # elevation per row
    map_x = np.broadcast_to(cx_f + fx_f * np.tan(theta), (h, w)).astype(np.float32)
    map_y = np.broadcast_to((cy_f + fy_f * np.tan(phi))[:, None], (h, w)).astype(np.float32)
    return np.ascontiguousarray(map_x), np.ascontiguousarray(map_y)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triplet", action="append", required=True)
    ap.add_argument("--fit-triplet", action="append", default=None,
                    help="Clips for the vertical-offset fit (default: all "
                         "--triplet clips).")
    ap.add_argument("--seg-root", default=str(REPO_ROOT / "data" / "seg"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "thermal_train"))
    ap.add_argument("--pdr-v", type=float, default=None,
                    help="Vertical pix/deg (default: horizontal 2.82: "
                         "square pixels).")
    args = ap.parse_args(argv)

    intrinsics = load_intrinsics()
    detection = load_detection()
    seg_root = Path(args.seg_root)
    out = Path(args.out)
    pdr_v = args.pdr_v if args.pdr_v is not None else float(
        intrinsics["thermal"]["pix_deg_ratio"])

    fit_clips = args.fit_triplet or args.triplet
    fit = fit_vertical_offset(fit_clips, seg_root, intrinsics, detection, pdr_v)
    print(f"vertical offset fit: b_row={fit['b_row']:.2f} px "
          f"(IQR {fit['iqr']:.2f}, n={fit['n_frames']} horizon pairs)")

    map_x, map_y = build_remap(intrinsics, fit["b_row"], pdr_v)
    K_t, D_t = intrinsics["thermal"]["K"], intrinsics["thermal"]["D"]
    dead_rows = detection["thermal"].get("dead_rows") or []

    (out / "overlays").mkdir(parents=True, exist_ok=True)
    report = {"fit": fit, "pdr_v": pdr_v, "clips": {}}
    for prefix in args.triplet:
        triplet = resolve_triplet(prefix)
        clip_key = f"{triplet.scene}__{triplet.timestamp}"
        img_dir = out / "images" / clip_key
        lab_dir = out / "labels" / clip_key
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        n_written = n_skipped = 0
        for ts, _fish, therm, _mm in _frames(triplet, detection):
            mp = _mask_path(seg_root, clip_key, ts)
            if not mp.exists():
                n_skipped += 1
                continue
            seg = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if seg is None:
                n_skipped += 1
                continue
            therm = repair_dead_rows(therm, dead_rows)
            und_t = undistort_thermal(therm, K_t, D_t)
            gray = (cv2.cvtColor(und_t, cv2.COLOR_BGR2GRAY)
                    if und_t.ndim == 3 else und_t)
            label = cv2.remap(seg, map_x, map_y,
                              interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=IGNORE)
            stem = f"ts={ts.replace(':', '-')}"
            cv2.imwrite(str(img_dir / f"{stem}.png"), gray)
            cv2.imwrite(str(lab_dir / f"{stem}.png"), label)
            if n_written % OVERLAY_EVERY == 0:
                ov = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for cls, col in _COLORS.items():
                    sel = label == cls
                    ov[sel] = (0.55 * ov[sel] + 0.45 * np.array(col)).astype(np.uint8)
                ov = cv2.resize(ov, (ov.shape[1] * 4, ov.shape[0] * 4),
                                interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(out / "overlays" / f"{clip_key}__{stem}.jpg"), ov)
            n_written += 1
        report["clips"][clip_key] = {"written": n_written, "no_mask": n_skipped}
        print(f"  {clip_key}: {n_written} pairs ({n_skipped} frames without mask)")

    (out / "fit_report.json").write_text(json.dumps(report, indent=2))
    print(f"report -> {out / 'fit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
