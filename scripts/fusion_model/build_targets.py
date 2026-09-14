"""Build bearing-space training targets from labels (Phase 1).

Joins labels/ (manual + qwen/GroundingDINO JSONL, dashboard schema) onto a
clip's exported feature tables (`build_features.py`) and writes
`data/features/<scene>__<ts>/targets.csv`: one row per (labelled frame ×
bearing bin):

  y_obstacle    1 if a label covers the bin's bearing within tolerance,
                looking ±dilation_frames around the frame (GroundingDINO
                flicker; same ±7 convention as thermal_gt_eval).
  y_relevant    False when every label supporting the bin reads beyond
                max_relevant_range_m (or above the horizon) by the same
                monocular water-plane method the pipeline uses: the
                far-bank / perception-GT-vs-nav-GT split, operationalized.
  y_nav         y_obstacle AND y_relevant: the nav-score training target.
  from_dilation 1 when the positive came only from a neighbouring frame.
  label_classes "|"-joined classes of same-frame supporting labels
                (person recall etc.).
  label_min_range_m  nearest supporting label's mono range (NaN unknown).

Label bearings come from the bbox centres under the PINHOLE model (labels
were drawn on undistorted fisheye frames), NOT from the stored
`obstacle_bins_fisheye`, which encode the legacy linear convention with a
known −8.6° median bias. Bearing-space first: bins are taken from the
feature table's meta, tolerance defaults to half a bin.

Rows exist only for frames that carry labels themselves (negatives on
unlabelled frames are unknowable, not zero). Unaudited labels are included
only with --include-unaudited; pseudo-labels are suggestions until
audited (label_tool audit mode).

Usage:
    python -m scripts.fusion_model.build_targets                # all clips w/ features
    python -m scripts.fusion_model.build_targets --clip Boats/2025-06-23_16-21-07
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.eval.metrics import DEFAULT_LABELS, Label, load_labels
from scripts.fusion_model.build_features import FEATURES_ROOT
from scripts.utils.geometry import (
    UP_LEVEL,
    range_from_water_plane_up,
    up_from_horizon_line,
)

# A label bin is "positive" for feature-bin i when |center_i - bearing| <=
# bin_step/2 + tolerance. Default tolerance = half a bin (plan §5: exact
# bin-center equality is too brittle to train against).
DEFAULT_DILATION_FRAMES = 7
# D.2 ruling (AuthorOne, 2026-08-24): nav-relevance is UNBOUNDED — a labelled
# obstacle is nav-relevant regardless of range (far-shore boats included),
# so y_nav == y_obstacle by default. The old 30 m gate (= the
# fusion.vote_range_gate envelope) remains available as an ablation via
# --max-relevant-range; 0/negative = unbounded.
DEFAULT_MAX_RELEVANT_RANGE_M = 0.0
HORIZON_CONF_FLOOR = 0.5              # match range.horizon_min_confidence
THERMAL_WIDTH_PX = 160                # recorded Lepton frame width (pure-thermal audits)


def _thermal_linear_model() -> tuple[float, float]:
    """(cx_px, px_per_deg) of the thermal linear bearing model."""
    try:
        from scripts.utils.calibration import load_intrinsics
        t = load_intrinsics()["thermal"]
        return float(t["cx"]), float(t["pix_deg_ratio"])
    except Exception:
        return 77.0, 2.82


def label_bearings_px(label: Label, width: int, K: np.ndarray
                      ) -> list[tuple[float, tuple[float, float, float, float], str]]:
    """Per bbox: (bearing_deg, bbox_px, cls) under the pinhole model."""
    K = np.asarray(K, dtype=np.float64)
    fx, cx = float(K[0, 0]), float(K[0, 2])
    out = []
    for bb in label.bboxes:
        xyxy = bb.get("xyxy")
        if not xyxy or len(xyxy) != 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in xyxy)
        u = (x0 + x1) / 2.0 * width
        bearing = math.degrees(math.atan2(u - cx, fx))
        out.append((bearing, (x0, y0, x1, y1), str(bb.get("cls", ""))))
    return out


def label_mono_range(bbox_norm: tuple[float, float, float, float],
                     K: np.ndarray, image_size: tuple[int, int],
                     horizon: tuple[float, float, float] | None,
                     camera_height_m: float) -> float | None:
    """Monocular water-plane range of a label bbox (bottom-centre ray):
    the same method as the pipeline's raw_ranges (uncapped). None = the
    bbox bottom back-projects above the horizon (far bank / sky)."""
    w, h = image_size
    x0, y0, x1, y1 = bbox_norm
    bbox_px = (x0 * w, y0 * h, x1 * w, y1 * h)
    K = np.asarray(K, dtype=np.float64)
    if horizon is not None and horizon[2] >= HORIZON_CONF_FLOOR:
        up = up_from_horizon_line(float(horizon[0]), float(horizon[1]), K)
    else:
        up = UP_LEVEL
    return range_from_water_plane_up(bbox_px, K, up, camera_height_m,
                                     max_range_m=None)


def _bin_centers(meta: dict) -> tuple[np.ndarray, float]:
    edges = np.asarray(meta["bin_edges_deg"], dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    step = float(edges[1] - edges[0])
    return centers, step


def build_targets_for_clip(feature_dir: Path, labels: list[Label],
                           dilation_frames: int = DEFAULT_DILATION_FRAMES,
                           tolerance_deg: float | None = None,
                           max_relevant_range_m: float = DEFAULT_MAX_RELEVANT_RANGE_M,
                           ) -> pd.DataFrame:
    """Targets table for one clip's feature dir (empty df when no labels)."""
    meta = json.loads((feature_dir / "meta.json").read_text())
    frames_path = next((feature_dir / f"frames.{ext}"
                        for ext in ("parquet", "csv")
                        if (feature_dir / f"frames.{ext}").exists()), None)
    if frames_path is None:
        raise FileNotFoundError(f"no frames table in {feature_dir}")
    frames = (pd.read_parquet(frames_path) if frames_path.suffix == ".parquet"
              else pd.read_csv(frames_path))

    clip_labels = [l for l in labels if l.clip_id == meta["clip_id"]]
    if not clip_labels:
        return pd.DataFrame()

    centers, step = _bin_centers(meta)
    tol = step / 2.0 if tolerance_deg is None else float(tolerance_deg)
    K = np.asarray(meta.get("fisheye_K"), dtype=np.float64)
    image_size = tuple(meta.get("image_size", [864, 648]))
    cam_h = float(meta.get("fisheye_camera_height_m", 0.27))

    # Resolve every label to a frame_index of this export. ts-keyed labels
    # match the frames table's RoundedTime; index-keyed match frame_index.
    ts_to_idx = {str(t): int(i) for t, i in
                 zip(frames["timestamp"], frames["frame_index"])}
    max_idx = int(frames["frame_index"].max())
    horizon_by_idx = {
        int(r.frame_index): (r.fisheye_horizon_slope,
                             r.fisheye_horizon_intercept,
                             r.fisheye_horizon_conf)
        for r in frames.itertuples()
        if "fisheye_horizon_slope" in frames.columns}

    # per-frame: list of (bearing, cls, relevant, range_or_nan)
    per_frame: dict[int, list[tuple[float, str, bool, float]]] = {}
    audited_by_idx: dict[int, bool] = {}
    for lab in clip_labels:
        if lab.frame_idx >= 0:
            idx = lab.frame_idx
        elif lab.frame_ts is not None and lab.frame_ts in ts_to_idx:
            idx = ts_to_idx[lab.frame_ts]
        else:
            continue
        if idx > max_idx:
            continue   # label beyond this export's frame range (--limit)
        entries = per_frame.setdefault(idx, [])
        audited_by_idx[idx] = audited_by_idx.get(idx, False) or lab.audited
        horizon = horizon_by_idx.get(idx)
        for bearing, bbox_norm, cls in label_bearings_px(
                lab, image_size[0], K):
            rng = label_mono_range(bbox_norm, K, image_size, horizon, cam_h)
            relevant = (max_relevant_range_m <= 0
                        or (rng is not None and rng <= max_relevant_range_m))
            entries.append((bearing, cls, relevant,
                            float(rng) if rng is not None else float("nan")))
        # Pure-thermal audit boxes (2026-09-12): bearing from the thermal
        # linear model (configs/intrinsics.yaml thermal.cx / pix_deg_ratio),
        # no monocular range (relevance = unbounded default).
        for tb in getattr(lab, "thermal_bboxes", []) or []:
            xyxy = tb.get("xyxy")
            if not xyxy or len(xyxy) != 4:
                continue
            cx_t, pdr_t = _thermal_linear_model()
            u = (float(xyxy[0]) + float(xyxy[2])) / 2.0 * THERMAL_WIDTH_PX
            entries.append(((u - cx_t) / pdr_t, str(tb.get("cls", "")),
                            max_relevant_range_m <= 0, float("nan")))
        if not lab.bboxes and not getattr(lab, "thermal_bboxes", None):
            # bbox-less record (e.g. heading-only or explicit empty frame):
            # keep the frame so its all-negative bins are learnable.
            per_frame.setdefault(idx, [])

    rows: list[dict] = []
    for idx in sorted(per_frame):
        for i, c in enumerate(centers):
            support_same, support_any = [], []
            for j in range(idx - dilation_frames, idx + dilation_frames + 1):
                for bearing, cls, relevant, rng in per_frame.get(j, []):
                    if abs(c - bearing) <= step / 2.0 + tol:
                        support_any.append((cls, relevant, rng))
                        if j == idx:
                            support_same.append((cls, relevant, rng))
            y = bool(support_any)
            relevant = any(r for _, r, _ in support_any) if support_any else True
            ranges = [rng for _, _, rng in support_same
                      if not math.isnan(rng)]
            rows.append({
                "clip_id": meta["clip_id"],
                "frame_index": idx,
                "bin_index": i,
                "bin_center_deg": float(c),
                "y_obstacle": int(y),
                "y_relevant": int(relevant),
                "y_nav": int(y and relevant),
                "from_dilation": int(bool(support_any) and not support_same),
                "label_classes": "|".join(sorted({cls for cls, _, _
                                                  in support_same if cls})),
                "label_min_range_m": (min(ranges) if ranges else float("nan")),
                "n_labels_frame": len(per_frame.get(idx, [])),
                "audited": int(audited_by_idx.get(idx, False)),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(FEATURES_ROOT))
    ap.add_argument("--clip", action="append", default=[],
                    help="clip_id filter (repeatable); default = every "
                         "feature dir that has labels")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="label JSONL paths (default: metrics defaults = "
                         "manual + labels/qwen/*.jsonl)")
    ap.add_argument("--include-unaudited", action="store_true",
                    help="use unaudited pseudo-labels too (bounded trust; "
                         "the audited column records the difference)")
    ap.add_argument("--dilation-frames", type=int,
                    default=DEFAULT_DILATION_FRAMES)
    ap.add_argument("--tolerance-deg", type=float, default=None,
                    help="bearing tolerance beyond the half-bin (default: "
                         "half a bin)")
    ap.add_argument("--max-relevant-range", type=float,
                    default=DEFAULT_MAX_RELEVANT_RANGE_M)
    args = ap.parse_args(argv)

    paths = ([Path(p) for p in args.labels] if args.labels
             else list(DEFAULT_LABELS))
    labels = load_labels(paths, include_unaudited=args.include_unaudited)
    if not labels:
        print("no labels loaded: check paths / --include-unaudited",
              file=sys.stderr)
        return 1

    n_ok = 0
    feature_root = Path(args.features)
    for feature_dir in sorted(p for p in feature_root.iterdir()
                              if p.is_dir() and (p / "meta.json").exists()):
        meta = json.loads((feature_dir / "meta.json").read_text())
        if args.clip and meta["clip_id"] not in args.clip:
            continue
        df = build_targets_for_clip(
            feature_dir, labels,
            dilation_frames=args.dilation_frames,
            tolerance_deg=args.tolerance_deg,
            max_relevant_range_m=args.max_relevant_range)
        if df.empty:
            continue
        out = feature_dir / "targets.csv"
        df.to_csv(out, index=False)
        n_ok += 1
        n_pos = int(df.y_obstacle.sum())
        n_nav = int(df.y_nav.sum())
        print(f"[ok] {meta['clip_id']}: {df.frame_index.nunique()} labelled "
              f"frames, {len(df)} rows, {n_pos} positive bins "
              f"({n_nav} nav-relevant) -> {out}")
    print(f"Wrote targets for {n_ok} clips.", file=sys.stderr)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
