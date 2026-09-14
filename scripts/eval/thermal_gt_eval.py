"""Thermal detector scoring against fisheye pseudo-GT (bearing space).

Scores the thermal path of the canonical pipeline against the fisheye
GroundingDINO labels (labels/qwen/det_2026-07-08.jsonl), reduced to bearing
intervals and clipped to the thermal camera's usable FOV, with the radar as
a secondary corroboration axis. Built for the 2026-07-09 thermal software
tuning session (docs/history/2026-07-09_thermal_tuning.md).

Bearing conventions (deliberate: CLAUDE.md §16):
  * GT bearings are recomputed from the bbox pixels with the PINHOLE model
    (undistorted fisheye keeps the original K as its camera matrix). The
    stored `obstacle_bins_fisheye` encode the legacy linear model and are
    NOT used.
  * Thermal detection bearings come from the pipeline itself (default
    `thermal.bearing_model: pinhole`), so GT and predictions share one
    bearing frame; residual cross-camera bias is ~2-3 deg (fisheye -3.4,
    thermal -1.0 vs radar), well inside the default 6 deg tolerance.

Matching (per frame):
  * A GT interval is RECALLED if any thermal detection bearing lands inside
    the interval widened by `tol_deg`.
  * A thermal detection is a TP if it lands inside any widened GT interval;
    otherwise RADAR-CORROBORATED if within `tol_deg` of a radar return's
    bearing that frame (physical evidence, not chargeable as FP);
    otherwise an FP.
  * Bin-grain scoring mirrors what fusion consumes: GT bins = fusion bins
    overlapped by a GT interval (thermal FOV only), predicted bins = bins
    containing a thermal detection.

Proxy-GT hazards are made visible, not hidden: recall is broken out
per-class and static-vs-moving (the mog2 lesson: a motion detector wins
naive proxies while dropping moored boats).

Usage:
    python -m scripts.eval.thermal_gt_eval \
        --triplet data/captures/2026-07-08/2026-07-08_16-37-01 \
        --labels labels/qwen/det_2026-07-08.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.utils.datasets import REPO_ROOT

# Setup phase of the 16-37-01 quad clip: AuthorOne handling the boat
# (clip_overrides scene_note). Excluded from scoring by default.
DEFAULT_SKIP_FRAMES = 19
# Thermal linear bearing model (configs/intrinsics.yaml thermal.cx /
# pix_deg_ratio; confirmed afloat 2026-08-19, HFOV 58 deg): used to turn
# pure-thermal audit boxes into bearings.
THERMAL_CX = 77.0
THERMAL_PXDEG = 2.82
DEFAULT_TOL_DEG = 6.0

# Static-vs-moving GT split: a 2-deg bearing cell is "static" when GT covers
# it in >= this fraction of scored frames; an interval is static when at
# least half its cells are. Dockside moored boats / pontoons qualify; the
# walking person and manoeuvring boats do not.
# NOTE the low fraction: GroundingDINO labels FLICKER: measured on the
# 16-37-01 clip, even permanently-visible dock structure is boxed in only
# ~25% of frames (max 2-deg-cell occupancy 0.27). 0.15 separates persistent
# geography from pass-through movers; it cannot be tightened until audited
# labels exist.
STATIC_CELL_DEG = 2.0
STATIC_FRAC = 0.15

# GT intervals wider than this (deg, after FOV clipping) are dropped as
# near-field/degenerate: measured: GroundingDINO "person" boxes spanning
# 70-100% of the frame width at 16:37:08-24 (AuthorOne handling the boat just
# past the documented 0-18 setup window). A 70-deg GT interval widened by
# the tolerance matches ANY detection and corrupts both recall and
# precision. Dropped count is reported by load_gt_intervals.
MAX_GT_WIDTH_DEG = 35.0


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

@dataclass
class GTInterval:
    cls: str
    lo_deg: float
    hi_deg: float
    static: bool = False

    def widened(self, tol: float) -> tuple[float, float]:
        return self.lo_deg - tol, self.hi_deg + tol


def fisheye_bearing_from_norm_x(x_norm: float, width: int, K: np.ndarray) -> float:
    """Pinhole bearing (deg, right +) of a normalised x on the undistorted fisheye."""
    fx, cx = float(K[0, 0]), float(K[0, 2])
    return float(np.degrees(np.arctan2(x_norm * width - cx, fx)))


def load_gt_intervals(
    labels_path: Path,
    clip_id: str,
    K_fisheye: np.ndarray,
    fov_lo: float,
    fov_hi: float,
    classes: tuple[str, ...] | None = None,
    max_width_deg: float = MAX_GT_WIDTH_DEG,
) -> dict[str, list[GTInterval]]:
    """frame_ts ('HH:MM:SS.f') -> GT bearing intervals clipped to the thermal FOV.

    Intervals wholly outside [fov_lo, fov_hi] are dropped (thermal cannot see
    them); partial overlaps are clipped to the FOV edge so an object half in
    view is only expected where it is visible. Intervals wider than
    `max_width_deg` are dropped as near-field/degenerate (see
    MAX_GT_WIDTH_DEG): the dropped count is printed.
    """
    n_dropped_wide = 0
    out: dict[str, list[GTInterval]] = {}
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scene, ts = rec["frame_id"].split("/")[:2]
            if f"{scene}/{ts}" != clip_id:
                continue
            w = int(rec["width"])
            intervals: list[GTInterval] = []
            for bb in rec.get("fisheye_bboxes", []):
                if classes and bb["cls"] not in classes:
                    continue
                x0, _, x1, _ = bb["xyxy"]
                lo = fisheye_bearing_from_norm_x(min(x0, x1), w, K_fisheye)
                hi = fisheye_bearing_from_norm_x(max(x0, x1), w, K_fisheye)
                if hi < fov_lo or lo > fov_hi:
                    continue
                lo_c, hi_c = max(lo, fov_lo), min(hi, fov_hi)
                if hi_c - lo_c > max_width_deg:
                    n_dropped_wide += 1
                    continue
                intervals.append(GTInterval(
                    cls=bb["cls"], lo_deg=lo_c, hi_deg=hi_c,
                ))
            # Pure-thermal audit boxes (2026-09-12): already in the thermal
            # frame; bearing straight from the linear model, no FOV clip
            # needed beyond the frame edge.
            tw = int(rec.get("thermal_width", 160))
            for bb in rec.get("thermal_bboxes", []):
                if classes and bb["cls"] not in classes:
                    continue
                x0, _, x1, _ = bb["xyxy"]
                lo = (min(x0, x1) * tw - THERMAL_CX) / THERMAL_PXDEG
                hi = (max(x0, x1) * tw - THERMAL_CX) / THERMAL_PXDEG
                intervals.append(GTInterval(cls=bb["cls"], lo_deg=lo, hi_deg=hi))
            out[rec["frame_ts"]] = intervals
    if n_dropped_wide:
        print(f"  GT: dropped {n_dropped_wide} degenerate intervals "
              f"(> {max_width_deg:.0f} deg wide: near-field boxes)")
    return out


def mark_static_intervals(gt_by_ts: dict[str, list[GTInterval]],
                          fov_lo: float, fov_hi: float) -> None:
    """Flag intervals as static via cross-frame bearing-cell occupancy."""
    n_frames = len(gt_by_ts)
    if n_frames == 0:
        return
    n_cells = max(1, int(np.ceil((fov_hi - fov_lo) / STATIC_CELL_DEG)))
    occupancy = np.zeros(n_cells, dtype=np.int64)

    def cells(iv: GTInterval) -> range:
        a = int((iv.lo_deg - fov_lo) / STATIC_CELL_DEG)
        b = int((iv.hi_deg - fov_lo) / STATIC_CELL_DEG)
        return range(max(0, a), min(n_cells - 1, b) + 1)

    for ivs in gt_by_ts.values():
        seen: set[int] = set()
        for iv in ivs:
            seen.update(cells(iv))
        for c in seen:
            occupancy[c] += 1

    static_cell = occupancy >= STATIC_FRAC * n_frames
    for ivs in gt_by_ts.values():
        for iv in ivs:
            cs = list(cells(iv))
            iv.static = bool(cs) and (
                float(np.count_nonzero(static_cell[cs])) / len(cs) >= 0.5)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ThermalGTMetrics:
    n_frames: int = 0
    n_gt: int = 0
    n_gt_recalled: int = 0
    n_det: int = 0
    n_det_tp: int = 0
    n_det_radar: int = 0            # unmatched by GT, corroborated by radar
    n_det_fp: int = 0
    sum_abs_bearing_err: float = 0.0   # matched detections, distance to interval
    n_bearing_err: int = 0
    per_class_gt: dict = field(default_factory=dict)       # cls -> [n, recalled]
    static_gt: list = field(default_factory=lambda: [0, 0])   # [n, recalled]
    moving_gt: list = field(default_factory=lambda: [0, 0])
    # Bin grain (what fusion consumes).
    bin_tp: int = 0
    bin_fp: int = 0
    bin_fn: int = 0

    # -- derived -----------------------------------------------------------
    def recall(self) -> float:
        return self.n_gt_recalled / self.n_gt if self.n_gt else float("nan")

    def precision_strict(self) -> float:
        d = self.n_det_tp + self.n_det_radar + self.n_det_fp
        return self.n_det_tp / d if d else float("nan")

    def precision_lenient(self) -> float:
        """Radar-corroborated detections are not chargeable."""
        d = self.n_det_tp + self.n_det_fp
        return self.n_det_tp / d if d else float("nan")

    def f1(self) -> float:
        p, r = self.precision_lenient(), self.recall()
        if not (p == p and r == r) or (p + r) == 0:   # NaN-safe
            return float("nan")
        return 2 * p * r / (p + r)

    def bearing_mae(self) -> float:
        return (self.sum_abs_bearing_err / self.n_bearing_err
                if self.n_bearing_err else float("nan"))

    def fp_per_frame(self) -> float:
        return self.n_det_fp / self.n_frames if self.n_frames else float("nan")

    def bin_precision(self) -> float:
        d = self.bin_tp + self.bin_fp
        return self.bin_tp / d if d else float("nan")

    def bin_recall(self) -> float:
        d = self.bin_tp + self.bin_fn
        return self.bin_tp / d if d else float("nan")

    def bin_f1(self) -> float:
        p, r = self.bin_precision(), self.bin_recall()
        if not (p == p and r == r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    def static_recall(self) -> float:
        n, rec = self.static_gt
        return rec / n if n else float("nan")

    def moving_recall(self) -> float:
        n, rec = self.moving_gt
        return rec / n if n else float("nan")

    def as_row(self) -> dict:
        return {
            "n_frames": self.n_frames, "n_gt": self.n_gt,
            "recall": round(self.recall(), 4),
            "recall_static": round(self.static_recall(), 4),
            "recall_moving": round(self.moving_recall(), 4),
            "precision_lenient": round(self.precision_lenient(), 4),
            "precision_strict": round(self.precision_strict(), 4),
            "f1": round(self.f1(), 4),
            "bearing_mae": round(self.bearing_mae(), 3),
            "fp_per_frame": round(self.fp_per_frame(), 3),
            "n_det": self.n_det, "n_det_tp": self.n_det_tp,
            "n_det_radar": self.n_det_radar, "n_det_fp": self.n_det_fp,
            "bin_p": round(self.bin_precision(), 4),
            "bin_r": round(self.bin_recall(), 4),
            "bin_f1": round(self.bin_f1(), 4),
            "per_class": {k: {"n": v[0], "recalled": v[1]}
                          for k, v in sorted(self.per_class_gt.items())},
        }


def _interval_distance(angle: float, iv: GTInterval) -> float:
    """0 inside the interval, else distance to the nearest edge (deg)."""
    if iv.lo_deg <= angle <= iv.hi_deg:
        return 0.0
    return min(abs(angle - iv.lo_deg), abs(angle - iv.hi_deg))


def score_frames(
    gt_by_ts: dict[str, list[GTInterval]],
    thermal_angles_by_ts: dict[str, list[float]],
    radar_angles_by_ts: dict[str, list[float]],
    tol_deg: float = DEFAULT_TOL_DEG,
    bin_edges: np.ndarray | None = None,
    fov: tuple[float, float] | None = None,
    dilate_frames: int = 0,
) -> ThermalGTMetrics:
    """Score one detector run. Frames = keys of thermal_angles_by_ts that
    also have a GT record (frames without labels are not scored).

    `dilate_frames`: temporal GT dilation for the PRECISION side only: a
    detection also counts as TP when it matches GT from any frame within
    +-dilate_frames. Rationale (measured, 2026-07-09): GroundingDINO labels
    flicker (persistent structure boxed in only ~25% of frames), so a
    correct persistent thermal detection is otherwise charged FP in the
    frames where the labeller forgot the box: 32% of the baseline run's
    charged FPs matched GT within +-7 frames. Recall stays per-frame
    (undilated): the labelled frames are a fair subsample for "did thermal
    see it too".
    """
    from scripts.sensor_processing.pipeline import angle_to_bin

    order = sorted(gt_by_ts.keys())
    ts_index = {t: i for i, t in enumerate(order)}

    def dilated(ts: str) -> list[GTInterval]:
        if not dilate_frames:
            return gt_by_ts[ts]
        i = ts_index[ts]
        return [iv
                for j in range(max(0, i - dilate_frames),
                               min(len(order), i + dilate_frames + 1))
                for iv in gt_by_ts[order[j]]]

    m = ThermalGTMetrics()
    for ts, det_angles in thermal_angles_by_ts.items():
        if ts not in gt_by_ts:
            continue
        ivs = gt_by_ts[ts]
        radar = radar_angles_by_ts.get(ts, [])
        m.n_frames += 1

        # GT recall.
        for iv in ivs:
            lo, hi = iv.widened(tol_deg)
            hit = any(lo <= a <= hi for a in det_angles)
            m.n_gt += 1
            m.n_gt_recalled += int(hit)
            cls_row = m.per_class_gt.setdefault(iv.cls, [0, 0])
            cls_row[0] += 1
            cls_row[1] += int(hit)
            row = m.static_gt if iv.static else m.moving_gt
            row[0] += 1
            row[1] += int(hit)

        # Detection precision (against temporally-dilated GT when enabled).
        ivs_p = dilated(ts)
        for a in det_angles:
            m.n_det += 1
            matched = [iv for iv in ivs_p
                       if iv.widened(tol_deg)[0] <= a <= iv.widened(tol_deg)[1]]
            if matched:
                m.n_det_tp += 1
                m.sum_abs_bearing_err += min(
                    _interval_distance(a, iv) for iv in matched)
                m.n_bearing_err += 1
            elif any(abs(a - ra) <= tol_deg for ra in radar):
                m.n_det_radar += 1
            else:
                m.n_det_fp += 1

        # Bin grain.
        if bin_edges is not None:
            gt_bins: set[int] = set()
            for iv in ivs:
                for b in range(len(bin_edges) - 1):
                    if iv.lo_deg < bin_edges[b + 1] and iv.hi_deg >= bin_edges[b]:
                        gt_bins.add(b)
            det_bins = {angle_to_bin(a, bin_edges) for a in det_angles}
            det_bins.discard(None)
            # Restrict both sides to bins inside the thermal FOV.
            if fov is not None:
                lo_f, hi_f = fov
                vis = {b for b in range(len(bin_edges) - 1)
                       if bin_edges[b + 1] > lo_f and bin_edges[b] < hi_f}
                gt_bins &= vis
                det_bins &= vis
            m.bin_tp += len(gt_bins & det_bins)
            m.bin_fp += len(det_bins - gt_bins)
            m.bin_fn += len(gt_bins - det_bins)
    return m


# ---------------------------------------------------------------------------
# Pipeline runner (baseline / single-config CLI)
# ---------------------------------------------------------------------------

def thermal_fov_deg(intrinsics: dict, bearing_model: str = "linear",
                    ) -> tuple[float, float, np.ndarray, tuple[int, int]]:
    """(lo, hi) usable thermal bearing span + P_thermal + undistorted (w, h).

    The span follows the CONFIGURED bearing model so GT clipping matches
    what process_thermal can actually report:
      linear: the datasheet geometry (cx=77 / 2.82 -> ~±28°). This is
                the validated model (2026-07-09 follow-up: the calibrated
                fx=63.8 is off ~2.3x; fx-ladder arbitration vs fisheye GT
                puts the truth at the datasheet 57° HFOV).
      pinhole: the calibrated-K span (~±53°), kept for A/B only.
    """
    import cv2  # noqa: F401 (undistort path)
    from scripts.utils.cv_common import pinhole_new_K, undistort_thermal

    K, D = intrinsics["thermal"]["K"], intrinsics["thermal"]["D"]
    dummy = np.zeros((120, 160, 3), dtype=np.uint8)
    und = undistort_thermal(dummy, K, D)
    h, w = und.shape[:2]
    P = pinhole_new_K(K, D, (w, h))
    if bearing_model == "pinhole":
        fx, cx = float(P[0, 0]), float(P[0, 2])
        lo = float(np.degrees(np.arctan2(0 - cx, fx)))
        hi = float(np.degrees(np.arctan2((w - 1) - cx, fx)))
    else:
        cx0 = float(intrinsics["thermal"]["cx"])
        pdr = float(intrinsics["thermal"]["pix_deg_ratio"])
        lo = (0 - cx0) / pdr
        hi = ((w - 1) - cx0) / pdr
    return lo, hi, P, (w, h)


def collect_run(
    triplet_prefix: str,
    detection: dict,
    intrinsics: dict,
    skip_frames: int = DEFAULT_SKIP_FRAMES,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, dict]]:
    """Run the pipeline's thermal + mmwave paths over a triplet.

    Returns (thermal_angles_by_ts, radar_angles_by_ts, thermal_stats_by_ts).
    Fisheye frames are ignored (not processed): the sweep objective is
    thermal-only; radar uses the current detection.yaml radar config.
    """
    from scripts.sensor_processing.pipeline import (
        ObstacleDetectionPipeline,
        iterate_triplet,
    )
    from scripts.utils.datasets import resolve_triplet

    triplet = resolve_triplet(triplet_prefix)
    pipeline = ObstacleDetectionPipeline(intrinsics, detection)
    thermal_by_ts: dict[str, list[float]] = {}
    radar_by_ts: dict[str, list[float]] = {}
    stats_by_ts: dict[str, dict] = {}
    for fidx, (ts, _fish, therm, mm_pts) in enumerate(
            iterate_triplet(triplet, detection)):
        t_res = pipeline.process_thermal(therm)
        m_res = pipeline.process_mmwave(mm_pts, timestamp=ts)
        if fidx < skip_frames:
            continue
        thermal_by_ts[ts] = list(t_res.angles)
        radar_by_ts[ts] = list(m_res.angles)
        stats_by_ts[ts] = {"std": t_res.quality_std,
                           "dyn": t_res.quality_dyn_range}
    return thermal_by_ts, radar_by_ts, stats_by_ts


def main():
    from scripts.sensor_processing.pipeline import make_bins
    from scripts.utils.calibration import load_detection, load_intrinsics

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet",
                    default="data/captures/2026-07-08/2026-07-08_16-37-01")
    ap.add_argument("--labels", default="labels/qwen/det_2026-07-08.jsonl")
    ap.add_argument("--detection", default=None,
                    help="Override path to detection.yaml (for A/B runs).")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_DEG)
    ap.add_argument("--dilate", type=int, default=0,
                    help="Temporal GT dilation (frames) for the precision "
                         "side: counters GroundingDINO label flicker.")
    ap.add_argument("--skip-frames", type=int, default=DEFAULT_SKIP_FRAMES)
    ap.add_argument("--json-out", default=None,
                    help="Write the metrics row as JSON here.")
    args = ap.parse_args()

    intrinsics = load_intrinsics()
    detection = load_detection(args.detection) if args.detection else load_detection()

    bearing_model = str(detection["thermal"].get("bearing_model", "linear"))
    fov_lo, fov_hi, _P, _wh = thermal_fov_deg(intrinsics, bearing_model)
    print(f"thermal usable FOV ({bearing_model}): "
          f"[{fov_lo:.1f}, {fov_hi:.1f}] deg")

    triplet_path = Path(args.triplet)
    # Nested per-boot sessions flatten their scene to <mission>_<sub>
    # (datasets.resolve_triplet), which is what the label frame_ids carry.
    try:
        from scripts.utils.datasets import resolve_triplet
        _t = resolve_triplet(str(triplet_path))
        clip_id = f"{_t.scene}/{_t.timestamp}"
    except Exception:
        clip_id = f"{triplet_path.parent.name}/{triplet_path.name}"
    gt = load_gt_intervals(REPO_ROOT / args.labels, clip_id,
                           intrinsics["fisheye"]["K"], fov_lo, fov_hi)
    mark_static_intervals(gt, fov_lo, fov_hi)
    n_iv = sum(len(v) for v in gt.values())
    n_static = sum(iv.static for v in gt.values() for iv in v)
    print(f"GT: {len(gt)} labelled frames, {n_iv} intervals in thermal FOV "
          f"({n_static} static, {n_iv - n_static} moving)")

    thermal_by_ts, radar_by_ts, _stats = collect_run(
        args.triplet, detection, intrinsics, skip_frames=args.skip_frames)
    edges, _centers = make_bins(detection["fusion"])
    m = score_frames(gt, thermal_by_ts, radar_by_ts, tol_deg=args.tol,
                     bin_edges=edges, fov=(fov_lo, fov_hi),
                     dilate_frames=args.dilate)
    row = m.as_row()
    print(json.dumps(row, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(row, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
