"""Quantitative evaluation: Phase I.4.6 of PLAN.md.

Loads labels (manual + audited Qwen), runs the canonical pipeline on
the corresponding frames, compares per-bin fused predictions against
label-derived ground truth, and writes a markdown report with per-scene
precision / recall / F1, per-sensor contribution, sector IoU, and the
OpenWater false-positive rate.

Usage:
    python -m scripts.eval.metrics
    python -m scripts.eval.metrics --include-unaudited --labels labels/qwen.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from scripts.sensor_processing.heading import (
    HeadingSmoother,
    recommend_heading,
)
from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    make_bins,
)
from scripts.sensor_processing.tracker import SectorTracker
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import REPO_ROOT, load_mmwave_csv, resolve_triplet

LABELS_DIR = REPO_ROOT / "labels"
RESULTS_DIR = REPO_ROOT / "results"
# Default label sources: the manual file plus every JSONL under labels/qwen/
# (qwen_*.jsonl Qwen3-VL runs, det_*.jsonl GroundingDINO runs). The flat
# labels/qwen.jsonl never existed on disk: the old default silently loaded
# manual.jsonl only. Kept in the list for forward compatibility (load_labels
# skips missing files); labels/qwen/cache/ and logs/ are dirs, not globbed.
DEFAULT_LABELS = [LABELS_DIR / "manual.jsonl", LABELS_DIR / "qwen.jsonl",
                  *sorted((LABELS_DIR / "qwen").glob("*.jsonl"))]


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

@dataclass
class Label:
    frame_id: str
    scene: str
    clip_id: str           # e.g. "Boats/2025-06-23_16-21-07"
    frame_idx: int
    source: str
    audited: bool
    bboxes: list[dict]
    obstacle_bins: list[int]
    image_size: tuple[int, int] | None = None  # (w, h)
    # PLAN.md §I.4.11: heading ground truth. `safe_heading_labelled`
    # marks "operator gave a number (possibly null)" vs. "this frame
    # was bbox-only labelled, ignore for heading metrics".
    safe_heading_labelled: bool = False
    safe_heading_deg: float | None = None
    # Timestamp-keyed labels (the dashboard's `ts=` frame_id scheme) carry
    # the radar RoundedTime here and frame_idx = -1 until _evaluate_clip
    # resolves it against the clip's timestamp list.
    frame_ts: str | None = None
    # Pure-thermal audit boxes (2026-09-12): normalised to the recorded
    # 160x120 Lepton frame; bearing via the thermal linear model, no range.
    thermal_bboxes: list[dict] = field(default_factory=list)


def _parse_frame_id(frame_id: str) -> tuple[str, str, int, str | None]:
    """Split a frame_id into (scene, ts, frame_idx, frame_ts).

    Two schemes coexist in labels/ (CLAUDE.md, dashboard._frame_id_for):
      - 'Boats/2025-06-23_16-21-07/000123'      -> frame_idx = 123
      - 'Boats/2025-06-23_16-21-07/ts=12-41-23.0' -> frame_ts = '12:41:23.0',
        frame_idx = -1 (resolved to an index per clip in _evaluate_clip).
    """
    parts = frame_id.split("/")
    if len(parts) != 3:
        raise ValueError(f"unexpected frame_id: {frame_id}")
    scene, ts, fidx = parts
    if fidx.startswith("ts="):
        return scene, ts, -1, fidx[3:].replace("-", ":")
    return scene, ts, int(fidx.lstrip("0") or "0"), None


def load_labels(paths: list[Path], include_unaudited: bool = False) -> list[Label]:
    out: list[Label] = []
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not obj.get("audited", False) and not include_unaudited:
                    continue
                scene, ts, fidx, fts = _parse_frame_id(obj["frame_id"])
                head_labelled = "safe_heading_deg" in obj
                out.append(Label(
                    frame_id=obj["frame_id"],
                    scene=scene,
                    clip_id=f"{scene}/{ts}",
                    frame_idx=fidx,
                    frame_ts=fts,
                    source=obj.get("source", "unknown"),
                    audited=obj.get("audited", False),
                    bboxes=obj.get("fisheye_bboxes", []),
                    thermal_bboxes=obj.get("thermal_bboxes", []),
                    obstacle_bins=obj.get("obstacle_bins_fisheye", []),
                    image_size=(obj.get("width", 0), obj.get("height", 0)) or None,
                    safe_heading_labelled=head_labelled,
                    safe_heading_deg=obj.get("safe_heading_deg"),
                ))
    return out


# ---------------------------------------------------------------------------
# Per-bin comparison
# ---------------------------------------------------------------------------

@dataclass
class Counters:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, predicted: bool, actual: bool):
        if predicted and actual:
            self.tp += 1
        elif predicted and not actual:
            self.fp += 1
        elif not predicted and actual:
            self.fn += 1
        else:
            self.tn += 1

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Comparison core
# ---------------------------------------------------------------------------

@dataclass
class HeadingCounters:
    """PLAN.md §I.4.11 / §I.6: heading evaluation accumulators.

    A frame is counted once per (raw, smoothed) prediction track, so
    sums of n_compared + n_recommender_abstained + n_we_recommended_when_no_safe
    equal the total of heading-labelled frames.
    """
    n_labelled: int = 0           # heading was labelled at all
    n_label_null: int = 0         # operator labelled "no safe direction"
    # Compared-against-non-null-label-and-non-null-prediction:
    n_compared_raw: int = 0
    sum_abs_err_raw: float = 0.0
    n_within_15_raw: int = 0
    n_compared_smoothed: int = 0
    sum_abs_err_smoothed: float = 0.0
    n_within_15_smoothed: int = 0
    # Abstention calibration: how the predictions handle "label is null".
    n_correct_abstain_raw: int = 0           # label null, pred null
    n_false_recommend_raw: int = 0           # label null, pred not null  → over-confident
    n_missed_recommend_raw: int = 0          # label not null, pred null  → over-cautious
    n_correct_abstain_smoothed: int = 0
    n_false_recommend_smoothed: int = 0
    n_missed_recommend_smoothed: int = 0

    def add_frame(self, label_deg, raw_deg, smoothed_deg):
        self.n_labelled += 1
        if label_deg is None:
            self.n_label_null += 1
            if raw_deg is None:
                self.n_correct_abstain_raw += 1
            else:
                self.n_false_recommend_raw += 1
            if smoothed_deg is None:
                self.n_correct_abstain_smoothed += 1
            else:
                self.n_false_recommend_smoothed += 1
            return
        if raw_deg is None:
            self.n_missed_recommend_raw += 1
        else:
            err = abs(raw_deg - label_deg)
            self.sum_abs_err_raw += err
            self.n_compared_raw += 1
            if err <= 15.0:
                self.n_within_15_raw += 1
        if smoothed_deg is None:
            self.n_missed_recommend_smoothed += 1
        else:
            err = abs(smoothed_deg - label_deg)
            self.sum_abs_err_smoothed += err
            self.n_compared_smoothed += 1
            if err <= 15.0:
                self.n_within_15_smoothed += 1

    def mae_raw(self) -> float:
        return (self.sum_abs_err_raw / self.n_compared_raw
                if self.n_compared_raw else float("nan"))

    def mae_smoothed(self) -> float:
        return (self.sum_abs_err_smoothed / self.n_compared_smoothed
                if self.n_compared_smoothed else float("nan"))

    def within15_raw(self) -> float:
        return (self.n_within_15_raw / self.n_compared_raw
                if self.n_compared_raw else float("nan"))

    def within15_smoothed(self) -> float:
        return (self.n_within_15_smoothed / self.n_compared_smoothed
                if self.n_compared_smoothed else float("nan"))


@dataclass
class ClipResult:
    clip_id: str
    scene: str
    n_frames_labelled: int = 0
    fused: Counters = field(default_factory=Counters)
    fisheye_only: Counters = field(default_factory=Counters)
    thermal_only: Counters = field(default_factory=Counters)
    mmwave_only: Counters = field(default_factory=Counters)
    sector_iou_sum: float = 0.0
    fp_bins_total: int = 0
    fp_frames_total: int = 0
    confusion: dict[tuple[int, int], int] = field(default_factory=dict)
    heading: HeadingCounters = field(default_factory=HeadingCounters)


def _evaluate_clip(labels: list[Label], pipeline_factory, intrinsics, detection,
                   tracker_factory=None) -> ClipResult:
    """labels is the set of labels for a single clip, sorted by frame_idx.

    `tracker_factory`, if provided, is a zero-arg callable returning a
    fresh SectorTracker per clip. When set, per-bin predictions are gated
    through the tracker, mirroring `fusion.py --track`.
    """
    clip_id = labels[0].clip_id
    scene = labels[0].scene
    try:
        triplet = resolve_triplet(REPO_ROOT / "data" / clip_id)
    except FileNotFoundError:
        # Field captures live under data/captures/<mission>/ (their clip_id
        # is "<mission>/<ts>", e.g. the institutionone_day1 qwen-labelled clips).
        triplet = resolve_triplet(REPO_ROOT / "data" / "captures" / clip_id)

    fisheye_cap = cv2.VideoCapture(str(triplet.fisheye))
    thermal_cap = cv2.VideoCapture(str(triplet.thermal))
    if not fisheye_cap.isOpened() or not thermal_cap.isOpened():
        fisheye_cap.release()
        thermal_cap.release()
        raise RuntimeError(f"cannot open videos for {clip_id}")

    df = load_mmwave_csv(triplet.mmwave)
    grouped = df.groupby("RoundedTime")
    timestamps = sorted(grouped.groups.keys())

    # Timestamp-keyed labels (`ts=` scheme) resolve to frame indices via the
    # clip's radar timestamp order: the same index pairing iterate_triplet
    # and the dashboard use. Clips with an EMPTY radar CSV (e.g. the day-1
    # regression captures) fall back to iterate_triplet's synthesized
    # wall-clock timeline: base = filename HH:MM:SS, one frame per 1/3 s:
    # invert that to recover the index. Unresolvable labels are dropped
    # with a note.
    from scripts.sensor_processing.pipeline import (
        _format_hhmmss,
        _wallclock_seconds_from_timestamp,
    )
    ts_to_idx = {t: i for i, t in enumerate(timestamps)}
    base_s = _wallclock_seconds_from_timestamp(triplet.timestamp)

    def _resolve_ts(fts: str) -> int | None:
        if fts in ts_to_idx:
            return ts_to_idx[fts]
        if timestamps:
            return None
        try:
            h, m, s = fts.split(":")
            secs = int(h) * 3600 + int(m) * 60 + float(s)
        except ValueError:
            return None
        idx = int(round((secs - base_s) * 3.0))
        # Round-trip check against the synthesized formatting.
        if idx < 0 or _format_hhmmss(base_s + idx / 3.0) != fts:
            return None
        return idx

    dropped = 0
    for lbl in labels:
        if lbl.frame_idx < 0 and lbl.frame_ts is not None:
            idx = _resolve_ts(lbl.frame_ts)
            if idx is None:
                dropped += 1
            else:
                lbl.frame_idx = idx
    labels = [lbl for lbl in labels if lbl.frame_idx >= 0]
    if dropped:
        print(f"  [{clip_id}] {dropped} ts-keyed labels not in the radar "
              f"timeline: dropped")
    if not labels:
        fisheye_cap.release()
        thermal_cap.release()
        raise RuntimeError(f"no resolvable labels for {clip_id}")
    target_indices = sorted({lbl.frame_idx for lbl in labels})
    label_by_idx = {lbl.frame_idx: lbl for lbl in labels}

    pipeline = pipeline_factory()
    tracker = tracker_factory() if tracker_factory else None
    edges, centers = make_bins(detection["fusion"])
    threshold = float(detection["fusion"].get("hit_threshold", 0.33))

    head_cfg = detection.get("heading", {}) or {}
    smoother = HeadingSmoother(
        window_n=int(head_cfg.get("smoothing_window", 9)),
        min_samples=int(head_cfg.get("smoothing_min_samples", 3)),
    )

    result = ClipResult(clip_id=clip_id, scene=scene)
    max_target = target_indices[-1]

    frame_idx = 0
    while frame_idx <= max_target:
        ret_f, fish = fisheye_cap.read()
        ret_t, therm = thermal_cap.read()
        if not ret_f or not ret_t:
            break
        # Get radar points for this index if a matching timestamp exists.
        if frame_idx < len(timestamps):
            group = grouped.get_group(timestamps[frame_idx])
            mm_pts = group[["X", "Y", "Z"]].values.astype(np.float64)
        else:
            mm_pts = np.empty((0, 3))

        # Always step the pipeline so thermal running mean stays correct.
        # frame_id lets the SegProvider find per-frame water masks (the
        # radar timestamp keys the mask filenames, same scheme as fusion.py;
        # empty-radar clips use the synthesized wall-clock timeline).
        if frame_idx < len(timestamps):
            ts = timestamps[frame_idx]
        elif not timestamps:
            ts = _format_hhmmss(base_s + frame_idx / 3.0)
        else:
            ts = None
        fid = f"{clip_id}/ts={ts.replace(':', '-')}" if ts is not None else None
        res = pipeline.process_frame(fish, therm, mm_pts, timestamp=ts,
                                     frame_id=fid)

        # Gate scores through the tracker if enabled.
        scores = res.fusion.scores
        if tracker is not None:
            hits = [s >= threshold for s in scores]
            tracker.update(hits)
            scores = tracker.gated_scores(scores)

        # Heading is computed on *every* frame so the smoother sees an
        # in-order stream and the smoothed predictions at labelled
        # frames match what fusion.py --out would have emitted.
        n_bins = len(centers)
        v_list = list(res.fusion.per_bin_velocity_mps or [])
        t_list = list(res.fusion.per_bin_ttc_s or [])
        if len(v_list) != n_bins:
            v_list = [None] * n_bins
        if len(t_list) != n_bins:
            t_list = [None] * n_bins
        head_rec = recommend_heading(
            bin_centers_deg=list(centers),
            scores=list(scores),
            min_range_m=list(res.fusion.min_ranges),
            per_bin_velocity_mps=v_list,
            per_bin_ttc_s=t_list,
            config=head_cfg,
            current_heading_deg=0.0,   # IMU stub: PLAN.md §IV.1 #14
        )
        smoothed_deg = smoother.update(head_rec.recommended_heading_deg)

        if frame_idx in label_by_idx:
            lbl = label_by_idx[frame_idx]
            actual = set(int(b) for b in lbl.obstacle_bins)
            predicted = {int(centers[i]) for i, s in enumerate(scores) if s >= threshold}
            fisheye_pred = {int(centers[i]) for i in range(len(centers))
                            if res.fusion.sensor_hit_mask[i, 0]}
            thermal_pred = {int(centers[i]) for i in range(len(centers))
                            if res.fusion.sensor_hit_mask[i, 1]}
            mmwave_pred = {int(centers[i]) for i in range(len(centers))
                           if res.fusion.sensor_hit_mask[i, 2]}

            for c in centers:
                key = int(c)
                result.fused.add(key in predicted, key in actual)
                result.fisheye_only.add(key in fisheye_pred, key in actual)
                result.thermal_only.add(key in thermal_pred, key in actual)
                result.mmwave_only.add(key in mmwave_pred, key in actual)

            # Sector IoU on the fused prediction.
            union = predicted | actual
            inter = predicted & actual
            result.sector_iou_sum += (len(inter) / len(union)) if union else 1.0

            # FP bins / frames bookkeeping (anything predicted but not in label).
            fps = predicted - actual
            result.fp_bins_total += len(fps)
            if fps:
                result.fp_frames_total += 1

            # Confusion: for each actual bin, which (closest) predicted bin was hit?
            for ab in actual:
                if not predicted:
                    continue
                nearest = min(predicted, key=lambda pb: abs(pb - ab))
                key = (ab, nearest)
                result.confusion[key] = result.confusion.get(key, 0) + 1

            if lbl.safe_heading_labelled:
                result.heading.add_frame(
                    lbl.safe_heading_deg,
                    head_rec.recommended_heading_deg,
                    smoothed_deg,
                )

            result.n_frames_labelled += 1

        frame_idx += 1

    fisheye_cap.release()
    thermal_cap.release()
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _scene_table(scene: str, clip_results: list[ClipResult],
                 frame_rate_fps: float) -> str:
    fused = Counters()
    fisheye = Counters()
    thermal = Counters()
    mmwave = Counters()
    iou_sum = 0.0
    iou_n = 0
    fp_bins = 0
    fp_frames = 0
    n_frames = 0
    for cr in clip_results:
        for c, agg in ((cr.fused, fused), (cr.fisheye_only, fisheye),
                       (cr.thermal_only, thermal), (cr.mmwave_only, mmwave)):
            agg.tp += c.tp; agg.fp += c.fp; agg.fn += c.fn; agg.tn += c.tn
        iou_sum += cr.sector_iou_sum
        iou_n += cr.n_frames_labelled
        fp_bins += cr.fp_bins_total
        fp_frames += cr.fp_frames_total
        n_frames += cr.n_frames_labelled

    iou = iou_sum / iou_n if iou_n else float("nan")
    fp_per_min = (fp_bins / n_frames * frame_rate_fps * 60) if n_frames else float("nan")
    lines = [
        f"### {scene}  (n={n_frames} labelled frames across {len(clip_results)} clips)",
        "",
        "| sensor  | P     | R     | F1    | TP | FP | FN |",
        "|---------|-------|-------|-------|----|----|----|",
        f"| fused   | {fused.precision():.3f} | {fused.recall():.3f} | {fused.f1():.3f} | {fused.tp} | {fused.fp} | {fused.fn} |",
        f"| fisheye | {fisheye.precision():.3f} | {fisheye.recall():.3f} | {fisheye.f1():.3f} | {fisheye.tp} | {fisheye.fp} | {fisheye.fn} |",
        f"| thermal | {thermal.precision():.3f} | {thermal.recall():.3f} | {thermal.f1():.3f} | {thermal.tp} | {thermal.fp} | {thermal.fn} |",
        f"| mmwave  | {mmwave.precision():.3f} | {mmwave.recall():.3f} | {mmwave.f1():.3f} | {mmwave.tp} | {mmwave.fp} | {mmwave.fn} |",
        "",
        f"- sector IoU (fused): **{iou:.3f}**",
        f"- FP bins per frame (fused): **{fp_bins/n_frames if n_frames else float('nan'):.3f}**",
        f"- FP bins per minute @ {frame_rate_fps:g} fps: **{fp_per_min:.2f}**",
        "",
    ]
    return "\n".join(lines)


def _heading_table(scene: str, clip_results: list[ClipResult]) -> str:
    """Aggregate per-scene heading metrics across clips.

    Returns the empty string when no heading labels exist for the scene;
    keeps the report clean during the bbox-only labelling phase.
    """
    agg = HeadingCounters()
    for cr in clip_results:
        h = cr.heading
        agg.n_labelled += h.n_labelled
        agg.n_label_null += h.n_label_null
        agg.n_compared_raw += h.n_compared_raw
        agg.sum_abs_err_raw += h.sum_abs_err_raw
        agg.n_within_15_raw += h.n_within_15_raw
        agg.n_compared_smoothed += h.n_compared_smoothed
        agg.sum_abs_err_smoothed += h.sum_abs_err_smoothed
        agg.n_within_15_smoothed += h.n_within_15_smoothed
        agg.n_correct_abstain_raw += h.n_correct_abstain_raw
        agg.n_false_recommend_raw += h.n_false_recommend_raw
        agg.n_missed_recommend_raw += h.n_missed_recommend_raw
        agg.n_correct_abstain_smoothed += h.n_correct_abstain_smoothed
        agg.n_false_recommend_smoothed += h.n_false_recommend_smoothed
        agg.n_missed_recommend_smoothed += h.n_missed_recommend_smoothed

    if agg.n_labelled == 0:
        return ""

    def _fmt(x: float) -> str:
        return f"{x:.3f}" if not math.isnan(x) else "—"

    def _pct(x: float) -> str:
        return f"{x:.1%}" if not math.isnan(x) else "—"

    lines = [
        f"### {scene} heading  (n={agg.n_labelled} labelled frames, "
        f"{agg.n_label_null} labelled as 'no safe direction')",
        "",
        "| signal      | MAE   | within ±15° | n compared |",
        "|-------------|-------|-------------|------------|",
        f"| raw         | {_fmt(agg.mae_raw())} | {_pct(agg.within15_raw())} | {agg.n_compared_raw} |",
        f"| smoothed    | {_fmt(agg.mae_smoothed())} | {_pct(agg.within15_smoothed())} | {agg.n_compared_smoothed} |",
        "",
        "**Abstain calibration** (against labelled 'no safe direction'):",
        "",
        "| signal      | correct abstain | false recommend | missed recommend |",
        "|-------------|-----------------|-----------------|------------------|",
        f"| raw         | {agg.n_correct_abstain_raw} | {agg.n_false_recommend_raw} | {agg.n_missed_recommend_raw} |",
        f"| smoothed    | {agg.n_correct_abstain_smoothed} | {agg.n_false_recommend_smoothed} | {agg.n_missed_recommend_smoothed} |",
        "",
    ]
    return "\n".join(lines)


def _confusion_block(clip_results: list[ClipResult]) -> str:
    agg: dict[tuple[int, int], int] = {}
    for cr in clip_results:
        for k, v in cr.confusion.items():
            agg[k] = agg.get(k, 0) + v
    if not agg:
        return ""
    actual_bins = sorted({a for a, _ in agg.keys()})
    pred_bins = sorted({p for _, p in agg.keys()})
    lines = ["### Confusion (actual bin -> closest predicted bin, fused)", ""]
    head = "| actual \\ pred |" + " ".join(f" {p:+d} |" for p in pred_bins)
    sep = "|" + " --- |" * (len(pred_bins) + 1)
    lines.append(head)
    lines.append(sep)
    for a in actual_bins:
        row = f"| {a:+d} |" + " ".join(f" {agg.get((a, p), 0)} |" for p in pred_bins)
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="JSONL label files (defaults to labels/{manual,qwen}.jsonl).")
    ap.add_argument("--include-unaudited", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--frame-rate", type=float, default=3.0,
                    help="Capture FPS for FP-per-minute calculation.")
    ap.add_argument("--track", action="store_true",
                    help="Gate per-bin predictions through a SectorTracker, "
                         "mirroring fusion.py --track.")
    ap.add_argument("--track-min-hits", type=int, default=3)
    ap.add_argument("--track-max-age", type=int, default=5)
    ap.add_argument("--detection", default=None,
                    help="Override path to detection.yaml (for A/B runs).")
    ap.add_argument("--seg", action="store_true",
                    help="Force segmentation on (masks under data/seg/).")
    ap.add_argument("--seg-fp-filter", action="store_true",
                    help="With --seg: enable segmentation.fp_filter.")
    ap.add_argument("--assoc", action="store_true",
                    help="Enable fusion.association (radar-ranged detections).")
    args = ap.parse_args()

    label_paths = [Path(p) for p in args.labels] if args.labels else DEFAULT_LABELS
    labels = load_labels(label_paths, include_unaudited=args.include_unaudited)
    if not labels:
        raise SystemExit(f"No labels found in {[str(p) for p in label_paths]}")

    by_clip: dict[str, list[Label]] = defaultdict(list)
    for lbl in labels:
        by_clip[lbl.clip_id].append(lbl)

    intrinsics = load_intrinsics()
    detection = load_detection(args.detection) if args.detection else load_detection()
    if args.seg:
        detection.setdefault("segmentation", {})["enabled"] = True
        if args.seg_fp_filter:
            detection["segmentation"].setdefault("fp_filter", {})["enabled"] = True
    if args.assoc:
        detection.setdefault("fusion", {}).setdefault(
            "association", {})["enabled"] = True

    def make_pipeline():
        return ObstacleDetectionPipeline(intrinsics, detection)

    n_bins = len(make_bins(detection["fusion"])[1])
    def make_tracker():
        return SectorTracker(n_bins=n_bins,
                             min_hits=args.track_min_hits,
                             max_age=args.track_max_age)
    tracker_factory = make_tracker if args.track else None

    clip_results: list[ClipResult] = []
    for clip_id, clip_labels in tqdm(by_clip.items(), desc="clips"):
        try:
            cr = _evaluate_clip(clip_labels, make_pipeline, intrinsics, detection,
                                tracker_factory=tracker_factory)
            clip_results.append(cr)
        except Exception as e:
            print(f"!! clip {clip_id} failed: {e}")

    # Group by scene.
    by_scene: dict[str, list[ClipResult]] = defaultdict(list)
    for cr in clip_results:
        by_scene[cr.scene].append(cr)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"metrics_{run_id}.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# metrics_{run_id}",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- labels: {', '.join(str(p) for p in label_paths)}",
        f"- audited only: {not args.include_unaudited}",
        f"- detection config: configs/detection.yaml",
        f"- intrinsics: configs/intrinsics.yaml",
        f"- tracking: {'on (min_hits=' + str(args.track_min_hits) + ', max_age=' + str(args.track_max_age) + ')' if args.track else 'off'}",
        f"- total labelled frames: {sum(cr.n_frames_labelled for cr in clip_results)}",
        "",
    ]
    for scene in sorted(by_scene):
        lines.append(_scene_table(scene, by_scene[scene], args.frame_rate))
    # Heading metrics (PLAN.md §I.4.11). One section per scene; the
    # function returns "" for scenes without heading labels so the
    # report stays clean during the bbox-only phase.
    heading_lines: list[str] = []
    for scene in sorted(by_scene):
        block = _heading_table(scene, by_scene[scene])
        if block:
            heading_lines.append(block)
    if heading_lines:
        lines.append("## Heading evaluation (§I.4.11)")
        lines.append("")
        lines.extend(heading_lines)
    lines.append(_confusion_block(clip_results))

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
