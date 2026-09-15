"""Export fusion-scorer feature tables from triplets (Phase 0).

Runs the canonical ObstacleDetectionPipeline over a clip and writes, per
clip, under data/features/<scene>__<ts>/:

  - bins.csv|parquet    one row per (frame x bearing-bin): per-sensor
                        evidence + geometry + the incumbent rule-based
                        score (kept only as the baseline-to-beat).
  - frames.csv|parquet  one row per frame: context features (attitude,
                        luminance, camera exposure/gain (v4), thermal
                        quality, sun position, radar frame stats,
                        split/group assignment).
  - meta.json           schema + config provenance for the run.

Design rules (plan doc §4):
  - Bearing-space first: binning is a parameter of this exporter
    (defaults to the detection.yaml fusion bins), never baked upstream.
  - Missing modalities are first-class: NaN + *_available flags.
  - The incumbent n/3 score must be exactly recomputable from the hit
    columns: verified on every run (see check_score_reproduction).
  - Reserved columns for the deferred typed-YOLO channel are emitted
    now (all-NaN/False) so adding it later is a retrain, not a schema
    migration.

Sun position needs no GPS: RTC-backed clip timestamps + a fixed
lake fallback fix are enough (sun elevation varies <1 deg
across the lake). Once gps_<ts>.csv sidecars exist, pass --lat/--lon
from them (loader wiring is a follow-up alongside the capture change).

Usage:
    python -m scripts.fusion_model.build_features --triplet data/Boats/2025-06-23_16-21-07
    python -m scripts.fusion_model.build_features --all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.sensor_processing.fusion import _frame_id_for
from scripts.utils.detections import DetProvider
from scripts.sensor_processing.gps_boat1 import sun_position
from scripts.sensor_processing.imu_bno085 import load_imu_config
from scripts.sensor_processing.imu_replay import ImuLogAttitudeProvider
from scripts.sensor_processing.pipeline import (
    FrameResult,
    ObstacleDetectionPipeline,
    angle_to_bin,
    iterate_triplet,
    make_bins,
)
from scripts.sensor_processing.target_motion import targets_by_bin
from scripts.data.splits import SplitConfig, assign_split, group_key
from scripts.utils.association import pinhole_bearings
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.curation import load_curation
from scripts.utils.datasets import (REPO_ROOT, list_triplets, load_frames_csv,
                                    resolve_triplet)
from scripts.utils.geometry import UP_LEVEL, load_extrinsics, up_from_horizon_line
from scripts.utils.segmentation import (
    OBSTACLE as SEG_OBSTACLE,
    SegDetectParams,
    _clip_dirname,
    free_space_profile,
    horizon_from_water,
    seg_obstacle_detections,
)

# v3 (2026-08-24): per-bin target_* motion columns (motion.targets tracker;
# reserved-style: all-NaN/False/"" while the tracker is off, filled when on,
# so Phase-3 consumption is a retrain, not a schema migration).
# v4 (2026-08-28): per-frame camera exposure context from the frames_<ts>.csv
# sidecar (exposure_time_us, analogue_gain, digital_gain, lux_est; NaN on
# clips captured before the sidecar carried them). Auto-exposure at 3 fps
# turns near-darkness into a bright-looking 300 ms high-gain frame, so
# luminance alone cannot tell "well lit" from "long integration of dusk" --
# the scorer's context gate needs the camera's own numbers.
SCHEMA_VERSION = 4
FEATURES_ROOT = REPO_ROOT / "data" / "features"

#: frames_<ts>.csv exposure columns -> feature-table column names.
EXPOSURE_COLUMNS = {
    "ExposureTime": "exposure_time_us",
    "AnalogueGain": "analogue_gain",
    "DigitalGain": "digital_gain",
    "Lux": "lux_est",
}


def exposure_table(triplet) -> dict[int, dict[str, float]]:
    """frame_index -> exposure context from the clip's frames sidecar, or {}
    when the clip has no sidecar or a pre-exposure one (every value NaN
    downstream). frame_index here is the mp4 frame order, which is what
    iterate_triplet's enumeration walks."""
    frames_path = getattr(triplet, "frames", None)
    if not frames_path:
        return {}
    df = load_frames_csv(frames_path)
    present = [c for c in EXPOSURE_COLUMNS if c in df.columns]
    if df.empty or not present:
        return {}
    out: dict[int, dict[str, float]] = {}
    for rec in df[["frame_index", *present]].itertuples(index=False):
        if rec[0] != rec[0]:          # NaN frame_index: a torn sidecar row (power-cut chunk, 2026-09-08 21:36)
            continue
        out[int(rec[0])] = {EXPOSURE_COLUMNS[c]: float(v)
                            for c, v in zip(present, rec[1:])}
    return out

# Fallback fix for sun geometry when no GPS sidecar exists (all clips so
# far). InstitutionOne shoreline; anywhere on the lake changes solar
# elevation by well under a degree, so one fixed point serves the corpus.
DEFAULT_LAT = 46.0
DEFAULT_LON = 9.0
# Capture timestamps are the Pi system clock = local time (RTC-backed).
DEFAULT_TZ = "CET"


# ---------------------------------------------------------------------------
# Time + sun helpers
# ---------------------------------------------------------------------------

def frame_datetime_utc(clip_date: str, rounded_time: str, tz: str,
                       chunk_start_hms: str | None = None) -> datetime:
    """(clip date 'YYYY-MM-DD', RoundedTime 'HH:MM:SS.f', tz name) -> UTC.

    Capture timestamps are local wall-clock (the Pi's RTC-backed system
    clock). A clip that crosses midnight keeps the start date in its
    filename, so a frame time far *before* the chunk-start time means the
    clock wrapped: roll the date forward one day.
    """
    local = datetime.strptime(f"{clip_date} {rounded_time}",
                              "%Y-%m-%d %H:%M:%S.%f")
    if chunk_start_hms is not None:
        try:
            start = datetime.strptime(f"{clip_date} {chunk_start_hms}",
                                      "%Y-%m-%d %H-%M-%S")
            if (start - local) > timedelta(hours=12):
                local += timedelta(days=1)
        except ValueError:
            pass
    return local.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Bearing <-> column mapping (pinhole, matching the default bearing model)
# ---------------------------------------------------------------------------

def bearing_to_column(theta_deg: float, K: np.ndarray) -> float:
    """Azimuth (deg, right +) -> undistorted-image column, u = cx + fx*tan."""
    K = np.asarray(K, dtype=np.float64)
    return float(K[0, 2] + K[0, 0] * math.tan(math.radians(theta_deg)))


def bin_column_bands(edges: np.ndarray, K: np.ndarray, width: int
                     ) -> list[tuple[int, int] | None]:
    """Per bin: the [u0, u1) column band inside the image, or None when the
    bin lies entirely outside the frame."""
    bands: list[tuple[int, int] | None] = []
    for a, b in zip(edges[:-1], edges[1:]):
        u0 = int(round(bearing_to_column(float(a), K)))
        u1 = int(round(bearing_to_column(float(b), K)))
        u0, u1 = min(u0, u1), max(u0, u1)
        u0, u1 = max(u0, 0), min(u1, width)
        bands.append((u0, u1) if u1 > u0 else None)
    return bands


# ---------------------------------------------------------------------------
# Per-bin aggregation primitives
# ---------------------------------------------------------------------------

def _bin_indices(angles: list[float], edges: np.ndarray) -> list[int | None]:
    return [angle_to_bin(a, edges) for a in angles]


def _agg(values: list[float], how: str) -> float:
    if not values:
        return float("nan")
    if how == "min":
        return float(min(values))
    if how == "max":
        return float(max(values))
    if how == "mean":
        return float(np.mean(values))
    if how == "median":
        return float(np.median(values))
    raise ValueError(how)


def seg_bin_features(seg_mask: np.ndarray | None, edges: np.ndarray,
                     K: np.ndarray, detect_params: SegDetectParams,
                     ) -> dict[str, list[float]]:
    """Segmentation-derived e_fisheye evidence per bin (plan doc §1.3).

    - seg_obstacle_frac: OBSTACLE-pixel fraction of the bin's column band
      (pinhole bearing->column mapping, the default bearing model).
    - seg_comp_count / seg_comp_max_size_px: connected-component obstacle
      detections (seg_obstacle_detections) binned by pinhole bearing.
    All-NaN (and zero counts stay NaN) when there is no mask.
    """
    n_bins = len(edges) - 1
    nanl = [float("nan")] * n_bins
    if seg_mask is None:
        return {"seg_obstacle_frac": list(nanl),
                "seg_comp_count": list(nanl),
                "seg_comp_max_size_px": list(nanl)}

    h, w = seg_mask.shape[:2]
    obstacle = seg_mask == SEG_OBSTACLE
    frac: list[float] = []
    for band in bin_column_bands(edges, K, w):
        if band is None:
            frac.append(float("nan"))
        else:
            u0, u1 = band
            frac.append(float(obstacle[:, u0:u1].mean()))

    coords, sizes = seg_obstacle_detections(seg_mask, detect_params)
    counts = [0.0] * n_bins
    max_size = [float("nan")] * n_bins
    if coords:
        for idx, size in zip(_bin_indices(pinhole_bearings(coords, K), edges),
                             sizes):
            if idx is None:
                continue
            counts[idx] += 1.0
            if math.isnan(max_size[idx]) or size > max_size[idx]:
                max_size[idx] = float(size)
    return {"seg_obstacle_frac": frac,
            "seg_comp_count": counts,
            "seg_comp_max_size_px": max_size}


def detection_bin_features(res, edges: np.ndarray, prefix: str
                           ) -> dict[str, list[float]]:
    """Bin the configured detector's per-detection outputs (count, max
    size, min range/raw-range, votes, radar association evidence)."""
    n_bins = len(edges) - 1

    def _empty() -> dict[str, list[float]]:
        return {
            f"{prefix}_det_count": [0.0] * n_bins,
            f"{prefix}_det_max_size_px": [float("nan")] * n_bins,
            f"{prefix}_det_min_range_m": [float("nan")] * n_bins,
            f"{prefix}_det_min_raw_range_m": [float("nan")] * n_bins,
            f"{prefix}_det_votes": [0.0] * n_bins,
            f"{prefix}_det_radar_hits": [0.0] * n_bins,
            f"{prefix}_det_min_radar_range_m": [float("nan")] * n_bins,
        }

    out = _empty()
    if res is None or not res.angles:
        return out
    idxs = _bin_indices(res.angles, edges)

    def _get(seq, i):
        return seq[i] if seq and i < len(seq) else None

    for i, idx in enumerate(idxs):
        if idx is None:
            continue
        out[f"{prefix}_det_count"][idx] += 1.0
        size = _get(res.sizes, i)
        if size is not None:
            cur = out[f"{prefix}_det_max_size_px"][idx]
            if math.isnan(cur) or size > cur:
                out[f"{prefix}_det_max_size_px"][idx] = float(size)
        for col, seq in ((f"{prefix}_det_min_range_m", res.ranges),
                         (f"{prefix}_det_min_raw_range_m", res.raw_ranges),
                         (f"{prefix}_det_min_radar_range_m",
                          getattr(res, "radar_ranges", None))):
            v = _get(seq, i)
            if v is not None:
                cur = out[col][idx]
                if math.isnan(cur) or v < cur:
                    out[col][idx] = float(v)
        if res.votes and i < len(res.votes) and res.votes[i]:
            out[f"{prefix}_det_votes"][idx] += 1.0
        hits = _get(getattr(res, "radar_hits", None), i)
        if hits:
            out[f"{prefix}_det_radar_hits"][idx] += float(hits)
    return out


def radar_bin_features(m_res, edges: np.ndarray) -> dict[str, list[float]]:
    """Per-bin radar point-cloud evidence: counts, ranges, SNR/noise dB,
    Doppler coverage. NaN aggregates where the bin holds no points."""
    n_bins = len(edges) - 1
    per_bin: dict[str, list[list[float]]] = {
        "range": [[] for _ in range(n_bins)],
        "snr": [[] for _ in range(n_bins)],
        "noise": [[] for _ in range(n_bins)],
    }
    counts = [0.0] * n_bins
    n_doppler = [0.0] * n_bins
    if m_res is not None and m_res.angles:
        idxs = _bin_indices(m_res.angles, edges)
        for i, idx in enumerate(idxs):
            if idx is None:
                continue
            counts[idx] += 1.0
            per_bin["range"][idx].append(float(m_res.ranges[i]))
            if m_res.snr_db and i < len(m_res.snr_db) \
                    and m_res.snr_db[i] is not None:
                per_bin["snr"][idx].append(float(m_res.snr_db[i]))
            if m_res.noise_db and i < len(m_res.noise_db) \
                    and m_res.noise_db[i] is not None:
                per_bin["noise"][idx].append(float(m_res.noise_db[i]))
            if m_res.velocities_xy and i < len(m_res.velocities_xy) \
                    and m_res.velocities_xy[i] is not None:
                n_doppler[idx] += 1.0
    return {
        "radar_n_points": counts,
        "radar_min_range_m": [_agg(v, "min") for v in per_bin["range"]],
        "radar_median_range_m": [_agg(v, "median") for v in per_bin["range"]],
        "radar_max_snr_db": [_agg(v, "max") for v in per_bin["snr"]],
        "radar_mean_snr_db": [_agg(v, "mean") for v in per_bin["snr"]],
        "radar_median_noise_db": [_agg(v, "median") for v in per_bin["noise"]],
        "radar_n_velocity": n_doppler,
    }


# Canonical undistorted fisheye width; every canonical frame is 864 px
# (clip_overrides resizes odd captures to canonical at read time).
UNDISTORTED_W = 864


def yolo_bin_features(det_boxes, edges: np.ndarray,
                      K: np.ndarray) -> dict[str, list]:
    """Per-bin typed-detection evidence from data/det (DetProvider boxes).

    det_boxes None = the typed stream has no record for this frame
    (yolo_available False everywhere); [] = ran and saw nothing. Bearing
    uses the same pinhole convention as build_targets.label_bearings_px:
    atan2(u - cx, fx) on the undistorted image."""
    n_bins = len(edges) - 1
    out = {
        "yolo_available": [det_boxes is not None] * n_bins,
        "yolo_max_conf": [float("nan")] * n_bins,
        "yolo_top_cls": [""] * n_bins,
    }
    if not det_boxes:
        return out
    fx, cx = float(K[0, 0]), float(K[0, 2])
    for b in det_boxes:
        x0, _, x1, _ = b.get("xyxy", (0, 0, 0, 0))
        u = (x0 + x1) / 2.0 * UNDISTORTED_W
        bearing = math.degrees(math.atan2(u - cx, fx))
        idx = int(np.searchsorted(edges, bearing, side="right")) - 1
        if not 0 <= idx < n_bins:
            continue
        conf = float(b.get("confidence") or 0.0)
        prev = out["yolo_max_conf"][idx]
        if math.isnan(prev) or conf > prev:
            out["yolo_max_conf"][idx] = conf
            out["yolo_top_cls"][idx] = str(b.get("cls", ""))
    return out


def target_bin_features(targets_res, edges: np.ndarray) -> dict[str, list]:
    """Per-bin tracked-target motion evidence (motion.targets tracker).

    Nearest-range target per bin (targets_by_bin: matches the min_range
    semantics of every other per-bin aggregate). Tracker off / no target
    in the bin -> False/NaN/"" defaults, so the columns are shape-stable
    across config generations (the reserved-column pattern)."""
    n_bins = len(edges) - 1
    out: dict[str, list] = {
        "target_present": [False] * n_bins,
        "target_range_m": [float("nan")] * n_bins,
        "target_closing_mps": [float("nan")] * n_bins,
        "target_v_tangential_mps": [float("nan")] * n_bins,
        "target_speed_mps": [float("nan")] * n_bins,
        "target_cpa_m": [float("nan")] * n_bins,
        "target_t_cpa_s": [float("nan")] * n_bins,
        "target_age_frames": [float("nan")] * n_bins,
        "target_motion_state": [""] * n_bins,
    }
    if targets_res is None or not targets_res.targets:
        return out
    for i, t in enumerate(targets_by_bin(targets_res.targets, edges)):
        if t is None:
            continue
        out["target_present"][i] = True
        out["target_range_m"][i] = float(t.range_m)
        if t.closing_mps is not None:
            out["target_closing_mps"][i] = float(t.closing_mps)
        if t.v_tangential_mps is not None:
            out["target_v_tangential_mps"][i] = float(t.v_tangential_mps)
        if t.speed_mps is not None:
            out["target_speed_mps"][i] = float(t.speed_mps)
        if t.cpa_m is not None:
            out["target_cpa_m"][i] = float(t.cpa_m)
        if t.t_cpa_s is not None:
            out["target_t_cpa_s"][i] = float(t.t_cpa_s)
        out["target_age_frames"][i] = float(t.age_frames)
        out["target_motion_state"][i] = t.motion_state
    return out


# ---------------------------------------------------------------------------
# Free space (mirrors fusion._free_space_dist, but keeps the blocked flags)
# ---------------------------------------------------------------------------

def free_space_bin_features(result: FrameResult, intrinsics: dict,
                            detection: dict, camera_height_m: float,
                            edges: np.ndarray, attitude=None
                            ) -> dict[str, list]:
    n_bins = len(edges) - 1
    seg = result.fisheye.seg_mask if result.fisheye is not None else None
    if seg is None:
        return {"free_space_m": [float("nan")] * n_bins,
                "free_space_blocked": [False] * n_bins}
    intr = intrinsics["fisheye"]
    K = np.asarray(intr["K"], float)
    max_range = float((detection.get("range", {}) or {}).get("max_range_m", 15.0))
    if attitude is not None:
        up = attitude.up_vector_camera()
    else:
        s, b, conf = horizon_from_water(seg)
        up = up_from_horizon_line(s, b, K) if conf > 0.3 else UP_LEVEL
    prof = free_space_profile(seg, K, up, camera_height_m,
                              float(intr["cx"]), float(intr["pix_deg_ratio"]),
                              edges, max_range_m=max_range)
    return {"free_space_m": [float("nan") if d is None else float(d)
                             for d in prof.free_dist_m],
            "free_space_blocked": list(prof.blocked)}


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

def bin_rows_for_frame(result: FrameResult, frame_index: int, clip_id: str,
                       detection: dict, intrinsics: dict,
                       seg_detect_params: SegDetectParams,
                       camera_height_m: float, attitude=None
                       ,
                       det_boxes: list | None = None) -> list[dict]:
    """One feature dict per bearing bin for a processed frame."""
    fr = result.fusion
    edges, centers = fr.bin_edges, fr.bin_centers
    n_bins = len(centers)
    num_sensors = int(detection["fusion"].get("num_sensors", 3))

    K_f = np.asarray(intrinsics["fisheye"]["K"], float)
    seg = result.fisheye.seg_mask if result.fisheye is not None else None
    seg_f = seg_bin_features(seg, edges, K_f, seg_detect_params)
    fish_f = detection_bin_features(result.fisheye, edges, "fisheye")
    therm_f = detection_bin_features(result.thermal, edges, "thermal")
    radar_f = radar_bin_features(result.mmwave, edges)
    free_f = free_space_bin_features(result, intrinsics, detection,
                                     camera_height_m, edges, attitude)
    tgt_f = target_bin_features(getattr(result, "targets", None), edges)
    yolo_f = yolo_bin_features(det_boxes, np.asarray(edges), K_f)

    rows = []
    for i in range(n_bins):
        row = {
            "clip_id": clip_id,
            "timestamp": result.timestamp,
            "frame_index": frame_index,
            "bin_index": i,
            "bin_center_deg": float(centers[i]),
            "bin_left_deg": float(edges[i]),
            "bin_right_deg": float(edges[i + 1]),
            # Incumbent rule-based fusion (baseline-to-beat; the hit
            # columns are what the n/num_sensors score consumes: with
            # the default config hit_fisheye is the retired legacy blob
            # channel and is excluded from every model feature set).
            "hit_fisheye": bool(fr.sensor_hit_mask[i, 0]),
            "hit_thermal": bool(fr.sensor_hit_mask[i, 1]),
            "hit_mmwave": bool(fr.sensor_hit_mask[i, 2]),
            "num_sensors": num_sensors,
            "score_legacy": float(fr.scores[i]),
            "min_range_m": (float(fr.min_ranges[i])
                            if fr.min_ranges[i] is not None else float("nan")),
            "per_bin_velocity_mps": (
                float(fr.per_bin_velocity_mps[i])
                if fr.per_bin_velocity_mps
                and fr.per_bin_velocity_mps[i] is not None else float("nan")),
            "per_bin_ttc_s": (
                float(fr.per_bin_ttc_s[i])
                if fr.per_bin_ttc_s
                and fr.per_bin_ttc_s[i] is not None else float("nan")),
            "confirmed": bool(fr.confirmed[i]) if fr.confirmed else False,
            # Availability flags (missing modalities are first-class).
            "fisheye_available": result.fisheye is not None,
            "thermal_available": result.thermal is not None,
            "mmwave_available": result.mmwave is not None,
            "seg_available": seg is not None,
        }
        for feats in (seg_f, fish_f, therm_f, radar_f, free_f, tgt_f, yolo_f):
            for k, v in feats.items():
                row[k] = v[i]
        rows.append(row)
    return rows


def frame_row(result: FrameResult, frame_index: int, triplet, split: str,
              group: str, att, sun: tuple[float, float] | None,
              sun_meta: dict, raw_point_count: int,
              exposure: dict[str, float] | None = None) -> dict:
    f, t, m = result.fisheye, result.thermal, result.mmwave
    seg = f.seg_mask if f is not None else None
    exposure = exposure or {}
    row = {
        "clip_id": triplet.clip_id,
        "scene": triplet.scene,
        "timestamp": result.timestamp,
        "frame_index": frame_index,
        "split": split,
        "group_key": group,
        "fisheye_available": f is not None,
        "thermal_available": t is not None,
        "mmwave_available": m is not None,
        "imu_available": att is not None,
        "seg_available": seg is not None,
        # Fisheye context (darkness must come from the fisheye: the
        # Lepton AGC fakes healthy contrast on night water).
        "luminance_mean": f.luminance_mean if f else float("nan"),
        "luminance_p05": f.luminance_p05 if f else float("nan"),
        "luminance_p95": f.luminance_p95 if f else float("nan"),
        "is_dark": bool(f.is_dark) if f else False,
        # Camera exposure context (schema v4; NaN before the sidecar had it).
        **{col: float(exposure.get(col, float("nan")))
           for col in EXPOSURE_COLUMNS.values()},
        "fisheye_horizon_conf": float(f.horizon_line[2]) if f else float("nan"),
        # Full horizon line (slope, intercept): lets the target builder
        # compute label bbox mono-ranges with the same water-plane method
        # as the pipeline (schema v2).
        "fisheye_horizon_slope": float(f.horizon_line[0]) if f else float("nan"),
        "fisheye_horizon_intercept": (float(f.horizon_line[1])
                                      if f else float("nan")),
        "range_reference_fisheye": (f.range_reference or "") if f else "",
        # Thermal context.
        "thermal_quality_std": (t.quality_std if t and t.quality_std is not None
                                else float("nan")),
        "thermal_quality_dyn_range": (
            t.quality_dyn_range if t and t.quality_dyn_range is not None
            else float("nan")),
        "thermal_quality_ok": bool(t.quality_ok) if t else False,
        "thermal_horizon_conf": float(t.horizon_line[2]) if t else float("nan"),
        "range_reference_thermal": (t.range_reference or "") if t else "",
        # Attitude (IMU replay; chop stats are a rolling-window follow-up).
        "attitude_roll_deg": att.roll_deg if att else float("nan"),
        "attitude_pitch_deg": att.pitch_deg if att else float("nan"),
        "attitude_yaw_deg": att.yaw_deg if att else float("nan"),
        # Radar frame stats.
        "radar_n_points": len(m.points_xyz) if m is not None else 0,
        "radar_n_points_raw": raw_point_count,
        "radar_velocity_source": m.velocity_source if m is not None else "",
        "radar_has_doppler": bool(m and any(v is not None
                                            for v in m.velocities_xy)),
        "radar_has_snr": bool(m and m.snr_db),
        "radar_median_snr_db": _agg(
            [s for s in (m.snr_db if m else []) if s is not None], "median"),
    }
    if sun is not None:
        row["sun_elevation_deg"], row["sun_azimuth_deg"] = sun
    else:
        row["sun_elevation_deg"] = row["sun_azimuth_deg"] = float("nan")
    row.update(sun_meta)
    return row


# ---------------------------------------------------------------------------
# Integrity: the incumbent score must be recomputable from the table
# ---------------------------------------------------------------------------

def check_score_reproduction(bins_df: pd.DataFrame) -> None:
    """Raise unless score_legacy == (hits summed) / num_sensors, bit-exact.

    This proves the exporter is faithful to the pipeline; it is NOT an
    architecture-preservation goal (the n/3 score is the incumbent to
    beat, then delete)."""
    hits = (bins_df["hit_fisheye"].astype(float)
            + bins_df["hit_thermal"].astype(float)
            + bins_df["hit_mmwave"].astype(float))
    recomputed = hits / bins_df["num_sensors"].astype(float)
    if not (recomputed == bins_df["score_legacy"]).all():
        bad = int((recomputed != bins_df["score_legacy"]).sum())
        raise AssertionError(
            f"score_legacy not reproducible from hit columns on {bad} rows "
            ": feature exporter out of sync with the pipeline")


# ---------------------------------------------------------------------------
# Per-clip driver
# ---------------------------------------------------------------------------

def build_for_triplet(prefix, intrinsics=None, detection=None, *,
                      lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON,
                      tz: str = DEFAULT_TZ, use_imu: bool = True,
                      limit: int | None = None,
                      respect_curation: bool = True,
                      det_root=None,
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the pipeline over one clip -> (bins_df, frames_df, meta).

    Curation (configs/curation.yaml, scripts/utils/curation.py): a deleted
    set raises FileNotFoundError (the caller's skip path); frames inside a
    cut are dropped here, with `frame_index` staying the RAW video index so
    the exposure sidecar lookup and the ±7-frame spacing metric remain
    honest. `respect_curation=False` exports everything on disk."""
    # Thermal-dead chunks (e.g. the 2026-08-19 mid-session sensor death) are
    # real navigation data with the thermal column honestly absent -- exactly
    # the missing-evidence case the scorer must learn. Radar stays required.
    triplet = (prefix if hasattr(prefix, "clip_id")
               else resolve_triplet(prefix, require=("fisheye", "mmwave")))
    curation = load_curation() if respect_curation else None
    if curation is not None and curation.is_deleted(triplet.clip_id):
        raise FileNotFoundError(
            f"{triplet.clip_id} is deleted in configs/curation.yaml "
            "(--ignore-curation exports it anyway)")
    intrinsics = intrinsics or load_intrinsics()
    detection = detection or load_detection()

    attitude = None
    if use_imu:
        attitude = ImuLogAttitudeProvider.for_triplet(
            triplet, (load_imu_config().get("replay", {}) or {}))

    pipeline = ObstacleDetectionPipeline(intrinsics, detection,
                                         attitude_provider=attitude)
    seg_det = (detection.get("segmentation", {}) or {}).get("detect", {}) or {}
    seg_detect_params = SegDetectParams(
        min_area=int(seg_det.get("min_area", 25)),
        water_edge_margin_px=int(seg_det.get("water_edge_margin_px", 10)),
        max_components=int(seg_det.get("max_components", 100)),
    )
    extrinsics = load_extrinsics()
    fisheye_height = float((extrinsics.get("camera_height_m", {}) or {})
                           .get("fisheye", 0.27))

    cfg = SplitConfig.load()
    split = assign_split(triplet.clip_id, cfg)
    group = group_key(triplet.clip_id, cfg)
    clip_date = triplet.timestamp[:10]
    chunk_hms = triplet.timestamp.split("_")[-1]
    sun_meta = {"sun_lat_deg": lat, "sun_lon_deg": lon,
                "sun_fix_source": "fallback", "tz": tz}

    det_provider = DetProvider(Path(det_root) if det_root
                               else REPO_ROOT / "data" / "det")
    exposure_by_index = exposure_table(triplet)
    bin_rows: list[dict] = []
    frame_rows: list[dict] = []
    for frame_index, (ts, fish, therm, mm_pts) in enumerate(
            iterate_triplet(triplet, detection, respect_curation=False)):
        if limit is not None and frame_index >= limit:
            break
        if curation is not None and not curation.keep_frame(triplet.clip_id, ts):
            continue   # curation cut: raw frame_index keeps counting
        if attitude is not None:
            attitude.set_time(ts)
        fid = _frame_id_for(triplet.scene, triplet.timestamp, ts)
        result = pipeline.process_frame(fish, therm, mm_pts, timestamp=ts,
                                        frame_id=fid)
        att = attitude.get() if attitude is not None else None
        try:
            utc = frame_datetime_utc(clip_date, ts, tz, chunk_hms)
            sun = sun_position(lat, lon, utc)
        except ValueError:
            sun = None
        bin_rows.extend(bin_rows_for_frame(
            result, frame_index, triplet.clip_id, detection,
            intrinsics, seg_detect_params, fisheye_height, attitude=att,
            det_boxes=det_provider.get(fid)))
        frame_rows.append(frame_row(
            result, frame_index, triplet, split, group, att, sun, sun_meta,
            raw_point_count=len(mm_pts) if mm_pts is not None else 0,
            exposure=exposure_by_index.get(frame_index)))

    bins_df = pd.DataFrame(bin_rows)
    frames_df = pd.DataFrame(frame_rows)
    if not bins_df.empty:
        check_score_reproduction(bins_df)

    fusion_cfg = detection["fusion"]
    edges, _ = make_bins(fusion_cfg)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "clip_id": triplet.clip_id,
        "n_frames": len(frame_rows),
        "n_bin_rows": len(bin_rows),
        "bin_edges_deg": [float(e) for e in edges],
        "num_sensors": int(fusion_cfg.get("num_sensors", 3)),
        "fisheye_detector": detection["fisheye"].get("detector", "blob"),
        "bearing_model": {
            "fisheye": detection["fisheye"].get("bearing_model", "linear"),
            "thermal": detection["thermal"].get("bearing_model", "linear"),
        },
        "segmentation_enabled": bool(
            (detection.get("segmentation", {}) or {}).get("enabled", False)),
        "targets_enabled": bool(
            ((detection.get("motion", {}) or {}).get("targets", {}) or {})
            .get("enabled", False)),
        "imu_replayed": attitude is not None,
        "split": split,
        "group_key": group,
        "sun": sun_meta,
        # Geometry for downstream consumers (target builder computes label
        # bbox mono-ranges without re-loading configs; schema v2).
        "fisheye_K": np.asarray(intrinsics["fisheye"]["K"],
                                dtype=float).tolist(),
        "fisheye_camera_height_m": fisheye_height,
        "image_size": [864, 648],
    }
    return bins_df, frames_df, meta


def write_tables(bins_df: pd.DataFrame, frames_df: pd.DataFrame, meta: dict,
                 out_root: Path, fmt: str = "auto") -> Path:
    """Write bins/frames/meta under <out_root>/<scene>__<ts>/; returns dir.

    fmt: "parquet" (requires pyarrow/fastparquet), "csv", or "auto"
    (parquet if an engine is importable, else csv: the repo's default
    deps don't include one)."""
    scene, _, ts = meta["clip_id"].partition("/")
    out_dir = Path(out_root) / _clip_dirname(scene, ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(df: pd.DataFrame, stem: str) -> None:
        if fmt in ("parquet", "auto"):
            try:
                df.to_parquet(out_dir / f"{stem}.parquet", index=False)
                return
            except (ImportError, ValueError):
                if fmt == "parquet":
                    raise
        df.to_csv(out_dir / f"{stem}.csv", index=False)

    _write(bins_df, "bins")
    _write(frames_df, "frames")
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", action="append", default=[],
                    help="Triplet prefix (repeatable), e.g. "
                         "data/Boats/2025-06-23_16-21-07")
    ap.add_argument("--all", action="store_true",
                    help="Every resolvable triplet under data/ + captures/.")
    ap.add_argument("--out", default=str(FEATURES_ROOT),
                    help="Output root (default data/features/, gitignored).")
    ap.add_argument("--format", choices=["auto", "csv", "parquet"],
                    default="auto")
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--tz", default=DEFAULT_TZ)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip clips whose feature tables already exist under --out (resume a run)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max frames per clip (smoke runs).")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--intrinsics", default=None)
    ap.add_argument("--detection", default=None)
    ap.add_argument("--det-root", default=None,
                    help="typed-detection JSONL root for the yolo_* columns "
                         "(default data/det); e.g. a leak-free detector's "
                         "output for a typed-evidence ablation")
    ap.add_argument("--ignore-curation", action="store_true",
                    help="Export deleted sets and cut frames anyway "
                         "(configs/curation.yaml is honoured by default).")
    args = ap.parse_args(argv)

    intrinsics = load_intrinsics(args.intrinsics) if args.intrinsics \
        else load_intrinsics()
    detection = load_detection(args.detection) if args.detection \
        else load_detection()

    targets: list = list(args.triplet)
    if args.all:
        targets.extend(list_triplets(respect_curation=False)
                       if args.ignore_curation else list_triplets())
    if not targets:
        ap.error("give --triplet at least once, or --all")

    n_ok = 0
    for target in targets:
        if args.skip_existing and hasattr(target, "clip_id"):
            scene_, _, ts_ = target.clip_id.partition("/")
            done_dir = Path(args.out) / _clip_dirname(scene_, ts_)
            if any(done_dir.glob("bins.*")):
                print(f"[skip-existing] {target.clip_id}", file=sys.stderr)
                n_ok += 1
                continue
        try:
            bins_df, frames_df, meta = build_for_triplet(
                target, intrinsics, detection,
                lat=args.lat, lon=args.lon, tz=args.tz,
                use_imu=not args.no_imu, limit=args.limit,
                respect_curation=not args.ignore_curation,
                det_root=args.det_root)
        except (FileNotFoundError, RuntimeError) as exc:
            name = target.clip_id if hasattr(target, "clip_id") else target
            print(f"[skip] {name}: {exc}", file=sys.stderr)
            continue
        out_dir = write_tables(bins_df, frames_df, meta, Path(args.out),
                               fmt=args.format)
        n_ok += 1
        print(f"[ok] {meta['clip_id']}: {meta['n_frames']} frames -> "
              f"{out_dir} (split={meta['split']})")
    print(f"Built feature tables for {n_ok}/{len(targets)} clips.",
          file=sys.stderr)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
