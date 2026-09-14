# ****************************************************************************
# *  Sensor-fusion driver: per-clip iterate the (fisheye, thermal, mmwave)
# *  triplet and emit per-frame angular-sector obstacle scores.
# *
# *  Algorithm and bin layout match AuthorZero's original; implementation has
# *  been refactored to drive a canonical ObstacleDetectionPipeline (see
# *  pipeline.py) so the same code path is used by viewer / sweep / metrics.
# *
# *  Usage:
# *    python -m scripts.sensor_processing.fusion \
# *        --triplet data/Boats/2025-06-23_16-21-07 \
# *        --out results/sectors_boats_2025-06-23_16-21-07.jsonl

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.sensor_processing.heading import HeadingSmoother, recommend_heading
from scripts.sensor_processing.imu_bno085 import load_imu_config
from scripts.sensor_processing.imu_replay import ImuLogAttitudeProvider
from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    iterate_triplet,
    make_bins,
)
from scripts.sensor_processing.target_motion import (
    target_to_dict,
    targets_by_bin,
)
from scripts.sensor_processing.tracker import SectorTracker
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import resolve_triplet
from scripts.utils.geometry import UP_LEVEL, load_extrinsics, up_from_horizon_line
from scripts.utils.segmentation import (
    FreeSpaceSmoother,
    free_space_profile,
    horizon_from_water,
)

PROTOCOL_VERSION = 1


def _frame_id_for(scene: str, triplet_ts: str, frame_ts: str) -> str:
    """Stable frame identifier: same scheme as dashboard._frame_id_for, so
    SegProvider finds the per-frame masks under data/seg/."""
    return f"{scene}/{triplet_ts}/ts={frame_ts.replace(':', '-')}"


def _free_space_dist(res, intrinsics, detection, camera_height_m, attitude=None):
    """Per-bin navigable free-space distance from the frame's water mask.

    Same computation the dashboard runs, with one upgrade: a replayed/live IMU
    attitude, when available, supplies the up-vector ahead of the water-edge
    line (matching the pipeline's range priority). None when no mask.
    """
    seg = res.fisheye.seg_mask if res.fisheye is not None else None
    if seg is None:
        return None
    intr = intrinsics["fisheye"]
    K = np.asarray(intr["K"], float)
    f = detection["fusion"]
    edges = np.arange(f["bin_min_deg"], f["bin_max_deg"] + f["bin_step_deg"],
                      f["bin_step_deg"])
    max_range = float((detection.get("range", {}) or {}).get("max_range_m", 15.0))
    if attitude is not None:
        up = attitude.up_vector_camera()
    else:
        s, b, conf = horizon_from_water(seg)
        up = up_from_horizon_line(s, b, K) if conf > 0.3 else UP_LEVEL
    prof = free_space_profile(seg, K, up, camera_height_m,
                              float(intr["cx"]), float(intr["pix_deg_ratio"]),
                              edges, max_range_m=max_range)
    return prof.free_dist_m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", required=True,
                    help="Triplet prefix, e.g. data/Boats/2025-06-23_16-21-07")
    ap.add_argument("--out", default=None,
                    help="Optional path to write per-frame sector JSONL.")
    ap.add_argument("--print-every", type=int, default=10,
                    help="Print a summary line every N frames (0 = silent).")
    ap.add_argument("--intrinsics", default=None,
                    help="Override path to intrinsics.yaml.")
    ap.add_argument("--detection", default=None,
                    help="Override path to detection.yaml.")
    ap.add_argument("--track", action="store_true",
                    help="Gate per-bin scores through a SectorTracker.")
    ap.add_argument("--track-min-hits", type=int, default=3)
    ap.add_argument("--track-max-age", type=int, default=5)
    ap.add_argument("--current-heading-deg", type=float, default=None,
                    help="Own-boat heading at clip start (deg from bow). "
                         "When unavailable, the heading recommender treats "
                         "bow as 0° (see PLAN.md §IV.1 question 14).")
    ap.add_argument("--no-heading", action="store_true",
                    help="Skip heading recommendation; emit v0 schema only.")
    ap.add_argument("--no-imu", action="store_true",
                    help="Ignore the clip's imu_<ts>.csv sidecar even when "
                         "present (range falls back to horizon/level).")
    ap.add_argument("--seg", action="store_true",
                    help="Force segmentation on (masks under data/seg/): "
                         "seg range + water-edge horizon; equivalent to "
                         "segmentation.enabled=true in detection.yaml.")
    ap.add_argument("--seg-fp-filter", action="store_true",
                    help="With --seg: drop blob detections that sit (almost) "
                         "entirely in water (glint/reflection FPs).")
    ap.add_argument("--free-space-heading", action="store_true",
                    help="With --seg: feed the per-bearing navigable free-"
                         "space into the heading threat field "
                         "(heading.use_free_space).")
    ap.add_argument("--assoc", action="store_true",
                    help="Pixel-level radar<->camera association: confirmed "
                         "bins + radar-ranged detections "
                         "(fusion.association.enabled=true).")
    ap.add_argument("--targets", action="store_true",
                    help="Per-target motion estimation (motion.targets."
                         "enabled=true): tracked radar objects with 2-D "
                         "velocity, closing/crossing/diverging state and "
                         "CPA, emitted as the additive `targets` JSONL "
                         "field (protocol v1.2).")
    ap.add_argument("--thermal-profile", default=None,
                    choices=["auto", "day", "twilight", "night"],
                    help="Enable the thermal sun-elevation profile "
                         "scheduler (thermal.profiles). 'auto' resolves "
                         "the profile per frame from the clip's computed "
                         "sun elevation; a profile name forces it for the "
                         "whole run (replay A/B). Default: whatever "
                         "detection.yaml says (scheduler off).")
    ap.add_argument("--identity", action="store_true",
                    help="Cross-sensor track identity (motion.targets."
                         "identity.enabled=true; implies --targets and "
                         "--assoc): camera tracks paired with radar target "
                         "ids; each target gains the additive "
                         "camera_track_id JSONL field (protocol v1.3).")
    ap.add_argument("--camera-rate", action="store_true",
                    help="With --identity (implied): paired camera bearings "
                         "as a second observation for the target bearing-"
                         "rate fit + camera sustain through radar dropouts "
                         "(motion.targets.camera_bearing_rate=true).")
    ap.add_argument("--bearing", choices=["linear", "pinhole"], default=None,
                    help="Override the bearing model for BOTH cameras "
                         "(default: detection.yaml per-sensor setting).")
    ap.add_argument("--vote-gate", type=float, default=None, metavar="M",
                    help="Enable range-gated voting: camera detections whose "
                         "raw range reads beyond M metres don't raise bin "
                         "scores (fusion.vote_range_gate).")
    ap.add_argument("--scorer", default=None, metavar="MODEL.json",
                    help="SHADOW MODE: load a trained gated-mixture scorer "
                         "(models/fusion_scorer_v1a.json) and emit per-bin "
                         "p_obstacle + threat alongside the legacy fields "
                         "in the JSONL. Additive, default off; the legacy "
                         "scores are untouched.")
    ap.add_argument("--scorer-smoothing", type=float, default=None,
                    metavar="TAU_S",
                    help="With --scorer: per-bin temporal persistence for "
                         "the calibrated score with this decay time "
                         "constant (seconds); emits the additive "
                         "p_obstacle_smooth field. Rises are instant, only "
                         "the decay is damped. Equivalent to "
                         "fusion.scorer_smoothing.enabled=true with "
                         "decay_time_s=TAU_S.")
    ap.add_argument("--nav", action="store_true",
                    help="Navigation profile: everything --seg --seg-fp-filter "
                         "--free-space-heading --assoc enables, PLUS the "
                         "segmentation-mode fisheye detector (detections only "
                         "inside the water region) and --vote-gate 30. "
                         "Threat semantics tighten to the actionable envelope; "
                         "perception recall of far shore objects is "
                         "deliberately excluded (2026-07-09 A/B, PLAN.md).")
    args = ap.parse_args()

    triplet = resolve_triplet(args.triplet, require=("fisheye", "mmwave"))
    intrinsics = load_intrinsics(args.intrinsics) if args.intrinsics else load_intrinsics()
    detection = load_detection(args.detection) if args.detection else load_detection()
    if args.nav:
        args.seg = True
        args.seg_fp_filter = True
        args.free_space_heading = True
        args.assoc = True
        if args.vote_gate is None:
            args.vote_gate = 30.0
        detection["fisheye"]["detector"] = "segmentation"
    if args.seg:
        detection.setdefault("segmentation", {})["enabled"] = True
        if args.seg_fp_filter:
            detection["segmentation"].setdefault("fp_filter", {})["enabled"] = True
    if args.free_space_heading:
        detection.setdefault("heading", {})["use_free_space"] = True
    if args.camera_rate:
        args.identity = True
    if args.identity:
        args.targets = True
        args.assoc = True
    if args.assoc:
        detection.setdefault("fusion", {}).setdefault(
            "association", {})["enabled"] = True
    if args.targets:
        detection.setdefault("motion", {}).setdefault(
            "targets", {})["enabled"] = True
    if args.identity:
        detection["motion"]["targets"].setdefault(
            "identity", {})["enabled"] = True
    if args.camera_rate:
        detection["motion"]["targets"]["camera_bearing_rate"] = True
    identity_on = bool((((detection.get("motion", {}) or {})
                         .get("targets", {}) or {})
                        .get("identity", {}) or {}).get("enabled", False))
    if args.bearing:
        detection["fisheye"]["bearing_model"] = args.bearing
        detection["thermal"]["bearing_model"] = args.bearing
    if args.thermal_profile:
        prof = detection["thermal"].setdefault("profiles", {})
        prof["enabled"] = True
        if args.thermal_profile != "auto":
            prof["force"] = args.thermal_profile
    if args.vote_gate is not None:
        detection.setdefault("fusion", {})["vote_range_gate"] = {
            "enabled": True, "max_vote_range_m": float(args.vote_gate)}

    # 4th stream: replayed IMU attitude (imu_<ts>.csv sidecar, captured over
    # BLE from boat1 since 2026-07-08). Feeds the monocular water-plane range
    # exactly like a live BNO085; absent/empty logs leave behaviour unchanged.
    attitude = None
    if not args.no_imu:
        attitude = ImuLogAttitudeProvider.for_triplet(
            triplet, (load_imu_config().get("replay", {}) or {}))
        if attitude is not None:
            print(f"[imu] replaying {len(attitude)} attitude samples from "
                  f"{triplet.imu.name}", file=sys.stderr)

    pipeline = ObstacleDetectionPipeline(intrinsics, detection,
                                         attitude_provider=attitude)

    # Thermal profile scheduler (thermal.profiles / --thermal-profile):
    # compute the frame's sun elevation the same way build_features does
    # (clip-local wall clock -> UTC -> offline solar geometry). Position:
    # the clip's gps_<ts>.csv sidecar when present, else the fixed
    # lake point (solar elevation varies well under a degree
    # across the lake). Lazy imports keep the default path light.
    sun_elev_for = None
    profiles_on = bool((detection["thermal"].get("profiles") or {})
                       .get("enabled", False))
    if profiles_on:
        from scripts.fusion_model.build_features import (
            DEFAULT_LAT,
            DEFAULT_LON,
            DEFAULT_TZ,
            frame_datetime_utc,
        )
        from scripts.sensor_processing.gps_boat1 import sun_position

        lat, lon = DEFAULT_LAT, DEFAULT_LON
        if triplet.gps is not None:
            try:
                from scripts.utils.datasets import load_gps_csv
                g = load_gps_csv(triplet.gps)
                if len(g):
                    lat, lon = float(g.iloc[0]["Lat"]), float(g.iloc[0]["Lon"])
            except Exception as exc:      # noqa: BLE001 - sidecar is optional
                print(f"[thermal.profiles] gps sidecar unreadable "
                      f"({exc}); using the fixed point", file=sys.stderr)
        _pdate = triplet.timestamp[:10]
        _phms = triplet.timestamp.split("_")[-1]

        def sun_elev_for(frame_ts: str) -> float | None:
            try:
                el, _ = sun_position(
                    lat, lon,
                    frame_datetime_utc(_pdate, frame_ts, DEFAULT_TZ, _phms))
                return el
            except ValueError:
                return None

    tracker = None
    if args.track:
        _, centers = make_bins(detection["fusion"])
        tracker = SectorTracker(n_bins=len(centers),
                                 min_hits=args.track_min_hits,
                                 max_age=args.track_max_age)
    hit_threshold = float(detection["fusion"].get("hit_threshold", 0.33))

    out_fp = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_fp = open(args.out, "w")

    heading_cfg = detection.get("heading", {}) or {}
    emit_heading = not args.no_heading
    smoother = HeadingSmoother(
        window_n=int(heading_cfg.get("smoothing_window", 9)),
        min_samples=int(heading_cfg.get("smoothing_min_samples", 3)),
    )
    use_free_space = bool(heading_cfg.get("use_free_space", False))
    extrinsics = load_extrinsics()
    fisheye_height = float((extrinsics.get("camera_height_m", {}) or {})
                           .get("fisheye", 0.27))
    freespace_smoother = FreeSpaceSmoother(
        alpha=float(heading_cfg.get("free_space_smoothing_alpha", 0.5)),
        max_range_m=float((detection.get("range", {}) or {}).get("max_range_m", 15.0)),
    )

    # Shadow scorer (plan Phase 2/4: log the learned score alongside the
    # incumbent for N outings before anything consumes it). Lazy imports:
    # the default path stays pandas-free.
    scorer = None
    if args.scorer:
        import pandas as pd

        from scripts.fusion_model.build_features import (
            DEFAULT_LAT,
            DEFAULT_LON,
            DEFAULT_TZ,
            bin_rows_for_frame,
            frame_datetime_utc,
        )
        from scripts.fusion_model.models import (
            load_any_scorer,
            load_fusion_model_config,
            prepare_matrix,
            threat_from_score,
        )
        from scripts.sensor_processing.gps_boat1 import sun_position
        from scripts.utils.segmentation import SegDetectParams

        scorer = load_any_scorer(args.scorer)
        fm_cfg = load_fusion_model_config()
        # Typed-evidence artefacts read per-frame boxes from data/det; the
        # provider is lazy per clip file, so this is free when absent.
        det_provider = None
        if "yolo" in tuple(scorer.sensors):
            from scripts.utils.detections import DetProvider
            from scripts.utils.datasets import REPO_ROOT as _RR
            det_provider = DetProvider(_RR / "data" / "det")
        _seg_det = (detection.get("segmentation", {}) or {}).get("detect", {}) or {}
        scorer_seg_params = SegDetectParams(
            min_area=int(_seg_det.get("min_area", 25)),
            water_edge_margin_px=int(_seg_det.get("water_edge_margin_px", 10)),
            max_components=int(_seg_det.get("max_components", 100)),
        )
        _clip_date = triplet.timestamp[:10]
        _chunk_hms = triplet.timestamp.split("_")[-1]
        print(f"[scorer] shadow-scoring with {args.scorer}", file=sys.stderr)

    # Per-bin temporal persistence for the calibrated score (additive
    # p_obstacle_smooth; fusion.scorer_smoothing, default OFF = the JSONL is
    # byte-identical). Instant rise / exponential decay: see
    # scripts/fusion_model/smoothing.py.
    score_smoother = None
    if scorer is not None:
        sm_cfg = dict((detection.get("fusion", {}) or {})
                      .get("scorer_smoothing", {}) or {})
        if args.scorer_smoothing is not None:
            sm_cfg["enabled"] = True
            sm_cfg["decay_time_s"] = args.scorer_smoothing
        if bool(sm_cfg.get("enabled", False)):
            from scripts.fusion_model.smoothing import ScorePersistence
            score_smoother = ScorePersistence(
                decay_time_s=float(sm_cfg.get("decay_time_s", 1.0)),
                gap_reset_s=float(sm_cfg.get("gap_reset_s", 5.0)))
            print(f"[scorer] smoothing p_obstacle: decay_time_s="
                  f"{score_smoother.tau}, gap_reset_s={score_smoother.gap}",
                  file=sys.stderr)

    n_frames = 0
    try:
        for ts, fish, therm, mm_pts in iterate_triplet(triplet, detection):
            if attitude is not None:
                attitude.set_time(ts)
            if sun_elev_for is not None:
                pipeline.set_sun_elevation(sun_elev_for(ts))
            fid = _frame_id_for(triplet.scene, triplet.timestamp, ts)
            result = pipeline.process_frame(fish, therm, mm_pts, timestamp=ts,
                                            frame_id=fid)
            fr = result.fusion
            n_frames += 1
            att = attitude.get() if attitude is not None else None
            free_dist = None
            if use_free_space:
                fd = _free_space_dist(result, intrinsics, detection,
                                      fisheye_height, attitude=att)
                if fd is not None:
                    free_dist = freespace_smoother.update(fd)

            scores = fr.scores
            if tracker is not None:
                hits = [s >= hit_threshold for s in scores]
                tracker.update(hits)
                scores = tracker.gated_scores(fr.scores)

            if args.print_every and n_frames % args.print_every == 0:
                _print_summary(ts, fr.bin_centers, scores, fr.min_ranges)

            if out_fp is not None:
                rec = {
                    "protocol": PROTOCOL_VERSION if emit_heading else 0,
                    "timestamp": ts,
                    "clip_id": triplet.clip_id,
                    "bin_centers_deg": fr.bin_centers.tolist(),
                    "scores": scores,
                    "min_range_m": [float(r) if r is not None else None for r in fr.min_ranges],
                    "sensor_hits": fr.sensor_hit_mask.astype(int).tolist(),
                    "tracked": tracker is not None,
                }
                # Replayed IMU attitude (v1 extension; null when no sample
                # within the staleness window). Yaw is the boat's absolute
                # heading reference: the recommended heading below stays
                # bow-relative; a consumer adds yaw to get compass.
                rec["attitude"] = (
                    {"roll_deg": att.roll_deg, "pitch_deg": att.pitch_deg,
                     "yaw_deg": att.yaw_deg}
                    if att is not None else None)
                if free_dist is not None:
                    rec["free_space_m"] = [
                        float(d) if d is not None else None for d in free_dist]
                if args.assoc:
                    # v1 extension: object-level cross-sensor confirmation per
                    # bin (radar returns projected inside a camera detection).
                    rec["confirmed"] = list(fr.confirmed)
                if profiles_on:
                    # Additive: which thermal parameter profile the
                    # sun-elevation scheduler had in force this frame
                    # (null when the frame carried no thermal).
                    rec["thermal_profile"] = (
                        result.thermal.profile
                        if result.thermal is not None else None)
                if result.targets is not None:
                    # v1.2 extension: per-target motion states (2-D velocity,
                    # closing/crossing/diverging, CPA). Additive; consumers
                    # of the per-bin schema are unaffected.
                    rec["targets"] = [
                        target_to_dict(t, include_identity=identity_on)
                        for t in result.targets.targets]
                if scorer is not None:
                    # Shadow fields (additive; consumers of the legacy
                    # schema are unaffected). Same feature construction as
                    # build_features + train: one code path, no drift.
                    sdf = pd.DataFrame(bin_rows_for_frame(
                        result, n_frames - 1, triplet.clip_id, detection,
                        intrinsics, scorer_seg_params, fisheye_height,
                        attitude=att,
                        det_boxes=(det_provider.get(fid)
                                   if det_provider is not None else None)))
                    try:
                        _elev, _ = sun_position(
                            DEFAULT_LAT, DEFAULT_LON,
                            frame_datetime_utc(_clip_date, ts, DEFAULT_TZ,
                                               _chunk_hms))
                    except ValueError:
                        _elev = float("nan")
                    f_res = result.fisheye
                    sdf["sun_elevation_deg"] = _elev
                    sdf["luminance_mean"] = (
                        f_res.luminance_mean if f_res else float("nan"))
                    sdf["is_dark"] = bool(f_res.is_dark) if f_res else False
                    sdf["thermal_quality_ok"] = (
                        bool(result.thermal.quality_ok)
                        if result.thermal else False)
                    sdf["imu_available"] = att is not None
                    if result.targets is not None:
                        from scripts.fusion_model.build_features import (
                            target_bin_features)
                        bin_edges = np.append(
                            sdf["bin_left_deg"].to_numpy(),
                            sdf["bin_right_deg"].iloc[-1])
                        for k, v in target_bin_features(
                                result.targets, bin_edges).items():
                            sdf[k] = v
                    if hasattr(scorer, "predict_proba_calibrated_df"):
                        # v1b GBT bundle: consumes the bins-table row frame
                        # (sdf carries GBT_COLUMNS natively) — no e/c
                        # factoring for this scorer.
                        p_shadow = scorer.predict_proba_calibrated_df(sdf)
                    else:
                        e_m, c_m, _ = prepare_matrix(
                            sdf, fm_cfg,
                            columns=tuple(scorer.context_columns),
                            sensors=tuple(scorer.sensors))
                        p_shadow = scorer.predict_proba_calibrated(e_m, c_m)
                    # Per-bin CPA arrays from the target tracker (NaN = no
                    # tracked target in the bin). Feeds the urgency motion
                    # term, itself gated by fusion_model.yaml urgency.motion.
                    cpa_arr = tcpa_arr = None
                    if result.targets is not None:
                        per_bin_t = targets_by_bin(result.targets.targets,
                                                   fr.bin_edges)
                        cpa_arr = np.array(
                            [t.cpa_m if t is not None and t.cpa_m is not None
                             else np.nan for t in per_bin_t])
                        tcpa_arr = np.array(
                            [t.t_cpa_s if t is not None
                             and t.t_cpa_s is not None
                             else np.nan for t in per_bin_t])
                    threat = threat_from_score(
                        p_shadow, sdf["min_range_m"].to_numpy(dtype=float),
                        sdf["per_bin_ttc_s"].to_numpy(dtype=float),
                        severity=1.0, cfg=fm_cfg,
                        cpa_m=cpa_arr, t_cpa_s=tcpa_arr)
                    rec["p_obstacle"] = [round(float(v), 4) for v in p_shadow]
                    if score_smoother is not None:
                        # v1.2 additive: persistence-smoothed score. Fed the
                        # unrounded probabilities; rises are instant so the
                        # onset frame matches p_obstacle exactly.
                        rec["p_obstacle_smooth"] = [
                            round(float(v), 4)
                            for v in score_smoother.update(ts, p_shadow)]
                    rec["threat"] = [round(float(v), 4) for v in threat]
                if emit_heading:
                    # v1 schema additions.
                    rec["per_bin_velocity_mps"] = [
                        float(v) if v is not None else None
                        for v in (fr.per_bin_velocity_mps or [])
                    ]
                    rec["per_bin_ttc_s"] = [
                        float(t) if t is not None else None
                        for t in (fr.per_bin_ttc_s or [])
                    ]
                    heading = recommend_heading(
                        bin_centers_deg=list(fr.bin_centers),
                        scores=scores,
                        min_range_m=list(fr.min_ranges),
                        per_bin_velocity_mps=list(fr.per_bin_velocity_mps or []),
                        per_bin_ttc_s=list(fr.per_bin_ttc_s or []),
                        config=heading_cfg,
                        current_heading_deg=args.current_heading_deg,
                        free_dist_m=free_dist,
                    )
                    rec["recommended_heading_deg"] = heading.recommended_heading_deg
                    rec["heading_cost_curve"] = heading.cost_curve
                    rec["heading_candidates_deg"] = heading.candidate_headings_deg
                    rec["heading_reason"] = heading.reason
                    rec["smoothed_heading_deg"] = smoother.update(
                        heading.recommended_heading_deg
                    )
                out_fp.write(json.dumps(rec) + "\n")
        print(f"Processed {n_frames} frames.", file=sys.stderr)
    finally:
        if out_fp is not None:
            out_fp.close()


def _print_summary(ts, centers, scores, min_ranges):
    cells = []
    for c, s, r in zip(centers, scores, min_ranges):
        cells.append(f"{int(c):+3d}:{s:.2f}{'/'+f'{r:.1f}m' if r is not None else ''}")
    print(f"{ts}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
