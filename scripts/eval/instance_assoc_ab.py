"""Instance-mask vs box association A/B + polygon-margin measurement.

Two modes against one clip:

1. **A/B** (default): run the full pipeline twice: `fusion.association`
   box gate vs `use_instance_masks` polygon gate: and report the 2026-07-09
   arbitration-stack verification proxies side by side: range attribution
   (radar-sourced / compat-rejected / veto), announced near-read medians,
   the frames-0-18 edge-pillar window, and confirmed-bin counts.

2. **--sweep-margins**: single pipeline pass caching, for every radar
   return matched by the BOX gate on a polygon-covered detection, its
   signed distance to the detection's instance polygon, classified as
   on-silhouette (projects onto seg OBSTACLE pixels: the object's own
   return) or foreground (radar < 0.5x camera-raw: the clutter the compat
   rule rejects). The margin choice is then a measurement: retention of
   true returns vs admission of clutter per candidate margin.

    python -m scripts.eval.instance_assoc_ab \
        --triplet data/captures/2026-07-08/2026-07-08_16-37-01 --imu
    python -m scripts.eval.instance_assoc_ab \
        --triplet data/captures/2026-07-08/2026-07-08_16-37-01 --imu \
        --sweep-margins 0,2,4,6,8,12,16,24,32
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

import cv2
import numpy as np

from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    iterate_triplet,
)
from scripts.utils.association import (
    AssociationParams,
    match_polygons_to_detections,
    polygons_from_instances,
)
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import resolve_triplet
from scripts.utils.detections import InstanceSegProvider
from scripts.utils.geometry import project_radar_to_undistorted
from scripts.utils.segmentation import OBSTACLE as SEG_OBSTACLE

EDGE_BEARING_DEG = 35.0   # |bearing| >= this = "edge" (pillar window proxy)
EDGE_FRAMES = 19          # frames 0-18 = the quad clip's setup phase
NEAR_BAND_M = (1.0, 8.0)  # the compat/veto rules' near band


def _attitude_for(triplet, use_imu: bool):
    if not use_imu or triplet.imu is None:
        return None
    from scripts.sensor_processing.imu_bno085 import load_imu_config
    from scripts.sensor_processing.imu_replay import ImuLogAttitudeProvider
    return ImuLogAttitudeProvider.for_triplet(
        triplet, (load_imu_config().get("replay", {}) or {}))


def _med(vals):
    return float(np.median(vals)) if vals else None


def run_mode(triplet, detection, intrinsics, use_instances: bool,
             use_imu: bool, max_frames: int = 0,
             margin_px: float | None = None) -> dict:
    """One full pipeline pass; returns the arbitration verification proxies."""
    det = copy.deepcopy(detection)
    det["fusion"]["association"]["enabled"] = True
    det["fusion"]["association"]["use_instance_masks"] = use_instances
    if margin_px is not None:
        det["fusion"]["association"]["instance_margin_px"] = margin_px
    attitude = _attitude_for(triplet, use_imu)
    pipeline = ObstacleDetectionPipeline(intrinsics, det,
                                         attitude_provider=attitude)
    m = {
        "mode": "instance" if use_instances else "box",
        "n_frames": 0,
        "confirmed_bin_instances": 0,
        "frames_with_confirmed": 0,
    }
    for cam in ("fisheye", "thermal"):
        m[cam] = {
            "n_det": 0, "n_instance_gated": 0, "n_matched": 0,
            "n_radar_sourced": 0, "n_compat_rejected": 0, "n_vetoed": 0,
            "radar_sourced_ranges": [], "near_reads": [],
            "edge_window_near": 0, "edge_window_det": 0,
        }
    for ts, fish, therm, pts in iterate_triplet(triplet, det):
        if max_frames and m["n_frames"] >= max_frames:
            break
        if attitude is not None:
            attitude.set_time(ts)
        fid = f"{triplet.clip_id}/ts={ts.replace(':', '-')}"
        res = pipeline.process_frame(fish, therm, pts, timestamp=ts,
                                     frame_id=fid)
        frame_idx = m["n_frames"]
        m["n_frames"] += 1
        n_conf = sum(res.fusion.confirmed)
        m["confirmed_bin_instances"] += n_conf
        m["frames_with_confirmed"] += bool(n_conf)
        for cam, r in (("fisheye", res.fisheye), ("thermal", res.thermal)):
            if r is None:
                continue
            o = m[cam]
            o["n_det"] += len(r.coords)
            if not r.radar_ranges:
                continue
            gates = r.radar_gates or ["box"] * len(r.radar_ranges)
            monos = r.mono_ranges or list(r.ranges)
            for ang, rng, rr, hits, gate, mono in zip(
                    r.angles, r.ranges, r.radar_ranges, r.radar_hits,
                    gates, monos):
                o["n_instance_gated"] += gate == "instance"
                o["n_matched"] += hits > 0
                if rr is not None:
                    o["n_radar_sourced"] += 1
                    o["radar_sourced_ranges"].append(float(rr))
                elif hits > 0:
                    o["n_compat_rejected"] += 1
                # Veto proxy: association left a near range, veto nulled it.
                if (rng is None and rr is None and hits == 0
                        and mono is not None
                        and NEAR_BAND_M[0] <= mono <= NEAR_BAND_M[1]):
                    o["n_vetoed"] += 1
                if rng is not None and rng < 3.0:
                    o["near_reads"].append(float(rng))
                if cam == "fisheye" and frame_idx < EDGE_FRAMES \
                        and abs(ang) >= EDGE_BEARING_DEG:
                    o["edge_window_det"] += 1
                    if rng is not None \
                            and NEAR_BAND_M[0] <= rng <= NEAR_BAND_M[1]:
                        o["edge_window_near"] += 1
    for cam in ("fisheye", "thermal"):
        o = m[cam]
        o["median_radar_sourced_m"] = _med(o.pop("radar_sourced_ranges"))
        nr = o.pop("near_reads")
        o["n_near_reads"] = len(nr)
        o["median_near_read_m"] = _med(nr)
    return m


def sweep_margins(triplet, detection, intrinsics, margins: list[float],
                  use_imu: bool, max_frames: int = 0,
                  dump_points: str | None = None) -> dict:
    """Single pass; distance-to-polygon distribution of box-gate matches.

    Only detections a polygon covers contribute (elsewhere the gates are
    identical by construction). Points classified on-silhouette vs
    foreground; the sweep reports, per margin, how many of each the polygon
    gate would keep.
    """
    det = copy.deepcopy(detection)
    det["fusion"]["association"]["enabled"] = True
    det["fusion"]["association"]["use_instance_masks"] = False
    assoc_cfg = det["fusion"]["association"]
    params = AssociationParams.from_config(assoc_cfg)
    inst = InstanceSegProvider(assoc_cfg.get("instance_root", "data/det_seg"))
    attitude = _attitude_for(triplet, use_imu)
    pipeline = ObstacleDetectionPipeline(intrinsics, det,
                                         attitude_provider=attitude)
    P = np.asarray(intrinsics["fisheye"]["K"], float)
    T = pipeline.extrinsics["T_radar_to_fisheye"]
    on_sil, foreground, other = [], [], []   # signed distances, px
    rows = []       # per matched point, for offline analysis (--dump-points)
    n_frames = n_covered = 0
    det_uid = 0
    for ts, fish, therm, pts in iterate_triplet(triplet, det):
        if max_frames and n_frames >= max_frames:
            break
        if attitude is not None:
            attitude.set_time(ts)
        fid = f"{triplet.clip_id}/ts={ts.replace(':', '-')}"
        res = pipeline.process_frame(fish, therm, pts, timestamp=ts,
                                     frame_id=fid)
        n_frames += 1
        r = res.fisheye
        if r is None or not r.coords or not len(res.mmwave.points_xyz):
            continue
        instances = inst.get(fid)
        if not instances:
            continue
        h_img, w_img = r.undistorted.shape[:2]
        polys = polygons_from_instances(instances, (w_img, h_img))
        if not polys:
            continue
        match_idx = match_polygons_to_detections(polys, r.coords, r.sizes)
        px = project_radar_to_undistorted(res.mmwave.points_xyz, P, T)
        seg = r.seg_mask
        for (u, v), size, raw, k in zip(r.coords, r.sizes, r.raw_ranges,
                                        match_idx):
            if k is None:
                continue
            n_covered += 1
            det_uid += 1
            contour = polys[k].reshape(-1, 1, 2)
            box_r = size / 2.0 + params.max_px
            for j in range(len(px)):
                pu, pv = px[j]
                if not (np.isfinite(pu) and np.isfinite(pv)):
                    continue
                if abs(pu - u) > box_r or abs(pv - v) > box_r:
                    continue
                d = cv2.pointPolygonTest(contour, (float(pu), float(pv)),
                                         True)
                y = float(res.mmwave.points_xyz[j, 1])
                sil = False
                if seg is not None:
                    ui, vi = int(round(pu)), int(round(pv))
                    if 0 <= vi < seg.shape[0] and 0 <= ui < seg.shape[1]:
                        sil = seg[vi, ui] == SEG_OBSTACLE
                if sil:
                    on_sil.append(d)
                elif raw is not None and y < 0.5 * raw:
                    foreground.append(d)
                else:
                    other.append(d)
                rows.append((det_uid, float(d), y, int(sil),
                             float(raw) if raw is not None else np.nan))
    if dump_points:
        with open(dump_points, "w") as fp:
            fp.write("det_uid,dist_px,range_m,on_sil,raw_m\n")
            for row in rows:
                fp.write(",".join(str(x) for x in row) + "\n")
    out = {"n_frames": n_frames, "n_covered_detections": n_covered,
           "n_on_silhouette": len(on_sil), "n_foreground": len(foreground),
           "n_other": len(other), "margins": {}}
    sil_a, fg_a, ot_a = (np.asarray(x) for x in (on_sil, foreground, other))
    if len(sil_a):
        out["on_sil_dist_p10_p50_p90"] = [
            float(np.percentile(sil_a, p)) for p in (10, 50, 90)]
    if len(fg_a):
        out["foreground_dist_p10_p50_p90"] = [
            float(np.percentile(fg_a, p)) for p in (10, 50, 90)]
    for mg in margins:
        out["margins"][mg] = {
            "on_sil_kept": float(np.mean(sil_a >= -mg)) if len(sil_a) else None,
            "foreground_admitted": (float(np.mean(fg_a >= -mg))
                                    if len(fg_a) else None),
            "other_admitted": float(np.mean(ot_a >= -mg)) if len(ot_a) else None,
        }
    return out


def _print_ab(box: dict, inst: dict) -> None:
    def row(label, f):
        print(f"  {label:<38}{f(box):>14}{f(inst):>14}")

    print(f"\n{'':<40}{'box':>14}{'instance':>14}")
    print(f"  frames: {box['n_frames']}")
    row("confirmed-bin instances",
        lambda m: m["confirmed_bin_instances"])
    row("frames with a confirmed bin",
        lambda m: m["frames_with_confirmed"])
    for cam in ("fisheye", "thermal"):
        print(f"  -- {cam}")
        for key, label in (
                ("n_det", "detections"),
                ("n_instance_gated", "instance-gated"),
                ("n_matched", "radar-matched (hits > 0)"),
                ("n_radar_sourced", "radar-sourced ranges"),
                ("n_compat_rejected", "compat-rejected"),
                ("n_vetoed", "radar-vetoed (proxy)"),
                ("n_near_reads", "announced reads < 3 m"),
        ):
            row(label, lambda m, c=cam, k=key: m[c][k])
        for key, label in (
                ("median_radar_sourced_m", "median radar-sourced range (m)"),
                ("median_near_read_m", "median near read (m)"),
        ):
            row(label, lambda m, c=cam, k=key:
                "-" if m[c][k] is None else f"{m[c][k]:.2f}")
        if cam == "fisheye":
            row("edge-window near reads (of dets)",
                lambda m, c=cam: f"{m[c]['edge_window_near']}"
                                 f"/{m[c]['edge_window_det']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", required=True)
    ap.add_argument("--imu", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--sweep-margins", default=None,
                    help="comma-separated px margins; sweep mode instead of A/B")
    ap.add_argument("--dump-points", default=None,
                    help="sweep mode: write per-matched-point CSV here")
    ap.add_argument("--margin", type=float, default=None,
                    help="A/B mode: override instance_margin_px")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args(argv)

    triplet = resolve_triplet(args.triplet)
    intrinsics = load_intrinsics()
    detection = load_detection()

    if args.sweep_margins:
        margins = [float(x) for x in args.sweep_margins.split(",")]
        r = sweep_margins(triplet, detection, intrinsics, margins,
                          args.imu, args.max_frames,
                          dump_points=args.dump_points)
        print(f"\nmargin sweep: {triplet.clip_id} "
              f"({r['n_frames']} frames, {r['n_covered_detections']} "
              f"polygon-covered detections)")
        print(f"  box-gate matched points: {r['n_on_silhouette']} "
              f"on-silhouette, {r['n_foreground']} foreground, "
              f"{r['n_other']} other")
        for k in ("on_sil_dist_p10_p50_p90", "foreground_dist_p10_p50_p90"):
            if k in r:
                p10, p50, p90 = r[k]
                print(f"  {k}: {p10:+.1f} / {p50:+.1f} / {p90:+.1f} px")
        print(f"  {'margin':>8}{'on-sil kept':>14}{'fg admitted':>14}"
              f"{'other adm.':>12}")
        for mg, row in r["margins"].items():
            fmt = lambda x: "-" if x is None else f"{x:.1%}"
            print(f"  {mg:>8.0f}{fmt(row['on_sil_kept']):>14}"
                  f"{fmt(row['foreground_admitted']):>14}"
                  f"{fmt(row['other_admitted']):>12}")
        payload = r
    else:
        box = run_mode(triplet, detection, intrinsics, False, args.imu,
                       args.max_frames)
        inst = run_mode(triplet, detection, intrinsics, True, args.imu,
                        args.max_frames, margin_px=args.margin)
        print(f"\nA/B: {triplet.clip_id}")
        _print_ab(box, inst)
        payload = {"clip": triplet.clip_id, "box": box, "instance": inst}

    if args.out:
        with open(args.out, "w") as fp:
            json.dump(payload, fp, indent=2)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
