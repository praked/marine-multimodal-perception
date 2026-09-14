"""Validate the IMU->camera axis mapping against the detected horizon.

The BNO08x rides on boat1 with an unmeasured mounting, so the sensor-frame
(roll, pitch) extracted from its quaternion log relate to the CAMERA's
(roll, pitch) by an unknown swap/sign combination (+ a fixed offset). The
fisheye's RANSAC horizon, on frames where it is confidently the true water
horizon, is an independent measurement of the same two angles:

    up = up_from_horizon_line(slope, intercept, K)
       = (-sin r,  -cos r cos p,  cos r sin p)      (geometry.py convention)
    =>  roll  r = asin(-up_x)
        pitch p = atan2(up_z, -up_y)

This script walks a clip, collects (horizon roll/pitch) vs (IMU roll/pitch)
pairs, scores all 8 swap/sign mappings by Pearson correlation of the
median-centred traces, and prints the winner + the residual offsets: the
values to put in configs/imu.yaml `replay:`. Optionally writes a trace plot.

    python -m scripts.eval.imu_attitude_check \
        --triplet data/captures/2026-07-08/2026-07-08_16-37-01 \
        --plot results/imu_check/2026-07-08_16-37-01.png

TWO OPTIONS MATTER FOR THE CURRENT BOX MOUNT (both added 2026-08-18):

`--attitude-source rotation` uses the shipped rotation composition instead of
the raw Euler angles. The box IMU sits on the enclosure's back panel, which
parks its Euler parameterisation ~12 deg from gimbal lock, so the swap/sign
search below is meaningless there: scored on raw angles it returns 0.08 pitch
correlation and offsets of -80/+101 deg on data that is actually fine. With
`rotation` the mapping is already known, so the tool reports agreement instead
of searching for a mapping.

`--reference water-edge` takes the horizon from the SEGMENTATION water mask
rather than RANSAC. On a lake that never offers a shoreline-free horizon,
RANSAC locks to the treeline and contributes ~3.4 deg of noise; measured
2026-08-18, switching the reference took roll correlation 0.44 -> 0.80 and cut
the camera's own pitch scatter from 3.61 to 1.13 deg. Needs masks:
    python -m scripts.eval.local_seg_masks --triplet <clip>

    python -m scripts.eval.imu_attitude_check --triplet <clip> \
        --attitude-source rotation --reference water-edge
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np

import os

import cv2
import yaml

from scripts.sensor_processing.imu_replay import (
    ImuLogAttitudeProvider,
    _seconds_of_day,
)
from scripts.utils.segmentation import horizon_from_water
from scripts.sensor_processing.pipeline import iterate_triplet
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.cv_common import (
    detect_horizon,
    horizon_kwargs,
    undistort_fisheye,
)
from scripts.utils.datasets import load_imu_csv, resolve_triplet
from scripts.utils.geometry import up_from_horizon_line


def pitch_roll_from_horizon(slope: float, intercept: float, K: np.ndarray
                            ) -> tuple[float, float]:
    """(pitch_rad, roll_rad) of the camera implied by a detected horizon."""
    up = up_from_horizon_line(slope, intercept, K)
    roll = math.asin(max(-1.0, min(1.0, -float(up[0]))))
    pitch = math.atan2(float(up[2]), -float(up[1]))
    return pitch, roll


def _centred_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.median(a)
    b = b - np.median(b)
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.mean(a * b) / (sa * sb))


def _plot(path, t_pairs, ref_roll, ref_pitch, imu_roll, imu_pitch, clip_id, title):
    """Trace plot of reference vs IMU attitude, both already offset-aligned."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(t_pairs) - t_pairs[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, ref, imu, name in ((axes[0], ref_roll, imu_roll, "roll"),
                               (axes[1], ref_pitch, imu_pitch, "pitch")):
        ax.plot(t, np.degrees(ref), ".-", lw=0.8, ms=3, label=f"reference {name}")
        ax.plot(t, np.degrees(imu), ".-", lw=0.8, ms=3, label=f"IMU {name}")
        ax.set_ylabel(f"{name} [deg]")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel(f"seconds since first pair ({clip_id})")
    fig.suptitle(title)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", required=True)
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="RANSAC horizon confidence gate.")
    ap.add_argument("--max-tilt-px", type=float, default=170.0,
                    help="Reject horizons further than this from the "
                         "principal point at image centre (shoreline locks).")
    ap.add_argument("--max-imu-gap-s", type=float, default=0.6,
                    help="Pair a frame only when an IMU sample is this close.")
    ap.add_argument("--attitude-source", choices=("euler", "rotation"),
                    default="euler",
                    help="euler: search the 8 swap/sign mappings (legacy BLE "
                         "mounts). rotation: use the shipped composition, whose "
                         "mapping is already known: required for the box mount, "
                         "where the raw angles sit beside gimbal lock.")
    ap.add_argument("--reference", choices=("horizon", "water-edge"),
                    default="horizon",
                    help="Attitude reference. water-edge reads the seg masks "
                         "and is the right choice on a lake, where RANSAC locks "
                         "to the treeline.")
    ap.add_argument("--seg-root", default="data/seg",
                    help="Where local_seg_masks wrote the masks.")
    ap.add_argument("--plot", default=None, help="Optional output PNG path.")
    args = ap.parse_args(argv)

    triplet = resolve_triplet(args.triplet)
    if triplet.imu is None:
        print("Clip has no imu_<ts>.csv sidecar: nothing to check.",
              file=sys.stderr)
        return 1
    imu_df = load_imu_csv(triplet.imu)
    if imu_df.empty:
        print("IMU log is empty: nothing to check.", file=sys.stderr)
        return 1

    intrinsics = load_intrinsics()
    detection = load_detection()
    intr = intrinsics["fisheye"]
    K = np.asarray(intr["K"], float)
    cy = float(K[1, 2])

    # Raw (unmapped, unzeroed) IMU euler traces on the log's own timeline.
    if args.attitude_source == "rotation":
        # The composition path: the mapping is already known and measured, so
        # take the shipped config (offsets included) rather than re-searching.
        cfg = dict(yaml.safe_load(open("configs/imu.yaml"))["replay"])
        cfg["max_age_s"] = args.max_imu_gap_s
        cfg["attitude_source"] = "rotation"
    else:
        cfg = {"max_age_s": args.max_imu_gap_s, "zero": "none"}
    raw = ImuLogAttitudeProvider(imu_df, cfg)

    seg_dir = None
    if args.reference == "water-edge":
        seg_dir = os.path.join(args.seg_root, triplet.clip_id.replace("/", "__"))
        if not os.path.isdir(seg_dir):
            print(f"No masks at {seg_dir}. Generate them first:\n"
                  f"  python -m scripts.eval.local_seg_masks --triplet "
                  f"{args.triplet}", file=sys.stderr)
            return 1

    t_pairs, h_pitch, h_roll, i_pitch, i_roll = [], [], [], [], []
    n_frames = n_conf = 0
    for ts, fish, _therm, _pts in iterate_triplet(triplet, detection):
        n_frames += 1
        if seg_dir is not None:
            mask_path = os.path.join(seg_dir, f"ts={str(ts).replace(':', '-')}.png")
            seg = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if seg is None:
                continue
            slope, intercept, conf = horizon_from_water(seg)
        else:
            und = undistort_fisheye(fish, intr["K"], intr["D"])
            _mask, slope, intercept, conf = detect_horizon(
                und, **horizon_kwargs(detection["horizon"]))
        if conf < args.min_confidence:
            continue
        row_c = slope * float(K[0, 2]) + intercept
        if abs(row_c - cy) > args.max_tilt_px:
            continue
        n_conf += 1
        raw.set_time(ts)
        att = raw.get()
        if att is None:
            continue
        p, r = pitch_roll_from_horizon(slope, intercept, K)
        t_pairs.append(_seconds_of_day(ts))
        h_pitch.append(p)
        h_roll.append(r)
        i_pitch.append(att.pitch_rad)
        i_roll.append(att.roll_rad)

    print(f"{n_frames} frames · {n_conf} confident horizons · "
          f"{len(t_pairs)} paired with IMU samples")
    if len(t_pairs) < 10:
        print("Too few pairs to score mappings.", file=sys.stderr)
        return 1

    hp, hr = np.array(h_pitch), np.array(h_roll)
    ip, ir = np.array(i_pitch), np.array(i_roll)

    if args.attitude_source == "rotation":
        # Mapping already known: report agreement, not a search.
        off_r = float(np.median(hr) - np.median(ir))
        off_p = float(np.median(hp) - np.median(ip))
        print(f"\nreference: {args.reference}   attitude: rotation composition "
              f"(configs/imu.yaml)")
        print(f"{'axis':>6} {'corr':>7} {'bias':>9} {'resid sd':>9} "
              f"{'cam sd':>8} {'imu sd':>8}")
        for name, cam, imu_a in (("roll", hr, ir), ("pitch", hp, ip)):
            d = imu_a - cam
            print(f"{name:>6} {_centred_corr(cam, imu_a):>+7.3f} "
                  f"{math.degrees(d.mean()):>+8.2f}d {math.degrees(d.std()):>8.2f}d "
                  f"{math.degrees(cam.std()):>7.2f}d {math.degrees(imu_a.std()):>7.2f}d")
        print(f"\nresidual offsets still unaccounted for (reference - IMU): "
              f"roll {math.degrees(off_r):+.2f} deg  pitch {math.degrees(off_p):+.2f} deg")
        print("A well-calibrated mount reads near-zero bias and a correlation "
              "close to 1. Low correlation with small bias means the offsets are "
              "right but the motion is too small to track: repeat with the boat "
              "rocked deliberately (slow, several degrees).")
        if args.plot:
            _plot(args.plot, t_pairs, hr, hp, ir + off_r, ip + off_p,
                  triplet.clip_id,
                  f"IMU (rotation) vs {args.reference}")
        return 0

    results = []
    for swap, s_r, s_p in product((False, True), (1.0, -1.0), (1.0, -1.0)):
        cr = s_r * (ip if swap else ir)      # candidate camera roll
        cp = s_p * (ir if swap else ip)      # candidate camera pitch
        score = _centred_corr(cr, hr) + _centred_corr(cp, hp)
        results.append((score, swap, s_r, s_p, cr, cp))
    results.sort(key=lambda x: -x[0])

    print(f"\n{'swap':>5} {'sign_roll':>9} {'sign_pitch':>10} "
          f"{'corr_roll':>9} {'corr_pitch':>10} {'sum':>6}")
    for score, swap, s_r, s_p, cr, cp in results:
        print(f"{str(swap):>5} {s_r:>+9.0f} {s_p:>+10.0f} "
              f"{_centred_corr(cr, hr):>9.3f} {_centred_corr(cp, hp):>10.3f} "
              f"{score:>6.3f}")

    score, swap, s_r, s_p, cr, cp = results[0]
    off_r = float(np.median(hr) - np.median(cr))
    off_p = float(np.median(hp) - np.median(cp))
    res_r = np.degrees(np.std((cr + off_r) - hr))
    res_p = np.degrees(np.std((cp + off_p) - hp))
    print(f"\nBest mapping: swap_roll_pitch={swap} sign_roll={s_r:+.0f} "
          f"sign_pitch={s_p:+.0f}  (corr sum {score:.3f})")
    print(f"Mount offsets (horizon - IMU medians): "
          f"roll {math.degrees(off_r):+.2f} deg  pitch {math.degrees(off_p):+.2f} deg")
    print(f"Residual std after offset: roll {res_r:.2f} deg  pitch {res_p:.2f} deg")
    print("-> configs/imu.yaml replay: "
          f"swap_roll_pitch: {str(swap).lower()}, sign_roll: {s_r:.0f}, "
          f"sign_pitch: {s_p:.0f}, mount_offset_rad: "
          f"{{pitch: {off_p:.4f}, roll: {off_r:.4f}}} (with zero: none)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t0 = t_pairs[0]
        t = np.array(t_pairs) - t0
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for ax, h, c, name, off in (
            (axes[0], hr, cr, "roll", off_r),
            (axes[1], hp, cp, "pitch", off_p),
        ):
            ax.plot(t, np.degrees(h), ".-", lw=0.8, ms=3,
                    label=f"horizon {name}")
            ax.plot(t, np.degrees(c + off), ".-", lw=0.8, ms=3,
                    label=f"IMU {name} (mapped+offset)")
            ax.set_ylabel(f"{name} [deg]")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)
        axes[1].set_xlabel(f"seconds since first pair ({triplet.clip_id})")
        fig.suptitle(
            f"IMU vs horizon attitude: swap={swap} sr={s_r:+.0f} "
            f"sp={s_p:+.0f}, corr sum {score:.3f}")
        out = Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
