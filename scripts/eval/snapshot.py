"""Non-interactive snapshot of the viewer's 2x2 panel at a single frame.

Same overlays as scripts/eval/viewer.py, but renders to a PNG instead
of an interactive window. Useful for documentation, headless test
environments, and showing screenshots in a chat. Skips matplotlib's
event loop entirely by using the Agg backend.

Usage:
    python -m scripts.eval.snapshot \\
        --triplet data/Boats/2025-07-07_17-11-00 \\
        --frame 100 --out results/demo/boats_f100.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

from scripts.sensor_processing.pipeline import (  # noqa: E402
    ObstacleDetectionPipeline,
    iterate_triplet,
    make_bins,
)
from scripts.utils.calibration import load_detection, load_intrinsics  # noqa: E402
from scripts.utils.datasets import resolve_triplet  # noqa: E402
from scripts.utils.geometry import (  # noqa: E402
    load_extrinsics,
    project_radar_to_fisheye,
    project_radar_to_thermal,
)


def _overlay_camera(sensor_result, projected, ranges):
    if sensor_result is None:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    img = cv2.cvtColor(sensor_result.undistorted, cv2.COLOR_BGR2RGB).copy()
    h, w = img.shape[:2]
    slope, intercept, conf = sensor_result.horizon_line
    if conf > 0:
        y0 = int(np.clip(intercept, 0, h - 1))
        y1 = int(np.clip(slope * (w - 1) + intercept, 0, h - 1))
        cv2.line(img, (0, y0), (w - 1, y1), (0, 255, 255), 1)
    if projected is not None and ranges is not None:
        for (px, py), r in zip(projected, ranges):
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            if not (0 <= px < w and 0 <= py < h):
                continue
            norm = float(np.clip(r / 9.0, 0, 1))
            color = (int(255 * (1 - norm)), int(255 * norm * 0.5), int(255 * norm))
            cv2.circle(img, (int(px), int(py)), 3, color, -1)
    for (x, y) in sensor_result.coords:
        cv2.circle(img, (x, y), 6, (255, 0, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", required=True)
    ap.add_argument("--frame", type=int, default=100,
                    help="Frame index in the triplet's mmwave timestamp order.")
    ap.add_argument("--out", required=True, help="Path for the output PNG.")
    args = ap.parse_args()

    triplet = resolve_triplet(args.triplet)
    intrinsics = load_intrinsics()
    detection = load_detection()
    extrinsics = load_extrinsics()
    pipeline = ObstacleDetectionPipeline(intrinsics, detection)

    target = None
    for i, (ts, fish, therm, mm_pts) in enumerate(iterate_triplet(triplet, detection)):
        res = pipeline.process_frame(fish, therm, mm_pts, timestamp=ts)
        if i == args.frame:
            target = (ts, fish, therm, mm_pts, res)
            break
    if target is None:
        raise SystemExit(f"clip ended before frame {args.frame}")
    ts, fish, therm, mm_pts, res = target

    radar_pts = res.mmwave.points_xyz if res.mmwave is not None else np.empty((0, 3))
    ranges = radar_pts[:, 1] if len(radar_pts) else None
    fish_proj = thermal_proj = None
    if len(radar_pts):
        fish_proj = project_radar_to_fisheye(
            radar_pts, intrinsics["fisheye"]["K"], intrinsics["fisheye"]["D"],
            extrinsics["T_radar_to_fisheye"],
        )
        thermal_proj = project_radar_to_thermal(
            radar_pts, intrinsics["thermal"]["K"], intrinsics["thermal"]["D"],
            extrinsics["T_radar_to_thermal"],
        )

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    ax_fish, ax_therm = axs[0]
    ax_radar, ax_bar = axs[1]
    for a in (ax_fish, ax_therm):
        a.set_xticks([]); a.set_yticks([])

    fish_img = _overlay_camera(res.fisheye, fish_proj, ranges)
    therm_img = _overlay_camera(res.thermal, thermal_proj, ranges)
    ax_fish.imshow(fish_img); ax_fish.set_title("Fisheye RGB + horizon + detections + radar overlay")
    ax_therm.imshow(therm_img); ax_therm.set_title("Thermal + same overlays")

    if len(radar_pts):
        ax_radar.scatter(radar_pts[:, 0], radar_pts[:, 1],
                         c=radar_pts[:, 1], cmap="turbo", s=40, vmin=0, vmax=9)
    ax_radar.set_xlim(-5, 5); ax_radar.set_ylim(0, 9)
    ax_radar.set_aspect("equal", adjustable="box")
    ax_radar.set_xlabel("X lateral (m)"); ax_radar.set_ylabel("Y forward (m)")
    ax_radar.grid(True, alpha=0.3); ax_radar.set_title("Radar bird's-eye")
    edges, centers = make_bins(detection["fusion"])
    for e0, e1 in zip(edges[:-1], edges[1:]):
        ax_radar.add_patch(Wedge((0, 0), 9, 90 - float(e1), 90 - float(e0),
                                 facecolor="none", edgecolor="gray", linewidth=0.3, alpha=0.6))

    bar_x = np.arange(len(centers))
    bars = ax_bar.bar(bar_x, res.fusion.scores)
    for bar, hits in zip(bars, res.fusion.sensor_hit_mask):
        n = int(hits.sum())
        bar.set_color({0: "#888", 1: "#f6b26b", 2: "#f1c232", 3: "#6aa84f"}[n])
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_xticks(bar_x)
    ax_bar.set_xticklabels([f"{int(c):+d}°" for c in centers])
    ax_bar.axhline(detection["fusion"].get("hit_threshold", 0.33),
                   color="orange", linestyle="--", linewidth=0.8)
    ax_bar.set_ylabel("fused score")
    ax_bar.set_title("Fusion bin scores (grey=0, orange=1, yellow=2, green=3 sensors)")

    fig.suptitle(f"{triplet.clip_id}   frame {args.frame}   ts={ts}", fontsize=10)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
