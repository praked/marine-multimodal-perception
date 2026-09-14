"""Animate the radar bird's-eye view across a clip and write an mp4.

Shows current-frame points in red and a fading grey trail of the last
N frames so the spatial-temporal pattern is visible. Useful for
diagnosing mount clutter, surface clutter clusters, and frame-to-frame
density without staring at 200+ single-frame snapshots.

Usage:
    python -m scripts.eval.radar_video --triplet data/Boats/2025-07-07_17-11-00 \\
        --out results/demo/radar_boats.mp4
    python -m scripts.eval.radar_video --triplet data/Ducks/2025-07-15_05-36-18 \\
        --out results/demo/radar_ducks.mp4 --fps 6 --trail 20
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

from scripts.sensor_processing.pipeline import make_bins  # noqa: E402
from scripts.utils.calibration import load_detection  # noqa: E402
from scripts.utils.datasets import load_mmwave_csv, resolve_triplet  # noqa: E402


def _draw_frame(ax, current_pts, trail, edges, y_max):
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X lateral (m)")
    ax.set_ylabel("Y forward (m)")
    ax.grid(True, alpha=0.3)
    for e0, e1 in zip(edges[:-1], edges[1:]):
        ax.add_patch(Wedge((0, 0), y_max, 90 - float(e1), 90 - float(e0),
                           facecolor="none", edgecolor="gray",
                           linewidth=0.3, alpha=0.4))
    # Trail (older = more transparent)
    n_trail = len(trail)
    for age, trail_pts in enumerate(trail):
        if len(trail_pts) == 0:
            continue
        alpha_val = (age + 1) / max(n_trail, 1) * 0.45
        ax.scatter(trail_pts[:, 0], trail_pts[:, 1],
                   c="gray", alpha=alpha_val, s=18, edgecolors="none")
    if len(current_pts):
        ax.scatter(current_pts[:, 0], current_pts[:, 1],
                   c="red", s=70, edgecolors="black", linewidths=0.8, zorder=10)


def _fig_to_bgr(fig):
    """Render matplotlib figure to BGR numpy for cv2.VideoWriter."""
    fig.canvas.draw()
    try:
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
    except AttributeError:
        # Older matplotlib
        rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        rgb = rgb.reshape((h, w, 3))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", required=True)
    ap.add_argument("--out", required=True, help="Output mp4 path.")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--trail", type=int, default=25,
                    help="Number of past frames in the fading trail.")
    ap.add_argument("--y-max", type=float, default=9.0,
                    help="Display Y limit in metres (radar max unambiguous range is 9.02).")
    ap.add_argument("--filter", action="store_true",
                    help="Apply detection.yaml's y_min/y_max gate before plotting "
                         "(otherwise shows all sanity-checked radar points).")
    args = ap.parse_args()

    triplet = resolve_triplet(args.triplet)
    df = load_mmwave_csv(triplet.mmwave)
    timestamps = sorted(df["RoundedTime"].unique())
    grouped = df.groupby("RoundedTime")

    detection = load_detection()
    _, centers = make_bins(detection["fusion"])
    edges, _ = make_bins(detection["fusion"])

    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    _draw_frame(ax, np.empty((0, 2)), deque(maxlen=args.trail), edges, args.y_max)
    bgr = _fig_to_bgr(fig)
    h, w = bgr.shape[:2]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))

    trail: deque = deque(maxlen=args.trail)
    y_min, y_max = detection["mmwave"]["y_min"], detection["mmwave"]["y_max"]

    try:
        for i, ts in enumerate(timestamps):
            g = grouped.get_group(ts)
            if args.filter:
                g = g[(g["Y"] >= y_min) & (g["Y"] <= y_max)]
            pts = g[["X", "Y"]].values

            _draw_frame(ax, pts, trail, edges, args.y_max)
            ax.set_title(f"{triplet.clip_id}   frame {i+1}/{len(timestamps)}   "
                         f"ts={ts}   N={len(pts)}",
                         fontsize=10)
            writer.write(_fig_to_bgr(fig))
            trail.append(pts)
    finally:
        writer.release()
        plt.close(fig)
    print(f"wrote {len(timestamps)} frames at {args.fps} fps -> {args.out}")


if __name__ == "__main__":
    main()
