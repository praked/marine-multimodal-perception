"""Visual eval harness: 2x2 viewer of fisheye / thermal / radar / scores.

Step through a triplet frame-by-frame, hot-reload detection params, and
screenshot interesting cases. Built for Phase I.4.2 of PLAN.md.

Usage:
    python -m scripts.eval.viewer --triplet data/Boats/2025-06-23_16-21-07

Keys:
    left / right    step one frame
    space           play / pause (3 fps)
    s               screenshot to results/screens/
    r               hot-reload detection.yaml (rebuilds pipeline)
    q               quit
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

from scripts.sensor_processing.pipeline import (
    FrameResult,
    ObstacleDetectionPipeline,
    iterate_triplet,
    make_bins,
)
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.datasets import resolve_triplet
from scripts.utils.geometry import (
    load_extrinsics,
    project_radar_to_fisheye,
    project_radar_to_thermal,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCREEN_DIR = REPO_ROOT / "results" / "screens"


class Viewer:
    def __init__(self, triplet_prefix: str, detection_path: str | None = None,
                 show_projection: bool = True):
        self.triplet = resolve_triplet(triplet_prefix)
        self.detection_path = detection_path
        self.intrinsics = load_intrinsics()
        self.detection = load_detection(detection_path) if detection_path else load_detection()
        self.extrinsics = load_extrinsics()
        self.show_projection = show_projection
        # Recorded-IMU replay for quad clips (imu_<ts>.csv sidecar); None
        # otherwise: pipeline falls back to horizon/level as before.
        self._imu_replay = None
        try:
            from scripts.sensor_processing.imu_bno085 import load_imu_config
            from scripts.sensor_processing.imu_replay import ImuLogAttitudeProvider
            self._imu_replay = ImuLogAttitudeProvider.for_triplet(
                self.triplet, (load_imu_config().get("replay", {}) or {}))
            if self._imu_replay is not None:
                print(f"[viewer] IMU replay: {len(self._imu_replay)} samples")
        except Exception as exc:  # noqa: BLE001
            print(f"[viewer] IMU replay unavailable ({exc})")
        self.pipeline = ObstacleDetectionPipeline(self.intrinsics, self.detection,
                                                  attitude_provider=self._imu_replay)

        # Cached frames for instant backward stepping.
        self.cached: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, FrameResult]] = []
        self._gen = iterate_triplet(self.triplet, self.detection)
        self.idx = 0
        self.playing = False
        self._build_figure()
        self._advance_to(0)

    # -- figure ---------------------------------------------------------

    def _build_figure(self):
        self.fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        self.fig.canvas.manager.set_window_title(f"viewer: {self.triplet.clip_id}")
        self.ax_fish, self.ax_therm = axs[0]
        self.ax_radar, self.ax_bar = axs[1]
        for a in (self.ax_fish, self.ax_therm):
            a.set_xticks([])
            a.set_yticks([])
        self.ax_fish.set_title("Fisheye RGB")
        self.ax_therm.set_title("Thermal")
        self.ax_radar.set_title("Radar (bird's eye)")
        self.ax_bar.set_title("Fusion bin scores")

        self.im_fish = self.ax_fish.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
        self.im_therm = self.ax_therm.imshow(np.zeros((10, 10, 3), dtype=np.uint8))

        self.radar_scat = self.ax_radar.scatter([], [], c=[], cmap="turbo", s=30, vmin=0, vmax=8)
        self.ax_radar.set_xlim(-5, 5)
        self.ax_radar.set_ylim(0, 9)
        self.ax_radar.set_xlabel("X lateral (m)")
        self.ax_radar.set_ylabel("Y forward (m)")
        self.ax_radar.set_aspect("equal", adjustable="box")
        self.ax_radar.grid(True, alpha=0.3)

        edges, centers = make_bins(self.detection["fusion"])
        self.bin_edges = edges
        self.bin_centers = centers
        # Wedge overlay on radar showing bin layout.
        radius = self.ax_radar.get_ylim()[1]
        for e0, e1 in zip(edges[:-1], edges[1:]):
            theta_a = 90 - float(e1)
            theta_b = 90 - float(e0)
            self.ax_radar.add_patch(
                Wedge(
                    (0, 0), radius, theta_a, theta_b,
                    facecolor="none", edgecolor="gray", linewidth=0.3, alpha=0.6,
                )
            )

        # Bar chart.
        self.bar_x = np.arange(len(centers))
        self.bars = self.ax_bar.bar(self.bar_x, np.zeros_like(centers, dtype=float))
        self.ax_bar.set_ylim(0, 1.05)
        self.ax_bar.set_xticks(self.bar_x)
        self.ax_bar.set_xticklabels([f"{int(c):+d}°" for c in centers], rotation=0)
        self.ax_bar.axhline(self.detection["fusion"].get("hit_threshold", 0.33),
                            color="orange", linestyle="--", linewidth=0.8)
        self.ax_bar.set_ylabel("fused score")

        self.status = self.fig.text(0.01, 0.01, "", fontsize=9, family="monospace")

        # Event hooks.
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._timer = self.fig.canvas.new_timer(interval=333)  # ~3 fps
        self._timer.add_callback(self._tick)

    # -- frame advance --------------------------------------------------

    def _ensure_cached_up_to(self, target_idx: int) -> bool:
        while len(self.cached) <= target_idx:
            try:
                ts, fish, therm, mm_pts = next(self._gen)
            except StopIteration:
                return False
            if self._imu_replay is not None:
                self._imu_replay.set_time(ts)
            res = self.pipeline.process_frame(fish, therm, mm_pts, timestamp=ts)
            self.cached.append((ts, fish, therm, mm_pts, res))
        return True

    def _advance_to(self, idx: int):
        if idx < 0:
            idx = 0
        if not self._ensure_cached_up_to(idx):
            idx = len(self.cached) - 1
        self.idx = idx
        self._render(*self.cached[idx])

    # -- rendering ------------------------------------------------------

    def _render(self, ts, fish_bgr, therm_bgr, mm_pts, res: FrameResult):
        # Pre-compute radar projections if extrinsics are loaded.
        radar_pts = res.mmwave.points_xyz if (res.mmwave is not None and self.show_projection) else np.empty((0, 3))

        # Fisheye panel.
        fish_proj = None
        if len(radar_pts):
            fish_proj = project_radar_to_fisheye(
                radar_pts,
                self.intrinsics["fisheye"]["K"],
                self.intrinsics["fisheye"]["D"],
                self.extrinsics["T_radar_to_fisheye"],
            )
        fish_rgb = self._overlay_camera(res.fisheye, projected=fish_proj, ranges=radar_pts[:, 1] if len(radar_pts) else None)
        self.im_fish.set_data(fish_rgb)
        self.ax_fish.set_xlim(0, fish_rgb.shape[1])
        self.ax_fish.set_ylim(fish_rgb.shape[0], 0)

        # Thermal panel.
        therm_proj = None
        if len(radar_pts):
            therm_proj = project_radar_to_thermal(
                radar_pts,
                self.intrinsics["thermal"]["K"],
                self.intrinsics["thermal"]["D"],
                self.extrinsics["T_radar_to_thermal"],
            )
        therm_rgb = self._overlay_camera(res.thermal, projected=therm_proj, ranges=radar_pts[:, 1] if len(radar_pts) else None)
        self.im_therm.set_data(therm_rgb)
        self.ax_therm.set_xlim(0, therm_rgb.shape[1])
        self.ax_therm.set_ylim(therm_rgb.shape[0], 0)

        # Radar.
        pts = res.mmwave.points_xyz if res.mmwave is not None else np.empty((0, 3))
        if len(pts):
            self.radar_scat.set_offsets(np.column_stack((pts[:, 0], pts[:, 1])))
            self.radar_scat.set_array(pts[:, 1])
        else:
            self.radar_scat.set_offsets(np.empty((0, 2)))
            self.radar_scat.set_array(np.array([]))

        # Bars.
        for bar, score, hits in zip(self.bars, res.fusion.scores,
                                     res.fusion.sensor_hit_mask):
            bar.set_height(score)
            n_hits = int(hits.sum())
            bar.set_color({0: "#888", 1: "#f6b26b", 2: "#f1c232", 3: "#6aa84f"}[n_hits])

        # Status.
        hp = res.fisheye.horizon_line if res.fisheye else (0, 0, 0)
        self.status.set_text(
            f"clip={self.triplet.clip_id}  frame={self.idx}/{len(self.cached) - 1}  "
            f"ts={ts}  fisheye_horizon_conf={hp[2]:.2f}  thermal_avg={self.pipeline.thermal_average:.1f}  "
            f"playing={self.playing}"
        )
        self.fig.canvas.draw_idle()

    def _overlay_camera(self, sensor_result, projected=None, ranges=None) -> np.ndarray:
        if sensor_result is None:
            return np.zeros((10, 10, 3), dtype=np.uint8)
        img = cv2.cvtColor(sensor_result.undistorted, cv2.COLOR_BGR2RGB).copy()

        h, w = img.shape[:2]
        slope, intercept, conf = sensor_result.horizon_line
        if conf > 0:
            x0, x1 = 0, w - 1
            y0 = int(np.clip(slope * x0 + intercept, 0, h - 1))
            y1 = int(np.clip(slope * x1 + intercept, 0, h - 1))
            cv2.line(img, (x0, y0), (x1, y1), (0, 255, 255), 1)

        # Radar projection overlay (Branch A.1). Colors by Y range:
        # blue near, red far.
        if projected is not None and ranges is not None:
            r_min, r_max = 0.0, 9.0
            for (px, py), r in zip(projected, ranges):
                if not (np.isfinite(px) and np.isfinite(py)):
                    continue
                if not (0 <= px < w and 0 <= py < h):
                    continue
                norm = float(np.clip((r - r_min) / (r_max - r_min), 0, 1))
                color = (int(255 * (1 - norm)), int(255 * norm * 0.5), int(255 * norm))
                cv2.circle(img, (int(px), int(py)), 3, color, -1)

        for (x, y) in sensor_result.coords:
            cv2.circle(img, (x, y), 6, (255, 0, 0), 2)

        return img

    # -- input handlers -------------------------------------------------

    def _on_key(self, event):
        if event.key in ("right", "n"):
            self.playing = False
            self._timer.stop()
            self._advance_to(self.idx + 1)
        elif event.key in ("left", "p"):
            self.playing = False
            self._timer.stop()
            self._advance_to(self.idx - 1)
        elif event.key == " ":
            self.playing = not self.playing
            if self.playing:
                self._timer.start()
            else:
                self._timer.stop()
            self._render(*self.cached[self.idx])
        elif event.key == "s":
            self._screenshot()
        elif event.key == "r":
            self._hot_reload()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def _tick(self):
        if not self.playing:
            return
        if not self._ensure_cached_up_to(self.idx + 1):
            self.playing = False
            self._timer.stop()
            return
        self._advance_to(self.idx + 1)

    def _screenshot(self):
        SCREEN_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{self.triplet.scene}_{self.triplet.timestamp}_frame{self.idx:05d}_{datetime.now().strftime('%H%M%S')}.png"
        path = SCREEN_DIR / name
        self.fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"saved {path}")

    def _hot_reload(self):
        try:
            self.detection = load_detection(self.detection_path) if self.detection_path else load_detection()
            # Rebuild pipeline, replay from frame 0 lazily (clear cache).
            self.pipeline = ObstacleDetectionPipeline(self.intrinsics, self.detection,
                                                      attitude_provider=self._imu_replay)
            self.cached.clear()
            self._gen = iterate_triplet(self.triplet, self.detection)
            print("reloaded detection.yaml, re-iterating from frame 0")
            self._advance_to(self.idx)
        except Exception as e:
            print(f"reload failed: {e}")

    def run(self):
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", required=True)
    ap.add_argument("--detection", default=None,
                    help="Override path to detection.yaml.")
    ap.add_argument("--no-projection", action="store_true",
                    help="Disable the radar-on-image projection overlay.")
    args = ap.parse_args()
    Viewer(args.triplet, args.detection, show_projection=not args.no_projection).run()


if __name__ == "__main__":
    main()
