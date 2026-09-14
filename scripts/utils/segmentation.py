"""Water-segmentation consumption for robust monocular range.

A learned water/sky/obstacle segmentation (eWaSR trained on LaRS; see
docs/reference/segmentation.md, or the internal docs/lars_segmentation_plan.md)
gives, per undistorted fisheye frame:

  * the **water edge** (water<->non-water boundary), a robust horizon /
    up-vector that replaces the tilt-fragile RANSAC-Canny line, and
  * per detection, the **true waterline-contact point**: the lowest obstacle
    pixel that borders water inside the box: instead of the bbox bottom, which
    over-captures water when the horizon is tilted and reports objects as far
    too close.

Everything here is pure numpy (no torch) and operates on a class-id mask
(0=obstacle, 1=water, 2=sky) in the SAME coordinate space as the range
geometry: the undistorted fisheye image. Masks are produced on the GPU
(scripts/gpu_seg/predict_masks.py) and read back into data/seg/.

This module is consumed by scripts/sensor_processing/pipeline.py behind the
`segmentation:` block of configs/detection.yaml (default off), so absent masks
leave the existing behaviour byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Class ids (the LaRS / MaSTr / eWaSR convention).
OBSTACLE = 0
WATER = 1
SKY = 2

# Colour-coded fallback decoding (eWaSR SEGMENTATION_COLORS order: obs/water/sky).
_SEG_COLORS = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], np.uint8)


# ---------------------------------------------------------------------------
# Mask IO
# ---------------------------------------------------------------------------

def load_seg_mask(path: str | Path) -> np.ndarray:
    """Load a segmentation mask as an HxW uint8 of class ids {0,1,2}.

    Accepts either a label-encoded grayscale PNG (values already class ids,
    with anything >2 treated as ignore->obstacle-neutral) or an RGB
    colour-coded PNG (nearest of the eWaSR palette).
    """
    arr = np.array(Image.open(path))
    return decode_seg_array(arr)


def decode_seg_array(arr: np.ndarray) -> np.ndarray:
    """Normalise a raw mask array to HxW class ids {0,1,2}."""
    if arr.ndim == 3:
        # int32: squared 8-bit colour diffs (up to ~65k) overflow int16.
        rgb = arr[:, :, :3].astype(np.int32)
        # nearest palette colour per pixel
        d = ((rgb[:, :, None, :] - _SEG_COLORS[None, None, :, :].astype(np.int32))
             ** 2).sum(-1)
        return d.argmin(-1).astype(np.uint8)
    out = arr.astype(np.uint8).copy()
    # Any non-{0,1,2} value (e.g. 4 or 255 ignore) -> leave as obstacle-neutral
    # by clamping to a value that the consumers treat as "not water/not sky".
    out[(out != WATER) & (out != SKY)] = OBSTACLE
    return out


# ---------------------------------------------------------------------------
# Waterline-contact point for a detection
# ---------------------------------------------------------------------------

@dataclass
class ContactParams:
    search_pad_px: int = 40     # extend the vertical search beyond the bbox
    min_columns: int = 5        # need this many columns with a valid contact
    water_below_px: int = 3     # require water within this many px below obstacle
    percentile: float = 80.0    # robust "deepest contact" percentile across cols


def water_edge_contact(
    seg: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    params: ContactParams | None = None,
) -> tuple[float, float] | None:
    """Return the (u, v) waterline-contact pixel for a detection, or None.

    Within the bbox's horizontal extent (and a small vertical pad, since a tight
    box often clips the waterline), for each column we find the *deepest*
    obstacle pixel that has water just below it: the per-column hull/water
    contact. The object's contact row is a high percentile of those per-column
    rows (robust to a few noisy columns); the contact column is their median.

    Returns None when too few columns show an obstacle->water transition (e.g.
    the detection is sky/glint, fully surrounded by water, or off the water
    edge), so the caller can fall back to the bbox-bottom estimate.
    """
    p = params or ContactParams()
    h, w = seg.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    c0 = max(0, int(np.floor(min(x0, x1))))
    c1 = min(w - 1, int(np.ceil(max(x0, x1))))
    r0 = max(0, int(np.floor(min(y0, y1))) - p.search_pad_px)
    r1 = min(h - 1, int(np.ceil(max(y0, y1))) + p.search_pad_px)
    if c1 < c0 or r1 <= r0:
        return None

    k = max(1, int(p.water_below_px))
    # Vectorised over the whole search window (2026-09-03): the per-column
    # Python walk cost ~30 ms per detection on the Pi 4 (65 detections/frame
    # on a dock scene = 2 s per tick, 80% of the live shadow tick). Same
    # rule, same answer: per column, the DEEPEST obstacle row that has water
    # within k rows below it inside the window (rows below the window are
    # not consulted, exactly as the slice `col[ro+1:ro+1+k]` never was).
    win = seg[r0:r1 + 1, c0:c1 + 1]
    is_obs = win == OBSTACLE
    is_water = win == WATER
    water_below = np.zeros_like(is_obs)
    for j in range(1, k + 1):
        water_below[:-j] |= is_water[j:]
    ok = is_obs & water_below
    has = ok.any(axis=0)
    if int(has.sum()) < p.min_columns:
        return None
    n_rows = ok.shape[0]
    # argmax over the row-reversed window = the deepest True row per column.
    deepest = (n_rows - 1) - np.argmax(ok[::-1], axis=0)
    contact_rows = r0 + deepest[has]
    contact_cols = c0 + np.flatnonzero(has)

    v = float(np.percentile(contact_rows, p.percentile))
    u = float(np.median(contact_cols))
    return u, v


# ---------------------------------------------------------------------------
# Water-edge horizon (robust up-vector source)
# ---------------------------------------------------------------------------

def horizon_from_water(
    seg: np.ndarray,
    min_columns_frac: float = 0.3,
    trim_sigma: float = 2.0,
) -> tuple[float, float, float]:
    """Fit the water-edge line and return (slope, intercept, confidence).

    The line is `row = slope*col + intercept` (the cv_common.detect_horizon
    convention), fit to the topmost water pixel of each column. Obstacles
    protruding above the open-water line show up as outliers and are trimmed by
    one robust pass. Confidence combines coverage (fraction of columns that have
    a water boundary) with straightness (1 - normalised residual). A frame with
    little/no water returns confidence 0.
    """
    h, w = seg.shape[:2]
    cols = np.arange(w)
    top_water = np.full(w, -1, dtype=np.int64)
    water = seg == WATER
    has_water = water.any(axis=0)
    # argmax on a boolean column gives the first (topmost) True row.
    top_water[has_water] = water[:, has_water].argmax(axis=0)

    valid = top_water >= 0
    coverage = float(valid.sum()) / max(1, w)
    if valid.sum() < max(2, int(min_columns_frac * w)):
        return 0.0, 0.0, 0.0

    xs = cols[valid].astype(np.float64)
    ys = top_water[valid].astype(np.float64)

    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    std = float(resid.std()) or 1.0
    keep = np.abs(resid) < trim_sigma * std
    if keep.sum() >= 2 and keep.sum() < xs.size:
        slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
        resid = ys[keep] - (slope * xs[keep] + intercept)

    rms = float(np.sqrt(np.mean(resid ** 2)))
    straightness = 1.0 / (1.0 + rms / max(1.0, 0.02 * h))
    confidence = float(coverage * straightness)
    return float(slope), float(intercept), confidence


def water_fraction(seg: np.ndarray) -> float:
    """Fraction of the frame labelled water (used for gating / telemetry)."""
    return float((seg == WATER).mean()) if seg.size else 0.0


def detection_obstacle_fraction(
    seg: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
) -> float:
    """Fraction of OBSTACLE-class pixels inside a detection box.

    A detection the segmentation calls water (a glint / wave / reflection: the
    classic night/open-water false positive) scores ~0; a real obstacle (boat,
    buoy, structure) scores high. Used by the seg FP filter to drop detections
    with little obstacle support.
    """
    h, w = seg.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    c0 = max(0, int(np.floor(min(x0, x1))))
    c1 = min(w, int(np.ceil(max(x0, x1))))
    r0 = max(0, int(np.floor(min(y0, y1))))
    r1 = min(h, int(np.ceil(max(y0, y1))))
    if c1 <= c0 or r1 <= r0:
        return 0.0
    patch = seg[r0:r1, c0:c1]
    return float((patch == OBSTACLE).mean())


# ---------------------------------------------------------------------------
# Navigable free-space profile (Phase 1: segmentation-driven navigation)
# ---------------------------------------------------------------------------

@dataclass
class FreeSpaceProfile:
    """Per-azimuth navigable free distance from the water mask."""
    bin_centers: np.ndarray            # azimuth degrees, len n_bins
    free_dist_m: list[float | None]    # nearest navigable limit (m); None = open past max_range
    blocked: list[bool]                # True where there is no near water (blocked at the boat)
    boundary_rows: np.ndarray          # per-column top-of-navigable-water row (for overlay)


def free_space_profile(
    seg: np.ndarray,
    P: np.ndarray,
    up_cam: np.ndarray,
    camera_height_m: float,
    cx: float,
    pix_deg_ratio: float,
    bin_edges: np.ndarray,
    max_range_m: float | None = None,
) -> FreeSpaceProfile:
    """Turn the water mask into a per-bearing "how far can I go" profile.

    Per column the navigable extent is the contiguous run of WATER pixels from
    the bottom of the frame; its top edge is the nearest non-navigable boundary
    (an obstacle in the water, or the shore / water edge). That boundary pixel is
    back-projected onto the water plane to a range. Per azimuth bin we keep the
    MIN range over its columns (conservative for navigation).

    `free_dist_m[i]` is metres to the nearest limit in bin i; `None` if the bin
    is open past `max_range_m` (water runs to the horizon); `blocked[i]` is True
    when a column had no near water at all. Bearing per column uses the same
    `(col - cx) / pix_deg_ratio` convention as `extract_kp`.

    Assumes the near field (bottom of the undistorted frame) is water: true for
    a forward-looking low-mounted camera. cf. WaSR free-space.
    """
    from scripts.utils.geometry import range_from_contact_point

    h, w = seg.shape[:2]
    water = seg == WATER
    # contiguous water rows from the bottom, per column
    contig = np.cumprod(water[::-1, :], axis=0).astype(bool)
    count = contig.sum(axis=0)                       # navigable rows from bottom
    boundary_rows = (h - count).astype(np.int64)     # first non-water going up (h if all water)

    n_bins = len(bin_edges) - 1
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    best = [np.inf] * n_bins
    blocked = [False] * n_bins

    for c in range(w):
        bearing = (c - cx) / pix_deg_ratio
        if bearing < bin_edges[0] or bearing >= bin_edges[-1]:
            continue
        bi = int(np.searchsorted(bin_edges, bearing, side="right") - 1)
        cnt = int(count[c])
        if cnt == 0:                                 # no near water -> blocked
            blocked[bi] = True
            best[bi] = min(best[bi], 0.0)
            continue
        if cnt >= h:                                 # all water -> open
            continue
        rng = range_from_contact_point(
            (float(c), float(boundary_rows[c])), P, up_cam, camera_height_m,
            max_range_m=max_range_m,
        )
        if rng is None:                              # boundary above horizon -> open
            continue
        best[bi] = min(best[bi], rng)

    free = [None if b == np.inf else float(b) for b in best]
    return FreeSpaceProfile(bin_centers=centers, free_dist_m=free,
                            blocked=blocked, boundary_rows=boundary_rows)


class FreeSpaceSmoother:
    """Temporal EMA over the per-bin free-space distance.

    The per-frame profile flickers (a one-frame spurious close reading, or a
    momentary mask gap); for steady navigation we damp it. `None` (open to the
    horizon) is treated as `max_range_m` for the EMA, so a transient close
    reading is pulled toward open over a few frames rather than trusted
    instantly, and vice-versa. One instance per consumer (per dashboard / per
    `fusion.py --out` run); call `update` per frame in order.

    `alpha` is the weight on the current frame (1.0 = no smoothing).
    """

    def __init__(self, alpha: float = 0.5, max_range_m: float = 15.0):
        self.alpha = float(alpha)
        self.max = float(max_range_m)
        self._ema: np.ndarray | None = None

    def reset(self) -> None:
        self._ema = None

    def update(self, free_dist_m: list[float | None]) -> list[float | None]:
        vals = np.array([self.max if d is None else float(d) for d in free_dist_m],
                        dtype=np.float64)
        if self._ema is None or len(self._ema) != len(vals):
            self._ema = vals.copy()
        else:
            self._ema = self.alpha * vals + (1.0 - self.alpha) * self._ema
        # report only genuinely-open bins (EMA at the cap) as None
        return [None if v >= 0.999 * self.max else float(v) for v in self._ema]


# ---------------------------------------------------------------------------
# Segmentation-driven obstacle detection (Phase B of "both, staged")
# ---------------------------------------------------------------------------

@dataclass
class SegDetectParams:
    min_area: int = 25          # drop components smaller than this (px)
    water_edge_margin_px: int = 10   # keep obstacles below (edge_row - margin)
    max_components: int = 100   # safety cap


def seg_obstacle_detections(
    seg: np.ndarray,
    params: SegDetectParams | None = None,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Detect discrete obstacles as connected components of the obstacle class
    that sit in/near the water region.

    WaSR-style: the obstacle class covers shore + dynamic objects, so we keep
    only components whose top is below the water edge (minus a margin): i.e.
    things floating in or rising out of the water, and discard the large
    above-horizon background (sky/shore). Returns (coords, sizes) with the same
    meaning as cv_common.extract_kp output: centres and blob "diameters".

    cv2 is used for connected components (available on the laptop too).
    """
    import cv2  # local import keeps module import cheap / cv2-optional

    p = params or SegDetectParams()
    h, w = seg.shape[:2]
    obstacle = (seg == OBSTACLE).astype(np.uint8)
    if obstacle.sum() == 0:
        return [], []

    slope, intercept, conf = horizon_from_water(seg)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(obstacle, 8)

    coords: list[tuple[int, int]] = []
    sizes: list[float] = []
    for i in range(1, n):                       # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < p.min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        # Water-edge gate: skip components entirely above the edge (background).
        if conf > 0.0:
            edge_at_cx = slope * cx + intercept
            comp_bottom = y + ch
            if comp_bottom < edge_at_cx - p.water_edge_margin_px:
                continue
        coords.append((int(round(cx)), int(round(cy))))
        sizes.append(float(max(cw, ch)))
        if len(coords) >= p.max_components:
            break
    return coords, sizes


# ---------------------------------------------------------------------------
# Frame-id -> mask provider
# ---------------------------------------------------------------------------

def _clip_dirname(scene: str, triplet_ts: str) -> str:
    return f"{scene}__{triplet_ts}"


def mask_path_for_frame_id(seg_root: str | Path, frame_id: str) -> Path | None:
    """Map a dashboard frame_id to its mask file on disk.

    frame_id is `<scene>/<triplet_ts>/ts=<HH-MM-SS.f>` (dashboard._frame_id_for).
    The mask lives at <seg_root>/<scene>__<triplet_ts>/ts=<...>.png (the layout
    export_undistorted_frames.py + predict_masks.py produce).
    """
    parts = frame_id.split("/")
    if len(parts) < 3:
        return None
    scene, triplet_ts, ts_token = parts[0], parts[1], parts[-1]
    return Path(seg_root) / _clip_dirname(scene, triplet_ts) / f"{ts_token}.png"


@dataclass
class SegProvider:
    """Lazy, cached reader of per-frame masks keyed by dashboard frame_id."""

    seg_root: Path
    _cache: dict[str, np.ndarray | None] = field(default_factory=dict)

    def __init__(self, seg_root: str | Path):
        self.seg_root = Path(seg_root)
        self._cache = {}

    def available(self) -> bool:
        return self.seg_root.is_dir()

    def get(self, frame_id: str | None) -> np.ndarray | None:
        if not frame_id:
            return None
        if frame_id in self._cache:
            return self._cache[frame_id]
        path = mask_path_for_frame_id(self.seg_root, frame_id)
        mask = load_seg_mask(path) if path and path.exists() else None
        self._cache[frame_id] = mask
        return mask
