"""Shared CV helpers: horizon detection, undistortion, keypoint extraction.

De-duplicated from the three copies that the original fisheye, thermal and
fusion scripts each carried (those standalone scripts have since been removed;
see the git history). The functions are byte-equivalent to those originals;
the only changes are typed args and an LRU undistort-map cache so the
viewer/sweep don't pay map construction per frame.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Undistortion
# ---------------------------------------------------------------------------

def _key_K(K: np.ndarray) -> tuple:
    return tuple(K.flatten().tolist())


def _key_D(D: np.ndarray) -> tuple:
    return tuple(D.flatten().tolist())


@lru_cache(maxsize=4)
def _fisheye_maps(w: int, h: int, K_key: tuple, D_key: tuple):
    K = np.array(K_key, dtype=np.float64).reshape(3, 3)
    D = np.array(D_key, dtype=np.float64).reshape(-1, 1)
    return cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )


def undistort_fisheye(frame: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    map1, map2 = _fisheye_maps(w, h, _key_K(K), _key_D(D))
    return cv2.remap(frame, map1, map2,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


@lru_cache(maxsize=4)
def _pinhole_new_K(w: int, h: int, K_key: tuple, D_key: tuple):
    K = np.array(K_key, dtype=np.float64).reshape(3, 3)
    D = np.array(D_key, dtype=np.float64).reshape(-1)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
    return new_K


def undistort_thermal(frame: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    new_K = _pinhole_new_K(w, h, _key_K(K), _key_D(D))
    return cv2.undistort(frame, K, D, None, new_K)


def repair_dead_rows(frame: np.ndarray, rows) -> np.ndarray:
    """Replace defective sensor rows by the mean of their nearest healthy
    neighbours (edge rows copy the one neighbour they have).

    The box's Lepton 3 has a stuck-hot FPA row (sensor row 21, present since
    at least 2026-07-14; row 98 in the 180-degree-rotated capture frame
    since 2026-08-14 — detection.yaml thermal.dead_rows). Apply BEFORE
    undistortion: it is a sensor row, and a remap would smear it into a
    curve. Recordings stay raw; every consumer (pipeline, thermal student
    labels, evals) repairs through this one function so train and deploy
    see identical frames. Empty/None rows -> the input is returned as-is
    (byte-identical path). Gray or multi-channel frames alike.
    """
    if not rows:
        return frame
    h = frame.shape[0]
    dead = sorted({int(r) for r in rows if 0 <= int(r) < h})
    if not dead:
        return frame
    out = frame.copy()
    dead_set = set(dead)
    for r in dead:
        up = r - 1
        while up >= 0 and up in dead_set:
            up -= 1
        dn = r + 1
        while dn < h and dn in dead_set:
            dn += 1
        if up >= 0 and dn < h:
            out[r] = ((frame[up].astype(np.float32) + frame[dn].astype(np.float32)) / 2.0
                      ).round().astype(frame.dtype)
        elif up >= 0:
            out[r] = frame[up]
        elif dn < h:
            out[r] = frame[dn]
    return out


def pinhole_new_K(K: np.ndarray, D: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Optimal new camera matrix for the undistorted (pinhole) image.

    `size` is (width, height). This is the matrix `undistort_thermal` uses
    as the destination intrinsics, so geometry that reasons about points in
    the undistorted thermal image must back-project with this, not the raw K.
    """
    w, h = size
    return _pinhole_new_K(w, h, _key_K(K), _key_D(D))


# ---------------------------------------------------------------------------
# Horizon detection (RANSAC on Canny edges)
# ---------------------------------------------------------------------------

def detect_horizon(
    frame: np.ndarray,
    ransac_iters: int = 200,
    inlier_thresh: float = 1.5,
    confidence_thresh: float = 0.5,
    fallback_row: int = 0,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_kernel: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, float, float, float]:
    """Find a single horizon line in `frame` and return a mask of pixels
    *below* it.

    Returns (mask, slope, intercept, confidence). If RANSAC fails or
    confidence < `confidence_thresh`, mask covers the area below
    `fallback_row` (0 = whole frame).

    The RANSAC sampling is seeded (`seed`) so the result is DETERMINISTIC for
    a given frame. This matters a lot for monocular range: at the low camera
    mount height an object near the horizon sits only a few pixels below it, so
    an unseeded horizon that jittered a few pixels run-to-run made the range
    estimate swing wildly. The best model is also refit by least squares on its
    inliers, which is more stable and accurate than the 2-point seed line.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    pts = np.column_stack(np.where(edges > 0))   # rows x [y,x]
    rows, cols = gray.shape

    if len(pts) < 2:
        return _whole_mask(gray, fallback_row), 0.0, float(fallback_row), 0.0

    best_inliers: np.ndarray = np.empty((0, 2))
    best_model: tuple[float, float] | None = None

    rng = np.random.RandomState(seed)
    for _ in range(ransac_iters):
        idxs = rng.choice(len(pts), 2, replace=False)
        (y1, x1), (y2, x2) = pts[idxs]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        distances = np.abs(slope * pts[:, 1] - pts[:, 0] + intercept) / np.sqrt(slope ** 2 + 1)
        inliers = pts[distances < inlier_thresh]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (slope, intercept)

    if best_model is None:
        return _whole_mask(gray, fallback_row), 0.0, float(fallback_row), 0.0

    slope, intercept = best_model
    # Refit the line on all inliers (least squares): steadier than the random
    # 2-point seed model, which sharpens the horizon row the range depends on.
    if len(best_inliers) >= 2:
        xs = best_inliers[:, 1].astype(np.float64)
        ys = best_inliers[:, 0].astype(np.float64)
        if xs.max() - xs.min() > 1e-6:
            slope, intercept = np.polyfit(xs, ys, 1)
    x_span = best_inliers[:, 1].max() - best_inliers[:, 1].min() if len(best_inliers) else 0
    confidence = float(x_span) / cols

    if confidence < confidence_thresh:
        return _whole_mask(gray, fallback_row), 0.0, float(fallback_row), confidence

    mask = np.zeros_like(gray, dtype=np.uint8)
    for x in range(cols):
        y_line = int(slope * x + intercept)
        y_line = max(0, min(rows - 1, y_line))
        mask[y_line:, x] = 255

    return mask, float(slope), float(intercept), confidence


def _whole_mask(gray: np.ndarray, fallback_row: int) -> np.ndarray:
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[max(0, fallback_row):, :] = 255
    return mask


#: detect_horizon's keyword parameters. The `horizon:` YAML block may carry
#: extra sub-blocks (e.g. `imu_seed`) that are consumed by the pipeline, not
#: by detect_horizon, always splat through `horizon_kwargs()`.
_HORIZON_KEYS = ("ransac_iters", "inlier_thresh", "confidence_thresh",
                 "fallback_row", "canny_low", "canny_high", "blur_kernel",
                 "seed")


def horizon_kwargs(cfg: dict) -> dict:
    """Filter a `horizon:` config block down to detect_horizon's kwargs."""
    return {k: v for k, v in (cfg or {}).items() if k in _HORIZON_KEYS}


def mask_below_line(shape: tuple[int, int], slope: float, intercept: float
                    ) -> np.ndarray:
    """Binary mask (255 below, 0 above) of the line row = slope*col +
    intercept: the same construction detect_horizon uses for its RANSAC
    line, exposed so an IMU-predicted horizon can produce an identical mask.
    """
    rows, cols = shape[:2]
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for x in range(cols):
        y_line = int(slope * x + intercept)
        y_line = max(0, min(rows - 1, y_line))
        mask[y_line:, x] = 255
    return mask


# ---------------------------------------------------------------------------
# Keypoint → bearing
# ---------------------------------------------------------------------------

def extract_kp(keypoints: Any, cx: float, pix_deg_ratio: float
               ) -> tuple[list[tuple[int, int]], list[float], list[float]]:
    """Convert blob keypoints to (pixel coords, sizes, azimuth-degrees)."""
    coords: list[tuple[int, int]] = []
    sizes: list[float] = []
    angles: list[float] = []
    for kp in keypoints:
        coords.append((int(kp.pt[0]), int(kp.pt[1])))
        sizes.append(float(kp.size))
        angles.append((kp.pt[0] - cx) / pix_deg_ratio)
    return coords, sizes, angles


# ---------------------------------------------------------------------------
# Blob-detector factory
# ---------------------------------------------------------------------------

def build_blob_detector(params_dict: dict) -> "cv2.SimpleBlobDetector":
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = float(params_dict.get("minThreshold", 180))
    params.maxThreshold = float(params_dict.get("maxThreshold", 255))
    params.filterByColor = True
    params.blobColor = int(params_dict.get("blobColor", 255))
    params.filterByArea = True
    params.minArea = float(params_dict.get("minArea", 20))
    params.filterByCircularity = bool(params_dict.get("filterByCircularity", False))
    params.filterByConvexity = bool(params_dict.get("filterByConvexity", False))
    params.filterByInertia = bool(params_dict.get("filterByInertia", False))
    return cv2.SimpleBlobDetector_create(params)
