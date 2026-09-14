"""Cross-sensor geometry: Branch A.1 scaffolding.

Implements radar -> image projection and a single-image range estimate
under a flat-water-plane assumption. Works against the placeholder
configs/extrinsics.yaml; when V.1 measurements land, only the YAML
changes.

Coordinate conventions (kept consistent across the codebase):
    Radar (TI):    X right, Y forward, Z up.
    Camera (CV):   X right, Y down,    Z forward (along optical axis).

The 4x4 transforms in extrinsics.yaml express
`p_camera = T_radar_to_camera @ p_radar`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

EXTRINSICS_PATH = Path(__file__).resolve().parents[2] / "configs" / "extrinsics.yaml"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_extrinsics(path: str | Path = EXTRINSICS_PATH) -> dict[str, Any]:
    """Return the extrinsics dict with transforms as numpy 4x4 arrays.

    If the file does not exist, returns a dict with `measured=False` and
    identity transforms; callers should check `measured` before trusting
    the results.
    """
    if not Path(path).exists():
        return {
            "measured": False,
            "T_radar_to_fisheye": np.eye(4),
            "T_radar_to_thermal": np.eye(4),
            "camera_height_m": {"fisheye": 0.4, "thermal": 0.4},
        }
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    out: dict[str, Any] = {"measured": bool(raw.get("measured", False))}
    for key in ("T_radar_to_fisheye", "T_radar_to_thermal"):
        if key in raw:
            out[key] = np.array(raw[key], dtype=np.float64)
        else:
            out[key] = np.eye(4)
    out["camera_height_m"] = dict(raw.get("camera_height_m", {"fisheye": 0.4, "thermal": 0.4}))
    return out


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _apply_transform(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to Nx3 points. Returns Nx3."""
    if len(points_xyz) == 0:
        return np.empty((0, 3))
    homog = np.hstack([points_xyz, np.ones((len(points_xyz), 1))])
    return (T @ homog.T).T[:, :3]


def project_radar_to_fisheye(
    points_xyz: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    T_radar_to_cam: np.ndarray,
) -> np.ndarray:
    """Project Nx3 radar points (TI frame, metres) to Nx2 fisheye pixels.

    Uses the OpenCV fisheye projection model (Kannala-Brandt).
    Returns Nx2 array; rows for points behind the camera (camera Z <= 0)
    are NaN.
    """
    return _project(points_xyz, K, D, T_radar_to_cam, model="fisheye")


def project_radar_to_thermal(
    points_xyz: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    T_radar_to_cam: np.ndarray,
) -> np.ndarray:
    """Project Nx3 radar points to Nx2 thermal pixels via standard pinhole."""
    return _project(points_xyz, K, D, T_radar_to_cam, model="pinhole")


def _project(points_xyz: np.ndarray, K: np.ndarray, D: np.ndarray,
             T: np.ndarray, model: str) -> np.ndarray:
    n = len(points_xyz)
    result = np.full((n, 2), np.nan)
    if n == 0:
        return result
    cam_pts = _apply_transform(points_xyz, T)
    visible = cam_pts[:, 2] > 1e-6  # in front of camera
    if not visible.any():
        return result
    visible_idx = np.where(visible)[0]
    pts_v = cam_pts[visible].astype(np.float64)
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    if model == "fisheye":
        pts_in = pts_v.reshape(-1, 1, 3)
        D_v = np.asarray(D, dtype=np.float64).reshape(-1, 1)
        img_pts, _ = cv2.fisheye.projectPoints(pts_in, rvec, tvec, K, D_v)
        img_pts = img_pts.reshape(-1, 2)
    else:
        D_v = np.asarray(D, dtype=np.float64).reshape(-1)
        img_pts, _ = cv2.projectPoints(pts_v, rvec, tvec, K, D_v)
        img_pts = img_pts.reshape(-1, 2)
    result[visible_idx] = img_pts
    return result


def project_radar_to_undistorted(
    points_xyz: np.ndarray,
    P: np.ndarray,
    T_radar_to_cam: np.ndarray,
) -> np.ndarray:
    """Project Nx3 radar points into an UNDISTORTED image (pure pinhole).

    `project_radar_to_{fisheye,thermal}` land points in the RAW image (they
    apply the lens distortion model). Detections, seg masks, and overlays
    live in the undistorted image instead, whose geometry is a plain pinhole
    with camera matrix `P`:
      - fisheye: the original K (cv_common.undistort_fisheye remaps with K as
        the new camera matrix);
      - thermal: cv_common.pinhole_new_K's optimal matrix.

    Returns Nx2 pixels; rows behind the camera are NaN. This is the right
    projection for pixel-level radar<->camera association.
    """
    n = len(points_xyz)
    result = np.full((n, 2), np.nan)
    if n == 0:
        return result
    cam = _apply_transform(np.asarray(points_xyz, dtype=np.float64)[:, :3],
                           T_radar_to_cam)
    z = cam[:, 2]
    vis = z > 1e-6
    if not vis.any():
        return result
    P = np.asarray(P, dtype=np.float64)
    fx, fy = P[0, 0], P[1, 1]
    cx, cy = P[0, 2], P[1, 2]
    result[vis, 0] = fx * cam[vis, 0] / z[vis] + cx
    result[vis, 1] = fy * cam[vis, 1] / z[vis] + cy
    return result


# ---------------------------------------------------------------------------
# Range from water plane
# ---------------------------------------------------------------------------

def range_from_water_plane(
    bbox_xyxy: tuple[float, float, float, float],
    K: np.ndarray,
    D: np.ndarray,
    camera_height_m: float,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0,
    model: str = "fisheye",
    image_size: tuple[int, int] | None = None,
) -> float | None:
    """Estimate range to a detected object assuming it sits on a flat water plane.

    The bbox bottom-centre pixel is back-projected to a 3-D ray in the
    camera frame; the ray is then rotated by the boat pitch/roll (rotation
    of the water plane relative to the camera) and intersected with the
    plane Y = +camera_height_m (camera Y points down).

    `bbox_xyxy` is given in PIXEL coordinates. If `image_size = (w,h)` is
    provided and any bbox component exceeds 1.5, the coords are treated
    as pixels directly; otherwise as normalised [0,1] scaled by image_size.

    Returns metres, or None if the ray points above the horizon (no
    intersection) or below the camera (negative range).
    """
    x0, y0, x1, y1 = bbox_xyxy
    if image_size is not None and max(x0, y0, x1, y1) <= 1.5:
        w, h = image_size
        x0, y0, x1, y1 = x0 * w, y0 * h, x1 * w, y1 * h
    px = np.array([[(x0 + x1) / 2.0, y1]], dtype=np.float64).reshape(1, 1, 2)

    if model == "fisheye":
        D_v = np.asarray(D, dtype=np.float64).reshape(-1, 1)
        norm = cv2.fisheye.undistortPoints(px, K, D_v)
    else:
        D_v = np.asarray(D, dtype=np.float64).reshape(-1)
        norm = cv2.undistortPoints(px, K, D_v)
    nx, ny = float(norm[0, 0, 0]), float(norm[0, 0, 1])
    return _range_from_normalized_ray(nx, ny, camera_height_m, pitch_rad, roll_rad)


def _range_from_normalized_ray(
    nx: float,
    ny: float,
    camera_height_m: float,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0,
) -> float | None:
    """Intersect a normalised camera ray (x/z, y/z, 1) with the water plane.

    The plane is Y = +camera_height_m in the camera frame (camera Y points
    down). Returns the slant range in metres, or None if the ray points at
    or above the horizon (no forward intersection).
    """
    ray = np.array([nx, ny, 1.0], dtype=np.float64)

    # Rotate ray by IMU pitch/roll. Pitch about camera X axis (nose up = +pitch),
    # roll about camera Z axis. Build R_world_to_cam and apply its inverse.
    if pitch_rad != 0.0 or roll_rad != 0.0:
        cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
        cr, sr = np.cos(roll_rad), np.sin(roll_rad)
        R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        R_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        ray = R_roll @ R_pitch @ ray

    if ray[1] <= 1e-6:
        return None  # ray points up; no water-plane intersection
    t = camera_height_m / ray[1]
    if t <= 0:
        return None
    point = t * ray
    return float(np.linalg.norm(point))


def range_from_undistorted_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    P: np.ndarray,
    camera_height_m: float,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0,
) -> float | None:
    """Water-plane range for a bbox already in UNDISTORTED pixel coordinates.

    Unlike `range_from_water_plane` (which takes raw, distorted pixels and
    removes distortion via the lens model), this assumes the detection lives
    in a rectified image and back-projects with a plain pinhole inverse using
    `P`, the camera matrix of that undistorted image:
      - fisheye: the same K passed as the new matrix to
        `cv2.fisheye.initUndistortRectifyMap` (see cv_common.undistort_fisheye);
      - thermal: the optimal new K from `cv2.getOptimalNewCameraMatrix`
        (see cv_common.pinhole_new_K).

    The bbox bottom-centre is taken as the waterline contact point. Returns
    metres, or None if it back-projects above the horizon.
    """
    x0, _y0, x1, y1 = bbox_xyxy
    u = (x0 + x1) / 2.0
    v = y1
    fx, fy = float(P[0, 0]), float(P[1, 1])
    cx, cy = float(P[0, 2]), float(P[1, 2])
    if fx == 0.0 or fy == 0.0:
        return None
    nx = (u - cx) / fx
    ny = (v - cy) / fy
    return _range_from_normalized_ray(nx, ny, camera_height_m, pitch_rad, roll_rad)


# ---------------------------------------------------------------------------
# Attitude-aware range: uses an explicit "up" vector instead of assuming the
# camera is level. The up vector (world up / gravity, expressed in camera
# coordinates) can come from the IMU (BNO085) or, when no IMU is connected,
# from the per-frame detected horizon line. Both reduce to the level case
# up = (0, -1, 0) (camera Y points down, so world up is -Y).
# ---------------------------------------------------------------------------

# Level camera: world up is straight up = -Y in the camera frame.
UP_LEVEL = np.array([0.0, -1.0, 0.0])


def up_from_pitch_roll(pitch_rad: float, roll_rad: float) -> np.ndarray:
    """World-up unit vector in camera coordinates from boat pitch/roll.

    Matches the rotation convention in `_range_from_normalized_ray`
    (pitch about camera X, nose-up positive; roll about camera Z): a ray is
    transformed to the level frame by `R_roll @ R_pitch`, so level-up maps
    back into the camera frame by the inverse. With pitch=roll=0 this is
    exactly UP_LEVEL.
    """
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    R_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    up = R_pitch.T @ R_roll.T @ UP_LEVEL
    n = np.linalg.norm(up)
    return up / n if n else UP_LEVEL.copy()


def up_from_horizon_line(slope: float, intercept: float, P: np.ndarray) -> np.ndarray:
    """World-up unit vector in camera coords from a detected horizon line.

    The horizon line (row = slope*col + intercept, the cv_common.detect_horizon
    convention) is the image of the level plane at infinity; its pre-image is a
    plane through the optical centre whose normal is world-up. For an image line
    l = (a, b, c) with a*u + b*v + c = 0, that plane normal in camera coords is
    P^T @ l. Here a*u + b*v + c = slope*u - v + intercept = 0, so
    l = (slope, -1, intercept). The sign is chosen so up points to -Y (level
    case gives exactly UP_LEVEL).

    Only meaningful when the horizon was detected with adequate confidence;
    the caller must gate on that (a low-confidence detect_horizon returns
    intercept=0, which would put the reference at the image top).
    """
    l = np.array([float(slope), -1.0, float(intercept)])
    up = np.asarray(P, dtype=np.float64).T @ l
    n = np.linalg.norm(up)
    if n == 0:
        return UP_LEVEL.copy()
    up = up / n
    if up[1] > 0:          # ensure it points "up" (camera up is -Y)
        up = -up
    return up


def pitch_roll_from_up(up_cam: np.ndarray) -> tuple[float, float]:
    """(pitch_rad, roll_rad) implied by a world-up vector in camera coords.

    Exact inverse of `up_from_pitch_roll`, which produces
    up = (-sin r, -cos r cos p, cos r sin p). Used to compare attitude
    sources (IMU vs water-edge vs RANSAC horizon) in a common space.
    """
    up = np.asarray(up_cam, dtype=np.float64)
    n = np.linalg.norm(up)
    if n == 0:
        return 0.0, 0.0
    up = up / n
    roll = float(np.arcsin(np.clip(-up[0], -1.0, 1.0)))
    pitch = float(np.arctan2(up[2], -up[1]))
    return pitch, roll


def horizon_line_from_up(up_cam: np.ndarray, P: np.ndarray
                         ) -> tuple[float, float]:
    """(slope, intercept) of the horizon line implied by a world-up vector.

    Inverse of `up_from_horizon_line`: the horizon is the image of the level
    plane at infinity, i.e. the line l = P^{-T} @ up, written back in the
    detect_horizon convention row = slope*col + intercept. Lets an IMU
    attitude predict where the horizon must be: used to seed/gate the RANSAC
    line (horizon.imu_seed) and to draw attitude overlays.
    """
    P = np.asarray(P, dtype=np.float64)
    l = np.linalg.solve(P.T, np.asarray(up_cam, dtype=np.float64))
    a, b, c = l
    if abs(b) < 1e-12:
        return 0.0, 0.0   # degenerate (horizon vertical): caller gates
    return float(-a / b), float(-c / b)


def range_from_water_plane_up(
    bbox_xyxy: tuple[float, float, float, float],
    P: np.ndarray,
    up_cam: np.ndarray,
    camera_height_m: float,
    max_range_m: float | None = None,
) -> float | None:
    """Water-plane range for an UNDISTORTED bbox given the world-up vector.

    Generalises `range_from_undistorted_bbox` to an arbitrary camera attitude:
    the water plane sits `camera_height_m` below the optical centre along
    `up_cam`. The bbox bottom-centre ray is intersected with that plane.

    Returns metres, or None if the ray does not point below the horizon
    (no forward intersection) or the result exceeds `max_range_m`: beyond
    which a monocular estimate is not trustworthy (let the radar own it).
    """
    x0, _y0, x1, y1 = bbox_xyxy
    u = (x0 + x1) / 2.0
    v = y1
    fx, fy = float(P[0, 0]), float(P[1, 1])
    cx, cy = float(P[0, 2]), float(P[1, 2])
    if fx == 0.0 or fy == 0.0:
        return None
    d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    up = np.asarray(up_cam, dtype=np.float64)
    denom = float(up @ d)
    if denom >= -1e-9:
        return None  # ray at or above the horizon; no water intersection
    t = -camera_height_m / denom
    if t <= 0:
        return None
    rng = float(t * np.linalg.norm(d))
    if max_range_m is not None and rng > max_range_m:
        return None
    return rng


def range_from_contact_point(
    contact_xy: tuple[float, float],
    P: np.ndarray,
    up_cam: np.ndarray,
    camera_height_m: float,
    max_range_m: float | None = None,
) -> float | None:
    """Water-plane range from an EXPLICIT waterline-contact pixel.

    Same geometry as `range_from_water_plane_up`, but the contact point is given
    directly (e.g. from water segmentation: scripts/utils/segmentation.
    water_edge_contact) rather than taken as the bbox bottom-centre. This is the
    fix for tilted-horizon boxes that over-capture water and report objects as
    far too close: the segmentation contact sits at the true hull/water line,
    typically above the box bottom, yielding a correctly larger range.

    `contact_xy` is (u, v) in the UNDISTORTED image; `P` is that image's camera
    matrix; `up_cam` is world-up in camera coords (IMU or water-edge derived).
    Returns metres, or None if the ray is at/above the horizon or beyond
    `max_range_m`.
    """
    u, v = float(contact_xy[0]), float(contact_xy[1])
    fx, fy = float(P[0, 0]), float(P[1, 1])
    cx, cy = float(P[0, 2]), float(P[1, 2])
    if fx == 0.0 or fy == 0.0:
        return None
    d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    up = np.asarray(up_cam, dtype=np.float64)
    denom = float(up @ d)
    if denom >= -1e-9:
        return None  # ray at or above the horizon; no water intersection
    t = -camera_height_m / denom
    if t <= 0:
        return None
    rng = float(t * np.linalg.norm(d))
    if max_range_m is not None and rng > max_range_m:
        return None
    return rng
