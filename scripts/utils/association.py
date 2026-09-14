"""Pixel-level radar <-> camera detection association: Branch A.1 payoff.

With measured extrinsics (configs/extrinsics.yaml, 2026-07-07) a radar return
projects into the undistorted image within a few pixels, so "the radar sees the
same OBJECT the camera detected" is decidable per detection instead of the old
"both voted somewhere in the same 10° bin". Two things fall out:

  1. **Radar-ranged camera detections**: an associated detection takes the
     radar's range (accurate 0–9 m) instead of the monocular water-plane
     estimate, sidestepping the attitude/horizon sensitivity exactly where it
     matters (near field). The monocular value is kept for telemetry.
  2. **Confirmed fusion evidence**: a bin containing an associated detection
     carries object-level cross-sensor agreement, exported as a `confirmed`
     flag next to the score (stronger than coincidental same-bin votes).

Association is deliberately simple (padded-bbox point-in-box test): radar
azimuth res is 15° and detections are blobs, so anything fancier than a gate +
range aggregate would be false precision. All consumers are config-gated
(`fusion.association`, default off).

Instance-mask upgrade (2026-08-24, `fusion.association.use_instance_masks`):
the box gate's known floor is pixel ATTRIBUTION: a tall edge box sweeps up
foreground returns that merely project inside its rectangle (the 2-3 m
dock-clutter-in-pillar-boxes failure the range_compat_ratio rule mitigates).
When a YOLOv8-seg instance polygon covers a detection, the gate becomes
point-in-polygon (dilated by `instance_margin_px` to absorb projection +
mask-boundary error); detections no instance covers keep the box gate, so
coverage gaps degrade to the old behaviour, never to silence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class AssociationParams:
    """Gate + aggregation knobs (detection.yaml fusion.association)."""
    max_px: float = 32.0        # bbox padding, px, in the undistorted image
    min_points: int = 1         # radar points inside the gate to confirm
    range_agg: str = "min"      # min | median of matched radar forward ranges
    # Instance-mask gate (fisheye only: the det_seg polygons live in the
    # undistorted-fisheye frame). Off -> pure box gate, byte-identical.
    use_instance_masks: bool = False
    instance_margin_px: float = 12.0   # polygon dilation, px (measured; see
    #                                    docs/history/2026-08-24_instance_association.md)

    @classmethod
    def from_config(cls, cfg: dict | None) -> "AssociationParams":
        cfg = cfg or {}
        return cls(
            max_px=float(cfg.get("max_px", 32.0)),
            min_points=int(cfg.get("min_points", 1)),
            range_agg=str(cfg.get("range_agg", "min")),
            use_instance_masks=bool(cfg.get("use_instance_masks", False)),
            instance_margin_px=float(cfg.get("instance_margin_px", 12.0)),
        )


@dataclass
class DetectionMatch:
    """Radar evidence for one camera detection (aligned with coords/sizes)."""
    radar_range_m: float | None   # aggregated forward range of matched points
    n_points: int                 # how many radar points fell in the gate
    radar_bearing_deg: float | None = None  # median bearing of matched points
    # Indices (into the input point cloud) of the matched points: lets
    # consumers re-inspect them, e.g. the seg-silhouette arbitration
    # ("does this return project onto the object's own obstacle pixels?").
    point_indices: list[int] = None
    # Which pixel gate produced this match: "box" (padded bbox) or
    # "instance" (dilated instance polygon). Telemetry/eval only; the
    # range-attribution rules are gate-agnostic.
    gate: str = "box"


@dataclass
class AssociationResult:
    matches: list[DetectionMatch] = field(default_factory=list)

    @property
    def n_confirmed(self) -> int:
        return sum(1 for m in self.matches if m.radar_range_m is not None)


def polygons_from_instances(
    instances: list[dict],
    image_size: tuple[int, int],
) -> list[np.ndarray]:
    """Denormalise instance polygons to Nx2 float32 pixel arrays.

    `instances` are InstanceSegProvider records (`polygon` = flat
    [x1,y1,x2,y2,...] normalised to [0,1] in the undistorted-fisheye frame);
    `image_size` is (width, height). Degenerate polygons (< 3 vertices) are
    dropped.
    """
    w, h = image_size
    out: list[np.ndarray] = []
    for inst in instances or []:
        flat = inst.get("polygon") or []
        if len(flat) < 6:
            continue
        poly = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
        poly[:, 0] *= float(w)
        poly[:, 1] *= float(h)
        out.append(poly)
    return out


def match_polygons_to_detections(
    polys: list[np.ndarray],
    coords: list[tuple[int, int]],
    sizes: list[float],
) -> list[int | None]:
    """Pick the instance polygon covering each detection (or None).

    A polygon "covers" a detection when the blob centre lies inside it, or
    within the blob's own radius (size/2) of its boundary: blob centres sit
    slightly off the mask on glint halos / partial silhouettes. Preference
    order: the SMALLEST containing polygon (most specific: a buoy inside the
    dock's outline picks the buoy), else the nearest within tolerance.
    """
    out: list[int | None] = []
    contours = [poly.reshape(-1, 1, 2) for poly in polys]
    areas = [float(cv2.contourArea(c)) for c in contours]
    for (u, v), size in zip(coords, sizes):
        containing: list[tuple[float, int]] = []   # (area, idx)
        near: list[tuple[float, int]] = []         # (distance, idx)
        for k, c in enumerate(contours):
            d = cv2.pointPolygonTest(c, (float(u), float(v)), True)
            if d >= 0:
                containing.append((areas[k], k))
            elif d >= -(size / 2.0):
                near.append((-d, k))
        if containing:
            out.append(min(containing)[1])
        elif near:
            out.append(min(near)[1])
        else:
            out.append(None)
    return out


def _points_in_polygon(
    px: np.ndarray,
    poly: np.ndarray,
    margin_px: float,
) -> np.ndarray:
    """Boolean mask: projected pixels inside `poly` dilated by `margin_px`.

    Signed-distance containment via cv2.pointPolygonTest (>= -margin), with a
    cheap bbox+margin pre-filter so the exact test only runs on candidates.
    NaN rows never match.
    """
    inside = np.zeros(len(px), dtype=bool)
    finite = np.isfinite(px).all(axis=1)
    if not finite.any():
        return inside
    lo = poly.min(axis=0) - margin_px
    hi = poly.max(axis=0) + margin_px
    cand = finite & (px[:, 0] >= lo[0]) & (px[:, 0] <= hi[0]) \
        & (px[:, 1] >= lo[1]) & (px[:, 1] <= hi[1])
    contour = poly.reshape(-1, 1, 2)
    for j in np.where(cand)[0]:
        d = cv2.pointPolygonTest(contour, (float(px[j, 0]), float(px[j, 1])),
                                 True)
        inside[j] = d >= -margin_px
    return inside


def associate_radar_to_detections(
    points_xyz: np.ndarray,
    projected_px: np.ndarray,
    coords: list[tuple[int, int]],
    sizes: list[float],
    params: AssociationParams | None = None,
    det_polygons: list[np.ndarray | None] | None = None,
) -> AssociationResult:
    """Match projected radar points to blob detections by padded-bbox test.

    `points_xyz` is the (already y-filtered) Nx3 radar cloud from
    MMWaveResult; `projected_px` its Nx2 pixels in the SAME undistorted image
    the detections live in (geometry.project_radar_to_undistorted); NaN rows
    (behind camera) never match. `coords`/`sizes` are blob centres +
    diameters. A detection with >= min_points matched returns the aggregated
    forward range (radar Y, the convention min_ranges uses) and the median
    bearing of its matched points.

    `det_polygons` (optional, aligned with `coords`) upgrades individual
    detections to instance-level pixel attribution: where an entry is a
    Nx2 pixel polygon, the gate is point-in-polygon dilated by
    `params.instance_margin_px` instead of the padded bbox (None entries
    keep the box gate). Match aggregation is identical either way.
    """
    p = params or AssociationParams()
    matches: list[DetectionMatch] = []
    pts = np.asarray(points_xyz, dtype=np.float64)
    px = np.asarray(projected_px, dtype=np.float64)
    have_radar = len(pts) > 0 and len(px) == len(pts)
    for i, ((u, v), size) in enumerate(zip(coords, sizes)):
        poly = det_polygons[i] if det_polygons is not None else None
        gate = "instance" if poly is not None else "box"
        if not have_radar:
            matches.append(DetectionMatch(None, 0, gate=gate))
            continue
        if poly is not None:
            inside = _points_in_polygon(px, poly, p.instance_margin_px)
        else:
            r = size / 2.0 + p.max_px
            inside = (
                (px[:, 0] >= u - r) & (px[:, 0] <= u + r)
                & (px[:, 1] >= v - r) & (px[:, 1] <= v + r)
            )
            inside &= np.isfinite(px).all(axis=1)
        n = int(inside.sum())
        idxs = np.where(inside)[0].tolist()
        if n < p.min_points:
            matches.append(DetectionMatch(None, n, point_indices=idxs,
                                          gate=gate))
            continue
        y = pts[inside, 1]
        rng = float(np.median(y)) if p.range_agg == "median" else float(y.min())
        x = pts[inside, 0]
        bearing = float(np.degrees(np.median(np.arctan2(x, pts[inside, 1]))))
        matches.append(DetectionMatch(rng, n, bearing, point_indices=idxs,
                                      gate=gate))
    return AssociationResult(matches=matches)


def pinhole_bearings(
    coords: list[tuple[int, int]],
    P: np.ndarray,
) -> list[float]:
    """Bearing per detection from the calibrated pinhole geometry.

    theta = atan((u - cx_K) / fx), degrees, right positive: the geometric
    replacement for the legacy linear `(u - cx_magic) / pix_deg_ratio`
    (detection.yaml `<sensor>.bearing_model: pinhole`). Note the calibrated
    principal point differs from the magic cx (fisheye: 444.1 vs 472: a
    ~4° systematic bearing offset the radar A/B can arbitrate).
    """
    P = np.asarray(P, dtype=np.float64)
    fx, cx = P[0, 0], P[0, 2]
    return [float(np.degrees(np.arctan2(u - cx, fx))) for u, _v in coords]
