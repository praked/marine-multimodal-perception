"""Cross-frame motion estimation: Phase I.4.9.

The radar TLV format carries a per-point radial velocity (Doppler), but
the Pi-side parser drops it before writing the CSV (see CLAUDE.md §4c,
PLAN.md §I.4.7 deferred upstream fix). Until that parser is fixed,
motion has to be derived from cross-frame association: take the
previous frame's point cloud, greedy-match each current point to its
nearest previous neighbour within a gate, and difference positions.

Design contract:

  - `RadarPointTracker` is purely positional. It does not know about
    own-boat motion; when the sailboat itself is moving at ~1 m/s,
    every static object reads as approaching at 1 m/s. PLAN.md V.3
    + I.4.7 deal with this once the IMU/GPS bridge lands.

  - `aggregate_bin_velocity` reduces per-point velocities to one
    closing-speed per bin, by magnitude-weighted median over the
    radial (Y) component. The magnitude weighting makes the bin
    velocity robust to a single bad NN match.

  - `radar_velocity_from_doppler` is the future-proof entry point.
    When the parser fix lands and CSVs carry a `V` column, callers
    pass it here directly; the tracker's role downgrades to providing
    velocities for points that lacked Doppler (unusual). The fields
    on `MMWaveResult` stay identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def parse_radar_timestamp(ts: str | None) -> float | None:
    """Parse "HH:MM:SS.f" → seconds since midnight. None passes through."""
    if ts is None:
        return None
    try:
        dt = datetime.strptime(ts, "%H:%M:%S.%f")
    except ValueError:
        return None
    return dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1e6


def timestamp_delta_s(curr: str | None, prev: str | None) -> float | None:
    """Δt in seconds between two radar timestamps. None if either invalid
    or if the delta is non-positive after a midnight-rollover correction.
    """
    a = parse_radar_timestamp(curr)
    b = parse_radar_timestamp(prev)
    if a is None or b is None:
        return None
    dt = a - b
    if dt < -80000.0:
        # Both timestamps are seconds-since-midnight, so a stream that
        # crosses 00:00 sees prev≈86400 and curr≈small, giving dt≈-86400.
        # 80000 ≈ 22 h tolerance: well past any plausible chunk length
        # so we won't snap real clock rewinds back to positive.
        dt += 86400.0
    if dt <= 0 or dt > 5.0:
        # >5 s gap = stream gap, treat as fresh start. The 5 s window
        # comfortably exceeds the 100 ms radar frame at the upstream
        # 10 Hz cadence; longer gaps mean we shouldn't be matching
        # across them anyway.
        return None
    return dt


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@dataclass
class RadarPointTrackerConfig:
    gate_m: float = 0.5        # max NN distance to count as a match
    max_dt_s: float = 0.5      # ignore prior frame older than this


class RadarPointTracker:
    """Greedy-nearest-neighbour position-to-velocity associator.

    Stateless across instantiations; holds only the most recent frame's
    points + timestamp. Call `update(points_xyz, timestamp)` per frame;
    receive per-point `(vx, vy)` velocity vectors aligned with the input
    `points_xyz`. Unmatched (no prior neighbour within gate) points get
    `None`.
    """

    def __init__(self, config: RadarPointTrackerConfig | None = None):
        self.config = config or RadarPointTrackerConfig()
        self._prev_pts: np.ndarray | None = None
        self._prev_ts: str | None = None

    def reset(self) -> None:
        self._prev_pts = None
        self._prev_ts = None

    def update(
        self,
        points_xyz: np.ndarray,
        timestamp: str | None,
    ) -> list[tuple[float, float] | None]:
        """Return per-point velocity vectors (vx, vy) in m/s.

        First call always returns all-`None` since there is no prior
        frame to difference against. After that, `None` entries indicate
        no prior point within `gate_m` of the current point.
        """
        n = len(points_xyz)
        velocities: list[tuple[float, float] | None] = [None] * n

        if n == 0:
            self._prev_pts = points_xyz
            self._prev_ts = timestamp
            return velocities

        if self._prev_pts is None or len(self._prev_pts) == 0:
            self._prev_pts = points_xyz
            self._prev_ts = timestamp
            return velocities

        dt = timestamp_delta_s(timestamp, self._prev_ts)
        if dt is None or dt > self.config.max_dt_s:
            # Stale prior; treat as fresh start.
            self._prev_pts = points_xyz
            self._prev_ts = timestamp
            return velocities

        # Build distance matrix in the XY plane (Z is mostly noise on
        # the TI radar at maritime ranges).
        curr_xy = points_xyz[:, :2]
        prev_xy = self._prev_pts[:, :2]
        diffs = curr_xy[:, None, :] - prev_xy[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)

        # Greedy assignment: take the cheapest pair, lock both rows
        # and columns, repeat. Sufficient at <100 points per frame
        # (the radar config caps at 100 anyway).
        gate = self.config.gate_m
        used_prev: set[int] = set()
        order = np.argsort(dists, axis=None)
        for flat_idx in order:
            i, j = np.unravel_index(flat_idx, dists.shape)
            if dists[i, j] > gate:
                break
            if velocities[i] is not None:
                continue
            if j in used_prev:
                continue
            dx = float(curr_xy[i, 0] - prev_xy[j, 0])
            dy = float(curr_xy[i, 1] - prev_xy[j, 1])
            velocities[i] = (dx / dt, dy / dt)
            used_prev.add(int(j))

        self._prev_pts = points_xyz
        self._prev_ts = timestamp
        return velocities


# ---------------------------------------------------------------------------
# Doppler entry point (future-proof; no-op when the CSV lacks V column)
# ---------------------------------------------------------------------------

def radar_velocity_from_doppler(
    points_xyz: np.ndarray,
    doppler_radial_mps: np.ndarray | None,
) -> list[tuple[float, float] | None]:
    """Reconstruct (vx, vy) from per-point radial Doppler.

    The TI TLV's `v` field is the line-of-sight (radial) component of
    the velocity in m/s. To split it back into XY components we use the
    point's bearing: `vx = v * sin(θ)`, `vy = v * cos(θ)` where
    `θ = atan2(x, y)`. The result is missing the tangential component,
    so it's a *partial* velocity: accurate radial magnitude, zero
    tangential. Still strictly better than nothing because for
    obstacle avoidance the radial (closing) component is what TTC
    depends on.

    Returns `None` for any row whose Doppler is `NaN` so callers can
    fall back to the tracker on those.
    """
    if doppler_radial_mps is None or len(doppler_radial_mps) != len(points_xyz):
        return [None] * len(points_xyz)
    out: list[tuple[float, float] | None] = []
    for (x, y, _), v in zip(points_xyz, doppler_radial_mps):
        if not np.isfinite(v):
            out.append(None)
            continue
        r = float(np.hypot(x, y))
        if r == 0.0:
            out.append((0.0, float(v)))
            continue
        sin_t = x / r
        cos_t = y / r
        out.append((float(v) * sin_t, float(v) * cos_t))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def closing_speed(velocity: tuple[float, float] | None) -> float | None:
    """Radar-frame radial closing speed (positive = approaching boat).

    Radar Y is range (forward+); a point with `vy < 0` is getting
    closer. Closing speed is therefore `-vy`. The tangential `vx` is
    discarded, only the radial component drives TTC.
    """
    if velocity is None:
        return None
    return -velocity[1]


def aggregate_bin_velocity(
    points_xyz: np.ndarray,
    velocities: Iterable[tuple[float, float] | None],
    bin_edges: np.ndarray,
) -> list[float | None]:
    """Per-bin closing-speed aggregate (median of magnitudes).

    For each bin, gather the closing speeds of all points whose
    azimuth falls inside the bin. Take the **magnitude-weighted
    median** of the available closing speeds: rank by absolute value
    and pick the middle entry. This is robust to a single bad NN
    match without losing sign information.

    Returns one entry per bin: positive m/s for approaching, negative
    for opening, `None` if no point in that bin had a velocity match.
    """
    n_bins = len(bin_edges) - 1
    buckets: list[list[float]] = [[] for _ in range(n_bins)]
    velocities = list(velocities)
    for (x, y, _), v in zip(points_xyz, velocities):
        if v is None:
            continue
        if y == 0:
            continue
        theta = float(np.degrees(np.arctan2(x, y)))
        if theta < bin_edges[0] or theta >= bin_edges[-1]:
            continue
        idx = int(np.searchsorted(bin_edges, theta, side="right") - 1)
        cs = closing_speed(v)
        # Unreachable: closing_speed() only returns None for a None input, and
        # v is already guarded as non-None above. Kept as defensive depth.
        if cs is None:  # pragma: no cover
            continue
        buckets[idx].append(cs)

    out: list[float | None] = []
    for bucket in buckets:
        if not bucket:
            out.append(None)
            continue
        # Magnitude-weighted median: rank by |closing|, pick the median.
        ranked = sorted(bucket, key=lambda c: abs(c))
        out.append(float(ranked[len(ranked) // 2]))
    return out


def ttc_from_velocity(
    min_range_m: float | None,
    closing_mps: float | None,
    eps: float = 0.05,
    cap_s: float = 60.0,
) -> float | None:
    """Time-to-collision per bin.

    Returns `None` when:
      - no range known (`min_range_m is None`)
      - no velocity known (`closing_mps is None`)
      - closing speed at or below `eps` (opening or near-static)

    The `cap_s` clamp keeps numerics tame and matches the protocol
    contract (callers can treat 60 s as "effectively unbounded").
    """
    if min_range_m is None or closing_mps is None:
        return None
    if closing_mps <= eps:
        return None
    return float(min(min_range_m / closing_mps, cap_s))


# ---------------------------------------------------------------------------
# Bbox bearing-rate (sanity-check input for cross-sensor consistency)
# ---------------------------------------------------------------------------

def bbox_bearing_rate_deg_per_s(
    centroid_history: list[tuple[float, float]],
    pix_deg_ratio: float,
    cx: float,
    timestamps_s: list[float] | None = None,
) -> float | None:
    """Pixel-velocity → bearing-rate from a camera bbox track.

    `centroid_history` is a chronological list of (x, y) pixel
    centroids. `timestamps_s` is the matching wall-clock time per
    entry; if omitted, assume uniform spacing at 3 fps (matches the
    capture target). Returns None if fewer than two samples.

    This is *sanity input* for radar-camera consistency, not primary
    motion. The camera knows bearing well but not range, so it cannot
    contribute to TTC directly without the V.3 IMU + flat-water-plane
    range estimate.
    """
    if len(centroid_history) < 2:
        return None
    if timestamps_s is None:
        timestamps_s = [i / 3.0 for i in range(len(centroid_history))]
    if len(timestamps_s) != len(centroid_history):
        return None
    # Use the first and last samples; a least-squares fit is overkill
    # for a 3-sample track.
    dt = timestamps_s[-1] - timestamps_s[0]
    if dt <= 0:
        return None
    bearing0 = (centroid_history[0][0] - cx) / pix_deg_ratio
    bearing1 = (centroid_history[-1][0] - cx) / pix_deg_ratio
    return float((bearing1 - bearing0) / dt)
