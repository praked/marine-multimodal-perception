"""Per-target motion estimation: 2-D velocity + closing/crossing/diverging.

The pipeline's existing motion picture is radial-only: per-point Doppler
(`motion.py radar_velocity_from_doppler`) reduced to a per-bin closing speed
and TTC. That is blind to the classic COLREG give-way case: a CROSSING target
has near-zero Doppler exactly while it is the boat that must be avoided.
This module builds the missing tangential component from cross-frame
bearing-rate and combines both into a full 2-D velocity vector per tracked
object, in the boat frame.

Geometry (all conventions match the rest of the repo):

  - bearing theta = atan2(X, Y) degrees, right positive, 0 = bow;
  - Doppler V is NEGATIVE on approach (verified 2026-07-06 pacing test), so
    the radial range-rate is rdot = V and closing = -rdot (positive =
    approaching), matching `motion.closing_speed`;
  - target position p = r*(sin t, cos t); differentiating,
        velocity = rdot * (sin t, cos t)  +  r*tdot * (cos t, -sin t)
    i.e. v_radial along the line of sight and v_tangential = r * tdot
    across it (positive = target moving to the boat's right).

Ego motion: two very different corrections.

  - **Rotation** should be subtracted in principle: a turning boat sweeps
    every static target's boat-frame bearing at the yaw rate, which would
    classify the whole world as "crossing". The mechanism exists
    (`use_imu_yaw`: corrected bearing = theta + sign*yaw before the rate
    fit) but ships DEFAULT OFF. Two rounds of 2026-08-24 measurement: the
    RAW RVC yaw is gimbal-aliased on this mount (rock -> p95 0.73 m/s fake
    tangential motion, 6/23 fake crossings); the camera-frame heading now
    composed by `imu_replay.camera_heading_from_rvc` (the Attitude yaw
    under attitude_source: rotation) FIXES that -- rocked-clip p95 0.120
    vs 0.145 yaw-blind, zero fake crossings, sign +1 arbitrated -- but on
    the crossing GT clip recall reads 0.774 vs 0.806 yaw-blind (the
    corrected |v_t| lands nearer the physical ~0.4 m/s, brushing the 0.3
    threshold), and no GT clip contains a genuine sustained turn. So the
    default stays off until a deliberate-turn clip demonstrates the
    benefit (detection.yaml has the numbers; history doc section 7).
  - **Translation** is deliberately NOT subtracted: collision geometry
    (CPA, TTC, give-way) lives in relative velocity. A static buoy dead
    ahead of a moving boat IS closing. No GPS/ground-speed needed.

Danger by geometry: with a full relative-velocity vector the closest point
of approach becomes computable per target:

    t_cpa = -(p . v) / |v|^2      d_cpa = |p + v * t_cpa|

which is the quantity that makes a crossing target dangerous (small d_cpa)
or harmless (large d_cpa) regardless of its Doppler. Exposed per target and,
via `urgency(..., cpa_m=, t_cpa_s=)`, into threat.

Everything is config-gated under `motion.targets` (default OFF: the
pipeline is byte-identical without it). Noise handling follows the house
style: EMA position smoothing (FreeSpaceSmoother pattern) plus a
least-squares bearing-rate fit over a sliding window, because the radar's
15 deg azimuth resolution makes single-frame bearing differences useless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from scripts.sensor_processing.motion import (
    parse_radar_timestamp,
    timestamp_delta_s,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TargetMotionConfig:
    """Knobs for the target tracker (detection.yaml `motion.targets`).

    Defaults mirror the YAML; the rationale for each number lives there
    (single source of documentation, per house style).
    """
    cluster_gate_m: float = 1.0
    assoc_gate_m: float = 1.5
    min_hits: int = 3
    max_age: int = 5
    ema_alpha: float = 0.5
    rate_window_s: float = 2.0
    rate_min_samples: int = 5
    closing_min_mps: float = 0.15
    crossing_min_tangential_mps: float = 0.3
    use_imu_yaw: bool = False
    imu_yaw_sign: float = 1.0
    max_dt_s: float = 0.5
    ttc_cap_s: float = 60.0
    # Camera bearing as a second observation stream for PAIRED targets
    # (motion.targets.camera_bearing_rate, default OFF: byte-identical
    # without it). Requires the identity layer (track_identity.py) to
    # decide which camera track observes which radar target; the pipeline
    # feeds `apply_camera_observations` only when both flags are on.
    camera_bearing_rate: bool = False
    # A paired track whose radar went silent survives on camera
    # observations for at most this long past its last radar hit (the
    # documented mid-transit radar dropout on the 2026-08-19 crossing GT
    # is 4 s; 6 s covers it with margin while bounding how long the held
    # v_radial/range dead-reckoning can drift).
    camera_sustain_max_s: float = 6.0
    # Camera observation freshness for the sustain rule: ~3 camera frames
    # at the captures' effective 3 Hz cadence.
    camera_fresh_s: float = 1.0

    @classmethod
    def from_config(cls, cfg: dict | None,
                    motion_cfg: dict | None = None) -> "TargetMotionConfig":
        cfg = cfg or {}
        return cls(
            cluster_gate_m=float(cfg.get("cluster_gate_m", 1.0)),
            assoc_gate_m=float(cfg.get("assoc_gate_m", 1.5)),
            min_hits=int(cfg.get("min_hits", 3)),
            max_age=int(cfg.get("max_age", 5)),
            ema_alpha=float(cfg.get("ema_alpha", 0.5)),
            rate_window_s=float(cfg.get("rate_window_s", 2.0)),
            rate_min_samples=int(cfg.get("rate_min_samples", 5)),
            closing_min_mps=float(cfg.get("closing_min_mps", 0.15)),
            crossing_min_tangential_mps=float(
                cfg.get("crossing_min_tangential_mps", 0.3)),
            use_imu_yaw=bool(cfg.get("use_imu_yaw", False)),
            imu_yaw_sign=float(cfg.get("imu_yaw_sign", 1.0)),
            max_dt_s=float(cfg.get("max_dt_s", 0.5)),
            ttc_cap_s=float((motion_cfg or {}).get("ttc_cap_s", 60.0)),
            camera_bearing_rate=bool(cfg.get("camera_bearing_rate", False)),
            camera_sustain_max_s=float(cfg.get("camera_sustain_max_s", 6.0)),
            camera_fresh_s=float(cfg.get("camera_fresh_s", 1.0)),
        )


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

@dataclass
class TargetState:
    """One tracked object's motion snapshot (boat frame, rotation-corrected)."""
    track_id: int
    bearing_deg: float            # smoothed, boat frame (right +, 0 = bow)
    range_m: float                # smoothed slant range in the water plane
    x_m: float                    # smoothed lateral position (right +)
    y_m: float                    # smoothed forward position
    n_points: int                 # radar returns in this frame's cluster
    age_frames: int               # frames since the track was born
    # Velocity decomposition. None = that component not yet measurable
    # (no Doppler-carrying point / bearing-rate window not filled).
    v_radial_mps: float | None = None      # range-rate (negative = approaching)
    v_tangential_mps: float | None = None  # r * bearing-rate (positive = right)
    vx_mps: float | None = None            # boat-frame lateral velocity
    vy_mps: float | None = None            # boat-frame forward velocity
    speed_mps: float | None = None
    course_deg: float | None = None        # direction of motion, atan2(vx, vy)
    closing_mps: float | None = None       # -v_radial (motion.closing_speed sign)
    # Closest-point-of-approach prediction (relative motion, so ego
    # translation is correctly included). None when speed is below the
    # noise floor or the CPA lies in the past (already diverging).
    cpa_m: float | None = None
    t_cpa_s: float | None = None
    # closing | crossing | diverging | static | unknown
    motion_state: str = "unknown"
    # True when the bearing-rate was measured in the yaw-stabilised frame
    # (IMU yaw available for every sample in the fit window).
    ego_corrected: bool = False
    # Indices (into this frame's filtered point cloud) of the cluster
    # matched to this track: the physical evidence the identity layer
    # intersects with DetectionMatch.point_indices. Empty on
    # camera-sustained frames (no radar cluster this frame).
    point_indices: list[int] = field(default_factory=list)
    # Cross-sensor identity (track_identity.py): the camera track this
    # radar target is paired with. None when the identity layer is off or
    # no pairing is confirmed. Additive protocol field (v1.3).
    camera_track_id: int | None = None
    # Which observation stream produced v_tangential: "radar" (fresh LS
    # fit over radar bearings), "camera" (paired camera track's bearing
    # window, used when the radar window is stale/dropped:
    # camera_bearing_rate), "radar_stale" (dropout, camera window not yet
    # fit-able: the last radar slope). None = no rate available.
    rate_source: str | None = None


@dataclass
class TargetMotionResult:
    targets: list[TargetState] = field(default_factory=list)
    n_clusters: int = 0
    ego_yaw_deg: float | None = None


# ---------------------------------------------------------------------------
# Clustering (single-linkage, greedy union-find; N <= 500 per config cap)
# ---------------------------------------------------------------------------

def cluster_points(points_xyz: np.ndarray, gate_m: float) -> list[np.ndarray]:
    """Group a frame's cloud into targets: single-linkage in the XY plane.

    Returns a list of index arrays into `points_xyz`. Z is ignored (mostly
    noise on the TI radar at maritime ranges: same reasoning as
    RadarPointTracker).
    """
    n = len(points_xyz)
    if n == 0:
        return []
    xy = np.asarray(points_xyz, dtype=np.float64)[:, :2]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    diffs = xy[:, None, :] - xy[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    ii, jj = np.where(np.triu(dists <= gate_m, k=1))
    for i, j in zip(ii, jj):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(g, dtype=int) for g in groups.values()]


def _cluster_observation(
    points_xyz: np.ndarray,
    velocities_xy: list[tuple[float, float] | None],
    idxs: np.ndarray,
) -> tuple[float, float, int, float | None]:
    """(centroid_x, centroid_y, n_points, median radial speed | None).

    The radial speed per point is the projection of its velocity onto the
    line of sight: (x*vx + y*vy)/r. For Doppler-sourced velocities this
    recovers the TLV's V exactly (radar_velocity_from_doppler stores
    V*(sin t, cos t)); for tracker-sourced ones it is the NN estimate's
    radial component: the same per-point fallback semantics the per-bin
    velocity already uses.
    """
    pts = points_xyz[idxs]
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    radials: list[float] = []
    for i in idxs:
        v = velocities_xy[i] if i < len(velocities_xy) else None
        if v is None:
            continue
        x, y = float(points_xyz[i, 0]), float(points_xyz[i, 1])
        r = math.hypot(x, y)
        if r <= 0.0:
            continue
        radials.append((x * v[0] + y * v[1]) / r)
    radial = float(np.median(radials)) if radials else None
    return cx, cy, len(idxs), radial


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

@dataclass
class _Track:
    track_id: int
    x: float                      # EMA position
    y: float
    hits: int = 1
    age_since_hit: int = 0
    n_points: int = 0
    v_radial_ema: float | None = None
    # (t_seconds, corrected_bearing_deg) samples for the rate fit. The
    # bearing stored here is already yaw-stabilised (theta + sign*yaw)
    # when ego correction is active, and unwrapped for continuity.
    bearing_samples: list[tuple[float, float]] = field(default_factory=list)
    ego_corrected: bool = False   # whether samples are yaw-stabilised
    # Cluster point indices matched this frame (empty when unmatched):
    # the identity layer's pairing evidence.
    last_point_indices: list[int] = field(default_factory=list)
    # Camera bearing observations from the paired camera track
    # (camera_bearing_rate): a SEPARATE window from bearing_samples so a
    # residual radar<->camera bearing offset can never bias the radar
    # slope; the camera fit only substitutes when the radar window is
    # stale (see _bearing_rate_dps).
    camera_samples: list[tuple[float, float]] = field(default_factory=list)
    camera_ego_corrected: bool = False
    last_camera_t: float | None = None
    last_radar_t: float | None = None
    last_update_t: float | None = None   # position update (radar or camera)

    @property
    def range_m(self) -> float:
        return math.hypot(self.x, self.y)

    @property
    def bearing_deg(self) -> float:
        return math.degrees(math.atan2(self.x, self.y))


class TargetMotionTracker:
    """Cluster -> associate -> smooth -> differentiate, per radar frame.

    Stateful per clip (like ObstacleDetectionPipeline / RadarPointTracker).
    Call `update(m_res, timestamp, yaw_deg=...)` once per frame with the
    pipeline's filtered MMWaveResult; `yaw_deg` is the IMU yaw for the same
    frame (None = no attitude: rates fall back to the raw boat frame).
    """

    def __init__(self, config: TargetMotionConfig | None = None):
        self.config = config or TargetMotionConfig()
        self.tracks: list[_Track] = []
        self._next_id = count(0)
        self._prev_ts: str | None = None
        self._last_n_clusters: int = 0
        self._last_yaw: float | None = None

    def reset(self) -> None:
        self.tracks = []
        self._prev_ts = None
        self._last_n_clusters = 0
        self._last_yaw = None

    # -- internals ---------------------------------------------------------

    def _corrected_bearing(self, x: float, y: float,
                           yaw_deg: float | None) -> tuple[float, bool]:
        theta = math.degrees(math.atan2(x, y))
        if self.config.use_imu_yaw and yaw_deg is not None:
            return theta + self.config.imu_yaw_sign * float(yaw_deg), True
        return theta, False

    def _push_sample(self, samples: list[tuple[float, float]], t_s: float,
                     bearing: float) -> list[tuple[float, float]]:
        """Append an (unwrapped) bearing sample and trim to the fit window."""
        if samples:
            # Unwrap against the previous sample for slope continuity.
            prev = samples[-1][1]
            while bearing - prev > 180.0:
                bearing -= 360.0
            while bearing - prev < -180.0:
                bearing += 360.0
        samples = samples + [(t_s, bearing)]
        # Trim to the fit window (keep one extra sample so the span check
        # sees the full window edge).
        cutoff = t_s - self.config.rate_window_s
        return [(t, b) for t, b in samples if t >= cutoff]

    def _push_bearing(self, tr: _Track, t_s: float, bearing: float,
                      corrected: bool) -> None:
        # Mixing yaw-stabilised and raw samples in one fit corrupts the
        # slope; on an availability flip, restart the window.
        if tr.bearing_samples and tr.ego_corrected != corrected:
            tr.bearing_samples = []
        tr.ego_corrected = corrected
        tr.bearing_samples = self._push_sample(tr.bearing_samples, t_s,
                                               bearing)

    def _push_camera_bearing(self, tr: _Track, t_s: float, bearing: float,
                             corrected: bool) -> None:
        """Camera-window twin of _push_bearing (same flip/unwrap/trim rules)."""
        if tr.camera_samples and tr.camera_ego_corrected != corrected:
            tr.camera_samples = []
        tr.camera_ego_corrected = corrected
        tr.camera_samples = self._push_sample(tr.camera_samples, t_s, bearing)
        tr.last_camera_t = t_s

    def _fit_rate(self, samples: list[tuple[float, float]]) -> float | None:
        """Least-squares slope over one sample window, deg/s.

        A two-point difference at the radar's quantised azimuth is noise;
        the LS fit over `rate_window_s` averages it down (~sqrt(N^3)
        improvement for uniform sampling). Requires `rate_min_samples`
        samples spanning at least half the window.
        """
        if len(samples) < self.config.rate_min_samples:
            return None
        t = np.array([p[0] for p in samples])
        b = np.array([p[1] for p in samples])
        span = float(t[-1] - t[0])
        if span < 0.5 * self.config.rate_window_s:
            return None
        t = t - t.mean()
        denom = float(np.dot(t, t))
        if denom <= 0.0:
            return None
        return float(np.dot(t, b - b.mean()) / denom)

    def _bearing_rate_dps(self, tr: _Track, t_s: float | None
                          ) -> tuple[float | None, str | None]:
        """(bearing-rate deg/s, source) combining radar + camera windows.

        Without camera_bearing_rate this is exactly the radar LS fit (the
        pre-identity behaviour). With it, the radar fit is trusted only
        while its newest sample is fresh (within half the fit window);
        during a radar dropout the paired camera track's bearing window
        takes over, and the stale radar slope is the last resort (early
        in a dropout, before the camera window can satisfy the fit
        gates: that slope was measured <=1 s ago).
        """
        radar_rate = self._fit_rate(tr.bearing_samples)
        if not self.config.camera_bearing_rate:
            return radar_rate, ("radar" if radar_rate is not None else None)
        radar_fresh = bool(
            tr.bearing_samples and t_s is not None
            and (t_s - tr.bearing_samples[-1][0])
            <= 0.5 * self.config.rate_window_s)
        if radar_rate is not None and radar_fresh:
            return radar_rate, "radar"
        camera_rate = self._fit_rate(tr.camera_samples)
        if camera_rate is not None:
            return camera_rate, "camera"
        if radar_rate is not None:
            return radar_rate, "radar_stale"
        return None, None

    def _snapshot(self, tr: _Track, t_s: float | None = None) -> TargetState:
        cfg = self.config
        r = tr.range_m
        theta = tr.bearing_deg
        v_r = tr.v_radial_ema
        rate_dps, rate_source = self._bearing_rate_dps(tr, t_s)
        v_t = (r * math.radians(rate_dps)) if rate_dps is not None else None

        vx = vy = speed = course = cpa = t_cpa = None
        if v_r is not None or v_t is not None:
            vr0 = v_r if v_r is not None else 0.0
            vt0 = v_t if v_t is not None else 0.0
            st, ct = math.sin(math.radians(theta)), math.cos(math.radians(theta))
            vx = vr0 * st + vt0 * ct
            vy = vr0 * ct - vt0 * st
            speed = math.hypot(vx, vy)
            course = math.degrees(math.atan2(vx, vy)) if speed > 0.0 else None
            # CPA only when the velocity is above the radial noise floor:
            # dividing by a noise-magnitude |v|^2 yields arbitrary t_cpa.
            if speed >= cfg.closing_min_mps:
                pv = tr.x * vx + tr.y * vy
                t_hat = -pv / (speed * speed)
                if t_hat > 0.0:
                    t_cpa = min(t_hat, cfg.ttc_cap_s)
                    cpa = math.hypot(tr.x + vx * t_hat, tr.y + vy * t_hat)

        closing = -v_r if v_r is not None else None
        if v_r is None and v_t is None:
            state = "unknown"
        elif v_t is not None and abs(v_t) >= cfg.crossing_min_tangential_mps:
            state = "crossing"
        elif closing is not None and closing >= cfg.closing_min_mps:
            state = "closing"
        elif closing is not None and closing <= -cfg.closing_min_mps:
            state = "diverging"
        else:
            state = "static"

        return TargetState(
            track_id=tr.track_id,
            bearing_deg=theta,
            range_m=r,
            x_m=tr.x,
            y_m=tr.y,
            n_points=tr.n_points,
            age_frames=tr.hits,
            v_radial_mps=v_r,
            v_tangential_mps=v_t,
            vx_mps=vx,
            vy_mps=vy,
            speed_mps=speed,
            course_deg=course,
            closing_mps=closing,
            cpa_m=cpa,
            t_cpa_s=t_cpa,
            motion_state=state,
            ego_corrected=tr.ego_corrected,
            point_indices=list(tr.last_point_indices),
            rate_source=rate_source,
        )

    # -- public ------------------------------------------------------------

    def update(
        self,
        m_res,
        timestamp: str | None,
        yaw_deg: float | None = None,
    ) -> TargetMotionResult:
        """Step the tracker with one frame's filtered MMWaveResult.

        Returns confirmed (>= min_hits), this-frame-matched targets only:
        coasting tracks survive internally (max_age) but do not report a
        stale snapshot.
        """
        cfg = self.config
        pts = np.asarray(m_res.points_xyz, dtype=np.float64)
        t_s = parse_radar_timestamp(timestamp)

        # Stream-gap policy mirrors RadarPointTracker: an unusable delta
        # means differentiating across it is meaningless: start fresh.
        if self._prev_ts is not None:
            dt = timestamp_delta_s(timestamp, self._prev_ts)
            if dt is None or dt > cfg.max_dt_s:
                self.tracks = []
        self._prev_ts = timestamp

        clusters = cluster_points(pts, cfg.cluster_gate_m) if len(pts) else []
        obs = [_cluster_observation(pts, m_res.velocities_xy or [], idxs)
               for idxs in clusters]

        # Greedy NN association, cheapest pair first (house pattern).
        matched_tracks: set[int] = set()
        matched_obs: set[int] = set()
        if obs and self.tracks:
            cost = np.full((len(self.tracks), len(obs)), np.inf)
            for i, tr in enumerate(self.tracks):
                for j, (ox, oy, _n, _vr) in enumerate(obs):
                    cost[i, j] = math.hypot(tr.x - ox, tr.y - oy)
            order = np.argsort(cost, axis=None)
            for flat in order:
                i, j = np.unravel_index(flat, cost.shape)
                if cost[i, j] > cfg.assoc_gate_m:
                    break
                if i in matched_tracks or j in matched_obs:
                    continue
                matched_tracks.add(int(i))
                matched_obs.add(int(j))
                tr = self.tracks[i]
                ox, oy, n_pts, v_rad = obs[j]
                a = cfg.ema_alpha
                tr.x = a * ox + (1.0 - a) * tr.x
                tr.y = a * oy + (1.0 - a) * tr.y
                tr.n_points = n_pts
                tr.hits += 1
                tr.age_since_hit = 0
                tr.last_point_indices = clusters[j].tolist()
                tr.last_radar_t = t_s
                tr.last_update_t = t_s
                if v_rad is not None:
                    tr.v_radial_ema = (v_rad if tr.v_radial_ema is None
                                       else a * v_rad + (1.0 - a) * tr.v_radial_ema)
                if t_s is not None:
                    bearing, corr = self._corrected_bearing(tr.x, tr.y, yaw_deg)
                    self._push_bearing(tr, t_s, bearing, corr)

        # Age unmatched tracks; spawn from unmatched observations.
        for i, tr in enumerate(self.tracks):
            if i not in matched_tracks:
                tr.age_since_hit += 1
                tr.last_point_indices = []
        for j, (ox, oy, n_pts, v_rad) in enumerate(obs):
            if j in matched_obs:
                continue
            tr = _Track(track_id=next(self._next_id), x=ox, y=oy,
                        n_points=n_pts, v_radial_ema=v_rad,
                        last_point_indices=clusters[j].tolist(),
                        last_radar_t=t_s, last_update_t=t_s)
            if t_s is not None:
                bearing, corr = self._corrected_bearing(ox, oy, yaw_deg)
                self._push_bearing(tr, t_s, bearing, corr)
            self.tracks.append(tr)
        self.tracks = [tr for tr in self.tracks
                       if tr.age_since_hit < cfg.max_age
                       or self._camera_sustained(tr, t_s)]

        self._last_n_clusters = len(clusters)
        self._last_yaw = float(yaw_deg) if yaw_deg is not None else None
        targets = [self._snapshot(tr, t_s) for tr in self.tracks
                   if tr.hits >= cfg.min_hits and tr.age_since_hit == 0]
        targets.sort(key=lambda t: t.range_m)
        return TargetMotionResult(
            targets=targets,
            n_clusters=len(clusters),
            ego_yaw_deg=self._last_yaw,
        )

    def _camera_sustained(self, tr: _Track, t_s: float | None) -> bool:
        """Whether a radar-aged track survives on camera observations.

        Only with camera_bearing_rate: the paired camera track has seen
        the object recently (camera_fresh_s) AND the radar silence has not
        exceeded camera_sustain_max_s (bounding the held v_radial/range
        dead-reckoning). Off -> always False: pruning is byte-identical.
        """
        cfg = self.config
        if not cfg.camera_bearing_rate or t_s is None:
            return False
        return (tr.last_camera_t is not None
                and (t_s - tr.last_camera_t) <= cfg.camera_fresh_s
                and tr.last_radar_t is not None
                and (t_s - tr.last_radar_t) <= cfg.camera_sustain_max_s)

    def apply_camera_observations(
        self,
        obs: dict[int, float],
        timestamp: str | None,
    ) -> TargetMotionResult:
        """Second phase of a frame step: paired camera bearings.

        `obs` maps radar track_id -> offset-corrected camera bearing
        (boat frame, deg) for this frame, from the identity layer
        (TrackIdentityManager.update -> camera_obs). Call AFTER `update`
        with the same frame's timestamp, and only when
        `camera_bearing_rate` is on. Two effects:

          - every observed paired track gains a camera bearing sample in
            its separate camera window (warm before a dropout starts);
          - a track the radar MISSED this frame is repositioned from the
            camera bearing (range dead-reckoned by the held v_radial) and
            reported in the returned result: the camera fills the radar's
            documented mid-transit dropouts. Such snapshots carry
            n_points == 0 and empty point_indices (honest provenance).

        Returns the frame's refreshed TargetMotionResult (supersedes the
        one `update` returned).
        """
        cfg = self.config
        t_s = parse_radar_timestamp(timestamp)
        sustained_ids: set[int] = set()
        if t_s is not None:
            by_id = {tr.track_id: tr for tr in self.tracks}
            for rid, cam_bearing in obs.items():
                tr = by_id.get(rid)
                if tr is None:
                    continue
                # The rate window uses the same yaw stabilisation as the
                # radar window (use_imu_yaw): position stays boat-frame.
                stab, corr = cam_bearing, False
                if cfg.use_imu_yaw and self._last_yaw is not None:
                    stab = cam_bearing + cfg.imu_yaw_sign * self._last_yaw
                    corr = True
                self._push_camera_bearing(tr, t_s, stab, corr)
                if tr.age_since_hit > 0 and self._camera_sustained(tr, t_s):
                    # Radar missed this frame: camera sustains the track.
                    a = cfg.ema_alpha
                    b = a * cam_bearing + (1.0 - a) * tr.bearing_deg
                    dt = (t_s - tr.last_update_t
                          if tr.last_update_t is not None else 0.0)
                    r = max(0.1, tr.range_m + (tr.v_radial_ema or 0.0) * dt)
                    tr.x = r * math.sin(math.radians(b))
                    tr.y = r * math.cos(math.radians(b))
                    tr.n_points = 0
                    tr.last_point_indices = []
                    tr.last_update_t = t_s
                    sustained_ids.add(rid)
        targets = [self._snapshot(tr, t_s) for tr in self.tracks
                   if tr.hits >= cfg.min_hits
                   and (tr.age_since_hit == 0
                        or tr.track_id in sustained_ids)]
        targets.sort(key=lambda t: t.range_m)
        return TargetMotionResult(
            targets=targets,
            n_clusters=self._last_n_clusters,
            ego_yaw_deg=self._last_yaw,
        )


# ---------------------------------------------------------------------------
# Bin mapping (fusion/scorer consumers)
# ---------------------------------------------------------------------------

def targets_by_bin(targets: list[TargetState],
                   bin_edges: np.ndarray) -> list[TargetState | None]:
    """Nearest-range target per fusion bin (None = no target in the bin).

    Nearest-range matches the min_range semantics every other per-bin
    aggregate uses; a consumer that wants all targets has the flat list.
    """
    n_bins = len(bin_edges) - 1
    out: list[TargetState | None] = [None] * n_bins
    for t in targets:
        if t.bearing_deg < bin_edges[0] or t.bearing_deg >= bin_edges[-1]:
            continue
        idx = int(np.searchsorted(bin_edges, t.bearing_deg, side="right") - 1)
        if out[idx] is None or t.range_m < out[idx].range_m:
            out[idx] = t
    return out


def target_to_dict(t: TargetState, ndigits: int = 3,
                   include_identity: bool = False) -> dict:
    """JSON-safe dict for the sectors JSONL `targets` field (protocol v1.2).

    `include_identity` (protocol v1.3, additive) appends `camera_track_id`
    -- pass it only when the identity layer is enabled so pre-identity
    streams stay byte-identical.
    """
    def _r(v):
        return round(float(v), ndigits) if v is not None else None
    out = {
        "id": t.track_id,
        "bearing_deg": _r(t.bearing_deg),
        "range_m": _r(t.range_m),
        "n_points": t.n_points,
        "age_frames": t.age_frames,
        "v_radial_mps": _r(t.v_radial_mps),
        "v_tangential_mps": _r(t.v_tangential_mps),
        "vx_mps": _r(t.vx_mps),
        "vy_mps": _r(t.vy_mps),
        "speed_mps": _r(t.speed_mps),
        "course_deg": _r(t.course_deg),
        "closing_mps": _r(t.closing_mps),
        "cpa_m": _r(t.cpa_m),
        "t_cpa_s": _r(t.t_cpa_s),
        "motion_state": t.motion_state,
        "ego_corrected": t.ego_corrected,
    }
    if include_identity:
        out["camera_track_id"] = t.camera_track_id
    return out
