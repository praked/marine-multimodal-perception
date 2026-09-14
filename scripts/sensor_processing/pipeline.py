"""Canonical per-frame obstacle-detection pipeline.

Encapsulates the per-sensor processing that was originally inlined in the
standalone fisheye/thermal demo scripts (since removed; see the git history
for them). Single source of truth so viewer.py, sweep.py, metrics.py, and the
production fusion.py all see the same algorithm.

Algorithms are byte-equivalent to those originals, with their bugs fixed and
parameters externalised to YAML (see configs/{intrinsics,detection}.yaml).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from scripts.sensor_processing.motion import (
    RadarPointTracker,
    RadarPointTrackerConfig,
    aggregate_bin_velocity,
    radar_velocity_from_doppler,
    ttc_from_velocity,
)
from scripts.sensor_processing.target_motion import (
    TargetMotionConfig,
    TargetMotionResult,
    TargetMotionTracker,
)
from scripts.sensor_processing.track_identity import (
    IdentityConfig,
    TrackIdentityManager,
)
from scripts.utils.association import (
    AssociationParams,
    associate_radar_to_detections,
    match_polygons_to_detections,
    pinhole_bearings,
    polygons_from_instances,
)
from scripts.utils.detections import InstanceSegProvider
from scripts.utils.cv_common import (
    build_blob_detector,
    detect_horizon,
    extract_kp,
    pinhole_new_K,
    undistort_fisheye,
    repair_dead_rows,
    undistort_thermal,
)
from scripts.utils.geometry import (
    UP_LEVEL,
    load_extrinsics,
    project_radar_to_undistorted,
    range_from_contact_point,
    range_from_water_plane_up,
    up_from_horizon_line,
)
from scripts.utils.segmentation import (
    OBSTACLE as SEG_OBSTACLE,
    ContactParams,
    SegDetectParams,
    SegProvider,
    detection_obstacle_fraction,
    horizon_from_water,
    seg_obstacle_detections,
    water_edge_contact,
)


# ---------------------------------------------------------------------------
# Per-frame result containers
# ---------------------------------------------------------------------------

@dataclass
class FisheyeResult:
    undistorted: np.ndarray
    horizon_mask: np.ndarray
    horizon_line: tuple[float, float, float]   # slope, intercept, confidence
    obstacle_mask: np.ndarray
    coords: list[tuple[int, int]]
    sizes: list[float]
    angles: list[float]
    # Monocular water-plane range estimate per detection, aligned with
    # `coords`/`angles`. None where the blob back-projects above the horizon.
    # Vision-estimated (flat-water assumption); biased until camera_height_m
    # is measured (V.1) and IMU pitch/roll feeds the ray. See geometry.py.
    ranges: list[float | None] = field(default_factory=list)
    # Water-segmentation mask (HxW class ids 0/1/2) used for this frame's range,
    # in the undistorted-image space. None when segmentation is off/absent.
    # Kept for overlays/eval (viewer, seg_range_ablation).
    seg_mask: np.ndarray | None = None
    # Pixel-level radar association (fusion.association, default off; empty
    # lists otherwise). Aligned with `coords`. `radar_ranges[i]` is the
    # aggregated radar forward range of the points that project inside
    # detection i's padded bbox (None = unassociated); `radar_hits[i]` the
    # match count. When association.use_radar_range is set, `ranges` holds
    # the radar value for associated detections and `mono_ranges` preserves
    # the original monocular estimate for telemetry/A-B.
    radar_ranges: list[float | None] = field(default_factory=list)
    radar_hits: list[int] = field(default_factory=list)
    # Median bearing (deg) of the matched radar returns, aligned with
    # `coords`; None = unassociated. Confirmation anchors on THIS bearing
    # (fusion.association.confirm_anchor: radar) so the confirmed flag sits
    # where the radar evidence physically is, and the same-bin agreement
    # check (confirm_same_bin) compares it against the detection's bearing.
    radar_bearings: list[float | None] = field(default_factory=list)
    # Per-detection matched returns as Nx2 (bearing_deg, range_m) arrays
    # (None = no matches): _confirm_bins marks the bins these occupy.
    radar_match_points: list = field(default_factory=list)
    # Which pixel gate produced each detection's association: "box" or
    # "instance" (fusion.association.use_instance_masks). Aligned with
    # `coords`; empty when association is off. Telemetry/eval only.
    radar_gates: list[str] = field(default_factory=list)
    # Indices (into MMWaveResult.points_xyz) of each detection's matched
    # returns, aligned with `coords`; empty lists when unmatched or
    # association off. The identity layer's pairing evidence (intersected
    # against TargetState.point_indices).
    radar_point_indices: list[list[int]] = field(default_factory=list)
    # Cross-sensor identity (motion.targets.identity, default off ->
    # empty). Aligned with `coords`: the BBoxTracker camera track each
    # detection belongs to, and the paired radar target id (None =
    # unpaired). See scripts/sensor_processing/track_identity.py.
    camera_track_ids: list[int | None] = field(default_factory=list)
    radar_target_ids: list[int | None] = field(default_factory=list)
    mono_ranges: list[float | None] = field(default_factory=list)
    # Untruncated monocular estimates (no max_range_m cap; None = ray above
    # horizon). Back the vote gate and the association range-compatibility
    # check ("radar says 2 m, camera says far" = a foreground object's
    # return inside this box, not this object).
    raw_ranges: list[float | None] = field(default_factory=list)
    # Per-detection fusion-vote eligibility (fusion.vote_range_gate): False
    # when the raw water-plane estimate reads far beyond the nav envelope
    # (e.g. the shoreline hundreds of metres out): the detection is still
    # reported, it just doesn't raise bin threat scores. All-True when the
    # gate is off or ranges are unknown (conservative).
    votes: list[bool] = field(default_factory=list)
    # Frame luminance stats (grayscale of the undistorted frame, 0-255):
    # the fisheye is the only reliable darkness proxy on the boat (the
    # Lepton's AGC stretches night noise into healthy-looking contrast).
    # `is_dark` mirrors the SegWorker darkness gate's mean-luminance test
    # (fisheye.darkness_thresh, default 25: pi_seg_worker's threshold).
    # Context features for the fusion scorer; stats only, never gates
    # detection here.
    luminance_mean: float | None = None
    luminance_p05: float | None = None
    luminance_p95: float | None = None
    is_dark: bool = False
    # Which attitude source supplied the range up-vector for this frame
    # ("imu" / "water_edge" / "horizon" / "level"; None = range disabled).
    # See _range_up_vector: recorded so range provenance survives into
    # telemetry/feature tables instead of being reconstructed downstream.
    range_reference: str | None = None


@dataclass
class ThermalResult:
    undistorted: np.ndarray
    horizon_mask: np.ndarray
    horizon_line: tuple[float, float, float]
    obstacle_mask: np.ndarray
    coords: list[tuple[int, int]]
    sizes: list[float]
    angles: list[float]
    new_average: float
    # See FisheyeResult.ranges.
    ranges: list[float | None] = field(default_factory=list)
    # See FisheyeResult.radar_ranges / radar_hits / radar_bearings /
    # mono_ranges / raw_ranges / votes.
    radar_ranges: list[float | None] = field(default_factory=list)
    radar_hits: list[int] = field(default_factory=list)
    radar_bearings: list[float | None] = field(default_factory=list)
    radar_match_points: list = field(default_factory=list)
    radar_gates: list[str] = field(default_factory=list)
    radar_point_indices: list[list[int]] = field(default_factory=list)
    mono_ranges: list[float | None] = field(default_factory=list)
    raw_ranges: list[float | None] = field(default_factory=list)
    votes: list[bool] = field(default_factory=list)
    # Wet-cover quality guard (thermal.quality_guard). quality_ok False
    # means the frame's contrast collapsed (water film / fog on the
    # cover) and detections were suppressed: the thermal contributes
    # nothing to fusion for this frame. std/dyn are always reported so
    # dashboards can plot cover health even with the guard off.
    quality_ok: bool = True
    quality_std: float | None = None
    quality_dyn_range: float | None = None
    # See FisheyeResult.range_reference.
    range_reference: str | None = None
    # Active thermal parameter profile (thermal.profiles): "day" /
    # "twilight" / "night" when the sun-elevation scheduler is enabled,
    # None when it is off (the base thermal block is in force).
    profile: str | None = None


@dataclass
class MMWaveResult:
    points_xyz: np.ndarray         # Nx3 after y filter
    angles: list[float]            # azimuth deg
    ranges: list[float]            # Y in metres
    # Per-point (vx, vy) m/s from cross-frame association or Doppler.
    # Aligned with `points_xyz`. None entries = unmatched / no prior.
    velocities_xy: list[tuple[float, float] | None] = field(default_factory=list)
    # Provenance for telemetry. "doppler" = every velocity came from the
    # TLV's radial Doppler (CSV `V` column, captured since 2026-07-06);
    # "mixed" = Doppler where available, cross-frame tracker filling the
    # NaN gaps; "tracker" = no Doppler column (pre-2026-07-06 captures).
    velocity_source: str = "tracker"
    # Per-point detection SNR / noise floor in dB (TLV-7 side info, CSV
    # SNR/NOISE columns captured since 2026-07-09). Aligned with
    # `points_xyz`; None entries = empty cell in the CSV; empty lists =
    # the capture predates side info (or the frame was rejected).
    snr_db: list[float | None] = field(default_factory=list)
    noise_db: list[float | None] = field(default_factory=list)


@dataclass
class FusionResult:
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    scores: list[float]
    min_ranges: list[float | None]
    sensor_hit_mask: np.ndarray    # n_bins x 3 (fisheye, thermal, mmwave)
    # Per-bin radial closing speed (m/s, positive = approaching).
    # None when no radar point in that bin had a velocity match.
    per_bin_velocity_mps: list[float | None] = field(default_factory=list)
    # Per-bin time-to-collision (s, capped at the motion config cap).
    # None when closing speed is non-positive or range unknown.
    per_bin_ttc_s: list[float | None] = field(default_factory=list)
    # Per-bin object-level cross-sensor confirmation (fusion.association):
    # True when the bin contains a camera detection whose padded bbox
    # captured >= min_points projected radar returns: much stronger
    # evidence than two sensors coincidentally voting in the same 10° bin.
    # All-False when association is off (shape-stable).
    confirmed: list[bool] = field(default_factory=list)


@dataclass
class FrameResult:
    fisheye: FisheyeResult | None = None
    thermal: ThermalResult | None = None
    mmwave: MMWaveResult | None = None
    fusion: FusionResult | None = None
    timestamp: str | None = None
    # Per-target motion states (motion.targets, default off -> None).
    # See scripts/sensor_processing/target_motion.py.
    targets: TargetMotionResult | None = None


# ---------------------------------------------------------------------------
# Detection primitives (stateless)
# ---------------------------------------------------------------------------

def gradient_obstacle_detection(
    frame: np.ndarray,
    gaussian_window: int = 5,
    gradient_threshold: int = 30,
    median_blur: int = 3,
    glint_low: int = 240,
    glint_high: int = 255,
    glint_inpaint_radius: int = 3,
) -> np.ndarray:
    """Glint removal -> Sobel-x -> threshold. Returns a uint8 mask."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    glint_mask = cv2.inRange(v, glint_low, glint_high)
    v_inpaint = cv2.inpaint(v, glint_mask, inpaintRadius=glint_inpaint_radius, flags=cv2.INPAINT_TELEA)
    glint_reduced = cv2.cvtColor(cv2.merge([h, s, v_inpaint]), cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(glint_reduced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (gaussian_window, gaussian_window), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_x_abs = cv2.convertScaleAbs(grad_x)
    _, mask = cv2.threshold(grad_x_abs, gradient_threshold, 255, cv2.THRESH_BINARY)
    return cv2.medianBlur(mask, median_blur)


def thermal_obstacle_detection(
    frame: np.ndarray,
    running_average: float,
    mask: np.ndarray,
    object_thresh: int = 100,
    contrast_guard: int = 50,
) -> tuple[np.ndarray, float]:
    """Background-subtract + contrast-stretch + threshold."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    new_average = float(cv2.mean(gray)[0])

    avg_u8 = int(round(running_average)) & 0xFF
    sub = np.clip(gray.astype(np.int16) - avg_u8, 0, 255).astype(np.uint8)
    max_val = int(np.max(sub)) if sub.size else 0
    if max_val > contrast_guard:
        normalised = (sub.astype(np.float32) / max_val * 255).astype(np.uint8)
    else:
        normalised = sub

    normalised = cv2.bitwise_and(normalised, normalised, mask=mask)
    _, obstacle_mask = cv2.threshold(normalised, object_thresh, 255, cv2.THRESH_BINARY)
    return obstacle_mask, new_average


def thermal_preprocess(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """Config-gated denoise/contrast preprocessing (thermal.preprocess).

    Applied to the full (unmasked) grayscale so CLAHE tiles aren't distorted
    by the zeroed sky region; the caller masks afterwards. Order: median
    denoise first, then CLAHE: enhancing before denoising amplifies the
    160x120 sensor's fixed-pattern noise. Off (or empty cfg) returns the
    input unchanged, keeping the default path byte-identical.
    """
    k = int(cfg.get("median_ksize", 0) or 0)
    if k >= 3:
        gray = cv2.medianBlur(gray, k | 1)
    cl = cfg.get("clahe", {}) or {}
    if cl.get("enabled", False):
        tile = int(cl.get("tile", 8))
        clahe = cv2.createCLAHE(
            clipLimit=float(cl.get("clip_limit", 2.0)),
            tileGridSize=(tile, tile),
        )
        gray = clahe.apply(gray)
    return gray


def thermal_quality_stats(frame: np.ndarray) -> tuple[float, float]:
    """(std-dev, p99 - p1 dynamic range) of a thermal frame's grayscale.

    The wet-cover signature (docs/thermal_water_problem.pdf §2.3): a water
    film on the cover is near-opaque in LWIR, so the scene collapses to a
    uniform field: std-dev and dynamic range both crater. Computed on the
    full frame (pre-horizon-mask; the mask zeroes the sky and would fake
    low contrast). Same statistic smoke_capture.py checks in the field.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    g = gray.astype(np.float32)
    std = float(g.std())
    dyn = float(np.percentile(g, 99) - np.percentile(g, 1))
    return std, dyn


THERMAL_PROFILE_NAMES = ("day", "twilight", "night")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge: override wins, nested dicts merge, inputs are
    left untouched. Builds a thermal profile's effective parameter set."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_thermal_profile(sun_elevation_deg: float | None,
                            day_min_elev_deg: float = 10.0,
                            night_max_elev_deg: float = -6.0) -> str:
    """Thermal profile name for a solar elevation (deg above horizon).

    Band convention matches the enrichment dayparts (dashboard/tools/
    enrich_bundle.py) and the evaluate.py strata (civil twilight):
    day = elev >= day_min (the enrichment's golden band 0..10 deg joins
    twilight here), night = elev < night_max (below civil twilight, -6),
    twilight between. Absent/NaN elevation resolves to "day": the daytime
    defaults are the measured Pareto point (2026-07-09 stage-4 sweep) and
    the only profile validated on open water, so ignorance must not select
    an opt-in candidate.
    """
    if sun_elevation_deg is None or not np.isfinite(sun_elevation_deg):
        return "day"
    e = float(sun_elevation_deg)
    if e >= float(day_min_elev_deg):
        return "day"
    if e < float(night_max_elev_deg):
        return "night"
    return "twilight"


def mmwave_angles_ranges(points_xyz: np.ndarray) -> tuple[list[float], list[float]]:
    """Convert Nx3 (X, Y, Z) points to azimuth degrees and Y range."""
    angles: list[float] = []
    ranges: list[float] = []
    for x, y, z in points_xyz:
        if y == 0:
            continue
        angles.append(float(np.degrees(np.arctan2(x, y))))
        ranges.append(float(y))
    return angles, ranges


# ---------------------------------------------------------------------------
# Bin helpers
# ---------------------------------------------------------------------------

def make_bins(fusion_params: dict) -> tuple[np.ndarray, np.ndarray]:
    a = fusion_params["bin_min_deg"]
    b = fusion_params["bin_max_deg"]
    step = fusion_params["bin_step_deg"]
    edges = np.arange(a, b + step, step)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


def angle_to_bin(angle_deg: float, edges: np.ndarray) -> int | None:
    """Return bin index (0..n-1) or None if angle is outside edges."""
    if angle_deg < edges[0] or angle_deg >= edges[-1]:
        return None
    return int(np.searchsorted(edges, angle_deg, side="right") - 1)


def fuse_angle_streams(
    fisheye_angles: list[float],
    thermal_angles: list[float],
    mmwave_angles: list[float],
    mmwave_ranges: list[float],
    fusion_params: dict,
    mmwave_points: np.ndarray | None = None,
    mmwave_velocities: list[tuple[float, float] | None] | None = None,
    motion_params: dict | None = None,
) -> FusionResult:
    edges, centers = make_bins(fusion_params)
    n_bins = len(centers)
    n_sensors = int(fusion_params.get("num_sensors", 3))

    hit_mask = np.zeros((n_bins, 3), dtype=bool)
    min_ranges: list[float | None] = [None] * n_bins

    for j, stream in enumerate((fisheye_angles, thermal_angles, mmwave_angles)):
        for angle in stream:
            idx = angle_to_bin(angle, edges)
            if idx is not None:
                hit_mask[idx, j] = True

    # mmWave-only range bookkeeping
    for angle, r in zip(mmwave_angles, mmwave_ranges):
        idx = angle_to_bin(angle, edges)
        if idx is None:
            continue
        if min_ranges[idx] is None or r < min_ranges[idx]:
            min_ranges[idx] = r

    scores = [float(hit_mask[i].sum()) / n_sensors for i in range(n_bins)]

    # Per-bin velocity + TTC. Only attempted when we have a point cloud
    # (positions) AND velocities for those points. Otherwise we return
    # all-None placeholders so the schema is shape-stable.
    if mmwave_points is not None and mmwave_velocities is not None and len(mmwave_points):
        per_bin_velocity_mps = aggregate_bin_velocity(
            mmwave_points, mmwave_velocities, edges
        )
    else:
        per_bin_velocity_mps = [None] * n_bins
    eps = float((motion_params or {}).get("ttc_eps_mps", 0.05))
    cap = float((motion_params or {}).get("ttc_cap_s", 60.0))
    per_bin_ttc_s = [
        ttc_from_velocity(r, v, eps=eps, cap_s=cap)
        for r, v in zip(min_ranges, per_bin_velocity_mps)
    ]

    return FusionResult(
        bin_edges=edges,
        bin_centers=centers,
        scores=scores,
        min_ranges=min_ranges,
        sensor_hit_mask=hit_mask,
        per_bin_velocity_mps=per_bin_velocity_mps,
        per_bin_ttc_s=per_bin_ttc_s,
        confirmed=[False] * n_bins,
    )


# ---------------------------------------------------------------------------
# Stateful pipeline
# ---------------------------------------------------------------------------

class ObstacleDetectionPipeline:
    """Holds per-clip state (running average, blob detectors) and intrinsics.

    Construct once per clip, call process_* per frame.
    """

    def __init__(self, intrinsics: dict[str, Any], detection: dict[str, Any],
                 extrinsics: dict[str, Any] | None = None,
                 attitude_provider: Any | None = None,
                 seg_provider: Any | None = None):
        self.intrinsics = intrinsics
        self.detection = detection
        # Camera heights above the waterline drive the monocular water-plane
        # range estimate. 0.27 m measured 2026-06-22 (extrinsics.yaml); the
        # radar<->camera transforms there are still placeholders.
        self.extrinsics = extrinsics if extrinsics is not None else load_extrinsics()
        heights = self.extrinsics.get("camera_height_m", {}) or {}
        self._camera_height = {
            "fisheye": float(heights.get("fisheye", 0.27)),
            "thermal": float(heights.get("thermal", 0.27)),
        }
        # Attitude reference for range: a BNO085 reader (anything with .get() ->
        # Attitude|None) takes priority; otherwise the per-frame detected
        # horizon, otherwise level. See configs/{detection,imu}.yaml.
        self._attitude = attitude_provider
        self._range_cfg = detection.get("range", {}) or {}
        # Attitude source of the most recent _range_up_vector call (see
        # FisheyeResult.range_reference). None until ranges are computed.
        self._last_range_reference: str | None = None

        # Water-segmentation augmentation (Branch: LaRS/eWaSR). Default off so
        # absent masks leave the existing fisheye range/horizon path untouched.
        # When enabled, a per-frame water mask (in the undistorted fisheye space)
        # supplies the range up-vector (water edge) and each detection's true
        # waterline-contact point. See scripts/utils/segmentation.py.
        self._seg_cfg = detection.get("segmentation", {}) or {}
        self._seg = seg_provider
        if self._seg is None and self._seg_cfg.get("enabled", False):
            root = self._seg_cfg.get("seg_root", "data/seg")
            provider = SegProvider(root)
            self._seg = provider if provider.available() else None
        contact = self._seg_cfg.get("contact", {}) or {}
        self._contact_params = ContactParams(
            search_pad_px=int(contact.get("search_pad_px", 40)),
            min_columns=int(contact.get("min_columns", 5)),
            water_below_px=int(contact.get("water_below_px", 3)),
            percentile=float(contact.get("percentile", 80.0)),
        )

        self._fisheye_blob = build_blob_detector(detection["fisheye"]["blob"])
        self._thermal_blob = build_blob_detector(detection["thermal"]["blob"])

        # Fisheye detector mode: "blob" (gradient/glint + SimpleBlobDetector,
        # the default) or "segmentation" (connected components of the obstacle
        # class in the water region: Phase B of "both, staged"). The seg
        # detector needs a water mask; it falls back to blob when none is present.
        self._fisheye_detector = detection["fisheye"].get("detector", "blob")
        seg_det = self._seg_cfg.get("detect", {}) or {}
        self._seg_detect_params = SegDetectParams(
            min_area=int(seg_det.get("min_area", 25)),
            water_edge_margin_px=int(seg_det.get("water_edge_margin_px", 10)),
            max_components=int(seg_det.get("max_components", 100)),
        )

        self.thermal_average: float = float(detection["thermal"]["initial_average"])

        # Thermal background model (thermal.background). "running_mean" is
        # AuthorZero's global scene-mean subtraction (default, byte-identical).
        # "mog2" is the per-pixel candidate from PLAN §I.4.7 #5 for the
        # uniform-night-water collapse; stateful per clip like the pipeline.
        bg_cfg = (detection["thermal"].get("background", {}) or {})
        self._thermal_mog2_cfg = bg_cfg.get("mog2", {}) or {}
        if bg_cfg.get("method", "running_mean") == "mog2":
            self._thermal_mog2 = cv2.createBackgroundSubtractorMOG2(
                history=int(self._thermal_mog2_cfg.get("history", 90)),
                varThreshold=float(self._thermal_mog2_cfg.get("var_threshold", 16.0)),
                detectShadows=False,
            )
        else:
            self._thermal_mog2 = None

        # Sun-elevation-scheduled thermal profiles (thermal.profiles; the
        # 2026-07-09 "time-of-day thermal roles" close-out, capture_runbook
        # §3.4). Default off = byte-identical: the base params / blob /
        # background model above stay in force and _thermal_params() returns
        # the untouched detection["thermal"]. Enabled, each named profile's
        # overrides are deep-merged onto the base thermal block once here;
        # the active profile is resolved per frame from the sun elevation
        # the driver provides via set_sun_elevation() (or from `force`).
        prof_cfg = detection["thermal"].get("profiles") or {}
        self._thermal_profiles_enabled = bool(prof_cfg.get("enabled", False))
        self._thermal_profile_day_min = float(
            prof_cfg.get("day_min_elev_deg", 10.0))
        self._thermal_profile_night_max = float(
            prof_cfg.get("night_max_elev_deg", -6.0))
        forced = prof_cfg.get("force")
        self._thermal_profile_forced: str | None = (
            str(forced) if forced else None)
        self._sun_elevation_deg: float | None = None
        self._active_thermal_profile: str | None = None
        self._thermal_profile_params: dict[str, dict] = {}
        self._thermal_profile_blobs: dict[str, Any] = {}
        self._warned_no_elevation = False
        if self._thermal_profiles_enabled:
            if (self._thermal_profile_forced is not None
                    and self._thermal_profile_forced
                    not in THERMAL_PROFILE_NAMES):
                raise ValueError(
                    f"thermal.profiles.force: unknown profile "
                    f"{self._thermal_profile_forced!r} "
                    f"(expected one of {THERMAL_PROFILE_NAMES})")
            base_t = {k: v for k, v in detection["thermal"].items()
                      if k != "profiles"}
            for name in THERMAL_PROFILE_NAMES:
                merged = _deep_merge(base_t, prof_cfg.get(name) or {})
                self._thermal_profile_params[name] = merged
                self._thermal_profile_blobs[name] = build_blob_detector(
                    merged["blob"])

        motion_cfg = detection.get("motion", {}) or {}
        self._radar_tracker = RadarPointTracker(
            RadarPointTrackerConfig(
                gate_m=float(motion_cfg.get("gate_m", 0.5)),
                max_dt_s=float(motion_cfg.get("max_dt_s", 0.5)),
            )
        )

        # Per-target motion tracker (motion.targets, default off): clusters
        # the filtered cloud into objects and combines Doppler radial speed
        # with cross-frame bearing-rate into a 2-D velocity + closing/
        # crossing/diverging state per target. Stateful per clip.
        tgt_cfg = motion_cfg.get("targets", {}) or {}
        self._target_tracker = (
            TargetMotionTracker(TargetMotionConfig.from_config(
                tgt_cfg, motion_cfg=motion_cfg))
            if tgt_cfg.get("enabled", False) else None)

        # Cross-sensor track identity (motion.targets.identity, default
        # off): pairs BBoxTracker camera tracks with radar target ids by
        # sustained point-overlap agreement; pairing evidence needs
        # fusion.association (the per-detection matched point indices).
        # Fisheye only: the thermal's 160 px frame + the value case
        # (dropout fill on the fisheye-visible subject) both live there.
        id_cfg = tgt_cfg.get("identity", {}) or {}
        self._identity = (
            TrackIdentityManager(IdentityConfig.from_config(id_cfg))
            if self._target_tracker is not None
            and id_cfg.get("enabled", False) else None)

        # Range-gated voting (fusion.vote_range_gate, default off): camera
        # detections whose raw water-plane estimate reads beyond the gate
        # don't raise bin threat scores. Unknown range keeps voting.
        gate_cfg = (detection.get("fusion", {}) or {}).get(
            "vote_range_gate", {}) or {}
        self._vote_gate_enabled = bool(gate_cfg.get("enabled", False))
        self._vote_gate_m = float(gate_cfg.get("max_vote_range_m", 30.0))

        # Pixel-level radar<->camera association (fusion.association,
        # default off: A.1 payoff, needs the measured extrinsics).
        assoc_cfg = (detection.get("fusion", {}) or {}).get("association", {}) or {}
        self._assoc_enabled = bool(assoc_cfg.get("enabled", False))
        self._assoc_params = AssociationParams.from_config(assoc_cfg)
        # Per-camera gate pad: max_px absorbs ANGULAR projection error, so a
        # pixel pad tuned on the 864-px fisheye (32 px ~ 4.3 deg) is ~13 deg
        # on the 160-px thermal, wider than a whole fusion bin: thermal blobs
        # were confirming bins with radar returns that belong 1-2 bins away
        # (measured, quad clip frame 88: open-water 0 deg bin "confirmed" by
        # dock returns at +10..+20 deg). thermal_max_px expresses the same
        # tolerance in thermal pixels (default 12 px ~ 4.3 deg at 2.81 px/deg).
        self._assoc_params_thermal = AssociationParams(
            max_px=float(assoc_cfg.get("thermal_max_px", 12.0)),
            min_points=self._assoc_params.min_points,
            range_agg=self._assoc_params.range_agg,
        )
        # Instance-mask pixel attribution (use_instance_masks, default off):
        # gate the projected radar return by the detection's YOLOv8-seg
        # instance POLYGON (data/det_seg, undistorted-fisheye space) instead
        # of the padded bbox. Fisheye only: the polygons don't exist in the
        # thermal frame. Falls back to the box gate per detection when no
        # instance record covers the frame/detection (see _apply_association).
        self._inst_provider = None
        if self._assoc_enabled and self._assoc_params.use_instance_masks:
            inst_root = assoc_cfg.get("instance_root", "data/det_seg")
            provider = InstanceSegProvider(inst_root)
            self._inst_provider = provider if provider.available() else None
        self._assoc_use_radar_range = bool(assoc_cfg.get("use_radar_range", True))
        self._assoc_compat_ratio = float(
            assoc_cfg.get("range_compat_ratio", 0.5) or 0.0)
        # Confirmation semantics (see _confirm_bins): same-bin agreement
        # between detection and matched-return bearings, and which of the
        # two bearings anchors the confirmed flag.
        self._assoc_confirm_same_bin = bool(
            assoc_cfg.get("confirm_same_bin", True))
        self._assoc_confirm_anchor = str(
            assoc_cfg.get("confirm_anchor", "radar"))

        # Bearing model per camera: "linear" = legacy (x - cx)/pix_deg_ratio
        # magic numbers; "pinhole" = atan((x - cx_K)/fx) from the calibrated
        # camera matrix of the undistorted image (fisheye: K; thermal: the
        # optimal new K). The two disagree by ~4° on the fisheye (cx 472 vs
        # 444.1): arbitrated empirically via association_report.
        self._bearing_model = {
            "fisheye": str(detection["fisheye"].get("bearing_model", "linear")),
            "thermal": str(detection["thermal"].get("bearing_model", "linear")),
        }

    # -- thermal profiles (sun-elevation scheduler) ----------------------------

    def set_sun_elevation(self, elevation_deg: float | None) -> None:
        """Provide the current solar elevation (deg above horizon).

        The thermal profile scheduler (thermal.profiles) resolves the
        active profile from this value once per frame. Replay drivers
        compute it from the frame timestamp (fusion.py, same math as
        fusion_model/build_features.py); live drivers from the RTC-backed
        clock + GPS fix (gps_boat1.current_sun_elevation). Never calling
        this (or passing None) honestly resolves to the day profile, with
        a single stderr log line.
        """
        self._sun_elevation_deg = (
            float(elevation_deg) if elevation_deg is not None else None)

    def _activate_thermal_profile(self, name: str) -> None:
        """Swap the thermal per-profile state (blob detector, background
        model config) to `name`. Switching to a mog2 profile starts a FRESH
        background model — it needs ~history frames to settle, which is
        honest for the once-per-outing dusk/dawn transitions this is built
        for. thermal_average continuity is preserved (process_thermal keeps
        updating it under every mode)."""
        params = self._thermal_profile_params[name]
        self._thermal_blob = self._thermal_profile_blobs[name]
        bg_cfg = params.get("background", {}) or {}
        self._thermal_mog2_cfg = bg_cfg.get("mog2", {}) or {}
        if bg_cfg.get("method", "running_mean") == "mog2":
            self._thermal_mog2 = cv2.createBackgroundSubtractorMOG2(
                history=int(self._thermal_mog2_cfg.get("history", 90)),
                varThreshold=float(
                    self._thermal_mog2_cfg.get("var_threshold", 16.0)),
                detectShadows=False,
            )
        else:
            self._thermal_mog2 = None
        if self._active_thermal_profile is not None:
            print(f"[thermal.profiles] {self._active_thermal_profile} -> "
                  f"{name} (sun elevation "
                  f"{self._sun_elevation_deg if self._sun_elevation_deg is not None else 'n/a'})",
                  file=sys.stderr)
        self._active_thermal_profile = name

    def _thermal_params(self) -> dict:
        """The thermal parameter set in force for this frame.

        Scheduler off -> the base detection["thermal"] block, untouched
        (byte-identical path). On -> resolve the profile (force wins over
        elevation; absent elevation -> day, logged once), activate it if it
        changed, and return its merged params.
        """
        if not self._thermal_profiles_enabled:
            return self.detection["thermal"]
        if self._thermal_profile_forced is not None:
            name = self._thermal_profile_forced
        else:
            if (self._sun_elevation_deg is None
                    and not self._warned_no_elevation):
                print("[thermal.profiles] no sun elevation available -> "
                      "day profile (defaults)", file=sys.stderr)
                self._warned_no_elevation = True
            name = resolve_thermal_profile(
                self._sun_elevation_deg,
                self._thermal_profile_day_min,
                self._thermal_profile_night_max)
        if name != self._active_thermal_profile:
            self._activate_thermal_profile(name)
        return self._thermal_profile_params[name]

    # -- horizon ---------------------------------------------------------------

    def _detect_horizon_imu(self, undistorted: np.ndarray, h_cfg: dict,
                            P: np.ndarray
                            ) -> tuple[np.ndarray, float, float, float]:
        """detect_horizon, optionally seeded/gated by the IMU attitude.

        With `horizon.imu_seed.enabled` and a live/replayed attitude, the
        RANSAC line is accepted only when it lands within `max_dev_px` of the
        IMU-predicted horizon (measured at the principal point column).
        Otherwise (RANSAC low-confidence OR locked onto a shoreline/dock)
        the IMU-predicted line supplies the mask and line instead, with
        confidence 1.0 (the line's trust then comes from the IMU, not the
        edge fit). Off / no attitude -> byte-identical to detect_horizon.
        """
        from scripts.utils.cv_common import horizon_kwargs, mask_below_line
        mask, slope, intercept, conf = detect_horizon(
            undistorted, **horizon_kwargs(h_cfg))
        seed_cfg = h_cfg.get("imu_seed") or {}
        if not seed_cfg.get("enabled", False):
            return mask, slope, intercept, conf
        att = self._attitude.get() if self._attitude is not None else None
        if att is None:
            return mask, slope, intercept, conf
        from scripts.utils.geometry import horizon_line_from_up
        s_i, b_i = horizon_line_from_up(att.up_vector_camera(), P)
        cx = float(np.asarray(P, dtype=np.float64)[0, 2])
        dev = abs((slope * cx + intercept) - (s_i * cx + b_i))
        conf_ok = conf >= float(h_cfg.get("confidence_thresh", 0.5))
        if conf_ok and dev <= float(seed_cfg.get("max_dev_px", 60.0)):
            return mask, slope, intercept, conf
        return (mask_below_line(undistorted.shape[:2], s_i, b_i),
                float(s_i), float(b_i), 1.0)

    def _vote_mask(self, ranges_raw: list[float | None]) -> list[bool]:
        """Fusion-vote eligibility per detection (see FisheyeResult.votes)."""
        if not self._vote_gate_enabled:
            return [True] * len(ranges_raw)
        return [r is None or r <= self._vote_gate_m for r in ranges_raw]

    # -- bearings --------------------------------------------------------------

    def _bearings(self, sensor: str, coords: list[tuple[int, int]],
                  intr: dict, P: np.ndarray | None = None) -> list[float]:
        """Detection bearings under the configured model (see __init__)."""
        if self._bearing_model.get(sensor, "linear") == "pinhole":
            return pinhole_bearings(coords, P if P is not None else intr["K"])
        pix_deg = intr["pix_deg_ratio"]
        cx0 = intr["cx"]
        return [(u - cx0) / pix_deg for u, _v in coords]

    # -- monocular range -----------------------------------------------------

    def _range_up_vector(
        self,
        P: np.ndarray,
        horizon_line: tuple[float, float, float],
        seg_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """World-up (camera coords) for range, from the best available source.

        Sources, walked in `range.attitude_priority` order (default:
        imu > water_edge > horizon > level: the historical hard-coded
        order):
          - imu:        a connected/replayed BNO085 (None when stale);
          - water_edge: the segmentation water-edge line, gated on its own
                        confidence (far more robust than Canny-RANSAC, which
                        locks onto shorelines/docks);
          - horizon:    the detected RANSAC line, gated on (a) confidence and
                        (b) plausibility (`horizon_max_tilt_px`: a horizon
                        far from the principal point implies an absurd tilt:
                        almost always a shoreline lock that would fake near
                        ranges);
          - level:      pitch = roll = 0 (always succeeds).

        When `range.imu_gate_deg` > 0 and an IMU attitude exists, a vision
        source (water_edge/horizon) is additionally rejected if its implied
        pitch/roll deviates from the IMU's by more than the gate: the IMU
        acting as a sanity prior over per-frame optical references.

        Side effect: records the winning source name on
        `self._last_range_reference` so process_fisheye/process_thermal can
        stamp it onto their results (range provenance for telemetry /
        feature tables). Return type unchanged: external callers
        (dashboard, range_validate, tests) see the same np.ndarray.
        """
        up, source = self._range_up_vector_with_source(P, horizon_line, seg_mask)
        self._last_range_reference = source
        return up

    def _range_up_vector_with_source(
        self,
        P: np.ndarray,
        horizon_line: tuple[float, float, float],
        seg_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str]:
        att = self._attitude.get() if self._attitude is not None else None
        gate = float(self._range_cfg.get("imu_gate_deg", 0.0) or 0.0)

        def _gated(up: np.ndarray) -> np.ndarray | None:
            """Reject a vision up-vector too far from the IMU's attitude."""
            if att is None or gate <= 0.0:
                return up
            from scripts.utils.geometry import pitch_roll_from_up
            p_v, r_v = pitch_roll_from_up(up)
            dp = abs(np.degrees(p_v - att.pitch_rad))
            dr = abs(np.degrees(r_v - att.roll_rad))
            return up if (dp <= gate and dr <= gate) else None

        priority = list(self._range_cfg.get(
            "attitude_priority", ("imu", "water_edge", "horizon", "level")))
        for source in priority:
            if source == "imu":
                if att is not None:
                    return att.up_vector_camera(), "imu"
            elif source == "water_edge":
                if seg_mask is not None and self._seg_cfg.get("use_for_horizon", True):
                    s_slope, s_intercept, s_conf = horizon_from_water(seg_mask)
                    if s_conf >= float(self._seg_cfg.get("horizon_min_confidence", 0.4)):
                        up = _gated(up_from_horizon_line(s_slope, s_intercept, P))
                        if up is not None:
                            return up, "water_edge"
            elif source == "horizon":
                slope, intercept, conf = horizon_line
                if (self._range_cfg.get("use_detected_horizon", True)
                        and conf >= float(self._range_cfg.get("horizon_min_confidence", 0.5))):
                    cx, cy = float(P[0, 2]), float(P[1, 2])
                    horizon_row = slope * cx + intercept
                    max_tilt = float(self._range_cfg.get("horizon_max_tilt_px", 170))
                    if abs(horizon_row - cy) <= max_tilt:
                        up = _gated(up_from_horizon_line(slope, intercept, P))
                        if up is not None:
                            return up, "horizon"
            elif source == "level":
                return UP_LEVEL.copy(), "level"
        return UP_LEVEL.copy(), "level"

    def _seg_fp_filter(self, coords, sizes, angles, seg_mask):
        """Keep only detections with enough OBSTACLE support in the mask.

        A blob over (near-)pure water is a glint/wave reflection FP; we drop it.
        Threshold from segmentation.fp_filter.min_obstacle_frac.
        """
        fp = self._seg_cfg.get("fp_filter", {}) or {}
        min_frac = float(fp.get("min_obstacle_frac", 0.1))
        kc, ks, ka = [], [], []
        for (px, py), size, ang in zip(coords, sizes, angles):
            r = size / 2.0
            frac = detection_obstacle_fraction(seg_mask, (px - r, py - r, px + r, py + r))
            if frac >= min_frac:
                kc.append((px, py))
                ks.append(size)
                ka.append(ang)
        return kc, ks, ka

    def _detection_ranges(
        self,
        coords: list[tuple[int, int]],
        sizes: list[float],
        P: np.ndarray,
        camera_height_m: float,
        horizon_line: tuple[float, float, float],
        seg_mask: np.ndarray | None = None,
    ) -> list[float | None]:
        """Water-plane range per blob.

        `coords`/`sizes` are blob centres + diameters in the UNDISTORTED image;
        `P` is that image's camera matrix. Attitude comes from the IMU, the
        water-edge segmentation, or the RANSAC horizon (see `_range_up_vector`).

        When a water mask is available and `segmentation.use_for_range` is set,
        each blob's range is computed from its true waterline-contact point
        (where the obstacle meets water inside the box) rather than the bbox
        bottom-centre: the fix for tilted-horizon boxes that over-capture water
        and read as too close. If no obstacle->water transition is found in the
        box, we fall back to the bbox-bottom estimate. Ranges beyond
        `range.max_range_m` are dropped (None) as untrustworthy.
        """
        if not self._range_cfg.get("enabled", True):
            self._last_range_reference = None
            return [None] * len(coords), [None] * len(coords)
        up = self._range_up_vector(P, horizon_line, seg_mask)
        max_range = self._range_cfg.get("max_range_m")
        max_range = float(max_range) if max_range is not None else None
        # Pixel-noise reliability check (range.uncertainty_px): a range is
        # announced only when the optimistic bound (the same ray shifted up
        # by the mask/contact noise floor) stays inside max_range_m. Near
        # the envelope edge one pixel is worth 1-1.5 m, and mask boundary
        # dilation / water reflections put the contact BELOW the true
        # waterline (systematic under-estimation, e.g. the far pontoon at
        # frames 358-369 reading 7-14 m). Noise-dominated readings report
        # None (">15m") instead of a falsely-confident number.
        unc_px = float(self._range_cfg.get("uncertainty_px", 0.0) or 0.0)

        def _reliable(point_uv: tuple[float, float]) -> bool:
            if unc_px <= 0.0 or max_range is None:
                return True
            hi = range_from_contact_point(
                (point_uv[0], point_uv[1] - unc_px), P, up, camera_height_m,
                max_range_m=None,
            )
            return hi is not None and hi <= max_range

        use_contact = seg_mask is not None and self._seg_cfg.get("use_for_range", True)
        ranges: list[float | None] = []
        # Raw (untruncated) estimates back the fusion vote gate: "reads as
        # 300 m" and "reads as unknown" must stay distinguishable even though
        # both report None in `ranges` (beyond max_range_m = untrustworthy
        # as a NUMBER, still trustworthy as evidence of "far").
        ranges_raw: list[float | None] = []
        for (px, py), size in zip(coords, sizes):
            r = size / 2.0
            bbox = (px - r, py - r, px + r, py + r)
            if use_contact:
                contact = water_edge_contact(seg_mask, bbox, self._contact_params)
                if contact is not None:
                    raw = range_from_contact_point(
                        contact, P, up, camera_height_m, max_range_m=None,
                    )
                    ranges_raw.append(raw)
                    ranges.append(raw if raw is not None
                                  and (max_range is None or raw <= max_range)
                                  and _reliable(contact)
                                  else None)
                    continue
            raw = range_from_water_plane_up(
                bbox, P, up, camera_height_m, max_range_m=None,
            )
            ranges_raw.append(raw)
            ranges.append(raw if raw is not None
                          and (max_range is None or raw <= max_range)
                          and _reliable((px, py + r))
                          else None)
        return ranges, ranges_raw

    # -- single-sensor entry points -----------------------------------------

    def process_fisheye(
        self,
        frame_bgr: np.ndarray,
        frame_id: str | None = None,
        seg_mask: np.ndarray | None = None,
    ) -> FisheyeResult:
        intr = self.intrinsics["fisheye"]
        h_params = self.detection["horizon"]
        f_params = self.detection["fisheye"]

        # Resolve the water mask (caller-supplied wins; else look up by frame_id).
        if seg_mask is None and self._seg is not None:
            seg_mask = self._seg.get(frame_id)

        undistorted = undistort_fisheye(frame_bgr, intr["K"], intr["D"])

        # Frame luminance stats (context features; see FisheyeResult).
        # Computed on the full undistorted gray, pre-mask: the horizon
        # mask zeroes the sky and would fake darkness.
        lum_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        lum_mean = float(lum_gray.mean())
        lum_p05, lum_p95 = (float(v) for v in np.percentile(lum_gray, (5, 95)))
        is_dark = lum_mean < float(f_params.get("darkness_thresh", 25.0))

        # segmentation.darkness_gate (2026-09-12, default ON): the live
        # SegWorker refuses frames below the luminance floor (a near-black
        # frame segments as "obstacle everywhere": measured 99.6-100 %
        # obstacle on every 2026-09-08 night chunk, fisheye mean luminance
        # 0.1-1.7), so an offline replay that consumes precomputed masks must
        # drop them on the same frames or it diverges from the box (same
        # bundle + config: box night intercept 0.19 vs offline 0.41) and
        # every seg-derived feature/label on those frames is noise.
        if (seg_mask is not None and is_dark
                and bool(self._seg_cfg.get("darkness_gate", True))):
            seg_mask = None

        # horizon.lazy (2026-09-04, default off = byte-identical): when a
        # water mask is present, the segmented water edge is already the
        # range reference (range.attitude_priority) and the seg detector does
        # not need the RANSAC mask, so the Canny/RANSAC horizon (~120 ms of a
        # 230 ms fisheye stage on the Pi 4, O(iters x edge points)) is
        # skipped and the water-edge line stands in for it. Falls back to the
        # RANSAC whenever the edge fit is weak or there is no mask.
        lazy_line = None
        if bool(h_params.get("lazy", False)) and seg_mask is not None:
            w_slope, w_intercept, w_conf = horizon_from_water(seg_mask)
            if w_conf >= float(h_params.get("lazy_min_conf", 0.5)):
                lazy_line = (float(w_slope), float(w_intercept), float(w_conf))
            elif self._fisheye_detector == "segmentation":
                # Measured on the box (2026-09-07, dockside bench clip): the
                # water-edge fit reads conf ~0.1 wherever the boundary is not
                # a straight line (piers, moored boats), so the RANSAC ran
                # twice per frame (173 ms) for a mask the segmentation
                # detector never uses. With the seg detector the horizon
                # mask only gates the classical blob path, so keep the whole
                # frame (row 0 = the documented fallback_row semantics) and
                # report the low confidence: the range reference then goes
                # water_edge -> IMU -> level as configured, never this line.
                lazy_line = (0.0, 0.0, float(w_conf))
        if lazy_line is not None:
            slope, intercept, conf = lazy_line
            rows_i, cols_i = undistorted.shape[:2]
            y_line = np.clip((slope * np.arange(cols_i) + intercept).astype(int), 0, rows_i - 1)
            horizon_mask = (np.arange(rows_i)[:, None] >= y_line[None, :]).astype(np.uint8) * 255
        else:
            horizon_mask, slope, intercept, conf = self._detect_horizon_imu(
                undistorted, h_params, np.asarray(intr["K"], float)
            )
        masked = cv2.bitwise_and(undistorted, undistorted, mask=horizon_mask)

        if self._fisheye_detector == "segmentation" and seg_mask is not None:
            # Segmentation-driven detection: connected components of the
            # obstacle class in the water region (Phase B). obstacle_mask is the
            # binary obstacle layer (for viz/overlays).
            obstacle_mask = ((seg_mask == SEG_OBSTACLE).astype(np.uint8)) * 255
            coords, sizes = seg_obstacle_detections(seg_mask, self._seg_detect_params)
            angles = self._bearings("fisheye", coords, intr)
        else:
            obstacle_mask = gradient_obstacle_detection(
                masked,
                gaussian_window=f_params["gaussian_window"],
                gradient_threshold=f_params["gradient_threshold"],
                median_blur=f_params.get("median_blur", 3),
                glint_low=f_params["glint_inpaint_low"],
                glint_high=f_params["glint_inpaint_high"],
                glint_inpaint_radius=f_params["glint_inpaint_radius"],
            )
            kps = self._fisheye_blob.detect(obstacle_mask)
            coords, sizes, angles = extract_kp(kps, intr["cx"], intr["pix_deg_ratio"])
            if self._bearing_model["fisheye"] != "linear":
                angles = self._bearings("fisheye", coords, intr)

        # Seg-based false-positive filter (default off): drop detections that
        # sit (almost) entirely in water: glints / waves / reflections the
        # detector fired on. Needs a mask; gated by segmentation.fp_filter.
        if (seg_mask is not None and coords
                and (self._seg_cfg.get("fp_filter", {}) or {}).get("enabled", False)):
            coords, sizes, angles = self._seg_fp_filter(coords, sizes, angles, seg_mask)

        # Undistorted fisheye keeps the original K as its camera matrix
        # (cv_common.undistort_fisheye passes K as the new matrix).
        ranges, ranges_raw = self._detection_ranges(
            coords, sizes, intr["K"], self._camera_height["fisheye"],
            (slope, intercept, conf), seg_mask=seg_mask,
        )
        return FisheyeResult(
            undistorted=undistorted,
            horizon_mask=horizon_mask,
            horizon_line=(slope, intercept, conf),
            obstacle_mask=obstacle_mask,
            coords=coords,
            sizes=sizes,
            angles=angles,
            ranges=ranges,
            seg_mask=seg_mask,
            raw_ranges=ranges_raw,
            votes=self._vote_mask(ranges_raw),
            luminance_mean=lum_mean,
            luminance_p05=lum_p05,
            luminance_p95=lum_p95,
            is_dark=is_dark,
            range_reference=self._last_range_reference,
        )

    def process_thermal(self, frame_bgr: np.ndarray) -> ThermalResult:
        intr = self.intrinsics["thermal"]
        # Resolved per frame: the base thermal block, or — with the
        # thermal.profiles sun-elevation scheduler enabled — the active
        # profile's merged params (also swaps blob detector + background
        # model on a profile change).
        t_params = self._thermal_params()
        # Thermal-specific horizon overrides (thermal.horizon) merged over
        # the shared block: the 160x120 LWIR frame wants different Canny/
        # RANSAC settings than the 864x648 fisheye. Absent -> identical.
        h_params = {**self.detection["horizon"],
                    **(t_params.get("horizon") or {})}

        # Stuck sensor row(s) repaired first: it is a sensor defect, so it is
        # fixed in sensor coordinates before the remap (thermal.dead_rows;
        # base block, inherited by every profile). [] -> byte-identical.
        frame_bgr = repair_dead_rows(
            frame_bgr, self.detection["thermal"].get("dead_rows") or [])
        undistorted = undistort_thermal(frame_bgr, intr["K"], intr["D"])

        # Wet-cover quality guard: contrast stats on the full frame,
        # detections suppressed below the floors (see detection.yaml).
        q_std, q_dyn = thermal_quality_stats(undistorted)
        qg = t_params.get("quality_guard", {}) or {}
        quality_ok = not (qg.get("enabled", False) and (
            q_std < float(qg.get("min_std", 25.0))
            or q_dyn < float(qg.get("min_dyn_range", 120.0))))

        # The undistorted thermal's camera matrix (needed by the IMU horizon
        # seed and pinhole bearings below; lru-cached, so cheap per frame).
        h_px0, w_px0 = undistorted.shape[:2]
        P_pre = pinhole_new_K(intr["K"], intr["D"], (w_px0, h_px0))
        horizon_mask, slope, intercept, conf = self._detect_horizon_imu(
            undistorted, h_params, P_pre
        )
        masked = cv2.bitwise_and(undistorted, undistorted, mask=horizon_mask)

        # Optional denoise/CLAHE preprocessing (thermal.preprocess, default
        # off = byte-identical). Applied to the full gray BEFORE masking so
        # CLAHE tiles aren't skewed by the zeroed sky, then re-masked.
        pp_cfg = t_params.get("preprocess", {}) or {}
        pp_on = bool(pp_cfg.get("enabled", False))
        if self._thermal_mog2 is not None:
            # Per-pixel MOG2 background. Learned on the full (unmasked)
            # gray frame so the horizon mask isn't baked into the model;
            # the mask is applied to the foreground output instead.
            gray = (cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                    if undistorted.ndim == 3 else undistorted)
            if pp_on:
                gray = thermal_preprocess(gray, pp_cfg)
            lr = float(self._thermal_mog2_cfg.get("learning_rate", -1))
            fg = self._thermal_mog2.apply(gray, learningRate=lr)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            obstacle_mask = cv2.bitwise_and(fg, fg, mask=horizon_mask)
            new_average = float(cv2.mean(gray)[0])
        else:
            detect_in = masked
            if pp_on:
                gray = (cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                        if undistorted.ndim == 3 else undistorted)
                proc = thermal_preprocess(gray, pp_cfg)
                detect_in = cv2.bitwise_and(proc, proc, mask=horizon_mask)
            obstacle_mask, new_average = thermal_obstacle_detection(
                detect_in,
                self.thermal_average,
                horizon_mask,
                object_thresh=t_params["object_thresh"],
                contrast_guard=t_params["contrast_guard"],
            )
        # Running mean update (same recurrence as thermal.py). Kept even
        # under MOG2 / a tripped guard so new_average telemetry and the
        # running_mean state stay continuous across mode switches.
        self.thermal_average = (self.thermal_average + new_average) / 2.0

        # Undistorted thermal uses the optimal new K (pinhole), not the raw K.
        h_px, w_px = undistorted.shape[:2]
        P_thermal = pinhole_new_K(intr["K"], intr["D"], (w_px, h_px))
        if quality_ok:
            kps = self._thermal_blob.detect(obstacle_mask)
            coords, sizes, angles = extract_kp(kps, intr["cx"], intr["pix_deg_ratio"])
            if self._bearing_model["thermal"] != "linear":
                angles = self._bearings("thermal", coords, intr, P=P_thermal)
        else:
            # Blind cover: whatever structure survives is water-film smear,
            # not scene content. No detections -> no fusion vote.
            coords, sizes, angles = [], [], []
        ranges, ranges_raw = self._detection_ranges(
            coords, sizes, P_thermal, self._camera_height["thermal"],
            (slope, intercept, conf),
        )
        return ThermalResult(
            undistorted=undistorted,
            horizon_mask=horizon_mask,
            horizon_line=(slope, intercept, conf),
            obstacle_mask=obstacle_mask,
            coords=coords,
            sizes=sizes,
            angles=angles,
            new_average=new_average,
            ranges=ranges,
            quality_ok=quality_ok,
            quality_std=q_std,
            quality_dyn_range=q_dyn,
            raw_ranges=ranges_raw,
            votes=self._vote_mask(ranges_raw),
            range_reference=self._last_range_reference,
            profile=(self._active_thermal_profile
                     if self._thermal_profiles_enabled else None),
        )

    def process_mmwave(
        self,
        points_xyz: np.ndarray,
        timestamp: str | None = None,
    ) -> MMWaveResult:
        """Filter the point cloud and attach per-point velocities.

        `points_xyz` is Nx3 (X, Y, Z), Nx4 (X, Y, Z, Doppler radial m/s;
        the CSV `V` column captured since 2026-07-06), or Nx6 with the TLV-7
        side-info columns appended (X, Y, Z, V, SNR, NOISE dB;
        captured since 2026-07-09). With Doppler present it wins per
        point; the cross-frame tracker fills the NaN gaps and remains the
        sole source for Nx3 input. SNR/NOISE ride along as per-point
        metadata (`snr_db`/`noise_db`, filtered in lockstep with the
        cloud). `MMWaveResult` always carries Nx3 positions so downstream
        consumers are unaffected by the extra columns.
        """
        m_params = self.detection["mmwave"]
        pts_in = np.asarray(points_xyz, dtype=np.float64)
        n = len(pts_in)
        if n == 0 or n > m_params.get("max_objects", 100) or n < m_params.get("min_objects", 1):
            # Still tick the tracker so its prev state ages out cleanly.
            self._radar_tracker.update(np.empty((0, 3)), timestamp)
            return MMWaveResult(points_xyz=np.empty((0, 3)), angles=[], ranges=[],
                                velocities_xy=[], velocity_source="tracker")

        doppler = pts_in[:, 3] if pts_in.shape[1] >= 4 else None
        snr_col = pts_in[:, 4] if pts_in.shape[1] >= 5 else None
        noise_col = pts_in[:, 5] if pts_in.shape[1] >= 6 else None
        pts3 = pts_in[:, :3]
        # Radar mount yaw: the board is rotated slightly about the vertical
        # relative to the cameras, so every reported bearing carries a constant
        # offset. Rotate the CLOUD rather than patching the bearing, so the
        # y-window, the self-clutter zone, the tracker, association and the
        # image projection all work in one corrected frame.
        # `mount_yaw_deg` is the measured radar-minus-truth offset; correcting
        # it means rotating by its negative.
        yaw_deg = float(m_params.get("mount_yaw_deg", 0.0) or 0.0)
        if yaw_deg:
            th = np.radians(-yaw_deg)
            c, s = np.cos(th), np.sin(th)
            xr = pts3[:, 0] * c + pts3[:, 1] * s
            yr = -pts3[:, 0] * s + pts3[:, 1] * c
            pts3 = np.column_stack([xr, yr, pts3[:, 2]])
        y = pts3[:, 1]
        keep = (y >= m_params["y_min"]) & (y <= m_params["y_max"])
        # Boat-frame self-clutter box (mmwave.self_clutter, default off):
        # suppress the measured mount/enclosure reflection zone, but keep
        # points with real Doppler: a mover crossing the zone still counts.
        sc = m_params.get("self_clutter", {}) or {}
        if sc.get("enabled", False):
            x = pts3[:, 0]
            in_zone = ((x >= float(sc.get("x_min", -0.75)))
                       & (x <= float(sc.get("x_max", 0.25)))
                       & (y >= float(sc.get("zone_y_min", 0.0)))
                       & (y <= float(sc.get("zone_y_max", 1.2))))
            if doppler is not None:
                moving = np.abs(np.nan_to_num(doppler, nan=0.0)) \
                    > float(sc.get("keep_if_doppler_above", 0.15))
                in_zone &= ~moving
            keep &= ~in_zone
        pts = pts3[keep]
        angles, ranges = mmwave_angles_ranges(pts)
        # Always tick the tracker (keeps its prev-frame state warm) even
        # when Doppler covers every point this frame.
        tracker_v = self._radar_tracker.update(pts, timestamp)
        if doppler is not None:
            doppler_v = radar_velocity_from_doppler(pts, doppler[keep])
            velocities = [d if d is not None else t
                          for d, t in zip(doppler_v, tracker_v)]
            used_doppler = any(d is not None for d in doppler_v)
            used_tracker = any(d is None and t is not None
                               for d, t in zip(doppler_v, tracker_v))
            if used_doppler and used_tracker:
                source = "mixed"
            elif used_doppler:
                source = "doppler"
            else:
                source = "tracker"
        else:
            velocities = tracker_v
            source = "tracker"
        def _side_info(col: np.ndarray | None) -> list[float | None]:
            if col is None:
                return []
            return [float(v) if np.isfinite(v) else None for v in col[keep]]

        return MMWaveResult(
            points_xyz=pts,
            angles=angles,
            ranges=ranges,
            velocities_xy=velocities,
            velocity_source=source,
            snr_db=_side_info(snr_col),
            noise_db=_side_info(noise_col),
        )

    # -- radar <-> camera association -----------------------------------------

    def _apply_association(self, res, m_res: "MMWaveResult",
                           P: np.ndarray, T_radar_to_cam: np.ndarray,
                           params: "AssociationParams | None" = None,
                           instances: list | None = None) -> None:
        """Attach radar evidence to one camera result (mutates `res`).

        Projects the (y-filtered) radar cloud into the camera's undistorted
        image and gates it against each detection's padded bbox (`params`
        selects the per-camera gate; default = the fisheye-scaled one). When
        `association.use_radar_range` is set, associated detections take the
        radar range (the monocular estimate moves to `mono_ranges`).

        `instances` (use_instance_masks, fisheye only) upgrades the pixel
        gate per detection: where a YOLOv8-seg instance polygon covers the
        detection, the gate is point-in-polygon (dilated by
        instance_margin_px) instead of the padded bbox: a tall edge box no
        longer sweeps up foreground returns that merely project inside its
        rectangle. Detections without a covering instance keep the box gate.
        """
        if res is None or not res.coords or not len(m_res.points_xyz):
            return
        p = params or self._assoc_params
        det_polys = None
        undist = getattr(res, "undistorted", None)
        if p.use_instance_masks and instances and undist is not None:
            h_img, w_img = undist.shape[:2]
            polys = polygons_from_instances(instances, (w_img, h_img))
            if polys:
                match_idx = match_polygons_to_detections(
                    polys, res.coords, res.sizes)
                det_polys = [polys[k] if k is not None else None
                             for k in match_idx]
        px = project_radar_to_undistorted(m_res.points_xyz, P, T_radar_to_cam)
        assoc = associate_radar_to_detections(
            m_res.points_xyz, px, res.coords, res.sizes, p,
            det_polygons=det_polys)
        res.radar_gates = [m.gate for m in assoc.matches]
        res.radar_ranges = [m.radar_range_m for m in assoc.matches]
        res.radar_hits = [m.n_points for m in assoc.matches]
        res.radar_bearings = [m.radar_bearing_deg for m in assoc.matches]
        res.radar_point_indices = [list(m.point_indices or [])
                                   for m in assoc.matches]
        # Per-detection matched returns as (bearing_deg, range_m) rows:
        # _confirm_bins anchors the confirmed flag on these (the bins the
        # evidence actually occupies) instead of any single centroid.
        res.radar_match_points = [
            (np.column_stack([
                np.degrees(np.arctan2(m_res.points_xyz[m.point_indices, 0],
                                      m_res.points_xyz[m.point_indices, 1])),
                m_res.points_xyz[m.point_indices, 1]])
             if m.point_indices else None)
            for m in assoc.matches
        ]
        # Foreground-return arbitration. Two rules, in order:
        #
        # 1. SILHOUETTE (needs a seg mask): a matched return that projects
        #    onto OBSTACLE pixels belongs to this object (or something
        #    visually occluding it): trust the radar even when the camera's
        #    own estimate disagrees (the camera reads wrong-far on near
        #    objects whose contact lands at the water boundary). The range
        #    re-aggregates over on-silhouette points only.
        # 2. COMPATIBILITY (association.range_compat_ratio): with no
        #    on-silhouette point (or no mask), a radar range dramatically
        #    NEARER than the camera's raw estimate (or camera-raw = above-
        #    horizon/far) is a foreground object's return swept up by the
        #    pixel gate (e.g. dock clutter at 2.9 m inside a far pillar's
        #    box): reject for range/confirmation. radar_hits stays either
        #    way (the return is real; radar_veto rightly stays quiet).
        seg = getattr(res, "seg_mask", None)
        for i, m in enumerate(assoc.matches):
            if m.radar_range_m is None:
                continue
            on_sil_ranges = []
            if seg is not None and m.point_indices:
                h_s, w_s = seg.shape[:2]
                for j in m.point_indices:
                    u, v = px[j]
                    if not (np.isfinite(u) and np.isfinite(v)):
                        continue
                    ui, vi = int(round(u)), int(round(v))
                    if 0 <= vi < h_s and 0 <= ui < w_s \
                            and seg[vi, ui] == SEG_OBSTACLE:
                        on_sil_ranges.append(float(m_res.points_xyz[j, 1]))
            if on_sil_ranges:
                agg = (float(np.median(on_sil_ranges))
                       if self._assoc_params.range_agg == "median"
                       else float(min(on_sil_ranges)))
                res.radar_ranges[i] = agg
                continue
            if self._assoc_compat_ratio > 0 and res.raw_ranges:
                raw = (res.raw_ranges[i]
                       if i < len(res.raw_ranges) else None)
                if raw is None or m.radar_range_m < self._assoc_compat_ratio * raw:
                    res.radar_ranges[i] = None
        if self._assoc_use_radar_range:
            res.mono_ranges = list(res.ranges)
            res.ranges = [
                rr if rr is not None else mono
                for rr, mono in zip(res.radar_ranges, res.ranges)
            ]

    def _apply_radar_veto(self, res, m_res: "MMWaveResult") -> None:
        """Demote near camera ranges the radar contradicts (range.radar_veto).

        A camera detection announcing 1-8 m inside the radar's usable cone,
        with ZERO radar points in its box while the radar returned points
        this frame, is physically contradictory: in practice a water
        reflection or mask artefact read as near (the far-pontoon case).
        The range moves to mono_ranges and reports None (">15m" display).
        Requires association (radar_hits per detection); no-ops without it.
        """
        veto = self._range_cfg.get("radar_veto", {}) or {}
        if not veto.get("enabled", False):
            return
        if res is None or not res.radar_ranges or not len(m_res.points_xyz):
            return   # association absent or radar silent -> no veto
        lo = float(veto.get("min_m", 1.0))
        hi = float(veto.get("max_m", 8.0))
        max_b = float(veto.get("max_bearing_deg", 50.0))
        if not res.mono_ranges:
            res.mono_ranges = list(res.ranges)
        for i, (rng, hits, ang) in enumerate(
                zip(res.ranges, res.radar_hits, res.angles)):
            if (rng is not None and hits == 0 and lo <= rng <= hi
                    and abs(ang) <= max_b):
                res.ranges[i] = None

    def _confirm_bins(self, fused: "FusionResult",
                      *results) -> None:
        """Mark bins holding a radar-associated detection as `confirmed`
        and tighten their min_range with the associated radar range.

        Two physical-sense tightenings (2026-08-05, both default on):
        - confirm_anchor "radar": confirmation marks the bins the matched
          returns actually OCCUPY (per point), not the detection's centre
          bin, and each marked bin's min_range tightens with the nearest
          matched return IN that bin. Extended objects (the dock spans
          three bins) keep every genuinely-backed bin; nothing is marked
          on borrowed evidence. "detection" restores the legacy
          centre-bin anchoring.
        - confirm_same_bin: additionally require the detection's own bin to
          be among the occupied bins; a FULLY cross-bin match (quad clip
          frame 88: an open-water thermal blob at +3 deg whose only matched
          returns were the dock's at +6..+9 deg) confirms nothing: the
          association survives for per-detection range, but it is not
          bin-level cross-sensor proof.
        """
        for res in results:
            if res is None or not res.radar_ranges:
                continue
            match_pts = (getattr(res, "radar_match_points", None)
                         or [None] * len(res.radar_ranges))
            for angle, rr, pts in zip(res.angles, res.radar_ranges,
                                      match_pts):
                if rr is None:
                    continue
                det_idx = angle_to_bin(angle, fused.bin_edges)
                if (self._assoc_confirm_anchor == "radar"
                        and pts is not None and len(pts)):
                    by_bin: dict[int, float] = {}
                    for b_deg, r_m in pts:
                        idx = angle_to_bin(float(b_deg), fused.bin_edges)
                        if idx is None:
                            continue
                        r_m = float(r_m)
                        if idx not in by_bin or r_m < by_bin[idx]:
                            by_bin[idx] = r_m
                    if self._assoc_confirm_same_bin and det_idx not in by_bin:
                        continue
                    for idx, rmin in by_bin.items():
                        fused.confirmed[idx] = True
                        if (fused.min_ranges[idx] is None
                                or rmin < fused.min_ranges[idx]):
                            fused.min_ranges[idx] = rmin
                else:
                    if det_idx is None:
                        continue
                    fused.confirmed[det_idx] = True
                    if (fused.min_ranges[det_idx] is None
                            or rr < fused.min_ranges[det_idx]):
                        fused.min_ranges[det_idx] = rr

    # -- combined ------------------------------------------------------------

    @staticmethod
    def _voting_angles(res) -> list[float]:
        """Angles of vote-eligible detections (all, when votes is empty)."""
        if res is None:
            return []
        if res.votes and len(res.votes) == len(res.angles):
            return [a for a, v in zip(res.angles, res.votes) if v]
        return res.angles

    def fuse(
        self,
        fisheye: FisheyeResult | None,
        thermal: ThermalResult | None,
        mmwave: MMWaveResult | None,
    ) -> FusionResult:
        f_a = self._voting_angles(fisheye)
        t_a = self._voting_angles(thermal)
        m_a = mmwave.angles if mmwave else []
        m_r = mmwave.ranges if mmwave else []
        m_pts = mmwave.points_xyz if mmwave else None
        m_vels = mmwave.velocities_xy if mmwave else None
        return fuse_angle_streams(
            f_a, t_a, m_a, m_r, self.detection["fusion"],
            mmwave_points=m_pts,
            mmwave_velocities=m_vels,
            motion_params=self.detection.get("motion"),
        )

    def process_frame(
        self,
        fisheye_frame: np.ndarray | None,
        thermal_frame: np.ndarray | None,
        mmwave_points: np.ndarray | None,
        timestamp: str | None = None,
        frame_id: str | None = None,
        sun_elevation_deg: float | None = None,
    ) -> FrameResult:
        # Sun elevation feeds the thermal profile scheduler
        # (thermal.profiles). None here means "not provided this call" and
        # leaves any previously set value in force; drivers with no
        # elevation at all simply never provide one (-> day profile,
        # logged once).
        if sun_elevation_deg is not None:
            self.set_sun_elevation(sun_elevation_deg)
        f_res = (self.process_fisheye(fisheye_frame, frame_id=frame_id)
                 if fisheye_frame is not None else None)
        t_res = self.process_thermal(thermal_frame) if thermal_frame is not None else None
        m_res = self.process_mmwave(mmwave_points, timestamp=timestamp) if mmwave_points is not None else None
        if self._assoc_enabled and m_res is not None:
            instances = (self._inst_provider.get(frame_id)
                         if self._inst_provider is not None else None)
            self._apply_association(
                f_res, m_res, np.asarray(self.intrinsics["fisheye"]["K"], float),
                self.extrinsics["T_radar_to_fisheye"], instances=instances)
            if t_res is not None and t_res.undistorted is not None:
                h_px, w_px = t_res.undistorted.shape[:2]
                P_t = pinhole_new_K(self.intrinsics["thermal"]["K"],
                                    self.intrinsics["thermal"]["D"],
                                    (w_px, h_px))
                self._apply_association(
                    t_res, m_res, P_t, self.extrinsics["T_radar_to_thermal"],
                    params=self._assoc_params_thermal)
            self._apply_radar_veto(f_res, m_res)
            self._apply_radar_veto(t_res, m_res)
        fused = self.fuse(f_res, t_res, m_res)
        if self._assoc_enabled:
            self._confirm_bins(fused, f_res, t_res)
        targets = None
        if self._target_tracker is not None and m_res is not None:
            att = self._attitude.get() if self._attitude is not None else None
            targets = self._target_tracker.update(
                m_res, timestamp,
                yaw_deg=(att.yaw_deg if att is not None else None))
            if self._identity is not None and f_res is not None:
                idf = self._identity.update(
                    f_res.coords, f_res.sizes, f_res.radar_point_indices,
                    f_res.angles, targets.targets, timestamp=timestamp)
                f_res.camera_track_ids = idf.camera_track_ids
                f_res.radar_target_ids = idf.radar_target_ids
                if (self._target_tracker.config.camera_bearing_rate
                        and idf.camera_obs):
                    # Second observation stream: paired camera bearings
                    # (offset-corrected) refresh the frame's targets --
                    # and sustain paired tracks through radar dropouts.
                    targets = self._target_tracker.apply_camera_observations(
                        idf.camera_obs, timestamp)
                self._identity.annotate(targets.targets)
        return FrameResult(fisheye=f_res, thermal=t_res, mmwave=m_res,
                           fusion=fused, timestamp=timestamp, targets=targets)


# ---------------------------------------------------------------------------
# Triplet iterator (used by viewer, sweep, fusion, metrics)
# ---------------------------------------------------------------------------

def iterate_triplet(
    triplet,
    detection_cfg: dict,
    clip_overrides: dict | None = None,
    respect_curation: bool = True,
):
    """Generator yielding (timestamp, fisheye_frame, thermal_frame, mmwave_points)
    in lockstep, keyed by mmwave RoundedTime as the source of truth.

    Per-clip overrides (rotation, resize) are looked up automatically
    from `configs/clip_overrides.yaml` unless an explicit override dict
    is passed (used by tests to avoid loading the global file).

    Frames inside a curation CUT (configs/curation.yaml, trimmed in the
    dashboard; scripts/utils/curation.py) are skipped by default — the
    video is still read in lockstep so the streams never drift, the frame
    just is not yielded. `respect_curation=False` yields every frame; use
    it when the caller needs the raw video frame index (the baker's
    `every`-th sampling, the feature exporter's exposure lookup) and
    applies `keep_frame` itself. Byte-identical when no cuts exist.

    A fisheye-only capture (`triplet.thermal` / `triplet.mmwave` are None;
    see `resolve_triplet(..., require=("fisheye",))`) iterates by fisheye frame
    on the synthesized timeline, yielding `thermal_frame=None` and an empty
    point cloud. Only fisheye-only consumers ever see that: the fusion pipeline
    resolves triplets strictly, so its frames are never None.
    """
    from scripts.utils.clip_overrides import (
        apply_overrides,
        get_for_clip,
        load_overrides,
    )
    from scripts.utils.datasets import load_mmwave_csv

    if clip_overrides is None:
        clip_overrides = get_for_clip(load_overrides(), triplet.clip_id)
    fisheye_o = clip_overrides.get("fisheye", {}) if clip_overrides else {}
    thermal_o = clip_overrides.get("thermal", {}) if clip_overrides else {}

    cuts = ()
    if respect_curation:
        from scripts.utils.curation import in_cut, load_curation
        cuts = load_curation().cuts_for(triplet.clip_id)

    def _cut(ts: str) -> bool:
        return bool(cuts) and in_cut(cuts, ts)

    # A fisheye-only capture has no radar CSV at all; treat it exactly like a
    # radar CSV that decoded nothing, which the fallback branch below handles.
    df = load_mmwave_csv(triplet.mmwave) if triplet.mmwave is not None else None
    grouped = (df.groupby("RoundedTime")
               if df is not None and not df.empty else None)
    timestamps = sorted(grouped.groups.keys()) if grouped is not None else None
    # Doppler-aware CSVs (captured since 2026-07-06) carry a V column;
    # points then yield as Nx4 (X, Y, Z, V) and process_mmwave uses the
    # Doppler per point. Older CSVs yield Nx3 exactly as before. TLV-7
    # side-info columns (SNR/NOISE dB, captured since 2026-07-09) are
    # appended after V when present, so per-point radar confidence
    # reaches process_mmwave (fusion-scorer evidence): they only ever
    # ship alongside V, keeping Doppler fixed at column 3.
    point_cols = ["X", "Y", "Z"]
    if grouped is not None and "V" in df.columns:
        point_cols.append("V")
        if "SNR" in df.columns and "NOISE" in df.columns:
            point_cols += ["SNR", "NOISE"]   # both or neither (column order contract)

    fisheye_cap = cv2.VideoCapture(str(triplet.fisheye))
    thermal_cap = (cv2.VideoCapture(str(triplet.thermal))
                   if triplet.thermal is not None else None)

    def _release():
        fisheye_cap.release()
        if thermal_cap is not None:
            thermal_cap.release()

    if not fisheye_cap.isOpened():
        _release()
        raise RuntimeError(f"Cannot open {triplet.fisheye}")
    if thermal_cap is not None and not thermal_cap.isOpened():
        _release()
        raise RuntimeError(f"Cannot open {triplet.thermal}")

    def _read_thermal():
        """(ok, frame) for the thermal stream; a fisheye-only capture has no
        thermal video, so it always 'succeeds' with a None frame rather than
        ending the iteration at frame 0."""
        if thermal_cap is None:
            return True, None
        return thermal_cap.read()

    try:
        if timestamps is not None:
            for ts in timestamps:
                ret_f, fisheye_frame = fisheye_cap.read()
                ret_t, thermal_frame = _read_thermal()
                if not ret_f or not ret_t:
                    break
                fisheye_frame = apply_overrides(fisheye_frame, fisheye_o)
                if thermal_frame is not None:
                    thermal_frame = apply_overrides(thermal_frame, thermal_o)
                # Dropping NaN-position rows strips sentinel rows (empty
                # X/Y/Z) so frames where the radar reported no objects
                # yield an empty point cloud rather than rows of NaNs
                # that downstream processing would mishandle. The
                # timestamp survives. A NaN in the V column alone keeps
                # the row: the point's position is fine, it just has no
                # Doppler (process_mmwave falls back to the tracker).
                if _cut(ts):
                    continue   # curation cut: frames read (lockstep), not yielded
                xyz_df = (grouped.get_group(ts)[point_cols]
                          .dropna(subset=["X", "Y", "Z"]))
                pts = (xyz_df.values.astype(np.float64)
                       if len(xyz_df) else np.empty((0, len(point_cols))))
                yield ts, fisheye_frame, thermal_frame, pts
        else:
            # Empty-radar fallback: iterate by fisheye frame so the
            # dashboard can still scrub fisheye+thermal. Timestamps are
            # synthesized from the clip's filename wall-clock plus a
            # 1/FPS_TARGET step (CLAUDE.md §3: capture nominally 3 fps).
            # Radar points come back empty, which process_mmwave handles.
            base = _wallclock_seconds_from_timestamp(triplet.timestamp)
            step = 1.0 / 3.0
            empty_pts = np.empty((0, 3), dtype=np.float64)
            i = 0
            while True:
                ret_f, fisheye_frame = fisheye_cap.read()
                ret_t, thermal_frame = _read_thermal()
                if not ret_f or not ret_t:
                    break
                fisheye_frame = apply_overrides(fisheye_frame, fisheye_o)
                if thermal_frame is not None:
                    thermal_frame = apply_overrides(thermal_frame, thermal_o)
                ts = _format_hhmmss(base + i * step)
                i += 1
                if _cut(ts):
                    continue
                yield ts, fisheye_frame, thermal_frame, empty_pts
    finally:
        _release()


def _wallclock_seconds_from_timestamp(triplet_ts: str) -> float:
    """Pull HH:MM:SS out of e.g. '2026-06-17_12-31-23' → 45083.0 s.

    Defaults to 0.0 if the suffix doesn't match the YYYY-MM-DD_HH-MM-SS
    convention `data_collection/continuous_capture.py` writes.
    """
    parts = triplet_ts.split("_")
    if len(parts) < 2:
        return 0.0
    hms = parts[-1].split("-")
    if len(hms) != 3:
        return 0.0
    try:
        h, m, s = (int(x) for x in hms)
    except ValueError:
        return 0.0
    return h * 3600 + m * 60 + s


def _format_hhmmss(t_sec: float) -> str:
    """Match load_mmwave_csv's 100 ms-rounded `HH:MM:SS.f` format."""
    t_sec %= 86400
    h, rem = divmod(t_sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:04.1f}"
