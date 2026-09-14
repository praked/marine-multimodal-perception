"""Per-target motion estimation (scripts/sensor_processing/target_motion.py).

Synthetic-geometry validation: the two ground-truth clips (2026-08-19 canoe
crossing, 2026-07-06 pacing) live on ROS2_SSD and are scored by
scripts.eval.target_motion_eval; these tests pin the math and the sign
conventions with trajectories whose answers are analytic.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from scripts.sensor_processing.pipeline import (
    MMWaveResult,
    ObstacleDetectionPipeline,
)
from scripts.sensor_processing.target_motion import (
    TargetMotionConfig,
    TargetMotionTracker,
    TargetState,
    cluster_points,
    target_to_dict,
    targets_by_bin,
)


# ---------------------------------------------------------------------------
# Synthetic trajectory helpers
# ---------------------------------------------------------------------------

def _ts(i: int, fps: float = 10.0) -> str:
    t = 12 * 3600 + i / fps
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:04.1f}"


def _frame(px: float, py: float, vx: float, vy: float,
           n_pts: int = 3, spread: float = 0.2) -> MMWaveResult:
    """A cluster of points around (px, py) whose velocities encode the
    Doppler radial projection of (vx, vy): exactly what
    radar_velocity_from_doppler produces from the TLV's V."""
    offs = np.linspace(-spread, spread, n_pts)
    pts = np.column_stack([px + offs, np.full(n_pts, py), np.zeros(n_pts)])
    vels = []
    for x, y, _ in pts:
        r = math.hypot(x, y)
        v_r = (x * vx + y * vy) / r          # range-rate (Doppler sign)
        vels.append((v_r * x / r, v_r * y / r))
    return MMWaveResult(points_xyz=pts,
                        angles=[math.degrees(math.atan2(x, y))
                                for x, y, _ in pts],
                        ranges=[float(y) for _, y, _ in pts],
                        velocities_xy=vels, velocity_source="doppler")


def _run(tracker: TargetMotionTracker, p0, v, n_frames: int,
         yaw_fn=None, fps: float = 10.0):
    """Propagate a constant-velocity target; returns the last result."""
    res = None
    for i in range(n_frames):
        t = i / fps
        px, py = p0[0] + v[0] * t, p0[1] + v[1] * t
        yaw = yaw_fn(t) if yaw_fn is not None else None
        res = tracker.update(_frame(px, py, v[0], v[1]), _ts(i), yaw_deg=yaw)
    return res


def _one_target(res) -> TargetState:
    assert res is not None and len(res.targets) == 1, \
        f"expected one target, got {res and [t.track_id for t in res.targets]}"
    return res.targets[0]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def test_cluster_points_groups_by_gate():
    pts = np.array([[0.0, 3.0, 0.0], [0.4, 3.2, 0.0],   # one target (chained)
                    [0.8, 3.4, 0.0],
                    [4.0, 6.0, 0.0]])                    # far singleton
    clusters = cluster_points(pts, gate_m=0.6)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 3]


def test_cluster_points_empty():
    assert cluster_points(np.empty((0, 3)), 1.0) == []


# ---------------------------------------------------------------------------
# Classification: the four synthetic type cases
# ---------------------------------------------------------------------------

def test_head_on_closing():
    # Approach at 0.5 m/s dead ahead: Doppler negative, closing positive.
    res = _run(TargetMotionTracker(), (0.0, 8.0), (0.0, -0.5), 30)
    t = _one_target(res)
    assert t.motion_state == "closing"
    assert t.closing_mps == pytest.approx(0.5, abs=0.05)
    assert t.v_radial_mps == pytest.approx(-0.5, abs=0.05)
    # CPA of a head-on track is (near) zero, soon.
    assert t.cpa_m == pytest.approx(0.0, abs=0.3)
    assert t.t_cpa_s is not None and t.t_cpa_s < 20.0


def test_receding_diverging():
    res = _run(TargetMotionTracker(), (0.0, 5.0), (0.0, 0.5), 30)
    t = _one_target(res)
    assert t.motion_state == "diverging"
    assert t.closing_mps == pytest.approx(-0.5, abs=0.05)
    # CPA in the past -> not predicted.
    assert t.cpa_m is None and t.t_cpa_s is None


def test_static_target():
    res = _run(TargetMotionTracker(), (1.0, 5.0), (0.0, 0.0), 30)
    t = _one_target(res)
    assert t.motion_state == "static"
    assert abs(t.closing_mps) < 0.05
    assert t.v_tangential_mps == pytest.approx(0.0, abs=0.05)


def test_crossing_left_to_right():
    # Canoe-type geometry: crossing the bow left->right at 1 m/s, 6 m out.
    # Doppler alone barely sees it near the crossing point; the bearing-rate
    # channel must carry the classification.
    res = _run(TargetMotionTracker(), (-3.0, 6.0), (1.0, 0.0), 40)
    t = _one_target(res)
    assert t.motion_state == "crossing"
    assert t.v_tangential_mps is not None and t.v_tangential_mps > 0.3
    # Recovered 2-D velocity points along +x.
    assert t.vx_mps == pytest.approx(1.0, abs=0.25)
    assert abs(t.vy_mps) < 0.25
    assert t.course_deg == pytest.approx(90.0, abs=15.0)


def test_crossing_cpa_matches_analytic():
    # p0 (-4, 6), v (1, 0): CPA is directly abeam of the path's closest
    # point: d_cpa = 6 m at the moment x = 0.
    res = _run(TargetMotionTracker(), (-4.0, 6.0), (1.0, 0.0), 20)
    t = _one_target(res)
    assert t.cpa_m == pytest.approx(6.0, abs=0.5)
    # ~4 s of travel consumed by the 20 simulated frames (2 s): ~2 s left.
    assert t.t_cpa_s == pytest.approx(2.0, abs=1.0)


def test_collision_course_crossing_small_cpa():
    # Constant-bearing-decreasing-range: crossing velocity aimed at the
    # boat. Doppler IS nonzero here; the CPA must read (near) zero: this is
    # the geometry the urgency motion term exists for.
    p0 = (-3.0, 6.0)
    speed = 1.0
    norm = math.hypot(*p0)
    v = (-p0[0] / norm * speed, -p0[1] / norm * speed)
    res = _run(TargetMotionTracker(), p0, v, 30)
    t = _one_target(res)
    assert t.cpa_m is not None and t.cpa_m < 0.5
    assert t.closing_mps == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# Ego-rotation correction
# ---------------------------------------------------------------------------

def _yawing_boat_frames(tracker, yaw_rate_dps: float, use_yaw: bool,
                        n_frames: int = 30, fps: float = 10.0):
    """Static world target while the boat yaws: its boat-frame bearing
    sweeps at -yaw_rate. Doppler is zero (rotation has no radial part)."""
    r0, b0 = 6.0, 20.0
    res = None
    for i in range(n_frames):
        t = i / fps
        yaw = yaw_rate_dps * t
        b = b0 - yaw                       # boat-frame bearing sweeps
        px = r0 * math.sin(math.radians(b))
        py = r0 * math.cos(math.radians(b))
        res = tracker.update(_frame(px, py, 0.0, 0.0), _ts(i),
                             yaw_deg=yaw if use_yaw else None)
    return _one_target(res)


def test_turning_boat_without_imu_fakes_crossing():
    cfg = TargetMotionConfig()
    t = _yawing_boat_frames(TargetMotionTracker(cfg), 8.0, use_yaw=False)
    assert t.motion_state == "crossing"      # the failure mode
    assert not t.ego_corrected


def test_turning_boat_with_imu_reads_static():
    # The correction mechanism, with a TRUE yaw signal. use_imu_yaw ships
    # default-off because the real RVC yaw is gimbal-aliased on this mount
    # (measured 2026-08-24); the math itself must stay correct for the day
    # a camera-frame yaw lands.
    cfg = TargetMotionConfig(use_imu_yaw=True)
    t = _yawing_boat_frames(TargetMotionTracker(cfg), 8.0, use_yaw=True)
    assert t.ego_corrected
    assert t.motion_state == "static"
    assert abs(t.v_tangential_mps) < 0.1


def test_use_imu_yaw_off_ignores_supplied_yaw():
    # Default config: yaw may be passed by the pipeline but must not
    # perturb the rates (the measured aliasing failure mode).
    t = _yawing_boat_frames(TargetMotionTracker(), 8.0, use_yaw=True)
    assert not t.ego_corrected
    assert t.motion_state == "crossing"      # same as yaw-blind


def test_yaw_availability_flip_restarts_rate_window():
    # Yaw arriving mid-track must not mix corrected/raw samples in one fit.
    tracker = TargetMotionTracker(TargetMotionConfig(use_imu_yaw=True))
    for i in range(10):
        tracker.update(_frame(0.0, 6.0, 0.0, 0.0), _ts(i), yaw_deg=None)
    res = tracker.update(_frame(0.0, 6.0, 0.0, 0.0), _ts(10), yaw_deg=15.0)
    t = _one_target(res)
    assert t.ego_corrected
    assert t.v_tangential_mps is None        # window restarted


# ---------------------------------------------------------------------------
# Tracker mechanics
# ---------------------------------------------------------------------------

def test_min_hits_gate_before_reporting():
    tracker = TargetMotionTracker()
    r1 = tracker.update(_frame(0.0, 5.0, 0.0, 0.0), _ts(0))
    r2 = tracker.update(_frame(0.0, 5.0, 0.0, 0.0), _ts(1))
    r3 = tracker.update(_frame(0.0, 5.0, 0.0, 0.0), _ts(2))
    assert r1.targets == [] and r2.targets == []
    assert len(r3.targets) == 1


def test_stream_gap_resets_tracks():
    tracker = TargetMotionTracker()
    for i in range(5):
        tracker.update(_frame(0.0, 5.0, 0.0, 0.0), _ts(i))
    # 10 s gap >> max_dt_s: differentiating across it is meaningless.
    res = tracker.update(_frame(0.0, 5.0, 0.0, 0.0), "12:10:00.0")
    assert res.targets == []                 # fresh, unconfirmed track


def test_empty_frames_age_out_tracks():
    tracker = TargetMotionTracker()
    for i in range(5):
        tracker.update(_frame(0.0, 5.0, 0.0, 0.0), _ts(i))
    empty = MMWaveResult(points_xyz=np.empty((0, 3)), angles=[], ranges=[],
                         velocities_xy=[])
    for i in range(5, 11):
        res = tracker.update(empty, _ts(i))
    assert res.targets == [] and tracker.tracks == []


def test_two_targets_tracked_independently():
    tracker = TargetMotionTracker()
    res = None
    for i in range(30):
        t = i / 10.0
        a = _frame(-3.0 + 1.0 * t, 6.0, 1.0, 0.0)       # crossing
        b = _frame(3.0, 7.0 - 0.5 * t, 0.0, -0.5)       # closing
        pts = np.vstack([a.points_xyz, b.points_xyz])
        vels = a.velocities_xy + b.velocities_xy
        m = MMWaveResult(points_xyz=pts, angles=[], ranges=[],
                         velocities_xy=vels, velocity_source="doppler")
        res = tracker.update(m, _ts(i))
    states = sorted(t.motion_state for t in res.targets)
    assert states == ["closing", "crossing"]


# ---------------------------------------------------------------------------
# Bin mapping + serialization
# ---------------------------------------------------------------------------

def _state(bearing=0.0, range_m=5.0, **kw) -> TargetState:
    x = range_m * math.sin(math.radians(bearing))
    y = range_m * math.cos(math.radians(bearing))
    return TargetState(track_id=kw.pop("track_id", 0), bearing_deg=bearing,
                       range_m=range_m, x_m=x, y_m=y, n_points=3,
                       age_frames=5, **kw)


def test_targets_by_bin_nearest_wins():
    edges = np.arange(-55, 65, 10)
    near = _state(bearing=2.0, range_m=3.0, track_id=1)
    far = _state(bearing=4.0, range_m=8.0, track_id=2)
    out = targets_by_bin([near, far], edges)
    centre = out[5]                          # the -5..+5 bin
    assert centre is not None and centre.track_id == 1
    assert sum(1 for t in out if t is not None) == 1


def test_targets_by_bin_out_of_fov_dropped():
    edges = np.arange(-55, 65, 10)
    out = targets_by_bin([_state(bearing=80.0)], edges)
    assert all(t is None for t in out)


def test_target_to_dict_json_safe():
    d = target_to_dict(_state(bearing=10.0, v_radial_mps=None,
                              motion_state="crossing"))
    assert d["motion_state"] == "crossing"
    assert d["v_radial_mps"] is None
    assert d["bearing_deg"] == pytest.approx(10.0)
    import json
    json.dumps(d)                            # must serialize


# ---------------------------------------------------------------------------
# Pipeline integration + config gating
# ---------------------------------------------------------------------------

def test_pipeline_default_off_targets_none(intrinsics_real, detection_real):
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    assert pl._target_tracker is None
    pts = np.array([[0.5, 3.0, 0.0]])
    res = pl.process_frame(None, None, pts, timestamp="12:00:00.0")
    assert res.targets is None


def test_pipeline_disabled_block_matches_absent(intrinsics_real,
                                                detection_real):
    # Presence of the (disabled) motion.targets block must not perturb the
    # fusion output: byte-identical contract.
    det_off = copy.deepcopy(detection_real)
    det_off["motion"].pop("targets", None)
    pts = np.array([[0.5, 3.0, 0.0, -0.3]])
    outs = []
    for det in (detection_real, det_off):
        pl = ObstacleDetectionPipeline(intrinsics_real, copy.deepcopy(det))
        r = pl.process_frame(None, None, pts, timestamp="12:00:00.0")
        outs.append((r.fusion.scores, r.fusion.min_ranges,
                     r.fusion.per_bin_velocity_mps, r.targets))
    assert outs[0] == outs[1]


def test_pipeline_enabled_emits_targets(intrinsics_real, detection_real):
    det = copy.deepcopy(detection_real)
    det["motion"]["targets"]["enabled"] = True
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    assert pl._target_tracker is not None
    res = None
    for i in range(5):
        pts = np.array([[0.5, 3.0, 0.0, -0.4]])
        res = pl.process_frame(None, None, pts, timestamp=_ts(i))
    assert res.targets is not None
    assert len(res.targets.targets) == 1
    assert res.targets.targets[0].motion_state in ("closing", "static")


# ---------------------------------------------------------------------------
# Feature columns (fusion-scorer schema)
# ---------------------------------------------------------------------------

def test_target_bin_features_defaults_when_off():
    from scripts.fusion_model.build_features import target_bin_features
    edges = np.arange(-55, 65, 10)
    out = target_bin_features(None, edges)
    assert out["target_present"] == [False] * 11
    assert all(math.isnan(v) for v in out["target_cpa_m"])
    assert out["target_motion_state"] == [""] * 11


def test_target_bin_features_filled():
    from scripts.fusion_model.build_features import target_bin_features
    from scripts.sensor_processing.target_motion import TargetMotionResult
    edges = np.arange(-55, 65, 10)
    t = _state(bearing=12.0, range_m=4.0, motion_state="crossing",
               v_tangential_mps=0.8, closing_mps=0.1, speed_mps=0.81,
               cpa_m=1.2, t_cpa_s=5.0)
    out = target_bin_features(TargetMotionResult(targets=[t]), edges)
    idx = 6                                  # +5..+15 bin
    assert out["target_present"][idx] is True
    assert out["target_motion_state"][idx] == "crossing"
    assert out["target_cpa_m"][idx] == pytest.approx(1.2)
    assert out["target_present"].count(True) == 1


# ---------------------------------------------------------------------------
# Urgency motion term (config-gated, default off)
# ---------------------------------------------------------------------------

def _urgency_cfg(enabled: bool) -> dict:
    return {"urgency": {"range_cap_m": 15.0, "ttc_cap_s": 60.0,
                        "unknown_floor": 0.3,
                        "motion": {"enabled": enabled, "cpa_cap_m": 5.0,
                                   "t_cpa_cap_s": 60.0}}}


def test_urgency_motion_term_off_is_unchanged():
    from scripts.fusion_model.models import urgency
    base = urgency(10.0, None, cfg=_urgency_cfg(False))
    with_cpa = urgency(10.0, None, cfg=_urgency_cfg(False),
                       cpa_m=0.5, t_cpa_s=5.0)
    assert with_cpa == base


def test_urgency_motion_term_raises_for_close_soon_cpa():
    from scripts.fusion_model.models import urgency
    cfg = _urgency_cfg(True)
    base = urgency(10.0, None, cfg=cfg)
    # Crossing target predicted to pass at 0.5 m in 6 s: nearly max term.
    raised = urgency(10.0, None, cfg=cfg, cpa_m=0.5, t_cpa_s=6.0)
    assert raised > base
    # A far CPA contributes nothing (max() semantics: never lowers).
    assert urgency(10.0, None, cfg=cfg, cpa_m=20.0, t_cpa_s=6.0) == base


def test_threat_from_score_cpa_arrays():
    from scripts.fusion_model.models import threat_from_score
    cfg = _urgency_cfg(True)
    p = np.array([0.8, 0.8])
    r = np.array([10.0, 10.0])
    t = np.array([np.nan, np.nan])
    plain = threat_from_score(p, r, t, cfg=cfg)
    cpa = np.array([0.5, np.nan])
    tcpa = np.array([6.0, np.nan])
    with_m = threat_from_score(p, r, t, cfg=cfg, cpa_m=cpa, t_cpa_s=tcpa)
    assert with_m[0] > plain[0]              # crossing bin raised
    assert with_m[1] == plain[1]             # NaN bin untouched


def test_resolve_triplet_unreadable_thermal_treated_absent(tmp_path):
    """A corrupt thermal mp4 (sensor-death chunk, no moov) resolves to None
    when thermal is not required — the require mechanism alone only covers
    MISSING files (bit us twice on the 2026-08-19 session)."""
    from scripts.utils.datasets import resolve_triplet

    d = tmp_path / "scene"
    d.mkdir()
    (d / "fisheye_2026-01-01_00-00-00.mp4").write_bytes(b"\x00" * 64)
    (d / "thermal_2026-01-01_00-00-00.mp4").write_bytes(b"junk")
    (d / "mmwave_2026-01-01_00-00-00.csv").write_text("Date,Time,X,Y,Z\n")
    t = resolve_triplet(str(d / "2026-01-01_00-00-00"),
                        require=("mmwave",))
    assert t.thermal is None  # unreadable -> honestly absent
    assert t.fisheye is None  # fisheye junk video also unreadable
