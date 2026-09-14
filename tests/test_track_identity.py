"""Cross-sensor track identity (scripts/sensor_processing/track_identity.py)
and the camera-bearing-rate second observation (target_motion.py).

Synthetic-analytic, like test_target_motion.py: a crossing object seen by
both sensors must pair and stay paired; two objects must not swap ids;
camera-only and radar-only frames must degrade gracefully; a paired
camera track must carry the target's bearing-rate (and the target itself)
through a radar dropout -- the documented 4 s mid-transit dropout on the
2026-08-19 crossing GT is the value proposition. The real-clip numbers
live in scripts.eval.track_identity_eval.
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
    target_to_dict,
)
from scripts.sensor_processing.track_identity import (
    IdentityConfig,
    TrackIdentityManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(i: int, fps: float = 10.0) -> str:
    t = 12 * 3600 + i / fps
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:04.1f}"


def _target(track_id: int, bearing: float, range_m: float,
            point_indices: list[int]) -> TargetState:
    x = range_m * math.sin(math.radians(bearing))
    y = range_m * math.cos(math.radians(bearing))
    return TargetState(track_id=track_id, bearing_deg=bearing,
                       range_m=range_m, x_m=x, y_m=y,
                       n_points=len(point_indices), age_frames=5,
                       point_indices=point_indices)


def _det(bearing: float, size: float = 20.0,
         cx: float = 432.0, px_per_deg: float = 7.0):
    """A blob detection at the given bearing (pixel-space synthetic)."""
    return (int(round(cx + bearing * px_per_deg)), 300), size


def _step(mgr, dets, timestamp=None):
    """dets: list of (bearing, point_indices) per detection; targets:
    list of TargetState. Returns IdentityFrame."""
    det_list, targets = dets
    coords, sizes, det_pts, bearings = [], [], [], []
    for bearing, pts in det_list:
        c, s = _det(bearing)
        coords.append(c)
        sizes.append(s)
        det_pts.append(list(pts))
        bearings.append(bearing)
    return mgr.update(coords, sizes, det_pts, bearings, targets,
                      timestamp=timestamp)


# ---------------------------------------------------------------------------
# Identity manager: pairing hysteresis
# ---------------------------------------------------------------------------

def test_pair_forms_after_min_hits_and_stays():
    mgr = TrackIdentityManager()
    idf = None
    for i in range(10):
        dets = [( -10.0 + i, [1, 2] )]
        tgts = [_target(7, -10.0 + i, 6.0, [0, 1, 2])]
        idf = _step(mgr, (dets, tgts), _ts(i))
        if i < 2:
            assert idf.pairs == {}           # 3-of-5 not yet met
    assert idf.pairs == {7: 0}               # first camera track id is 0
    assert idf.radar_target_ids == [7]
    assert idf.camera_track_ids == [0]
    assert mgr.pairs_formed == 1 and mgr.pairs_broken == 0
    assert mgr.id_switches == 0
    # The paired camera observation is emitted for the sustain feed.
    assert 7 in idf.camera_obs


def test_two_objects_do_not_swap_ids():
    mgr = TrackIdentityManager()
    idf = None
    for i in range(20):
        dets = [(-20.0 + 0.5 * i, [1, 2]), (25.0, [10, 11])]
        tgts = [_target(3, -20.0 + 0.5 * i, 6.0, [0, 1, 2]),
                _target(4, 25.0, 7.0, [10, 11, 12])]
        idf = _step(mgr, (dets, tgts), _ts(i))
    assert idf.pairs == {3: 0, 4: 1}
    assert mgr.id_switches == 0
    assert mgr.pairs_formed == 2


def test_camera_only_frames_degrade_gracefully():
    mgr = TrackIdentityManager()
    for i in range(8):
        idf = _step(mgr, ([(5.0, [])], []), _ts(i))   # no radar targets
    assert idf.pairs == {}
    assert idf.camera_track_ids == [0]                # camera still tracks
    assert idf.radar_target_ids == [None]
    assert mgr.pairs_formed == 0


def test_radar_only_frames_degrade_gracefully():
    mgr = TrackIdentityManager()
    tgts = [_target(1, 0.0, 5.0, [0, 1])]
    for i in range(8):
        idf = _step(mgr, ([], tgts), _ts(i))          # no detections
    assert idf.pairs == {} and idf.camera_track_ids == []
    mgr.annotate(tgts)
    assert tgts[0].camera_track_id is None


def test_association_off_never_pairs():
    # No point indices (fusion.association disabled): agreement evidence
    # cannot accumulate even with perfectly co-located tracks.
    mgr = TrackIdentityManager()
    for i in range(10):
        idf = _step(mgr, ([(0.0, [])],
                          [_target(1, 0.0, 5.0, [0, 1, 2])]), _ts(i))
    assert idf.pairs == {}


def test_pair_survives_radar_dropout():
    mgr = TrackIdentityManager()
    tgt = _target(2, 0.0, 6.0, [0, 1])
    for i in range(5):
        _step(mgr, ([(0.0, [0, 1])], [tgt]), _ts(i))
    assert 2 in mgr.pairs
    # Radar drops the target for 8 frames; camera keeps seeing it.
    for i in range(5, 13):
        idf = _step(mgr, ([(0.0, [])], []), _ts(i))
    assert idf.pairs == {2: 0}               # held, not divorced
    assert 2 in idf.camera_obs               # sustain feed still emitted
    # Radar returns: same pair, no switch.
    idf = _step(mgr, ([(0.0, [0, 1])], [tgt]), _ts(13))
    assert idf.pairs == {2: 0} and mgr.id_switches == 0
    assert mgr.pairs_broken == 0


def test_pair_breaks_on_sustained_disagreement():
    cfg = IdentityConfig(break_misses=5)
    mgr = TrackIdentityManager(cfg)
    for i in range(5):
        _step(mgr, ([(0.0, [0, 1])], [_target(2, 0.0, 6.0, [0, 1])]),
              _ts(i))
    assert 2 in mgr.pairs
    # Both sides observed, zero overlap, 5 consecutive frames.
    for i in range(5, 10):
        idf = _step(mgr, ([(0.0, [7, 8])],
                          [_target(2, 0.0, 6.0, [0, 1])]), _ts(i))
    assert idf.pairs == {}
    assert mgr.pairs_broken == 1


def test_pair_dies_when_radar_target_stale():
    cfg = IdentityConfig(stale_after=6)
    mgr = TrackIdentityManager(cfg)
    for i in range(5):
        _step(mgr, ([(0.0, [0, 1])], [_target(2, 0.0, 6.0, [0, 1])]),
              _ts(i))
    assert 2 in mgr.pairs
    for i in range(5, 13):
        idf = _step(mgr, ([(0.0, [])], []), _ts(i))
    assert idf.pairs == {}
    assert mgr.pairs_broken == 1


def test_bearing_offset_learned_and_applied():
    # Camera reads 3 deg low relative to the radar target: camera_obs must
    # converge to the target's frame (offset-corrected).
    mgr = TrackIdentityManager()
    idf = None
    for i in range(15):
        idf = _step(mgr, ([(7.0, [0, 1])],
                          [_target(9, 10.0, 6.0, [0, 1, 2])]), _ts(i))
    assert idf.pairs == {9: 0}
    assert idf.camera_obs[9] == pytest.approx(10.0, abs=0.3)


def test_annotate_stamps_camera_track_id():
    mgr = TrackIdentityManager()
    tgt = _target(5, 0.0, 6.0, [0, 1])
    for i in range(5):
        _step(mgr, ([(0.0, [0, 1])], [tgt]), _ts(i))
    other = _target(6, 30.0, 8.0, [9])
    mgr.annotate([tgt, other])
    assert tgt.camera_track_id == 0
    assert other.camera_track_id is None


# ---------------------------------------------------------------------------
# Camera bearing-rate: second observation stream in TargetMotionTracker
# ---------------------------------------------------------------------------

def _crossing_frame(t: float, r: float = 6.0, v: float = 1.0,
                    x0: float = -3.0, n_pts: int = 3,
                    spread: float = 0.2) -> MMWaveResult:
    """Constant-velocity crossing target, Doppler-exact (test_target_motion
    _frame geometry)."""
    px, py = x0 + v * t, r
    offs = np.linspace(-spread, spread, n_pts)
    pts = np.column_stack([px + offs, np.full(n_pts, py), np.zeros(n_pts)])
    vels = []
    for x, y, _ in pts:
        rr = math.hypot(x, y)
        v_r = (x * v) / rr
        vels.append((v_r * x / rr, v_r * y / rr))
    return MMWaveResult(points_xyz=pts,
                        angles=[math.degrees(math.atan2(x, y))
                                for x, y, _ in pts],
                        ranges=[float(y) for _, y, _ in pts],
                        velocities_xy=vels, velocity_source="doppler")


_EMPTY = MMWaveResult(points_xyz=np.empty((0, 3)), angles=[], ranges=[],
                      velocities_xy=[])


def _cam_bearing(t: float, r: float = 6.0, v: float = 1.0,
                 x0: float = -3.0) -> float:
    return math.degrees(math.atan2(x0 + v * t, r))


def test_camera_rate_off_is_byte_identical():
    # Same frame sequence, apply_camera_observations never called with the
    # flag off (the pipeline contract): states must match a pre-identity
    # tracker exactly.
    a = TargetMotionTracker(TargetMotionConfig())
    b = TargetMotionTracker(TargetMotionConfig(camera_bearing_rate=True))
    ra = rb = None
    for i in range(30):
        t = i / 10.0
        ra = a.update(_crossing_frame(t), _ts(i))
        rb = b.update(_crossing_frame(t), _ts(i))
    ta, tb = ra.targets[0], rb.targets[0]
    assert ta.motion_state == tb.motion_state == "crossing"
    assert ta.v_tangential_mps == pytest.approx(tb.v_tangential_mps)
    assert ta.rate_source == "radar" and tb.rate_source == "radar"


def test_camera_sustains_target_through_radar_dropout():
    # Radar sees the crossing subject for 2 s, then drops it for 1.5 s
    # (>> max_age = 5 frames) while the camera keeps observing. The paired
    # track must keep reporting "crossing" on camera bearing-rate, with
    # honest provenance (n_points 0, empty point_indices), and re-attach
    # to the SAME id when the radar returns.
    cfg = TargetMotionConfig(camera_bearing_rate=True)
    tracker = TargetMotionTracker(cfg)
    res = None
    for i in range(20):                       # 0..1.9 s: radar + camera
        t = i / 10.0
        tracker.update(_crossing_frame(t), _ts(i))
        res = tracker.apply_camera_observations(
            {0: _cam_bearing(t)}, _ts(i))
    assert res.targets and res.targets[0].track_id == 0
    for i in range(20, 35):                   # 2.0..3.4 s: camera only
        t = i / 10.0
        r_up = tracker.update(_EMPTY, _ts(i))
        assert r_up.targets == []             # radar alone reports nothing
        res = tracker.apply_camera_observations(
            {0: _cam_bearing(t)}, _ts(i))
        assert len(res.targets) == 1          # camera sustains it
    t_sus = res.targets[0]
    assert t_sus.track_id == 0
    assert t_sus.n_points == 0 and t_sus.point_indices == []
    assert t_sus.rate_source == "camera"
    assert t_sus.motion_state == "crossing"
    assert t_sus.v_tangential_mps == pytest.approx(1.0, abs=0.3)
    # Radar returns near the camera-propagated position: same track id.
    t = 3.5
    r_back = tracker.update(_crossing_frame(t), _ts(35))
    assert [tr.track_id for tr in r_back.targets] == [0]


def test_sustain_expires_at_camera_sustain_max_s():
    cfg = TargetMotionConfig(camera_bearing_rate=True,
                             camera_sustain_max_s=1.0)
    tracker = TargetMotionTracker(cfg)
    for i in range(20):
        t = i / 10.0
        tracker.update(_crossing_frame(t), _ts(i))
        tracker.apply_camera_observations({0: _cam_bearing(t)}, _ts(i))
    res = None
    for i in range(20, 35):
        t = i / 10.0
        tracker.update(_EMPTY, _ts(i))
        res = tracker.apply_camera_observations({0: _cam_bearing(t)}, _ts(i))
    # 1.5 s of dropout > 1.0 s cap: the track must be gone.
    assert res.targets == [] and tracker.tracks == []


def test_camera_obs_for_unknown_track_is_ignored():
    cfg = TargetMotionConfig(camera_bearing_rate=True)
    tracker = TargetMotionTracker(cfg)
    for i in range(5):
        tracker.update(_crossing_frame(i / 10.0), _ts(i))
    res = tracker.apply_camera_observations({99: 10.0}, _ts(4))
    assert [t.track_id for t in res.targets] == [0]


def test_radar_rate_preferred_while_fresh():
    # With both streams live the radar fit (calibrated bearing source)
    # must win; the camera stream only enriches the window.
    cfg = TargetMotionConfig(camera_bearing_rate=True)
    tracker = TargetMotionTracker(cfg)
    res = None
    for i in range(30):
        t = i / 10.0
        tracker.update(_crossing_frame(t), _ts(i))
        # Deliberately biased camera bearings: must not perturb the rate.
        res = tracker.apply_camera_observations(
            {0: _cam_bearing(t) + 3.0}, _ts(i))
    tgt = res.targets[0]
    assert tgt.rate_source == "radar"
    assert tgt.v_tangential_mps == pytest.approx(1.0, abs=0.25)


# ---------------------------------------------------------------------------
# Serialization (protocol v1.3: additive camera_track_id)
# ---------------------------------------------------------------------------

def test_target_to_dict_identity_key_gated():
    t = _target(1, 5.0, 6.0, [0, 1])
    t.camera_track_id = 4
    d_legacy = target_to_dict(t)
    assert "camera_track_id" not in d_legacy      # pre-identity streams
    d_id = target_to_dict(t, include_identity=True)
    assert d_id["camera_track_id"] == 4
    import json
    json.dumps(d_id)


# ---------------------------------------------------------------------------
# Pipeline integration + config gating
# ---------------------------------------------------------------------------

def test_pipeline_identity_default_off(intrinsics_real, detection_real):
    det = copy.deepcopy(detection_real)
    det["motion"]["targets"]["enabled"] = True
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    assert pl._identity is None
    pts = np.array([[0.5, 3.0, 0.0, -0.4]])
    res = None
    for i in range(5):
        res = pl.process_frame(None, None, pts, timestamp=_ts(i))
    assert res.targets is not None
    assert all(t.camera_track_id is None for t in res.targets.targets)


def test_pipeline_identity_off_matches_absent(intrinsics_real,
                                              detection_real):
    # Presence of the (disabled) identity block must not perturb targets:
    # byte-identical contract.
    det_a = copy.deepcopy(detection_real)
    det_a["motion"]["targets"]["enabled"] = True
    det_b = copy.deepcopy(det_a)
    det_b["motion"]["targets"].pop("identity", None)
    det_b["motion"]["targets"].pop("camera_bearing_rate", None)
    pts = np.array([[0.5, 3.0, 0.0, -0.4]])
    outs = []
    for det in (det_a, det_b):
        pl = ObstacleDetectionPipeline(intrinsics_real, copy.deepcopy(det))
        r = None
        for i in range(5):
            r = pl.process_frame(None, None, pts, timestamp=_ts(i))
        outs.append([target_to_dict(t) for t in r.targets.targets])
    assert outs[0] == outs[1]


def test_pipeline_identity_enabled_requires_tracker(intrinsics_real,
                                                    detection_real):
    # identity.enabled without motion.targets.enabled stays off (the
    # layer joins tracks that would not exist).
    det = copy.deepcopy(detection_real)
    det["motion"]["targets"]["identity"] = {"enabled": True}
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    assert pl._identity is None


def test_pipeline_identity_enabled_radar_only_frames(intrinsics_real,
                                                     detection_real):
    # Radar-only process_frame calls (target_motion_eval pattern) must not
    # crash the identity layer (no fisheye result to pair against).
    det = copy.deepcopy(detection_real)
    det["motion"]["targets"]["enabled"] = True
    det["motion"]["targets"]["identity"] = {"enabled": True}
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    assert pl._identity is not None
    pts = np.array([[0.5, 3.0, 0.0, -0.4]])
    res = None
    for i in range(5):
        res = pl.process_frame(None, None, pts, timestamp=_ts(i))
    assert res.targets is not None and len(res.targets.targets) == 1
    assert res.targets.targets[0].camera_track_id is None
