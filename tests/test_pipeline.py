import numpy as np
import pytest

from scripts.sensor_processing.pipeline import (
    FrameResult,
    ObstacleDetectionPipeline,
    angle_to_bin,
    fuse_angle_streams,
    gradient_obstacle_detection,
    iterate_triplet,
    make_bins,
    mmwave_angles_ranges,
    thermal_obstacle_detection,
)


# ---------------------------------------------------------------------------
# Bin helpers
# ---------------------------------------------------------------------------

def test_make_bins_count():
    edges, centers = make_bins({"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10})
    assert len(centers) == 11
    assert centers[0] == -50
    assert centers[-1] == 50


def test_angle_to_bin_within_range():
    edges, _ = make_bins({"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10})
    assert angle_to_bin(0.0, edges) == 5


def test_angle_to_bin_out_of_range():
    edges, _ = make_bins({"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10})
    assert angle_to_bin(-60, edges) is None
    assert angle_to_bin(60, edges) is None


def test_angle_to_bin_edge_inclusive_lower():
    edges, _ = make_bins({"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10})
    assert angle_to_bin(-55, edges) == 0
    assert angle_to_bin(-45, edges) == 1


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------

def test_gradient_obstacle_detection_finds_bright_blob(small_fisheye_frame):
    mask = gradient_obstacle_detection(small_fisheye_frame,
                                       gaussian_window=3,
                                       gradient_threshold=10)
    assert mask.shape == small_fisheye_frame.shape[:2]
    assert mask.sum() > 0


def test_thermal_obstacle_detection_returns_mask_and_average(small_thermal_frame):
    full_mask = np.full(small_thermal_frame.shape[:2], 255, dtype=np.uint8)
    mask, new_avg = thermal_obstacle_detection(small_thermal_frame, 50.0,
                                               full_mask, object_thresh=50,
                                               contrast_guard=10)
    assert mask.shape == small_thermal_frame.shape[:2]
    assert mask.sum() > 0
    assert 0 < new_avg < 255


def test_thermal_obstacle_detection_low_contrast_path():
    """If max(sub) <= contrast_guard, take the no-normalize branch."""
    img = np.full((20, 20, 3), 60, dtype=np.uint8)
    mask = np.full((20, 20), 255, dtype=np.uint8)
    out, _ = thermal_obstacle_detection(img, running_average=55.0,
                                        mask=mask, object_thresh=20,
                                        contrast_guard=100)
    assert out.shape == (20, 20)


def test_mmwave_angles_ranges_basic():
    pts = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    angles, ranges = mmwave_angles_ranges(pts)
    assert angles[0] == pytest.approx(0)
    assert angles[1] == pytest.approx(45)
    assert angles[2] == pytest.approx(-45)
    assert ranges == [1.0, 1.0, 1.0]


def test_mmwave_angles_ranges_skips_zero_y():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    angles, _ = mmwave_angles_ranges(pts)
    assert len(angles) == 1


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def test_fuse_angle_streams_all_three_in_same_bin():
    f_params = {"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10, "num_sensors": 3}
    res = fuse_angle_streams([0.0], [0.0], [0.0], [3.0], f_params)
    assert max(res.scores) == 1.0
    # Min range recorded for that bin.
    centre_idx = 5
    assert res.min_ranges[centre_idx] == 3.0


def test_fuse_angle_streams_single_sensor():
    f_params = {"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10, "num_sensors": 3}
    res = fuse_angle_streams([0.0], [], [], [], f_params)
    assert max(res.scores) == pytest.approx(1.0 / 3)


def test_fuse_angle_streams_no_hits():
    f_params = {"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10, "num_sensors": 3}
    res = fuse_angle_streams([], [], [], [], f_params)
    assert all(s == 0.0 for s in res.scores)
    assert all(r is None for r in res.min_ranges)


def test_fuse_angle_streams_min_range_keeps_smallest():
    f_params = {"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10, "num_sensors": 3}
    res = fuse_angle_streams([], [], [0.0, 0.0, 0.0], [5.0, 2.0, 4.0], f_params)
    centre_idx = 5
    assert res.min_ranges[centre_idx] == 2.0


# ---------------------------------------------------------------------------
# Pipeline (stateful)
# ---------------------------------------------------------------------------

def test_pipeline_process_fisheye(intrinsics_real, detection_real, small_fisheye_frame):
    import cv2
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    img = cv2.resize(small_fisheye_frame, (864, 648))
    out = pl.process_fisheye(img)
    assert out.undistorted.shape == img.shape
    assert isinstance(out.angles, list)
    # Monocular range is computed per detection, aligned with coords/angles.
    assert isinstance(out.ranges, list)
    assert len(out.ranges) == len(out.coords)
    assert all(r is None or r > 0 for r in out.ranges)


def test_pipeline_process_thermal_has_ranges(intrinsics_real, detection_real,
                                             small_thermal_frame):
    import cv2
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    img = cv2.resize(small_thermal_frame, (160, 120))
    out = pl.process_thermal(img)
    assert len(out.ranges) == len(out.coords)
    assert all(r is None or r > 0 for r in out.ranges)


def test_range_up_vector_rejects_implausible_horizon(intrinsics_real, detection_real):
    """A confident but absurdly-tilted horizon (RANSAC locked on a shore edge)
    must be rejected in favour of level, so far objects aren't faked as near."""
    import numpy as np
    from scripts.utils.geometry import UP_LEVEL, up_from_horizon_line
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    K = intrinsics_real["fisheye"]["K"]
    cy = float(K[1, 2])
    # Plausible horizon near the principal point -> used.
    plausible = (0.0, cy + 10.0, 0.95)
    up_ok = pl._range_up_vector(K, plausible)
    np.testing.assert_allclose(up_ok, up_from_horizon_line(*plausible[:2], K))
    # Horizon row way below the principal point (huge intercept) -> level.
    implausible = (0.0, cy + 400.0, 0.95)
    np.testing.assert_allclose(pl._range_up_vector(K, implausible), UP_LEVEL)


def test_pipeline_process_thermal_updates_average(intrinsics_real, detection_real,
                                                   small_thermal_frame):
    import cv2
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    img = cv2.resize(small_thermal_frame, (160, 120))
    initial = pl.thermal_average
    out = pl.process_thermal(img)
    assert out.undistorted.shape == img.shape
    # Running average should have moved.
    assert pl.thermal_average != initial


def test_pipeline_process_mmwave_filters_y(intrinsics_real, detection_real):
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    pts = np.array([
        [0.0, 0.05, 0.0],   # below y_min
        [0.0, 10.0, 0.0],   # above y_max
        [0.0, 2.0, 0.0],    # keeps
    ])
    out = pl.process_mmwave(pts)
    assert len(out.points_xyz) == 1


def test_pipeline_process_mmwave_rejects_too_many():
    intr = {"fisheye": {"K": np.eye(3), "D": np.zeros((4,1)), "cx": 0, "pix_deg_ratio": 1.0,
                       "image_size": (10,10), "model": "fisheye"},
            "thermal": {"K": np.eye(3), "D": np.zeros(5), "cx": 0, "pix_deg_ratio": 1.0,
                       "image_size": (10,10), "model": "pinhole"}}
    det = {
        "horizon": {}, "fisheye": {"blob": {"minArea": 5}}, "thermal": {"initial_average": 100, "blob": {"minArea": 1}},
        "mmwave": {"y_min": 0.0, "y_max": 10.0, "max_objects": 5, "min_objects": 1},
        "fusion": {"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10},
    }
    pl = ObstacleDetectionPipeline(intr, det)
    pts = np.tile(np.array([[0.0, 1.0, 0.0]]), (10, 1))  # 10 > max_objects
    out = pl.process_mmwave(pts)
    assert len(out.points_xyz) == 0


def test_pipeline_process_mmwave_doppler_column(intrinsics_real, detection_real):
    """Nx4 input (X, Y, Z, Doppler) uses radar_velocity_from_doppler:
    a point dead ahead closing at 1 m/s (V = -1, away-positive TI
    convention) yields velocity (0, -1) and velocity_source 'doppler'.

    mount_yaw_deg is pinned to 0: this covers the Doppler decomposition
    mechanics on synthetic points, and the per-box mount rotation would
    otherwise rotate the velocity vector along with the cloud."""
    detection_real = {**detection_real,
                      "mmwave": {**detection_real["mmwave"], "mount_yaw_deg": 0.0}}
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    pts = np.array([[0.0, 2.0, 0.0, -1.0]])
    out = pl.process_mmwave(pts, timestamp="00:00:00.0")
    assert out.velocity_source == "doppler"
    assert out.points_xyz.shape == (1, 3)   # positions stay Nx3 downstream
    vx, vy = out.velocities_xy[0]
    assert vx == pytest.approx(0.0)
    assert vy == pytest.approx(-1.0)


def test_pipeline_process_mmwave_doppler_nan_falls_back_to_tracker(
        intrinsics_real, detection_real):
    """A NaN Doppler entry falls back to the cross-frame tracker for that
    point; provenance reports 'mixed' when both sources contributed."""
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    # Prime the tracker with a prior frame (Nx3, no Doppler).
    pl.process_mmwave(np.array([[0.0, 2.0, 0.0]]), timestamp="00:00:00.0")
    # Two points: one with Doppler, one NaN. The NaN one sits within the
    # tracker gate of the prior point, so the tracker supplies it.
    pts = np.array([
        [0.5, 3.0, 0.0, -0.7],       # doppler
        [0.0, 1.9, 0.0, np.nan],     # tracker fallback (prior at y=2.0)
    ])
    out = pl.process_mmwave(pts, timestamp="00:00:00.1")
    assert out.velocity_source == "mixed"
    assert out.velocities_xy[0] is not None
    assert out.velocities_xy[1] is not None
    # Tracker-derived: moved -0.1 m in y over 0.1 s -> vy ~ -1 m/s.
    assert out.velocities_xy[1][1] == pytest.approx(-1.0, abs=0.05)


def test_pipeline_process_mmwave_nx3_stays_tracker(intrinsics_real, detection_real):
    """Nx3 input (pre-Doppler CSVs) keeps the tracker-only path and
    provenance, byte-identical to the old behaviour."""
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    out = pl.process_mmwave(np.array([[0.0, 2.0, 0.0]]), timestamp="00:00:00.0")
    assert out.velocity_source == "tracker"
    assert out.velocities_xy == [None]


def test_iterate_triplet_yields_doppler_column(tmp_path, detection_real,
                                               synthetic_triplet):
    """A CSV with a V column yields Nx4 points; sentinel rows still drop and
    a NaN V keeps its (positionally valid) row."""
    from scripts.utils.datasets import Triplet
    csv = tmp_path / "mmwave_v.csv"
    csv.write_text(
        "Date,Time,X,Y,Z,V\n"
        "2099-01-01,00:00:00.0,0.1,1.5,0.0,-0.8\n"
        "2099-01-01,00:00:00.0,0.5,2.0,0.0,\n"    # point without doppler
        "2099-01-01,00:00:01.0,,,,\n"             # sentinel heartbeat
        "2099-01-01,00:00:02.0,0.2,2.5,0.0,0.3\n"
    )
    triplet = Triplet(
        scene=synthetic_triplet.scene,
        timestamp=synthetic_triplet.timestamp,
        fisheye=synthetic_triplet.fisheye,
        thermal=synthetic_triplet.thermal,
        mmwave=csv,
    )
    out = list(iterate_triplet(triplet, detection_real))
    assert len(out) == 3
    # Frame 1: two points, Nx4, second one's V is NaN.
    pts0 = out[0][3]
    assert pts0.shape == (2, 4)
    assert pts0[0, 3] == pytest.approx(-0.8)
    assert np.isnan(pts0[1, 3])
    # Frame 2: sentinel -> empty cloud (width follows the V-aware schema).
    assert out[1][3].shape[0] == 0
    # Frame 3: single doppler point.
    assert out[2][3].shape == (1, 4)
    assert out[2][3][0, 3] == pytest.approx(0.3)


def test_pipeline_iterate_triplet_runs(intrinsics_real, detection_real, synthetic_triplet):
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    n = 0
    for ts, fish, therm, mm_pts in iterate_triplet(synthetic_triplet, detection_real):
        res = pl.process_frame(fish, therm, mm_pts, timestamp=ts)
        assert isinstance(res, FrameResult)
        n += 1
    assert n == 5


def test_iterate_triplet_raises_on_bad_video(tmp_path, detection_real):
    from scripts.utils.datasets import Triplet
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"not a video")
    csv = tmp_path / "x.csv"
    csv.write_text("Date,Time,X,Y,Z\n2099-01-01,00:00:00.0,0,1,0\n")
    triplet = Triplet(scene="X", timestamp="t", fisheye=bad, thermal=bad, mmwave=csv)
    gen = iterate_triplet(triplet, detection_real)
    with pytest.raises(RuntimeError):
        next(gen)


# ---------------------------------------------------------------------------
# fuse_angle_streams: mmWave range angle outside bin range (line 224)
# ---------------------------------------------------------------------------

def test_fuse_angle_streams_mmwave_range_out_of_bin_skipped():
    """An mmWave point whose azimuth is outside the bin span must not
    record a min_range anywhere (covers the `continue` in the range loop)."""
    f_params = {"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10, "num_sensors": 3}
    # 89° is far outside the [-55, 55) bin span.
    res = fuse_angle_streams([], [], [89.0], [4.0], f_params)
    assert all(r is None for r in res.min_ranges)


# ---------------------------------------------------------------------------
# Monocular range branches
# ---------------------------------------------------------------------------

class _FakeAttitude:
    def up_vector_camera(self):
        import numpy as np
        return np.array([0.0, -1.0, 0.0])


class _FakeAttitudeProvider:
    """Mimics a BNO085 reader: .get() -> Attitude | None."""
    def get(self):
        return _FakeAttitude()


def test_range_up_vector_uses_attitude_provider(intrinsics_real, detection_real):
    """When an attitude provider yields a non-None Attitude, the up vector
    comes from the IMU (line 315)."""
    import numpy as np
    pl = ObstacleDetectionPipeline(
        intrinsics_real, detection_real,
        attitude_provider=_FakeAttitudeProvider(),
    )
    P = np.array(intrinsics_real["fisheye"]["K"], dtype=np.float64)
    up = pl._range_up_vector(P, (0.0, 100.0, 0.9))
    assert np.allclose(up, np.array([0.0, -1.0, 0.0]))


def test_range_up_vector_uses_detected_horizon(intrinsics_real, detection_real):
    """No IMU + a confident detected horizon -> up vector from the horizon
    line (line 319)."""
    import numpy as np
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    P = np.array(intrinsics_real["fisheye"]["K"], dtype=np.float64)
    # High confidence and a non-trivial intercept -> use_detected_horizon path.
    up = pl._range_up_vector(P, (0.0, 300.0, 0.99))
    assert up.shape == (3,)
    # It is NOT the bare level vector (the horizon shifted it).
    from scripts.utils.geometry import UP_LEVEL, up_from_horizon_line
    assert np.allclose(up, up_from_horizon_line(0.0, 300.0, P))


def test_detection_ranges_disabled_returns_none(intrinsics_real, detection_real):
    """range.enabled = False short-circuits to all-None (line 338)."""
    import copy

    import numpy as np
    det = copy.deepcopy(detection_real)
    det.setdefault("range", {})
    det["range"]["enabled"] = False
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    coords = [(10, 10), (20, 20)]
    sizes = [4.0, 6.0]
    P = np.array(intrinsics_real["fisheye"]["K"], dtype=np.float64)
    ranges, _raw_ranges = pl._detection_ranges(coords, sizes, P, 0.27, (0.0, 100.0, 0.9))
    assert ranges == [None, None]


def test_process_fisheye_with_attitude_provider(intrinsics_real, detection_real,
                                                small_fisheye_frame):
    """Full process_fisheye path with an IMU attitude provider attached,
    exercising the IMU branch inside _detection_ranges."""
    import cv2
    pl = ObstacleDetectionPipeline(
        intrinsics_real, detection_real,
        attitude_provider=_FakeAttitudeProvider(),
    )
    img = cv2.resize(small_fisheye_frame, (864, 648))
    out = pl.process_fisheye(img)
    assert len(out.ranges) == len(out.coords)


# ---------------------------------------------------------------------------
# Thermal quality guard (wet-cover detector) + background scaffolding
# ---------------------------------------------------------------------------

def test_thermal_quality_stats_flat_vs_structured():
    from scripts.sensor_processing.pipeline import thermal_quality_stats
    flat = np.full((120, 160, 3), 100, dtype=np.uint8)
    std_f, dyn_f = thermal_quality_stats(flat)
    assert std_f == pytest.approx(0.0)
    assert dyn_f == pytest.approx(0.0)
    structured = np.full((120, 160, 3), 50, dtype=np.uint8)
    structured[40:80, 40:120] = 220
    std_s, dyn_s = thermal_quality_stats(structured)
    assert std_s > 25
    assert dyn_s > 120


def test_thermal_quality_guard_default_off_reports_stats(
        intrinsics_real, detection_real, small_thermal_frame):
    """Guard disabled (repo default): quality_ok stays True even on a flat
    frame, but the stats are still reported for dashboards."""
    import cv2
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    flat = np.full((120, 160, 3), 100, dtype=np.uint8)
    out = pl.process_thermal(flat)
    assert out.quality_ok is True
    assert out.quality_std == pytest.approx(0.0, abs=1.0)
    assert out.quality_dyn_range is not None


def test_thermal_quality_guard_trips_and_suppresses(intrinsics_real,
                                                    detection_real,
                                                    small_thermal_frame):
    """Guard enabled with a floor no real frame can clear: detections are
    suppressed (no fusion vote) and quality_ok is False, while the same
    frame with the guard off keeps its detections."""
    import copy

    import cv2
    det = copy.deepcopy(detection_real)
    det["thermal"]["quality_guard"] = {
        "enabled": True, "min_std": 1e6, "min_dyn_range": 1e6,
    }
    img = cv2.resize(small_thermal_frame, (160, 120))

    guarded = ObstacleDetectionPipeline(intrinsics_real, det)
    out = guarded.process_thermal(img)
    assert out.quality_ok is False
    assert out.coords == [] and out.angles == [] and out.ranges == []

    unguarded = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    base = unguarded.process_thermal(img)
    assert base.quality_ok is True
    # Same inputs, same masks: only the detection suppression differs.
    np.testing.assert_array_equal(out.obstacle_mask, base.obstacle_mask)


def test_thermal_horizon_override_merges(intrinsics_real, detection_real,
                                         small_thermal_frame):
    """thermal.horizon keys override the shared horizon block for the
    thermal path only; bogus-but-valid overrides must not crash."""
    import copy

    import cv2
    det = copy.deepcopy(detection_real)
    det["thermal"]["horizon"] = {"canny_low": 10, "canny_high": 60}
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    img = cv2.resize(small_thermal_frame, (160, 120))
    out = pl.process_thermal(img)
    assert out.horizon_mask.shape == img.shape[:2]


def test_thermal_mog2_background_runs(intrinsics_real, detection_real,
                                      small_thermal_frame):
    """thermal.background.method == mog2 exercises the per-pixel model: a
    hot blob appearing after a static background warm-up is foreground."""
    import copy

    import cv2
    det = copy.deepcopy(detection_real)
    det["thermal"]["background"] = {
        "method": "mog2",
        "mog2": {"history": 10, "var_threshold": 16.0, "learning_rate": -1},
    }
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    bg = np.full((120, 160, 3), 60, dtype=np.uint8)
    for _ in range(5):
        pl.process_thermal(bg)
    hot = bg.copy()
    hot[40:80, 60:100] = 220
    out = pl.process_thermal(hot)
    # The new hot region shows up in the foreground mask.
    assert out.obstacle_mask.sum() > 0
    assert len(out.ranges) == len(out.coords)


def test_thermal_preprocess_noop_when_disabled():
    """thermal_preprocess with an empty/disabled cfg returns the input
    unchanged (the default path's byte-identity depends on it)."""
    from scripts.sensor_processing.pipeline import thermal_preprocess
    gray = np.random.default_rng(0).integers(
        0, 255, (120, 160), dtype=np.uint8)
    out = thermal_preprocess(gray, {})
    assert out is gray
    out = thermal_preprocess(gray, {"median_ksize": 0,
                                    "clahe": {"enabled": False}})
    assert out is gray


def test_thermal_preprocess_clahe_and_median_change_output():
    from scripts.sensor_processing.pipeline import thermal_preprocess
    rng = np.random.default_rng(1)
    gray = rng.integers(90, 110, (120, 160), dtype=np.uint8)
    med = thermal_preprocess(gray, {"median_ksize": 3})
    assert med.shape == gray.shape and not np.array_equal(med, gray)
    cl = thermal_preprocess(
        gray, {"clahe": {"enabled": True, "clip_limit": 4.0, "tile": 4}})
    # CLAHE stretches the narrow 90-110 band far wider.
    assert int(cl.max()) - int(cl.min()) > int(gray.max()) - int(gray.min())


def test_thermal_preprocess_enabled_changes_pipeline_mask(
        intrinsics_real, detection_real):
    """thermal.preprocess.enabled routes detection through the preprocessed
    gray: a hot spot too subtle for the default path (sub-guard, sub-thresh)
    fires only after CLAHE amplification."""
    import copy

    # Uniform 100 background with a faint 140 spot. With the running
    # average pinned at 100: base path sub = 40 max < contrast_guard 50 ->
    # no stretch -> below object_thresh -> empty mask. CLAHE lifts the spot
    # past the guard, the stretch then carries it past the threshold.
    img = np.full((120, 160, 3), 100, dtype=np.uint8)
    img[55:67, 75:87] = 140

    det_base = copy.deepcopy(detection_real)
    det_base["thermal"]["initial_average"] = 100.0
    base = ObstacleDetectionPipeline(intrinsics_real, det_base)
    out_base = base.process_thermal(img)
    assert out_base.obstacle_mask.sum() == 0

    det = copy.deepcopy(det_base)
    det["thermal"]["preprocess"] = {
        "enabled": True,
        "clahe": {"enabled": True, "clip_limit": 16.0, "tile": 4},
    }
    pp = ObstacleDetectionPipeline(intrinsics_real, det)
    out_pp = pp.process_thermal(img)
    assert out_pp.obstacle_mask.sum() > 0
    assert len(out_pp.ranges) == len(out_pp.coords)


def test_thermal_default_background_matches_pre_scaffold(
        intrinsics_real, detection_real, small_thermal_frame):
    """With the default config (running_mean, guard off) the obstacle mask
    equals a direct call to thermal_obstacle_detection: the scaffolding
    changes nothing unless opted into."""
    import cv2

    from scripts.utils.cv_common import detect_horizon, undistort_thermal
    img = cv2.resize(small_thermal_frame, (160, 120))
    pl = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    initial_avg = pl.thermal_average
    out = pl.process_thermal(img)

    intr = intrinsics_real["thermal"]
    und = undistort_thermal(img, intr["K"], intr["D"])
    from scripts.utils.cv_common import horizon_kwargs
    mask, s, b, c = detect_horizon(und, **horizon_kwargs(detection_real["horizon"]))
    masked = cv2.bitwise_and(und, und, mask=mask)
    ref_mask, ref_avg = thermal_obstacle_detection(
        masked, initial_avg, mask,
        object_thresh=detection_real["thermal"]["object_thresh"],
        contrast_guard=detection_real["thermal"]["contrast_guard"],
    )
    np.testing.assert_array_equal(out.obstacle_mask, ref_mask)
    assert out.new_average == pytest.approx(ref_avg)


# ---------------------------------------------------------------------------
# iterate_triplet: thermal-cap-not-opened branch (lines 535-537)
# ---------------------------------------------------------------------------

def test_iterate_triplet_raises_on_bad_thermal(tmp_path, detection_real,
                                               synthetic_triplet):
    """A valid fisheye but broken thermal raises RuntimeError naming the
    thermal path (lines 535-537)."""
    from scripts.utils.datasets import Triplet
    bad_therm = tmp_path / "bad_thermal.mp4"
    bad_therm.write_bytes(b"not a video")
    triplet = Triplet(
        scene=synthetic_triplet.scene,
        timestamp=synthetic_triplet.timestamp,
        fisheye=synthetic_triplet.fisheye,
        thermal=bad_therm,
        mmwave=synthetic_triplet.mmwave,
    )
    gen = iterate_triplet(triplet, detection_real)
    with pytest.raises(RuntimeError, match="bad_thermal"):
        next(gen)


# ---------------------------------------------------------------------------
# iterate_triplet: empty-radar fallback (lines 562-575) + timestamp helpers
# ---------------------------------------------------------------------------

def test_iterate_triplet_empty_radar_fallback(tmp_path, detection_real,
                                              synthetic_triplet):
    """When the radar CSV is empty (header only), iterate_triplet falls back
    to iterating fisheye frames and synthesizing timestamps from the clip's
    wall-clock filename (lines 562-575, plus the timestamp helpers)."""
    from scripts.utils.datasets import Triplet
    empty_csv = tmp_path / "empty_mmwave.csv"
    empty_csv.write_text("Date,Time,X,Y,Z\n")
    triplet = Triplet(
        scene=synthetic_triplet.scene,
        timestamp="2099-01-01_12-31-23",
        fisheye=synthetic_triplet.fisheye,
        thermal=synthetic_triplet.thermal,
        mmwave=empty_csv,
    )
    out = list(iterate_triplet(triplet, detection_real))
    assert len(out) == 5
    # First synthesized timestamp == 12:31:23 -> "12:31:23.0".
    first_ts = out[0][0]
    assert first_ts == "12:31:23.0"
    # Radar points come back empty in the fallback.
    for ts, fish, therm, pts in out:
        assert pts.shape == (0, 3)


def test_iterate_triplet_breaks_when_video_shorter_than_csv(
        tmp_path, detection_real, synthetic_triplet):
    """More radar timestamps than video frames: the loop stops at the last
    readable frame (the `break` at line 545)."""
    from scripts.utils.datasets import Triplet
    # 8 timestamps but the synthetic videos only have 5 frames.
    rows = ["Date,Time,X,Y,Z"]
    for i in range(8):
        rows.append(f"2099-01-01,00:00:{i:02d}.0,0.1,1.5,0.0")
    csv = tmp_path / "long_mmwave.csv"
    csv.write_text("\n".join(rows) + "\n")
    triplet = Triplet(
        scene=synthetic_triplet.scene,
        timestamp=synthetic_triplet.timestamp,
        fisheye=synthetic_triplet.fisheye,
        thermal=synthetic_triplet.thermal,
        mmwave=csv,
    )
    out = list(iterate_triplet(triplet, detection_real))
    # Stops at the 5 available frames, not the 8 timestamps.
    assert len(out) == 5


def test_wallclock_seconds_from_timestamp():
    from scripts.sensor_processing.pipeline import (
        _wallclock_seconds_from_timestamp,
    )
    assert _wallclock_seconds_from_timestamp("2026-06-17_12-31-23") == (
        12 * 3600 + 31 * 60 + 23
    )


def test_wallclock_seconds_no_underscore_suffix():
    from scripts.sensor_processing.pipeline import (
        _wallclock_seconds_from_timestamp,
    )
    assert _wallclock_seconds_from_timestamp("nounderscore") == 0.0


def test_wallclock_seconds_bad_hms_segments():
    from scripts.sensor_processing.pipeline import (
        _wallclock_seconds_from_timestamp,
    )
    # Suffix doesn't split into exactly three H-M-S parts.
    assert _wallclock_seconds_from_timestamp("2026-06-17_1231") == 0.0


def test_wallclock_seconds_non_integer_hms():
    from scripts.sensor_processing.pipeline import (
        _wallclock_seconds_from_timestamp,
    )
    # Three parts but not integers -> ValueError branch -> 0.0.
    assert _wallclock_seconds_from_timestamp("2026-06-17_aa-bb-cc") == 0.0


def test_format_hhmmss_basic_and_wrap():
    from scripts.sensor_processing.pipeline import _format_hhmmss
    assert _format_hhmmss(45083.0) == "12:31:23.0"
    # Wraps past midnight.
    assert _format_hhmmss(86400.0 + 1.0) == "00:00:01.0"


def test_vote_range_gate(intrinsics_real, detection_real):
    """fusion.vote_range_gate: far detections don't vote; unknown does."""
    import copy

    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline

    det = copy.deepcopy(detection_real)
    det["fusion"]["vote_range_gate"] = {"enabled": True,
                                        "max_vote_range_m": 30.0}
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    assert pl._vote_mask([5.0, 300.0, None, 29.9]) == [True, False, True, True]

    det["fusion"]["vote_range_gate"]["enabled"] = False
    pl2 = ObstacleDetectionPipeline(intrinsics_real, det)
    assert pl2._vote_mask([5.0, 300.0, None]) == [True, True, True]


def test_voting_angles_filters_fused_scores(intrinsics_real, detection_real):
    """A far (non-voting) detection is reported but raises no bin score."""
    import copy
    from types import SimpleNamespace

    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline

    det = copy.deepcopy(detection_real)
    det["fusion"]["vote_range_gate"] = {"enabled": True,
                                        "max_vote_range_m": 30.0}
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    fake_fish = SimpleNamespace(angles=[0.0, 20.0], votes=[True, False],
                                radar_ranges=[], ranges=[5.0, None])
    fused = pl.fuse(fake_fish, None, None)
    from scripts.sensor_processing.pipeline import angle_to_bin
    i0 = angle_to_bin(0.0, fused.bin_edges)
    i20 = angle_to_bin(20.0, fused.bin_edges)
    assert fused.sensor_hit_mask[i0, 0]          # near detection votes
    assert not fused.sensor_hit_mask[i20, 0]     # far detection gated out


def test_iterate_triplet_fisheye_only(tmp_path, detection_real):
    """A fisheye-only capture (no thermal video, no radar CSV) iterates on the
    synthesized timeline, yielding a None thermal frame and an empty cloud,
    so the fisheye-only tools (undistort export, segmentation) still run."""
    import cv2
    import numpy as np

    from scripts.sensor_processing.pipeline import iterate_triplet
    from scripts.utils.datasets import Triplet

    scene = tmp_path / "Synth"
    scene.mkdir()
    ts = "2099-01-01_12-00-00"
    fish = scene / f"fisheye_{ts}.mp4"
    fw = cv2.VideoWriter(str(fish), cv2.VideoWriter_fourcc(*"mp4v"), 3.0,
                         (160, 120))
    for _ in range(5):
        fw.write(np.full((120, 160, 3), 70, dtype=np.uint8))
    fw.release()

    triplet = Triplet(scene="Synth", timestamp=ts, fisheye=fish,
                      thermal=None, mmwave=None)
    frames = list(iterate_triplet(triplet, detection_real, clip_overrides={}))
    assert len(frames) == 5
    for _ts, fisheye_frame, thermal_frame, pts in frames:
        assert fisheye_frame is not None
        assert thermal_frame is None
        assert len(pts) == 0
    # Timeline is synthesized from the clip filename at the 3 fps nominal rate.
    assert frames[0][0].startswith("12:00:00")
