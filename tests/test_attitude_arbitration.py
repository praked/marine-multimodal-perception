"""Tests for attitude-source arbitration + the IMU-seeded horizon (A-now.2)."""

import math

import numpy as np
import pytest

from scripts.utils.geometry import (
    UP_LEVEL,
    horizon_line_from_up,
    pitch_roll_from_up,
    up_from_horizon_line,
    up_from_pitch_roll,
)

P = np.array([[400.0, 0.0, 432.0],
              [0.0, 400.0, 324.0],
              [0.0, 0.0, 1.0]])


class TestGeometryInverses:
    @pytest.mark.parametrize("pitch,roll", [
        (0.0, 0.0), (0.1, 0.0), (0.0, -0.2), (0.15, 0.08), (-0.3, 0.25),
    ])
    def test_pitch_roll_roundtrip(self, pitch, roll):
        up = up_from_pitch_roll(pitch, roll)
        p, r = pitch_roll_from_up(up)
        assert p == pytest.approx(pitch, abs=1e-9)
        assert r == pytest.approx(roll, abs=1e-9)

    def test_level_up_gives_zero(self):
        assert pitch_roll_from_up(UP_LEVEL) == (0.0, 0.0)

    @pytest.mark.parametrize("pitch,roll", [
        (0.0, 0.0), (0.05, 0.0), (0.0, 0.1), (-0.08, -0.12),
    ])
    def test_horizon_line_roundtrip(self, pitch, roll):
        up = up_from_pitch_roll(pitch, roll)
        slope, intercept = horizon_line_from_up(up, P)
        up2 = up_from_horizon_line(slope, intercept, P)
        # up vectors equal up to sign normalisation (line has no orientation)
        assert np.allclose(np.abs(up2 @ up), 1.0, atol=1e-9)

    def test_level_horizon_is_principal_row(self):
        slope, intercept = horizon_line_from_up(UP_LEVEL, P)
        assert slope == pytest.approx(0.0, abs=1e-12)
        assert intercept == pytest.approx(324.0)

    def test_pitch_down_moves_horizon_up(self):
        # nose down (negative pitch) -> horizon row above the principal point
        up = up_from_pitch_roll(-0.1, 0.0)
        _s, intercept = horizon_line_from_up(up, P)
        assert intercept < 324.0


class _FixedAttitude:
    """Minimal attitude provider stub."""

    def __init__(self, pitch_rad=0.0, roll_rad=0.0):
        from scripts.sensor_processing.imu_bno085 import Attitude
        self._att = Attitude(pitch_rad=pitch_rad, roll_rad=roll_rad)

    def get(self):
        return self._att


class TestRangeUpPriority:
    def _pipeline(self, detection, intrinsics, attitude=None):
        from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
        return ObstacleDetectionPipeline(intrinsics, detection,
                                         attitude_provider=attitude)

    def test_default_order_imu_wins(self, detection_real, intrinsics_real):
        att = _FixedAttitude(pitch_rad=0.1, roll_rad=-0.05)
        p = self._pipeline(detection_real, intrinsics_real, att)
        up = p._range_up_vector(P, (0.0, 324.0, 1.0))
        assert np.allclose(up, up_from_pitch_roll(0.1, -0.05))

    def test_priority_reorder_horizon_first(self, detection_real,
                                            intrinsics_real):
        detection_real["range"]["attitude_priority"] = ["horizon", "imu", "level"]
        att = _FixedAttitude(pitch_rad=0.1)
        p = self._pipeline(detection_real, intrinsics_real, att)
        # confident, plausible horizon at the principal row -> level-ish up
        up = p._range_up_vector(P, (0.0, 324.0, 1.0))
        pitch, roll = pitch_roll_from_up(up)
        assert pitch == pytest.approx(0.0, abs=1e-6)

    def test_imu_gate_rejects_deviant_horizon(self, detection_real,
                                              intrinsics_real):
        # horizon implies ~14 deg pitch (100 px / fx=418.5); IMU says level;
        # gate 5 deg -> horizon rejected -> falls through to imu.
        detection_real["range"]["attitude_priority"] = ["horizon", "imu", "level"]
        detection_real["range"]["imu_gate_deg"] = 5.0
        detection_real["range"]["horizon_max_tilt_px"] = 1000.0
        att = _FixedAttitude()
        p = self._pipeline(detection_real, intrinsics_real, att)
        up = p._range_up_vector(P, (0.0, 224.0, 1.0))
        pitch, roll = pitch_roll_from_up(up)
        assert abs(math.degrees(pitch)) < 1e-6   # IMU (level), not the horizon

    def test_gate_off_keeps_horizon(self, detection_real, intrinsics_real):
        detection_real["range"]["attitude_priority"] = ["horizon", "imu", "level"]
        detection_real["range"]["imu_gate_deg"] = 0.0
        detection_real["range"]["horizon_max_tilt_px"] = 1000.0
        att = _FixedAttitude()
        p = self._pipeline(detection_real, intrinsics_real, att)
        up = p._range_up_vector(P, (0.0, 224.0, 1.0))
        pitch, _ = pitch_roll_from_up(up)
        assert abs(math.degrees(pitch)) > 5.0    # the deviant horizon survives


class TestImuSeededHorizon:
    def _pipeline(self, detection, intrinsics, attitude=None):
        from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
        return ObstacleDetectionPipeline(intrinsics, detection,
                                         attitude_provider=attitude)

    def _frame(self):
        # Strong synthetic horizon: dark top half, bright bottom half at row
        # 200 of a 648x864 frame (far from the IMU-level prediction at cy).
        img = np.zeros((648, 864, 3), dtype=np.uint8)
        img[200:, :] = 200
        return img

    def test_off_keeps_ransac_line(self, detection_real, intrinsics_real):
        # The undistortion curves the synthetic straight edge, capping RANSAC
        # confidence ~0.34: drop the gate so the line is accepted.
        detection_real["horizon"]["confidence_thresh"] = 0.25
        p = self._pipeline(detection_real, intrinsics_real, _FixedAttitude())
        res = p.process_fisheye(self._frame())
        _s, intercept, conf = res.horizon_line
        assert conf > 0.25 and abs(intercept - 200) < 40

    def test_seed_replaces_deviant_line(self, detection_real, intrinsics_real):
        detection_real["horizon"]["imu_seed"] = {"enabled": True,
                                                 "max_dev_px": 60}
        att = _FixedAttitude()   # level -> predicted horizon near cy (~300)
        p = self._pipeline(detection_real, intrinsics_real, att)
        res = p.process_fisheye(self._frame())
        slope, intercept, conf = res.horizon_line
        K = np.asarray(intrinsics_real["fisheye"]["K"], float)
        cy = K[1, 2]
        row_c = slope * K[0, 2] + intercept
        # the row-200 edge deviates > 60 px from the IMU-level prediction ->
        # replaced by the IMU line at the principal row, confidence 1.0
        assert conf == 1.0
        assert abs(row_c - cy) < 2.0

    def test_seed_no_attitude_is_noop(self, detection_real, intrinsics_real):
        detection_real["horizon"]["confidence_thresh"] = 0.25
        detection_real["horizon"]["imu_seed"] = {"enabled": True,
                                                 "max_dev_px": 60}
        p = self._pipeline(detection_real, intrinsics_real, attitude=None)
        res = p.process_fisheye(self._frame())
        _s, intercept, conf = res.horizon_line
        assert abs(intercept - 200) < 40
