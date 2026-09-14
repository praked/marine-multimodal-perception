"""Tests for the boat-frame self-clutter box (mmwave.self_clutter)."""

import copy

import numpy as np
import pytest

from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline


def _pipeline(detection, intrinsics, **sc):
    det = copy.deepcopy(detection)
    det["mmwave"]["y_min"] = 0.1   # keep the near field open for these tests
    det["mmwave"]["self_clutter"] = {
        "enabled": True, "x_min": -0.75, "x_max": 0.25,
        "zone_y_min": 0.0, "zone_y_max": 1.2,
        "keep_if_doppler_above": 0.15, **sc,
    }
    return ObstacleDetectionPipeline(intrinsics, det)


class TestSelfClutterBox:
    def test_static_point_in_zone_dropped(self, detection_real, intrinsics_real):
        p = _pipeline(detection_real, intrinsics_real)
        res = p.process_mmwave(np.array([[-0.3, 0.6, 0.0, 0.0]]))
        assert len(res.points_xyz) == 0

    def test_moving_point_in_zone_kept(self, detection_real, intrinsics_real):
        p = _pipeline(detection_real, intrinsics_real)
        res = p.process_mmwave(np.array([[-0.3, 0.6, 0.0, -0.4]]))
        assert len(res.points_xyz) == 1

    def test_point_outside_zone_kept(self, detection_real, intrinsics_real):
        p = _pipeline(detection_real, intrinsics_real)
        res = p.process_mmwave(np.array([[1.5, 0.6, 0.0, 0.0]]))
        assert len(res.points_xyz) == 1

    def test_nx3_input_zone_drops_without_doppler(self, detection_real,
                                                  intrinsics_real):
        # No Doppler column -> no escape hatch; zone point is dropped.
        p = _pipeline(detection_real, intrinsics_real)
        res = p.process_mmwave(np.array([[-0.3, 0.6, 0.0]]))
        assert len(res.points_xyz) == 0

    def test_nan_doppler_treated_as_static(self, detection_real,
                                           intrinsics_real):
        p = _pipeline(detection_real, intrinsics_real)
        res = p.process_mmwave(np.array([[-0.3, 0.6, 0.0, np.nan]]))
        assert len(res.points_xyz) == 0

    def test_disabled_is_noop(self, detection_real, intrinsics_real):
        det = copy.deepcopy(detection_real)
        det["mmwave"]["y_min"] = 0.1
        det["mmwave"].setdefault("self_clutter", {})["enabled"] = False
        p = ObstacleDetectionPipeline(intrinsics_real, det)
        res = p.process_mmwave(np.array([[-0.3, 0.6, 0.0, 0.0]]))
        assert len(res.points_xyz) == 1
