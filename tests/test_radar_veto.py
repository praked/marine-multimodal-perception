"""Tests for the radar-contradiction range veto (range.radar_veto)."""

import copy

import numpy as np

from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline


def _res(ranges, hits, angles):
    from types import SimpleNamespace
    return SimpleNamespace(ranges=list(ranges), radar_hits=list(hits),
                           angles=list(angles), radar_ranges=[1] * len(hits),
                           mono_ranges=[])


def _mm(n=3):
    from types import SimpleNamespace
    return SimpleNamespace(points_xyz=np.ones((n, 3)))


def _pipeline(detection, intrinsics, **veto):
    det = copy.deepcopy(detection)
    det["range"]["radar_veto"] = {"enabled": True, "min_m": 1.0, "max_m": 8.0,
                                  "max_bearing_deg": 50.0, **veto}
    return ObstacleDetectionPipeline(intrinsics, det)


class TestRadarVeto:
    def test_near_unconfirmed_demoted(self, detection_real, intrinsics_real):
        p = _pipeline(detection_real, intrinsics_real)
        r = _res([3.0], [0], [10.0])
        p._apply_radar_veto(r, _mm())
        assert r.ranges == [None]
        assert r.mono_ranges == [3.0]

    def test_radar_confirmed_kept(self, detection_real, intrinsics_real):
        p = _pipeline(detection_real, intrinsics_real)
        r = _res([3.0], [2], [10.0])
        p._apply_radar_veto(r, _mm())
        assert r.ranges == [3.0]

    def test_far_reading_not_vetoed(self, detection_real, intrinsics_real):
        p = _pipeline(detection_real, intrinsics_real)
        r = _res([12.0], [0], [10.0])
        p._apply_radar_veto(r, _mm())
        assert r.ranges == [12.0]

    def test_outside_bearing_cone_not_vetoed(self, detection_real,
                                             intrinsics_real):
        p = _pipeline(detection_real, intrinsics_real)
        r = _res([3.0], [0], [60.0])
        p._apply_radar_veto(r, _mm())
        assert r.ranges == [3.0]

    def test_radar_silent_no_veto(self, detection_real, intrinsics_real):
        p = _pipeline(detection_real, intrinsics_real)
        r = _res([3.0], [0], [10.0])
        p._apply_radar_veto(r, _mm(n=0))
        assert r.ranges == [3.0]

    def test_no_association_no_veto(self, detection_real, intrinsics_real):
        from types import SimpleNamespace
        p = _pipeline(detection_real, intrinsics_real)
        r = SimpleNamespace(ranges=[3.0], radar_hits=[], angles=[10.0],
                            radar_ranges=[], mono_ranges=[])
        p._apply_radar_veto(r, _mm())
        assert r.ranges == [3.0]

    def test_disabled_is_noop(self, detection_real, intrinsics_real):
        det = copy.deepcopy(detection_real)
        det["range"]["radar_veto"] = {"enabled": False}
        p = ObstacleDetectionPipeline(intrinsics_real, det)
        r = _res([3.0], [0], [10.0])
        p._apply_radar_veto(r, _mm())
        assert r.ranges == [3.0]
