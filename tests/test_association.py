"""Tests for pixel-level radar<->camera association (A.1)."""

import numpy as np
import pytest

from scripts.utils.association import (
    AssociationParams,
    associate_radar_to_detections,
    match_polygons_to_detections,
    pinhole_bearings,
    polygons_from_instances,
)
from scripts.utils.geometry import project_radar_to_undistorted

# Flat-plate axis swap (extrinsics.yaml rotation block), zero translation:
# cam.X = radar.X, cam.Y = -radar.Z, cam.Z = radar.Y.
T_SWAP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
P = np.array([[400.0, 0.0, 432.0],
              [0.0, 400.0, 324.0],
              [0.0, 0.0, 1.0]])


class TestProjectUndistorted:
    def test_dead_ahead_lands_on_principal_point(self):
        px = project_radar_to_undistorted(np.array([[0.0, 3.0, 0.0]]), P, T_SWAP)
        assert px[0, 0] == pytest.approx(432.0)
        assert px[0, 1] == pytest.approx(324.0)

    def test_right_offset_projects_right(self):
        px = project_radar_to_undistorted(np.array([[1.0, 3.0, 0.0]]), P, T_SWAP)
        assert px[0, 0] == pytest.approx(432.0 + 400.0 / 3.0)

    def test_behind_camera_is_nan(self):
        px = project_radar_to_undistorted(np.array([[0.0, -1.0, 0.0]]), P, T_SWAP)
        assert np.isnan(px).all()

    def test_empty(self):
        assert project_radar_to_undistorted(np.empty((0, 3)), P, T_SWAP).shape == (0, 2)


class TestAssociate:
    def test_match_takes_min_range(self):
        pts = np.array([[0.0, 3.0, 0.0], [0.0, 5.0, 0.0]])
        px = np.array([[100.0, 100.0], [105.0, 98.0]])
        res = associate_radar_to_detections(
            pts, px, [(100, 100)], [20.0], AssociationParams(max_px=10))
        assert res.matches[0].n_points == 2
        assert res.matches[0].radar_range_m == pytest.approx(3.0)
        assert res.n_confirmed == 1

    def test_median_agg(self):
        pts = np.array([[0.0, 3.0, 0.0], [0.0, 5.0, 0.0], [0.0, 7.0, 0.0]])
        px = np.tile([100.0, 100.0], (3, 1))
        res = associate_radar_to_detections(
            pts, px, [(100, 100)], [10.0],
            AssociationParams(max_px=5, range_agg="median"))
        assert res.matches[0].radar_range_m == pytest.approx(5.0)

    def test_outside_gate_no_match(self):
        pts = np.array([[0.0, 3.0, 0.0]])
        px = np.array([[300.0, 300.0]])
        res = associate_radar_to_detections(
            pts, px, [(100, 100)], [20.0], AssociationParams(max_px=10))
        assert res.matches[0].radar_range_m is None
        assert res.matches[0].n_points == 0

    def test_nan_projection_never_matches(self):
        pts = np.array([[0.0, -3.0, 0.0]])
        px = np.array([[np.nan, np.nan]])
        res = associate_radar_to_detections(
            pts, px, [(100, 100)], [20.0], AssociationParams(max_px=1e9))
        assert res.matches[0].radar_range_m is None

    def test_min_points_gate(self):
        pts = np.array([[0.0, 3.0, 0.0]])
        px = np.array([[100.0, 100.0]])
        res = associate_radar_to_detections(
            pts, px, [(100, 100)], [20.0],
            AssociationParams(max_px=10, min_points=2))
        assert res.matches[0].radar_range_m is None
        assert res.matches[0].n_points == 1

    def test_no_radar(self):
        res = associate_radar_to_detections(
            np.empty((0, 3)), np.empty((0, 2)), [(1, 1)], [4.0])
        assert res.matches[0].radar_range_m is None

    def test_bearing_of_matched_points(self):
        pts = np.array([[3.0, 3.0, 0.0]])   # 45 deg right
        px = np.array([[100.0, 100.0]])
        res = associate_radar_to_detections(
            pts, px, [(100, 100)], [10.0], AssociationParams(max_px=5))
        assert res.matches[0].radar_bearing_deg == pytest.approx(45.0)


def _square(cx, cy, half):
    """Axis-aligned square polygon (Nx2 float32 px) centred on (cx, cy)."""
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                     [cx + half, cy + half], [cx - half, cy + half]],
                    dtype=np.float32)


class TestInstanceGate:
    """Polygon containment replaces the padded-bbox gate per detection
    (fusion.association.use_instance_masks). The tall-edge-box failure:
    a foreground return inside the RECTANGLE but outside the SILHOUETTE
    must not match."""

    def test_polygon_excludes_in_box_but_off_silhouette_point(self):
        # Detection box would span +-30 px around (100, 100); the instance
        # silhouette is a 10-px square. One return projects on the object
        # (5.0 m), one inside the box but off the polygon (the 2 m
        # foreground clutter). Box gate takes min = 2.0; instance gate
        # must keep only the on-silhouette 5.0 m.
        pts = np.array([[0.0, 5.0, 0.0], [0.0, 2.0, 0.0]])
        px = np.array([[100.0, 100.0], [125.0, 100.0]])
        poly = _square(100, 100, 10)
        box = associate_radar_to_detections(
            pts, px, [(100, 100)], [40.0], AssociationParams(max_px=10))
        inst = associate_radar_to_detections(
            pts, px, [(100, 100)], [40.0],
            AssociationParams(max_px=10, instance_margin_px=0.0),
            det_polygons=[poly])
        assert box.matches[0].radar_range_m == pytest.approx(2.0)
        assert inst.matches[0].radar_range_m == pytest.approx(5.0)
        assert inst.matches[0].n_points == 1
        assert inst.matches[0].point_indices == [0]
        assert box.matches[0].gate == "box"
        assert inst.matches[0].gate == "instance"

    def test_margin_dilates_the_polygon(self):
        # Point 6 px outside the polygon boundary: excluded at margin 0,
        # matched at margin 8.
        pts = np.array([[0.0, 3.0, 0.0]])
        px = np.array([[116.0, 100.0]])   # 6 px right of the square's edge
        poly = _square(100, 100, 10)
        for margin, expect in ((0.0, None), (8.0, 3.0)):
            res = associate_radar_to_detections(
                pts, px, [(100, 100)], [40.0],
                AssociationParams(instance_margin_px=margin),
                det_polygons=[poly])
            if expect is None:
                assert res.matches[0].radar_range_m is None
            else:
                assert res.matches[0].radar_range_m == pytest.approx(expect)

    def test_none_entry_falls_back_to_box_gate(self):
        # Mixed list: detection 0 has a polygon, detection 1 does not and
        # keeps the padded-box behaviour.
        pts = np.array([[0.0, 4.0, 0.0]])
        px = np.array([[300.0, 300.0]])
        poly = _square(100, 100, 10)
        res = associate_radar_to_detections(
            pts, px, [(100, 100), (300, 300)], [40.0, 40.0],
            AssociationParams(max_px=10),
            det_polygons=[poly, None])
        assert res.matches[0].radar_range_m is None       # off-polygon
        assert res.matches[0].gate == "instance"
        assert res.matches[1].radar_range_m == pytest.approx(4.0)
        assert res.matches[1].gate == "box"

    def test_nan_projection_never_matches_polygon(self):
        pts = np.array([[0.0, -3.0, 0.0]])
        px = np.array([[np.nan, np.nan]])
        res = associate_radar_to_detections(
            pts, px, [(100, 100)], [40.0],
            AssociationParams(instance_margin_px=1e9),
            det_polygons=[_square(100, 100, 10)])
        assert res.matches[0].radar_range_m is None

    def test_no_polygons_arg_is_identical_to_box(self):
        # det_polygons=None must reproduce the pure box-gate result
        # exactly (the flag-off path).
        rng = np.random.default_rng(3)
        pts = np.column_stack([rng.uniform(-4, 4, 40),
                               rng.uniform(0.5, 9, 40),
                               np.zeros(40)])
        px = np.column_stack([rng.uniform(0, 864, 40),
                              rng.uniform(0, 648, 40)])
        coords, sizes = [(200, 300), (600, 320)], [30.0, 55.0]
        a = associate_radar_to_detections(pts, px, coords, sizes,
                                          AssociationParams())
        b = associate_radar_to_detections(pts, px, coords, sizes,
                                          AssociationParams(),
                                          det_polygons=None)
        for ma, mb in zip(a.matches, b.matches):
            assert ma.radar_range_m == mb.radar_range_m
            assert ma.n_points == mb.n_points
            assert ma.point_indices == mb.point_indices


class TestPolygonHelpers:
    def test_polygons_from_instances_denormalises(self):
        inst = [{"cls": "boat_ship", "confidence": 0.8,
                 "polygon": [0.25, 0.5, 0.75, 0.5, 0.5, 1.0]}]
        polys = polygons_from_instances(inst, (864, 648))
        assert len(polys) == 1
        assert polys[0] == pytest.approx(
            np.array([[216.0, 324.0], [648.0, 324.0], [432.0, 648.0]]))

    def test_degenerate_polygons_dropped(self):
        inst = [{"polygon": [0.1, 0.1, 0.2, 0.2]},   # 2 vertices
                {"polygon": []}, {}]
        assert polygons_from_instances(inst, (864, 648)) == []

    def test_match_prefers_smallest_containing_polygon(self):
        # A buoy silhouette inside the dock's outline: the detection centre
        # is inside both; the smaller (more specific) polygon wins.
        big = _square(100, 100, 80)
        small = _square(100, 100, 12)
        assert match_polygons_to_detections(
            [big, small], [(100, 100)], [20.0]) == [1]

    def test_match_within_blob_radius_tolerance(self):
        # Centre 15 px outside the polygon, blob radius 20 px -> matched;
        # blob radius 10 px -> not covered.
        poly = _square(100, 100, 10)
        assert match_polygons_to_detections([poly], [(125, 100)], [40.0]) \
            == [0]
        assert match_polygons_to_detections([poly], [(125, 100)], [20.0]) \
            == [None]


class TestPinholeBearings:
    def test_principal_point_is_zero(self):
        assert pinhole_bearings([(432, 100)], P)[0] == pytest.approx(0.0)

    def test_known_angle(self):
        u = 432.0 + 400.0 * np.tan(np.radians(10.0))
        assert pinhole_bearings([(u, 0)], P)[0] == pytest.approx(10.0, abs=1e-6)


class TestPipelineIntegration:
    def _pipeline(self, detection, intrinsics):
        from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
        return ObstacleDetectionPipeline(intrinsics, detection)

    def _frames(self):
        rng = np.random.default_rng(7)
        fish = rng.integers(0, 255, (648, 864, 3), dtype=np.uint8)
        therm = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        return fish, therm

    def test_off_is_shape_stable_and_unconfirmed(self, detection_real,
                                                 intrinsics_real):
        fish, therm = self._frames()
        detection_real["fusion"]["association"]["enabled"] = False
        p = self._pipeline(detection_real, intrinsics_real)
        res = p.process_frame(fish, therm, np.array([[0.0, 2.0, 0.0]]))
        assert res.fusion.confirmed == [False] * len(res.fusion.bin_centers)
        assert res.fisheye.radar_ranges == []
        assert res.fisheye.mono_ranges == []

    def test_on_attaches_radar_evidence(self, detection_real, intrinsics_real):
        fish, therm = self._frames()
        detection_real["fusion"]["association"]["enabled"] = True
        # Huge gate: every projected in-front point matches every detection,
        # so any detection at all must pick up the radar range. The range-
        # compatibility gate is disabled here: this test covers the pure
        # attachment mechanics (TestRangeCompatGate covers the gate).
        detection_real["fusion"]["association"]["max_px"] = 1e6
        detection_real["fusion"]["association"]["range_compat_ratio"] = 0.0
        # Pin the per-box mount yaw: this covers attachment mechanics on a
        # synthetic dead-ahead point, and the rotation would move it off axis.
        detection_real["mmwave"] = {**detection_real["mmwave"], "mount_yaw_deg": 0.0}
        p = self._pipeline(detection_real, intrinsics_real)
        res = p.process_frame(fish, therm, np.array([[0.0, 2.0, 0.0]]))
        f = res.fisheye
        if f.coords:   # noise frame usually yields blobs; guard regardless
            assert len(f.radar_ranges) == len(f.coords)
            assert all(r == pytest.approx(2.0) for r in f.radar_ranges)
            assert len(f.mono_ranges) == len(f.coords)
            assert any(res.fusion.confirmed)
            # min_range in a confirmed bin is tightened to the radar range
            for i, c in enumerate(res.fusion.confirmed):
                if c:
                    assert res.fusion.min_ranges[i] <= 2.0 + 1e-9

    def test_bearing_model_pinhole_changes_angles(self, detection_real,
                                                  intrinsics_real):
        fish, therm = self._frames()
        import copy
        det2 = copy.deepcopy(detection_real)
        detection_real["fisheye"]["bearing_model"] = "linear"
        det2["fisheye"]["bearing_model"] = "pinhole"
        p1 = self._pipeline(detection_real, intrinsics_real)
        p2 = self._pipeline(det2, intrinsics_real)
        r1 = p1.process_fisheye(fish)
        r2 = p2.process_fisheye(fish)
        assert r1.coords == r2.coords
        if r1.angles:
            # cx 472 vs calibrated 444.1 shifts bearings ~+4 deg near the
            # centre; at the edges the tangent compression pulls the other
            # way. Either way the two models must disagree measurably.
            deltas = [abs(a2 - a1) for a1, a2 in zip(r1.angles, r2.angles)]
            assert max(deltas) > 0.5


class TestRangeCompatGate:
    def _pipeline(self, detection, intrinsics, ratio=0.5):
        import copy

        from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
        det = copy.deepcopy(detection)
        det["fusion"]["association"] = {"enabled": True, "max_px": 1e6,
                                        "min_points": 1, "range_agg": "min",
                                        "use_radar_range": True,
                                        "range_compat_ratio": ratio}
        return ObstacleDetectionPipeline(intrinsics, det)

    def _res(self, raw):
        from types import SimpleNamespace
        return SimpleNamespace(coords=[(400, 400)], sizes=[40.0],
                               ranges=[raw], raw_ranges=[raw],
                               radar_ranges=[], radar_hits=[],
                               mono_ranges=[])

    def test_foreground_return_rejected(self, detection_real, intrinsics_real):
        # camera raw says 20 m; radar point in box at 2.9 m -> foreground
        import numpy as np
        p = self._pipeline(detection_real, intrinsics_real)
        r = self._res(20.0)
        p._apply_association(r, __import__('types').SimpleNamespace(
            points_xyz=np.array([[0.0, 2.9, 0.0]])),
            np.asarray(intrinsics_real["fisheye"]["K"], float),
            p.extrinsics["T_radar_to_fisheye"])
        assert r.radar_ranges == [None]      # match rejected for range
        assert r.radar_hits == [1]           # ...but the return is counted
        assert r.ranges == [20.0]            # camera estimate kept

    def test_camera_far_none_rejects_near_radar(self, detection_real,
                                                intrinsics_real):
        import numpy as np
        p = self._pipeline(detection_real, intrinsics_real)
        r = self._res(None)                  # ray above horizon = far
        p._apply_association(r, __import__('types').SimpleNamespace(
            points_xyz=np.array([[0.0, 2.9, 0.0]])),
            np.asarray(intrinsics_real["fisheye"]["K"], float),
            p.extrinsics["T_radar_to_fisheye"])
        assert r.radar_ranges == [None]

    def test_compatible_match_accepted(self, detection_real, intrinsics_real):
        import numpy as np
        p = self._pipeline(detection_real, intrinsics_real)
        r = self._res(3.5)                   # camera 3.5 m, radar 2.9 m -> ok
        p._apply_association(r, __import__('types').SimpleNamespace(
            points_xyz=np.array([[0.0, 2.9, 0.0]])),
            np.asarray(intrinsics_real["fisheye"]["K"], float),
            p.extrinsics["T_radar_to_fisheye"])
        assert r.radar_ranges == [2.9]
        assert r.ranges == [2.9]             # radar range substituted

    def test_ratio_zero_disables_gate(self, detection_real, intrinsics_real):
        import numpy as np
        p = self._pipeline(detection_real, intrinsics_real, ratio=0.0)
        r = self._res(20.0)
        p._apply_association(r, __import__('types').SimpleNamespace(
            points_xyz=np.array([[0.0, 2.9, 0.0]])),
            np.asarray(intrinsics_real["fisheye"]["K"], float),
            p.extrinsics["T_radar_to_fisheye"])
        assert r.radar_ranges == [2.9]


class TestSilhouetteArbitration:
    """Seg-silhouette rule: a return on the object's own obstacle pixels
    overrides the compatibility gate (the camera reads wrong-far on near
    objects whose contact lands at the water boundary)."""

    def _pipeline(self, detection, intrinsics):
        import copy

        from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
        det = copy.deepcopy(detection)
        det["fusion"]["association"] = {"enabled": True, "max_px": 1e6,
                                        "min_points": 1, "range_agg": "min",
                                        "use_radar_range": True,
                                        "range_compat_ratio": 0.5}
        return ObstacleDetectionPipeline(intrinsics, det)

    def _res(self, raw, seg):
        from types import SimpleNamespace
        return SimpleNamespace(coords=[(400, 400)], sizes=[40.0],
                               ranges=[raw], raw_ranges=[raw],
                               radar_ranges=[], radar_hits=[],
                               mono_ranges=[], seg_mask=seg)

    def _mm(self):
        import numpy as np
        from types import SimpleNamespace
        # dead-ahead point at 2.9 m -> projects near the principal point
        return SimpleNamespace(points_xyz=np.array([[0.0, 2.9, 0.0]]))

    def test_on_silhouette_overrides_compat(self, detection_real,
                                            intrinsics_real):
        import numpy as np
        p = self._pipeline(detection_real, intrinsics_real)
        seg = np.full((648, 864), 0, np.uint8)      # all OBSTACLE
        r = self._res(20.0, seg)                     # camera wrongly far
        p._apply_association(r, self._mm(),
                             np.asarray(intrinsics_real["fisheye"]["K"], float),
                             p.extrinsics["T_radar_to_fisheye"])
        assert r.radar_ranges == [2.9]               # silhouette wins
        assert r.ranges == [2.9]

    def test_on_water_still_rejected(self, detection_real, intrinsics_real):
        import numpy as np
        p = self._pipeline(detection_real, intrinsics_real)
        seg = np.full((648, 864), 1, np.uint8)      # all WATER
        r = self._res(20.0, seg)
        p._apply_association(r, self._mm(),
                             np.asarray(intrinsics_real["fisheye"]["K"], float),
                             p.extrinsics["T_radar_to_fisheye"])
        assert r.radar_ranges == [None]              # compat gate holds
        assert r.ranges == [20.0]


class TestConfirmSemantics:
    """_confirm_bins: per-point radar anchoring + same-bin agreement.

    2026-08-05 tightenings: the confirmed flag marks the bins the matched
    returns actually occupy (confirm_anchor: radar, per point), and only
    when the detection's own bin is among them (confirm_same_bin). A fully
    cross-bin match (an open-water thermal blob matched to a neighbouring
    object's returns) must confirm nothing.
    """

    def _pipeline(self, detection, intrinsics, **assoc):
        from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
        # These tests pin bearings to specific bins: fix the 10-degree
        # bin geometry explicitly so the 2026-08-24 15-degree default
        # (D.2) does not shift the fixtures.
        detection["fusion"].update(
            {"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10})
        detection["fusion"]["association"]["enabled"] = True
        detection["fusion"]["association"].update(assoc)
        return ObstacleDetectionPipeline(intrinsics, detection)

    @staticmethod
    def _fused(p):
        from types import SimpleNamespace
        from scripts.sensor_processing.pipeline import make_bins
        edges, centers = make_bins(p.detection["fusion"])
        return SimpleNamespace(bin_edges=edges, bin_centers=centers,
                               confirmed=[False] * len(centers),
                               min_ranges=[None] * len(centers))

    @staticmethod
    def _res(angle, rr, match_pts):
        from types import SimpleNamespace
        return SimpleNamespace(
            angles=[angle], radar_ranges=[rr],
            radar_match_points=[np.asarray(match_pts, float)
                                if match_pts is not None else None])

    def test_per_point_marks_every_occupied_bin(self, detection_real,
                                                intrinsics_real):
        # The dock case: one wide detection centred in bin +30 whose matched
        # returns span three bins. Every occupied bin is marked and each
        # min_range is the nearest matched return IN that bin.
        p = self._pipeline(detection_real, intrinsics_real)
        fused = self._fused(p)
        res = self._res(34.5, 0.77, [(12.0, 1.75), (22.0, 1.57), (34.0, 0.77)])
        p._confirm_bins(fused, res)
        marked = [c for c, cf in zip(fused.bin_centers, fused.confirmed) if cf]
        assert marked == [10.0, 20.0, 30.0]
        by_center = dict(zip(fused.bin_centers, fused.min_ranges))
        assert by_center[10.0] == pytest.approx(1.75)
        assert by_center[20.0] == pytest.approx(1.57)
        assert by_center[30.0] == pytest.approx(0.77)

    def test_cross_bin_match_confirms_nothing(self, detection_real,
                                              intrinsics_real):
        # The frame-88 case: a detection at +3 deg (bin 0) whose only
        # matched returns sit in bin +10. Nothing is marked, and no
        # min_range is borrowed.
        p = self._pipeline(detection_real, intrinsics_real)
        fused = self._fused(p)
        res = self._res(3.1, 2.5, [(7.0, 2.5), (8.0, 2.6)])
        p._confirm_bins(fused, res)
        assert not any(fused.confirmed)
        assert all(mr is None for mr in fused.min_ranges)

    def test_same_bin_off_anchors_on_the_evidence(self, detection_real,
                                                  intrinsics_real):
        # With the agreement gate off, the mark goes to the bin the
        # evidence occupies, never the detection's own (empty) bin.
        p = self._pipeline(detection_real, intrinsics_real,
                           confirm_same_bin=False)
        fused = self._fused(p)
        res = self._res(3.1, 2.5, [(7.0, 2.5)])
        p._confirm_bins(fused, res)
        marked = [c for c, cf in zip(fused.bin_centers, fused.confirmed) if cf]
        assert marked == [10.0]
        assert dict(zip(fused.bin_centers, fused.min_ranges))[0.0] is None

    def test_legacy_detection_anchor(self, detection_real, intrinsics_real):
        # confirm_anchor: detection restores the pre-2026-08-05 behaviour:
        # the detection's centre bin is marked with the aggregated range.
        p = self._pipeline(detection_real, intrinsics_real,
                           confirm_same_bin=False, confirm_anchor="detection")
        fused = self._fused(p)
        res = self._res(3.1, 2.5, [(7.0, 2.5)])
        p._confirm_bins(fused, res)
        marked = [c for c, cf in zip(fused.bin_centers, fused.confirmed) if cf]
        assert marked == [0.0]
        assert dict(zip(fused.bin_centers, fused.min_ranges))[0.0] \
            == pytest.approx(2.5)

    def test_unassociated_detection_is_ignored(self, detection_real,
                                               intrinsics_real):
        p = self._pipeline(detection_real, intrinsics_real)
        fused = self._fused(p)
        p._confirm_bins(fused, self._res(3.1, None, None))
        assert not any(fused.confirmed)

    def test_thermal_gate_pad_default_and_override(self, detection_real,
                                                   intrinsics_real):
        # The pad absorbs angular error, so the thermal gate defaults to
        # 12 px (the fisheye's ~4.3 deg at the thermal's ~2.81 px/deg)
        # and is overridable independently of the fisheye's.
        p = self._pipeline(detection_real, intrinsics_real)
        assert p._assoc_params.max_px == pytest.approx(32.0)
        assert p._assoc_params_thermal.max_px == pytest.approx(12.0)
        p2 = self._pipeline(detection_real, intrinsics_real,
                            thermal_max_px=6.0)
        assert p2._assoc_params_thermal.max_px == pytest.approx(6.0)
        assert p2._assoc_params.max_px == pytest.approx(32.0)


class TestInstanceGatePipeline:
    """use_instance_masks through _apply_association: the range-attribution
    rules keep their semantics on top of the polygon gate, and absent
    masks/records degrade byte-identically to the box gate."""

    def _pipeline(self, detection, intrinsics, **assoc):
        import copy

        from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
        det = copy.deepcopy(detection)
        det["fusion"]["association"] = {
            "enabled": True, "max_px": 1e6, "min_points": 1,
            "range_agg": "min", "use_radar_range": True,
            "range_compat_ratio": 0.0, "use_instance_masks": True,
            "instance_margin_px": 0.0, **assoc}
        return ObstacleDetectionPipeline(intrinsics, det)

    @staticmethod
    def _res(coords, sizes, raws):
        from types import SimpleNamespace
        return SimpleNamespace(
            coords=coords, sizes=sizes, ranges=list(raws),
            raw_ranges=list(raws), radar_ranges=[], radar_hits=[],
            mono_ranges=[], undistorted=np.zeros((648, 864, 3), np.uint8))

    def test_polygon_excludes_foreground_end_to_end(self, detection_real,
                                                    intrinsics_real):
        from types import SimpleNamespace

        from scripts.utils.geometry import project_radar_to_undistorted
        p = self._pipeline(detection_real, intrinsics_real,
                           instance_margin_px=6.0)
        P = np.asarray(intrinsics_real["fisheye"]["K"], float)
        T = p.extrinsics["T_radar_to_fisheye"]
        # Two returns: dead-ahead on the object at 2.9 m, and foreground
        # clutter at 1.0 m projecting well to the right (inside the huge
        # padded box, outside the instance polygon).
        pts = np.array([[0.0, 2.9, 0.0], [0.6, 1.0, 0.0]])
        px = project_radar_to_undistorted(pts, P, T)
        assert np.isfinite(px).all()
        u0, v0 = px[0]
        # Instance record: a small square silhouette around the on-object
        # projection, normalised to the 864x648 undistorted frame.
        half = 12.0
        poly_px = [(u0 - half, v0 - half), (u0 + half, v0 - half),
                   (u0 + half, v0 + half), (u0 - half, v0 + half)]
        instances = [{"cls": "boat_ship", "confidence": 0.9,
                      "polygon": [c for (x, y) in poly_px
                                  for c in (x / 864.0, y / 648.0)]}]
        mm = SimpleNamespace(points_xyz=pts)
        r_inst = self._res([(int(u0), int(v0))], [40.0], [3.5])
        p._apply_association(r_inst, mm, P, T, instances=instances)
        assert r_inst.radar_gates == ["instance"]
        assert r_inst.radar_hits == [1]
        assert r_inst.radar_ranges == [pytest.approx(2.9)]
        assert r_inst.ranges == [pytest.approx(2.9)]
        # Same frame without instances: box gate sweeps up the 1.0 m
        # foreground return (min aggregation).
        r_box = self._res([(int(u0), int(v0))], [40.0], [3.5])
        p._apply_association(r_box, mm, P, T)
        assert r_box.radar_gates == ["box"]
        assert r_box.radar_ranges == [pytest.approx(1.0)]

    def test_missing_instance_root_degrades_to_box(self, detection_real,
                                                   intrinsics_real):
        # Provider root absent -> pipeline runs with the box gate, results
        # identical to use_instance_masks: false (the flag-off contract).
        rng = np.random.default_rng(7)
        fish = rng.integers(0, 255, (648, 864, 3), dtype=np.uint8)
        therm = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        pts = np.array([[0.0, 2.0, 0.0]])
        p_on = self._pipeline(detection_real, intrinsics_real,
                              instance_root="does/not/exist")
        p_off = self._pipeline(detection_real, intrinsics_real,
                               use_instance_masks=False)
        assert p_on._inst_provider is None
        a = p_on.process_frame(fish, therm, pts.copy())
        b = p_off.process_frame(fish, therm, pts.copy())
        assert a.fisheye.radar_ranges == b.fisheye.radar_ranges
        assert a.fisheye.radar_hits == b.fisheye.radar_hits
        assert a.fisheye.ranges == b.fisheye.ranges
        assert a.fusion.confirmed == b.fusion.confirmed
        assert a.fusion.min_ranges == b.fusion.min_ranges

    def test_silhouette_rule_still_wins_on_polygon_matches(
            self, detection_real, intrinsics_real):
        # Rule-interaction: a polygon-gated match on OBSTACLE seg pixels
        # keeps the silhouette override (compat gate at 0.5 would
        # otherwise reject radar 2.9 m against camera-raw 20 m).
        from types import SimpleNamespace

        from scripts.utils.geometry import project_radar_to_undistorted
        p = self._pipeline(detection_real, intrinsics_real,
                           range_compat_ratio=0.5, instance_margin_px=6.0)
        P = np.asarray(intrinsics_real["fisheye"]["K"], float)
        T = p.extrinsics["T_radar_to_fisheye"]
        pts = np.array([[0.0, 2.9, 0.0]])
        px = project_radar_to_undistorted(pts, P, T)
        u0, v0 = px[0]
        instances = [{"polygon": [c for (x, y) in
                                  ((u0 - 12, v0 - 12), (u0 + 12, v0 - 12),
                                   (u0 + 12, v0 + 12), (u0 - 12, v0 + 12))
                                  for c in (x / 864.0, y / 648.0)]}]
        r = self._res([(int(u0), int(v0))], [40.0], [20.0])
        r.seg_mask = np.zeros((648, 864), np.uint8)      # all OBSTACLE
        p._apply_association(r, SimpleNamespace(points_xyz=pts), P, T,
                             instances=instances)
        assert r.radar_gates == ["instance"]
        assert r.radar_ranges == [pytest.approx(2.9)]    # silhouette wins
        r2 = self._res([(int(u0), int(v0))], [40.0], [20.0])
        r2.seg_mask = np.full((648, 864), 1, np.uint8)   # all WATER
        p._apply_association(r2, SimpleNamespace(points_xyz=pts), P, T,
                             instances=instances)
        assert r2.radar_ranges == [None]                 # compat gate holds
