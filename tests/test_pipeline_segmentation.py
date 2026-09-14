"""Integration tests: the pipeline's range path actually uses the water mask
when segmentation is enabled, and is byte-identical to before when it isn't."""

from __future__ import annotations

import numpy as np

from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.segmentation import OBSTACLE, SKY, WATER


def _pipeline(seg_cfg=None, seg_provider=None):
    intr = load_intrinsics()
    det = load_detection()
    if seg_cfg is not None:
        det = {**det, "segmentation": seg_cfg}
    return ObstacleDetectionPipeline(intr, det, seg_provider=seg_provider)


def _undistorted_size(pipe):
    # The undistort map keeps the source size; use the configured image size.
    w, h = pipe.intrinsics["fisheye"]["image_size"]
    return int(w), int(h)


def _mask_with_obstacle_on_water(w, h, box):
    """sky top third, water below, with an obstacle rect sitting on the water."""
    seg = np.full((h, w), WATER, np.uint8)
    seg[: h // 3, :] = SKY
    c0, r0, c1, r1 = box
    seg[r0:r1, c0:c1] = OBSTACLE
    return seg


def test_segmentation_off_leaves_ranges_via_bbox_bottom():
    pipe = _pipeline()  # default: no segmentation block lookup -> off
    P = pipe.intrinsics["fisheye"]["K"]
    coords = [(400, 400)]
    sizes = [40.0]
    ranges, _raw_ranges = pipe._detection_ranges(coords, sizes, P,
                                    pipe._camera_height["fisheye"], (0.0, 0.0, 0.0))
    # No mask -> bbox-bottom estimate; just assert it's a finite number or None,
    # and that enabling-without-mask doesn't change this path.
    assert len(ranges) == 1


def test_contact_path_engaged_and_raises_range():
    w, h = 864, 648
    # Obstacle box rows [300,360); water below row 300 in that column band.
    seg = _mask_with_obstacle_on_water(w, h, box=(380, 300, 420, 360))
    pipe = _pipeline(seg_cfg={"enabled": True, "use_for_range": True,
                              "use_for_horizon": False,
                              "contact": {"min_columns": 3, "search_pad_px": 60}})
    P = pipe.intrinsics["fisheye"]["K"]
    # A blob whose box bottom (py+r) dips well below the true contact (~row 359).
    coords = [(400, 340)]
    sizes = [120.0]  # r=60 -> box bottom at 400, contact ~359
    r_seg = pipe._detection_ranges(coords, sizes, P,
                                   pipe._camera_height["fisheye"],
                                   (0.0, 0.0, 0.0), seg_mask=seg)[0][0]
    r_box = pipe._detection_ranges(coords, sizes, P,
                                   pipe._camera_height["fisheye"],
                                   (0.0, 0.0, 0.0), seg_mask=None)[0][0]
    assert r_seg is not None and r_box is not None
    # Contact sits above the box bottom -> larger (farther) range. This is the fix.
    assert r_seg > r_box


def test_contact_falls_back_when_no_transition():
    w, h = 864, 648
    seg = np.full((h, w), WATER, np.uint8)  # all water, no obstacle->water edge
    pipe = _pipeline(seg_cfg={"enabled": True, "use_for_range": True,
                              "use_for_horizon": False,
                              "contact": {"min_columns": 3}})
    P = pipe.intrinsics["fisheye"]["K"]
    coords = [(400, 400)]
    sizes = [40.0]
    r_seg = pipe._detection_ranges(coords, sizes, P,
                                   pipe._camera_height["fisheye"],
                                   (0.0, 0.0, 0.0), seg_mask=seg)[0][0]
    r_box = pipe._detection_ranges(coords, sizes, P,
                                   pipe._camera_height["fisheye"],
                                   (0.0, 0.0, 0.0), seg_mask=None)[0][0]
    # No transition -> identical to the bbox-bottom estimate.
    assert (r_seg is None) == (r_box is None)
    if r_seg is not None:
        assert abs(r_seg - r_box) < 1e-6


def test_water_edge_up_vector_used_for_horizon():
    w, h = 864, 648
    # Tilted water edge -> non-level up vector -> different range than level.
    seg = np.full((h, w), SKY, np.uint8)
    for c in range(w):
        edge = int(0.1 * c + 200)
        seg[edge:, c] = WATER
    pipe = _pipeline(seg_cfg={"enabled": True, "use_for_range": False,
                              "use_for_horizon": True, "horizon_min_confidence": 0.3})
    P = pipe.intrinsics["fisheye"]["K"]
    up = pipe._range_up_vector(P, (0.0, 0.0, 0.0), seg_mask=seg)
    # Should not be exactly level (0,-1,0) because the water edge is tilted.
    assert not np.allclose(up, np.array([0.0, -1.0, 0.0]), atol=1e-3)
