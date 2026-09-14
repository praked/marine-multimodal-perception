"""Tests for scripts/utils/segmentation.py: the water-mask consumption that
drives the robust monocular range fix (waterline-contact point + water edge)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.utils.segmentation import (
    OBSTACLE,
    SKY,
    WATER,
    ContactParams,
    SegDetectParams,
    SegProvider,
    decode_seg_array,
    horizon_from_water,
    mask_path_for_frame_id,
    seg_obstacle_detections,
    water_edge_contact,
    water_fraction,
)


# ---------------------------------------------------------------------------
# Mask decoding
# ---------------------------------------------------------------------------

def test_decode_label_encoded_passthrough():
    arr = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.uint8)
    out = decode_seg_array(arr)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, arr)


def test_decode_ignore_values_become_obstacle_neutral():
    # MaSTr ignore (4) and LaRS ignore (255) must not masquerade as water/sky.
    arr = np.array([[4, 255, 1], [2, 4, 255]], dtype=np.uint8)
    out = decode_seg_array(arr)
    assert out[0, 0] == OBSTACLE and out[0, 1] == OBSTACLE
    assert out[0, 2] == WATER and out[1, 0] == SKY


def test_decode_color_coded_to_nearest_palette():
    # eWaSR palette: obstacle [247,195,37], water [41,167,224], sky [90,75,164]
    rgb = np.array([
        [[247, 195, 37], [41, 167, 224]],
        [[90, 75, 164], [40, 170, 220]],   # last is ~water
    ], dtype=np.uint8)
    out = decode_seg_array(rgb)
    assert out[0, 0] == OBSTACLE
    assert out[0, 1] == WATER
    assert out[1, 0] == SKY
    assert out[1, 1] == WATER


# ---------------------------------------------------------------------------
# Waterline-contact point
# ---------------------------------------------------------------------------

def _scene(h=200, w=200, water_top=120, obstacle=None):
    """sky above water_top, water below; optional obstacle rect dict."""
    seg = np.full((h, w), SKY, np.uint8)
    seg[water_top:, :] = WATER
    if obstacle is not None:
        r0, r1, c0, c1 = obstacle
        seg[r0:r1, c0:c1] = OBSTACLE
    return seg


def test_contact_is_obstacle_water_boundary_not_box_bottom():
    # Obstacle rect rows [80,140) over water starting at row 120. The hull/water
    # contact is row ~139 (last obstacle row), regardless of how far the box
    # bottom extends into the water below.
    seg = _scene(obstacle=(80, 140, 90, 130))
    # Box deliberately over-captures water down to row 180.
    bbox = (90, 80, 130, 180)
    contact = water_edge_contact(seg, bbox, ContactParams(min_columns=3))
    assert contact is not None
    u, v = contact
    assert 90 <= u <= 130
    assert 135 <= v <= 141      # at the obstacle base, NOT the box bottom (180)


def test_contact_none_when_fully_water():
    seg = _scene(water_top=0)            # all water, no obstacle
    bbox = (50, 50, 100, 150)
    assert water_edge_contact(seg, bbox, ContactParams(min_columns=3)) is None


def test_contact_none_when_no_water_below_obstacle():
    # Obstacle hanging in sky with no water beneath it inside the search window.
    seg = np.full((200, 200), SKY, np.uint8)
    seg[20:60, 80:120] = OBSTACLE
    bbox = (80, 20, 120, 60)
    assert water_edge_contact(seg, bbox, ContactParams(min_columns=3,
                                                       search_pad_px=5)) is None


def test_contact_robust_to_a_few_noisy_columns():
    seg = _scene(obstacle=(80, 140, 90, 130))
    # Punch one column down deeper (a spurious obstacle finger into water).
    seg[140:170, 110] = OBSTACLE
    contact = water_edge_contact(seg, (90, 80, 130, 180),
                                 ContactParams(min_columns=3, percentile=80))
    assert contact is not None
    _, v = contact
    # 80th percentile shrugs off the single deep column.
    assert v <= 160


# ---------------------------------------------------------------------------
# Water-edge horizon
# ---------------------------------------------------------------------------

def test_horizon_flat_water_edge():
    seg = _scene(water_top=120)
    slope, intercept, conf = horizon_from_water(seg)
    assert abs(slope) < 1e-3
    assert abs(intercept - 120) < 1.5
    assert conf > 0.8


def test_horizon_tilted_water_edge_recovered():
    # Build a slanted water edge: row = 0.2*col + 60.
    h, w = 240, 320
    seg = np.full((h, w), SKY, np.uint8)
    true_slope, true_int = 0.2, 60.0
    for c in range(w):
        edge = int(true_slope * c + true_int)
        seg[edge:, c] = WATER
    slope, intercept, conf = horizon_from_water(seg)
    assert abs(slope - true_slope) < 0.03
    assert abs(intercept - true_int) < 4.0
    assert conf > 0.8


def test_horizon_no_water_zero_confidence():
    seg = np.full((100, 100), SKY, np.uint8)
    _, _, conf = horizon_from_water(seg)
    assert conf == 0.0


def test_horizon_ignores_protruding_obstacle():
    # Flat water at row 120 with a tall obstacle spiking up to row 40: the
    # obstacle column's "top water" is far above the line and must be trimmed.
    seg = _scene(water_top=120)
    seg[40:120, 150:160] = OBSTACLE
    slope, intercept, conf = horizon_from_water(seg)
    assert abs(slope) < 0.02
    assert abs(intercept - 120) < 4.0


def test_water_fraction():
    seg = _scene(water_top=100, obstacle=None)  # 100/200 rows water
    assert abs(water_fraction(seg) - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Frame-id -> mask path + provider
# ---------------------------------------------------------------------------

def test_mask_path_for_frame_id():
    fid = "Boats/2025-06-23_16-21-07/ts=16-21-08.3"
    p = mask_path_for_frame_id("data/seg", fid)
    assert p is not None
    assert p.as_posix() == "data/seg/Boats__2025-06-23_16-21-07/ts=16-21-08.3.png"


def test_mask_path_bad_frame_id_returns_none():
    assert mask_path_for_frame_id("data/seg", "garbage") is None


def test_seg_provider_get(tmp_path):
    clip = tmp_path / "Boats__2025-06-23_16-21-07"
    clip.mkdir()
    from PIL import Image
    seg = _scene(water_top=50)
    Image.fromarray(seg).save(clip / "ts=16-21-08.3.png")

    prov = SegProvider(tmp_path)
    assert prov.available()
    fid = "Boats/2025-06-23_16-21-07/ts=16-21-08.3"
    m = prov.get(fid)
    assert m is not None and m.shape == seg.shape
    # cached + missing handling
    assert prov.get(fid) is m or np.array_equal(prov.get(fid), m)
    assert prov.get("Boats/2025-06-23_16-21-07/ts=99-99-99.9") is None
    assert prov.get(None) is None


# ---------------------------------------------------------------------------
# Segmentation-driven obstacle detection
# ---------------------------------------------------------------------------

def test_seg_detect_finds_obstacle_in_water():
    seg = _scene(water_top=120, obstacle=(110, 160, 90, 130))  # box in water
    coords, sizes = seg_obstacle_detections(seg, SegDetectParams(min_area=10))
    assert len(coords) == 1
    cx, cy = coords[0]
    assert 90 <= cx <= 130 and 110 <= cy <= 160
    assert sizes[0] >= 30


def test_seg_detect_skips_background_above_water_edge():
    # An obstacle blob entirely in the sky region, well above the water edge,
    # is background (shore/sky) and should be dropped.
    seg = _scene(water_top=150)
    seg[10:40, 60:90] = OBSTACLE          # high above the edge
    coords, _ = seg_obstacle_detections(seg, SegDetectParams(min_area=10,
                                                             water_edge_margin_px=10))
    assert coords == []


def test_seg_detect_min_area_filter():
    seg = _scene(water_top=100, obstacle=(110, 113, 100, 103))  # tiny 3x3
    coords, _ = seg_obstacle_detections(seg, SegDetectParams(min_area=50))
    assert coords == []


def test_seg_detect_empty_when_no_obstacle():
    seg = _scene(water_top=100)
    assert seg_obstacle_detections(seg) == ([], [])


# ---------------------------------------------------------------------------
# Navigable free-space profile
# ---------------------------------------------------------------------------

def _free_space(seg):
    from scripts.utils.segmentation import free_space_profile
    from scripts.utils.geometry import UP_LEVEL
    P = np.array([[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1.0]])
    edges = np.arange(-55, 56, 10)
    return free_space_profile(seg, P, UP_LEVEL, camera_height_m=0.27,
                              cx=320.0, pix_deg_ratio=7.2, bin_edges=edges,
                              max_range_m=15.0)


def test_free_space_obstacle_gives_finite_limit_open_elsewhere():
    # sky above 200, water below; an obstacle straight ahead (cols 300-340)
    # sitting in the water (rows 300-360, below the cy=240 horizon).
    seg = np.full((480, 640), WATER, np.uint8)
    seg[:200, :] = SKY
    seg[300:360, 300:340] = OBSTACLE
    prof = _free_space(seg)
    centers = list(prof.bin_centers)
    c0 = centers.index(0.0)                      # bin straight ahead
    assert prof.free_dist_m[c0] is not None       # blocked by the obstacle
    assert 0.3 < prof.free_dist_m[c0] < 5.0       # a few metres ahead
    # a side bearing with only open water -> water edge is above the horizon -> open
    cside = centers.index(40.0)
    assert prof.free_dist_m[cside] is None
    assert not any(prof.blocked)


def test_free_space_blocked_when_no_near_water():
    # straight-ahead columns are obstacle all the way down -> blocked at the boat
    seg = np.full((480, 640), WATER, np.uint8)
    seg[:200, :] = SKY
    seg[200:, 300:340] = OBSTACLE
    prof = _free_space(seg)
    c0 = list(prof.bin_centers).index(0.0)
    assert prof.blocked[c0] is True
    assert prof.free_dist_m[c0] == 0.0


def test_free_space_all_water_is_open():
    seg = np.full((480, 640), WATER, np.uint8)
    prof = _free_space(seg)
    assert all(d is None for d in prof.free_dist_m)
    assert not any(prof.blocked)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Seg-based false-positive filter
# ---------------------------------------------------------------------------

def test_detection_obstacle_fraction():
    from scripts.utils.segmentation import detection_obstacle_fraction
    seg = np.full((200, 200), WATER, np.uint8)
    seg[50:100, 50:100] = OBSTACLE          # a 50x50 obstacle
    # box fully on the obstacle -> ~1.0
    assert detection_obstacle_fraction(seg, (50, 50, 100, 100)) > 0.95
    # box fully on open water -> 0.0 (the glint FP case)
    assert detection_obstacle_fraction(seg, (120, 120, 160, 160)) == 0.0
    # half/half
    f = detection_obstacle_fraction(seg, (50, 50, 150, 100))
    assert 0.3 < f < 0.7


def test_seg_fp_filter_drops_water_only_detections():
    from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
    from scripts.utils.calibration import load_detection, load_intrinsics
    seg = np.full((648, 864), WATER, np.uint8)
    seg[300:360, 400:460] = OBSTACLE        # one real obstacle
    det = {**load_detection(), "segmentation": {"enabled": True,
           "fp_filter": {"enabled": True, "min_obstacle_frac": 0.2}}}
    pipe = ObstacleDetectionPipeline(load_intrinsics(), det)
    coords = [(430, 330), (700, 500)]       # on-obstacle, open-water
    sizes = [50.0, 50.0]
    angles = [0.0, 10.0]
    kc, ks, ka = pipe._seg_fp_filter(coords, sizes, angles, seg)
    assert (430, 330) in kc                  # real obstacle kept
    assert (700, 500) not in kc              # water-only dropped
    assert len(kc) == 1


# ---------------------------------------------------------------------------
# Free-space temporal smoothing
# ---------------------------------------------------------------------------

def test_free_space_smoother_converges_to_steady():
    from scripts.utils.segmentation import FreeSpaceSmoother
    sm = FreeSpaceSmoother(alpha=0.5, max_range_m=15.0)
    prof = [5.0, None, 0.0]
    out = None
    for _ in range(20):
        out = sm.update(prof)
    assert out[0] == pytest.approx(5.0, abs=0.05)   # converges to the steady value
    assert out[1] is None                            # open stays open
    assert out[2] == pytest.approx(0.0, abs=0.05)    # blocked stays blocked


def test_free_space_smoother_damps_transient():
    from scripts.utils.segmentation import FreeSpaceSmoother
    sm = FreeSpaceSmoother(alpha=0.4, max_range_m=15.0)
    for _ in range(10):
        sm.update([None])                 # settle to open
    spike = sm.update([2.0])              # one close reading
    # damped: not instantly 2.0, somewhere between 2 and 15
    assert spike[0] is not None and 2.0 < spike[0] < 15.0


def test_free_space_smoother_alpha_one_is_passthrough():
    from scripts.utils.segmentation import FreeSpaceSmoother
    sm = FreeSpaceSmoother(alpha=1.0, max_range_m=15.0)
    assert sm.update([3.0, None])[0] == pytest.approx(3.0)
    assert sm.update([7.0, None])[0] == pytest.approx(7.0)


def test_free_space_smoother_reset():
    from scripts.utils.segmentation import FreeSpaceSmoother
    sm = FreeSpaceSmoother(alpha=0.3)
    sm.update([1.0]); sm.reset()
    # after reset, first update seeds directly (no lag from old state)
    assert sm.update([9.0])[0] == pytest.approx(9.0)
