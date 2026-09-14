"""Tests for geometry.range_from_contact_point: the explicit-contact-point
water-plane range used by the segmentation range fix."""

from __future__ import annotations

import numpy as np

from scripts.utils.geometry import (
    UP_LEVEL,
    range_from_contact_point,
    range_from_water_plane_up,
)

# A simple pinhole camera matrix for an undistorted frame.
P = np.array([[400.0, 0.0, 320.0],
              [0.0, 400.0, 240.0],
              [0.0, 0.0, 1.0]])
H = 0.27  # camera height (m), the measured Vessel A value


def _analytic_range(u, v, height=H):
    """Closed form for a level camera (up=-Y): r = t*|d|, t = h*fy/(v-cy)."""
    fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]
    d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    denom = -(v - cy) / fy
    t = -height / denom
    return float(t * np.linalg.norm(d))


def test_contact_matches_analytic_level():
    u, v = 320.0, 300.0   # straight ahead, below the horizon
    r = range_from_contact_point((u, v), P, UP_LEVEL, H)
    assert r is not None
    assert abs(r - _analytic_range(u, v)) < 1e-6


def test_above_horizon_returns_none():
    # v above the principal point (level camera) -> ray points up, no water hit.
    assert range_from_contact_point((320.0, 200.0), P, UP_LEVEL, H) is None
    # exactly on the horizon row
    assert range_from_contact_point((320.0, 240.0), P, UP_LEVEL, H) is None


def test_higher_contact_gives_larger_range():
    # The core property of the fix: a contact point HIGHER in the image (smaller
    # v, i.e. the true hull/water line) yields a LARGER range than the bbox
    # bottom that dips lower into the water (larger v -> looks closer).
    u = 320.0
    box_bottom_v = 360.0
    true_contact_v = 300.0
    r_box = range_from_contact_point((u, box_bottom_v), P, UP_LEVEL, H)
    r_contact = range_from_contact_point((u, true_contact_v), P, UP_LEVEL, H)
    assert r_box is not None and r_contact is not None
    assert r_contact > r_box


def test_consistent_with_bbox_bottom_helper():
    # range_from_contact_point at (x_center, y1) must equal
    # range_from_water_plane_up for the same box (which uses bottom-centre).
    bbox = (300.0, 250.0, 340.0, 320.0)
    u = (bbox[0] + bbox[2]) / 2.0
    v = bbox[3]
    r_contact = range_from_contact_point((u, v), P, UP_LEVEL, H)
    r_box = range_from_water_plane_up(bbox, P, UP_LEVEL, H)
    assert r_contact is not None and r_box is not None
    assert abs(r_contact - r_box) < 1e-6


def test_max_range_clip():
    # A contact just below the horizon -> huge range -> clipped to None.
    r = range_from_contact_point((320.0, 241.0), P, UP_LEVEL, H, max_range_m=15.0)
    assert r is None


def test_tilted_up_vector_changes_range():
    # With a rolled "up" vector the same pixel maps to a different range, proving
    # the attitude reference is actually applied.
    u, v = 320.0, 300.0
    up_level = UP_LEVEL
    up_tilt = np.array([np.sin(np.radians(10)), -np.cos(np.radians(10)), 0.0])
    r0 = range_from_contact_point((u, v), P, up_level, H)
    r1 = range_from_contact_point((u, v), P, up_tilt, H)
    assert r0 is not None and r1 is not None
    assert abs(r0 - r1) > 1e-3
