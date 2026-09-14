import numpy as np
import pytest

from scripts.sensor_processing.motion import (
    RadarPointTracker,
    RadarPointTrackerConfig,
    aggregate_bin_velocity,
    bbox_bearing_rate_deg_per_s,
    closing_speed,
    parse_radar_timestamp,
    radar_velocity_from_doppler,
    timestamp_delta_s,
    ttc_from_velocity,
)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def test_parse_radar_timestamp_basic():
    assert parse_radar_timestamp("00:00:00.0") == pytest.approx(0.0)
    assert parse_radar_timestamp("01:02:03.4") == pytest.approx(3723.4)


def test_parse_radar_timestamp_invalid():
    assert parse_radar_timestamp(None) is None
    assert parse_radar_timestamp("garbage") is None


def test_timestamp_delta_positive():
    assert timestamp_delta_s("00:00:00.5", "00:00:00.3") == pytest.approx(0.2)


def test_timestamp_delta_non_positive_returns_none():
    # Same timestamp = no progress.
    assert timestamp_delta_s("00:00:00.5", "00:00:00.5") is None
    # Going backwards = clock weird; bail.
    assert timestamp_delta_s("00:00:00.3", "00:00:00.5") is None


def test_timestamp_delta_too_large_returns_none():
    # >5 s gap means we treat it as a fresh start, not a delta.
    assert timestamp_delta_s("00:00:10.0", "00:00:00.0") is None


def test_timestamp_delta_invalid_input_returns_none():
    # An unparseable timestamp on either side yields None (line 60).
    assert timestamp_delta_s("garbage", "00:00:00.0") is None
    assert timestamp_delta_s("00:00:00.0", None) is None


def test_timestamp_delta_midnight_rollover():
    # prev just before midnight, curr just after -> rollover correction (line 67).
    dt = timestamp_delta_s("00:00:00.0", "23:59:59.9")
    assert dt == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# RadarPointTracker
# ---------------------------------------------------------------------------

def test_tracker_first_call_returns_none():
    tr = RadarPointTracker()
    pts = np.array([[0.0, 2.0, 0.0]])
    out = tr.update(pts, "00:00:00.0")
    assert out == [None]


def test_tracker_matches_within_gate():
    """A single point translating by (0, -0.2) over 0.1 s should
    report velocity ~(0, -2.0) m/s."""
    tr = RadarPointTracker(RadarPointTrackerConfig(gate_m=1.0))
    tr.update(np.array([[0.0, 2.0, 0.0]]), "00:00:00.0")
    out = tr.update(np.array([[0.0, 1.8, 0.0]]), "00:00:00.1")
    assert out[0] is not None
    vx, vy = out[0]
    assert vx == pytest.approx(0.0, abs=0.01)
    assert vy == pytest.approx(-2.0, abs=0.01)


def test_tracker_unmatched_beyond_gate():
    tr = RadarPointTracker(RadarPointTrackerConfig(gate_m=0.1))
    tr.update(np.array([[0.0, 2.0, 0.0]]), "00:00:00.0")
    out = tr.update(np.array([[0.0, 1.0, 0.0]]), "00:00:00.1")
    assert out == [None]


def test_tracker_resets_after_stale_gap():
    tr = RadarPointTracker()
    tr.update(np.array([[0.0, 2.0, 0.0]]), "00:00:00.0")
    # Gap exceeds max_dt_s; should reset.
    out = tr.update(np.array([[0.0, 1.9, 0.0]]), "00:00:10.0")
    assert out == [None]


def test_tracker_greedy_pairs_closest_first():
    """Two prior points + two current points: confirm the closer
    pair gets matched even when the order is adversarial."""
    tr = RadarPointTracker(RadarPointTrackerConfig(gate_m=1.0))
    tr.update(np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]), "00:00:00.0")
    # New points: (0, 1.9) closest to prev (0, 2.0); (1, 1.9) closest to (1, 2.0).
    out = tr.update(np.array([[1.0, 1.9, 0.0], [0.0, 1.9, 0.0]]), "00:00:00.1")
    assert out[0] is not None and out[1] is not None
    # Both should report ~ (0, -1) m/s (no x-translation, -0.1/0.1 in y).
    assert out[0][1] == pytest.approx(-1.0, abs=0.05)
    assert out[1][1] == pytest.approx(-1.0, abs=0.05)


def test_tracker_handles_empty_frame():
    tr = RadarPointTracker()
    out = tr.update(np.empty((0, 3)), "00:00:00.0")
    assert out == []
    # Following frame should also be empty since prev was empty.
    out2 = tr.update(np.array([[0.0, 1.0, 0.0]]), "00:00:00.1")
    assert out2 == [None]


def test_tracker_reset_clears_prior_frame():
    """reset() drops the cached prior frame so the next update is a fresh
    start (lines 103-104)."""
    tr = RadarPointTracker(RadarPointTrackerConfig(gate_m=1.0))
    tr.update(np.array([[0.0, 2.0, 0.0]]), "00:00:00.0")
    tr.reset()
    # Without reset this would match; after reset it returns None (no prior).
    out = tr.update(np.array([[0.0, 1.9, 0.0]]), "00:00:00.1")
    assert out == [None]


# ---------------------------------------------------------------------------
# Doppler entry point
# ---------------------------------------------------------------------------

def test_radar_velocity_from_doppler_aligns_with_bearing():
    # A point straight ahead with v_radial = -1 m/s should split to (0, -1).
    pts = np.array([[0.0, 2.0, 0.0]])
    out = radar_velocity_from_doppler(pts, np.array([-1.0]))
    assert out[0][0] == pytest.approx(0.0)
    assert out[0][1] == pytest.approx(-1.0)


def test_radar_velocity_from_doppler_handles_missing():
    pts = np.array([[0.0, 2.0, 0.0]])
    assert radar_velocity_from_doppler(pts, None) == [None]
    assert radar_velocity_from_doppler(pts, np.array([np.nan])) == [None]


def test_radar_velocity_from_doppler_origin_point():
    """A point at the radar origin (r == 0) returns the radial speed straight
    along Y with no tangential split (lines 199-200)."""
    pts = np.array([[0.0, 0.0, 0.0]])
    out = radar_velocity_from_doppler(pts, np.array([1.5]))
    assert out[0] == (0.0, 1.5)


# ---------------------------------------------------------------------------
# Bin aggregation + TTC
# ---------------------------------------------------------------------------

def test_aggregate_bin_velocity_picks_median_magnitude():
    # Two points dead ahead (bin 5 in the standard layout) with closing
    # speeds 0.5 and 2.0; magnitude median = 2.0 (ranked higher).
    pts = np.array([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]])
    vels = [(0.0, -0.5), (0.0, -2.0)]
    edges = np.arange(-55, 56, 10)
    out = aggregate_bin_velocity(pts, vels, edges)
    centre = 5
    # Magnitude-weighted median over 2 entries picks the larger half.
    assert out[centre] == pytest.approx(2.0)


def test_aggregate_bin_velocity_skips_none_velocities():
    pts = np.array([[0.0, 2.0, 0.0]])
    edges = np.arange(-55, 56, 10)
    out = aggregate_bin_velocity(pts, [None], edges)
    assert all(v is None for v in out)


def test_aggregate_bin_velocity_skips_zero_range_point():
    """A point with Y == 0 is skipped (line 246) so its bin stays None."""
    pts = np.array([[0.0, 0.0, 0.0]])
    edges = np.arange(-55, 56, 10)
    out = aggregate_bin_velocity(pts, [(0.0, -1.0)], edges)
    assert all(v is None for v in out)


def test_aggregate_bin_velocity_skips_out_of_range_bearing():
    """A point whose azimuth falls outside [bin_edges[0], bin_edges[-1]) is
    dropped (line 249). Here the point is at +90 deg (X large, Y small)."""
    pts = np.array([[100.0, 1.0, 0.0]])  # theta ~= +89.4 deg, beyond +55 edge
    edges = np.arange(-55, 56, 10)
    out = aggregate_bin_velocity(pts, [(0.0, -1.0)], edges)
    assert all(v is None for v in out)


def test_closing_speed_sign_convention():
    # vy < 0 = approaching boat (Y decreasing = getting closer)
    assert closing_speed((0.0, -1.0)) == 1.0
    assert closing_speed((0.0, 1.0)) == -1.0
    assert closing_speed(None) is None


def test_ttc_basic():
    # 2 m away, closing at 1 m/s → TTC = 2 s.
    assert ttc_from_velocity(2.0, 1.0) == pytest.approx(2.0)


def test_ttc_static_returns_none():
    # No closing motion → TTC undefined.
    assert ttc_from_velocity(2.0, 0.01) is None
    assert ttc_from_velocity(2.0, -1.0) is None


def test_ttc_missing_inputs_returns_none():
    assert ttc_from_velocity(None, 1.0) is None
    assert ttc_from_velocity(2.0, None) is None


def test_ttc_caps_at_60s():
    # 2 m, 0.01 m/s → would be 200 s without the cap and eps gate.
    # Above eps but past cap.
    assert ttc_from_velocity(2.0, 0.06, cap_s=60.0) == pytest.approx(min(2.0 / 0.06, 60.0))


# ---------------------------------------------------------------------------
# Bbox bearing-rate (sanity helper)
# ---------------------------------------------------------------------------

def test_bbox_bearing_rate_constant_motion():
    # Centroid moves from x=400 to x=472 over 1 s. With cx=400 and
    # pix_deg_ratio=7.2, bearing goes from 0 to 10 deg → rate = 10 deg/s.
    rate = bbox_bearing_rate_deg_per_s(
        centroid_history=[(400, 200), (472, 200)],
        pix_deg_ratio=7.2,
        cx=400,
        timestamps_s=[0.0, 1.0],
    )
    assert rate == pytest.approx(10.0)


def test_bbox_bearing_rate_too_few_samples():
    rate = bbox_bearing_rate_deg_per_s(
        centroid_history=[(400, 200)],
        pix_deg_ratio=7.2,
        cx=400,
    )
    assert rate is None


def test_bbox_bearing_rate_default_timestamps():
    """No timestamps -> assume uniform 3 fps spacing (line 315). Two samples
    are 1/3 s apart; a 10 deg bearing change gives 30 deg/s."""
    rate = bbox_bearing_rate_deg_per_s(
        centroid_history=[(400, 200), (472, 200)],
        pix_deg_ratio=7.2,
        cx=400,
    )
    assert rate == pytest.approx(30.0)


def test_bbox_bearing_rate_mismatched_timestamps_returns_none():
    """timestamps_s length != centroid_history length -> None (line 317)."""
    rate = bbox_bearing_rate_deg_per_s(
        centroid_history=[(400, 200), (472, 200)],
        pix_deg_ratio=7.2,
        cx=400,
        timestamps_s=[0.0],  # too short
    )
    assert rate is None


def test_bbox_bearing_rate_non_positive_dt_returns_none():
    """Non-increasing timestamps (dt <= 0) -> None (line 322)."""
    rate = bbox_bearing_rate_deg_per_s(
        centroid_history=[(400, 200), (472, 200)],
        pix_deg_ratio=7.2,
        cx=400,
        timestamps_s=[1.0, 1.0],  # dt == 0
    )
    assert rate is None
