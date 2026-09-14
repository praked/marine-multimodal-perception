import numpy as np
import pytest

from scripts.sensor_processing.heading import (
    HeadingRecommendation,
    HeadingSmoother,
    _course_cost,
    _free_space_amps,
    _gaussian_field_cost,
    _is_ambiguous,
    recommend_from_fusion,
    recommend_heading,
    rolling_average_headings,
)
from scripts.sensor_processing.pipeline import FusionResult


# Standard bin layout used across tests.
BIN_CENTERS = list(range(-50, 51, 10))      # -50..+50
N_BINS = len(BIN_CENTERS)

BASE_CONFIG = {
    "candidate_step_deg": 1.0,
    "blocked_threshold": 0.66,
    "block_sigma_deg": 12.0,
    "max_swing_deg": 30.0,
    "weights": {"block": 1.0, "close": 0.3, "ttc": 0.6, "course": 0.2, "nogo": 1.0},
    "tied_gap_frac": 0.05,
    "tied_min_separation_deg": 30.0,
    "range_eps_m": 0.5,
    "ttc_eps_s": 0.5,
}


def _empty_bins() -> dict:
    return {
        "scores": [0.0] * N_BINS,
        "min_range_m": [None] * N_BINS,
        "per_bin_velocity_mps": [None] * N_BINS,
        "per_bin_ttc_s": [None] * N_BINS,
    }


# ---------------------------------------------------------------------------
# Empty / no-obstacle scenes
# ---------------------------------------------------------------------------

def test_empty_inputs_returns_none():
    rec = recommend_heading([], [], [], [], [], BASE_CONFIG, current_heading_deg=0.0)
    assert rec.recommended_heading_deg is None
    assert rec.reason == "ambiguous"


def test_clear_scene_picks_zero_with_current_heading_zero():
    """No blocked bins, current heading = 0°. Course term is the only
    non-zero one, minimised at θ = 0."""
    bins = _empty_bins()
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=0.0)
    assert rec.recommended_heading_deg == pytest.approx(0.0)
    assert rec.reason in ("course", "clear")


# ---------------------------------------------------------------------------
# Single-obstacle scenes
# ---------------------------------------------------------------------------

def test_blocked_centre_steers_off_centre():
    """A blocked bin straight ahead with the boat going forward: the
    recommendation must swing away from 0°."""
    bins = _empty_bins()
    centre_idx = BIN_CENTERS.index(0)
    bins["scores"][centre_idx] = 1.0
    bins["min_range_m"][centre_idx] = 3.0
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=0.0)
    # Symmetric scene: should abstain (tied alternatives on left/right).
    assert rec.recommended_heading_deg is None
    assert rec.reason == "ambiguous"


def test_blocked_starboard_steers_port():
    """A blocked bin on the right should push the recommendation left."""
    bins = _empty_bins()
    bins["scores"][BIN_CENTERS.index(20)] = 1.0
    bins["min_range_m"][BIN_CENTERS.index(20)] = 3.0
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=0.0)
    assert rec.recommended_heading_deg is not None
    assert rec.recommended_heading_deg < 0


def test_blocked_port_steers_starboard():
    bins = _empty_bins()
    bins["scores"][BIN_CENTERS.index(-20)] = 1.0
    bins["min_range_m"][BIN_CENTERS.index(-20)] = 3.0
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=0.0)
    assert rec.recommended_heading_deg is not None
    assert rec.recommended_heading_deg > 0


# ---------------------------------------------------------------------------
# Velocity / TTC influence
# ---------------------------------------------------------------------------

def test_low_ttc_outweighs_high_ttc():
    """Two blocked bins at +20° and -20°, but the +20° one is closing
    fast (TTC = 1 s) while -20° is closing slow (TTC = 30 s). The
    recommendation should run from the threat: i.e. left."""
    bins = _empty_bins()
    pos_idx = BIN_CENTERS.index(20)
    neg_idx = BIN_CENTERS.index(-20)
    bins["scores"][pos_idx] = 0.66
    bins["scores"][neg_idx] = 0.66
    bins["min_range_m"][pos_idx] = 3.0
    bins["min_range_m"][neg_idx] = 3.0
    bins["per_bin_ttc_s"][pos_idx] = 1.0
    bins["per_bin_ttc_s"][neg_idx] = 30.0
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=0.0)
    assert rec.recommended_heading_deg is not None
    assert rec.recommended_heading_deg < 0


def test_squeeze_picks_gap():
    """Two blockers at ±30°, gap in the middle. Course term keeps
    recommendation near 0°."""
    bins = _empty_bins()
    for b in (30, -30):
        bins["scores"][BIN_CENTERS.index(b)] = 1.0
        bins["min_range_m"][BIN_CENTERS.index(b)] = 4.0
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=0.0)
    # Either dead centre or a small swing: but must not pick beyond ±30°.
    assert rec.recommended_heading_deg is not None
    assert abs(rec.recommended_heading_deg) < 20


# ---------------------------------------------------------------------------
# Course-change influence
# ---------------------------------------------------------------------------

def test_course_term_biases_toward_current_heading():
    """No blockers, no ranges, only course term active. Heading at 20°
    means the cheapest candidate is 20°."""
    bins = _empty_bins()
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=20.0)
    assert rec.recommended_heading_deg == pytest.approx(20.0)


def test_course_term_defaults_to_zero_when_heading_unknown():
    bins = _empty_bins()
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=None)
    assert rec.recommended_heading_deg == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tied-alternatives guard
# ---------------------------------------------------------------------------

def test_symmetric_centre_blocker_is_ambiguous():
    bins = _empty_bins()
    bins["scores"][BIN_CENTERS.index(0)] = 1.0
    bins["min_range_m"][BIN_CENTERS.index(0)] = 3.0
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=0.0)
    assert rec.recommended_heading_deg is None
    assert rec.reason == "ambiguous"


def test_clear_does_not_trigger_ambiguous():
    """All-zero cost is flat but not 'ambiguous'; that's a valid
    'go straight' answer."""
    bins = _empty_bins()
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=0.0)
    assert rec.recommended_heading_deg is not None


# ---------------------------------------------------------------------------
# Abstain-when-no-safe-option guard
# ---------------------------------------------------------------------------

def test_abstain_when_everywhere_blocked():
    """Entire FOV blocked → recommender should not pin to the FOV
    edge; it should abstain and let the autopilot hold course."""
    bins = _empty_bins()
    for b in BIN_CENTERS:
        bins["scores"][BIN_CENTERS.index(b)] = 1.0
    cfg = {**BASE_CONFIG, "abstain_cost": 0.3}
    rec = recommend_heading(BIN_CENTERS, **bins, config=cfg, current_heading_deg=0.0)
    assert rec.recommended_heading_deg is None
    assert rec.reason == "all_blocked"


def test_no_abstain_when_disabled():
    """Same wall of blockers, but with abstain disabled; recommender
    is forced to pick *something*. Default config triggers the
    tied-alternatives guard on symmetric block, so use an asymmetric
    pattern that has a clearly cheapest single direction."""
    bins = _empty_bins()
    for b in (-30, -20, -10, 0, 10):   # asymmetric: right side clearer
        bins["scores"][BIN_CENTERS.index(b)] = 1.0
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=0.0)
    assert rec.recommended_heading_deg is not None
    assert rec.recommended_heading_deg > 0   # swing starboard, away from blockers


def test_graded_amplitude_prefers_clearer_path_over_weak_blocker():
    """A weak (0.33) bin at +20° should still bias the recommendation
    leftward, but a strong (1.0) bin at the same spot should bias it
    much more strongly. The graded amplitude bakes this into the cost."""
    bins_weak = _empty_bins()
    bins_weak["scores"][BIN_CENTERS.index(20)] = 0.33
    rec_weak = recommend_heading(BIN_CENTERS, **bins_weak, config=BASE_CONFIG, current_heading_deg=0.0)
    bins_strong = _empty_bins()
    bins_strong["scores"][BIN_CENTERS.index(20)] = 1.0
    rec_strong = recommend_heading(BIN_CENTERS, **bins_strong, config=BASE_CONFIG, current_heading_deg=0.0)
    # Both should steer left, strong more so.
    assert rec_weak.recommended_heading_deg <= 0
    assert rec_strong.recommended_heading_deg < rec_weak.recommended_heading_deg


# ---------------------------------------------------------------------------
# FusionResult adapter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HeadingSmoother
# ---------------------------------------------------------------------------

def test_smoother_returns_none_before_min_samples():
    sm = HeadingSmoother(window_n=5, min_samples=3)
    assert sm.update(10.0) is None
    assert sm.update(12.0) is None
    assert sm.update(14.0) == pytest.approx((10 + 12 + 14) / 3)


def test_smoother_rolling_window_drops_oldest():
    sm = HeadingSmoother(window_n=3, min_samples=1)
    sm.update(0)
    sm.update(10)
    sm.update(20)
    assert sm.update(30) == pytest.approx((10 + 20 + 30) / 3)


def test_smoother_holds_last_value_through_none():
    sm = HeadingSmoother(window_n=3, min_samples=2)
    sm.update(10)
    last = sm.update(20)            # 15
    # None update: should not change the smoothed value.
    held = sm.update(None)
    assert held == pytest.approx(last)
    # Following non-None continues smoothing as if None never happened.
    assert sm.update(30) == pytest.approx((10 + 20 + 30) / 3)


def test_smoother_reset_clears_history():
    sm = HeadingSmoother(window_n=3, min_samples=1)
    sm.update(50.0)
    sm.reset()
    assert sm.smoothed is None
    assert sm.update(0.0) == pytest.approx(0.0)


def test_smoother_invalid_config():
    with pytest.raises(ValueError):
        HeadingSmoother(window_n=0)
    with pytest.raises(ValueError):
        HeadingSmoother(window_n=3, min_samples=4)


def test_rolling_average_batch_matches_streaming():
    headings = [10.0, 20.0, None, 30.0, 40.0]
    batch = rolling_average_headings(headings, window_n=3, min_samples=2)
    sm = HeadingSmoother(window_n=3, min_samples=2)
    streamed = [sm.update(h) for h in headings]
    assert batch == streamed


# ---------------------------------------------------------------------------
# FusionResult adapter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Private cost-term edge cases
# ---------------------------------------------------------------------------

def test_gaussian_field_cost_zero_sigma_returns_zero():
    """sigma <= 0 short-circuits to 0.0 (line 126)."""
    assert _gaussian_field_cost(0.0, [0.0], [1.0], sigma_deg=0.0) == 0.0


def test_gaussian_field_cost_no_amplitudes_returns_zero():
    """Empty amplitudes also short-circuits to 0.0 (line 126)."""
    assert _gaussian_field_cost(0.0, [], [], sigma_deg=12.0) == 0.0


def test_course_cost_zero_max_swing_returns_zero():
    """max_swing <= 0 short-circuits to 0.0 (line 142)."""
    assert _course_cost(30.0, current_heading_deg=0.0, max_swing_deg=0.0) == 0.0


def test_is_ambiguous_flat_zero_curve_returns_false():
    """An all-zero cost curve is flat (no preference) and must not be
    flagged ambiguous (line 293)."""
    cost_curve = [0.0, 0.0, 0.0]
    candidates = [-50.0, 0.0, 50.0]
    assert _is_ambiguous(cost_curve, candidates, best_i=1,
                         tied_gap_frac=0.05, tied_min_sep=30.0) is False


def test_recommend_from_fusion_adapter():
    edges = np.arange(-55, 56, 10, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    fr = FusionResult(
        bin_edges=edges,
        bin_centers=centers,
        scores=[0.0] * 11,
        min_ranges=[None] * 11,
        sensor_hit_mask=np.zeros((11, 3), dtype=bool),
        per_bin_velocity_mps=[None] * 11,
        per_bin_ttc_s=[None] * 11,
    )
    rec = recommend_from_fusion(fr, BASE_CONFIG, current_heading_deg=10.0)
    assert isinstance(rec, HeadingRecommendation)
    assert rec.recommended_heading_deg == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Segmentation free-space integration (Phase 1)
# ---------------------------------------------------------------------------

def test_free_space_none_is_noop():
    """free_dist_m=None must leave the recommendation byte-identical."""
    bins = _empty_bins()
    bins["scores"][BIN_CENTERS.index(30)] = 1.0
    bins["min_range_m"][BIN_CENTERS.index(30)] = 3.0
    base = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=0.0)
    same = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG,
                             current_heading_deg=0.0, free_dist_m=None)
    assert base.recommended_heading_deg == same.recommended_heading_deg
    assert base.cost_curve == same.cost_curve


def test_free_space_close_starboard_steers_port():
    """No fused threat, but a close navigable limit to starboard (+30°) from the
    water mask must push the recommendation to port."""
    bins = _empty_bins()
    free = [None] * N_BINS
    free[BIN_CENTERS.index(30)] = 1.0          # ~1 m navigable limit to starboard
    rec = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG,
                            current_heading_deg=0.0, free_dist_m=free)
    assert rec.recommended_heading_deg is not None
    assert rec.recommended_heading_deg < 0     # steer away from the starboard limit


def test_free_space_open_everywhere_is_noop():
    bins = _empty_bins()
    base = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG, current_heading_deg=0.0)
    openf = recommend_heading(BIN_CENTERS, **bins, config=BASE_CONFIG,
                              current_heading_deg=0.0, free_dist_m=[None] * N_BINS)
    assert base.cost_curve == openf.cost_curve


def test_free_space_zero_max_range_no_crash():
    """`free_space_max_range_m: 0` (or negative) in config must not
    ZeroDivisionError; a 0 m (blocked) bin still reads as full threat and
    finite distances clamp to no-threat."""
    for bad_max in (0.0, -5.0):
        amps = _free_space_amps([None, 0.0, 1.0], bad_max)
        assert amps[0] == 0.0          # open water stays no-threat
        assert amps[1] == 1.0          # hard-blocked bin stays full threat
        assert 0.0 <= amps[2] <= 1.0   # finite distance stays in range

    bins = _empty_bins()
    free = [None] * N_BINS
    free[BIN_CENTERS.index(30)] = 0.0  # blocked to starboard
    cfg = dict(BASE_CONFIG, free_space_max_range_m=0.0)
    rec = recommend_heading(BIN_CENTERS, **bins, config=cfg,
                            current_heading_deg=0.0, free_dist_m=free)
    assert rec.recommended_heading_deg is not None
    assert rec.recommended_heading_deg < 0     # still steers away, no crash


def test_free_space_threads_through_fusion_adapter():
    fr = FusionResult(
        bin_edges=np.array(BIN_CENTERS + [60]), bin_centers=np.array(BIN_CENTERS),
        scores=[0.0] * N_BINS, min_ranges=[None] * N_BINS,
        sensor_hit_mask=np.zeros((N_BINS, 3), bool),
        per_bin_velocity_mps=[None] * N_BINS, per_bin_ttc_s=[None] * N_BINS,
    )
    free = [None] * N_BINS
    free[BIN_CENTERS.index(-30)] = 1.0         # close limit to port
    rec = recommend_from_fusion(fr, BASE_CONFIG, current_heading_deg=0.0, free_dist_m=free)
    assert rec.recommended_heading_deg is not None
    assert rec.recommended_heading_deg > 0     # steer starboard, away from the port limit
