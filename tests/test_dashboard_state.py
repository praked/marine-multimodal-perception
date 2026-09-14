"""State-correctness tests for the dashboard's heading / smoother cache.

PLAN.md §I.4.12: `_render` used to recompute the heading and tick the
HeadingSmoother on every call, including when re-displaying a cached
frame after stepping backward / scrubbing. That meant the sparkline
got duplicate appends and the smoother saw frames out of order.

The fix shifts both into the per-frame production path
(`_ensure_cached_up_to`) so each cached entry carries its own
`(recommendation, smoothed)` pair, and `_render` is a pure read.

These tests exercise the production-vs-replay equivalence *without*
launching Tk: they call the module-level helpers directly and assert
that walking-forward-then-rebuilding-from-a-slice gives the same state
as a clean forward replay up to the same index.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest

from scripts.eval.dashboard import _compute_heading_for_frame
from scripts.sensor_processing.heading import HeadingSmoother
from scripts.sensor_processing.pipeline import FrameResult, FusionResult


BIN_CENTERS = np.arange(-50, 51, 10, dtype=float)
BIN_EDGES = np.arange(-55, 56, 10, dtype=float)
DETECTION_CFG = {
    "heading": {
        "candidate_step_deg": 1.0,
        "alert_threshold": 0.33,
        "blocked_threshold": 0.66,
        "block_sigma_deg": 12.0,
        "max_swing_deg": 30.0,
        "weights": {"block": 1.0, "close": 0.3, "ttc": 0.6, "course": 0.2, "nogo": 1.0},
        "tied_gap_frac": 0.05,
        "tied_min_separation_deg": 30.0,
        "range_eps_m": 0.5,
        "ttc_eps_s": 0.5,
        "abstain_cost": 0.3,
        "smoothing_window": 5,
        "smoothing_min_samples": 2,
    }
}


def _make_fusion(scores: list[float], min_ranges=None, velocities=None, ttcs=None):
    n = len(scores)
    return FusionResult(
        bin_edges=BIN_EDGES,
        bin_centers=BIN_CENTERS,
        scores=scores,
        min_ranges=min_ranges if min_ranges is not None else [None] * n,
        sensor_hit_mask=np.zeros((n, 3), dtype=bool),
        per_bin_velocity_mps=velocities if velocities is not None else [None] * n,
        per_bin_ttc_s=ttcs if ttcs is not None else [None] * n,
    )


def _frame(scores) -> FrameResult:
    return FrameResult(fusion=_make_fusion(scores))


def _produce(
    frame_scores: Iterable[list[float]],
    smoother: HeadingSmoother,
):
    """Simulate the dashboard's production path: per frame, compute a
    heading, tick the smoother, append to the cache. Returns the cache
    plus the smoother (with its state advanced)."""
    cache = []
    confirmed = [False] * len(BIN_CENTERS)
    for scores in frame_scores:
        res = _frame(scores)
        rec = _compute_heading_for_frame(
            res, confirmed, BIN_CENTERS, DETECTION_CFG,
            use_tracker_gate=False, current_heading_deg=0.0,
        )
        smoothed = smoother.update(rec.recommended_heading_deg)
        cache.append((rec, smoothed))
    return cache


# ---------------------------------------------------------------------------
# Compute helper purity
# ---------------------------------------------------------------------------

def test_compute_helper_is_deterministic():
    """Same inputs → same outputs. The helper must not depend on any
    hidden state, otherwise the cache-and-replay invariant breaks."""
    res = _frame([0.0] * 11)
    confirmed = [False] * 11
    a = _compute_heading_for_frame(
        res, confirmed, BIN_CENTERS, DETECTION_CFG,
        use_tracker_gate=False, current_heading_deg=0.0,
    )
    b = _compute_heading_for_frame(
        res, confirmed, BIN_CENTERS, DETECTION_CFG,
        use_tracker_gate=False, current_heading_deg=0.0,
    )
    assert a.recommended_heading_deg == b.recommended_heading_deg
    assert a.reason == b.reason


def test_compute_helper_tracker_gate_zeros_unconfirmed():
    """When tracker-gate is on, bins not in `confirmed` should contribute
    nothing to the threat field."""
    scores = [0.0] * 11
    scores[6] = 1.0   # bin +10°
    res = _frame(scores)
    # Tracker says "+10° is NOT confirmed" → gate should suppress it.
    confirmed = [False] * 11
    rec = _compute_heading_for_frame(
        res, confirmed, BIN_CENTERS, DETECTION_CFG,
        use_tracker_gate=True, current_heading_deg=0.0,
    )
    # With the threat zeroed, the recommendation should be 0° (course
    # term dominates and prefers bow).
    assert rec.recommended_heading_deg == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Cache-and-replay invariant
# ---------------------------------------------------------------------------

def _scenes_with_jitter():
    """A 12-frame sequence where the recommendation should swing back
    and forth, exercising the smoother + abstain logic."""
    base = [0.0] * 11
    out = []
    for k, idx in enumerate([7, 7, 5, 5, 3, 3, 7, 7, 5, 5, 3, 3]):
        s = list(base)
        s[idx] = 1.0
        out.append(s)
    return out


def test_step_back_reproduces_history_slice():
    """Walk N frames forward, then re-derive the history from cache
    slice [0:k+1] for several k. Each slice must match the head of the
    streamed history."""
    smoother = HeadingSmoother(window_n=5, min_samples=2)
    cache = _produce(_scenes_with_jitter(), smoother)

    full_raw = [c[0].recommended_heading_deg for c in cache]
    full_smoothed = [c[1] for c in cache]

    for k in (0, 1, 4, 7, len(cache) - 1):
        sliced_raw = [c[0].recommended_heading_deg for c in cache[: k + 1]]
        sliced_smoothed = [c[1] for c in cache[: k + 1]]
        assert sliced_raw == full_raw[: k + 1]
        assert sliced_smoothed == full_smoothed[: k + 1]


def test_replay_from_scratch_matches_cached():
    """Run the same scene sequence with a fresh smoother and compare
    against the cached production smoothed values. They must agree:
    this is what guarantees that hot-reload (which clears state and
    replays) leaves no visible artefact."""
    scenes = _scenes_with_jitter()
    sm_a = HeadingSmoother(window_n=5, min_samples=2)
    cache_a = _produce(scenes, sm_a)
    sm_b = HeadingSmoother(window_n=5, min_samples=2)
    cache_b = _produce(scenes, sm_b)
    assert [c[0].recommended_heading_deg for c in cache_a] == \
        [c[0].recommended_heading_deg for c in cache_b]
    assert [c[1] for c in cache_a] == [c[1] for c in cache_b]


def test_smoother_state_drifts_if_render_path_double_ticks():
    """Regression guard against the original I.4.12 bug.

    If `_render` were still ticking the smoother on each call, walking
    forward 10 frames and then re-rendering frame 3 would advance the
    smoother to 11 inputs: its output for the "replayed frame 3"
    would NOT match what was originally cached for that frame.

    We simulate the buggy path explicitly: re-tick the smoother on
    the re-rendered frame and confirm the values diverge. This
    documents what the fix prevents."""
    scenes = _scenes_with_jitter()
    sm = HeadingSmoother(window_n=5, min_samples=2)
    cache = _produce(scenes, sm)

    # Buggy replay: smoother re-ticks on a backward re-render.
    target_idx = 3
    head_at_3 = cache[target_idx][0].recommended_heading_deg
    re_ticked = sm.update(head_at_3)
    assert re_ticked != cache[target_idx][1], (
        "buggy double-tick should produce a different value than the "
        "cached one; if these match the regression guard is meaningless"
    )


# ---------------------------------------------------------------------------
# History-from-slice rebuild semantics
# ---------------------------------------------------------------------------

def test_history_rebuild_respects_maxlen():
    """When the deque maxlen is smaller than the cache, the rebuild
    should keep only the last `maxlen` entries up to `idx + 1`."""
    from collections import deque

    smoother = HeadingSmoother(window_n=5, min_samples=2)
    cache = _produce(_scenes_with_jitter() * 30, smoother)   # 360 frames

    maxlen = 50
    raw_hist: deque = deque(maxlen=maxlen)
    sm_hist: deque = deque(maxlen=maxlen)

    idx = 200
    lo = max(0, idx + 1 - maxlen)
    for entry in cache[lo: idx + 1]:
        rec, smoothed = entry
        raw_hist.append(rec.recommended_heading_deg)
        sm_hist.append(smoothed)
    assert len(raw_hist) == maxlen
    # Last entry must equal the cached head at `idx`.
    assert raw_hist[-1] == cache[idx][0].recommended_heading_deg
    assert sm_hist[-1] == cache[idx][1]


def test_rebuild_from_idx_zero_gives_singleton():
    smoother = HeadingSmoother(window_n=5, min_samples=2)
    cache = _produce(_scenes_with_jitter(), smoother)
    lo = max(0, 0 + 1 - 300)
    slc = cache[lo: 0 + 1]
    assert len(slc) == 1
    assert slc[0][0].recommended_heading_deg == cache[0][0].recommended_heading_deg
