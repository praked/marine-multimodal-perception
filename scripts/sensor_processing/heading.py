"""Heading recommendation: Phase I.4.10.

Pure function over the per-frame fusion + motion output. Sweeps
candidate headings across the usable FOV at 1° resolution, evaluates a
small bag of cost terms, and returns the argmin plus the full cost
curve so the dashboard and downstream consumers can audit the decision.

Design contract:

  - Advisory, not commanded. The autopilot owns the final action; we
    publish a *suggestion* with its full reasoning.
  - Pure: no I/O, no state. Trivially testable.
  - Fail-safe: when the cost curve is flat or has tied alternatives in
    distinct directions, return `None` rather than picking arbitrarily.
    "No recommendation" is the safer signal than "swerve confidently
    in a random direction" (PLAN.md §III.1 safety bias).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class HeadingRecommendation:
    """Per-frame heading output.

    - `recommended_heading_deg`: argmin of the cost curve, or `None`
      when ambiguous / tied across distinct directions.
    - `candidate_headings_deg` + `cost_curve` align 1:1.
    - `per_term_at_choice` lets the dashboard show the breakdown that
      produced the recommendation; empty when no recommendation.
    - `reason` is the single dominant weighted cost term at the chosen
      heading, or `"ambiguous"` when the recommender abstained.
    """
    recommended_heading_deg: float | None
    candidate_headings_deg: list[float]
    cost_curve: list[float]
    per_term_at_choice: dict[str, float]
    reason: str


# ---------------------------------------------------------------------------
# Cost terms (each returns a value in [0, 1] before weighting)
#
# The three threat-flavoured signals (presence, proximity, urgency) all
# emanate from the *same* per-bin source, so they're folded into one
# Gaussian-falloff "threat field": each blocked bin produces a bump in
# heading-space whose amplitude depends on its score, range, and TTC.
# Picking a heading inside that bump = bad. This avoids the bug where
# a fast-closing threat at +20° doesn't penalise candidates near 0°
# because there's no per-bin TTC value at the 0° bin.
#
# `block`, `close`, `ttc` weights then *select* which of those signals
# the final cost is most sensitive to:
#   - block: the raw score-amplitude bump
#   - close: the range-amplified bump
#   - ttc: the TTC-amplified bump
# All three peak at the same bearing; weights tune relative influence.
# ---------------------------------------------------------------------------

def _bin_amplitudes(
    scores: Sequence[float],
    min_range_m: Sequence[float | None],
    ttc_s: Sequence[float | None],
    blocked_threshold: float,
    range_eps_m: float,
    ttc_eps_s: float,
    range_scale_m: float = 5.0,
    ttc_scale_s: float = 5.0,
    alert_threshold: float = 0.33,
) -> list[tuple[float, float, float]]:
    """For each bin, compute three amplitudes (block_amp, close_amp, ttc_amp).

    The block amplitude is a *graded* function of the fused score:
      - score < `alert_threshold`     → 0
      - alert_threshold ≤ s < blocked → linear ramp 0 → 0.5
      - blocked ≤ s ≤ 1               → linear ramp 0.5 → 1
    so a 1.0 (all-three) bin pushes the field twice as hard as a 0.66
    (two sensors) bin, instead of contributing nearly the same. Without
    this grading, scenes with many low-confidence firings (e.g. Boats
    with fisheye dominance) flatten the cost curve and the recommender
    pins to the FOV edge.

    `close_amp` and `ttc_amp` only fire above `alert_threshold` so a
    single-sensor blip doesn't drag the threat field around just because
    its bin happens to have a range / TTC reading.
    """
    amps: list[tuple[float, float, float]] = []
    span_low = max(blocked_threshold - alert_threshold, 1e-6)
    span_high = max(1.0 - blocked_threshold, 1e-6)
    for s, r, t in zip(scores, min_range_m, ttc_s):
        if s < alert_threshold:
            amps.append((0.0, 0.0, 0.0))
            continue
        if s < blocked_threshold:
            block = 0.5 * float(s - alert_threshold) / span_low
        else:
            block = 0.5 + 0.5 * float(s - blocked_threshold) / span_high
        block = max(0.0, min(block, 1.0))
        if r is None:
            close = 0.0
        else:
            close = float(range_scale_m / (max(r, range_eps_m) + range_scale_m))
        if t is None:
            ttc = 0.0
        else:
            ttc = float(ttc_scale_s / (max(t, ttc_eps_s) + ttc_scale_s))
        amps.append((block, close, ttc))
    return amps


def _gaussian_field_cost(
    candidate_deg: float,
    bin_centers_deg: Sequence[float],
    amplitudes: Sequence[float],
    sigma_deg: float,
) -> float:
    """Max contribution of `amplitudes[b] * gaussian(θ - center_b)` across
    bins. Max (not sum) so a single nearby threat is not diluted by
    distant zeros."""
    if sigma_deg <= 0 or not amplitudes:
        return 0.0
    worst = 0.0
    two_sigma_sq = 2.0 * sigma_deg * sigma_deg
    for c, a in zip(bin_centers_deg, amplitudes):
        if a <= 0:
            continue
        d = candidate_deg - c
        contrib = a * math.exp(-(d * d) / two_sigma_sq)
        if contrib > worst:
            worst = contrib
    return min(worst, 1.0)


def _course_cost(candidate_deg: float, current_heading_deg: float | None,
                 max_swing_deg: float) -> float:
    if max_swing_deg <= 0:
        return 0.0
    ref = current_heading_deg if current_heading_deg is not None else 0.0
    swing = abs(candidate_deg - ref)
    return float(min(swing / max_swing_deg, 1.0))


def _nogo_cost(candidate_deg: float) -> float:
    """Stub: wired to wind / no-go zone once V.3 IMU + autopilot
    bridge lands. Always 0 today."""
    return 0.0


def _free_space_amps(
    free_dist_m: Sequence[float | None],
    max_range_m: float,
) -> list[float]:
    """Per-bin block amplitude from the segmentation free-space profile.

    `None` (open water to the horizon) -> 0; a blocked / 0 m bin -> 1; a finite
    free distance d -> 1 - d/max_range (closer navigable limit = higher threat).
    Lets the water-mask free-space feed the same threat field as the fused
    bin scores (Phase 1 segmentation-driven navigation).
    """
    # Guard: a misconfigured max range of 0/negative must not
    # ZeroDivisionError the recommender; clamp the denominator to a small
    # positive value (a 0 m bin still reads as full threat, any finite
    # distance then clamps to 0).
    max_range_m = max(float(max_range_m), 1e-6)
    out: list[float] = []
    for d in free_dist_m:
        if d is None:
            out.append(0.0)
        else:
            out.append(max(0.0, min(1.0, 1.0 - float(d) / max_range_m)))
    return out


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

def recommend_heading(
    bin_centers_deg: Sequence[float],
    scores: Sequence[float],
    min_range_m: Sequence[float | None],
    per_bin_velocity_mps: Sequence[float | None],
    per_bin_ttc_s: Sequence[float | None],
    config: dict,
    current_heading_deg: float | None = None,
    free_dist_m: Sequence[float | None] | None = None,
) -> HeadingRecommendation:
    """Sweep candidate headings, sum weighted costs, return the argmin
    (or `None` when ambiguous).

    `per_bin_velocity_mps` is currently surfaced for telemetry only:
    the TTC term is the velocity-informed signal that drives heading
    choice. The raw velocities are kept on the dataclass interface so
    consumers (and the dashboard) can display them without recomputing.
    """
    if not bin_centers_deg or not scores:
        return HeadingRecommendation(
            recommended_heading_deg=None,
            candidate_headings_deg=[],
            cost_curve=[],
            per_term_at_choice={},
            reason="ambiguous",
        )

    step = float(config.get("candidate_step_deg", 1.0))
    blocked_threshold = float(config.get("blocked_threshold", 0.66))
    sigma_deg = float(config.get("block_sigma_deg", 12.0))
    max_swing = float(config.get("max_swing_deg", 30.0))
    weights = dict(config.get("weights", {}))
    weights.setdefault("block", 1.0)
    weights.setdefault("close", 0.3)
    weights.setdefault("ttc", 0.6)
    weights.setdefault("course", 0.2)
    weights.setdefault("nogo", 1.0)
    tied_gap_frac = float(config.get("tied_gap_frac", 0.05))
    tied_min_sep = float(config.get("tied_min_separation_deg", 30.0))
    range_eps = float(config.get("range_eps_m", 0.5))
    ttc_eps = float(config.get("ttc_eps_s", 0.5))
    abstain_cost = float(config.get("abstain_cost", 0.0))
    alert_threshold = float(config.get("alert_threshold", 0.33))

    # Candidate grid covers the bin span exactly. Avoid float-walk
    # off-by-one issues by using integer counts.
    lo = float(bin_centers_deg[0])
    hi = float(bin_centers_deg[-1])
    n = max(int(round((hi - lo) / step)) + 1, 1)
    candidates = [lo + i * step for i in range(n)]

    amps = _bin_amplitudes(
        scores, min_range_m, per_bin_ttc_s,
        blocked_threshold=blocked_threshold,
        range_eps_m=range_eps,
        ttc_eps_s=ttc_eps,
        alert_threshold=alert_threshold,
    )
    block_amps = [a[0] for a in amps]
    close_amps = [a[1] for a in amps]
    ttc_amps = [a[2] for a in amps]

    # Optional segmentation free-space threat (Phase 1, off unless the caller
    # passes free_dist_m). A near navigable limit / blocked bearing raises the
    # block field, so the water-mask free-space and the fused bin scores share
    # one threat representation. max() so it only ever adds threat.
    if free_dist_m is not None and len(free_dist_m) == len(block_amps):
        fs_max = float(config.get("free_space_max_range_m", 15.0))
        fs_amps = _free_space_amps(free_dist_m, fs_max)
        block_amps = [max(b, f) for b, f in zip(block_amps, fs_amps)]

    cost_curve: list[float] = []
    per_term: list[dict[str, float]] = []
    for theta in candidates:
        terms = {
            "block": weights["block"] * _gaussian_field_cost(
                theta, bin_centers_deg, block_amps, sigma_deg),
            "close": weights["close"] * _gaussian_field_cost(
                theta, bin_centers_deg, close_amps, sigma_deg),
            "ttc": weights["ttc"] * _gaussian_field_cost(
                theta, bin_centers_deg, ttc_amps, sigma_deg),
            "course": weights["course"] * _course_cost(
                theta, current_heading_deg, max_swing),
            "nogo": weights["nogo"] * _nogo_cost(theta),
        }
        cost_curve.append(sum(terms.values()))
        per_term.append(terms)

    best_i = min(range(len(cost_curve)), key=cost_curve.__getitem__)
    best_cost = cost_curve[best_i]

    # All-blocked guard runs *first*: if even the cheapest candidate has
    # high residual block-threat cost, "swerve to the edge" is bad
    # advice: there genuinely is no safe option in our FOV. Abstain
    # so the autopilot holds course / reduces speed instead. Takes
    # precedence over the tied-alternatives check because an "all
    # blocked, symmetric" scene would otherwise read as "ambiguous"
    # when the stronger signal is "stop".
    if abstain_cost > 0 and per_term[best_i].get("block", 0.0) >= abstain_cost:
        return HeadingRecommendation(
            recommended_heading_deg=None,
            candidate_headings_deg=candidates,
            cost_curve=cost_curve,
            per_term_at_choice={},
            reason="all_blocked",
        )

    # Tied-alternatives guard: if any *distant* candidate is within
    # tied_gap_frac of the best, we abstain. Distance is measured
    # against the chosen heading, not 0°.
    if _is_ambiguous(cost_curve, candidates, best_i, tied_gap_frac, tied_min_sep):
        return HeadingRecommendation(
            recommended_heading_deg=None,
            candidate_headings_deg=candidates,
            cost_curve=cost_curve,
            per_term_at_choice={},
            reason="ambiguous",
        )

    chosen_terms = per_term[best_i]
    reason = max(chosen_terms.items(), key=lambda kv: kv[1])[0] if chosen_terms else "flat"
    if best_cost == 0.0:
        reason = "clear"

    return HeadingRecommendation(
        recommended_heading_deg=float(candidates[best_i]),
        candidate_headings_deg=candidates,
        cost_curve=cost_curve,
        per_term_at_choice=chosen_terms,
        reason=reason,
    )


def _is_ambiguous(
    cost_curve: Sequence[float],
    candidates: Sequence[float],
    best_i: int,
    tied_gap_frac: float,
    tied_min_sep: float,
) -> bool:
    """Return True iff a distinct-direction candidate is within
    `tied_gap_frac` of the best cost."""
    best_cost = cost_curve[best_i]
    # Flat-cost case (everything zero): no preference, suppress.
    if best_cost == 0.0 and all(c == 0.0 for c in cost_curve):
        return False
    # Threshold: cost must be at most best * (1 + gap_frac) to count
    # as "tied". When best_cost is exactly 0 but neighbours aren't,
    # only exact zeros tie: but those would be at adjacent bearings,
    # not distinct directions.
    if best_cost == 0.0:
        ref = 1e-9
    else:
        ref = best_cost * (1.0 + tied_gap_frac)
    best_heading = candidates[best_i]
    for i, c in enumerate(cost_curve):
        if i == best_i:
            continue
        if c > ref:
            continue
        if abs(candidates[i] - best_heading) >= tied_min_sep:
            return True
    return False


# ---------------------------------------------------------------------------
# Convenience: derive from a FusionResult-like object
# ---------------------------------------------------------------------------

def recommend_from_fusion(
    fusion_result,                     # duck-typed FusionResult
    config: dict,
    current_heading_deg: float | None = None,
    free_dist_m: Sequence[float | None] | None = None,
) -> HeadingRecommendation:
    """Adapter for the canonical FusionResult dataclass: used by
    fusion.py and the dashboard. `free_dist_m` (optional) is the per-bin
    segmentation free-space profile; when given it augments the block field."""
    return recommend_heading(
        bin_centers_deg=list(fusion_result.bin_centers),
        scores=list(fusion_result.scores),
        min_range_m=list(fusion_result.min_ranges),
        per_bin_velocity_mps=list(getattr(fusion_result, "per_bin_velocity_mps", []) or []),
        per_bin_ttc_s=list(getattr(fusion_result, "per_bin_ttc_s", []) or []),
        config=config,
        current_heading_deg=current_heading_deg,
        free_dist_m=free_dist_m,
    )


# ---------------------------------------------------------------------------
# Temporal smoothing
#
# The per-frame recommendation is jittery: a single noisy frame can
# swing a 1° argmin by 20°. That's fine for the cost-curve display but
# wrong for steering: an autopilot following the raw signal would
# oscillate. A rolling mean over the last N frames gives the autopilot
# a steady number, and *holds* the last smoothed value across
# ambiguous / null frames so brief abstentions don't drop the line.
# ---------------------------------------------------------------------------

class HeadingSmoother:
    """Rolling-mean smoother over the recommended-heading stream.

    Construct one instance per consumer (per dashboard, per
    `fusion.py --out` run). Call `update(heading_deg | None)` per
    frame; receive the smoothed value or `None` if not enough
    samples have been seen yet.

    Design notes:
      - None inputs do not enter the buffer; the smoother carries
        forward the last smoothed value. Rationale: an "ambiguous"
        recommendation should not change the steering signal.
      - `window_n` is the max buffer length; once warmed up,
        smoothing is a simple arithmetic mean over the buffer.
      - `min_samples` gates the first emission. Below it, `update`
        returns `None` so the autopilot can hold course.
    """

    def __init__(self, window_n: int = 9, min_samples: int = 3):
        if window_n < 1:
            raise ValueError("window_n must be >= 1")
        if min_samples < 1 or min_samples > window_n:
            raise ValueError("min_samples must be in [1, window_n]")
        self.window_n = window_n
        self.min_samples = min_samples
        self._buf: deque[float] = deque(maxlen=window_n)
        self._last_smoothed: float | None = None

    def reset(self) -> None:
        self._buf.clear()
        self._last_smoothed = None

    def update(self, heading_deg: float | None) -> float | None:
        if heading_deg is not None:
            self._buf.append(float(heading_deg))
        if len(self._buf) >= self.min_samples:
            self._last_smoothed = sum(self._buf) / len(self._buf)
        return self._last_smoothed

    @property
    def smoothed(self) -> float | None:
        return self._last_smoothed


def rolling_average_headings(
    headings: Sequence[float | None],
    window_n: int = 9,
    min_samples: int = 3,
) -> list[float | None]:
    """Offline / batch version of `HeadingSmoother.update`.

    Returns one smoothed value per input frame, applying the same
    "hold last value across None" semantics.
    """
    smoother = HeadingSmoother(window_n=window_n, min_samples=min_samples)
    return [smoother.update(h) for h in headings]
