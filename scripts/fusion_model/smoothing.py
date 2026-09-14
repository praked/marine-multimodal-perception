# ****************************************************************************
# *  Temporal persistence for the calibrated per-bin learned score.
# *
# *  `fusion.py --scorer` scores every frame independently, so p_obstacle
# *  flickers frame-to-frame (a one-frame evidence dropout reads as the
# *  obstacle vanishing). ScorePersistence damps the DECAY only:
# *
# *      p_smooth[t] = max(p_raw[t], exp(-dt / decay_time_s) * p_smooth[t-1])
# *
# *  Rises are instant — a fresh detection is never smoothed away, so the
# *  first frame where the smoothed series crosses any threshold is never
# *  later than the raw series' crossing (zero onset delay by construction;
# *  p_smooth >= p_raw always). Decay follows an exponential with the
# *  configured time constant, converted per frame from the ACTUAL timestamp
# *  spacing (the corpus cadence is ~3 fps but drifts; RoundedTime is the
# *  authority). State resets — output := raw — on the first frame, on a
# *  backwards timestamp jump, on a gap larger than `gap_reset_s` (chunk
# *  rolls produce multi-second jumps), and on a bin-count change.
# *
# *  ScoreEma is the symmetric baseline (same time-constant parametrisation)
# *  kept for EVALUATION comparison only: it lags onsets, so it must not be
# *  emitted on the nav path.
# *
# *  House pattern: FreeSpaceSmoother (scripts/utils/segmentation.py) — one
# *  instance per consumer, `update` once per frame in stream order.

from __future__ import annotations

import math


def parse_ts_seconds(ts: str | float) -> float:
    """`HH:MM:SS.f` (RoundedTime convention) -> seconds since midnight.

    Floats/ints pass through so callers with a numeric clock can reuse the
    smoothers directly.
    """
    if isinstance(ts, (int, float)):
        return float(ts)
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


class _TimestampedSmoother:
    """Shared reset/cadence logic for the score smoothers."""

    def __init__(self, decay_time_s: float = 2.0, gap_reset_s: float = 5.0):
        if decay_time_s <= 0:
            raise ValueError(f"decay_time_s must be > 0, got {decay_time_s}")
        if gap_reset_s <= 0:
            raise ValueError(f"gap_reset_s must be > 0, got {gap_reset_s}")
        self.tau = float(decay_time_s)
        self.gap = float(gap_reset_s)
        self._prev: list[float] | None = None
        self._prev_t: float | None = None

    def reset(self) -> None:
        self._prev = None
        self._prev_t = None

    def _decay_for(self, t: float, n_bins: int) -> float:
        """exp(-dt/tau) for a valid step, 0.0 when state must reset."""
        if (self._prev is None or self._prev_t is None
                or len(self._prev) != n_bins):
            return 0.0
        dt = t - self._prev_t
        if dt <= 0 or dt > self.gap:
            # Backwards jump or chunk-roll/stall gap: stale state would
            # assert obstacles across a discontinuity — drop it.
            return 0.0
        return math.exp(-dt / self.tau)


class ScorePersistence(_TimestampedSmoother):
    """Asymmetric persistence: instant rise, exponential decay."""

    def update(self, ts: str | float, p_raw) -> list[float]:
        t = parse_ts_seconds(ts)
        vals = [float(v) for v in p_raw]
        decay = self._decay_for(t, len(vals))
        prev = self._prev if decay > 0.0 else vals
        out = [max(v, decay * p) for v, p in zip(vals, prev)]
        self._prev = out
        self._prev_t = t
        return out


class ScoreEma(_TimestampedSmoother):
    """Symmetric EMA with the same time-constant parametrisation
    (alpha = 1 - exp(-dt/tau)). Eval-only baseline — lags onsets."""

    def update(self, ts: str | float, p_raw) -> list[float]:
        t = parse_ts_seconds(ts)
        vals = [float(v) for v in p_raw]
        decay = self._decay_for(t, len(vals))
        prev = self._prev if decay > 0.0 else vals
        out = [(1.0 - decay) * v + decay * p for v, p in zip(vals, prev)]
        self._prev = out
        self._prev_t = t
        return out
