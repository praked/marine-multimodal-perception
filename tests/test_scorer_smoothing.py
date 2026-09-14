"""Per-bin temporal persistence for the calibrated learned score
(scripts/fusion_model/smoothing.py + fusion.py --scorer-smoothing).

Safety property under test: rises are INSTANT — the smoothed series must
cross every threshold on exactly the frame the raw series does (zero onset
delay), because p_smooth >= p_raw by construction. Decay follows
exp(-dt/tau) off the real timestamps; state resets across chunk-roll-sized
timestamp gaps and backwards jumps.
"""

import json
import math
import sys

import numpy as np
import pytest

from scripts.fusion_model.smoothing import (
    ScoreEma,
    ScorePersistence,
    parse_ts_seconds,
)

DT = 1.0 / 3.0  # nominal corpus cadence


def _ts(seconds):
    h = int(seconds) // 3600
    m = int(seconds) % 3600 // 60
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def test_parse_ts_seconds():
    assert parse_ts_seconds("00:00:01.5") == 1.5
    assert parse_ts_seconds("16:37:01.2") == 16 * 3600 + 37 * 60 + 1.2
    assert parse_ts_seconds(42.5) == 42.5  # numeric passthrough


def test_first_frame_is_raw():
    sm = ScorePersistence(decay_time_s=2.0)
    assert sm.update("12:00:00.0", [0.7, 0.1]) == [0.7, 0.1]


def test_rise_is_instant():
    sm = ScorePersistence(decay_time_s=2.0)
    sm.update(0.0, [0.05])
    out = sm.update(DT, [0.9])
    assert out == [0.9]  # fresh detection never smoothed away


def test_decay_follows_exponential():
    tau = 1.5
    sm = ScorePersistence(decay_time_s=tau)
    sm.update(0.0, [0.8])
    t = 0.0
    expected = 0.8
    for _ in range(6):
        t += DT
        expected *= math.exp(-DT / tau)
        out = sm.update(t, [0.0])
        assert out[0] == pytest.approx(expected, rel=1e-12)


def test_decay_uses_actual_dt_not_nominal():
    tau = 2.0
    sm = ScorePersistence(decay_time_s=tau)
    sm.update(0.0, [0.8])
    out = sm.update(1.0, [0.0])  # a 1 s stall, still inside the gap window
    assert out[0] == pytest.approx(0.8 * math.exp(-1.0 / tau))


def test_smoothed_never_below_raw():
    rng = np.random.default_rng(0)
    sm = ScorePersistence(decay_time_s=1.0)
    t = 0.0
    for _ in range(200):
        raw = rng.uniform(0, 1, 5)
        out = sm.update(t, raw)
        assert all(o >= r - 1e-12 for o, r in zip(out, raw))
        t += DT


def test_gap_resets_state():
    sm = ScorePersistence(decay_time_s=10.0, gap_reset_s=5.0)
    sm.update(0.0, [0.9])
    # chunk roll: 60 s jump. Stale 0.9 must NOT persist across it.
    out = sm.update(60.0, [0.1])
    assert out == [0.1]


def test_backwards_timestamp_resets_state():
    sm = ScorePersistence(decay_time_s=10.0)
    sm.update(100.0, [0.9])
    out = sm.update(99.0, [0.1])
    assert out == [0.1]


def test_bin_count_change_resets_state():
    sm = ScorePersistence(decay_time_s=10.0)
    sm.update(0.0, [0.9, 0.9])
    out = sm.update(DT, [0.1, 0.1, 0.1])
    assert out == [0.1, 0.1, 0.1]


def test_reset_method():
    sm = ScorePersistence(decay_time_s=10.0)
    sm.update(0.0, [0.9])
    sm.reset()
    assert sm.update(DT, [0.2]) == [0.2]


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        ScorePersistence(decay_time_s=0.0)
    with pytest.raises(ValueError):
        ScorePersistence(decay_time_s=1.0, gap_reset_s=-1.0)


def test_zero_onset_delay_synthetic():
    """The frame where raw first crosses each threshold is unchanged in the
    smoothed series — for every threshold, on a series with decays, gaps and
    re-detections."""
    rng = np.random.default_rng(7)
    n, bins = 300, 4
    raw = np.zeros((n, bins))
    ts = np.arange(n) * DT
    # sparse detection events with noise floors
    for b in range(bins):
        raw[:, b] = rng.uniform(0, 0.15, n)
        for start in rng.choice(n - 20, 5, replace=False):
            raw[start:start + 6, b] = rng.uniform(0.5, 1.0, 6)
    ts[200:] += 30.0  # a chunk roll mid-series

    sm = ScorePersistence(decay_time_s=2.0, gap_reset_s=5.0)
    smooth = np.array([sm.update(t, r) for t, r in zip(ts, raw)])

    for thr in (0.2, 0.3, 0.5, 0.7, 0.9):
        for b in range(bins):
            raw_first = np.argmax(raw[:, b] >= thr) \
                if (raw[:, b] >= thr).any() else None
            sm_first = np.argmax(smooth[:, b] >= thr) \
                if (smooth[:, b] >= thr).any() else None
            if raw_first is None:
                continue
            # smoothed crosses no later than raw (>= raw pointwise), and is
            # already over the threshold on raw's own onset frame
            assert sm_first is not None and sm_first <= raw_first
            assert smooth[raw_first, b] >= thr


def test_ema_is_symmetric_and_lags_rise():
    """The eval-only EMA baseline: continuous, but it DOES delay onsets —
    which is exactly why it must not ship on the nav path."""
    ema = ScoreEma(decay_time_s=2.0)
    ema.update(0.0, [0.0])
    out = ema.update(DT, [1.0])
    assert 0.0 < out[0] < 1.0  # lagged rise
    # and symmetric decay: same smoothing factor down as up
    a = 1.0 - math.exp(-DT / 2.0)
    assert out[0] == pytest.approx(a * 1.0)


# ---------------------------------------------------------------------------
# fusion.py emission wiring
# ---------------------------------------------------------------------------

def _run_fusion(argv):
    from scripts.sensor_processing import fusion as fusion_mod
    old = sys.argv
    sys.argv = ["fusion"] + argv
    try:
        fusion_mod.main()
    finally:
        sys.argv = old


@pytest.fixture
def scorer_path(tmp_path):
    from scripts.fusion_model.models import GatedMixtureScorer
    rng = np.random.default_rng(0)
    e = rng.uniform(0, 1, (300, 3))
    c = rng.uniform(0, 1, (300, 8))
    y = (e.sum(axis=1) > 1.5).astype(float)
    m = GatedMixtureScorer.fit(e, c, y, seed=0, epochs=20)
    p = tmp_path / "scorer.json"
    m.save(p)
    return p


def test_fusion_off_by_default_and_emits_when_enabled(tmp_path, scorer_path,
                                                      make_fusion_triplet):
    trip = make_fusion_triplet(tmp_path / "clip", "2099-07-03_12-00-00", n=4)
    trip_arg = str(trip.fisheye.parent / trip.timestamp)

    out_off = tmp_path / "off.jsonl"
    _run_fusion(["--triplet", trip_arg, "--out", str(out_off),
                 "--scorer", str(scorer_path), "--print-every", "0"])
    recs_off = [json.loads(l) for l in out_off.read_text().splitlines()]
    assert recs_off
    for r in recs_off:
        assert "p_obstacle" in r
        assert "p_obstacle_smooth" not in r  # default OFF

    out_on = tmp_path / "on.jsonl"
    _run_fusion(["--triplet", trip_arg, "--out", str(out_on),
                 "--scorer", str(scorer_path), "--scorer-smoothing", "2.0",
                 "--print-every", "0"])
    recs_on = [json.loads(l) for l in out_on.read_text().splitlines()]
    assert len(recs_on) == len(recs_off)
    for r_on, r_off in zip(recs_on, recs_off):
        smooth = r_on.pop("p_obstacle_smooth")
        assert len(smooth) == len(r_on["bin_centers_deg"])
        # instant rise: never below the raw score (rounding tolerance)
        assert all(s >= p - 1e-4
                   for s, p in zip(smooth, r_on["p_obstacle"]))
        # byte-identical otherwise: the field is purely additive
        assert r_on == r_off
