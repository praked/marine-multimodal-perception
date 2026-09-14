# ****************************************************************************
# *  Flicker + onset-delay evaluation for scorer-score smoothing.
# *
# *  Reads sectors JSONLs that carry the raw per-bin `p_obstacle` (emitted by
# *  `fusion.py --scorer`, smoothing OFF) and replays the smoothing offline —
# *  the smoothers are pure functions of (timestamp, p_raw) stream, so one
# *  fusion run per clip supports the whole decay-constant sweep.
# *
# *  Reports, per clip x variant:
# *    - flicker: mean per-bin |p[t] - p[t-1]| over consecutive frames
# *      (reset boundaries — timestamp gaps > gap_reset_s — excluded so a
# *      chunk roll is not billed as flicker)
# *    - onset delay: for each threshold, the number of (bin, onset) events
# *      where the smoothed series first crosses LATER than the raw one.
# *      Must be 0 for ScorePersistence (instant rise, p_smooth >= p_raw);
# *      the symmetric EMA baseline is included to show it is NOT 0 there.
# *    - decay tail: mean frames a variant stays >= 0.5 after raw drops
# *      below 0.5 (how long a vanished detection lingers).
# *
# *  Usage:
# *    python -m scripts.eval.scorer_smoothing_eval \
# *        results/sectors_A.jsonl results/sectors_B.jsonl \
# *        --taus 0.5 1.0 2.0 3.0 --thresholds 0.3 0.5 0.7
# ****************************************************************************

import argparse
import json
from pathlib import Path

from scripts.fusion_model.smoothing import (
    ScoreEma,
    ScorePersistence,
    parse_ts_seconds,
)


def load_series(path):
    """-> (timestamps: list[str], p: list[list[float]]) from a sectors JSONL."""
    ts, p = [], []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if "p_obstacle" not in rec:
                raise SystemExit(f"{path}: record without p_obstacle — "
                                 "emit with fusion.py --scorer first")
            ts.append(rec["timestamp"])
            p.append([float(v) for v in rec["p_obstacle"]])
    return ts, p


def _step_valid(ts, i, gap_reset_s):
    dt = parse_ts_seconds(ts[i]) - parse_ts_seconds(ts[i - 1])
    return 0 < dt <= gap_reset_s


def flicker(ts, p, gap_reset_s=5.0):
    """Mean per-bin |delta p| across consecutive valid frame steps."""
    num = den = 0.0
    for i in range(1, len(p)):
        if not _step_valid(ts, i, gap_reset_s):
            continue
        for a, b in zip(p[i - 1], p[i]):
            num += abs(b - a)
            den += 1
    return num / den if den else 0.0


def onset_events(ts, p, thr, gap_reset_s=5.0):
    """Onset frames per bin: i where p crosses thr upward (or holds thr at a
    stream/reset start). -> set of (bin, frame_index)."""
    events = set()
    for b in range(len(p[0])):
        for i in range(len(p)):
            fresh_start = i == 0 or not _step_valid(ts, i, gap_reset_s)
            below_prev = fresh_start or p[i - 1][b] < thr
            if p[i][b] >= thr and below_prev:
                events.add((b, i))
    return events


def onset_delays(ts, p_raw, p_smooth, thr, gap_reset_s=5.0):
    """Count raw onsets NOT matched at the same frame in the smoothed
    series (i.e. smoothed still below thr when raw crosses = onset delay)."""
    delayed = 0
    raw_ev = onset_events(ts, p_raw, thr, gap_reset_s)
    for (b, i) in raw_ev:
        if p_smooth[i][b] < thr:
            delayed += 1
    return delayed, len(raw_ev)


def dropouts(ts, p, thr, max_len=2, gap_reset_s=5.0):
    """Count short DIPS: runs of <= max_len consecutive frames below `thr`
    bounded by frames >= thr on both sides (no reset boundary inside).
    These one-to-two-frame evidence dropouts are the flicker a nav consumer
    actually acts on — a blocked bin momentarily reading clear."""
    count = 0
    n_bins = len(p[0])
    for b in range(n_bins):
        run = 0
        above_before = False
        for i in range(len(p)):
            if i and not _step_valid(ts, i, gap_reset_s):
                run, above_before = 0, False
            if p[i][b] >= thr:
                if above_before and 0 < run <= max_len:
                    count += 1
                run, above_before = 0, True
            elif above_before:
                run += 1
    return count


def decay_tail(ts, p_raw, p_smooth, thr=0.5, gap_reset_s=5.0):
    """Mean extra frames the smoothed series holds >= thr after the raw
    series has dropped below it (ghost-tail length)."""
    tails = []
    n_bins = len(p_raw[0])
    for b in range(n_bins):
        run = None
        for i in range(len(p_raw)):
            if i and not _step_valid(ts, i, gap_reset_s):
                run = None
            if p_raw[i][b] < thr and p_smooth[i][b] >= thr:
                run = (run or 0) + 1
            else:
                if run:
                    tails.append(run)
                run = None
        if run:
            tails.append(run)
    return (sum(tails) / len(tails)) if tails else 0.0


def evaluate(paths, taus, thresholds, gap_reset_s):
    rows = []
    for path in paths:
        ts, p_raw = load_series(path)
        name = Path(path).stem
        base = flicker(ts, p_raw, gap_reset_s)
        rows.append((name, "raw", None, base, 1.0,
                     {t: (0, len(onset_events(ts, p_raw, t, gap_reset_s)))
                      for t in thresholds}, 0.0,
                     {t: dropouts(ts, p_raw, t, 2, gap_reset_s)
                      for t in thresholds}))
        for kind, cls in (("persistence", ScorePersistence),
                          ("ema", ScoreEma)):
            for tau in taus:
                sm = cls(decay_time_s=tau, gap_reset_s=gap_reset_s)
                p_s = [sm.update(t, pr) for t, pr in zip(ts, p_raw)]
                fl = flicker(ts, p_s, gap_reset_s)
                delays = {t: onset_delays(ts, p_raw, p_s, t, gap_reset_s)
                          for t in thresholds}
                tail = decay_tail(ts, p_raw, p_s, 0.5, gap_reset_s)
                dips = {t: dropouts(ts, p_s, t, 2, gap_reset_s)
                        for t in thresholds}
                rows.append((name, kind, tau, fl, fl / base if base else 1.0,
                             delays, tail, dips))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+", help="sectors JSONL(s) with p_obstacle")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7])
    ap.add_argument("--gap-reset-s", type=float, default=5.0)
    args = ap.parse_args(argv)

    rows = evaluate(args.jsonl, args.taus, args.thresholds, args.gap_reset_s)
    thr_hdr = " | ".join(f"delayed@{t:g}" for t in args.thresholds)
    dip_hdr = " | ".join(f"dips@{t:g}" for t in args.thresholds)
    print(f"| clip | variant | tau_s | flicker | vs raw | {thr_hdr} "
          f"| {dip_hdr} | tail@0.5 (frames) |")
    print("|---" * (5 + 2 * len(args.thresholds) + 1) + "|")
    for name, kind, tau, fl, ratio, delays, tail, dips in rows:
        thr_cells = " | ".join(f"{delays[t][0]}/{delays[t][1]}"
                               for t in args.thresholds)
        dip_cells = " | ".join(f"{dips[t]}" for t in args.thresholds)
        tau_s = f"{tau:g}" if tau is not None else "—"
        print(f"| {name} | {kind} | {tau_s} | {fl:.4f} | {ratio:.2f}x "
              f"| {thr_cells} | {dip_cells} | {tail:.2f} |")


if __name__ == "__main__":
    main()
