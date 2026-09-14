"""Table IV from boat1's closed-loop trial logs.

Reads every ``sector_experiment_*.jsonl`` written by boatv1's
``sector_experiment.py`` (one trial per file: HOLD -> DECIDE -> GO -> DONE)
and prints one row per trial plus a summary, as Markdown (and CSV with
``--csv``).

    python -m scripts.eval.closed_loop_table /Volumes/ROS2_SSD/asvproject/captures/<session>/boat1_logs/
    python -m scripts.eval.closed_loop_table logs/*.jsonl --csv results/closed_loop/table.csv

A launch with ``--trials N`` writes one file holding N trials, each row
tagged with its ``trial`` index (0 = the start/hardware rows, before the
first trial); those are split into one table row per trial (``<log>#k``).

Per trial: the launch ``--name`` and ``--note``, decisions attempted /
refused ("blocked") before a heading was taken, time from start to the
accepted heading, the bow-relative and absolute heading, checkpoint
distance, GO duration, station-keeping duration and pattern when
``--station-keep-s`` was used, outcome (reached / abort / timeout / stale
link / interrupted), and -- from the ``sectors`` rows the experiment logs
during GO -- the closest radar range and the highest obstacle probability
within +-15 deg of the course. Nothing is inferred that the log does not
carry: a trial without GO sector rows shows those two columns empty.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

HALF_DEG = 15.0


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _trial_index(row: dict) -> int:
    """``trial`` of one row: 0 for the launch-wide start/hardware rows, 1-based
    per trial, and 1 for the pre-``--trials`` logs that carry no field at all."""
    v = row.get("trial")
    return 1 if v is None else int(v)


def load_trial(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def summarise(path: str, rows: list[dict]) -> dict:
    out = {"trial": Path(path).stem.replace("sector_experiment_", ""),
           "name": None, "note": None,
           "decisions": 0, "refused": 0, "t_decide_s": None,
           "rel_heading_deg": None, "abs_heading_deg": None, "field_p": None,
           "checkpoint_m": None, "go_s": None, "outcome": "no log",
           "station_s": None, "station_pattern": None,
           "closest_m": None, "closest_deg": None, "max_p_course": None,
           "armed": None, "n_records_hold": None}
    go_start = go_end = None
    station_start = None
    abs_hdg = None
    for r in rows:
        k = r.get("kind")
        if k == "start":
            out["name"] = r.get("name") or None
            out["note"] = r.get("note") or None
        elif k == "hardware":
            out["armed"] = not r.get("dry_run", True)
        elif k == "decide":
            out["decisions"] += 1
            if r.get("result") == "ok" or r.get("result") == "fixed":
                out["rel_heading_deg"] = r.get("rel_heading_deg")
                out["field_p"] = r.get("field_p")
                out["n_records_hold"] = r.get("n")
            elif r.get("result") in ("blocked", "insufficient", "no_heading_or_fix"):
                out["refused"] += 1
        elif k == "checkpoint":
            abs_hdg = r.get("abs_heading_deg")
            out["abs_heading_deg"] = abs_hdg
            out["checkpoint_m"] = r.get("distance_m")
            out["t_decide_s"] = r.get("t_s")
        elif k == "phase" and r.get("phase") == "GO":
            go_start = r.get("t_s")
        elif k == "phase" and r.get("phase") == "STATION":
            # --station-keep-s: the leg ended here ("checkpoint reached") and
            # the trial goes on holding the checkpoint. Close the GO clock now
            # so the station time is not counted as leg time.
            go_end = r.get("t_s")
            station_start = r.get("t_s")
            if go_start is not None and go_end is not None:
                out["go_s"] = round(float(go_end) - float(go_start), 1)
        elif k == "station_pattern":
            out["station_pattern"] = r.get("pattern")
        elif k == "station":
            out["station_pattern"] = r.get("pattern") or out["station_pattern"] or "hold"
        elif k == "phase" and r.get("phase") == "DONE":
            why = str(r.get("why", ""))
            out["outcome"] = ("reached" if why.startswith("checkpoint reached")
                              else "reached" if why.startswith("station keeping done")
                              else "abort" if why.startswith("abort")
                              else "stale link" if "stale" in why
                              else "timeout" if "timeout" in why
                              else "decide-only" if "decide-only" in why
                              else why or "done")
            if station_start is not None:
                out["station_s"] = round(float(r.get("t_s", 0.0)) - float(station_start), 1)
            if go_start is not None and go_end is None:
                out["go_s"] = round(float(r.get("t_s", 0.0)) - float(go_start), 1)
        elif k == "sectors" and r.get("phase") == "GO" and abs_hdg is not None:
            bins = r.get("bins") or []
            p = r.get("p") or []
            ranges = r.get("min_range_m") or [None] * len(bins)
            hdg = r.get("heading")
            if hdg is None or not bins:
                continue
            course_rel = _wrap180(float(abs_hdg) - float(hdg))
            for b, pb, rg in zip(bins, p, ranges):
                if abs(float(b) - course_rel) > HALF_DEG:
                    continue
                if pb is not None and (out["max_p_course"] is None or pb > out["max_p_course"]):
                    out["max_p_course"] = round(float(pb), 2)
                if rg is not None and (out["closest_m"] is None or rg < out["closest_m"]):
                    out["closest_m"] = round(float(rg), 1)
                    out["closest_deg"] = float(b)
    return out


COLUMNS = [("trial", "trial"), ("name", "name"), ("armed", "armed"),
           ("decisions", "decisions"),
           ("refused", "refused"), ("n_records_hold", "records in window"),
           ("t_decide_s", "t to heading (s)"), ("rel_heading_deg", "rel hdg (deg)"),
           ("abs_heading_deg", "abs hdg (deg)"), ("field_p", "p at choice"),
           ("checkpoint_m", "checkpoint (m)"), ("go_s", "GO (s)"),
           ("station_s", "station (s)"), ("station_pattern", "station"),
           ("outcome", "outcome"), ("closest_m", "closest radar on course (m)"),
           ("max_p_course", "max p on course"), ("note", "note")]

NOTE_CHARS = 40          # the markdown table truncates; the CSV keeps it whole


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v) >= 10 or v == int(v) else f"{v:.2f}"
    return str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="directories or sector_experiment_*.jsonl files")
    ap.add_argument("--csv", default=None, help="also write the per-trial rows here")
    args = ap.parse_args(argv)
    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "sector_experiment_*.jsonl"), recursive=True))
        else:
            files += sorted(glob.glob(p))
    if not files:
        print("no sector_experiment_*.jsonl found", file=sys.stderr)
        return 1
    trials = []
    for f in files:
        rows = load_trial(f)
        idx = sorted({i for i in map(_trial_index, rows) if i > 0})
        if len(idx) <= 1:
            trials.append(summarise(f, rows))
        else:                                   # --trials N: one row per trial
            for k in idx:
                # trial 0 = the start/hardware rows written before the first
                # trial: they describe the launch, so every trial gets them
                sub = [r for r in rows if _trial_index(r) in (0, k)]
                t = summarise(f, sub); t["trial"] = f"{t['trial']}#{k}"
                trials.append(t)
    print("| " + " | ".join(h for _, h in COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for t in trials:
        cells = []
        for k, _ in COLUMNS:
            v = _fmt(t[k])
            if k == "note" and len(v) > NOTE_CHARS:
                v = v[:NOTE_CHARS - 1] + "…"
            cells.append(v.replace("|", "/"))
        print("| " + " | ".join(cells) + " |")
    n = len(trials)
    reached = sum(t["outcome"] == "reached" for t in trials)
    aborts = sum(t["outcome"] == "abort" for t in trials)
    decided = [t for t in trials if t["t_decide_s"] is not None]
    med = (sorted(t["t_decide_s"] for t in decided)[len(decided) // 2] if decided else None)
    closest = [t["closest_m"] for t in trials if t["closest_m"] is not None]
    kept = [t["station_s"] for t in trials if t["station_s"] is not None]
    print(f"\n{n} trials · {len(decided)} took a heading · {reached} reached · {aborts} aborted"
          f" · refused decisions {sum(t['refused'] for t in trials)}"
          f" · median t to heading {_fmt(med)} s"
          f" · closest radar on course {_fmt(min(closest)) if closest else '—'} m"
          + (f" · {len(kept)} held station, {_fmt(sum(kept))} s total" if kept else ""))
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[k for k, _ in COLUMNS])
            w.writeheader()
            for t in trials:
                w.writerow({k: t[k] for k, _ in COLUMNS})
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
