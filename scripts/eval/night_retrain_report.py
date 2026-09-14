#!/usr/bin/env python3
"""Before/after report for a scorer retrain on the 2026-09-08 lake session.

Two things, both on IDENTICAL inputs:

1. Sensor handover table: mean learned p per sensor-hit combination and light
   band (day / dusk / night) from per-chunk sector JSONL sets, before vs after.
   Also a per-sensor marginal weight (least squares of p on the three hit
   flags), overall and inside the night trial window (20:27-21:10; the pooled
   night number was shown to be a return-to-pontoon artefact, see
   docs/history/2026-09-08_lake_trials.md section 7).

2. Decision re-scoring: for every launch in the boat's sector_experiment logs
   the hold window's records are taken from the AFTER sector set (same
   timestamps as the box streamed), aggregated with the boat policy
   (sector_policy.aggregate + choose_heading, evidence mode from the launch's
   hardware row) and compared with the heading the boat actually took.

    python -m scripts.eval.night_retrain_report \\
        --before results/sectors_night --after results/sectors_night_retrain \\
        --logs <SSD>/captures/2026-09-08_afloat/boat_logs/boat-a \\
        --policy scratch/boatv1-sector-experiment --out results/closed_loop/2026-09-08/retrain_report.md

`--before`/`--after` are directories of `<scene>__<chunk_ts>.jsonl` protocol
records (the baker's sector source). The live shadow records were placed in
results/sectors_night on 2026-09-09; a re-emission with a new bundle goes to
its own directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

CEST = timezone(timedelta(hours=2))
LAT, LON = 47.6956, 9.1938
SCENE = "2026-09-08_afloat_2026-09-08_17-10-55"
NIGHT_TRIALS = ("20:27:00", "21:10:00")


def sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def sun_elev(ts: str) -> float:
    from scripts.sensor_processing.gps_boat1 import sun_position
    h, m, s = ts.split(":")
    t = datetime(2026, 9, 8, int(h), int(m), int(float(s)), tzinfo=CEST)
    return sun_position(LAT, LON, t.astimezone(timezone.utc))[0]


def load_records(d: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(d, f"{SCENE}__*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("timestamp") and r.get("p_obstacle") and r.get("sensor_hits"):
                out.append(r)
    out.sort(key=lambda r: sec(r["timestamp"]))
    return out


COMBOS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
NAMES = {(0, 0, 0): "none", (1, 0, 0): "fisheye only", (0, 1, 0): "thermal only", (0, 0, 1): "radar only",
         (1, 1, 0): "fisheye+thermal", (1, 0, 1): "fisheye+radar", (0, 1, 1): "thermal+radar", (1, 1, 1): "all three"}


def handover(recs: list[dict]) -> dict:
    """{band: {combo: (n, mean p)}, band_fit: [b, wf, wt, wr]} over camera-mode records."""
    rows = []
    cache = {}
    for r in recs:
        # Live shadow records carry fresh-flags; keep only camera-mode ticks.
        # Offline re-emissions (fusion.py) have no `shadow` block and always
        # ran the cameras, so they all count.
        if "shadow" in r:
            sh = r["shadow"] or {}
            if not (sh.get("seg_fresh") or sh.get("yolo_fresh")):
                continue
        ts = r["timestamp"]
        key = ts[:5]
        el = cache.get(key)
        if el is None:
            el = cache[key] = sun_elev(ts)
        for hits, p in zip(r["sensor_hits"], r["p_obstacle"]):
            rows.append((el, sec(ts), tuple(int(x) for x in hits), float(p)))
    bands = {"day": lambda e, t: e > 6, "dusk": lambda e, t: -6 <= e <= 6, "night": lambda e, t: e < -6,
             "night trial window": lambda e, t: sec(NIGHT_TRIALS[0]) <= t <= sec(NIGHT_TRIALS[1])}
    out = {}
    for band, pred in bands.items():
        sel = [x for x in rows if pred(x[0], x[1])]
        if len(sel) < 50:
            continue
        table = {}
        for c in COMBOS:
            m = [x[3] for x in sel if x[2] == c]
            if len(m) >= 20:
                table[c] = (len(m), float(np.mean(m)))
        X = np.array([[1, *x[2]] for x in sel], float)
        y = np.array([x[3] for x in sel], float)
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[band] = {"n": len(sel), "table": table, "fit": [float(v) for v in w]}
    return out


def rescore(after: list[dict], logs_dir: str, policy_dir: str) -> list[dict]:
    sys.path.insert(0, policy_dir)
    import sector_policy as policy  # the boat's policy module, verbatim
    by_t = np.array([sec(r["timestamp"]) for r in after])
    out = []
    for L in sorted(glob.glob(os.path.join(logs_dir, "sector_experiment_20260908_*.jsonl"))):
        rows = []
        for line in open(L):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        hw = next((r for r in rows if r["kind"] == "hardware"), {})
        if hw.get("dry_run", True):
            continue
        st = next((r for r in rows if r["kind"] == "start"), {})
        args = st.get("args", {})
        mode = hw.get("evidence", args.get("evidence", "learned"))
        hold_s = float(args.get("hold_s", 10.0))
        for d in [r for r in rows if r["kind"] == "decide" and r["result"] in ("ok", "fixed")]:
            t1 = sec(d["wall"][11:23])
            i0, i1 = np.searchsorted(by_t, t1 - hold_s), np.searchsorted(by_t, t1 + 0.6)
            recs = after[i0:i1]
            policy.EVIDENCE = mode
            agg = policy.aggregate(recs) if recs else None
            new = policy.choose_heading(agg["bins"], agg["p"], agg["min_range_m"],
                                        {"max_swing_deg": float(args.get("max_swing_deg", 45.0)),
                                         "hard_stop_m": float(args.get("hard_stop_m", 4.0))}) if agg else None
            out.append({"launch": os.path.basename(L)[18:33], "decided_at": d["wall"][11:19],
                        "evidence": mode, "boat_heading": d.get("rel_heading_deg"),
                        "boat_p": d.get("p"), "n_records_after": len(recs),
                        "after_result": None if new is None else new["reason"],
                        "after_heading": None if new is None else new["heading_deg"],
                        "after_p": None if agg is None else [round(v, 2) for v in agg["p"]]})
    return out


def fmt_handover(h: dict, title: str) -> str:
    lines = [f"### {title}", ""]
    for band, v in h.items():
        f = v["fit"]
        lines.append(f"**{band}** (n={v['n']} sector-records): p = {f[0]:.2f} + {f[1]:+.2f}·fisheye {f[2]:+.2f}·thermal {f[3]:+.2f}·radar")
        lines.append("")
        lines.append("| combination | n | mean p |")
        lines.append("|---|---|---|")
        for c in COMBOS:
            if c in v["table"]:
                n, m = v["table"][c]
                lines.append(f"| {NAMES[c]} | {n} | {m:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--policy", default="scratch/boatv1-sector-experiment")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    before, after = load_records(a.before), load_records(a.after)
    print(f"before {len(before)} records, after {len(after)} records")
    hb, ha = handover(before), handover(after)
    md = ["# Night retrain report: 2026-09-08 lake session", "",
          f"before = `{a.before}` ({len(before)} records), after = `{a.after}` ({len(after)} records)", "",
          fmt_handover(hb, "Sensor handover BEFORE"), fmt_handover(ha, "Sensor handover AFTER")]
    if len(after):
        rs = rescore(after, a.logs, a.policy)
        md += ["### Decision re-scoring on the AFTER sectors", "",
               "| launch | decided | evidence | boat heading | after result | after heading | after p (−45…+45) |", "|---|---|---|---|---|---|---|"]
        changed = 0
        for r in rs:
            md.append(f"| {r['launch']} | {r['decided_at']} | {r['evidence']} | {r['boat_heading']} | {r['after_result']} | {r['after_heading']} | {r['after_p']} |")
            if r["after_heading"] is not None and r["boat_heading"] is not None and abs(r["after_heading"] - r["boat_heading"]) >= 15:
                changed += 1
        md += ["", f"{len(rs)} decisions re-scored; {changed} would have moved by ≥ 15°."]
        json.dump(rs, open(os.path.splitext(a.out)[0] + "_decisions.json", "w"), indent=1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(md) + "\n")
    json.dump({"before": {b: {"n": v["n"], "fit": v["fit"]} for b, v in hb.items()},
               "after": {b: {"n": v["n"], "fit": v["fit"]} for b, v in ha.items()}},
              open(os.path.splitext(a.out)[0] + "_handover.json", "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
