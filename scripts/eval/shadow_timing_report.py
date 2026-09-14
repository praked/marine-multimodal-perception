"""Real-time report + figure from shadow-service logs (one or more modes).

    python -m scripts.eval.shadow_timing_report \
        tail=/path/_shadow/shadow_2026-09-08_18-00-00.jsonl \
        full=/path/_shadow/shadow_2026-09-08_18-10-00.jsonl \
        --out images/pi_timing/shadow_modes.png [--md results/pi_timing/shadow_modes.md]

Each positional is ``label=path`` (or just a path; the label is then the file
stem). Per mode it reports what the record carries: tick p50/p95, achieved
sector rate (records per second of source time), source latency, scorer
success, camera/segmentation/typed-detection freshness and skipped sidecar
rows (full mode), and -- when the service ran on the Pi -- the firmware ARM
clock, SoC temperature and throttle flags it sampled (``shadow.board``). The
figure has four panels: tick per mode, rate/budget per mode, evidence
freshness per mode, and clock+temperature over time for every mode that
logged them. Nothing is modelled: a mode without a field shows it blank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FISHEYE, THERMAL, RADAR = "#1487B8", "#B8741A", "#7A6BB5"
NAVY, MUTED, GOOD, WARN, LINE = "#0A2A3A", "#4C7B94", "#2E8B57", "#D97E0C", "#DCE9F0"
CAMERA_PERIOD_MS = 333.3


def _sod(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load(path: str) -> list[dict]:
    out = []
    if not Path(path).is_file():
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def summarise(label: str, recs: list[dict]) -> dict:
    sh = [r.get("shadow", {}) for r in recs]
    ticks = np.array([x.get("tick_ms", np.nan) for x in sh], float)
    t = np.array([_sod(r["timestamp"]) for r in recs if "timestamp" in r], float)
    span = float(t.max() - t.min()) if len(t) > 1 else 0.0
    lat = np.array([x.get("source_latency_s", np.nan) for x in sh], float)
    def frac(key):
        vals = [x.get(key) for x in sh if key in x]
        return (sum(bool(v) for v in vals) / len(vals)) if vals else None
    boards = [x["board"] for x in sh if isinstance(x.get("board"), dict)]
    clock = np.array([b.get("clock_mhz") or np.nan for b in boards], float)
    temp = np.array([b.get("temp_c") or np.nan for b in boards], float)
    thr = [b.get("throttled") for b in boards if b.get("throttled")]
    return {
        "label": label, "n": len(recs), "span_s": span,
        "rate_hz": (len(recs) - 1) / span if span > 0 else None,
        "tick_p50": float(np.nanmedian(ticks)) if len(ticks) else None,
        "tick_p95": float(np.nanpercentile(ticks, 95)) if len(ticks) else None,
        "over_budget": float(np.nanmean(ticks > CAMERA_PERIOD_MS)) if len(ticks) else None,
        "latency_p50": float(np.nanmedian(lat)) if np.isfinite(lat).any() else None,
        "scorer_ok": frac("scorer_ok"), "camera": frac("camera"),
        "seg_fresh": frac("seg_fresh"), "yolo_fresh": frac("yolo_fresh"),
        "skipped_rows": (max(x.get("skipped_rows", 0) for x in sh) if sh else 0),
        "clock_mhz": (float(np.nanmedian(clock)) if np.isfinite(clock).any() else None),
        "clock_min": (float(np.nanmin(clock)) if np.isfinite(clock).any() else None),
        "temp_max": (float(np.nanmax(temp)) if np.isfinite(temp).any() else None),
        "throttled_any": (any(v not in ("0x0", "0") for v in thr) if thr else None),
        "_t": t, "_ticks": ticks, "_clock": clock, "_temp": temp,
        "_board_t": np.array([_sod(r["timestamp"]) for r in recs
                              if isinstance(r.get("shadow", {}).get("board"), dict)], float),
    }


def _f(v, unit="", nd=0):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float) and nd == 0 and abs(v) < 10:
        return f"{v:.2f}{unit}"
    return f"{v:.{nd}f}{unit}" if isinstance(v, float) else f"{v}{unit}"


def table(rows: list[dict]) -> str:
    cols = [("label", "mode"), ("n", "records"), ("rate_hz", "sector rate (Hz)"),
            ("tick_p50", "tick p50 (ms)"), ("tick_p95", "tick p95 (ms)"),
            ("over_budget", "ticks > 333 ms"), ("latency_p50", "source latency (s)"),
            ("scorer_ok", "scorer ok"), ("camera", "camera"), ("seg_fresh", "seg fresh"),
            ("yolo_fresh", "typed fresh"), ("skipped_rows", "rows skipped"),
            ("clock_mhz", "ARM clock median (MHz)"), ("clock_min", "clock min"),
            ("temp_max", "SoC max (°C)"), ("throttled_any", "throttled")]
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r[k]
            if k in ("over_budget", "scorer_ok", "camera", "seg_fresh", "yolo_fresh") and v is not None:
                cells.append(f"{100 * v:.0f}%")
            elif k in ("tick_p50", "tick_p95", "clock_mhz", "clock_min", "temp_max") and v is not None:
                cells.append(f"{v:.0f}" if k != "temp_max" else f"{v:.1f}")
            elif k == "rate_hz" and v is not None:
                cells.append(f"{v:.2f}")
            elif k == "latency_p50" and v is not None:
                cells.append(f"{v:.2f}")
            else:
                cells.append(_f(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def figure(rows: list[dict], out: Path, title: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_board = any(np.isfinite(r["_clock"]).any() or np.isfinite(r["_temp"]).any() for r in rows)
    fig, axes = plt.subplots(1, 4 if has_board else 3, figsize=(13 if has_board else 10, 3.4),
                             gridspec_kw={"wspace": 0.42})
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    ax = axes[0]
    p50 = [r["tick_p50"] or 0 for r in rows]; p95 = [r["tick_p95"] or 0 for r in rows]
    ax.bar(x - 0.18, p50, 0.36, color=NAVY, label="tick p50")
    ax.bar(x + 0.18, p95, 0.36, color=MUTED, label="tick p95")
    ax.axhline(CAMERA_PERIOD_MS, color=WARN, ls="--", lw=1); ax.text(x.max() + 0.45, CAMERA_PERIOD_MS, "333 ms", color=WARN, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(x, labels); ax.set_ylabel("ms"); ax.set_title("tick", fontsize=10); ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    rate = [r["rate_hz"] or 0 for r in rows]
    ax.bar(x, rate, 0.5, color=[GOOD if v >= 1.0 else WARN for v in rate])
    ax.axhline(3.0, color=MUTED, ls=":", lw=1); ax.text(x.max() + 0.45, 3.0, "camera 3 Hz", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.axhline(1.0, color=WARN, ls="--", lw=1); ax.text(x.max() + 0.45, 1.0, "1 Hz floor", color=WARN, fontsize=8, va="bottom", ha="right")
    for xi, v in zip(x, rate):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels); ax.set_ylabel("sector records / s"); ax.set_title("achieved rate", fontsize=10)
    ax = axes[2]
    keys = [("scorer_ok", "scorer", NAVY), ("camera", "camera", MUTED), ("seg_fresh", "seg fresh", FISHEYE), ("yolo_fresh", "typed fresh", THERMAL)]
    w = 0.8 / len(keys)
    for i, (k, lab, c) in enumerate(keys):
        vals = [(r[k] if r[k] is not None else 0) for r in rows]
        ax.bar(x - 0.4 + w * (i + 0.5), vals, w, color=c, label=lab)
    ax.set_ylim(0, 1.05); ax.set_xticks(x, labels); ax.set_title("evidence available per tick", fontsize=10); ax.legend(fontsize=7.5, frameon=False, ncol=2)
    if has_board:
        ax = axes[3]; ax2 = ax.twinx()
        for r, c in zip(rows, [NAVY, FISHEYE, THERMAL, RADAR]):
            if len(r["_board_t"]) and np.isfinite(r["_clock"]).any():
                tt = r["_board_t"] - r["_board_t"].min()
                ax.plot(tt, r["_clock"], color=c, lw=1.4, label=f"{r['label']} clock")
                ax2.plot(tt, r["_temp"], color=c, lw=1, ls="--")
        ax.set_ylabel("ARM clock (MHz)"); ax2.set_ylabel("SoC °C (dashed)"); ax.set_xlabel("s"); ax.set_title("board while running", fontsize=10)
        ax.axhline(600, color=WARN, ls=":", lw=1); ax.legend(fontsize=7.5, frameon=False)
    for a in axes:
        a.grid(color=LINE, lw=0.6, axis="y"); a.spines[["top", "right"]].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=10, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modes", nargs="+", help="label=path.jsonl (or path)")
    ap.add_argument("--out", default=None, help="figure PNG")
    ap.add_argument("--md", default=None, help="write the table here too")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)
    rows = []
    for m in args.modes:
        label, _, path = m.partition("=") if "=" in m else (Path(m).stem, "", m)
        recs = load(path)
        if not recs:
            print(f"no records in {path}", file=sys.stderr)
            return 1
        rows.append(summarise(label, recs))
    md = table(rows)
    print(md)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md + "\n")
    if args.out:
        figure(rows, Path(args.out), args.title)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
