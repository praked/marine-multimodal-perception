"""Column-width vector figures for the ICRA 2027 paper (2026-09 edit).

    python -m scripts.eval.paper_figures_2026_09 --out AuthorOne_ICRA2027_AutonomySuite_2.0/res

Writes, per figure, a PDF (vector) and a 300 dpi PNG twin:

  stage_budget_paper       per-frame stage budget on the Raspberry Pi 4 at
                           full clock (results/pi_timing/pi_timing_2026-09-03.json
                           for the two classical rows; the seg-primary rows
                           come from --segprimary-json when it is given, else
                           from the values recorded in docs/reference/pi_timing.md
                           section 4b, see SEGPRIMARY_FALLBACK)
  shadow_modes_paper       the three on-vessel configurations beside capture
                           (results/pi_timing/shadow_2026-09-07_*.jsonl)
  range_height_bars_paper  monocular range vs radar on the isolated wading
                           target at the two camera heights (2026-08-19)

House rules: 3.5 in wide, at most 2.3 in tall, 8 pt text with 7 pt ticks and
annotations, no title, white background, no em or en dashes, legible in
grayscale (lightness ladder plus hatches), bbox_inches="tight".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

WIDTH_IN = 3.5
MAX_HEIGHT_IN = 2.3
CAMERA_PERIOD_MS = 333.3
CAMERA_HZ = 3.0
DPI_PNG = 300

# Grayscale-safe ladder: dark solid, mid hatched, light hatched, ...
DARK, MID, LIGHT, PALE = "#1f4e79", "#4f9d69", "#b3a6d6", "#e5b35a"
GREY = "#cfcfcf"
GATE = "#b5541c"


def style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5, "legend.columnspacing": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": 0.4, "axes.unicode_minus": False,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })


PAD_IN = 0.02


def fit_width(fig, width_in: float = WIDTH_IN, iters: int = 4) -> None:
    """Rescale the figure so that the tight bounding box is `width_in` wide.

    Text is fixed in points, so the tight box does not scale linearly with
    the figure; a few iterations converge to well under 0.01 in.
    """
    for _ in range(iters):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * (width_in - 2 * PAD_IN) / bb.width, h)


def save(fig, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    fit_width(fig)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": DPI_PNG})):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=PAD_IN, **kw)
        paths.append(p)
        print("wrote", p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# 1. stage budget

STAGES = [  # (legend label, stage key, colour, hatch)
    ("fisheye classical", "fisheye", DARK, None),
    ("thermal classical", "thermal", PALE, "\\\\\\\\"),
    ("radar", "mmwave", LIGHT, "...."),
    ("scorer", "score", MID, "////"),
    ("fusion / misc", "rest", GREY, "xxxx"),
]

#: Rows taken from the canonical 2026-09-03 report, keyed by report label.
STAGE_ROWS_FROM_JSON = [
    ("classical + scorer", "classical + v1b scorer"),
    ("+ seg 512", "+ seg student 512 int8"),
]
#: Rows from the 2026-09-07 bench (pi_timing_2026-09-07_segprimary_v2.json).
SEGPRIMARY_LABELS = [
    ("seg-primary", "SEG-PRIMARY: seg 512 + lazy horizon + seg detector"),
    ("seg-primary + YOLOv8n", "SEG-PRIMARY + YOLOv8n-640 worker"),
]
#: Fallback when that JSON is not on disk (it was not archived; the Pi holds
#: it under ~/results/).  seg-primary: the stage medians written in
#: docs/reference/pi_timing.md section 4b (tick 171 = 114 + 6 + 18 + 33 rest;
#: scorer 144).  seg-primary + YOLOv8n: tick 236 and total 422 from the same
#: doc and the committed images/pi_timing/stage_budget.png; the split of the
#: 236 ms tick (fisheye 168, thermal 7, radar 20, rest 41) was measured from
#: that PNG's bar segments (about 3 ms per segment), scorer = 422 - 236.
SEGPRIMARY_FALLBACK = {
    "seg-primary": {"fisheye": 114.0, "thermal": 6.0, "mmwave": 18.0,
                    "score": 144.0, "tick": 171.0},
    "seg-primary + YOLOv8n": {"fisheye": 168.0, "thermal": 7.0, "mmwave": 20.0,
                              "score": 186.0, "tick": 236.0},
}
TAIL_LABELS = ("tail", "radar + IMU only", "radar+IMU")


def _p50(st: dict) -> float:
    if st.get("p50_full_ms") is not None:
        return float(st["p50_full_ms"])
    return float(st.get("p50_all_ms") or 0.0)


def _parts_from_row(row: dict) -> dict:
    st = row["stages"]
    parts = {k: (_p50(st[k]) if st.get(k, {}).get("n") else 0.0)
             for k in ("fisheye", "thermal", "mmwave", "score")}
    parts["tick"] = _p50(st["tick"])
    return parts


def _finish(parts: dict) -> dict:
    """tick excludes the scorer (timed outside process_frame); total = tick + scorer."""
    p = dict(parts)
    p["rest"] = max(0.0, p["tick"] - p["fisheye"] - p["thermal"] - p["mmwave"])
    p["total"] = p["tick"] + p["score"]
    return p


def stage_rows(report: dict, segprimary: dict | None) -> tuple[list[tuple[str, dict]], list[str]]:
    by_label = {r["label"]: r for r in report.get("pipeline", []) if "stages" in r}
    rows, notes = [], []
    for short, label in STAGE_ROWS_FROM_JSON:
        rows.append((short, _finish(_parts_from_row(by_label[label]))))
    seg_by_label = {r["label"]: r for r in (segprimary or {}).get("pipeline", []) if "stages" in r}
    for short, label in SEGPRIMARY_LABELS:
        hit = next((r for lab, r in seg_by_label.items() if lab.startswith(label)), None)
        if hit is not None:
            rows.append((short, _finish(_parts_from_row(hit))))
        else:
            rows.append((short, _finish(SEGPRIMARY_FALLBACK[short])))
            notes.append(f"{short}: reconstructed (no JSON on disk)")
    tail = next((r for lab, r in {**by_label, **seg_by_label}.items()
                 if lab.lower().startswith(TAIL_LABELS)), None)
    if tail is not None:
        rows.append(("radar + IMU (tail)", _finish(_parts_from_row(tail))))
    else:
        notes.append("no tail-mode pipeline row in any report: omitted")
    return rows, notes


def fig_stage_budget(report: dict, segprimary: dict | None, out: Path) -> dict:
    rows, notes = stage_rows(report, segprimary)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 1.75))
    y = np.arange(n)[::-1]
    for yi, (short, p) in zip(y, rows):
        left = 0.0
        for _, key, col, hatch in STAGES:
            w = p[key]
            ax.barh(yi, w, 0.62, left=left, color=col, hatch=hatch,
                    edgecolor="black", linewidth=0.3)
            left += w
        ax.text(left + 8, yi, f"{p['total']:.0f} ms", va="center", ha="left", fontsize=7)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.axvline(CAMERA_PERIOD_MS, color=GATE, lw=0.8, ls=(0, (4, 2)))
    ax.set_ylim(-0.55, n - 0.45 + 0.55)
    ax.text(CAMERA_PERIOD_MS + 6, n - 0.45 + 0.5, "camera period 333 ms",
            color=GATE, fontsize=7, va="top", ha="left")
    xmax = max(p["total"] for _, p in rows)
    ax.set_xlim(0, xmax * 1.16)
    ax.set_xlabel("ms per frame (median over 60 frames)")
    handles = [Patch(facecolor=c, hatch=h, edgecolor="black", linewidth=0.3, label=l)
               for l, _, c, h in STAGES]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.0),
              ncol=3, fontsize=7, borderaxespad=0.2)
    save(fig, out, "stage_budget_paper")
    return {"rows": [(s, {k: round(v, 1) for k, v in p.items()}) for s, p in rows],
            "notes": notes}


# ---------------------------------------------------------------------------
# 2. shadow modes beside capture

SHADOW_FILES = [  # (short label, log file)
    ("radar+IMU", "shadow_2026-09-07_13-48-07.jsonl"),
    ("seg", "shadow_2026-09-07_13-57-41.jsonl"),
    ("seg+YOLO", "shadow_2026-09-07_13-51-01.jsonl"),
]
#: Same words, broken so three labels fit under 0.8 in wide panels.
TICK_LABELS = {"radar+IMU": "radar\n+IMU", "seg": "seg", "seg+YOLO": "seg\n+YOLO"}


def shadow_rows(shadow_dir: Path) -> list[dict]:
    from scripts.eval.shadow_timing_report import load, summarise
    rows = []
    for label, name in SHADOW_FILES:
        recs = load(str(shadow_dir / name))
        if not recs:
            raise SystemExit(f"no records in {shadow_dir / name}")
        rows.append(summarise(label, recs))
    return rows


def fig_shadow_modes(rows: list[dict], out: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 1.7),
                             gridspec_kw={"wspace": 0.5, "width_ratios": [1.1, 1, 1]})
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    bar_kw = dict(edgecolor="black", linewidth=0.3)

    # (a) tick p50 / p95
    ax = axes[0]
    p50 = [r["tick_p50"] for r in rows]
    p95 = [r["tick_p95"] for r in rows]
    ax.bar(x - 0.2, p50, 0.4, color=DARK, label="p50", **bar_kw)
    ax.bar(x + 0.2, p95, 0.4, color=LIGHT, hatch="////", label="p95", **bar_kw)
    for xi, v in zip(x - 0.2, p50):
        ax.text(xi, v + 40, f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
    for xi, v in zip(x + 0.2, p95):
        ax.text(xi, v + 40, f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axhline(CAMERA_PERIOD_MS, color=GATE, lw=0.8, ls=(0, (4, 2)), label="333 ms")
    ax.set_ylim(0, 2750)
    ax.set_yticks([0, 500, 1000, 1500, 2000, 2500])
    ax.set_ylabel("tick (ms)")
    ax.set_title("(a) tick", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=7,
              handlelength=1.2, borderaxespad=0.0, labelspacing=0.2)

    # (b) achieved sector rate
    ax = axes[1]
    rate = [r["rate_hz"] for r in rows]
    ax.bar(x, rate, 0.55, color=[DARK, MID, LIGHT], hatch=[None, "////", "...."], **bar_kw)
    for xi, v in zip(x, rate):
        ax.text(xi, v + 0.06, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(CAMERA_HZ, color=GATE, lw=0.8, ls=(0, (4, 2)))
    ax.text(2.55, CAMERA_HZ - 0.08, "camera\n3 Hz", color=GATE, fontsize=7, ha="right",
            va="top", linespacing=1.0)
    ax.set_ylim(0, 3.6)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("sector records / s")
    ax.set_title("(b) rate", loc="left")

    # (c) fresh evidence per tick
    ax = axes[2]
    seg = [r["seg_fresh"] for r in rows]
    yolo = [r["yolo_fresh"] for r in rows]
    for i, (vals, off, col, hatch, lab) in enumerate((
            (seg, -0.2, DARK, None, "fresh mask"),
            (yolo, 0.2, LIGHT, "////", "fresh typed boxes"))):
        for xi, v in zip(x, vals):
            if v is None or (lab == "fresh mask" and rows[int(xi)]["camera"] is None):
                ax.text(xi + off, 0.03, "n/a", ha="center", va="bottom", fontsize=7,
                        rotation=90, color="0.35")
                continue
            ax.bar(xi + off, v, 0.4, color=col, hatch=hatch, label=lab if xi == 1 or (lab != "fresh mask" and xi == 2) else None, **bar_kw)
            ax.text(xi + off, v + 0.03, f"{100 * v:.0f} %", ha="center", va="bottom",
                    fontsize=7, rotation=90)
    ax.set_ylim(0, 1.75)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("fraction of ticks")
    ax.set_title("(c) fresh evidence", loc="left")
    handles = [Patch(facecolor=DARK, edgecolor="black", linewidth=0.3, label="mask"),
               Patch(facecolor=LIGHT, hatch="////", edgecolor="black", linewidth=0.3, label="boxes")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=7,
              handlelength=1.2, borderaxespad=0.0, labelspacing=0.2)

    for ax in axes:
        ax.set_xticks(x, [TICK_LABELS[l] for l in labels])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlim(-0.6, 2.6)
    save(fig, out, "shadow_modes_paper")
    return {"rows": [{k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in r.items() if not k.startswith("_")} for r in rows]}


# ---------------------------------------------------------------------------
# 3. range vs radar at two camera heights

RANGE_METRICS = ["median |error|", "mean error", "outside the gate"]
RANGE_VALUES = {  # percent; 2026-08-19 afloat calibration, n = 11, 6.9 to 8.9 m
    "0.27 m (dry)": [8.3, 9.4, 27.0],
    "0.25 m (afloat)": [3.5, 1.3, 0.0],
}


def fig_range_height(out: Path) -> dict:
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 1.9))
    x = np.arange(len(RANGE_METRICS))
    w = 0.36
    series = [("0.27 m (dry)", -w / 2, LIGHT, "////"), ("0.25 m (afloat)", w / 2, DARK, None)]
    for lab, off, col, hatch in series:
        vals = RANGE_VALUES[lab]
        ax.bar(x + off, vals, w, color=col, hatch=hatch, edgecolor="black", linewidth=0.3, label=lab)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.5, f"{v:.1f} %", ha="center", va="bottom", fontsize=7)
    ax.axhline(20, color=GATE, lw=0.8, ls=(0, (4, 2)))
    ax.text(1.5, 20.6, "20 % gate", color=GATE, fontsize=7, ha="center", va="bottom")
    ax.set_xticks(x, RANGE_METRICS)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("range error (%)")
    ax.set_ylim(0, 31)
    ax.set_yticks([0, 10, 20, 30])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7, borderaxespad=0.2,
              title="camera height", title_fontsize=7, alignment="left")
    ax.text(1.1, 30.0, "n = 11 detections\n6.9 to 8.9 m", fontsize=7, ha="center", va="top",
            linespacing=1.05)
    save(fig, out, "range_height_bars_paper")
    return {"values": RANGE_VALUES}


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="AuthorOne_ICRA2027_AutonomySuite_2.0/res")
    ap.add_argument("--timing-json", default="results/pi_timing/pi_timing_2026-09-03.json")
    ap.add_argument("--segprimary-json", default="results/pi_timing/pi_timing_2026-09-07_segprimary_v2.json",
                    help="2026-09-07 bench report; falls back to the documented values when absent")
    ap.add_argument("--shadow-dir", default="results/pi_timing")
    ap.add_argument("--which", nargs="*", default=["stage", "shadow", "range"],
                    choices=["stage", "shadow", "range"])
    args = ap.parse_args(argv)
    style()
    out = Path(args.out)
    summary = {}
    if "stage" in args.which:
        report = json.loads(Path(args.timing_json).read_text())
        seg = Path(args.segprimary_json)
        seg_report = json.loads(seg.read_text()) if seg.is_file() else None
        if seg_report is None:
            print(f"note: {seg} not found; seg-primary rows use SEGPRIMARY_FALLBACK")
        summary["stage"] = fig_stage_budget(report, seg_report, out)
    if "shadow" in args.which:
        summary["shadow"] = fig_shadow_modes(shadow_rows(Path(args.shadow_dir)), out)
    if "range" in args.which:
        summary["range"] = fig_range_height(out)
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
