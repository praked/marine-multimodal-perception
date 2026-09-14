"""Figures from a `pi_timing_bench` report (deck style, one PNG per panel).

    python -m scripts.eval.pi_timing_figures results/pi_timing/pi_timing_2026-09-03.json \
        --out images/pi_timing

Panels:
  models_fps.png       every model, frames/s at full clock (2 threads solid,
                       4 threads faint), camera-rate line
  stage_budget.png     per-combo stacked stage budget (ms) against 333 ms
  combos_fps.png       per-combo sustained pipeline rate vs the camera rate,
                       seg-worker Hz annotated
  latency_chain.png    frame -> capture CSV -> shadow tick -> BLE -> boat1
                       (measured medians, from the report's `chain` block or
                       the 2026-09-03 defaults)
Full-clock medians are used wherever >= 3 full-clock samples exist; bars
that had to fall back to all-sample medians are hatched (throttled run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Uni InstitutionOne CD (presentation/tools/palette.py, inlined: that dir is not tracked)
SEEBLAU, DEEP, NAVY, INK = "#00A9E0", "#1487B8", "#0A2A3A", "#17384C"
MUTED, FAINT, PAGE, CARD, LINE = "#4C7B94", "#7FA6BC", "#FCFCFA", "#FFFFFF", "#DCE9F0"
FISHEYE, THERMAL, RADAR = "#1487B8", "#B8741A", "#7A6BB5"
GOOD, WARN, SERIOUS = "#2E8B57", "#D97E0C", "#C0392B"
FAMILY_COLOR = {"seg": DEEP, "thermal": THERMAL, "yolo": RADAR}
CAMERA_FPS = 3.0
BUDGET_MS = 333.0
DPI = 300

#: Measured 2026-09-03 (history doc §9): medians, seconds.
DEFAULT_CHAIN = [
    ("capture → CSV on disk\n(per-loop flush)", 0.10, MUTED),
    ("shadow tick\n(radar+IMU, v1b)", 0.066, DEEP),
    ("tailer poll + encode\n+ BLE notify → boat1", 0.30 - 0.10 - 0.066, RADAR),
]


def style():
    matplotlib.rcParams.update({
        "figure.facecolor": PAGE, "savefig.facecolor": PAGE, "axes.facecolor": CARD,
        "font.family": ["Helvetica Neue", "DejaVu Sans"], "font.size": 13,
        "text.color": INK, "axes.edgecolor": LINE, "axes.labelcolor": INK,
        "axes.titlecolor": NAVY, "axes.titleweight": "bold", "axes.titlesize": 16,
        "axes.labelsize": 13, "axes.linewidth": 1.0, "axes.grid": True,
        "grid.color": LINE, "grid.linewidth": 0.8, "xtick.color": MUTED,
        "ytick.color": MUTED, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 11.5})


def p50(st: dict) -> tuple[float | None, bool]:
    """(median ms, used_full_clock)."""
    if st.get("p50_full_ms") is not None:
        return st["p50_full_ms"], True
    return st.get("p50_all_ms"), False


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=DPI)
    plt.close(fig)
    print("wrote", out / name)


# ---------------------------------------------------------------------------

def models_fps(report: dict, out: Path):
    rows = [r for r in report.get("models", []) if not r.get("missing")]
    if not rows:
        return
    by_label: dict[str, dict] = {}
    for r in rows:
        by_label.setdefault(r["label"], {})[r["threads"]] = r
    labels = list(by_label)
    # slowest first -> plotted bottom-up -> fastest at the top
    order = sorted(labels, key=lambda l: -(p50(by_label[l].get(2) or next(iter(by_label[l].values())))[0] or 1e9))
    n = len(order)
    threads_seen = sorted({r["threads"] for r in rows})
    fig, ax = plt.subplots(figsize=(9.6, 0.42 * n + 2.4),
                           gridspec_kw={"left": 0.34, "right": 0.97, "top": 0.90, "bottom": 0.12})
    y = np.arange(n)
    h = 0.36
    for i, lab in enumerate(order):
        fam = by_label[lab][next(iter(by_label[lab]))]["family"]
        col = FAMILY_COLOR.get(fam, DEEP)
        for k, (thr, off, alpha) in enumerate(((2, -h / 2 - 0.02, 1.0), (4, h / 2 + 0.02, 0.45))):
            r = by_label[lab].get(thr)
            if r is None:
                continue
            ms, full = p50(r)
            if ms is None:
                continue
            fps = 1000.0 / ms
            ax.barh(i + off, fps, h, color=col, alpha=alpha,
                    hatch=None if full else "///", edgecolor=CARD, linewidth=0.5)
            ax.text(fps * 1.08, i + off, f"{fps:.2f}" if fps < 10 else f"{fps:.1f}",
                    va="center", fontsize=10.5, fontweight="bold", color=NAVY, family="Menlo")
    ax.set_yticks(y, order)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 80)
    ax.axvline(CAMERA_FPS, color=WARN, lw=1.4, ls="--")
    ax.text(CAMERA_FPS * 1.06, -0.55, "camera rate 3 fps", color=WARN, fontsize=11,
            fontweight="bold", va="top")
    thr = ("2 threads" if threads_seen == [2]
           else "solid: 2 threads, faint: 4 threads")
    ax.set_xlabel(f"frames / s, Raspberry Pi 4 at full clock, {thr} (log scale)")
    ax.set_title("every model on the box, alone (onnxruntime, CPU)", pad=10)
    ax.set_ylim(-1.0, n - 0.4)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=DEEP, label="fisheye segmentation"),
                       Patch(color=THERMAL, label="thermal segmentation"),
                       Patch(color=RADAR, label="typed detection (YOLO, no NMS)"),
                       Patch(facecolor=CARD, edgecolor=MUTED, hatch="///",
                             label="SoC ≥ 80 °C throughout: all-sample median")],
              loc="lower right", bbox_to_anchor=(1.0, 0.02))
    save(fig, out, "models_fps.png")


SHORT_LABELS = {
    "SEG-PRIMARY: seg 512 + lazy horizon + seg detector": "seg-primary: seg 512, lazy horizon",
    "SEG-PRIMARY + YOLOv8n-640 worker": "seg-primary + YOLOv8n-640 worker",
}


def _short(label: str) -> str:
    for k, v in SHORT_LABELS.items():
        if label.startswith(k):
            return label.replace(k, v)
    return label


def stage_budget(report: dict, out: Path):
    rows = [r for r in report.get("pipeline", []) if "stages" in r]
    if not rows:
        return
    stages = [("fisheye classical (incl. horizon RANSAC)", "fisheye", FISHEYE),
              ("thermal classical", "thermal", THERMAL),
              ("radar", "mmwave", RADAR),
              ("scorer", "score", GOOD),
              ("fusion / association / misc", "rest", MUTED)]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * n + 3.0),
                           gridspec_kw={"left": 0.28, "right": 0.97, "top": 0.80, "bottom": 0.11})
    y = np.arange(n)[::-1]
    any_fallback = False
    for i, r in zip(y, rows):
        st = r["stages"]
        tick, full = p50(st["tick"])
        parts = {}
        for _, key, _ in stages[:-1]:
            v, f = p50(st[key]) if st[key]["n"] else (0.0, True)
            parts[key] = v or 0.0
            full = full and f
        # scorer is timed outside process_frame (in _record): add to the tick
        total = (tick or 0.0) + parts["score"]
        parts["rest"] = max(0.0, total - sum(parts.values()))
        left = 0.0
        for _, key, col in stages:
            w = parts[key]
            ax.barh(i, w, 0.55, left=left, color=col, edgecolor=CARD, linewidth=0.5,
                    hatch=None if full else "///")
            left += w
        any_fallback |= not full
        seg = r.get("seg_infer")
        extra = ""
        if seg and seg.get("n"):
            sm, _ = p50(seg)
            extra = f"\nseg {sm:.0f} ms/mask, {r.get('seg_rate_hz', 0):.1f} Hz"
        ax.text(left + 8, i, f"{total:.0f} ms{extra}", va="center", fontsize=10,
                fontweight="bold", color=NAVY, family="Menlo", linespacing=1.15)
    ax.set_yticks(y, [_short(r["label"]) for r in rows])
    ax.axvline(BUDGET_MS, color=WARN, lw=1.4, ls="--")
    ax.text(BUDGET_MS + 6, n - 0.35, "333 ms\n(3 fps)", color=WARN, fontsize=11, fontweight="bold", va="top")
    ax.set_xlabel("milliseconds per frame, Raspberry Pi 4 at full clock (medians)")
    ax.set_title("where the tick goes: stage budget per configuration", pad=44)
    ax.set_xlim(0, max(BUDGET_MS * 1.3, ax.get_xlim()[1] * 1.22))
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for l, _, c in stages]
    if any_fallback:
        handles.append(Patch(facecolor=CARD, edgecolor=MUTED, hatch="///", label="throttled run"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=10.5)
    save(fig, out, "stage_budget.png")


def combos_fps(report: dict, out: Path):
    rows = [r for r in report.get("pipeline", []) if "stages" in r]
    for r in report.get("pipeline_beside", []):
        if "stages" in r:
            rows.append(dict(r, label=r["label"] + "  (beside capture)"))
    if not rows:
        return
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.6, 0.6 * n + 2.9),
                           gridspec_kw={"left": 0.32, "right": 0.97, "top": 0.82, "bottom": 0.11})
    y = np.arange(n)[::-1]
    for i, r in zip(y, rows):
        tick, full = p50(r["stages"]["tick"])
        sc, _ = p50(r["stages"]["score"]) if r["stages"]["score"]["n"] else (0.0, True)
        fps_full = 1000.0 / (tick + (sc or 0.0)) if tick else 0.0
        fps_wall = r.get("fps_wall") or 0.0
        ax.barh(i + 0.19, fps_full, 0.34, color=DEEP, hatch=None if full else "///", edgecolor=CARD)
        ax.barh(i - 0.19, fps_wall, 0.34, color=FAINT, edgecolor=CARD)
        ax.text(fps_full + 0.08, i + 0.19, f"{fps_full:.2f}", va="center", fontsize=10.5,
                fontweight="bold", color=NAVY, family="Menlo")
        ax.text(fps_wall + 0.08, i - 0.19, f"{fps_wall:.2f}", va="center", fontsize=10.5,
                fontweight="bold", color=MUTED, family="Menlo")
    ax.set_yticks(y, [_short(r["label"]) for r in rows])
    ax.axvline(CAMERA_FPS, color=WARN, lw=1.4, ls="--")
    ax.text(CAMERA_FPS + 0.05, n - 0.35, "camera 3 fps", color=WARN, fontsize=11, fontweight="bold", va="top")
    ax.set_xlabel("pipeline frames / s on the Raspberry Pi 4")
    ax.set_title("configurations: what the box can sustain", pad=40)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=DEEP, label="at full clock (median tick)"),
                       Patch(color=FAINT, label="as measured on the boat feed (wall clock, throttling included)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, fontsize=10.5)
    save(fig, out, "combos_fps.png")


def latency_chain(report: dict, out: Path):
    chain = report.get("chain") or DEFAULT_CHAIN
    fig, ax = plt.subplots(figsize=(9.6, 3.4),
                           gridspec_kw={"left": 0.04, "right": 0.97, "top": 0.80, "bottom": 0.30})
    left = 0.0
    for lab, s, col in chain:
        ax.barh(0, s, 0.5, left=left, color=col, edgecolor=CARD, linewidth=1)
        ax.text(left + s / 2, 0, f"{s*1000:.0f} ms", ha="center", va="center", fontsize=12,
                fontweight="bold", color=CARD, family="Menlo")
        ax.text(left + s / 2, -0.42, lab, ha="center", va="top", fontsize=10.5, color=INK)
        left += s
    ax.text(left + 0.01, 0, f"{left*1000:.0f} ms\nmedian", va="center", fontsize=12,
            fontweight="bold", color=NAVY, family="Menlo")
    ax.set_xlim(0, left * 1.18)
    ax.set_yticks([])
    ax.set_ylim(-1.1, 0.6)
    ax.set_xlabel("seconds after the camera frame")
    ax.set_title("frame → sector record on boat1: measured end-to-end (tail-mode shadow, BLE)", pad=8)
    ax.grid(axis="y", visible=False)
    save(fig, out, "latency_chain.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--out", default="images/pi_timing")
    ap.add_argument("--beside", default=None,
                    help="a pipeline-only report taken beside the running capture "
                         "service; its combos are added to combos_fps")
    args = ap.parse_args(argv)
    style()
    report = json.loads(Path(args.report).read_text())
    if args.beside:
        report["pipeline_beside"] = json.loads(Path(args.beside).read_text()).get("pipeline", [])
    out = Path(args.out)
    models_fps(report, out)
    stage_budget(report, out)
    combos_fps(report, out)
    latency_chain(report, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
