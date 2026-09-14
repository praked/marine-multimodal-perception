"""Re-measure the radar self-clutter zone on the CURRENT mount, and simulate a
candidate zone/hatch against the labels before changing detection.yaml.

Run after any remount and after a capture campaign (box_reassembly.md §6):

    python -m scripts.eval.radar_clutter_map --mmwave 'data/captures/2026-08-*/**/mmwave_*.csv'
    python -m scripts.eval.radar_clutter_map --mmwave ... --simulate \
        --hatch 0.30 --x-min -1.0 --x-max 0.25 --y-max 1.3 --features data/features_full

Section 1 prints the dead-Doppler occupancy map (points per frame in 0.25 m
cells inside 2 m): self-clutter is the cell(s) occupied in nearly every frame
with |V| <= one Doppler bin; anything the boat moves relative to (a pontoon,
a moored boat) comes and goes. Section 2 (--simulate) applies the current
and the candidate filter to the raw clouds and counts the near-return
(< 1.3 m) frame x sector rows that survive, split by the fusion targets'
label where a feature table has them, and by origin (in-box escapee with
2-bin Doppler noise, in-box with real Doppler, port fringe, other) -- the
2026-09-04 measurement that set hatch 0.30 / x_min -1.0 / y_max 1.3 and
showed that widening to starboard blinds the pontoon.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.calibration import load_detection  # noqa: E402
from scripts.utils.datasets import load_mmwave_csv  # noqa: E402

DOPPLER_BIN = 0.1217


def clip_id_for(path: str) -> str:
    """data/captures/<mission>[/<sub>]/mmwave_<ts>.csv -> <mission>[/<sub>]/<ts>."""
    p = Path(path)
    ts = p.stem[len("mmwave_"):]
    parts = p.parent.parts
    i = parts.index("captures") + 1 if "captures" in parts else len(parts) - 1
    return "/".join(parts[i:]) + "/" + ts


def load_clouds(patterns):
    files = sorted(set(sum((glob.glob(p, recursive=True) for p in patterns), [])))
    clouds = []
    for f in files:
        try:
            d = load_mmwave_csv(f)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f}: {e}", file=sys.stderr)
            continue
        if d.empty or "V" not in d.columns:
            continue
        clouds.append((clip_id_for(f), d))
    return clouds


def occupancy(clouds, cell=0.25, r_max=2.0):
    X, Y, V, n = [], [], [], 0
    for _, d in clouds:
        n += d["RoundedTime"].nunique()
        X.append(d["X"].to_numpy()); Y.append(d["Y"].to_numpy())
        V.append(np.nan_to_num(d["V"].to_numpy()))
    x, y, v = map(np.concatenate, (X, Y, V))
    near = (y >= 0.5) & (np.hypot(x, y) < r_max)
    dead = np.abs(v) <= DOPPLER_BIN * 1.05
    xe = np.arange(-1.5, 1.51, cell); ye = np.arange(0.5, r_max + 0.01, cell)
    H, _, _ = np.histogram2d(x[near & dead], y[near & dead], bins=[xe, ye])
    print(f"{len(clouds)} chunks, {n} frames, {len(x)} points; near (<{r_max} m) "
          f"{int(near.sum())}, of which dead-Doppler {int((near & dead).sum())}")
    print("dead-Doppler near-return occupancy, points per frame (rows x port- .. starboard+, cols y):")
    print("       y:", " ".join(f"{a:5.2f}" for a in ye[:-1]))
    for i in range(len(xe) - 1):
        print(f"x {xe[i]:+.2f}:", " ".join(f"{H[i, j] / n:5.2f}" for j in range(len(ye) - 1)))
    b = np.round(np.degrees(np.arctan2(x, y)) / 15) * 15
    print("dead-Doppler near points per frame by 15° bin:",
          {int(k): round(c / n, 2) for k, c in pd.Series(b[near & dead]).value_counts().sort_index().items()})
    return n


def labels_from_features(features_root):
    lab = {}
    if not features_root:
        return lab
    for t in glob.glob(str(Path(features_root) / "*" / "targets.csv")):
        d = pd.read_csv(t, usecols=["clip_id", "frame_index", "bin_center_deg", "y_nav"])
        for r in d.itertuples(index=False):
            lab[(r.clip_id, int(r.frame_index), float(r.bin_center_deg))] = r.y_nav
    return lab


def simulate(clouds, cfg: dict, lab: dict, y_min: float, y_max_range: float):
    h, xmin, xmax, ymax = (cfg["keep_if_doppler_above"], cfg["x_min"], cfg["x_max"], cfg["zone_y_max"])
    rows = pos = neg = unl = frames = 0
    origin = {"in_box_2bin_noise": 0, "in_box_real_doppler": 0, "port_fringe": 0, "other": 0}
    per_bin = {}
    for clip, d in clouds:
        x, y, v = d["X"].to_numpy(), d["Y"].to_numpy(), np.nan_to_num(d["V"].to_numpy())
        fidx = pd.factorize(d["RoundedTime"])[0]; frames += fidx.max() + 1
        keep = (y >= y_min) & (y <= y_max_range)
        inbox = (x >= xmin) & (x <= xmax) & (y <= ymax) & (y >= cfg.get("zone_y_min", 0.0))
        keep &= ~(inbox & ~(np.abs(v) > h))
        r = np.hypot(x, y); b = np.clip(np.round(np.degrees(np.arctan2(x, y)) / 15) * 15, -45, 45)
        near = keep & (r < 1.3)
        seen = set()
        for i in np.flatnonzero(near):
            key = (clip, int(fidx[i]), float(b[i]))
            if key in seen:
                continue
            seen.add(key); rows += 1; per_bin[b[i]] = per_bin.get(b[i], 0) + 1
            L = lab.get(key)
            if L is None or pd.isna(L):
                unl += 1
            elif L >= 0.5:
                pos += 1
            else:
                neg += 1
            if (-0.75 <= x[i] <= 0.25) and y[i] <= 1.2:
                origin["in_box_2bin_noise" if abs(v[i]) < 2.2 * DOPPLER_BIN else "in_box_real_doppler"] += 1
            elif x[i] < -0.75 and y[i] < 1.3:
                origin["port_fringe"] += 1
            else:
                origin["other"] += 1
    print(f"  hatch {h:.2f} box x {xmin:+.2f}..{xmax:+.2f} y..{ymax:.2f}: near-return (<1.3 m) frame×sector rows "
          f"{rows} ({rows / max(frames, 1) * 100:.1f}/100 frames) | label POS {pos} NEG {neg} unlabelled {unl} | "
          f"bins -45/-30: {per_bin.get(-45.0, 0)}/{per_bin.get(-30.0, 0)} | origin {origin}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mmwave", nargs="+", required=True, help="glob(s) of mmwave_*.csv (recursive ** ok)")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--features", default=None, help="feature root with targets.csv for the label split")
    ap.add_argument("--hatch", type=float); ap.add_argument("--x-min", type=float)
    ap.add_argument("--x-max", type=float); ap.add_argument("--y-max", type=float)
    args = ap.parse_args(argv)
    clouds = load_clouds(args.mmwave)
    if not clouds:
        print("no radar CSVs matched", file=sys.stderr); return 1
    occupancy(clouds)
    if args.simulate:
        det = load_detection(); mm = det["mmwave"]; cur = dict(mm["self_clutter"])
        cand = dict(cur)
        for k, a in (("keep_if_doppler_above", args.hatch), ("x_min", args.x_min),
                     ("x_max", args.x_max), ("zone_y_max", args.y_max)):
            if a is not None:
                cand[k] = a
        lab = labels_from_features(args.features)
        print("\nsimulation (current detection.yaml vs candidate):")
        simulate(clouds, cur, lab, mm["y_min"], mm["y_max"])
        simulate(clouds, cand, lab, mm["y_min"], mm["y_max"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
