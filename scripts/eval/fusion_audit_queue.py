"""Rank frames for the label-audit sprint: where auditing buys the most.

The scorer's numbers rest on unaudited GroundingDINO pseudo-labels; auditing
every frame is not happening, so rank by expected information: frames where
the label layer and the learned scorer DISAGREE are where either the label
or the model is wrong — both worth a human look. Score per frame:

  disagreement = mean over bins of |p_obstacle - label_bin|
  + a bonus when the disagreeing bin is high-threat (severity-weighted)

Inputs are artefacts that already exist: scored sector JSONLs (p per bin)
and the labels/qwen files (obstacle_bins_fisheye per frame). Output: a CSV
ranked worst-first, one row per frame, ready to drive dashboard audit
sessions (--pseudo flow) clip by clip.

    python -m scripts.eval.fusion_audit_queue \
        --sectors results/sectors_rescored --out results/audit_queue.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.utils.datasets import REPO_ROOT


def load_label_bins(paths: list[Path]) -> dict[str, set[int]]:
    """frame_id -> set of labelled bin centres."""
    out: dict[str, set[int]] = {}
    for p in paths:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = r.get("frame_id")
            if fid:
                out.setdefault(fid, set()).update(
                    int(b) for b in r.get("obstacle_bins_fisheye", []))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sectors", required=True,
                    help="dir of scored per-clip sector JSONLs")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="label JSONLs (default labels/qwen/*.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=500,
                    help="rows to keep (worst-first)")
    args = ap.parse_args(argv)

    label_paths = ([Path(p) for p in args.labels] if args.labels
                   else sorted((REPO_ROOT / "labels" / "qwen").glob("*.jsonl")))
    labels = load_label_bins(label_paths)
    print(f"labels: {len(labels)} frames from {len(label_paths)} files")

    rows = []
    for jf in sorted(Path(args.sectors).glob("*.jsonl")):
        clip = jf.stem
        scene, _, tts = clip.partition("__")
        for line in jf.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            p = r.get("p_obstacle")
            centers = r.get("bin_centers_deg")
            if not p or not centers:
                continue
            ts = r["timestamp"]
            fid = f"{scene}/{tts}/ts={ts.replace(':', '-')}"
            lab = labels.get(fid)
            if lab is None:
                continue  # unlabelled frames go to a separate campaign
            per_bin = [abs(pi - (1.0 if c in lab else 0.0))
                       for pi, c in zip(p, centers)]
            threat = r.get("threat") or [0.0] * len(p)
            score = (sum(per_bin) / len(per_bin)
                     + 0.5 * max((d * t for d, t in zip(per_bin, threat)),
                                 default=0.0))
            worst = max(range(len(per_bin)), key=lambda i: per_bin[i])
            rows.append({
                "clip": clip, "frame_ts": ts, "frame_id": fid,
                "disagreement": round(score, 4),
                "worst_bin_deg": centers[worst],
                "p_at_worst": round(p[worst], 3),
                "label_at_worst": int(centers[worst] in lab),
            })
    rows.sort(key=lambda r: -r["disagreement"])
    rows = rows[:args.top]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["clip", "frame_ts", "frame_id", "disagreement",
                            "worst_bin_deg", "p_at_worst", "label_at_worst"])
        w.writeheader()
        w.writerows(rows)
    by_clip: dict[str, int] = {}
    for r in rows:
        by_clip[r["clip"]] = by_clip.get(r["clip"], 0) + 1
    print(f"wrote {len(rows)} rows -> {out}")
    for c, n in sorted(by_clip.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {c}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
