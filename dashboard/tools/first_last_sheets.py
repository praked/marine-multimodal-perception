"""Compose contact sheets of each baked activity's FIRST and LAST frame,
for visual classification (indoor / pontoon / on-water). Reads the bundle;
writes numbered sheets + an index mapping row labels to clip_ids.

    .venv/bin/python dashboard/tools/first_last_sheets.py --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BUNDLE = Path("/Volumes/ROS2_SSD/asvproject/dashboard_bundle")
PER_SHEET = 12
THUMB_W, THUMB_H = 320, 240
LABEL_H = 22


def safe(ts: str) -> str:
    return ts.replace(":", "-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, default=BUNDLE)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    catalogue = json.loads((args.bundle / "clips.json").read_text())["clips"]
    index: list[dict] = []
    cells: list[np.ndarray] = []

    for n, clip in enumerate(catalogue):
        key = clip["clip_id"].replace("/", "__")
        meta = json.loads((args.bundle / key / "meta.json").read_text())
        first_ts = meta["frames"][0]["ts"]
        last_ts = meta["frames"][-1]["ts"]
        pair = []
        for which, ts in (("first", first_ts), ("last", last_ts)):
            img = cv2.imread(str(args.bundle / key / "frames" / f"ts={safe(ts)}.jpg"))
            if img is None:
                img = np.zeros((THUMB_H, THUMB_W, 3), np.uint8)
            img = cv2.resize(img, (THUMB_W, THUMB_H))
            bar = np.full((LABEL_H, THUMB_W, 3), 255, np.uint8)
            cv2.putText(bar, f"#{n:02d} {which}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            pair.append(np.vstack([bar, img]))
        cells.append(np.hstack(pair))
        index.append({"n": n, "clip_id": clip["clip_id"], "title": clip["title"],
                      "n_frames": clip["n_frames"]})

    per_row = 2  # two activities (four thumbs) per row
    sheet_idx = 0
    for start in range(0, len(cells), PER_SHEET):
        chunk = cells[start:start + PER_SHEET]
        rows = []
        for r in range(0, len(chunk), per_row):
            row = chunk[r:r + per_row]
            while len(row) < per_row:
                row.append(np.full_like(chunk[0], 255))
            rows.append(np.hstack(row))
        sheet = np.vstack(rows)
        cv2.imwrite(str(args.out / f"sheet_{sheet_idx:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        sheet_idx += 1

    (args.out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(cells)} activities -> {sheet_idx} sheets in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
