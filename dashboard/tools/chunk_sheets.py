"""Contact sheets of every capture CHUNK's first + last decodable frame,
for chunk-level indoor/pontoon screening (activities are screened per chunk:
missions start and end at the dock with hours of real footage between).

Uses the same discovery/skips as bake_corpus, so the sheets cover exactly
the chunks a bake would consider.

    .venv/bin/python dashboard/tools/chunk_sheets.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_corpus import (  # noqa: E402
    MIN_YEAR,
    SKIP_SCENES,
    discover_nested,
    parse_ts,
)

from scripts.utils.datasets import list_triplets  # noqa: E402

PER_SHEET = 12
THUMB_W, THUMB_H = 320, 240
LABEL_H = 22


def first_last_frames(video: Path):
    cap = cv2.VideoCapture(str(video))
    try:
        ok, first = cap.read()
        if not ok:
            return None, None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last = None
        for back in range(1, 12):  # some tails have unreadable trailing frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(n - back, 0))
            ok, frame = cap.read()
            if ok:
                last = frame
                break
        return first, last
    finally:
        cap.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    triplets = sorted(
        (t for t in list_triplets() + discover_nested()
         if not any(t.scene.startswith(s) for s in SKIP_SCENES)
         and parse_ts(t.timestamp).year >= MIN_YEAR),
        key=lambda t: (t.scene, t.timestamp),
    )

    cells: list[np.ndarray] = []
    index: list[dict] = []
    for n, t in enumerate(triplets):
        first, last = first_last_frames(t.fisheye)
        pair = []
        for which, img in (("first", first), ("last", last)):
            if img is None:
                img = np.zeros((THUMB_H, THUMB_W, 3), np.uint8)
            img = cv2.resize(img, (THUMB_W, THUMB_H))
            bar = np.full((LABEL_H, THUMB_W, 3), 255, np.uint8)
            cv2.putText(bar, f"#{n:02d} {which}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                        cv2.LINE_AA)
            pair.append(np.vstack([bar, img]))
        cells.append(np.hstack(pair))
        index.append({"n": n, "chunk_id": t.clip_id,
                      "readable": first is not None})

    sheet_idx = 0
    for start in range(0, len(cells), PER_SHEET):
        chunk = cells[start:start + PER_SHEET]
        rows = []
        for r in range(0, len(chunk), 2):
            row = chunk[r:r + 2]
            while len(row) < 2:
                row.append(np.full_like(chunk[0], 255))
            rows.append(np.hstack(row))
        cv2.imwrite(str(args.out / f"sheet_{sheet_idx:02d}.jpg"),
                    np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
        sheet_idx += 1

    (args.out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(cells)} chunks -> {sheet_idx} sheets in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
