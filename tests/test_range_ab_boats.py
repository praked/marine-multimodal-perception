"""Tests for scripts/eval/range_ab_boats.py: bbox-bottom vs seg-contact ranging
of typed YOLO boat boxes, on synthetic detections/masks/frames."""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
from PIL import Image

from scripts.eval.range_ab_boats import main
from scripts.utils.segmentation import OBSTACLE, SKY, WATER

CLIP = "Synth__2099-01-01_00-00-00"
FID = "Synth/2099-01-01_00-00-00/ts=00-00-00.0"
W, H = 864, 648


def _mask():
    """Sky top third, water below; a boat-shaped obstacle whose hull sits on
    the water around rows 300-360, cols 380-420 (matches the test box)."""
    m = np.full((H, W), WATER, np.uint8)
    m[:300, :] = SKY
    m[300:360, 380:420] = OBSTACLE
    return m


def _norm(x0, y0, x1, y1):
    return [x0 / W, y0 / H, x1 / W, y1 / H]


def _setup(tmp_path):
    det_root = tmp_path / "det"
    seg_root = tmp_path / "seg"
    frames_root = tmp_path / "frames"
    det_root.mkdir()
    (seg_root / CLIP).mkdir(parents=True)
    (frames_root / CLIP).mkdir(parents=True)

    Image.fromarray(_mask()).save(seg_root / CLIP / "ts=00-00-00.0.png")
    cv2.imwrite(str(frames_root / CLIP / "ts=00-00-00.0.jpg"),
                np.full((H, W, 3), 90, np.uint8))

    records = [
        {"frame_id": FID, "fisheye_bboxes": [
            # boat over-capturing water below its hull -> seg range farther
            {"cls": "boat_ship", "xyxy": _norm(380, 300, 420, 400), "confidence": 0.9},
            {"cls": "boat_ship", "xyxy": _norm(380, 300, 420, 400), "confidence": 0.1},  # low conf
            {"cls": "swimmer", "xyxy": _norm(380, 300, 420, 400), "confidence": 0.9},    # class filtered
            # box fully in the sky -> neither estimate -> dropped
            {"cls": "boat_ship", "xyxy": _norm(600, 10, 660, 100), "confidence": 0.9},
        ]},
        {"frame_id": "Synth/2099-01-01_00-00-00/ts=00-00-01.0",
         "fisheye_bboxes": [   # frame with no jpg/mask on disk -> skipped
            {"cls": "boat_ship", "xyxy": _norm(380, 300, 420, 400), "confidence": 0.9}]},
        {"frame_id": "Synth/2099-01-01_00-00-00/ts=00-00-02.0",
         "fisheye_bboxes": []},   # no qualifying boxes
    ]
    (det_root / f"{CLIP}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    return det_root, seg_root, frames_root


def test_main_ranges_and_reports(tmp_path, capsys):
    det_root, seg_root, frames_root = _setup(tmp_path)
    out = tmp_path / "out"
    rc = main(["--det-root", str(det_root), "--seg-root", str(seg_root),
               "--frames-root", str(frames_root), "--out", str(out),
               "--top", "5"])
    assert rc == 0

    with open(out / "range_ab_boats.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1                       # only the confident boat ranged
    r = rows[0]
    assert r["cls"] == "boat_ship"
    bbox_r, seg_r = float(r["bbox_range_m"]), float(r["seg_range_m"])
    # contact point (~row 359) sits above the box bottom (row 400) -> farther
    assert seg_r > bbox_r > 0
    assert abs(float(r["delta_m"]) - round(seg_r - bbox_r, 2)) < 0.01

    assert (out / "closest_boats.png").stat().st_size > 0
    printed = capsys.readouterr().out
    assert "1 have both estimates" in printed
    assert "% farther: 100%" in printed


def test_main_no_detections_still_writes_csv(tmp_path, capsys):
    det_root = tmp_path / "det"
    det_root.mkdir()
    (det_root / f"{CLIP}.jsonl").write_text(
        json.dumps({"frame_id": FID, "fisheye_bboxes": []}) + "\n")
    out = tmp_path / "out"
    rc = main(["--det-root", str(det_root), "--seg-root", str(tmp_path / "seg"),
               "--frames-root", str(tmp_path / "frames"), "--out", str(out)])
    assert rc == 0
    assert (out / "range_ab_boats.csv").exists()
    assert not (out / "closest_boats.png").exists()   # no top rows -> no sheet
    assert "0 boat detections ranged" in capsys.readouterr().out
