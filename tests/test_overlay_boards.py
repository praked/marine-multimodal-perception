"""Tests for scripts/eval/overlay_boards.py: smoke-set comparison boards."""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
from PIL import Image

import scripts.eval.overlay_boards as ob
from scripts.utils.segmentation import SKY, WATER

SCENE, TS = "SceneA", "2026-01-01_00-00-00"
CLIP = f"{SCENE}__{TS}"
W, H = 64, 48


def _orig(i):
    return f"{SCENE}/{TS}/ts=00-00-0{i}.0"


def _fake(i):
    return f"eval_smoke36/2026-06-18_00-00-00/ts=00-00-0{i}.0"


def test_hex_to_rgb():
    assert ob._hex_to_rgb("#ff0080") == (255, 0, 128)
    assert ob._hex_to_rgb("00ff00") == (0, 255, 0)


def test_clipdir_and_frame_jpg(tmp_path):
    assert ob._clipdir(_orig(0)) == CLIP
    p = ob._frame_jpg(tmp_path, _orig(0))
    assert p == tmp_path / CLIP / "ts=00-00-00.0.jpg"


def test_draw_gdino_box_variants():
    rgb = np.zeros((H, W, 3), np.uint8)
    boxes = [
        {"cls": "boat", "xyxy": [0.1, 0.1, 0.6, 0.6]},        # normalised
        {"cls": "unknown_cls", "xyxy": [5, 5, 30, 30]},        # pixels, fallback colour
        {"cls": "person"},                                     # no box -> skipped
        {"cls": "duck", "xyxy": [1, 2, 3]},                    # wrong len -> skipped
    ]
    out = ob._draw_gdino(rgb, boxes)
    assert out.shape == rgb.shape
    assert out.any()                                           # something was drawn
    assert not rgb.any()                                       # original untouched


def _setup(tmp_path):
    frames_root = tmp_path / "frames"
    (frames_root / CLIP).mkdir(parents=True)
    img = np.full((H, W, 3), 90, np.uint8)
    for i in range(2):                       # frame 2 in the mapping has no jpg
        cv2.imwrite(str(frames_root / CLIP / f"ts=00-00-0{i}.0.jpg"), img)

    seg_root = tmp_path / "seg"
    (seg_root / CLIP).mkdir(parents=True)
    m = np.full((H, W), WATER, np.uint8)
    m[: H // 3] = SKY
    Image.fromarray(m).save(seg_root / CLIP / "ts=00-00-00.0.png")

    det_root = tmp_path / "det"
    det_root.mkdir()
    (det_root / f"{CLIP}.jsonl").write_text(json.dumps(
        {"frame_id": _orig(0), "fisheye_bboxes": [
            {"cls": "boat_ship", "xyxy": [0.2, 0.3, 0.6, 0.8],
             "confidence": 0.9}]}) + "\n")

    inst_root = tmp_path / "inst"
    inst_root.mkdir()
    (inst_root / f"{CLIP}.jsonl").write_text(json.dumps(
        {"frame_id": _orig(0), "instances": [
            {"cls": "boat_ship", "confidence": 0.8,
             "polygon": [0.2, 0.4, 0.6, 0.4, 0.6, 0.8, 0.2, 0.8]}]}) + "\n")

    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps(
        {"fake_to_original": {_fake(i): _orig(i) for i in range(3)}}))

    # GroundingDINO audited boxes, keyed by the eval-clip (fake) frame_id.
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "manual.jsonl").write_text("\n".join([
        json.dumps({"frame_id": _fake(0), "fisheye_bboxes": [
            {"cls": "boat", "xyxy": [0.1, 0.1, 0.5, 0.5]}]}),
        "",
        json.dumps({"frame_id": "Other/clip/ts=0", "fisheye_bboxes": []}),
    ]) + "\n")
    return frames_root, seg_root, det_root, inst_root, mapping


def test_main_builds_boards_and_index(tmp_path, monkeypatch, capsys):
    frames_root, seg_root, det_root, inst_root, mapping = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)              # labels/manual.jsonl is cwd-relative
    out = tmp_path / "boards"
    rc = ob.main(["--mapping", str(mapping), "--frames-root", str(frames_root),
                  "--seg-root", str(seg_root), "--det-root", str(det_root),
                  "--inst-root", str(inst_root), "--out", str(out)])
    assert rc == 0

    for v in ob.VARIANTS:
        assert (out / f"board_{v}.png").stat().st_size > 0
    panels = sorted((out / "by_frame").glob("frame_*.png"))
    assert len(panels) == 2                  # third frame jpg missing -> skipped

    with open(out / "index.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["frame_id"] == _orig(0)
    assert rows[0]["n_typed"] == "1" and rows[0]["n_instances"] == "1"
    assert rows[0]["n_gdino"] == "1"
    assert rows[1]["n_typed"] == "0"

    printed = capsys.readouterr().out
    assert "Done. 2 frames." in printed
    assert "[skip] missing frame jpg" in printed


def test_main_without_manual_labels(tmp_path, monkeypatch):
    frames_root, seg_root, det_root, inst_root, mapping = _setup(tmp_path)
    monkeypatch.chdir(tmp_path / "frames")   # no labels/manual.jsonl here
    out = tmp_path / "boards2"
    rc = ob.main(["--mapping", str(mapping), "--frames-root", str(frames_root),
                  "--seg-root", str(seg_root), "--det-root", str(det_root),
                  "--inst-root", str(inst_root), "--out", str(out)])
    assert rc == 0
    assert (out / "index.csv").exists()
