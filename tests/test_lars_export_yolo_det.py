"""Tests for scripts/lars/export_yolo_det.py: LaRS panoptic -> YOLO det dataset.

Synthesises a tiny LaRS-like tree (panoptic_annotations.json + images) so the
exporter runs without the dataset SSD.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from scripts.lars.export_yolo_det import (
    CATID_TO_YOLO,
    NAMES,
    convert_split,
    main,
    write_data_yaml,
)


def _make_lars_panoptic(root, split, entries, size=(64, 48)):
    """entries: list of (stem, segments_info, write_image)."""
    w, h = size
    img_dir = root / "lars_v1.0.0_images" / split / "images"
    ann_dir = root / "lars_v1.0.0_annotations" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    for i, (stem, segments, write_image) in enumerate(entries):
        images.append({"id": i, "width": w, "height": h,
                       "file_name": f"{stem}.jpg"})
        annotations.append({"image_id": i, "file_name": f"{stem}.png",
                            "segments_info": segments})
        if write_image:
            Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(img_dir / f"{stem}.jpg")
    # one annotation pointing at a non-existent image id (skipped)
    annotations.append({"image_id": 999, "file_name": "ghost.png",
                        "segments_info": []})
    (ann_dir / "panoptic_annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations}))
    return root


def test_catid_mapping_covers_8_classes():
    assert len(NAMES) == 8
    assert CATID_TO_YOLO[11] == 0          # boat_ship
    assert CATID_TO_YOLO[19] == 7          # other
    assert 3 not in CATID_TO_YOLO          # stuff class


def test_convert_split_writes_labels_and_images(tmp_path):
    segs = [
        {"category_id": 11, "bbox": [8, 8, 16, 12], "id": 1},     # boat
        {"category_id": 3, "bbox": [0, 0, 10, 10], "id": 2},      # stuff -> skip
        {"category_id": 14, "bbox": [4, 4, 0, 6], "id": 3},       # zero width -> skip
        {"category_id": 15, "bbox": [-4, -4, 200, 200], "id": 4}, # clamped
    ]
    root = _make_lars_panoptic(tmp_path, "val", [
        ("a", segs, True),
        ("b", [], True),          # background negative -> empty label file
        ("c", segs, False),       # image missing on disk -> skipped
    ])
    out = tmp_path / "out"
    n_img, n_obj = convert_split(root, out, "val", max_side=0, limit=None)
    assert n_img == 2 and n_obj == 2
    lbl_a = (out / "labels" / "val" / "a.txt").read_text().strip().splitlines()
    assert len(lbl_a) == 2
    cls_a = {int(l.split()[0]) for l in lbl_a}
    assert cls_a == {CATID_TO_YOLO[11], CATID_TO_YOLO[15]}
    # every coordinate normalised into [0,1]
    for line in lbl_a:
        assert all(0.0 <= float(v) <= 1.0 for v in line.split()[1:])
    assert (out / "labels" / "val" / "b.txt").read_text() == ""
    assert (out / "images" / "val" / "a.jpg").exists()
    assert not (out / "images" / "val" / "c.jpg").exists()


def test_convert_split_downscales_with_max_side(tmp_path):
    root = _make_lars_panoptic(
        tmp_path, "val", [("big", [], True)], size=(128, 64))
    out = tmp_path / "out"
    convert_split(root, out, "val", max_side=32, limit=None)
    img = Image.open(out / "images" / "val" / "big.jpg")
    assert max(img.size) == 32
    assert img.size == (32, 16)


def test_convert_split_limit(tmp_path):
    root = _make_lars_panoptic(tmp_path, "val", [
        ("a", [], True), ("b", [], True), ("c", [], True)])
    out = tmp_path / "out"
    n_img, _ = convert_split(root, out, "val", max_side=0, limit=1)
    assert n_img == 1


def test_convert_split_missing_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_split(tmp_path, tmp_path / "out", "train", max_side=0, limit=None)


def test_write_data_yaml(tmp_path):
    p = write_data_yaml(tmp_path)
    text = p.read_text()
    assert "nc: 8" in text
    assert "0: boat_ship" in text and "7: other" in text
    assert "train: images/train" in text


def test_main_end_to_end(tmp_path, capsys):
    root = _make_lars_panoptic(tmp_path, "val", [
        ("a", [{"category_id": 11, "bbox": [2, 2, 10, 10], "id": 1}], True)])
    out = tmp_path / "yolo"
    rc = main(["--lars-root", str(root), "--out", str(out), "--splits", "val,"])
    assert rc == 0
    assert (out / "data.yaml").exists()
    assert (out / "labels" / "val" / "a.txt").read_text().startswith("0 ")
    assert "1 images, 1 object instances" in capsys.readouterr().out
