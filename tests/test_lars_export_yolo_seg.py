"""Tests for scripts/lars/export_yolo_seg.py: LaRS panoptic -> YOLOv8-seg labels."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from scripts.lars.export_yolo_seg import (
    convert_split,
    instance_polygons,
    main,
    write_data_yaml,
)


def _encode_id(seg_id):
    """COCO panoptic RGB encoding: id = R + 256 G + 65536 B."""
    return (seg_id % 256, (seg_id // 256) % 256, (seg_id // 65536) % 256)


def _make_lars_seg(root, split, size=(64, 48)):
    w, h = size
    ann_dir = root / "lars_v1.0.0_annotations" / split
    masks_dir = ann_dir / "panoptic_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # instance 300 = boat (cat 11), a solid 20x14 rectangle
    mask = np.zeros((h, w, 3), np.uint8)
    mask[10:24, 8:28] = _encode_id(300)
    Image.fromarray(mask).save(masks_dir / "a.png")

    images = [
        {"id": 0, "width": w, "height": h, "file_name": "a.jpg"},
        {"id": 1, "width": w, "height": h, "file_name": "b.jpg"},
    ]
    annotations = [
        {"image_id": 0, "file_name": "a.png", "segments_info": [
            {"category_id": 11, "id": 300},        # boat with a real mask
            {"category_id": 3, "id": 301},         # stuff -> skipped
            {"category_id": 14, "id": 999},        # id absent from mask -> no polygon
        ]},
        {"image_id": 1, "file_name": "missing.png", "segments_info": []},  # no mask file
        {"image_id": 42, "file_name": "ghost.png", "segments_info": []},   # unknown image
    ]
    (ann_dir / "panoptic_annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations}))
    return root


def test_instance_polygons_rectangle():
    m = np.zeros((48, 64), np.uint8)
    m[10:30, 10:40] = 1
    polys = instance_polygons(m, 64, 48)
    assert len(polys) == 1
    flat = polys[0]
    assert len(flat) >= 6 and len(flat) % 2 == 0
    assert all(0.0 <= v <= 1.0 for v in flat)


def test_instance_polygons_min_area_filters_specks():
    m = np.zeros((48, 64), np.uint8)
    m[5:7, 5:7] = 1                       # 4 px, below min_area=25
    assert instance_polygons(m, 64, 48) == []


def test_instance_polygons_empty_mask():
    assert instance_polygons(np.zeros((10, 10), np.uint8), 10, 10) == []


def test_convert_split_writes_polygon_labels(tmp_path):
    root = _make_lars_seg(tmp_path, "val")
    out = tmp_path / "out"
    n_img, n_obj = convert_split(root, out, "val", limit=None)
    assert n_img == 1 and n_obj == 1
    lines = (out / "labels" / "val" / "a.txt").read_text().strip().splitlines()
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"                # boat_ship
    assert len(parts) >= 7                # cls + >=3 (x,y) pairs
    assert not (out / "labels" / "val" / "b.txt").exists()


def test_convert_split_limit_zero(tmp_path):
    root = _make_lars_seg(tmp_path, "val")
    n_img, n_obj = convert_split(root, tmp_path / "out", "val", limit=0)
    assert n_img == 0 and n_obj == 0


def test_write_data_yaml(tmp_path):
    p = write_data_yaml(tmp_path, images_rel="imgs")
    text = p.read_text()
    assert "train: imgs/train" in text and "nc: 8" in text


def test_main_end_to_end(tmp_path, capsys):
    root = _make_lars_seg(tmp_path, "val")
    out = tmp_path / "yolo_seg"
    rc = main(["--lars-root", str(root), "--out", str(out), "--splits", "val"])
    assert rc == 0
    assert (out / "data.yaml").exists()
    assert "1 instance polygons" in capsys.readouterr().out
