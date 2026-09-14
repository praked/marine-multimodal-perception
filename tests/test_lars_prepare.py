"""Tests for scripts/lars/prepare.py: LaRS -> MaSTr-format staging.

Synthesises a tiny LaRS-like tree so it runs without the dataset SSD.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scripts.lars import MASTR_IGNORE
from scripts.lars.prepare import (
    _parse_size,
    prepare_split,
    remap_mask,
    write_config,
    write_image_list,
)


def _make_lars(root, split, stems, size=(64, 48)):
    img_dir = root / "lars_v1.0.0_images" / split / "images"
    msk_dir = root / "lars_v1.0.0_annotations" / split / "semantic_masks"
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)
    w, h = size
    for s in stems:
        Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(img_dir / f"{s}.jpg")
        m = np.full((h, w), 1, np.uint8)       # water
        m[: h // 3] = 2                        # sky
        m[h - 5 :, :] = 255                    # ignore strip
        m[h // 2 : h // 2 + 4, :4] = 0         # a little obstacle
        Image.fromarray(m).save(msk_dir / f"{s}.png")


def test_parse_size():
    assert _parse_size("512x384") == (512, 384)
    with pytest.raises(Exception):
        _parse_size("garbage")


def test_remap_mask():
    m = np.array([[0, 1, 2, 255, 7]], np.uint8)
    out = remap_mask(m)
    assert out[0, 0] == 0 and out[0, 1] == 1 and out[0, 2] == 2
    assert out[0, 3] == MASTR_IGNORE and out[0, 4] == MASTR_IGNORE


def test_prepare_split_outputs(tmp_path):
    root = tmp_path / "LaRS"
    stems = ["clipA_0001", "clipA_0002", "clipB_0003"]
    _make_lars(root, "val", stems)

    out = tmp_path / "staged"
    written = prepare_split(root, out, "val", size=(32, 24))
    assert sorted(written) == sorted(stems)

    for s in stems:
        assert (out / "images" / f"{s}.jpg").exists()
        assert (out / "masks" / f"{s}m.png").exists()       # 'm' suffix!
        assert (out / "imus" / f"{s}.png").exists()
        img = Image.open(out / "images" / f"{s}.jpg")
        assert img.size == (32, 24)
        msk = np.array(Image.open(out / "masks" / f"{s}m.png"))
        assert msk.shape == (24, 32)
        # only valid class ids + ignore(4) survive; 255 must be gone
        assert set(np.unique(msk)).issubset({0, 1, 2, MASTR_IGNORE})
        # dummy imu mask is all zeros
        assert np.array(Image.open(out / "imus" / f"{s}.png")).max() == 0


def test_write_config_and_list(tmp_path):
    out = tmp_path
    write_image_list(out, "train", ["a", "b"])
    cfg = write_config(out, "train")
    assert (out / "train_images.txt").read_text().split() == ["a", "b"]
    text = cfg.read_text()
    assert "image_dir: images" in text
    assert "mask_dir: masks" in text
    assert "imu_dir: imus" in text
    assert "image_list: train_images.txt" in text
