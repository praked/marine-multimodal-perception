"""Tests for scripts/lars/loader.py: pure-numpy LaRS readers."""

from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.lars import LARS_IGNORE, OBSTACLE, SKY, WATER
from scripts.lars.loader import class_fractions, read_image, read_semantic_mask


def test_read_semantic_mask_roundtrip(tmp_path):
    m = np.full((6, 8), WATER, np.uint8)
    m[0, :] = SKY
    m[-1, :] = LARS_IGNORE
    p = tmp_path / "m.png"
    Image.fromarray(m).save(p)
    out = read_semantic_mask(p)
    assert out.shape == (6, 8)
    assert (out == m).all()


def test_class_fractions_ignores_255():
    m = np.full((4, 4), WATER, np.uint8)
    m[0, :] = SKY
    m[1, :] = OBSTACLE
    m[2, :] = LARS_IGNORE
    fr = class_fractions(m)
    # 12 valid pixels: 4 sky, 4 obstacle, 4 water
    assert abs(fr["water"] - 4 / 12) < 1e-9
    assert abs(fr["sky"] - 4 / 12) < 1e-9
    assert abs(fr["obstacle"] - 4 / 12) < 1e-9


def test_class_fractions_all_ignore_no_zero_division():
    m = np.full((3, 3), LARS_IGNORE, np.uint8)
    fr = class_fractions(m)
    assert fr == {"obstacle": 0.0, "water": 0.0, "sky": 0.0}


def test_read_image_converts_to_rgb(tmp_path):
    p = tmp_path / "i.png"
    Image.fromarray(np.zeros((5, 7), np.uint8)).save(p)   # grayscale on disk
    out = read_image(p)
    assert out.shape == (5, 7, 3)
    assert out.dtype == np.uint8
