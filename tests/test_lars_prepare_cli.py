"""CLI + error-branch tests for scripts/lars/prepare.py (complements
tests/test_lars_prepare.py, which covers the split-staging internals)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scripts.lars.prepare import main, prepare_split


def _make_lars(root, split, stems, size=(64, 48), skip_masks=()):
    img_dir = root / "lars_v1.0.0_images" / split / "images"
    msk_dir = root / "lars_v1.0.0_annotations" / split / "semantic_masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    for s in stems:
        Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(img_dir / f"{s}.jpg")
        if s not in skip_masks:
            m = np.full((h, w), 1, np.uint8)
            m[: h // 3] = 2
            Image.fromarray(m).save(msk_dir / f"{s}.png")


def test_prepare_split_missing_images_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="images dir"):
        prepare_split(tmp_path, tmp_path / "out", "train", (32, 24))


def test_prepare_split_missing_masks_dir_raises(tmp_path):
    (tmp_path / "lars_v1.0.0_images" / "train" / "images").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="semantic_masks"):
        prepare_split(tmp_path, tmp_path / "out", "train", (32, 24))


def test_prepare_split_limit_and_missing_mask_warning(tmp_path, capsys):
    _make_lars(tmp_path, "train", ["a", "b", "c"], skip_masks={"b"})
    out = tmp_path / "out"
    written = prepare_split(tmp_path, out, "train", (32, 24), limit=2)
    # limit=2 keeps a,b; b has no mask -> skipped with a warning
    assert written == ["a"]
    assert "had no mask" in capsys.readouterr().err


def test_main_end_to_end(tmp_path, capsys):
    _make_lars(tmp_path, "val", ["a", "b"])
    out = tmp_path / "staged"
    rc = main(["--lars-root", str(tmp_path), "--out", str(out),
               "--size", "32x24", "--splits", "val,"])
    assert rc == 0
    assert (out / "val_images.txt").read_text().splitlines() == ["a", "b"]
    assert (out / "mastr_val.yaml").exists()
    assert (out / "images" / "a.jpg").exists()
    assert (out / "masks" / "am.png").exists()
    assert (out / "imus" / "a.png").exists()
    assert "Done." in capsys.readouterr().out
    # staged sizes honour --size (WxH)
    assert Image.open(out / "images" / "a.jpg").size == (32, 24)
