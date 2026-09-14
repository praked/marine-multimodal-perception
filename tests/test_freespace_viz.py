"""Tests for scripts/eval/freespace_viz.py: free-space overlay rendering."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from scripts.eval.freespace_viz import _clipdir, main
from scripts.utils.segmentation import OBSTACLE, SKY, WATER

TS = "2099-01-01_00-00-00"
W, H = 864, 648


def _mask():
    """Sky above row 200, water below, one obstacle band rising from the
    bottom (blocked bins): everything else navigable."""
    m = np.full((H, W), WATER, np.uint8)
    m[:200, :] = SKY
    m[400:, 100:200] = OBSTACLE
    return m


def _setup(tmp_path, synthetic_triplet, n_frames=2):
    frames_root = tmp_path / "frames"
    seg_root = tmp_path / "seg"
    clip = _clipdir("Synth", TS)
    (frames_root / clip).mkdir(parents=True)
    (seg_root / clip).mkdir(parents=True)
    img = np.full((H, W, 3), 90, np.uint8)
    for i in range(n_frames):
        cv2.imwrite(str(frames_root / clip / f"ts=00-00-0{i}.0.jpg"), img)
        Image.fromarray(_mask()).save(seg_root / clip / f"ts=00-00-0{i}.0.png")
    # a frame with no mask -> skipped
    cv2.imwrite(str(frames_root / clip / "ts=00-00-09.0.jpg"), img)
    return frames_root, seg_root


def test_clipdir():
    assert _clipdir("Boats", "x") == "Boats__x"


def test_main_renders_free_space_frames(tmp_path, synthetic_triplet, capsys):
    frames_root, seg_root = _setup(tmp_path, synthetic_triplet)
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    out = tmp_path / "out"
    rc = main(["--triplet", str(prefix), "--seg-root", str(seg_root),
               "--frames-root", str(frames_root), "--out", str(out),
               "--limit", "0"])
    assert rc == 0
    pngs = list((out / _clipdir("Synth", TS)).glob("ts=*.png"))
    assert len(pngs) == 2                       # maskless frame skipped
    assert all(p.stat().st_size > 0 for p in pngs)
    assert "2 free-space frames" in capsys.readouterr().out


def test_main_respects_limit(tmp_path, synthetic_triplet, capsys):
    frames_root, seg_root = _setup(tmp_path, synthetic_triplet)
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    rc = main(["--triplet", str(prefix), "--seg-root", str(seg_root),
               "--frames-root", str(frames_root), "--out", str(tmp_path / "o"),
               "--limit", "1"])
    assert rc == 0
    assert "1 free-space frames" in capsys.readouterr().out
