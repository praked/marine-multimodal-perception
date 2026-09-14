"""Crop-extraction geometry for the swarm self-recognition review gallery
(dashboard/tools/swarm_crops/extract_crops.py — not a package, so load it
by path)."""

import importlib.util
from pathlib import Path

_path = (
    Path(__file__).resolve().parents[1]
    / "dashboard" / "tools" / "swarm_crops" / "extract_crops.py"
)
_spec = importlib.util.spec_from_file_location("extract_crops", _path)
extract_crops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_crops)

crop_rect = extract_crops.crop_rect
crop_id_for = extract_crops.crop_id_for


def test_pad_is_15_percent_per_axis():
    # 100x50 px box centred well inside the frame
    rect = crop_rect([100 / 864, 100 / 648, 200 / 864, 150 / 648])
    x0, y0, x1, y1 = rect
    assert (x0, x1) == (85, 215)   # ±15 px (15% of 100)
    assert (y0, y1) == (92, 158)   # ±7.5 px rounded (15% of 50)


def test_clamped_to_image_bounds():
    rect = crop_rect([-0.01, -0.01, 0.5, 0.5])
    x0, y0, x1, y1 = rect
    assert x0 == 0 and y0 == 0
    rect = crop_rect([0.5, 0.5, 1.0, 1.0])
    assert rect[2] == 864 and rect[3] == 648


def test_small_boxes_skipped():
    # 10 px longest side even after padding -> below the 16 px floor
    assert crop_rect([0.5, 0.5, 0.5 + 10 / 864, 0.5 + 8 / 648]) is None
    # 16 px after padding survives (14 px box * 1.3 ≈ 18)
    assert crop_rect([0.5, 0.5, 0.5 + 14 / 864, 0.5 + 14 / 648]) is not None


def test_degenerate_boxes_skipped():
    assert crop_rect([0.5, 0.5, 0.5, 0.6]) is None
    assert crop_rect([0.6, 0.5, 0.5, 0.6]) is None


def test_crop_id_stable_and_distinct():
    fid = "2026-08-26_afloat/2026-08-26_17-06-04/ts=17-06-05.0"
    assert crop_id_for(fid, 0) == crop_id_for(fid, 0)
    assert crop_id_for(fid, 0) != crop_id_for(fid, 1)
    assert len(crop_id_for(fid, 0)) == 40
