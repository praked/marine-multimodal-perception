"""Reflection post-filter: water-fraction + mirror-partner geometry."""
import numpy as np

from scripts.eval.filter_reflections import reflection_indices, water_fraction
from scripts.utils.segmentation import OBSTACLE, WATER

H, W = 120, 160


def seg_with_water_below(row=60):
    seg = np.full((H, W), OBSTACLE, dtype=np.uint8)
    seg[row:, :] = WATER
    return seg


def box(cls, x0, y0, x1, y1):
    return {"cls": cls, "xyxy": [x0, y0, x1, y1]}


def test_water_fraction():
    seg = seg_with_water_below(60)  # bottom half water
    assert water_fraction(seg, [0.0, 0.5, 1.0, 1.0]) == 1.0
    assert water_fraction(seg, [0.0, 0.0, 1.0, 0.5]) == 0.0


def test_mirror_pair_drops_the_lower_box():
    seg = seg_with_water_below(60)
    real = box("person", 0.4, 0.2, 0.5, 0.45)        # above water
    mirror = box("person", 0.4, 0.55, 0.5, 0.8)       # on water, same column
    assert reflection_indices([real, mirror], seg) == {1}


def test_swimmer_without_partner_is_kept():
    seg = seg_with_water_below(60)
    swimmer = box("person", 0.4, 0.6, 0.5, 0.8)
    assert reflection_indices([swimmer], seg) == set()


def test_different_class_above_is_not_a_partner():
    seg = seg_with_water_below(60)
    boat = box("boat", 0.4, 0.2, 0.5, 0.45)
    blob = box("person", 0.4, 0.55, 0.5, 0.8)
    assert reflection_indices([boat, blob], seg) == set()


def test_offset_column_is_not_a_partner():
    seg = seg_with_water_below(60)
    real = box("person", 0.1, 0.2, 0.2, 0.45)
    blob = box("person", 0.6, 0.55, 0.7, 0.8)
    assert reflection_indices([real, blob], seg) == set()


def test_partner_on_water_does_not_suppress():
    # two stacked on-water boxes (wave pattern): neither may suppress the other
    seg = seg_with_water_below(20)
    a = box("duck", 0.4, 0.3, 0.5, 0.4)
    b = box("duck", 0.4, 0.55, 0.5, 0.7)
    assert reflection_indices([a, b], seg) == set()


def test_no_mask_is_a_noop():
    assert reflection_indices([box("person", 0.4, 0.6, 0.5, 0.8)], None) == set()
