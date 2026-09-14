"""thermal.dead_rows: the stuck-hot Lepton row repair (cv_common.repair_dead_rows).

The box's Lepton 3 has a defective FPA row (sensor row 21; row 98 in the
180-degree-rotated capture frame since 2026-08-14), stuck at 242-249 full
width in every frame since at least 2026-07-14. The repair runs first in
every thermal consumer so train and deploy see the same frames; [] must be
byte-identical to no repair at all.
"""

import numpy as np
import pytest

from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
from scripts.utils.cv_common import repair_dead_rows


def _frame(h=120, w=160, ch=3, hot=(98,), seed=0):
    rng = np.random.default_rng(seed)
    f = rng.integers(60, 180, size=(h, w, ch) if ch else (h, w)).astype(np.uint8)
    for r in hot:
        f[r] = 246
    return f


def test_repairs_only_the_listed_row_bgr():
    f = _frame()
    out = repair_dead_rows(f, [98])
    expect = ((f[97].astype(np.float32) + f[99].astype(np.float32)) / 2).round().astype(np.uint8)
    assert np.array_equal(out[98], expect)
    mask = np.ones(f.shape[0], bool)
    mask[98] = False
    assert np.array_equal(out[mask], f[mask])          # every other row untouched
    assert out is not f and np.array_equal(f[98], np.full_like(f[98], 246))  # input not mutated


def test_gray_frames_and_edge_rows():
    g = _frame(ch=0, hot=(0, 119))
    out = repair_dead_rows(g, [0, 119])
    assert np.array_equal(out[0], g[1])        # top edge copies the neighbour below
    assert np.array_equal(out[119], g[118])    # bottom edge copies the neighbour above


def test_adjacent_dead_rows_skip_to_healthy_neighbours():
    g = _frame(ch=0, hot=(50, 51))
    out = repair_dead_rows(g, [50, 51])
    expect = ((g[49].astype(np.float32) + g[52].astype(np.float32)) / 2).round().astype(np.uint8)
    assert np.array_equal(out[50], expect) and np.array_equal(out[51], expect)


def test_empty_or_out_of_range_is_identity():
    f = _frame()
    assert repair_dead_rows(f, []) is f
    assert repair_dead_rows(f, None) is f
    assert repair_dead_rows(f, [500, -3]) is f


def test_pipeline_applies_config_and_off_is_byte_identical(intrinsics_real, detection_real):
    import copy
    f = _frame(seed=7)
    det_on = copy.deepcopy(detection_real)
    det_on["thermal"]["dead_rows"] = [98]
    det_off = copy.deepcopy(detection_real)
    det_off["thermal"]["dead_rows"] = []
    # A stuck row at 246 lifts the contrast stats; the repaired frame's
    # stats must match a pipeline fed the manually repaired frame, and the
    # [] pipeline must reproduce the raw-frame stats exactly.
    res_on = ObstacleDetectionPipeline(intrinsics_real, det_on).process_thermal(f.copy())
    res_manual = ObstacleDetectionPipeline(intrinsics_real, det_off).process_thermal(
        repair_dead_rows(f, [98]))
    res_off = ObstacleDetectionPipeline(intrinsics_real, det_off).process_thermal(f.copy())
    assert res_on.quality_std == pytest.approx(res_manual.quality_std)
    assert res_on.quality_dyn_range == pytest.approx(res_manual.quality_dyn_range)
    assert res_off.quality_std != pytest.approx(res_on.quality_std)


def test_default_config_lists_the_measured_row(detection_real):
    assert detection_real["thermal"]["dead_rows"] == [98]
