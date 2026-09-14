"""thermal.profiles: the sun-elevation-scheduled thermal parameter profiles.

The scheduler wires the 2026-07-09 close-out's time-of-day thermal roles
(docs/guides/capture_runbook.md §3.4) into the pipeline: day = the current
defaults, twilight = the measured CLAHE dock profile, night = CLAHE +
MOG2(h45, v8). The profile CONTENTS are recorded measurements
(docs/history/2026-07-09_thermal_tuning.md); these tests cover the
MECHANISM only: band resolution at the edges, override precedence, state
swapping on a profile change, and the off = byte-identical contract.
"""

import copy

import numpy as np
import pytest

from scripts.sensor_processing.pipeline import (
    ObstacleDetectionPipeline,
    resolve_thermal_profile,
)


@pytest.fixture
def thermal_seq():
    """Deterministic 160x120 BGR frames with structure below a synthetic
    horizon: enough texture that the detectors have something to chew on,
    identical for every pipeline they are fed to."""
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(3):
        img = rng.integers(30, 70, (120, 160, 3), dtype=np.uint8)
        img[:40] = 40                       # uniform "sky"
        y, x = rng.integers(60, 100), rng.integers(30, 120)
        img[y:y + 12, x:x + 12] = 210       # a hot blob
        frames.append(np.ascontiguousarray(img))
    return frames


def _profiles_enabled(detection_real, **extra):
    det = copy.deepcopy(detection_real)
    det["thermal"]["profiles"]["enabled"] = True
    det["thermal"]["profiles"].update(extra)
    return det


# ---------------------------------------------------------------------------
# Band resolution
# ---------------------------------------------------------------------------

def test_resolve_bands_and_edges():
    # Convention matches the enrichment dayparts: day iff elev >= day_min,
    # night iff elev < night_max, twilight between.
    assert resolve_thermal_profile(45.0) == "day"
    assert resolve_thermal_profile(10.0) == "day"          # day edge inclusive
    assert resolve_thermal_profile(9.999) == "twilight"
    assert resolve_thermal_profile(0.0) == "twilight"      # enrichment "golden"
    assert resolve_thermal_profile(-6.0) == "twilight"     # night edge exclusive
    assert resolve_thermal_profile(-6.001) == "night"
    assert resolve_thermal_profile(-40.0) == "night"


def test_resolve_absent_elevation_is_day():
    assert resolve_thermal_profile(None) == "day"
    assert resolve_thermal_profile(float("nan")) == "day"


def test_resolve_custom_bands():
    assert resolve_thermal_profile(5.0, day_min_elev_deg=4.0) == "day"
    assert resolve_thermal_profile(-3.0, night_max_elev_deg=-2.0) == "night"


# ---------------------------------------------------------------------------
# Off = byte-identical
# ---------------------------------------------------------------------------

def test_scheduler_off_is_byte_identical(intrinsics_real, detection_real,
                                         thermal_seq):
    """The repo default (profiles present, enabled: false) must produce
    bit-identical thermal output to a config with no profiles block at all,
    even when an elevation is provided."""
    det_without = copy.deepcopy(detection_real)
    del det_without["thermal"]["profiles"]

    with_block = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    without = ObstacleDetectionPipeline(intrinsics_real, det_without)
    with_block.set_sun_elevation(-20.0)     # ignored while disabled

    for img in thermal_seq:
        a = with_block.process_thermal(img)
        b = without.process_thermal(img)
        np.testing.assert_array_equal(a.obstacle_mask, b.obstacle_mask)
        np.testing.assert_array_equal(a.undistorted, b.undistorted)
        assert a.coords == b.coords
        assert a.angles == b.angles
        assert a.profile is None and b.profile is None
    assert with_block.thermal_average == without.thermal_average


def test_day_profile_matches_defaults(intrinsics_real, detection_real,
                                      thermal_seq):
    """Enabled scheduler at high sun = the day profile = the base defaults:
    output must match the disabled pipeline exactly (deliverable: day clips
    do not change under the day profile)."""
    scheduled = ObstacleDetectionPipeline(
        intrinsics_real, _profiles_enabled(detection_real))
    baseline = ObstacleDetectionPipeline(intrinsics_real, detection_real)
    scheduled.set_sun_elevation(45.0)

    for img in thermal_seq:
        a = scheduled.process_thermal(img)
        b = baseline.process_thermal(img)
        np.testing.assert_array_equal(a.obstacle_mask, b.obstacle_mask)
        assert a.coords == b.coords
        assert a.profile == "day" and b.profile is None
    assert scheduled.thermal_average == baseline.thermal_average


# ---------------------------------------------------------------------------
# Profile activation + switching
# ---------------------------------------------------------------------------

def test_night_profile_activates_mog2_and_back(intrinsics_real,
                                               detection_real, thermal_seq):
    pl = ObstacleDetectionPipeline(
        intrinsics_real, _profiles_enabled(detection_real))
    pl.set_sun_elevation(-10.0)
    out = pl.process_thermal(thermal_seq[0])
    assert out.profile == "night"
    assert pl._thermal_mog2 is not None            # per-pixel background live
    assert float(pl._thermal_mog2_cfg["history"]) == 45
    assert float(pl._thermal_mog2_cfg["var_threshold"]) == 8.0

    pl.set_sun_elevation(45.0)                     # sunrise
    out2 = pl.process_thermal(thermal_seq[1])
    assert out2.profile == "day"
    assert pl._thermal_mog2 is None                # back to running mean


def test_twilight_profile_effective_params(intrinsics_real, detection_real):
    """The twilight profile is the measured dock profile (stage-2 candidate:
    obj 80, med3 + CLAHE(4, 4)) deep-merged over the base block, which
    itself must stay unmutated."""
    det = _profiles_enabled(detection_real)
    pl = ObstacleDetectionPipeline(intrinsics_real, det)
    tw = pl._thermal_profile_params["twilight"]
    assert tw["object_thresh"] == 80
    assert tw["preprocess"]["enabled"] is True
    assert tw["preprocess"]["median_ksize"] == 3
    assert tw["preprocess"]["clahe"]["clip_limit"] == 4.0
    assert tw["preprocess"]["clahe"]["tile"] == 4
    # Un-overridden keys come through from the base block.
    assert tw["contrast_guard"] == det["thermal"]["contrast_guard"]
    assert tw["background"]["method"] == "running_mean"
    # The base config was not mutated by the merge.
    assert det["thermal"]["object_thresh"] == 100
    assert det["thermal"]["preprocess"]["enabled"] is False
    # Night keeps the base object_thresh (unused under mog2) and adds both
    # the preprocess and the measured mog2 background.
    ni = pl._thermal_profile_params["night"]
    assert ni["background"]["method"] == "mog2"
    assert ni["background"]["mog2"]["history"] == 45
    assert ni["preprocess"]["enabled"] is True


# ---------------------------------------------------------------------------
# Override precedence + honest fallback
# ---------------------------------------------------------------------------

def test_forced_profile_beats_elevation(intrinsics_real, detection_real,
                                        thermal_seq):
    pl = ObstacleDetectionPipeline(
        intrinsics_real, _profiles_enabled(detection_real, force="night"))
    pl.set_sun_elevation(45.0)                     # says day; force wins
    out = pl.process_thermal(thermal_seq[0])
    assert out.profile == "night"
    assert pl._thermal_mog2 is not None


def test_unknown_forced_profile_raises(intrinsics_real, detection_real):
    with pytest.raises(ValueError, match="unknown profile"):
        ObstacleDetectionPipeline(
            intrinsics_real, _profiles_enabled(detection_real, force="dusk"))


def test_missing_elevation_resolves_day_and_logs_once(
        intrinsics_real, detection_real, thermal_seq, capsys):
    pl = ObstacleDetectionPipeline(
        intrinsics_real, _profiles_enabled(detection_real))
    for img in thermal_seq:                        # never given an elevation
        out = pl.process_thermal(img)
        assert out.profile == "day"
    err = capsys.readouterr().err
    assert err.count("no sun elevation") == 1      # logged once, not per frame


def test_process_frame_plumbs_sun_elevation(intrinsics_real, detection_real,
                                            thermal_seq):
    pl = ObstacleDetectionPipeline(
        intrinsics_real, _profiles_enabled(detection_real))
    res = pl.process_frame(None, thermal_seq[0], None,
                           sun_elevation_deg=-10.0)
    assert res.thermal.profile == "night"
    # None means "not provided this call": the previous value stays.
    res2 = pl.process_frame(None, thermal_seq[1], None)
    assert res2.thermal.profile == "night"
