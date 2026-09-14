import pytest

from scripts.sensor_processing.tracker import (
    BBoxTracker,
    SectorTracker,
    iou,
)


# ---------------------------------------------------------------------------
# SectorTracker
# ---------------------------------------------------------------------------

def test_sector_tracker_confirms_after_min_hits():
    t = SectorTracker(n_bins=3, min_hits=3, max_age=5)
    assert t.update([True, False, False]) == [False, False, False]
    assert t.update([True, False, False]) == [False, False, False]
    assert t.update([True, False, False]) == [True, False, False]


def test_sector_tracker_drops_after_max_age():
    t = SectorTracker(n_bins=2, min_hits=2, max_age=3)
    t.update([True, False])
    t.update([True, False])
    assert t.confirmed[0] is True
    for _ in range(3):
        t.update([False, False])
    assert t.confirmed[0] is False


def test_sector_tracker_gated_scores_zeros_unconfirmed():
    t = SectorTracker(n_bins=3, min_hits=1, max_age=5)
    t.update([True, False, True])
    gated = t.gated_scores([1.0, 0.5, 0.5])
    assert gated == [1.0, 0.0, 0.5]


def test_sector_tracker_reset_on_long_miss_before_confirm():
    t = SectorTracker(n_bins=1, min_hits=5, max_age=2)
    t.update([True])
    t.update([True])
    t.update([False])
    t.update([False])
    # Not yet confirmed and now over max_age -> hits reset to 0
    assert t.hits[0] == 0


def test_sector_tracker_invalid_n_bins():
    with pytest.raises(ValueError):
        SectorTracker(n_bins=0)


def test_sector_tracker_invalid_min_hits():
    with pytest.raises(ValueError):
        SectorTracker(n_bins=2, min_hits=0)


def test_sector_tracker_invalid_max_age():
    with pytest.raises(ValueError):
        SectorTracker(n_bins=2, max_age=0)


def test_sector_tracker_update_wrong_length():
    t = SectorTracker(n_bins=3)
    with pytest.raises(ValueError):
        t.update([True, False])


# ---------------------------------------------------------------------------
# IoU + BBoxTracker
# ---------------------------------------------------------------------------

def test_iou_identical_is_one():
    a = (0.0, 0.0, 10.0, 10.0)
    assert iou(a, a) == 1.0


def test_iou_disjoint_is_zero():
    a = (0.0, 0.0, 1.0, 1.0)
    b = (2.0, 2.0, 3.0, 3.0)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = (0.0, 0.0, 2.0, 2.0)
    b = (1.0, 1.0, 3.0, 3.0)
    # intersection 1x1 = 1; union = 4 + 4 - 1 = 7
    assert iou(a, b) == pytest.approx(1 / 7)


def test_iou_zero_area_returns_zero():
    a = (1.0, 1.0, 1.0, 1.0)
    b = (0.0, 0.0, 2.0, 2.0)
    assert iou(a, b) == 0.0


def test_bbox_tracker_invalid_params():
    with pytest.raises(ValueError):
        BBoxTracker(min_hits=0)


def test_bbox_tracker_confirms_persistent_detection():
    t = BBoxTracker(min_hits=3, max_age=5)
    box = (10.0, 10.0, 20.0, 20.0)
    confirmed = t.update([box], scores=[0.9])
    assert confirmed == []
    confirmed = t.update([box], scores=[0.9])
    assert confirmed == []
    confirmed = t.update([box], scores=[0.9])
    assert len(confirmed) == 1
    assert confirmed[0].confirmed is True


def test_bbox_tracker_drops_aged_track():
    t = BBoxTracker(min_hits=1, max_age=2)
    box = (0.0, 0.0, 10.0, 10.0)
    t.update([box], scores=[0.9])
    # Two empty frames -> dropped on second.
    t.update([], scores=[])
    t.update([], scores=[])
    assert len(t.tracks) == 0


def test_bbox_tracker_two_stage_recovery():
    """High-conf first, then low-conf for occlusion recovery."""
    t = BBoxTracker(min_hits=1, max_age=5, iou_threshold_high=0.3, iou_threshold_low=0.05)
    box = (0.0, 0.0, 10.0, 10.0)
    t.update([box], scores=[0.9])
    # Slight shift, low confidence: should still associate to the existing track.
    box2 = (2.0, 2.0, 12.0, 12.0)
    t.update([box2], scores=[0.1])
    assert len(t.tracks) == 1


def test_bbox_tracker_mismatched_scores_raises():
    t = BBoxTracker()
    with pytest.raises(ValueError):
        t.update([(0, 0, 1, 1)], scores=[0.9, 0.8])


def test_bbox_tracker_skips_already_matched_track_high_conf():
    """Two high-confidence detections overlapping the same existing track:
    once the first claims it, the second must skip the already-matched
    track (line 160) and spawn its own."""
    t = BBoxTracker(min_hits=1, max_age=5, iou_threshold_high=0.3)
    box = (0.0, 0.0, 10.0, 10.0)
    t.update([box], scores=[0.9])              # creates track 0
    # Two detections both overlapping track 0 in the same frame.
    a = (0.0, 0.0, 10.0, 10.0)                 # perfect overlap
    b = (1.0, 1.0, 11.0, 11.0)                 # also overlaps track 0
    t.update([a, b], scores=[0.9, 0.9])
    # Track 0 matched once; the second det spawned a new track.
    assert len(t.tracks) == 2


def test_bbox_tracker_skips_already_matched_track_low_conf():
    """Same, but both detections are low-confidence so the stage-2 loop
    runs and the second hits the already-matched guard (line 176)."""
    t = BBoxTracker(min_hits=1, max_age=5,
                    iou_threshold_high=0.3, iou_threshold_low=0.05)
    box = (0.0, 0.0, 10.0, 10.0)
    t.update([box], scores=[0.9])              # creates track 0
    a = (0.0, 0.0, 10.0, 10.0)
    b = (1.0, 1.0, 11.0, 11.0)
    # Both low-confidence -> only stage 2 associates; second det skips
    # the now-matched track. Low-conf dets do not spawn new tracks.
    t.update([a, b], scores=[0.1, 0.1])
    assert len(t.tracks) == 1
