"""Temporal tracking: Branch B.3.

Per-frame detections are inherently noisy; a transient firing in a
single frame is more often than not a false positive. Tracking gates
bin-level scores by confirmation across N frames, dramatically reducing
the false-positive rate (especially on the Rain scene where rainfall
generates one-frame radar spikes).

Two tracker shapes here:

  - `SectorTracker`: operates directly on the 11 fusion bins. Confirms
    a bin after `min_hits` consecutive frames with score >= threshold;
    drops it after `max_age` consecutive misses. Cheap, lossless integration
    with fusion.py.

  - `BBoxTracker`: ByteTrack-flavour tracker over camera bboxes,
    associated by IoU. Used by the eval tools for per-detection velocity
    + classification later (Branch B.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Iterable


# ---------------------------------------------------------------------------
# Sector-level tracker (primary path)
# ---------------------------------------------------------------------------

class SectorTracker:
    """Per-bin confirmation gate.

    Each bin has independent hit / miss counters. Returns per-bin
    `confirmed` booleans on every update.
    """

    def __init__(self, n_bins: int, min_hits: int = 3, max_age: int = 5):
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if max_age < 1:
            raise ValueError("max_age must be >= 1")
        self.n_bins = n_bins
        self.min_hits = min_hits
        self.max_age = max_age
        self.hits = [0] * n_bins
        self.misses = [0] * n_bins
        self.confirmed = [False] * n_bins

    def update(self, bin_hits: Iterable[bool]) -> list[bool]:
        """Feed per-bin hit booleans for the current frame. Returns the
        per-bin confirmed state after the update.
        """
        bin_hits = list(bin_hits)
        if len(bin_hits) != self.n_bins:
            raise ValueError(f"expected {self.n_bins} bin_hits, got {len(bin_hits)}")
        for i, hit in enumerate(bin_hits):
            if hit:
                self.hits[i] += 1
                self.misses[i] = 0
                if self.hits[i] >= self.min_hits:
                    self.confirmed[i] = True
            else:
                self.misses[i] += 1
                if self.confirmed[i] and self.misses[i] >= self.max_age:
                    self.confirmed[i] = False
                    self.hits[i] = 0
                elif not self.confirmed[i] and self.misses[i] >= self.max_age:
                    # Reset partial tracks too
                    self.hits[i] = 0
        return list(self.confirmed)

    def gated_scores(self, scores: Iterable[float]) -> list[float]:
        """Zero out scores for bins that are not currently confirmed."""
        return [s if c else 0.0 for s, c in zip(scores, self.confirmed)]


# ---------------------------------------------------------------------------
# Bbox-level tracker (ByteTrack-lite)
# ---------------------------------------------------------------------------

@dataclass
class BBoxTrack:
    track_id: int
    bbox: tuple[float, float, float, float]   # xyxy in image coords
    hits: int = 1
    age_since_hit: int = 0
    confirmed: bool = False
    history: list[tuple[float, float, float, float]] = field(default_factory=list)

    def predict(self) -> tuple[float, float, float, float]:
        """Constant-position predictor. Returns last bbox; B.2 can swap
        in linear motion."""
        return self.bbox


def iou(a: tuple[float, float, float, float],
        b: tuple[float, float, float, float]) -> float:
    """IoU between two xyxy bboxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


class BBoxTracker:
    """Per-frame IoU-greedy tracker. Two-stage (ByteTrack-lite):
    high-confidence detections first, then low-confidence to recover
    short occlusions.
    """

    def __init__(self,
                 min_hits: int = 3,
                 max_age: int = 5,
                 iou_threshold_high: float = 0.3,
                 iou_threshold_low: float = 0.1):
        if min_hits < 1 or max_age < 1:
            raise ValueError("min_hits and max_age must be >= 1")
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_high = iou_threshold_high
        self.iou_low = iou_threshold_low
        self.tracks: list[BBoxTrack] = []
        self._next_id = count(0)
        # Per-detection track id for the LAST update() call, aligned with
        # the `detections` argument (None = unmatched and not spawned).
        # Additive telemetry: the cross-sensor identity layer
        # (track_identity.py) needs "which track is detection i" per frame,
        # which the confirmed-tracks return value cannot express.
        self.last_det_track_ids: list[int | None] = []

    def update(self, detections: list[tuple[float, float, float, float]],
               scores: list[float] | None = None,
               high_conf: float = 0.5) -> list[BBoxTrack]:
        """Step the tracker by one frame.

        `detections` is a list of xyxy bboxes; `scores` (optional) is the
        per-detection confidence. Returns the *confirmed* tracks after
        the update.
        """
        scores = scores if scores is not None else [1.0] * len(detections)
        if len(scores) != len(detections):
            raise ValueError("scores and detections must align")

        high_idx = [i for i, s in enumerate(scores) if s >= high_conf]
        low_idx = [i for i, s in enumerate(scores) if s < high_conf]

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()
        det_track_ids: list[int | None] = [None] * len(detections)

        # Stage 1: associate high-confidence detections to existing tracks.
        for di in high_idx:
            best_track = None
            best_iou = self.iou_high
            for tr in self.tracks:
                if tr.track_id in matched_track_ids:
                    continue
                ov = iou(tr.predict(), detections[di])
                if ov > best_iou:
                    best_iou = ov
                    best_track = tr
            if best_track is not None:
                self._update_track(best_track, detections[di])
                matched_track_ids.add(best_track.track_id)
                matched_det_indices.add(di)
                det_track_ids[di] = best_track.track_id

        # Stage 2: low-confidence detections recover from short occlusions.
        for di in low_idx:
            best_track = None
            best_iou = self.iou_low
            for tr in self.tracks:
                if tr.track_id in matched_track_ids:
                    continue
                ov = iou(tr.predict(), detections[di])
                if ov > best_iou:
                    best_iou = ov
                    best_track = tr
            if best_track is not None:
                self._update_track(best_track, detections[di])
                matched_track_ids.add(best_track.track_id)
                matched_det_indices.add(di)
                det_track_ids[di] = best_track.track_id

        # Age unmatched tracks.
        for tr in self.tracks:
            if tr.track_id not in matched_track_ids:
                tr.age_since_hit += 1

        # Spawn new tracks from unmatched high-conf detections.
        for di in high_idx:
            if di in matched_det_indices:
                continue
            new = BBoxTrack(track_id=next(self._next_id), bbox=detections[di])
            new.history.append(detections[di])
            self.tracks.append(new)
            det_track_ids[di] = new.track_id

        # Drop aged tracks.
        self.tracks = [tr for tr in self.tracks if tr.age_since_hit < self.max_age]

        # Update confirmation flag.
        for tr in self.tracks:
            if tr.hits >= self.min_hits:
                tr.confirmed = True

        self.last_det_track_ids = det_track_ids
        return [tr for tr in self.tracks if tr.confirmed]

    def _update_track(self, tr: BBoxTrack, bbox: tuple[float, float, float, float]):
        tr.bbox = bbox
        tr.hits += 1
        tr.age_since_hit = 0
        tr.history.append(bbox)
