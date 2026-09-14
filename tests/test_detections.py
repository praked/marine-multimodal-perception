"""Tests for scripts/utils/detections.DetProvider (typed-detection overlay)."""

from __future__ import annotations

import json

from scripts.utils.detections import (
    CLASS_COLOURS,
    DetProvider,
    det_file_for_frame_id,
)

FID = "2026-06-17_institutionone_day1/2026-06-17_12-41-23/ts=12-41-24.0"


def test_det_file_for_frame_id():
    p = det_file_for_frame_id("data/det", FID)
    assert p is not None
    assert p.as_posix() == "data/det/2026-06-17_institutionone_day1__2026-06-17_12-41-23.jsonl"


def test_det_file_bad_frame_id():
    assert det_file_for_frame_id("data/det", "garbage") is None


def test_provider_get(tmp_path):
    path = det_file_for_frame_id(tmp_path, FID)
    rec = {"frame_id": FID,
           "fisheye_bboxes": [{"cls": "boat_ship", "xyxy": [.4, .4, .6, .6],
                               "confidence": 0.9}]}
    other = {"frame_id": "2026-06-17_institutionone_day1/2026-06-17_12-41-23/ts=12-41-25.0",
             "fisheye_bboxes": []}
    path.write_text(json.dumps(rec) + "\n" + json.dumps(other) + "\n")

    prov = DetProvider(tmp_path)
    assert prov.available()
    got = prov.get(FID)
    assert got and got[0]["cls"] == "boat_ship"
    # frame present but no objects -> empty list (not None)
    assert prov.get(other["frame_id"]) == []
    # unknown frame / clip -> None
    assert prov.get("2026-06-17_institutionone_day1/2026-06-17_12-41-23/ts=99-99-99.9") is None
    assert prov.get(None) is None


def test_all_lars_classes_have_colours():
    for name in ("boat_ship", "row_boats", "paddle_board", "buoy", "swimmer",
                 "animal", "float", "other"):
        assert name in CLASS_COLOURS
