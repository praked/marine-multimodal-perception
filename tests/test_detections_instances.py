"""Tests for scripts/utils/detections.InstanceSegProvider + DetProvider edge
branches (malformed lines, missing files, cache reuse)."""

from __future__ import annotations

import json

from scripts.utils.detections import (
    DetProvider,
    InstanceSegProvider,
    det_file_for_frame_id,
)

FID = "SceneA/2026-01-01_00-00-00/ts=00-00-01.0"
FID2 = "SceneA/2026-01-01_00-00-00/ts=00-00-02.0"


def _write_jsonl(path, records, garbage=True):
    lines = [json.dumps(r) for r in records]
    if garbage:
        lines.insert(0, "")                 # blank line
        lines.insert(1, "{not json")        # malformed line
        lines.append(json.dumps({"no_frame_id": True}))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def test_det_provider_skips_garbage_lines_and_caches(tmp_path):
    path = det_file_for_frame_id(tmp_path, FID)
    _write_jsonl(path, [{"frame_id": FID,
                         "fisheye_bboxes": [{"cls": "buoy", "xyxy": [0, 0, 1, 1]}]}])
    prov = DetProvider(tmp_path)
    assert prov.get(FID)[0]["cls"] == "buoy"
    # second call comes from the cache (delete the file to prove it)
    path.unlink()
    assert prov.get(FID)[0]["cls"] == "buoy"


def test_det_provider_missing_clip_file(tmp_path):
    prov = DetProvider(tmp_path / "nope")
    assert not prov.available()
    assert prov.get(FID) is None
    assert prov.get("bad_frame_id") is None


def test_instance_provider_get(tmp_path):
    path = det_file_for_frame_id(tmp_path, FID)
    _write_jsonl(path, [
        {"frame_id": FID, "instances": [
            {"cls": "boat_ship", "confidence": 0.8,
             "polygon": [0.1, 0.1, 0.5, 0.1, 0.5, 0.5]}]},
        {"frame_id": FID2, "instances": []},
    ])
    prov = InstanceSegProvider(tmp_path)
    assert prov.available()
    got = prov.get(FID)
    assert got and got[0]["cls"] == "boat_ship"
    assert prov.get(FID2) == []                 # present, no instances
    assert prov.get("SceneA/2026-01-01_00-00-00/ts=99-99-99.9") is None
    assert prov.get(None) is None
    assert prov.get("garbage") is None          # unparseable frame_id
    # cached second read
    path.unlink()
    assert prov.get(FID)[0]["confidence"] == 0.8


def test_instance_provider_missing_root(tmp_path):
    prov = InstanceSegProvider(tmp_path / "absent")
    assert not prov.available()
    assert prov.get(FID) is None
