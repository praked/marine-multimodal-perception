"""Tests for scripts/eval/seed_labels_from_detector.py."""

import json
import sys
from unittest import mock

import pytest

import scripts.eval.seed_labels_from_detector as seed


def test_frame_meta_ts_scheme():
    scene, tts, fts = seed._frame_meta("Boats/2025-01-01_00-00-00/ts=00-00-01.5")
    assert scene == "Boats"
    assert tts == "2025-01-01_00-00-00"
    assert fts == "00:00:01.5"


def test_frame_meta_legacy_scheme():
    scene, tts, fts = seed._frame_meta("Boats/2025-01-01_00-00-00/000005")
    assert scene == "Boats"
    assert tts == "2025-01-01_00-00-00"
    assert fts is None


def test_derive_bins_centre(detection_real, intrinsics_real):
    fusion = detection_real["fusion"]
    fish = intrinsics_real["fisheye"]
    # A box centred on cx maps to bin 0.
    cx = fish["cx"]
    pdr = fish["pix_deg_ratio"]
    centre_norm = cx / 864.0
    bins = seed._derive_bins(
        [{"cls": "boat", "xyxy": [centre_norm - 0.01, 0.4, centre_norm + 0.01, 0.5]}],
        864, 648, cx, pdr, fusion)
    assert bins == [0]


def test_derive_bins_empty():
    assert seed._derive_bins([], 864, 648, 472, 7.2,
                             {"bin_min_deg": -55, "bin_max_deg": 55,
                              "bin_step_deg": 10}) == []


def _mapping(tmp_path, fake="eval/2026-06-17_18-00-00/ts=18-00-00.0",
             orig="Boats/x/ts=00-00-00.0"):
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({
        "fake_to_original": {fake: orig},
        "original_to_fake": {orig: fake},
    }))
    return mp, fake, orig


def test_main_seeds_and_replaces(tmp_path, capsys):
    mp, fake, orig = _mapping(tmp_path)
    pred = tmp_path / "pred.jsonl"
    pred.write_text(json.dumps({
        "frame_id": orig, "width": 864, "height": 648,
        "fisheye_bboxes": [{"cls": "boat", "xyxy": [0.5, 0.5, 0.55, 0.55]}],
    }) + "\n")
    manual = tmp_path / "manual.jsonl"
    # Pre-existing records: one in the eval scene (to be replaced) and one
    # unrelated (to be preserved).
    manual.write_text(
        json.dumps({"frame_id": "eval/old/ts=0", "scene": "eval"}) + "\n" +
        "\n" +  # blank line -> exercises the skip-blank branch
        json.dumps({"frame_id": "Ducks/x/ts=0", "scene": "Ducks"}) + "\n"
    )
    argv = ["seed", "--pred", str(pred), "--mapping", str(mp),
            "--manual", str(manual)]
    with mock.patch.object(sys, "argv", argv):
        seed.main()
    recs = [json.loads(l) for l in manual.read_text().splitlines() if l.strip()]
    scenes = [r["scene"] for r in recs]
    # Old eval record dropped, Ducks kept, new eval seeded.
    assert "Ducks" in scenes
    seeded = [r for r in recs if r["scene"] == "eval"]
    assert len(seeded) == 1
    assert seeded[0]["source"] == "dashboard-manual"
    assert seeded[0]["frame_id"] == fake
    assert "seeded 1 eval-clip frames" in capsys.readouterr().out


def test_main_skips_unmapped_pred(tmp_path):
    mp, fake, orig = _mapping(tmp_path)
    pred = tmp_path / "pred.jsonl"
    pred.write_text(
        json.dumps({"frame_id": "not-in-mapping",  # unmapped -> skipped
                    "fisheye_bboxes": []}) + "\n\n" +  # blank line too
        json.dumps({"frame_id": orig,  # mapped -> seeded
                    "fisheye_bboxes": []}) + "\n"
    )
    manual = tmp_path / "manual.jsonl"
    argv = ["seed", "--pred", str(pred), "--mapping", str(mp),
            "--manual", str(manual)]
    with mock.patch.object(sys, "argv", argv):
        seed.main()
    # Only the mapped frame is seeded; the unmapped row is dropped.
    recs = [json.loads(l) for l in manual.read_text().splitlines() if l.strip()]
    assert len(recs) == 1
    assert recs[0]["frame_id"] == fake
