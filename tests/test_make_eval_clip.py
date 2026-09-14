"""Tests for scripts/eval/make_eval_clip.py."""

import json
import sys
from unittest import mock

import pytest

import scripts.eval.make_eval_clip as mec
from scripts.sensor_processing.pipeline import iterate_triplet
from scripts.utils.calibration import load_detection


def test_format_hhmmss_basic():
    assert mec._format_hhmmss(0) == "00:00:00.0"
    assert mec._format_hhmmss(3661.5) == "01:01:01.5"


def test_format_hhmmss_wraps_day():
    # 86400 == 24h wraps back to 0.
    assert mec._format_hhmmss(86400) == "00:00:00.0"


def _frame_ids_for(triplet):
    det = load_detection()
    ids = []
    for ts, _f, _t, _p in iterate_triplet(triplet, det):
        ids.append(f"{triplet.scene}/{triplet.timestamp}/ts={ts.replace(':', '-')}")
    return ids


def test_main_builds_clip_and_mapping(tmp_path, monkeypatch, synthetic_triplet,
                                      capsys):
    ids = _frame_ids_for(synthetic_triplet)
    assert ids, "synthetic triplet produced no frames"

    frames_file = tmp_path / "frames.txt"
    # Include a comment and a bogus id to exercise filtering.
    frames_file.write_text(
        "# a comment line\n" +
        ids[0] + "  # inline comment\n" +
        ids[1] + "\n" +
        "Synth/2099-01-01_00-00-00/ts=99-99-99.9\n"  # not resolvable
    )

    captures = tmp_path / "captures"
    monkeypatch.setattr(mec, "CAPTURES_DIR", captures)
    monkeypatch.chdir(tmp_path)  # mapping written under ./labels

    src_dir = synthetic_triplet.fisheye.parent  # tmp/Synth
    argv = ["mec", "--captures-dir", str(src_dir),
            "--frames-file", str(frames_file),
            "--out-mission", "evaltest", "--ts", "2099-01-01_18-00-00"]
    with mock.patch.object(sys, "argv", argv):
        mec.main()

    out_clip = captures / "evaltest"
    assert (out_clip / "fisheye_2099-01-01_18-00-00.mp4").exists()
    assert (out_clip / "thermal_2099-01-01_18-00-00.mp4").exists()
    assert (out_clip / "mmwave_2099-01-01_18-00-00.csv").exists()

    mapping = json.loads((tmp_path / "labels" / "evaltest_mapping.json").read_text())
    assert len(mapping["fake_to_original"]) == 2
    assert set(mapping["original_to_fake"]) == set(ids[:2])
    assert "1 frame_ids not found" in capsys.readouterr().out


def test_main_no_frames_resolved_exits(tmp_path, monkeypatch, synthetic_triplet):
    frames_file = tmp_path / "frames.txt"
    frames_file.write_text("Synth/2099-01-01_00-00-00/ts=99-99-99.9\n")
    monkeypatch.setattr(mec, "CAPTURES_DIR", tmp_path / "captures")
    monkeypatch.chdir(tmp_path)
    src_dir = synthetic_triplet.fisheye.parent
    argv = ["mec", "--captures-dir", str(src_dir),
            "--frames-file", str(frames_file)]
    with mock.patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit):
            mec.main()
