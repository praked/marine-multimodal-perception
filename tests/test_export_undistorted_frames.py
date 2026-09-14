"""Tests for scripts/eval/export_undistorted_frames.py against a synthetic triplet."""

from __future__ import annotations

import pytest

from scripts.eval.export_undistorted_frames import (
    clip_dirname,
    export_clip,
    main,
    safe_ts,
)


def test_clip_dirname_and_safe_ts():
    assert clip_dirname("Boats", "2025-06-23_16-21-07") == "Boats__2025-06-23_16-21-07"
    assert safe_ts("16:21:07.3") == "16-21-07.3"


def test_export_clip_writes_every_nth_frame(tmp_path, synthetic_triplet):
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    out = tmp_path / "frames"
    n = export_clip(str(prefix), out, every=2)
    assert n == 3                                # frames 0, 2, 4 of 5
    clip_dir = out / clip_dirname("Synth", synthetic_triplet.timestamp)
    jpgs = sorted(clip_dir.glob("ts=*.jpg"))
    assert len(jpgs) == 3
    assert jpgs[0].name == "ts=00-00-00.0.jpg"


def test_main_counts_and_skips_bad_clips(tmp_path, synthetic_triplet, capsys):
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    out = tmp_path / "frames"
    rc = main(["--triplet", str(prefix),
               "--triplet", str(tmp_path / "Nope" / "2099-01-01_00-00-00"),
               "--out", str(out), "--every", "1"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "Total: 5 frames across 1/2 clip(s)." in printed
    assert "SKIP" in printed and "Failed clips (1)" in printed


def test_main_requires_a_triplet():
    with pytest.raises(SystemExit):
        main(["--out", "somewhere"])
