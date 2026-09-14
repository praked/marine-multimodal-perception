"""Tests for scripts/eval/render_label_overlays.py."""

import json
import sys
from unittest import mock

import numpy as np

import scripts.eval.render_label_overlays as rlo


def test_draw_renders_box_and_banner():
    und = np.full((120, 160, 3), 40, dtype=np.uint8)
    rec = {"fisheye_bboxes": [{"cls": "boat", "xyxy": [0.2, 0.2, 0.6, 0.6]}]}
    out = rlo._draw(und, rec, "banner")
    # Original untouched, output differs where the box/banner were drawn.
    assert out.shape == und.shape
    assert not np.array_equal(out, und)


def test_draw_unknown_class_uses_white():
    und = np.full((120, 160, 3), 40, dtype=np.uint8)
    rec = {"fisheye_bboxes": [{"cls": "kraken", "xyxy": [0.1, 0.1, 0.3, 0.3]}]}
    out = rlo._draw(und, rec, "x")
    assert out.shape == und.shape


def test_montage_paginates(tmp_path):
    thumbs = [np.full((20, 30, 3), i, dtype=np.uint8) for i in range(5)]
    paths = rlo._montage(thumbs, tmp_path, "contact", cols=2, per_page=4)
    # 5 thumbs, per_page 4 -> 2 pages.
    assert len(paths) == 2
    for p in paths:
        assert p.exists()


def test_montage_empty(tmp_path):
    assert rlo._montage([], tmp_path, "contact", cols=2, per_page=4) == []


def test_main(tmp_path, monkeypatch, capsys):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"triplet_ts": "2025-01-01_00-00-00", "frame_ts": "00:00:00.0",
                    "fisheye_bboxes": [{"cls": "boat", "xyxy": [0.4, 0.4, 0.5, 0.5]}]}) + "\n" +
        json.dumps({"triplet_ts": "2025-01-01_00-00-00", "frame_ts": "00:00:01.0",
                    "fisheye_bboxes": []}) + "\n"
    )

    def fake_iter(captures_dir, clip_ts, every, det, intr):
        # The "99" frame is not in the labels -> exercises the skip branch.
        for fts in ("00:00:00.0", "00:00:99.9", "00:00:01.0"):
            und = np.full((120, 160, 3), 50, dtype=np.uint8)
            yield (f"Scene/{clip_ts}/ts={fts}", "Scene", clip_ts, fts, und)

    monkeypatch.setattr(rlo, "iter_clip_frames", fake_iter)
    out_dir = tmp_path / "out"
    argv = ["rlo", "--labels", str(labels), "--captures-dir", str(tmp_path),
            "--out-dir", str(out_dir), "--exposure", "1.5"]
    with mock.patch.object(sys, "argv", argv):
        rlo.main()
    out = capsys.readouterr().out
    assert "contact sheet" in out
    assert list(out_dir.glob("contact_*.png"))


def test_main_detections_only(tmp_path, monkeypatch, capsys):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"triplet_ts": "c", "frame_ts": "00:00:00.0",
                    "fisheye_bboxes": [{"cls": "boat", "xyxy": [0.4, 0.4, 0.5, 0.5]}]}) + "\n" +
        json.dumps({"triplet_ts": "c", "frame_ts": "00:00:01.0",
                    "fisheye_bboxes": []}) + "\n"
    )

    def fake_iter(captures_dir, clip_ts, every, det, intr):
        und = np.full((120, 160, 3), 50, dtype=np.uint8)
        yield (f"Scene/{clip_ts}/ts=00:00:00.0", "Scene", clip_ts, "00:00:00.0", und)

    monkeypatch.setattr(rlo, "iter_clip_frames", fake_iter)
    out_dir = tmp_path / "out"
    argv = ["rlo", "--labels", str(labels), "--captures-dir", str(tmp_path),
            "--out-dir", str(out_dir), "--detections-only"]
    with mock.patch.object(sys, "argv", argv):
        rlo.main()
    assert "rendered 1 frames" in capsys.readouterr().out
