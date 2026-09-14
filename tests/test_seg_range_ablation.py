"""Tests for scripts/eval/seg_range_ablation.py: bbox-bottom vs contact range.

Pipeline is stubbed (deterministic detections either side of the A/B); the
CSV/summary plumbing under test runs for real against synthetic masks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

import scripts.eval.seg_range_ablation as sra
from scripts.utils.segmentation import SKY, WATER

TS = "2099-01-01_00-00-00"


class _StubPipeline:
    def __init__(self, intr, det, seg_provider=None):
        self._seg = seg_provider is not None

    def process_fisheye(self, frame, frame_id=None):
        # seg run reports the contact-corrected (farther) range for det 0;
        # det 1 carries no range either way (None-handling branch).
        ranges = [5.0, None] if self._seg else [3.0, None]
        return SimpleNamespace(undistorted=frame,
                               coords=[(80.0, 60.0), (20.0, 20.0)],
                               sizes=[20.0, 10.0], ranges=ranges)


def _write_masks(seg_root, n=2):
    clip_dir = seg_root / f"Synth__{TS}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    m = np.full((120, 160), WATER, np.uint8)
    m[:40, :] = SKY
    for i in range(n):
        Image.fromarray(m).save(clip_dir / f"ts=00-00-0{i}.0.png")


def test_run_clip_measures_deltas(tmp_path, monkeypatch, synthetic_triplet):
    monkeypatch.setattr(sra, "ObstacleDetectionPipeline", _StubPipeline)
    seg_root = tmp_path / "seg"
    _write_masks(seg_root, n=2)
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    rows: list[dict] = []
    summary = sra.run_clip(str(prefix), str(seg_root), rows)
    assert summary["frames"] == 5
    assert summary["masked_frames"] == 2
    assert summary["detections"] == 4            # 2 per masked frame
    assert summary["range_changed"] == 2         # only det 0 changes
    assert summary["mean_delta_m"] == 2.0
    assert summary["frac_farther"] == 1.0
    assert len(rows) == 4
    assert rows[0]["delta_m"] == 2.0 and rows[1]["delta_m"] is None


def test_main_writes_csv_and_summary(tmp_path, monkeypatch, synthetic_triplet, capsys):
    monkeypatch.setattr(sra, "ObstacleDetectionPipeline", _StubPipeline)
    seg_root = tmp_path / "seg"
    _write_masks(seg_root, n=1)
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    out_csv = tmp_path / "res" / "ablation.csv"
    rc = sra.main(["--triplet", str(prefix), "--seg-root", str(seg_root),
                   "--out", str(out_csv)])
    assert rc == 0
    assert out_csv.exists()
    header = out_csv.read_text().splitlines()[0]
    assert "base_range_m" in header and "seg_range_m" in header
    printed = capsys.readouterr().out
    assert "Summary (segmentation range vs bbox-bottom)" in printed


def test_main_missing_seg_root_returns_1(tmp_path, capsys):
    rc = sra.main(["--seg-root", str(tmp_path / "absent")])
    assert rc == 1
    assert "No masks" in capsys.readouterr().out


def test_main_default_clips_skip_cleanly_outside_repo(tmp_path, monkeypatch, capsys):
    seg_root = tmp_path / "seg"
    seg_root.mkdir()
    monkeypatch.chdir(tmp_path)                  # default clips can't resolve
    rc = sra.main(["--seg-root", str(seg_root), "--out", str(tmp_path / "a.csv")])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "skipped" in printed and "(0 rows)" in printed
