"""Tests for scripts/eval/seg_overlay.py (Agg rendering, synthetic masks).

The blob detector is nondeterministic on synthetic footage, so the pipeline is
swapped for a stub with fixed detections; the overlay/mask/contact drawing code
under test runs for real."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

import scripts.eval.seg_overlay as so
from scripts.utils.segmentation import OBSTACLE, SKY, WATER

TS = "2099-01-01_00-00-00"


class _StubPipeline:
    def __init__(self, intr, det, seg_provider=None):
        self._seg = seg_provider is not None

    def process_fisheye(self, frame, frame_id=None):
        ranges = [5.0, None] if self._seg else [3.0, None]
        return SimpleNamespace(undistorted=frame,
                               coords=[(80.0, 60.0), (20.0, 20.0)],
                               sizes=[20.0, 10.0], ranges=ranges)


def _write_masks(seg_root, n=2):
    """Sky top, water below, an obstacle block sitting on water at (80, 60)."""
    clip_dir = seg_root / f"Synth__{TS}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    m = np.full((120, 160), WATER, np.uint8)
    m[:40, :] = SKY
    m[50:70, 70:90] = OBSTACLE
    for i in range(n):
        Image.fromarray(m).save(clip_dir / f"ts=00-00-0{i}.0.png")


def test_render_clip_writes_overlays(tmp_path, monkeypatch, synthetic_triplet, capsys):
    monkeypatch.setattr(so, "ObstacleDetectionPipeline", _StubPipeline)
    seg_root = tmp_path / "seg"
    _write_masks(seg_root, n=2)          # masks for 2 of 5 frames
    out = tmp_path / "out"
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    n = so.render_clip(str(prefix), str(seg_root), out, limit=0)
    assert n == 2                        # frames without a mask are skipped
    pngs = list((out / f"Synth__{TS}").glob("ts=*.png"))
    assert len(pngs) == 2 and all(p.stat().st_size > 0 for p in pngs)


def test_render_clip_respects_limit(tmp_path, monkeypatch, synthetic_triplet):
    monkeypatch.setattr(so, "ObstacleDetectionPipeline", _StubPipeline)
    seg_root = tmp_path / "seg"
    _write_masks(seg_root, n=2)
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    assert so.render_clip(str(prefix), str(seg_root), tmp_path / "o", limit=1) == 1


def test_main_missing_seg_root_returns_1(tmp_path, capsys):
    rc = so.main(["--seg-root", str(tmp_path / "absent")])
    assert rc == 1
    assert "No masks" in capsys.readouterr().out


def test_main_runs_clips_and_skips_failures(tmp_path, monkeypatch,
                                            synthetic_triplet, capsys):
    monkeypatch.setattr(so, "ObstacleDetectionPipeline", _StubPipeline)
    seg_root = tmp_path / "seg"
    _write_masks(seg_root, n=1)
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    rc = so.main(["--triplet", str(prefix),
                  "--triplet", str(tmp_path / "Missing" / TS),
                  "--seg-root", str(seg_root), "--out", str(tmp_path / "o"),
                  "--limit", "3"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "Total 1 overlays" in printed
    assert "skipped" in printed


def test_main_default_clips_all_skipped_outside_repo(tmp_path, monkeypatch, capsys):
    """With no --triplet the four frozen val clips are tried; from a scratch
    cwd they all fail to resolve and are reported as skipped."""
    seg_root = tmp_path / "seg"
    seg_root.mkdir()
    monkeypatch.chdir(tmp_path)
    rc = so.main(["--seg-root", str(seg_root), "--out", str(tmp_path / "o")])
    assert rc == 0
    assert "Total 0 overlays" in capsys.readouterr().out
