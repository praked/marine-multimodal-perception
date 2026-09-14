"""Tests for scripts/eval/local_seg_masks.py: laptop ONNX mask generation.

onnxruntime is not installed here (nor in CI), so a fake module is injected
into sys.modules with an InferenceSession whose .run returns deterministic
3-class logits. The script imports onnxruntime inside main(), so injection
before the call is sufficient."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import scripts.eval.local_seg_masks as lsm

# Fake model input size (the script reads it from the session, so any
# factor-friendly size works and keeps the resizes cheap).
_MW, _MH = 64, 48


class _FakeInput:
    name = "input"
    shape = [1, 3, _MH, _MW]


class _FakeSession:
    """Mimics onnxruntime.InferenceSession for the 3-class LRASPP student:
    returns (1, 3, H, W) logits: water everywhere, sky in the top third."""

    def __init__(self, path, providers=None):
        self.path = path
        self.providers = providers

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, output_names, feeds):
        x = feeds[_FakeInput.name]
        assert x.shape == (1, 3, _MH, _MW)
        assert x.dtype == np.float32
        b, _c, h, w = x.shape
        logits = np.zeros((b, 3, h, w), dtype=np.float32)
        logits[:, 1] = 1.0                 # water
        logits[:, 2, : h // 3] = 2.0       # sky wins in the top third
        return [logits]


@pytest.fixture
def fake_ort(monkeypatch):
    mod = types.ModuleType("onnxruntime")
    mod.InferenceSession = _FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return mod


def _prefix(triplet) -> str:
    return str(triplet.fisheye.parent / triplet.timestamp)


def _make_triplet(tmp_path: Path, n: int):
    """Synthetic n-frame triplet (local copy of the conftest builder so the
    frame count is controllable)."""
    import cv2

    from scripts.utils.datasets import Triplet

    scene_dir = tmp_path / "Synth"
    scene_dir.mkdir()
    ts = "2099-01-01_12-00-00"
    fish = scene_dir / f"fisheye_{ts}.mp4"
    therm = scene_dir / f"thermal_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(fish), fourcc, 3.0, (160, 120))
    tw = cv2.VideoWriter(str(therm), fourcc, 3.0, (160, 120))
    for _ in range(n):
        f = np.full((120, 160, 3), 90, dtype=np.uint8)
        fw.write(f)
        tw.write(f)
    fw.release()
    tw.release()
    rows = ["Date,Time,X,Y,Z"]
    for i in range(n):
        rows.append(f"2099-01-01,12:00:{i:02d}.0,0.1,1.5,0.0")
    mm = scene_dir / f"mmwave_{ts}.csv"
    mm.write_text("\n".join(rows) + "\n")
    return Triplet(scene="Synth", timestamp=ts, fisheye=fish, thermal=therm,
                   mmwave=mm)


def test_missing_onnxruntime_returns_1(monkeypatch, capsys):
    # None in sys.modules makes `import onnxruntime` raise ImportError even
    # when a real installation exists.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    rc = lsm.main(["--triplet", "does/not/matter"])
    assert rc == 1
    assert "needs onnxruntime" in capsys.readouterr().err


def test_writes_masks_in_segprovider_layout(fake_ort, tmp_path,
                                            synthetic_triplet, capsys):
    seg_root = tmp_path / "seg"
    rc = lsm.main(["--triplet", _prefix(synthetic_triplet),
                   "--seg-root", str(seg_root), "--min-luma", "-1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"input {_MW}x{_MH}" in out
    assert "5 frames -> 5 masks written (0 skipped dark)" in out

    out_dir = seg_root / f"Synth__{synthetic_triplet.timestamp}"
    pngs = sorted(out_dir.glob("ts=*.png"))
    assert len(pngs) == 5
    # colon->dash timestamp naming (radar-keyed)
    assert pngs[0].name == "ts=00-00-00.0.png"
    mask = np.asarray(Image.open(pngs[0]))
    # label-encoded at the undistorted frame's size, classes in {0,1,2}
    assert mask.shape == (120, 160)
    assert set(np.unique(mask)) <= {0, 1, 2}
    # fake logits: sky (2) top third, water (1) below
    assert mask[0, 0] == 2
    assert mask[-1, 0] == 1


def test_existing_masks_skipped_unless_overwrite(fake_ort, tmp_path,
                                                 synthetic_triplet, capsys):
    seg_root = tmp_path / "seg"
    args = ["--triplet", _prefix(synthetic_triplet),
            "--seg-root", str(seg_root), "--min-luma", "-1"]
    assert lsm.main(args) == 0
    capsys.readouterr()

    # second run: everything exists -> nothing written
    assert lsm.main(args) == 0
    assert "5 frames -> 0 masks written" in capsys.readouterr().out

    # --overwrite regenerates
    assert lsm.main(args + ["--overwrite"]) == 0
    assert "5 frames -> 5 masks written" in capsys.readouterr().out


def test_dark_frames_skipped_by_luma_gate(fake_ort, tmp_path,
                                          synthetic_triplet, capsys):
    seg_root = tmp_path / "seg_dark"
    rc = lsm.main(["--triplet", _prefix(synthetic_triplet),
                   "--seg-root", str(seg_root), "--min-luma", "1e9"])
    assert rc == 0
    assert "0 masks written (5 skipped dark)" in capsys.readouterr().out
    assert not list((seg_root / f"Synth__{synthetic_triplet.timestamp}")
                    .glob("*.png"))


def test_progress_line_every_50_masks(fake_ort, tmp_path, capsys):
    triplet = _make_triplet(tmp_path, 50)
    seg_root = tmp_path / "seg50"
    rc = lsm.main(["--triplet", _prefix(triplet),
                   "--seg-root", str(seg_root), "--min-luma", "-1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "50 masks (" in out
    assert "50 frames -> 50 masks written" in out
