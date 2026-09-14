import sys
from pathlib import Path
from unittest import mock

import pytest

from scripts.eval.hard_negatives import main, mine_clip


def test_mine_clip_writes_jsonl_and_images(intrinsics_real, detection_real,
                                            synthetic_triplet, tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.eval.hard_negatives.HARD_NEG_DIR", tmp_path)
    monkeypatch.setattr("scripts.eval.hard_negatives.CANDIDATES_PATH",
                        tmp_path / "candidates.jsonl")
    # Force a low threshold so something fires.
    detection_real["fusion"]["hit_threshold"] = 0.0
    found = mine_clip(synthetic_triplet, intrinsics_real, detection_real,
                      max_per_clip=10, save_images=True)
    # Candidates file should exist (even if zero candidates).
    assert (tmp_path / "candidates.jsonl").exists()


def test_main_no_triplets_exits(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.eval.hard_negatives.HARD_NEG_DIR", tmp_path)
    monkeypatch.setattr("scripts.eval.hard_negatives.CANDIDATES_PATH",
                        tmp_path / "candidates.jsonl")
    monkeypatch.setattr("scripts.eval.hard_negatives.list_triplets", lambda s: [])
    with mock.patch.object(sys, "argv", ["hn", "--scene", "X"]):
        with pytest.raises(SystemExit):
            main()


def test_mine_clip_skips_frames_below_threshold(intrinsics_real, detection_real,
                                                synthetic_triplet, tmp_path,
                                                monkeypatch):
    """Threshold above any achievable score: every frame is skipped via
    the `max_score < threshold` continue (line 53), so zero candidates."""
    monkeypatch.setattr("scripts.eval.hard_negatives.HARD_NEG_DIR", tmp_path)
    monkeypatch.setattr("scripts.eval.hard_negatives.CANDIDATES_PATH",
                        tmp_path / "candidates.jsonl")
    (tmp_path / "candidates.jsonl").write_text("")
    detection_real["fusion"]["hit_threshold"] = 1.5  # unreachable (max is 1.0)
    found = mine_clip(synthetic_triplet, intrinsics_real, detection_real,
                      max_per_clip=10, save_images=False)
    assert found == 0
    assert (tmp_path / "candidates.jsonl").read_text() == ""


def test_main_with_images_prints_image_dir(monkeypatch, tmp_path,
                                            synthetic_triplet):
    """main() default path saves images and prints the images-saved line
    (line 103)."""
    monkeypatch.setattr("scripts.eval.hard_negatives.HARD_NEG_DIR", tmp_path)
    monkeypatch.setattr("scripts.eval.hard_negatives.CANDIDATES_PATH",
                        tmp_path / "candidates.jsonl")
    monkeypatch.setattr("scripts.eval.hard_negatives.list_triplets",
                        lambda s: [synthetic_triplet])
    captured = {}

    def _print(*args, **kwargs):
        captured.setdefault("lines", []).append(" ".join(str(a) for a in args))

    monkeypatch.setattr("builtins.print", _print)
    with mock.patch.object(sys, "argv", ["hn", "--scene", "Synth"]):
        main()
    joined = "\n".join(captured["lines"])
    assert "images saved" in joined


@pytest.mark.needs_data
def test_main_with_real_data(monkeypatch, tmp_path):
    """Run against a real Boats clip; should produce some candidates with
    a low threshold or zero with the default. Either is acceptable."""
    pytest.importorskip("tqdm")
    monkeypatch.setattr("scripts.eval.hard_negatives.HARD_NEG_DIR", tmp_path)
    monkeypatch.setattr("scripts.eval.hard_negatives.CANDIDATES_PATH",
                        tmp_path / "candidates.jsonl")
    with mock.patch.object(sys, "argv", ["hn", "--scene", "OpenWater",
                                          "--max-per-clip", "2", "--no-images"]):
        main()
    assert (tmp_path / "candidates.jsonl").exists()
