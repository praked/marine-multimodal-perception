import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

from scripts.eval.sweep import _combo_id, _enumerate_grid, main, run


def test_enumerate_grid_simple():
    grid = {"a": [1, 2], "b": [3, 4]}
    combos = _enumerate_grid(grid)
    assert len(combos) == 4
    assert {tuple(sorted(c.items())) for c in combos} == {
        (("a", 1), ("b", 3)), (("a", 1), ("b", 4)),
        (("a", 2), ("b", 3)), (("a", 2), ("b", 4)),
    }


def test_enumerate_grid_single_key():
    combos = _enumerate_grid({"a": [1, 2, 3]})
    assert [c["a"] for c in combos] == [1, 2, 3]


def test_combo_id_format():
    assert _combo_id({"fisheye.gradient_threshold": 30}) == "gradient_threshold=30"


def test_combo_id_multi():
    s = _combo_id({"a.b": 1, "c.d": 2})
    assert "b=1" in s and "d=2" in s


def _write_smoke_grid(tmp_path):
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump({
        "eval_clips": {"Boats": "data/Boats/2025-06-23_16-21-07"},
        "scene_weights": {"Boats": 1.0},
        "grid": {"fisheye.gradient_threshold": [30]},
    }))
    return grid


@pytest.mark.needs_data
def test_run_smoke(tmp_path, monkeypatch, repo_root):
    grid = _write_smoke_grid(tmp_path)
    monkeypatch.setattr("scripts.eval.sweep.SWEEP_DIR", tmp_path / "out")
    out_dir = run(str(grid), run_id="smoke")
    assert (out_dir / "per_frame.csv").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "ranked.csv").exists()
    assert (out_dir / "config.yaml").exists()


@pytest.mark.needs_data
def test_run_skips_zero_frame_clip(tmp_path, monkeypatch, repo_root):
    """A clip that yields no frames hits the `n == 0: continue` guard (line 131)
    so it produces no summary row but the sweep still completes."""
    grid = _write_smoke_grid(tmp_path)
    monkeypatch.setattr("scripts.eval.sweep.SWEEP_DIR", tmp_path / "out")
    # Force iterate_triplet (as imported into the sweep module) to yield nothing.
    monkeypatch.setattr("scripts.eval.sweep.iterate_triplet",
                        lambda triplet, det: iter(()))
    out_dir = run(str(grid), run_id="zero")
    # summary.csv has only the header (the lone clip was skipped at n == 0).
    summary_lines = (out_dir / "summary.csv").read_text().splitlines()
    assert len(summary_lines) == 1  # header only
    # ranked.csv still written (combo present with score 0).
    assert (out_dir / "ranked.csv").exists()


@pytest.mark.needs_data
def test_main_invokes_run(tmp_path, monkeypatch):
    grid = _write_smoke_grid(tmp_path)
    monkeypatch.setattr("scripts.eval.sweep.SWEEP_DIR", tmp_path / "out")
    with mock.patch.object(sys, "argv", ["sweep", "--grid", str(grid),
                                          "--run-id", "main_smoke"]):
        main()
    assert (tmp_path / "out" / "main_smoke" / "summary.csv").exists()
