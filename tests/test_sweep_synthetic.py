"""Data-free sweep tests: run() + main() against the synthetic triplet
(complements the needs_data smoke tests in test_sweep.py)."""

from __future__ import annotations

import csv
import sys

import yaml

import scripts.eval.sweep as sweep


def _grid_yaml(tmp_path, prefix):
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump({
        # OpenWater exercises the fp_proxy branch; Boats the recall weighting
        "eval_clips": {"Boats": str(prefix), "OpenWater": str(prefix)},
        "scene_weights": {"Boats": 0.6, "OpenWater": 0.4},
        "grid": {"fisheye.gradient_threshold": [30, 60]},
    }))
    return grid


def test_run_synthetic_full_outputs(tmp_path, monkeypatch, synthetic_triplet):
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    grid = _grid_yaml(tmp_path, prefix)
    monkeypatch.setattr(sweep, "SWEEP_DIR", tmp_path / "sweeps")
    out_dir = sweep.run(str(grid), run_id="synth")
    assert out_dir == tmp_path / "sweeps" / "synth"
    for name in ("config.yaml", "per_frame.csv", "summary.csv", "ranked.csv"):
        assert (out_dir / name).exists()

    with open(out_dir / "per_frame.csv") as f:
        pf = list(csv.DictReader(f))
    # 2 combos x 2 clips x 5 frames
    assert len(pf) == 20
    assert {r["combo_id"] for r in pf} == {"gradient_threshold=30",
                                           "gradient_threshold=60"}

    with open(out_dir / "summary.csv") as f:
        sm = list(csv.DictReader(f))
    assert len(sm) == 4
    ow = [r for r in sm if r["scene"] == "OpenWater"]
    boats = [r for r in sm if r["scene"] == "Boats"]
    # fp_proxy only counts on OpenWater rows
    assert all(r["fp_proxy"] == r["hit_rate"] for r in ow)
    assert all(r["fp_proxy"] == "0.0000" for r in boats)

    with open(out_dir / "ranked.csv") as f:
        ranked = list(csv.DictReader(f))
    assert len(ranked) == 2
    scores = [float(r["proxy_score"]) for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_main_smoke(tmp_path, monkeypatch, synthetic_triplet):
    prefix = synthetic_triplet.fisheye.parent / synthetic_triplet.timestamp
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump({
        "eval_clips": {"Boats": str(prefix)},
        "scene_weights": {"Boats": 1.0},
        "grid": {"fisheye.gradient_threshold": [30]},
    }))
    monkeypatch.setattr(sweep, "SWEEP_DIR", tmp_path / "sweeps")
    monkeypatch.setattr(sys, "argv", ["sweep", "--grid", str(grid),
                                      "--run-id", "cli"])
    sweep.main()
    assert (tmp_path / "sweeps" / "cli" / "ranked.csv").exists()
