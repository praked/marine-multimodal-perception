"""Tests for scripts/eval/thermal_sweep.py: thermal-only cached-frame sweep.

The grid/dotted-key helpers and the worker pair (_init_worker/_run_combo)
are exercised in-process against a synthetic triplet + synthetic labels;
main() runs end-to-end with --workers 1 (and once through the Pool path)
with OUT_DIR redirected to tmp_path. No data/ footage needed."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

import scripts.eval.thermal_sweep as tsw

MAIN_TS = "2099-01-01_00-00-00"
CLIP_ID = f"Synth/{MAIN_TS}"


# ---------------------------------------------------------------------------
# Local fixtures (synthetic triplets + labels)
# ---------------------------------------------------------------------------

def _make_triplet(base: Path, ts: str, n: int) -> str:
    """Synthetic triplet under base/Synth (160x120 videos + mmwave CSV).

    Returns the resolve_triplet prefix string."""
    import cv2

    scene_dir = base / "Synth"
    scene_dir.mkdir(exist_ok=True, parents=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(scene_dir / f"fisheye_{ts}.mp4"),
                         fourcc, 3.0, (160, 120))
    tw = cv2.VideoWriter(str(scene_dir / f"thermal_{ts}.mp4"),
                         fourcc, 3.0, (160, 120))
    for i in range(n):
        f = np.full((120, 160, 3), 60 + (i % 3) * 5, dtype=np.uint8)
        f[40:80, 60:100] = 200
        fw.write(f)
        tw.write(f)
    fw.release()
    tw.release()
    rows = ["Date,Time,X,Y,Z"]
    for i in range(n):
        rows.append(f"{ts[:10]},00:00:{i:02d}.0,0.1,1.5,0.0")
        rows.append(f"{ts[:10]},00:00:{i:02d}.0,0.5,2.0,0.0")
    (scene_dir / f"mmwave_{ts}.csv").write_text("\n".join(rows) + "\n")
    return str(scene_dir / ts)


def _write_labels(path: Path, n_frames: int) -> None:
    """Centred boat box per frame (~[-4, +4.3] deg under the real fisheye K,
    inside the thermal linear FOV)."""
    recs = [{"frame_id": f"{CLIP_ID}/{i:06d}",
             "frame_ts": f"00:00:{i:02d}.0",
             "width": 864, "height": 648,
             "fisheye_bboxes": [
                 {"cls": "boat", "xyxy": [0.48, 0.55, 0.55, 0.80]}]}
            for i in range(n_frames)]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


@pytest.fixture
def sweep_data(tmp_path):
    """(main prefix, labels path, fp_clips dict): fp 'short' has too few
    frames to survive the 10-frame warm-up; fp 'long' has 3 scored frames."""
    main_prefix = _make_triplet(tmp_path, MAIN_TS, 6)
    fp_short = _make_triplet(tmp_path, "2099-01-02_00-00-00", 5)
    fp_long = _make_triplet(tmp_path, "2099-01-03_00-00-00", 13)
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, 6)
    return main_prefix, labels, {"short": fp_short, "long": fp_long}


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def test_enumerate_grid():
    combos = tsw._enumerate_grid({"a": [1, 2], "b": [3]})
    assert combos == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_combo_id_uses_last_key_component():
    assert tsw._combo_id({"thermal.object_thresh": 100,
                          "thermal.blob.minArea": 4}) == \
        "object_thresh=100_minArea=4"


def test_set_dotted_create_makes_intermediate_dicts():
    cfg: dict = {"thermal": {"object_thresh": 100}}
    tsw._set_dotted_create(cfg, "thermal.horizon.fallback_row", 5)
    assert cfg["thermal"]["horizon"] == {"fallback_row": 5}
    tsw._set_dotted_create(cfg, "thermal.object_thresh", 60)
    assert cfg["thermal"]["object_thresh"] == 60
    tsw._set_dotted_create(cfg, "a.b.c", 1)
    assert cfg["a"] == {"b": {"c": 1}}


# ---------------------------------------------------------------------------
# build_cache + worker pair (in-process)
# ---------------------------------------------------------------------------

def test_build_cache_and_run_combo(tmp_path, sweep_data, intrinsics_real,
                                   detection_real):
    main_prefix, labels, fp_clips = sweep_data
    cache_path = tmp_path / "cache" / "frame_cache.pkl"
    cache = tsw.build_cache(main_prefix, detection_real, intrinsics_real,
                            cache_path, fp_clips=fp_clips)
    assert cache_path.exists()
    assert len(cache["frames"]) == 6
    assert set(cache["radar_by_ts"]) == {f"00:00:{i:02d}.0" for i in range(6)}
    assert cache["triplet_prefix"] == main_prefix
    assert len(cache["fp_frames"]["short"]) == 5
    assert len(cache["fp_frames"]["long"]) == 13

    tsw._init_worker(str(cache_path), str(labels), CLIP_ID,
                     6.0, 1, 1)   # skip 1 frame, dilate 1
    assert tsw._G["fov"][0] < 0 < tsw._G["fov"][1]
    assert len(tsw._G["gt"]) == 6
    assert tsw._G["tol_deg"] == 6.0

    row = tsw._run_combo({"thermal.object_thresh": 120})
    assert row["combo_id"] == "object_thresh=120"
    assert json.loads(row["combo_json"]) == {"thermal.object_thresh": 120}
    assert row["n_frames"] == 5                     # 6 frames - 1 skipped
    # fp 'short' never clears the 10-frame warm-up -> NaN rate
    assert math.isnan(row["fpclip_short"])
    # fp 'long' scores frames 10-12 -> a finite non-negative rate
    assert row["fpclip_long"] >= 0.0
    assert "recall" in row and "bin_f1" in row


def test_build_cache_without_fp_clips(tmp_path, sweep_data, intrinsics_real,
                                      detection_real):
    main_prefix, _labels, _fp = sweep_data
    cache_path = tmp_path / "cache2.pkl"
    cache = tsw.build_cache(main_prefix, detection_real, intrinsics_real,
                            cache_path)
    assert cache["fp_frames"] == {}


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def _write_grid(path: Path, main_prefix: str, labels: Path, grid: dict,
                fp_clips: dict | None = None, **extra) -> None:
    cfg = {"triplet": main_prefix, "labels": str(labels),
           "skip_frames": 0, "gt_dilate_frames": 1, "tol_deg": 6.0,
           "grid": grid}
    if fp_clips is not None:
        cfg["fp_clips"] = fp_clips
    cfg.update(extra)
    path.write_text(yaml.safe_dump(cfg))


def test_main_rejects_non_thermal_keys(tmp_path, sweep_data):
    main_prefix, labels, _fp = sweep_data
    grid_path = tmp_path / "bad_grid.yaml"
    _write_grid(grid_path, main_prefix, labels,
                {"fisheye.gradient_threshold": [30]})
    with mock.patch.object(sys, "argv",
                           ["thermal_sweep", "--grid", str(grid_path)]):
        with pytest.raises(SystemExit, match="non-thermal keys"):
            tsw.main()


def test_main_workers1_end_to_end(tmp_path, monkeypatch, sweep_data, capsys):
    main_prefix, labels, fp_clips = sweep_data
    grid_path = tmp_path / "grid.yaml"
    # 12 combos -> exercises the every-10 progress print.
    _write_grid(grid_path, main_prefix, labels,
                {"thermal.object_thresh": [60, 100, 130, 160],
                 "thermal.blob.minArea": [2, 4, 8]},
                fp_clips=fp_clips)
    monkeypatch.setattr(tsw, "OUT_DIR", tmp_path / "out")
    with mock.patch.object(sys, "argv",
                           ["thermal_sweep", "--grid", str(grid_path),
                            "--run-id", "unit", "--workers", "1"]):
        tsw.main()
    out_dir = tmp_path / "out" / "sweep_unit"
    assert (out_dir / "grid.yaml").exists()
    assert not (out_dir / "frame_cache.pkl").exists()   # deleted at the end
    with open(out_dir / "ranked.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert {"combo_id", "f1", "fpclip_short", "fpclip_long",
            "combo_json", "per_class"} <= set(rows[0])
    assert all(json.loads(r["combo_json"]) for r in rows)
    printed = capsys.readouterr().out
    assert "12 combos" in printed
    assert "10/12" in printed
    assert "ranked.csv" in printed


def test_main_pool_path(tmp_path, monkeypatch, sweep_data, capsys):
    """--workers 2 goes through the spawn Pool (worker coverage itself comes
    from the in-process tests; this covers the parent's pool branch)."""
    main_prefix, labels, fp_clips = sweep_data
    grid_path = tmp_path / "grid_pool.yaml"
    # 20 combos -> exercises the pool branch's every-20 progress print.
    _write_grid(grid_path, main_prefix, labels,
                {"thermal.object_thresh": [60, 100, 130, 160],
                 "thermal.blob.minArea": [2, 4, 8, 16, 32]}, fp_clips=None)
    monkeypatch.setattr(tsw, "OUT_DIR", tmp_path / "out")
    with mock.patch.object(sys, "argv",
                           ["thermal_sweep", "--grid", str(grid_path),
                            "--run-id", "pool", "--workers", "2",
                            "--tol", "6.0"]):
        tsw.main()
    with open(tmp_path / "out" / "sweep_pool" / "ranked.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    assert "20/20" in capsys.readouterr().out
