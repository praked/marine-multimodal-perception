"""CLI tests for scripts/eval/range_validate.py main() + the raw-pixel and
unreadable-image branches of evaluate()."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import yaml

from scripts.eval.range_validate import evaluate, main


def _frame(tmp_path, name="f.jpg"):
    img = np.full((648, 864, 3), 80, np.uint8)
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return p


def test_evaluate_skips_unreadable_image(tmp_path, capsys):
    samples = [{"image": str(tmp_path / "missing.jpg"),
                "pixel": [432, 520], "true_range_m": 5.0}]
    rows = evaluate(samples, intrinsics_path=None, use_horizon=False)
    assert rows == []
    assert "cannot read" in capsys.readouterr().out


def test_evaluate_raw_pixel_space_undistorts(tmp_path):
    p = _frame(tmp_path)
    samples = [{"image": str(p), "pixel": [432, 520],
                "pixel_space": "raw", "true_range_m": 5.0}]
    rows = evaluate(samples, intrinsics_path=None, use_horizon=False)
    assert len(rows) == 1
    assert rows[0][2] is None or rows[0][2] > 0


def test_main_prints_report(tmp_path, capsys):
    p = _frame(tmp_path)
    doc = {"samples": [
        {"image": str(p), "pixel": [432, 520],
         "pixel_space": "undistorted", "true_range_m": 5.0},
        {"image": str(p), "pixel": [432, 50],       # above horizon -> None
         "pixel_space": "undistorted", "true_range_m": 5.0},
    ]}
    yml = tmp_path / "samples.yaml"
    yml.write_text(yaml.safe_dump(doc))
    rc = main(["--samples", str(yml)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "attitude: level" in out
    assert "mean |error|" in out
    assert "(above horizon)" in out


def test_main_use_horizon_flag(tmp_path, capsys):
    p = _frame(tmp_path)
    doc = {"samples": [{"image": str(p), "pixel": [432, 520],
                        "pixel_space": "undistorted", "true_range_m": 5.0}]}
    yml = tmp_path / "samples.yaml"
    yml.write_text(yaml.safe_dump(doc))
    assert main(["--samples", str(yml), "--use-horizon"]) == 0
    assert "detected-horizon" in capsys.readouterr().out


def test_main_no_samples_exits(tmp_path):
    yml = tmp_path / "empty.yaml"
    yml.write_text("{}")
    with pytest.raises(SystemExit, match="no 'samples'"):
        main(["--samples", str(yml)])
