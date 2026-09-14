"""Tests for scripts/eval/export_yolo.py: dashboard labels -> YOLO dataset."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

import scripts.eval.export_yolo as ey

SCENE, TS = "Synth", "2099-01-01_00-00-00"


def _fid(i):
    return f"{SCENE}/{TS}/ts=00-00-0{i}.0"


def _make_triplet(captures_root):
    scene_dir = captures_root / SCENE
    scene_dir.mkdir(parents=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(scene_dir / f"fisheye_{TS}.mp4"), fourcc, 3.0, (160, 120))
    tw = cv2.VideoWriter(str(scene_dir / f"thermal_{TS}.mp4"), fourcc, 3.0, (160, 120))
    for i in range(5):
        f = np.full((120, 160, 3), 60 + i * 5, dtype=np.uint8)
        fw.write(f)
        tw.write(f)
    fw.release()
    tw.release()
    rows = ["Date,Time,X,Y,Z"]
    for i in range(5):
        rows.append(f"2099-01-01,00:00:0{i}.0,0.1,1.5,0.0")
    (scene_dir / f"mmwave_{TS}.csv").write_text("\n".join(rows) + "\n")


def _box(cls="boat", conf=0.9):
    return {"cls": cls, "xyxy": [0.2, 0.3, 0.6, 0.9], "confidence": conf}


def _setup(tmp_path, monkeypatch):
    captures = tmp_path / "captures"
    _make_triplet(captures)
    monkeypatch.setattr(ey, "CAPTURES_ROOT", captures)
    labels = tmp_path / "human.jsonl"
    labels.write_text("\n".join(
        json.dumps({"frame_id": _fid(i), "fisheye_bboxes": [_box()]})
        for i in range(3)) + "\n")
    return labels


def test_main_human_only(tmp_path, monkeypatch, capsys):
    labels = _setup(tmp_path, monkeypatch)
    out = tmp_path / "yolo"
    rc = ey.main(["--labels", str(labels), "--out", str(out)])
    assert rc == 0
    # i=0 -> val, i=1,2 -> train (repeat 1)
    assert len(list((out / "images" / "val").glob("*.jpg"))) == 1
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 2
    data = (out / "data.yaml").read_text()
    assert "nc: 1" in data and "boat" in data
    lbl = next((out / "labels" / "train").glob("*.txt")).read_text().strip()
    cls, cx, cy, w, h = lbl.split()
    assert cls == "0"
    assert abs(float(cx) - 0.4) < 1e-6 and abs(float(w) - 0.4) < 1e-6


def test_main_mixed_pseudo_with_exclude_and_repeat(tmp_path, monkeypatch, capsys):
    labels = _setup(tmp_path, monkeypatch)
    pseudo = tmp_path / "pseudo.jsonl"
    pseudo.write_text("\n".join([
        json.dumps({"frame_id": _fid(3), "fisheye_bboxes": [_box(conf=0.5)]}),
        json.dumps({"frame_id": _fid(4), "fisheye_bboxes": [_box(conf=0.5)]}),
        json.dumps({"frame_id": _fid(0), "fisheye_bboxes": [_box()]}),   # human -> excluded
        json.dumps({"frame_id": _fid(4), "fisheye_bboxes": [_box("duck")]}),  # class not kept
    ]) + "\n")
    mapping = tmp_path / "eval_map.json"
    mapping.write_text(json.dumps({"fake_to_original": {"fake0": _fid(3)}}))

    out = tmp_path / "yolo_mixed"
    rc = ey.main(["--labels", str(labels), "--out", str(out),
                  "--classes", "boat",
                  "--pseudo", str(pseudo), "--pseudo-count", "2",
                  "--exclude", str(mapping), "--human-repeat", "2"])
    assert rc == 0
    # val: human i=0. train: human i=1,2 x2 copies + 1 pseudo (frame 4 only;
    # frame 3 is excluded by the eval mapping, frame 0 is already human).
    assert len(list((out / "images" / "val").glob("*.jpg"))) == 1
    train_imgs = sorted(p.name for p in (out / "images" / "train").glob("*.jpg"))
    assert len(train_imgs) == 5
    assert sum("_r0" in n for n in train_imgs) == 2   # oversampled copies
    printed = capsys.readouterr().out
    assert "pseudo 1" in printed

    # re-running wipes and rebuilds (rmtree branch)
    assert ey.main(["--labels", str(labels), "--out", str(out),
                    "--classes", "boat"]) == 0
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 2


def test_main_no_labelled_frames_exits(tmp_path, monkeypatch):
    labels = tmp_path / "empty.jsonl"
    labels.write_text(json.dumps({"frame_id": _fid(0), "fisheye_bboxes": []}) + "\n")
    with pytest.raises(SystemExit, match="no labelled frames"):
        ey.main(["--labels", str(labels), "--out", str(tmp_path / "o")])


def test_xyxy_to_cxcywh():
    got = ey._xyxy_to_cxcywh([0.2, 0.3, 0.6, 0.9])
    assert np.allclose(got, (0.4, 0.6, 0.4, 0.6))
