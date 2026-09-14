"""Tests for scripts/eval/score_detections.py."""

import json
import sys
from unittest import mock

import pytest

import scripts.eval.score_detections as sd


# ---------------------------------------------------------------------------
# _iou
# ---------------------------------------------------------------------------

def test_iou_identical_is_one():
    assert sd._iou([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert sd._iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_iou_half_overlap():
    # Two unit squares overlapping on half their area.
    v = sd._iou([0, 0, 2, 2], [1, 0, 3, 2])
    # inter = 2, union = 4 + 4 - 2 = 6
    assert v == pytest.approx(2 / 6)


# ---------------------------------------------------------------------------
# _match
# ---------------------------------------------------------------------------

def _box(x0, y0, x1, y1, cls="boat", conf=1.0):
    return {"cls": cls, "xyxy": [x0, y0, x1, y1], "confidence": conf}


def test_match_true_positive():
    preds = [_box(0, 0, 1, 1)]
    truths = [_box(0, 0, 1, 1)]
    tp, fp, fn = sd._match(preds, truths, 0.5, agnostic=False)
    assert tp["boat"] == 1
    assert fp["boat"] == 0
    assert fn["boat"] == 0


def test_match_false_positive_and_negative():
    preds = [_box(0, 0, 1, 1)]
    truths = [_box(5, 5, 6, 6)]
    tp, fp, fn = sd._match(preds, truths, 0.5, agnostic=False)
    assert tp["boat"] == 0
    assert fp["boat"] == 1
    assert fn["boat"] == 1


def test_match_skips_already_used_truth():
    # Two disjoint boxes both present in preds and truths. The first pred
    # claims truth[0]; the second pred must skip the used truth before
    # matching truth[1].
    a, b = _box(0, 0, 1, 1), _box(5, 5, 6, 6)
    tp, fp, fn = sd._match([a, b], [a, b], 0.5, agnostic=False)
    assert tp["boat"] == 2
    assert fp["boat"] == 0
    assert fn["boat"] == 0


def test_match_agnostic_collapses_classes():
    preds = [_box(0, 0, 1, 1, cls="duck")]
    truths = [_box(0, 0, 1, 1, cls="boat")]
    # Per-class this would be a FP + FN; agnostic it is a TP.
    tp, fp, fn = sd._match(preds, truths, 0.5, agnostic=True)
    assert tp["obstacle"] == 1
    assert fp["obstacle"] == 0
    assert fn["obstacle"] == 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def test_score_perfect():
    truth = {"f0": [_box(0, 0, 1, 1)]}
    pred = {"f0": [_box(0, 0, 1, 1)]}
    res = sd.score(truth, pred)
    assert res["per_class"]["boat"]["precision"] == 1.0
    assert res["per_class"]["boat"]["recall"] == 1.0
    assert res["per_class"]["boat"]["f1"] == 1.0
    assert res["agnostic"]["obstacle"]["f1"] == 1.0


def test_score_only_scores_annotated_frames():
    truth = {"f0": [_box(0, 0, 1, 1)]}
    # An extra predicted frame with no ground truth is ignored.
    pred = {"f0": [_box(0, 0, 1, 1)], "f99": [_box(0, 0, 1, 1)]}
    res = sd.score(truth, pred)
    assert res["per_class"]["boat"]["fp"] == 0


def test_score_missing_prediction_counts_fn():
    truth = {"f0": [_box(0, 0, 1, 1)]}
    res = sd.score(truth, {})  # no predictions at all
    assert res["per_class"]["boat"]["fn"] == 1
    assert res["per_class"]["boat"]["recall"] == 0.0


# ---------------------------------------------------------------------------
# _remap_pred
# ---------------------------------------------------------------------------

def test_remap_pred_uses_mapping_else_identity():
    pred = {"orig0": [1], "unmapped": [2]}
    out = sd._remap_pred(pred, {"orig0": "fake0"})
    assert out == {"fake0": [1], "unmapped": [2]}


# ---------------------------------------------------------------------------
# _load
# ---------------------------------------------------------------------------

def test_load_skips_blank_lines(tmp_path):
    p = tmp_path / "labels.jsonl"
    p.write_text(
        json.dumps({"frame_id": "f0", "fisheye_bboxes": [_box(0, 0, 1, 1)]}) +
        "\n\n" +
        json.dumps({"frame_id": "f1"}) + "\n"
    )
    out = sd._load(p)
    assert set(out) == {"f0", "f1"}
    assert out["f1"] == []  # missing key -> empty list


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def _write_mapping(tmp_path, fake="M/ts/ts=00-00-00.0", orig="Boats/x/ts=00-00-00.0"):
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({
        "fake_to_original": {fake: orig},
        "original_to_fake": {orig: fake},
    }))
    return mp, fake, orig


def test_main_pred_path(tmp_path, capsys):
    mp, fake, orig = _write_mapping(tmp_path)
    truth = tmp_path / "truth.jsonl"
    truth.write_text(json.dumps(
        {"frame_id": fake, "fisheye_bboxes": [_box(0, 0, 1, 1)]}) + "\n")
    pred = tmp_path / "pred.jsonl"
    pred.write_text(json.dumps(
        {"frame_id": orig, "fisheye_bboxes": [_box(0, 0, 1, 1)]}) + "\n")
    argv = ["sd", "--truth", str(truth), "--pred", str(pred),
            "--mapping", str(mp)]
    with mock.patch.object(sys, "argv", argv):
        sd.main()
    out = capsys.readouterr().out
    assert "agnostic" in out
    assert "per_class" in out


def test_main_grid_dir(tmp_path, capsys):
    mp, fake, orig = _write_mapping(tmp_path)
    truth = tmp_path / "truth.jsonl"
    truth.write_text(json.dumps(
        {"frame_id": fake, "fisheye_bboxes": [_box(0, 0, 1, 1)]}) + "\n")
    grid = tmp_path / "grid"
    cfg = grid / "configA"
    cfg.mkdir(parents=True)
    (cfg / "labels.jsonl").write_text(json.dumps(
        {"frame_id": orig, "fisheye_bboxes": [_box(0, 0, 1, 1)]}) + "\n")
    # A directory without labels.jsonl is skipped.
    (grid / "empty").mkdir()
    argv = ["sd", "--truth", str(truth), "--grid-dir", str(grid),
            "--mapping", str(mp)]
    with mock.patch.object(sys, "argv", argv):
        sd.main()
    out = capsys.readouterr().out
    assert "ranked by class-agnostic F1" in out
    assert "configA" in out


def test_main_no_truth_exits(tmp_path):
    mp, fake, orig = _write_mapping(tmp_path)
    truth = tmp_path / "truth.jsonl"
    # frame_id not in the eval id space -> filtered out -> empty truth.
    truth.write_text(json.dumps(
        {"frame_id": "Other/y/ts=00-00-00.0", "fisheye_bboxes": []}) + "\n")
    argv = ["sd", "--truth", str(truth), "--pred", str(truth), "--mapping", str(mp)]
    with mock.patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit):
            sd.main()


def test_main_requires_pred_or_grid(tmp_path):
    mp, fake, orig = _write_mapping(tmp_path)
    truth = tmp_path / "truth.jsonl"
    truth.write_text(json.dumps(
        {"frame_id": fake, "fisheye_bboxes": [_box(0, 0, 1, 1)]}) + "\n")
    argv = ["sd", "--truth", str(truth), "--mapping", str(mp)]
    with mock.patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit):
            sd.main()


def test_main_scene_filter(tmp_path, capsys):
    mp, fake, orig = _write_mapping(tmp_path)
    truth = tmp_path / "truth.jsonl"
    truth.write_text(json.dumps(
        {"frame_id": fake, "fisheye_bboxes": [_box(0, 0, 1, 1)]}) + "\n")
    pred = tmp_path / "pred.jsonl"
    pred.write_text(json.dumps(
        {"frame_id": orig, "fisheye_bboxes": [_box(0, 0, 1, 1)]}) + "\n")
    argv = ["sd", "--truth", str(truth), "--pred", str(pred),
            "--mapping", str(mp), "--scene", "M"]
    with mock.patch.object(sys, "argv", argv):
        sd.main()
    assert "annotated frames" in capsys.readouterr().out
