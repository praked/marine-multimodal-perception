import json

from scripts.eval.merge_teacher_suggestions import merge, merge_boxes, main


def _rec(fid, boxes, source):
    return {"frame_id": fid, "scene": fid.split("/")[0], "source": source,
            "fisheye_bboxes": boxes, "obstacle_bins_fisheye": [0]}


def test_matched_boxes_collapse_with_agree_flag():
    a = [{"cls": "boat", "xyxy": [0.1, 0.1, 0.3, 0.3], "confidence": 0.5}]
    b = [{"cls": "boat", "xyxy": [0.11, 0.1, 0.31, 0.3], "confidence": 0.9},
         {"cls": "duck", "xyxy": [0.6, 0.6, 0.62, 0.62], "confidence": 0.4}]
    out = merge_boxes(a, b, "grounding-dino", "dart-sam3", 0.5)
    assert len(out) == 2
    boat = next(x for x in out if x["cls"] == "boat")
    assert boat["source"] == "both" and boat["agree"] and boat["confidence"] == 0.9
    assert boat["confidence_other"] == 0.5
    duck = next(x for x in out if x["cls"] == "duck")
    assert duck["source"] == "dart-sam3" and not duck["agree"]


def test_class_mismatch_is_not_a_match():
    a = [{"cls": "boat", "xyxy": [0.1, 0.1, 0.3, 0.3], "confidence": 0.5}]
    b = [{"cls": "structure", "xyxy": [0.1, 0.1, 0.3, 0.3], "confidence": 0.9}]
    out = merge_boxes(a, b, "grounding-dino", "dart-sam3", 0.5)
    assert len(out) == 2 and not any(x["agree"] for x in out)


def test_merge_frames_union_and_stats(tmp_path):
    ra = {"s/c/ts=1": _rec("s/c/ts=1", [{"cls": "boat", "xyxy": [0, 0, .2, .2], "confidence": .5}], "grounding-dino"),
          "s/c/ts=2": _rec("s/c/ts=2", [], "grounding-dino")}
    rb = {"s/c/ts=1": _rec("s/c/ts=1", [{"cls": "boat", "xyxy": [0, 0, .2, .2], "confidence": .7}], "dart-sam3"),
          "s/c/ts=3": _rec("s/c/ts=3", [{"cls": "buoy", "xyxy": [.5, .5, .52, .52], "confidence": .6}], "dart-sam3")}
    merged, stats = merge(ra, rb, "grounding-dino", "dart-sam3", 0.5)
    assert stats == {"frames": 3, "only_a": 1, "only_b": 1, "both_frames": 1,
                     "boxes_a": 1, "boxes_b": 2, "boxes_out": 2, "agreed": 1}
    assert merged["s/c/ts=1"]["source"] == "union"
    assert merged["s/c/ts=1"]["teachers"] == ["dart-sam3", "grounding-dino"]
    assert "obstacle_bins_fisheye" not in merged["s/c/ts=1"]
    assert merged["s/c/ts=3"]["fisheye_bboxes"][0]["source"] == "dart-sam3"


def test_cli_writes_per_clip_files(tmp_path):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    a.mkdir(); b.mkdir()
    (a / "det_s__act.jsonl").write_text(json.dumps(_rec("s/c1/ts=1", [], "grounding-dino")) + "\n")
    (b / "det_s__c1.jsonl").write_text(json.dumps(_rec("s/c1/ts=1", [{"cls": "duck", "xyxy": [.1, .1, .12, .12], "confidence": .8}], "dart-sam3")) + "\n"
                                       + json.dumps(_rec("s/c2/ts=9", [], "dart-sam3")) + "\n")
    assert main(["--a", str(a), "--b", str(b), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["det_s__c1.jsonl", "det_s__c2.jsonl"]
    rec = json.loads((out / "det_s__c1.jsonl").read_text().splitlines()[0])
    assert rec["fisheye_bboxes"][0]["cls"] == "duck" and rec["source"] == "union"


def test_pairs_across_different_chunk_attribution():
    # Track-A DINO labels carry ACTIVITY-level frame_ids; DART per-chunk ones.
    # The same physical frame (same scene + frame_ts) must merge into ONE record.
    ra = {"s|17:16:04.0": {"frame_id": "s/act/ts=17-16-04.0", "scene": "s", "frame_ts": "17:16:04.0",
                           "triplet_ts": "act", "source": "grounding-dino",
                           "fisheye_bboxes": [{"cls": "boat", "xyxy": [0, 0, .2, .2], "confidence": .5}]}}
    rb = {"s|17:16:04.0": {"frame_id": "s/chunk/ts=17-16-04.0", "scene": "s", "frame_ts": "17:16:04.0",
                           "triplet_ts": "chunk", "source": "dart-sam3",
                           "fisheye_bboxes": [{"cls": "boat", "xyxy": [0, 0, .2, .2], "confidence": .9, "polygon": [0, 0, .2, 0, .2, .2]}]}}
    merged, stats = merge(ra, rb, "grounding-dino", "dart-sam3", 0.5)
    assert stats["both_frames"] == 1 and stats["agreed"] == 1
    rec = merged["s|17:16:04.0"]
    assert rec["frame_id"] == "s/act/ts=17-16-04.0"       # teacher A's id (bundle convention)
    box = rec["fisheye_bboxes"][0]
    assert box["source"] == "both" and box["confidence"] == 0.9 and "polygon" in box
