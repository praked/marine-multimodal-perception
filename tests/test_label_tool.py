import json
import sys
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

import scripts.eval.label_tool as lt
from scripts.eval.label_tool import (
    AuditSession,
    FrameLabel,
    LabelSession,
    _append_label,
    _derive_bins,
    _iter_dir_frames,
    _iter_triplet_frames,
    _load_provisional,
    _seek_frame,
    main,
)


def _mock_cv2_ui(monkeypatch):
    monkeypatch.setattr("cv2.namedWindow", lambda *a, **k: None)
    monkeypatch.setattr("cv2.destroyAllWindows", lambda: None)
    monkeypatch.setattr("cv2.destroyWindow", lambda *a, **k: None)
    monkeypatch.setattr("cv2.setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr("cv2.imshow", lambda *a, **k: None)


def test_derive_bins_centre_box(intrinsics_real, detection_real):
    cx = intrinsics_real["fisheye"]["cx"]
    pdr = intrinsics_real["fisheye"]["pix_deg_ratio"]
    bins = _derive_bins([{"cls": "boat", "xyxy": [0.49, 0.5, 0.51, 0.52]}],
                        image_w=864, image_h=648,
                        cx=cx, pix_deg_ratio=pdr,
                        fusion_params=detection_real["fusion"])
    assert isinstance(bins, list)


def test_derive_bins_empty():
    bins = _derive_bins([], 800, 600, 400, 7.2, {"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10})
    assert bins == []


def test_derive_bins_off_screen_excluded():
    """An angle outside the bin range is silently dropped."""
    fusion = {"bin_min_deg": -55, "bin_max_deg": 55, "bin_step_deg": 10}
    # Place bbox far to the right -> angle > 55 deg if pix_deg_ratio is small.
    bins = _derive_bins([{"cls": "boat", "xyxy": [0.95, 0.5, 1.0, 0.55]}],
                        image_w=800, image_h=600, cx=0, pix_deg_ratio=1.0,
                        fusion_params=fusion)
    assert bins == []


def test_append_label_writes_jsonl(tmp_path):
    path = tmp_path / "out.jsonl"
    lbl = FrameLabel(
        frame_id="Boats/ts/0", scene="Boats",
        source="manual", audited=True,
        fisheye_bboxes=[{"cls": "boat", "xyxy": [0.1, 0.2, 0.3, 0.4]}],
        obstacle_bins_fisheye=[-5, 5],
        width=800, height=600,
    )
    _append_label(lbl, path)
    line = path.read_text().strip()
    obj = json.loads(line)
    assert obj["frame_id"] == "Boats/ts/0"
    assert obj["audited"] is True


def test_append_label_appends_multiple(tmp_path):
    path = tmp_path / "out.jsonl"
    for i in range(3):
        _append_label(FrameLabel(
            frame_id=f"Scene/ts/{i}", scene="X", source="m", audited=True,
            fisheye_bboxes=[], obstacle_bins_fisheye=[], width=10, height=10,
        ), path)
    assert path.read_text().count("\n") == 3


def test_load_provisional_filters_audited(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        json.dumps({"frame_id": "A/b/0", "audited": False}) + "\n" +
        json.dumps({"frame_id": "A/b/1", "audited": True}) + "\n"
    )
    out = _load_provisional(p)
    assert len(out) == 1
    assert out[0]["frame_id"] == "A/b/0"


def test_load_provisional_missing_file(tmp_path):
    out = _load_provisional(tmp_path / "nope.jsonl")
    assert out == []


def test_load_provisional_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n\n" + json.dumps({"frame_id": "A/b/0", "audited": False}) + "\n\n")
    assert len(_load_provisional(p)) == 1


# ---------------------------------------------------------------------------
# Frame iteration helpers
# ---------------------------------------------------------------------------

def test_iter_triplet_frames_strides(monkeypatch, boats_triplet, intrinsics_real,
                                     detection_real):
    out = _iter_triplet_frames(
        str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        every=120, intrinsics=intrinsics_real, detection=detection_real,
    )
    assert len(out) > 0
    fr = out[0]
    assert fr.frame_id.startswith("Boats/")
    assert fr.scene == "Boats"
    assert fr.img.shape[2] == 3


def test_iter_triplet_frames_unreadable(tmp_path, intrinsics_real, detection_real):
    fake = tmp_path / "fisheye_x.mp4"; fake.write_bytes(b"x")
    fake_t = tmp_path / "thermal_x.mp4"; fake_t.write_bytes(b"x")
    fake_m = tmp_path / "mmwave_x.csv"; fake_m.write_text("Date,Time,X,Y,Z\n")
    with pytest.raises(RuntimeError):
        _iter_triplet_frames(str(tmp_path / "x"), every=1,
                             intrinsics=intrinsics_real, detection=detection_real)


def test_iter_dir_frames(tmp_path):
    # data/Scene/<dir>/jpgs layout per the helper.
    base = tmp_path / "data" / "Scene" / "ts"
    base.mkdir(parents=True)
    for i in range(3):
        img = np.full((10, 10, 3), 80, dtype=np.uint8)
        cv2.imwrite(str(base / f"{i:06d}.jpg"), img)
    out = _iter_dir_frames(str(base))
    assert len(out) == 3
    assert out[0][1] == "Scene"


def test_iter_dir_frames_skips_unreadable(tmp_path):
    base = tmp_path / "data" / "Scene" / "ts"
    base.mkdir(parents=True)
    (base / "broken.jpg").write_bytes(b"")
    out = _iter_dir_frames(str(base))
    assert out == []


def test_seek_frame_real(boats_triplet):
    img = _seek_frame(boats_triplet.fisheye, 0)
    assert img is not None
    assert img.shape[2] == 3


def test_seek_frame_unreadable(tmp_path):
    fake = tmp_path / "x.mp4"; fake.write_bytes(b"")
    assert _seek_frame(fake, 0) is None


# ---------------------------------------------------------------------------
# LabelSession behaviour (cv2 mocked)
# ---------------------------------------------------------------------------

def test_label_session_save_with_classed_bbox(tmp_path, monkeypatch,
                                              intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "manual.jsonl")
    monkeypatch.setattr(lt, "LABELS_DIR", tmp_path)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    frames = [lt.Frame("Scene/ts/0", "Scene", "ts", None, img)]
    sess = LabelSession(frames, intrinsics_real, detection_real)
    sess.bboxes = [{"cls": "boat", "xyxy": [0.1, 0.2, 0.3, 0.4]}]
    sess._save_current()
    obj = json.loads((tmp_path / "manual.jsonl").read_text())
    assert obj["fisheye_bboxes"][0]["cls"] == "boat"


def test_label_session_save_none_marks_empty(tmp_path, monkeypatch,
                                             intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "manual.jsonl")
    monkeypatch.setattr(lt, "LABELS_DIR", tmp_path)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)], intrinsics_real, detection_real)
    sess.bboxes = [{"cls": "boat", "xyxy": [0.1, 0.2, 0.3, 0.4]}]
    sess._save_current(none=True)
    obj = json.loads((tmp_path / "manual.jsonl").read_text())
    assert obj["fisheye_bboxes"] == []


def test_label_session_mouse_callback_draws_bbox(monkeypatch,
                                                  intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)], intrinsics_real, detection_real)
    sess._on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
    sess._on_mouse(cv2.EVENT_MOUSEMOVE, 150, 150, 0, None)
    sess._on_mouse(cv2.EVENT_LBUTTONUP, 200, 200, 0, None)
    assert len(sess.bboxes) == 1
    # A freshly drawn box takes the session's active class (default = first
    # configured label class).
    assert sess.bboxes[0]["cls"] == sess.active_cls


def test_label_session_mouse_callback_too_small_ignored(monkeypatch,
                                                        intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)], intrinsics_real, detection_real)
    sess._on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
    sess._on_mouse(cv2.EVENT_LBUTTONUP, 101, 101, 0, None)
    assert sess.bboxes == []


def test_label_session_run_keys(monkeypatch, tmp_path, intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "out.jsonl")
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    frames = [lt.Frame("Scene/ts/0", "Scene", "ts", None, img), lt.Frame("Scene/ts/1", "Scene", "ts", None, img)]
    sess = LabelSession(frames, intrinsics_real, detection_real)
    # Programmatically inject bboxes; the run loop will save + advance + quit.
    sess.bboxes = [{"cls": "unset", "xyxy": [0.1, 0.1, 0.2, 0.2]}]
    keys = iter([ord("b"), ord("n"),  # set class, next
                 ord(" "),             # mark none on frame 1
                 ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    assert (tmp_path / "out.jsonl").exists()
    n_lines = len((tmp_path / "out.jsonl").read_text().strip().splitlines())
    assert n_lines >= 1


def test_label_session_save_omits_heading_when_unset(tmp_path, monkeypatch,
                                                     intrinsics_real, detection_real):
    """A label where the operator never touched the heading caret must
    not write a `safe_heading_deg` key. Heading-aware metrics treat
    missing keys as 'unlabelled', not as 0°."""
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "manual.jsonl")
    monkeypatch.setattr(lt, "LABELS_DIR", tmp_path)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)], intrinsics_real, detection_real)
    sess._save_current()
    obj = json.loads((tmp_path / "manual.jsonl").read_text())
    assert "safe_heading_deg" not in obj


def test_label_session_save_writes_heading_when_set(tmp_path, monkeypatch,
                                                    intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "manual.jsonl")
    monkeypatch.setattr(lt, "LABELS_DIR", tmp_path)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)], intrinsics_real, detection_real)
    sess.safe_heading_deg = -22.0
    sess.safe_heading_set = True
    sess._save_current()
    obj = json.loads((tmp_path / "manual.jsonl").read_text())
    assert obj["safe_heading_deg"] == -22.0


def test_label_session_save_writes_null_heading_explicitly(tmp_path, monkeypatch,
                                                           intrinsics_real, detection_real):
    """`h` (null) should round-trip as JSON null: analogous to the
    recommender's `all_blocked` abstention. Differentiates from
    'unlabelled' (key missing entirely)."""
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "manual.jsonl")
    monkeypatch.setattr(lt, "LABELS_DIR", tmp_path)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)], intrinsics_real, detection_real)
    sess.safe_heading_deg = None
    sess.safe_heading_set = True
    sess._save_current()
    obj = json.loads((tmp_path / "manual.jsonl").read_text())
    assert "safe_heading_deg" in obj
    assert obj["safe_heading_deg"] is None


def test_label_session_nudge_heading_clamps_to_fov(monkeypatch,
                                                   intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)], intrinsics_real, detection_real)
    for _ in range(20):
        sess._nudge_heading(+5.0)
    assert sess.safe_heading_deg == 55.0
    assert sess.safe_heading_set is True
    for _ in range(40):
        sess._nudge_heading(-5.0)
    assert sess.safe_heading_deg == -55.0


def test_label_session_h_toggles_null_within_frame(monkeypatch,
                                                   intrinsics_real, detection_real, tmp_path):
    """Pressing `h` once marks 'null' (no safe direction); pressing
    `h` again brings the caret back to 0°. The toggle is intra-frame:
    a fresh frame always starts unset."""
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "out.jsonl")
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img),
                         lt.Frame("Scene/ts/1", "Scene", "ts", None, img)], intrinsics_real, detection_real)
    # Frame 0: h, h, n → ends back at 0°, saved as 0°.
    # Frame 1: h, n     → saved as null.
    keys = iter([ord("h"), ord("h"), ord("n"),
                 ord("h"), ord("n"),
                 ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    lines = (tmp_path / "out.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    a = json.loads(lines[0])
    b = json.loads(lines[1])
    assert a["safe_heading_deg"] == 0.0
    assert b["safe_heading_deg"] is None


def test_label_session_undo(monkeypatch, intrinsics_real, detection_real, tmp_path):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "out.jsonl")
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)], intrinsics_real, detection_real)
    sess.bboxes = [{"cls": "boat", "xyxy": [0.1, 0.1, 0.2, 0.2]}]
    keys = iter([ord("u"), ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    assert sess.bboxes == []


def test_label_session_skip(monkeypatch, intrinsics_real, detection_real, tmp_path):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "out.jsonl")
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img),
                         lt.Frame("Scene/ts/1", "Scene", "ts", None, img)], intrinsics_real, detection_real)
    keys = iter([ord("s"), ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    # Nothing was saved since we skipped.
    assert not (tmp_path / "out.jsonl").exists()


# ---------------------------------------------------------------------------
# AuditSession
# ---------------------------------------------------------------------------

def test_audit_accept_writes_master(monkeypatch, tmp_path, intrinsics_real,
                                     detection_real, boats_triplet):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MASTER_PATH", tmp_path / "master.jsonl")
    monkeypatch.setattr(lt, "LABELS_DIR", tmp_path)
    items = [{
        "frame_id": f"{boats_triplet.clip_id}/000000",
        "scene": "Boats", "source": "qwen-vl",
        "fisheye_bboxes": [{"cls": "boat", "xyxy": [0.4, 0.5, 0.5, 0.6]}],
        "obstacle_bins_fisheye": [0],
    }]
    sess = AuditSession(items, intrinsics_real, detection_real)
    keys = iter([ord("a"), ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    text = (tmp_path / "master.jsonl").read_text()
    obj = json.loads(text.strip())
    assert obj["audited"] is True
    assert sess.counts["accept"] == 1


def test_audit_reject(monkeypatch, tmp_path, intrinsics_real, detection_real,
                      boats_triplet):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MASTER_PATH", tmp_path / "master.jsonl")
    monkeypatch.setattr(lt, "LABELS_DIR", tmp_path)
    items = [{
        "frame_id": f"{boats_triplet.clip_id}/000000", "scene": "Boats",
        "fisheye_bboxes": [{"cls": "boat", "xyxy": [0.4, 0.5, 0.5, 0.6]}],
        "obstacle_bins_fisheye": [0],
    }]
    sess = AuditSession(items, intrinsics_real, detection_real)
    keys = iter([ord("r"), ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    obj = json.loads((tmp_path / "master.jsonl").read_text().strip())
    assert obj["fisheye_bboxes"] == []
    assert sess.counts["reject"] == 1


def test_audit_skip(monkeypatch, tmp_path, intrinsics_real, detection_real,
                    boats_triplet):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MASTER_PATH", tmp_path / "master.jsonl")
    items = [{
        "frame_id": f"{boats_triplet.clip_id}/000000", "scene": "Boats",
        "fisheye_bboxes": [], "obstacle_bins_fisheye": [],
    }]
    sess = AuditSession(items, intrinsics_real, detection_real)
    keys = iter([ord("n"), ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    assert sess.counts["skip"] == 1
    assert not (tmp_path / "master.jsonl").exists()


def test_audit_unresolvable_frame_skips(monkeypatch, tmp_path,
                                        intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MASTER_PATH", tmp_path / "master.jsonl")
    items = [{
        "frame_id": "Nonexistent/9999-99-99_99-99-99/000000",
        "scene": "Nonexistent",
        "fisheye_bboxes": [], "obstacle_bins_fisheye": [],
    }]
    sess = AuditSession(items, intrinsics_real, detection_real)
    sess.run()
    # _resolve_frame returns None -> incremented idx, no master write.
    assert not (tmp_path / "master.jsonl").exists()


# ---------------------------------------------------------------------------
# main() entry points
# ---------------------------------------------------------------------------

def test_main_audit_no_entries_exits(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with mock.patch.object(sys, "argv", ["lt", "--audit", str(p)]):
        with pytest.raises(SystemExit):
            main()


def test_main_no_frames_exits(monkeypatch, tmp_path):
    _mock_cv2_ui(monkeypatch)
    empty = tmp_path / "empty_dir"; empty.mkdir()
    with mock.patch.object(sys, "argv", ["lt", "--frames", str(empty)]):
        with pytest.raises(SystemExit):
            main()


# ---------------------------------------------------------------------------
# _frame_ts_from_id (covers lines 115, 118)
# ---------------------------------------------------------------------------

def test_frame_ts_from_id_wrong_parts():
    # Not exactly 3 "/"-separated parts -> (None, None).
    assert lt._frame_ts_from_id("Scene/ts") == (None, None)
    assert lt._frame_ts_from_id("a/b/c/d") == (None, None)


def test_frame_ts_from_id_ts_scheme():
    # The ts= scheme returns (triplet_ts, frame_ts) with dashes -> colons.
    tts, fts = lt._frame_ts_from_id("Boats/2025-06-23_16-21-07/ts=00-00-01.0")
    assert tts == "2025-06-23_16-21-07"
    assert fts == "00:00:01.0"


def test_frame_ts_from_id_legacy_scheme():
    # Legacy index scheme -> (triplet_ts, None).
    tts, fts = lt._frame_ts_from_id("Boats/2025-06-23_16-21-07/000000")
    assert tts == "2025-06-23_16-21-07"
    assert fts is None


# ---------------------------------------------------------------------------
# _append_label frame_ts branch (covers line 168)
# ---------------------------------------------------------------------------

def test_append_label_writes_frame_ts(tmp_path):
    path = tmp_path / "out.jsonl"
    lbl = FrameLabel(
        frame_id="Boats/ts/ts=00-00-01.0", scene="Boats",
        source="manual", audited=True, fisheye_bboxes=[],
        obstacle_bins_fisheye=[], width=10, height=10,
        triplet_ts="ts", frame_ts="00:00:01.0",
    )
    _append_label(lbl, path)
    obj = json.loads(path.read_text().strip())
    assert obj["triplet_ts"] == "ts"
    assert obj["frame_ts"] == "00:00:01.0"


# ---------------------------------------------------------------------------
# _draw branches (covers line 235 drag preview)
# ---------------------------------------------------------------------------

def test_label_session_draw_with_drag_preview(monkeypatch, intrinsics_real,
                                              detection_real):
    _mock_cv2_ui(monkeypatch)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)],
                        intrinsics_real, detection_real)
    # Active drag in progress -> preview rectangle path is exercised.
    sess.drag_start = (10, 10)
    sess.preview = (50, 50)
    sess.safe_heading_set = True
    sess.safe_heading_deg = 5.0
    sess._draw()  # should not raise


def test_label_session_draw_null_heading_label(monkeypatch, intrinsics_real,
                                               detection_real):
    _mock_cv2_ui(monkeypatch)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)],
                        intrinsics_real, detection_real)
    sess.safe_heading_set = True
    sess.safe_heading_deg = None
    sess._draw()  # null-heading label branch


# ---------------------------------------------------------------------------
# _save_current dropped-bbox warning (covers line 263)
# ---------------------------------------------------------------------------

def test_label_session_save_drops_unset_bbox(tmp_path, monkeypatch, capsys,
                                             intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "manual.jsonl")
    monkeypatch.setattr(lt, "LABELS_DIR", tmp_path)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)],
                        intrinsics_real, detection_real)
    sess.bboxes = [
        {"cls": "boat", "xyxy": [0.1, 0.2, 0.3, 0.4]},
        {"cls": "unset", "xyxy": [0.5, 0.5, 0.6, 0.6]},
    ]
    sess._save_current()
    out = capsys.readouterr().out
    assert "dropped 1 bboxes without a class" in out
    obj = json.loads((tmp_path / "manual.jsonl").read_text())
    assert len(obj["fisheye_bboxes"]) == 1


# ---------------------------------------------------------------------------
# _nudge_heading None-init + fine heading keys (covers 290, 311, 313, 315, 317)
# ---------------------------------------------------------------------------

def test_nudge_heading_from_none(monkeypatch, intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)],
                        intrinsics_real, detection_real)
    sess.safe_heading_deg = None  # null -> nudging re-initialises to 0
    sess._nudge_heading(+5.0)
    assert sess.safe_heading_deg == 5.0
    assert sess.safe_heading_set is True


def test_label_session_fine_and_coarse_heading_keys(monkeypatch, tmp_path,
                                                    intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "out.jsonl")
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)],
                        intrinsics_real, detection_real)
    # 255 (no key) -> continue; , . < > nudge; n save; q quit.
    keys = iter([255,
                 ord("."), ord("."),   # +10
                 ord(","),             # -5  -> +5
                 ord(">"),             # +1  -> +6
                 ord("<"), ord("<"),   # -2  -> +4
                 ord("n"), ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    obj = json.loads((tmp_path / "out.jsonl").read_text().strip())
    assert obj["safe_heading_deg"] == 4.0


# ---------------------------------------------------------------------------
# backspace / previous frame (covers 333-335)
# ---------------------------------------------------------------------------

def test_label_session_backspace_goes_back(monkeypatch, tmp_path,
                                           intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "out.jsonl")
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    frames = [lt.Frame("Scene/ts/0", "Scene", "ts", None, img),
              lt.Frame("Scene/ts/1", "Scene", "ts", None, img)]
    sess = LabelSession(frames, intrinsics_real, detection_real)
    # n -> advance to frame 1; backspace (8) -> back to frame 0; q.
    keys = iter([ord("n"), 8, ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    assert sess.idx == 0


def test_label_session_backspace_at_zero_noop(monkeypatch, tmp_path,
                                              intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "out.jsonl")
    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    sess = LabelSession([lt.Frame("Scene/ts/0", "Scene", "ts", None, img)],
                        intrinsics_real, detection_real)
    # backspace at idx 0 must not go negative.
    keys = iter([127, ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    assert sess.idx == 0


# ---------------------------------------------------------------------------
# AuditSession ts= resolution: _load_clip + _resolve_frame (432-449, 457-461)
# ---------------------------------------------------------------------------

def _patch_scene_dir(monkeypatch, scene_path):
    monkeypatch.setattr(lt, "_scene_dir", lambda scene: scene_path)


def test_audit_resolve_ts_scheme(monkeypatch, tmp_path, intrinsics_real,
                                 detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MASTER_PATH", tmp_path / "master.jsonl")
    monkeypatch.setattr(lt, "LABELS_DIR", tmp_path)
    scene_dir = tmp_path / "Synth"
    scene_dir.mkdir()
    _patch_scene_dir(monkeypatch, scene_dir)

    img = np.full((120, 160, 3), 70, dtype=np.uint8)
    fid = "Synth/2099-01-01_00-00-00/ts=00-00-00.0"

    def fake_iter(cdir, triplet_ts, every, det, intr):
        yield (fid, "Synth", triplet_ts, "00:00:00.0", img)

    monkeypatch.setattr(lt, "iter_clip_frames", fake_iter)

    items = [{
        "frame_id": fid, "scene": "Synth", "source": "qwen3-vl",
        "fisheye_bboxes": [{"cls": "boat", "xyxy": [0.4, 0.5, 0.5, 0.6]}],
        "obstacle_bins_fisheye": [0],
    }]
    sess = AuditSession(items, intrinsics_real, detection_real)
    keys = iter([ord("a"), ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    obj = json.loads((tmp_path / "master.jsonl").read_text().strip())
    assert obj["frame_id"] == fid
    assert sess.counts["accept"] == 1


def test_audit_load_clip_no_scene_dir(monkeypatch, tmp_path, intrinsics_real,
                                      detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "_scene_dir", lambda scene: None)
    items = [{"frame_id": "X/ts/ts=00-00-00.0", "scene": "X",
              "fisheye_bboxes": [], "obstacle_bins_fisheye": []}]
    sess = AuditSession(items, intrinsics_real, detection_real)
    # _load_clip returns {} (cdir None) -> resolve returns None -> skip.
    assert sess._load_clip("X", "ts") == {}
    assert sess._resolve_frame("X/ts/ts=00-00-00.0") is None


def test_audit_load_clip_no_wanted(monkeypatch, tmp_path, intrinsics_real,
                                   detection_real):
    _mock_cv2_ui(monkeypatch)
    scene_dir = tmp_path / "Y"; scene_dir.mkdir()
    monkeypatch.setattr(lt, "_scene_dir", lambda scene: scene_dir)
    # item is ts= for a *different* triplet, so wanted set is empty.
    items = [{"frame_id": "Y/other/ts=00-00-00.0", "scene": "Y",
              "fisheye_bboxes": [], "obstacle_bins_fisheye": []}]
    sess = AuditSession(items, intrinsics_real, detection_real)
    assert sess._load_clip("Y", "ts") == {}


def test_audit_load_clip_exception(monkeypatch, tmp_path, capsys,
                                   intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    scene_dir = tmp_path / "Z"; scene_dir.mkdir()
    monkeypatch.setattr(lt, "_scene_dir", lambda scene: scene_dir)
    fid = "Z/ts/ts=00-00-00.0"

    def boom(*a, **k):
        raise RuntimeError("bad clip")

    monkeypatch.setattr(lt, "iter_clip_frames", boom)
    items = [{"frame_id": fid, "scene": "Z",
              "fisheye_bboxes": [], "obstacle_bins_fisheye": []}]
    sess = AuditSession(items, intrinsics_real, detection_real)
    cache = sess._load_clip("Z", "ts")
    assert cache == {}
    assert "could not load clip" in capsys.readouterr().out


def test_resolve_frame_wrong_parts(monkeypatch, intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    sess = AuditSession([], intrinsics_real, detection_real)
    assert sess._resolve_frame("only/two") is None


# ---------------------------------------------------------------------------
# AuditSession legacy index resolution (covers 468-469, 472)
# ---------------------------------------------------------------------------

def test_resolve_frame_legacy_triplet_not_found(monkeypatch, tmp_path,
                                                intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    scene_dir = tmp_path / "S"; scene_dir.mkdir()
    monkeypatch.setattr(lt, "_scene_dir", lambda scene: scene_dir)

    def boom(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(lt, "resolve_triplet", boom)
    sess = AuditSession([], intrinsics_real, detection_real)
    assert sess._resolve_frame("S/ts/000000") is None


def test_resolve_frame_legacy_seek_none(monkeypatch, tmp_path,
                                        intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    scene_dir = tmp_path / "S"; scene_dir.mkdir()
    monkeypatch.setattr(lt, "_scene_dir", lambda scene: scene_dir)

    class FakeTriplet:
        fisheye = tmp_path / "fisheye_x.mp4"

    monkeypatch.setattr(lt, "resolve_triplet", lambda p: FakeTriplet())
    monkeypatch.setattr(lt, "_seek_frame", lambda v, i: None)
    sess = AuditSession([], intrinsics_real, detection_real)
    assert sess._resolve_frame("S/ts/000000") is None


def test_resolve_frame_legacy_success(monkeypatch, tmp_path,
                                      intrinsics_real, detection_real):
    _mock_cv2_ui(monkeypatch)
    scene_dir = tmp_path / "S"; scene_dir.mkdir()
    monkeypatch.setattr(lt, "_scene_dir", lambda scene: scene_dir)

    class FakeTriplet:
        fisheye = tmp_path / "fisheye_x.mp4"

    raw = np.full((648, 864, 3), 70, dtype=np.uint8)
    monkeypatch.setattr(lt, "resolve_triplet", lambda p: FakeTriplet())
    monkeypatch.setattr(lt, "_seek_frame", lambda v, i: raw)
    monkeypatch.setattr(lt, "undistort_fisheye", lambda r, K, D: r)
    sess = AuditSession([], intrinsics_real, detection_real)
    out = sess._resolve_frame("S/ts/000007")
    assert out is not None
    assert out.shape[2] == 3


# ---------------------------------------------------------------------------
# AuditSession _edit (covers 522-550) + run e/q branches (569, 572-573)
# ---------------------------------------------------------------------------

def test_audit_edit_saves_edited_label(monkeypatch, tmp_path, intrinsics_real,
                                       detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MASTER_PATH", tmp_path / "master.jsonl")
    monkeypatch.setattr(lt, "LABELS_DIR", tmp_path)
    scene_dir = tmp_path / "Synth"; scene_dir.mkdir()
    monkeypatch.setattr(lt, "_scene_dir", lambda scene: scene_dir)

    img = np.full((648, 864, 3), 70, dtype=np.uint8)
    fid = "Synth/ts/ts=00-00-00.0"

    def fake_iter(cdir, triplet_ts, every, det, intr):
        yield (fid, "Synth", triplet_ts, "00:00:00.0", img)

    monkeypatch.setattr(lt, "iter_clip_frames", fake_iter)

    items = [{
        "frame_id": fid, "scene": "Synth", "source": "qwen3-vl",
        "fisheye_bboxes": [
            {"cls": "boat", "xyxy": [0.4, 0.5, 0.5, 0.6]},
            {"cls": "unset", "xyxy": [0.1, 0.1, 0.2, 0.2]},
        ],
        "obstacle_bins_fisheye": [0],
    }]
    sess = AuditSession(items, intrinsics_real, detection_real)
    # Outer loop key 'e' -> _edit. Inside edit: 'd' reclass last bbox,
    # 'u' pop, 'b' reclass, then 'n' save+break. Then outer 'q'.
    keys = iter([ord("e"),
                 ord("d"), ord("u"), ord("b"), ord("n"),
                 ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    obj = json.loads((tmp_path / "master.jsonl").read_text().strip())
    assert obj["source"].endswith("+edited")
    assert sess.counts["edit"] == 1
    # The unset bbox was dropped by the kept-filter on save.
    assert all(bb["cls"] != "unset" for bb in obj["fisheye_bboxes"])


def test_audit_edit_quit_without_save(monkeypatch, tmp_path, intrinsics_real,
                                      detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MASTER_PATH", tmp_path / "master.jsonl")
    monkeypatch.setattr(lt, "LABELS_DIR", tmp_path)
    scene_dir = tmp_path / "Synth"; scene_dir.mkdir()
    monkeypatch.setattr(lt, "_scene_dir", lambda scene: scene_dir)
    img = np.full((648, 864, 3), 70, dtype=np.uint8)
    fid = "Synth/ts/ts=00-00-00.0"
    monkeypatch.setattr(lt, "iter_clip_frames",
                        lambda *a, **k: iter([(fid, "Synth", "ts",
                                               "00:00:00.0", img)]))
    items = [{"frame_id": fid, "scene": "Synth",
              "fisheye_bboxes": [], "obstacle_bins_fisheye": []}]
    sess = AuditSession(items, intrinsics_real, detection_real)
    # 'e' -> edit; inside: 'q' breaks edit without save; outer 'q'.
    keys = iter([ord("e"), ord("q"), ord("q")])
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    assert sess.counts["edit"] == 0
    assert not (tmp_path / "master.jsonl").exists()


# ---------------------------------------------------------------------------
# AuditSession run() key==255 continue + q quit (covers 563, 572-573)
# ---------------------------------------------------------------------------

def test_audit_run_noop_key_then_quit(monkeypatch, tmp_path, intrinsics_real,
                                      detection_real):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MASTER_PATH", tmp_path / "master.jsonl")
    scene_dir = tmp_path / "Synth"; scene_dir.mkdir()
    monkeypatch.setattr(lt, "_scene_dir", lambda scene: scene_dir)
    img = np.full((120, 160, 3), 70, dtype=np.uint8)
    fid = "Synth/ts/ts=00-00-00.0"
    monkeypatch.setattr(lt, "iter_clip_frames",
                        lambda *a, **k: iter([(fid, "Synth", "ts",
                                               "00:00:00.0", img)]))
    items = [{"frame_id": fid, "scene": "Synth",
              "fisheye_bboxes": [], "obstacle_bins_fisheye": []}]
    sess = AuditSession(items, intrinsics_real, detection_real)
    keys = iter([255, ord("q")])  # 255 -> continue, then quit
    monkeypatch.setattr("cv2.waitKey", lambda x: next(keys, ord("q")) | 0)
    sess.run()
    assert sess.counts == {"accept": 0, "reject": 0, "edit": 0, "skip": 0}


# ---------------------------------------------------------------------------
# main() audit branch (covers 599-601) and triplet branch (604)
# ---------------------------------------------------------------------------

def test_main_audit_runs(monkeypatch, tmp_path):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MASTER_PATH", tmp_path / "master.jsonl")
    p = tmp_path / "prov.jsonl"
    p.write_text(json.dumps({
        "frame_id": "Synth/ts/ts=00-00-00.0", "scene": "Synth",
        "audited": False, "fisheye_bboxes": [], "obstacle_bins_fisheye": [],
    }) + "\n")

    # AuditSession is constructed and .run() called; stub run to avoid GUI loop.
    ran = {}

    class FakeAudit:
        def __init__(self, prov, intr, det):
            ran["n"] = len(prov)

        def run(self):
            ran["ran"] = True

    monkeypatch.setattr(lt, "AuditSession", FakeAudit)
    with mock.patch.object(sys, "argv", ["lt", "--audit", str(p)]):
        main()
    assert ran == {"n": 1, "ran": True}


def test_main_triplet_runs(monkeypatch, tmp_path):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "manual.jsonl")
    img = np.full((100, 100, 3), 70, dtype=np.uint8)
    frame = lt.Frame("Scene/ts/ts=00-00-00.0", "Scene", "ts", "00:00:00.0", img)
    monkeypatch.setattr(lt, "_iter_triplet_frames",
                        lambda *a, **k: [frame])

    ran = {}

    class FakeLabel:
        def __init__(self, frames, intr, det):
            ran["n"] = len(frames)

        def run(self):
            ran["ran"] = True

    monkeypatch.setattr(lt, "LabelSession", FakeLabel)
    with mock.patch.object(sys, "argv",
                           ["lt", "--triplet", "data/Scene/ts", "--every", "5"]):
        main()
    assert ran == {"n": 1, "ran": True}


def test_main_frames_runs(monkeypatch, tmp_path):
    _mock_cv2_ui(monkeypatch)
    monkeypatch.setattr(lt, "MANUAL_PATH", tmp_path / "manual.jsonl")
    img = np.full((100, 100, 3), 70, dtype=np.uint8)
    frame = lt.Frame("Scene/dir/0", "Scene", None, None, img)
    monkeypatch.setattr(lt, "_iter_dir_frames", lambda d: [frame])

    ran = {}

    class FakeLabel:
        def __init__(self, frames, intr, det):
            ran["n"] = len(frames)

        def run(self):
            ran["ran"] = True

    monkeypatch.setattr(lt, "LabelSession", FakeLabel)
    with mock.patch.object(sys, "argv", ["lt", "--frames", str(tmp_path)]):
        main()
    assert ran == {"n": 1, "ran": True}
