import json
import types
from unittest import mock

import numpy as np
import pytest

from scripts.eval.qwen_labeler import (
    _extract_json,
    _frame_hash,
    _frame_to_jpeg_bytes,
    _qwen_to_record,
    call_with_retry,
    estimate_cost_usd,
)


def test_frame_to_jpeg_bytes_compresses():
    img = np.full((200, 200, 3), 200, dtype=np.uint8)
    out = _frame_to_jpeg_bytes(img)
    assert len(out) < 200 * 200 * 3
    assert out[:3] == b"\xff\xd8\xff"  # JPEG magic


def test_frame_to_jpeg_bytes_resizes_large():
    img = np.full((4000, 4000, 3), 100, dtype=np.uint8)
    out = _frame_to_jpeg_bytes(img, max_long_side=512)
    assert len(out) > 0


def test_frame_hash_deterministic():
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    j = _frame_to_jpeg_bytes(img)
    assert _frame_hash(j, "m", "p") == _frame_hash(j, "m", "p")


def test_frame_hash_changes_with_model():
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    j = _frame_to_jpeg_bytes(img)
    assert _frame_hash(j, "m1", "p") != _frame_hash(j, "m2", "p")


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def test_extract_json_plain():
    txt = '{"obstacles": []}'
    assert _extract_json(txt) == {"obstacles": []}


def test_extract_json_code_fenced():
    txt = "```json\n{\"obstacles\": [{\"class\": \"boat\"}]}\n```"
    assert _extract_json(txt)["obstacles"][0]["class"] == "boat"


def test_extract_json_chatty_prefix():
    txt = "Here is the JSON you requested:\n{\"obstacles\": []}"
    assert _extract_json(txt) == {"obstacles": []}


def test_extract_json_no_brace_raises():
    with pytest.raises(ValueError):
        _extract_json("nothing here")


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def test_qwen_to_record_normalises_bbox(intrinsics_real, detection_real):
    parsed = {"obstacles": [
        {"class": "boat", "bbox_xyxy_normalised": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9},
    ]}
    rec = _qwen_to_record("Boats/ts/0", "Boats", (648, 864, 3), parsed, "m",
                          detection_real["fusion"], intrinsics_real["fisheye"])
    assert rec["fisheye_bboxes"][0]["xyxy"] == [0.1, 0.2, 0.3, 0.4]
    assert rec["audited"] is False
    assert rec["source"] == "qwen-vl"


def test_qwen_to_record_handles_pixel_bbox(intrinsics_real, detection_real):
    parsed = {"obstacles": [
        {"class": "boat", "bbox_xyxy": [100, 200, 300, 400], "confidence": 0.5},
    ]}
    rec = _qwen_to_record("Boats/ts/0", "Boats", (648, 864, 3), parsed, "m",
                          detection_real["fusion"], intrinsics_real["fisheye"])
    # 100/864 ~= 0.116
    assert rec["fisheye_bboxes"][0]["xyxy"][0] < 0.5


def test_qwen_to_record_skips_malformed(intrinsics_real, detection_real):
    parsed = {"obstacles": [
        {"class": "boat"},  # no bbox
        {"class": "duck", "bbox": [1, 2, 3]},  # wrong length
    ]}
    rec = _qwen_to_record("Boats/ts/0", "Boats", (648, 864, 3), parsed, "m",
                          detection_real["fusion"], intrinsics_real["fisheye"])
    assert rec["fisheye_bboxes"] == []


def test_qwen_to_record_derives_bins(intrinsics_real, detection_real):
    # bbox centred in image -> bin around 0
    parsed = {"obstacles": [
        {"class": "boat", "bbox_xyxy_normalised": [0.49, 0.5, 0.51, 0.52], "confidence": 0.9},
    ]}
    rec = _qwen_to_record("Boats/ts/0", "Boats", (648, 864, 3), parsed, "m",
                          detection_real["fusion"], intrinsics_real["fisheye"])
    # Centre of image is at cx_px = 472 (offset from 432 actual centre by +40 px).
    assert isinstance(rec["obstacle_bins_fisheye"], list)


# ---------------------------------------------------------------------------
# call_with_retry + cost
# ---------------------------------------------------------------------------

def test_call_with_retry_succeeds_first_attempt():
    calls = []
    def fn(b, m):
        calls.append(1)
        return "ok"
    assert call_with_retry(fn, b"x", "m", max_attempts=3, base_delay=0) == "ok"
    assert len(calls) == 1


def test_call_with_retry_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []
    def fn(b, m):
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("transient")
        return "ok"
    assert call_with_retry(fn, b"x", "m", max_attempts=3, base_delay=0) == "ok"
    assert len(calls) == 2


def test_call_with_retry_gives_up(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    def fn(b, m):
        raise RuntimeError("always fails")
    with pytest.raises(RuntimeError, match="backend failed"):
        call_with_retry(fn, b"x", "m", max_attempts=2, base_delay=0)


def test_call_with_retry_passes_systemexit(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    def fn(b, m):
        raise SystemExit("die")
    with pytest.raises(SystemExit):
        call_with_retry(fn, b"x", "m", max_attempts=3, base_delay=0)


def test_estimate_cost_with_price():
    assert estimate_cost_usd(10, "m", 0.005) == pytest.approx(0.05)


def test_estimate_cost_without_price():
    assert estimate_cost_usd(10, "m", None) == 0.0


# ---------------------------------------------------------------------------
# iter_fisheye_frames + _process_frame + main with mocked backend
# ---------------------------------------------------------------------------

import sys as _sys

import scripts.eval.qwen_labeler as ql


def test_iter_fisheye_frames(boats_triplet, intrinsics_real):
    frames = list(ql.iter_fisheye_frames(
        str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        every=120, intrinsics=intrinsics_real,
    ))
    assert len(frames) > 0
    fid, scene, img = frames[0]
    assert scene == "Boats"
    assert fid.startswith("Boats/")


def test_iter_fisheye_frames_unreadable(tmp_path, intrinsics_real):
    fake_f = tmp_path / "fisheye_x.mp4"; fake_f.write_bytes(b"x")
    fake_t = tmp_path / "thermal_x.mp4"; fake_t.write_bytes(b"x")
    fake_m = tmp_path / "mmwave_x.csv"; fake_m.write_text("Date,Time,X,Y,Z\n")
    with pytest.raises(SystemExit):
        list(ql.iter_fisheye_frames(str(tmp_path / "x"), every=1, intrinsics=intrinsics_real))


def test_process_frame_cache_hit(monkeypatch, tmp_path,
                                  intrinsics_real, detection_real):
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ql, "LABELS_DIR", tmp_path)
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    item = ("Boats/ts/0", "Boats", img)
    jpeg = ql._frame_to_jpeg_bytes(img)
    h = ql._frame_hash(jpeg, "test-model", ql.PROMPT)
    (tmp_path / f"{h}.json").write_text(json.dumps({
        "frame_id": "Boats/ts/0", "model": "test-model",
        "raw": json.dumps({"obstacles": []}),
    }))
    def backend(b, m):
        raise AssertionError("should not call backend when cached")
    fid, rec, err = ql._process_frame(item, backend, "test-model", 1, False,
                                       intrinsics_real["fisheye"],
                                       detection_real["fusion"])
    assert err is None
    assert rec["frame_id"] == "Boats/ts/0"
    assert rec["_from_cache"] is True


def test_process_frame_calls_backend(monkeypatch, tmp_path,
                                      intrinsics_real, detection_real):
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path)
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    item = ("Boats/ts/0", "Boats", img)
    def backend(b, m):
        return json.dumps({"obstacles": [
            {"class": "boat", "bbox_xyxy_normalised": [0.4, 0.5, 0.5, 0.6], "confidence": 0.9},
        ]})
    fid, rec, err = ql._process_frame(item, backend, "test-model", 1, False,
                                       intrinsics_real["fisheye"],
                                       detection_real["fusion"])
    assert err is None
    assert len(rec["fisheye_bboxes"]) == 1
    # Cache file should now exist.
    h = ql._frame_hash(ql._frame_to_jpeg_bytes(img), "test-model", ql.PROMPT)
    assert (tmp_path / f"{h}.json").exists()


def test_process_frame_dry_run(monkeypatch, tmp_path,
                                intrinsics_real, detection_real):
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path)
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    item = ("Boats/ts/0", "Boats", img)
    fid, rec, err = ql._process_frame(item, None, "m", 1, True,
                                       intrinsics_real["fisheye"],
                                       detection_real["fusion"])
    assert err == "dry-run"
    assert rec is None


def test_process_frame_bad_json(monkeypatch, tmp_path,
                                 intrinsics_real, detection_real):
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path)
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    item = ("Boats/ts/0", "Boats", img)
    def backend(b, m):
        return "no json here"
    fid, rec, err = ql._process_frame(item, backend, "m", 1, False,
                                       intrinsics_real["fisheye"],
                                       detection_real["fusion"])
    assert "parse failed" in err
    assert rec is None


def test_process_frame_backend_exception(monkeypatch, tmp_path,
                                          intrinsics_real, detection_real):
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path)
    monkeypatch.setattr("time.sleep", lambda s: None)
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    item = ("Boats/ts/0", "Boats", img)
    def backend(b, m):
        raise RuntimeError("api down")
    fid, rec, err = ql._process_frame(item, backend, "m", 2, False,
                                       intrinsics_real["fisheye"],
                                       detection_real["fusion"])
    assert err and "api down" in err
    assert rec is None


def test_main_dry_run(monkeypatch, boats_triplet, tmp_path):
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ql, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(ql, "QWEN_PATH", tmp_path / "qwen.jsonl")
    with mock.patch.object(_sys, "argv", [
        "qwen", "--triplet", str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--every", "120", "--max-frames", "2", "--dry-run",
    ]):
        ql.main()
    # No labels written because dry-run.
    out = tmp_path / "qwen.jsonl"
    # File may exist but should be empty.
    if out.exists():
        assert out.read_text() == ""


def test_main_with_mocked_api(monkeypatch, boats_triplet, tmp_path):
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ql, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(ql, "QWEN_PATH", tmp_path / "qwen.jsonl")
    monkeypatch.setattr(ql, "call_api",
                        lambda b, m: json.dumps({"obstacles": []}))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with mock.patch.object(_sys, "argv", [
        "qwen", "--triplet", str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--every", "120", "--max-frames", "2", "--backend", "api",
        "--model", "fake-model",
    ]):
        ql.main()
    out = tmp_path / "qwen.jsonl"
    assert out.exists()
    lines = out.read_text().splitlines()
    assert len(lines) <= 2


def test_main_parallel(monkeypatch, boats_triplet, tmp_path):
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ql, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(ql, "QWEN_PATH", tmp_path / "qwen.jsonl")
    monkeypatch.setattr(ql, "call_api",
                        lambda b, m: json.dumps({"obstacles": []}))
    with mock.patch.object(_sys, "argv", [
        "qwen", "--triplet", str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--every", "120", "--max-frames", "3", "--workers", "2",
        "--backend", "api",
    ]):
        ql.main()
    assert (tmp_path / "qwen.jsonl").exists()


def test_main_transformers_workers_capped(monkeypatch, boats_triplet, tmp_path, capsys):
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ql, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(ql, "QWEN_PATH", tmp_path / "qwen.jsonl")
    monkeypatch.setattr(ql, "call_transformers",
                        lambda b, m: json.dumps({"obstacles": []}))
    with mock.patch.object(_sys, "argv", [
        "qwen", "--triplet", str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--every", "120", "--max-frames", "1", "--workers", "4",
        "--backend", "transformers",
    ]):
        ql.main()
    msg = capsys.readouterr().out
    assert "ignored" in msg


def test_main_cost_cap_triggers(monkeypatch, boats_triplet, tmp_path):
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ql, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(ql, "QWEN_PATH", tmp_path / "qwen.jsonl")
    monkeypatch.setattr(ql, "call_api",
                        lambda b, m: json.dumps({"obstacles": []}))
    with mock.patch.object(_sys, "argv", [
        "qwen", "--triplet", str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--every", "60", "--backend", "api",
        "--price-per-call-usd", "0.5", "--cost-cap-usd", "0.6",
    ]):
        with pytest.raises(SystemExit, match="cost cap"):
            ql.main()


# ---------------------------------------------------------------------------
# jpeg encode failure (line 106)
# ---------------------------------------------------------------------------

def test_frame_to_jpeg_bytes_encode_failure(monkeypatch):
    monkeypatch.setattr(ql.cv2, "imencode", lambda *a, **k: (False, None))
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="jpeg encode failed"):
        ql._frame_to_jpeg_bytes(img)


# ---------------------------------------------------------------------------
# call_api backend (lines 124-143): fake openai module, no network
# ---------------------------------------------------------------------------

def test_call_api_builds_request_and_returns_content(monkeypatch):
    captured = {}

    class FakeMessage:
        content = '{"obstacles": []}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResp:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, *a, **k):
            pass
        chat = FakeChat()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(_sys.modules, "openai", fake_openai)

    out = ql.call_api(b"\xff\xd8jpeg", "fake-model")
    assert out == '{"obstacles": []}'
    assert captured["model"] == "fake-model"
    # The image is base64-data-url embedded in the message content.
    content = captured["messages"][0]["content"]
    assert any(part.get("type") == "image_url" for part in content)


def test_call_api_content_none_returns_empty_string(monkeypatch):
    class FakeMessage:
        content = None

    class FakeChoice:
        message = FakeMessage()

    class FakeResp:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, *a, **k):
            pass
        chat = FakeChat()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(_sys.modules, "openai", fake_openai)
    assert ql.call_api(b"x", "m") == ""


def test_call_api_missing_openai_raises_systemexit(monkeypatch):
    # Simulate `import openai` failing.
    monkeypatch.setitem(_sys.modules, "openai", None)
    with pytest.raises(SystemExit, match="pip install openai"):
        ql.call_api(b"x", "m")


# ---------------------------------------------------------------------------
# call_transformers backend (lines 153-176): fake torch/PIL/transformers
# ---------------------------------------------------------------------------

def test_call_transformers_lazy_loads_and_decodes(monkeypatch):
    # Reset module-level cache so the load branch runs.
    monkeypatch.setattr(ql, "_TF_MODEL", None)
    monkeypatch.setattr(ql, "_TF_PROCESSOR", None)

    class FakeInputs(dict):
        # Must behave like a mapping for `**inputs` AND expose `.input_ids`
        # and `.to(device)` like a transformers BatchEncoding.
        input_ids = np.zeros((1, 3), dtype=int)
        def to(self, device):
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "prompt-text"
        def __call__(self, text, images, return_tensors):
            return FakeInputs()
        def batch_decode(self, seqs, skip_special_tokens):
            return ['{"obstacles": []}']

    class FakeModel:
        device = "cpu"
        def generate(self, **kwargs):
            # shape (1, 5): more than the 3 input ids so the slice is non-empty
            return np.zeros((1, 5), dtype=int)

    fake_torch = types.ModuleType("torch")

    fake_pil = types.ModuleType("PIL")
    fake_pil_image = types.ModuleType("PIL.Image")
    class FakeImage:
        @staticmethod
        def open(buf):
            return FakeImage()
        def convert(self, mode):
            return self
    fake_pil_image.open = FakeImage.open
    fake_pil.Image = fake_pil_image

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda m: FakeProcessor())
    fake_tf.AutoModelForVision2Seq = types.SimpleNamespace(
        from_pretrained=lambda m, **k: FakeModel())

    monkeypatch.setitem(_sys.modules, "torch", fake_torch)
    monkeypatch.setitem(_sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(_sys.modules, "PIL.Image", fake_pil_image)
    monkeypatch.setitem(_sys.modules, "transformers", fake_tf)

    jpeg = ql._frame_to_jpeg_bytes(np.zeros((20, 20, 3), dtype=np.uint8))
    out = ql.call_transformers(jpeg, "fake/Qwen")
    assert out == '{"obstacles": []}'
    # Second call reuses the cached model (no re-load); still returns.
    out2 = ql.call_transformers(jpeg, "fake/Qwen")
    assert out2 == '{"obstacles": []}'


def test_call_transformers_missing_deps_raises_systemexit(monkeypatch):
    monkeypatch.setattr(ql, "_TF_MODEL", None)
    monkeypatch.setattr(ql, "_TF_PROCESSOR", None)
    monkeypatch.setitem(_sys.modules, "torch", None)
    with pytest.raises(SystemExit, match="pip install transformers"):
        ql.call_transformers(b"x", "m")


# ---------------------------------------------------------------------------
# main() failure-print + max-calls breaks + cost summary
# ---------------------------------------------------------------------------

def test_main_prints_failure(monkeypatch, boats_triplet, tmp_path, capsys):
    """A backend that always errors drives the `!! frame_id: err` print and
    the n_failed counter (lines 355-356)."""
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ql, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(ql, "QWEN_PATH", tmp_path / "qwen.jsonl")
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(ql, "call_api",
                        lambda b, m: (_ for _ in ()).throw(RuntimeError("api boom")))
    with mock.patch.object(_sys, "argv", [
        "qwen", "--triplet", str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--every", "120", "--max-frames", "1", "--backend", "api", "--retries", "1",
    ]):
        ql.main()
    out = capsys.readouterr().out
    assert "!!" in out
    assert "api boom" in out
    assert "failed=1" in out


def test_main_serial_max_calls_break(monkeypatch, boats_triplet, tmp_path, capsys):
    """Serial mode stops once n_calls hits --max-calls (line 394)."""
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ql, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(ql, "QWEN_PATH", tmp_path / "qwen.jsonl")
    monkeypatch.setattr(ql, "call_api",
                        lambda b, m: json.dumps({"obstacles": []}))
    with mock.patch.object(_sys, "argv", [
        "qwen", "--triplet", str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--every", "30", "--backend", "api", "--max-calls", "1",
    ]):
        ql.main()
    out = capsys.readouterr().out
    # Only one actual call was made before the break.
    assert "calls=1" in out


def test_main_cost_summary_printed(monkeypatch, boats_triplet, tmp_path, capsys):
    """price-per-call without a cost cap prints the estimated-cost summary
    (lines 404-405)."""
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ql, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(ql, "QWEN_PATH", tmp_path / "qwen.jsonl")
    monkeypatch.setattr(ql, "call_api",
                        lambda b, m: json.dumps({"obstacles": []}))
    with mock.patch.object(_sys, "argv", [
        "qwen", "--triplet", str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--every", "120", "--max-frames", "1", "--backend", "api",
        "--price-per-call-usd", "0.01",
    ]):
        ql.main()
    out = capsys.readouterr().out
    assert "estimated cost:" in out


def test_main_parallel_max_calls_break(monkeypatch, boats_triplet, tmp_path, capsys):
    """Parallel mode honours --max-calls both at submit time (line 379) and
    after consuming a result (lines 388-390)."""
    monkeypatch.setattr(ql, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ql, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(ql, "QWEN_PATH", tmp_path / "qwen.jsonl")
    monkeypatch.setattr(ql, "call_api",
                        lambda b, m: json.dumps({"obstacles": []}))
    with mock.patch.object(_sys, "argv", [
        "qwen", "--triplet", str(boats_triplet.fisheye.parent / boats_triplet.timestamp),
        "--every", "20", "--workers", "2", "--backend", "api", "--max-calls", "1",
    ]):
        ql.main()
    out = capsys.readouterr().out
    assert "calls=1" in out
