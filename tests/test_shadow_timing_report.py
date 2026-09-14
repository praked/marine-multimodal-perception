import json

from scripts.eval.shadow_timing_report import main, summarise


def _rec(i, tick, extra=None, board=None):
    sh = {"tick_ms": tick, "scorer_ok": True, "source_latency_s": 0.1}
    if extra:
        sh.update(extra)
    if board:
        sh["board"] = board
    return {"timestamp": f"12:00:{i // 10:02d}.{i % 10}", "shadow": sh}


def test_summary_and_figure(tmp_path):
    tail = tmp_path / "tail.jsonl"
    tail.write_text("\n".join(json.dumps(_rec(i * 3, 60 + i, board={"clock_mhz": 1500 if i < 5 else 600, "temp_c": 70 + i, "throttled": "0x0" if i < 5 else "0x50005"})) for i in range(10)) + "\n")
    full = tmp_path / "full.jsonl"
    full.write_text("\n".join(json.dumps(_rec(i * 10, 900 + 10 * i, {"camera": True, "seg_fresh": i % 2 == 0, "yolo_fresh": True, "skipped_rows": i})) for i in range(6)) + "\n")
    s = summarise("tail", [json.loads(l) for l in tail.read_text().splitlines()])
    assert abs(s["rate_hz"] - 3.33) < 0.05 and s["tick_p50"] == 64.5 and s["throttled_any"] is True and s["clock_min"] == 600
    f = summarise("full", [json.loads(l) for l in full.read_text().splitlines()])
    assert f["camera"] == 1.0 and f["seg_fresh"] == 0.5 and f["skipped_rows"] == 5 and f["over_budget"] == 1.0 and f["clock_mhz"] is None
    out = tmp_path / "fig.png"; md = tmp_path / "t.md"
    assert main([f"tail={tail}", f"full={full}", "--out", str(out), "--md", str(md)]) == 0
    assert out.stat().st_size > 1000 and "| mode |" in md.read_text()
    assert main([str(tmp_path / "missing.jsonl")]) == 1
