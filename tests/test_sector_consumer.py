import json
import sys
from unittest import mock

import pytest

from scripts.eval.sector_consumer import consume, decide, format_line, main


def _bins(n=11):
    return list(range(-50, 60, 10))


def _record(scores, ranges=None, ts="00:00:00.0", tracked=False):
    if ranges is None:
        ranges = [None] * len(scores)
    return {
        "timestamp": ts,
        "clip_id": "Test/x",
        "bin_centers_deg": _bins(),
        "scores": scores,
        "min_range_m": ranges,
        "sensor_hits": [[0, 0, 0]] * len(scores),
        "tracked": tracked,
    }


def test_decide_no_blocked_returns_empty():
    rec = _record([0.0] * 11)
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    assert d["blocked"] == []
    assert d["alert"] == []
    assert d["closest"] is None
    assert d["hard_stop"] is False


def test_decide_blocked_above_threshold():
    scores = [0.0] * 11
    scores[5] = 1.0
    rec = _record(scores)
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    assert d["blocked"] == [0]


def test_decide_alert_band():
    scores = [0.0] * 11
    scores[5] = 0.5    # alert (above 0.33, below 0.66)
    scores[6] = 0.8    # blocked
    rec = _record(scores)
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    assert d["alert"] == [0]
    assert d["blocked"] == [10]


def test_decide_closest_range():
    scores = [0.0] * 11
    scores[5] = 0.5
    scores[6] = 0.5
    ranges = [None] * 11
    ranges[5] = 3.5
    ranges[6] = 1.8
    rec = _record(scores, ranges=ranges)
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    assert d["closest"] == 1.8
    assert d["closest_bin"] == 10


def test_decide_hard_stop_triggers():
    scores = [0.0] * 11
    scores[5] = 1.0
    ranges = [None] * 11
    ranges[5] = 1.5
    rec = _record(scores, ranges=ranges)
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=2.0)
    assert d["hard_stop"] is True


def test_decide_hard_stop_disabled_when_threshold_none():
    scores = [0.0] * 11
    scores[5] = 1.0
    ranges = [None] * 11
    ranges[5] = 1.5
    rec = _record(scores, ranges=ranges)
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    assert d["hard_stop"] is False


def test_decide_propagates_tracked_flag():
    rec = _record([0.0] * 11, tracked=True)
    assert decide(rec, 0.66, None)["tracked"] is True


def test_format_line_minimal():
    d = decide(_record([0.0] * 11), 0.66, None)
    assert format_line(d) == "00:00:00.0"


def test_format_line_has_blocked():
    scores = [0.0] * 11
    scores[5] = 1.0
    d = decide(_record(scores), 0.66, None)
    out = format_line(d)
    assert "blocked=[0]" in out


def test_format_line_has_hard_stop():
    scores = [0.0] * 11
    scores[5] = 1.0
    ranges = [None] * 11
    ranges[5] = 1.0
    d = decide(_record(scores, ranges=ranges), 0.66, hard_stop_m=2.0)
    assert "HARD_STOP" in format_line(d)


def test_consume_counts(tmp_path):
    path = tmp_path / "sectors.jsonl"
    lines = []
    for i in range(5):
        scores = [0.0] * 11
        if i % 2 == 0:
            scores[5] = 1.0   # blocked frames 0, 2, 4
        ranges = [None] * 11
        if i == 0:
            ranges[5] = 1.0   # hard-stop frame 0
        lines.append(json.dumps(_record(scores, ranges=ranges, ts=f"00:00:0{i}.0")))
    path.write_text("\n".join(lines) + "\n")
    stats = consume(path, blocked_threshold=0.66, hard_stop_m=2.0, quiet=True)
    assert stats["n_frames"] == 5
    assert stats["n_blocked"] == 3
    assert stats["n_hard_stop"] == 1


def test_consume_handles_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text("\n\n" + json.dumps(_record([0.0] * 11)) + "\n\n")
    stats = consume(path, 0.66, None, quiet=True)
    assert stats["n_frames"] == 1


def test_consume_prints_per_frame_when_loud(tmp_path, capsys):
    path = tmp_path / "x.jsonl"
    scores = [0.0] * 11
    scores[5] = 1.0
    path.write_text(json.dumps(_record(scores)) + "\n")
    consume(path, blocked_threshold=0.66, hard_stop_m=None, quiet=False)
    out = capsys.readouterr().out
    assert "blocked=[0]" in out


def test_main_missing_file(tmp_path):
    with mock.patch.object(sys, "argv", ["sc", "--input", str(tmp_path / "nope")]):
        with pytest.raises(SystemExit):
            main()


# ---------------------------------------------------------------------------
# v1 (motion + heading) coverage
# ---------------------------------------------------------------------------

def _v1_record(scores, ranges=None, velocities=None, ttcs=None,
               heading=None, reason=None, ts="00:00:00.0"):
    if ranges is None:
        ranges = [None] * len(scores)
    if velocities is None:
        velocities = [None] * len(scores)
    if ttcs is None:
        ttcs = [None] * len(scores)
    return {
        "protocol": 1,
        "timestamp": ts,
        "clip_id": "Test/x",
        "bin_centers_deg": _bins(),
        "scores": scores,
        "min_range_m": ranges,
        "sensor_hits": [[0, 0, 0]] * len(scores),
        "tracked": False,
        "per_bin_velocity_mps": velocities,
        "per_bin_ttc_s": ttcs,
        "recommended_heading_deg": heading,
        "heading_reason": reason,
    }


def test_decide_surfaces_v1_velocity_and_ttc():
    scores = [0.0] * 11
    scores[5] = 1.0
    velocities = [None] * 11
    velocities[5] = 1.5
    ttcs = [None] * 11
    ttcs[5] = 2.0
    d = decide(_v1_record(scores, velocities=velocities, ttcs=ttcs,
                          heading=-20.0, reason="ttc"),
               blocked_threshold=0.66, hard_stop_m=None)
    assert d["closest_ttc"] == 2.0
    assert d["closest_ttc_bin"] == 0
    assert d["fastest_closing"] == 1.5
    assert d["fastest_bin"] == 0
    assert d["recommended_heading_deg"] == -20.0
    assert d["heading_reason"] == "ttc"


def test_decide_v0_record_has_null_v1_fields():
    """A v0 record (no protocol field) decided cleanly returns
    null/None for the v1 fields without crashing."""
    rec = _record([0.0] * 11)
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    assert d["recommended_heading_deg"] is None
    assert d["closest_ttc"] is None
    assert d["protocol"] == 0


def test_format_line_includes_heading_when_present():
    scores = [0.0] * 11
    scores[6] = 1.0
    d = decide(_v1_record(scores, heading=-15.0, reason="ttc"),
               blocked_threshold=0.66, hard_stop_m=None)
    out = format_line(d)
    assert "head=-15°(ttc)" in out


def test_format_line_shows_question_mark_when_ambiguous():
    d = decide(_v1_record([0.0] * 11, heading=None, reason="ambiguous"),
               blocked_threshold=0.66, hard_stop_m=None)
    # Empty input → no blocked text, but heading reporting flags ambiguous.
    out = format_line(d)
    assert "head=?" in out


def test_consume_counts_heading_recommendations(tmp_path):
    path = tmp_path / "v1.jsonl"
    lines = [
        json.dumps(_v1_record([0.0] * 11, heading=10.0, reason="course",
                              ts="00:00:00.0")),
        json.dumps(_v1_record([0.0] * 11, heading=None, reason="ambiguous",
                              ts="00:00:00.1")),
        json.dumps(_v1_record([0.0] * 11, heading=15.0, reason="course",
                              ts="00:00:00.2")),
    ]
    path.write_text("\n".join(lines) + "\n")
    stats = consume(path, 0.66, None, quiet=True)
    assert stats["n_heading"] == 2
    assert stats["n_heading_ambiguous"] == 1


def test_format_line_includes_alert_ttc_velocity_steer():
    """A blocked frame with alert bins, TTC, velocity and a smoothed
    heading exercises the alert/ttc/v/steer format branches (lines
    97, 101, 103, 111)."""
    scores = [0.0] * 11
    scores[5] = 1.0    # blocked at 0°
    scores[6] = 0.5    # alert at +10°
    velocities = [None] * 11
    velocities[5] = 2.0
    ttcs = [None] * 11
    ttcs[5] = 3.0
    rec = _v1_record(scores, velocities=velocities, ttcs=ttcs,
                     heading=-10.0, reason="ttc")
    rec["smoothed_heading_deg"] = -8.0
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    out = format_line(d)
    assert "alert=" in out
    assert "ttc=" in out
    assert "v=" in out
    assert "steer=" in out


def test_consume_counts_alert_frames(tmp_path):
    """A frame whose only firing is in the alert band increments
    n_alert (line 136)."""
    path = tmp_path / "alert.jsonl"
    scores = [0.0] * 11
    scores[5] = 0.5    # alert only, not blocked
    path.write_text(json.dumps(_record(scores)) + "\n")
    stats = consume(path, blocked_threshold=0.66, hard_stop_m=None, quiet=True)
    assert stats["n_alert"] == 1
    assert stats["n_blocked"] == 0


def test_main_prints_heading_summary(tmp_path, capsys):
    """A v1 stream with heading recommendations triggers the heading
    summary block in main() (lines 182-185)."""
    path = tmp_path / "v1.jsonl"
    lines = [
        json.dumps(_v1_record([0.0] * 11, heading=10.0, reason="course",
                              ts="00:00:00.0")),
        json.dumps(_v1_record([0.0] * 11, heading=None, reason="ambiguous",
                              ts="00:00:00.1")),
    ]
    path.write_text("\n".join(lines) + "\n")
    with mock.patch.object(sys, "argv", ["sc", "--input", str(path), "--quiet"]):
        main()
    out = capsys.readouterr().out
    assert "heading frames" in out
    assert "recommended" in out


# ---------------------------------------------------------------------------
# v1.1 extensions: attitude/compass, confirmed bins, free-space
# ---------------------------------------------------------------------------

def test_absolute_heading_composes_and_wraps():
    from scripts.eval.sector_consumer import absolute_heading
    d = {"attitude": {"yaw_deg": 170.0}, "smoothed_heading_deg": 20.0}
    # 170 + 20 = 190 -> wraps to -170.
    assert absolute_heading(d) == pytest.approx(-170.0)
    # yaw_sign flips the steer: 170 - 20 = 150 (no wrap).
    assert absolute_heading(d, yaw_sign=-1.0) == pytest.approx(150.0)


def test_absolute_heading_none_cases():
    from scripts.eval.sector_consumer import absolute_heading
    assert absolute_heading({"attitude": None,
                             "smoothed_heading_deg": 5.0}) is None
    assert absolute_heading({"attitude": {"yaw_deg": 10.0},
                             "smoothed_heading_deg": None}) is None
    assert absolute_heading({"attitude": {"yaw_deg": None},
                             "smoothed_heading_deg": 5.0}) is None


def test_format_line_includes_compass_when_attitude_present():
    scores = [0.0] * 11
    scores[5] = 1.0
    rec = _v1_record(scores)
    rec["attitude"] = {"roll_deg": 1.0, "pitch_deg": 2.0, "yaw_deg": 90.0}
    rec["smoothed_heading_deg"] = 10.0
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    out = format_line(d)
    assert "compass=+100°" in out


def test_format_line_confirmed_blocked_bins():
    scores = [0.0] * 11
    scores[5] = 1.0     # blocked at 0°
    scores[6] = 1.0     # blocked at +10°
    rec = _record(scores)
    conf = [0] * 11
    conf[5] = 1          # only the 0° bin is radar-confirmed
    rec["confirmed"] = conf
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    out = format_line(d)
    assert "confirmed=[0]" in out


def test_format_line_confirmed_only_on_unblocked_bin_omitted():
    scores = [0.0] * 11
    scores[5] = 1.0      # blocked at 0°
    rec = _record(scores)
    conf = [0] * 11
    conf[7] = 1          # confirmation on a NON-blocked bin
    rec["confirmed"] = conf
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    assert "confirmed=" not in format_line(d)


def test_format_line_free_space_min():
    scores = [0.0] * 11
    scores[5] = 1.0
    rec = _record(scores)
    rec["free_space_m"] = [None, 12.0, 4.5] + [None] * 8
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    assert "freespace_min=4.5m" in format_line(d)


def test_format_line_free_space_all_null_omitted():
    scores = [0.0] * 11
    scores[5] = 1.0
    rec = _record(scores)
    rec["free_space_m"] = [None] * 11
    d = decide(rec, blocked_threshold=0.66, hard_stop_m=None)
    assert "freespace_min" not in format_line(d)


def test_main_summary(tmp_path, capsys):
    path = tmp_path / "x.jsonl"
    scores = [0.0] * 11
    scores[5] = 1.0
    ranges = [None] * 11
    ranges[5] = 1.0
    path.write_text(json.dumps(_record(scores, ranges=ranges)) + "\n")
    with mock.patch.object(sys, "argv", ["sc", "--input", str(path),
                                          "--blocked", "0.5",
                                          "--hard-stop-m", "2.0",
                                          "--quiet"]):
        main()
    out = capsys.readouterr().out
    assert "frames" in out
    assert "hard-stop frames" in out
