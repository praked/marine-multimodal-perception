import json

from scripts.eval.closed_loop_table import main, summarise


def _trial(tmp_path, name, rows):
    p = tmp_path / f"sector_experiment_{name}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


BINS = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]


def test_summarise_reached_trial(tmp_path):
    rows = [
        {"t_s": 0.0, "phase": "INIT", "kind": "hardware", "dry_run": False},
        {"t_s": 20.1, "phase": "HOLD", "kind": "decide", "result": "blocked", "n": 60},
        {"t_s": 40.3, "phase": "HOLD", "kind": "decide", "result": "ok", "rel_heading_deg": -30.0,
         "field_p": 0.21, "n": 61},
        {"t_s": 40.4, "phase": "HOLD", "kind": "checkpoint", "abs_heading_deg": 300.0, "distance_m": 15.0},
        {"t_s": 40.5, "phase": "GO", "kind": "phase"},
        {"t_s": 45.0, "phase": "GO", "kind": "sectors", "bins": BINS, "p": [0.1, 0.2, 0.6, 0.1, 0.1, 0.1, 0.1],
         "min_range_m": [None, None, 5.5, None, None, None, None], "heading": 330.0},   # course rel = -30
        {"t_s": 62.0, "phase": "DONE", "kind": "phase", "why": "checkpoint reached", "distance_m": 8.9},
    ]
    p = _trial(tmp_path, "20260910_213000", rows)
    t = summarise(str(p), rows)
    assert t["armed"] is True and t["decisions"] == 2 and t["refused"] == 1
    assert t["t_decide_s"] == 40.4 and t["rel_heading_deg"] == -30.0 and t["abs_heading_deg"] == 300.0
    assert t["outcome"] == "reached" and t["go_s"] == 21.5
    # the -30 bin is on the course (rel -30), -15 is within 15 deg too: closest 5.5 m at -15, max p 0.6
    assert t["closest_m"] == 5.5 and t["closest_deg"] == -15.0 and t["max_p_course"] == 0.6


def test_cli_table_and_csv(tmp_path, capsys):
    _trial(tmp_path, "a", [{"t_s": 0.0, "phase": "INIT", "kind": "hardware", "dry_run": True},
                           {"t_s": 30.0, "phase": "DONE", "kind": "phase", "why": "abort: p 0.80 at course with radar 3.1 m"}])
    out_csv = tmp_path / "t.csv"
    assert main([str(tmp_path), "--csv", str(out_csv)]) == 0
    text = capsys.readouterr().out
    assert "| trial |" in text and "abort" in text and "1 trials" in text
    assert out_csv.read_text().splitlines()[1].startswith("a,,False,0,0")   # trial,name,armed,...
    assert main([str(tmp_path / "nothing")]) == 1


def test_multi_trial_log_splits_per_trial(tmp_path, capsys):
    rows = [{"t_s": 0.0, "phase": "INIT", "kind": "hardware", "dry_run": False, "trial": 0},
            {"t_s": 12.0, "phase": "HOLD", "kind": "decide", "result": "ok", "rel_heading_deg": -30.0, "field_p": 0.2, "n": 12, "trial": 1},
            {"t_s": 12.1, "phase": "HOLD", "kind": "checkpoint", "abs_heading_deg": 300.0, "distance_m": 15.0, "trial": 1},
            {"t_s": 30.0, "phase": "DONE", "kind": "phase", "why": "checkpoint reached", "trial": 1},
            {"t_s": 60.0, "phase": "HOLD", "kind": "decide", "result": "blocked", "n": 12, "trial": 2},
            {"t_s": 75.0, "phase": "DONE", "kind": "phase", "why": "abort: p 0.9 at course with radar 2.0 m", "trial": 2}]
    _trial(tmp_path, "w", rows)
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "w#1" in out and "w#2" in out and "2 trials" in out and "1 reached" in out and "1 aborted" in out


def test_multi_trial_with_station_keeping(tmp_path, capsys):
    """--trials 2 --station-keep-s: the leg ends at STATION, the trial at DONE.

    Trial 1 drives a figure-eight around the checkpoint for 120 s; trial 2 uses
    the `hold` pattern. GO time must be the leg only, not leg + station.
    """
    rows = [
        {"t_s": 0.0, "phase": "INIT", "kind": "start", "trial": 0,
         "name": "port30_unlit", "note": "obstacle boat 7 m at 30 port, dusk, wind 2 m/s NW",
         "args": {"station_keep_s": 120.0, "station_pattern": "eight"}},
        {"t_s": 0.1, "phase": "INIT", "kind": "hardware", "trial": 0, "dry_run": False,
         "station_keep_s": 120.0, "station_pattern": "eight", "station_radius_m": 6.0},
        # trial 1: decide -> checkpoint -> GO 20 s -> STATION 120 s
        {"t_s": 20.0, "phase": "HOLD", "kind": "decide", "trial": 1, "result": "ok",
         "rel_heading_deg": 30.0, "field_p": 0.18, "n": 14},
        {"t_s": 20.1, "phase": "HOLD", "kind": "checkpoint", "trial": 1,
         "abs_heading_deg": 90.0, "distance_m": 20.0},
        {"t_s": 20.2, "phase": "GO", "kind": "phase", "trial": 1},
        {"t_s": 40.2, "phase": "STATION", "kind": "phase", "trial": 1,
         "why": "checkpoint reached", "distance_m": 3.6, "keep_s": 120.0, "radius_m": 6.0},
        {"t_s": 40.3, "phase": "STATION", "kind": "station_pattern", "trial": 1,
         "pattern": "eight", "radius_m": 6.0},
        {"t_s": 90.0, "phase": "STATION", "kind": "station", "trial": 1,
         "pattern": "eight", "wp": 3, "dist_m": 2.1, "driving": True},
        {"t_s": 160.2, "phase": "DONE", "kind": "phase", "trial": 1,
         "why": "station keeping done", "keep_s": 120.0},
        # trial 2: same but the hold pattern, and the station rows carry no pattern
        {"t_s": 200.0, "phase": "HOLD", "kind": "decide", "trial": 2, "result": "ok",
         "rel_heading_deg": -15.0, "field_p": 0.22, "n": 12},
        {"t_s": 200.1, "phase": "HOLD", "kind": "checkpoint", "trial": 2,
         "abs_heading_deg": 300.0, "distance_m": 20.0},
        {"t_s": 200.2, "phase": "GO", "kind": "phase", "trial": 2},
        {"t_s": 215.2, "phase": "STATION", "kind": "phase", "trial": 2,
         "why": "checkpoint reached", "keep_s": 60.0, "radius_m": 6.0},
        {"t_s": 240.0, "phase": "STATION", "kind": "station", "trial": 2,
         "dist_m": 4.4, "driving": False},
        {"t_s": 275.2, "phase": "DONE", "kind": "phase", "trial": 2, "why": "station keeping done"},
    ]
    p = _trial(tmp_path, "20260908_190000_port30_unlit", rows)

    t1 = summarise(str(p), [r for r in rows if r["trial"] in (0, 1)])
    assert t1["name"] == "port30_unlit" and t1["note"].startswith("obstacle boat 7 m")
    assert t1["outcome"] == "reached"                       # the station DONE still counts as reached
    assert t1["go_s"] == 20.0                               # leg only: GO -> STATION, not GO -> DONE
    assert t1["station_s"] == 120.0 and t1["station_pattern"] == "eight"

    t2 = summarise(str(p), [r for r in rows if r["trial"] in (0, 2)])
    assert t2["go_s"] == 15.0 and t2["station_s"] == 60.0 and t2["station_pattern"] == "hold"

    out_csv = tmp_path / "station.csv"
    assert main([str(tmp_path), "--csv", str(out_csv)]) == 0
    text = capsys.readouterr().out
    assert "| station (s) |" in text and "port30_unlit" in text
    assert "#1" in text and "#2" in text and "2 trials" in text and "2 reached" in text
    # the trial-0 start/hardware rows describe the launch: every trial row carries
    # the name and the armed flag, not only the first
    assert text.count("| port30_unlit | yes |") == 2
    assert "2 held station, 180.0 s total" in text
    # the markdown truncates the note, the CSV keeps it whole
    assert "obstacle boat 7 m at 30 port, dusk, win…" in text
    body = out_csv.read_text().splitlines()
    assert body[0].startswith("trial,name,armed")
    assert "obstacle boat 7 m at 30 port, dusk, wind 2 m/s NW" in body[1]
