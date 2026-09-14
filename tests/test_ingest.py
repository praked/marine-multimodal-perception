import json
import sys
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

from scripts.data.ingest import (
    _find_triplets_in_source,
    _hash_file,
    _validate,
    ingest_one,
    main,
)


def _write_triplet(dir_: Path, ts: str, valid_csv: bool = True,
                   add_frames: bool = True):
    fish = dir_ / f"fisheye_{ts}.mp4"
    therm = dir_ / f"thermal_{ts}.mp4"
    mm = dir_ / f"mmwave_{ts}.csv"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(fish), fourcc, 3.0, (40, 30))
    tw = cv2.VideoWriter(str(therm), fourcc, 3.0, (40, 30))
    if add_frames:
        for i in range(3):
            f = np.full((30, 40, 3), 80, dtype=np.uint8)
            fw.write(f); tw.write(f)
    fw.release(); tw.release()
    if valid_csv:
        mm.write_text("Date,Time,X,Y,Z\n2099-01-01,00:00:00.0,0,1,0\n")
    else:
        mm.write_text("A,B,C\n1,2,3\n")
    return fish, therm, mm


def test_find_triplets_in_source(tmp_path):
    _write_triplet(tmp_path, "2099-01-01_00-00-00")
    _write_triplet(tmp_path, "2099-01-01_00-00-01")
    out = _find_triplets_in_source(tmp_path)
    assert len(out) == 2


def test_find_triplets_includes_fisheye_only(tmp_path):
    """A lone fisheye is a legitimate fisheye-only capture (the service can be
    run with radar + thermal disabled), so discovery must NOT skip it: the
    previous 'all three or nothing' rule dropped such captures silently."""
    (tmp_path / "fisheye_ts.mp4").write_bytes(b"x")
    assert _find_triplets_in_source(tmp_path) == ["ts"]


def test_find_triplets_ignores_orphan_thermal(tmp_path):
    """Without a fisheye there is no capture to key on."""
    (tmp_path / "thermal_ts.mp4").write_bytes(b"x")
    (tmp_path / "mmwave_ts.csv").write_text("Date,Time,X,Y,Z\n")
    assert _find_triplets_in_source(tmp_path) == []


def test_hash_file_returns_hash(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    assert len(_hash_file(p)) == 16


def test_hash_file_missing_returns_none(tmp_path):
    assert _hash_file(tmp_path / "absent") is None


def test_validate_ok(tmp_path):
    _write_triplet(tmp_path, "ts1")
    ok, report = _validate(tmp_path, "ts1")
    assert ok
    assert report["files"]["fisheye"]["ok"] is True


def test_validate_missing_csv_columns(tmp_path):
    _write_triplet(tmp_path, "ts2", valid_csv=False)
    ok, report = _validate(tmp_path, "ts2")
    assert not ok
    assert "error" in report["files"]["mmwave"]


def test_validate_unreadable_video(tmp_path):
    # Write text files instead of mp4 contents.
    for n in ("fisheye_x.mp4", "thermal_x.mp4"):
        (tmp_path / n).write_bytes(b"not a video")
    (tmp_path / "mmwave_x.csv").write_text("Date,Time,X,Y,Z\n")
    ok, _ = _validate(tmp_path, "x")
    assert not ok


def test_ingest_one_dry_run(tmp_path):
    _write_triplet(tmp_path, "ts1")
    dest = tmp_path / "out"
    r = ingest_one(tmp_path, dest, "ts1", "cfg-hash", dry_run=True)
    assert r["ok"]
    assert not dest.exists()


def test_ingest_one_copies_and_writes_metadata(tmp_path):
    _write_triplet(tmp_path, "ts1")
    dest = tmp_path / "out"
    r = ingest_one(tmp_path, dest, "ts1", "cfg-hash", dry_run=False)
    assert r["ok"]
    assert (dest / "fisheye_ts1.mp4").exists()
    meta = json.loads((dest / "ts1.metadata.json").read_text())
    assert meta["sensor_config_hash"] == "cfg-hash"
    assert meta["mission_id"] == "out"


def test_main_no_triplets_exits(tmp_path):
    with mock.patch.object(sys, "argv", ["ingest", "--source", str(tmp_path),
                                          "--mission", "m"]):
        with pytest.raises(SystemExit):
            main()


def test_main_source_not_dir(tmp_path):
    with mock.patch.object(sys, "argv", ["ingest", "--source",
                                          str(tmp_path / "missing"),
                                          "--mission", "m"]):
        with pytest.raises(SystemExit):
            main()


def test_validate_video_with_no_frames(tmp_path, monkeypatch):
    """A video that opens but reports zero frames hits the 'no frames'
    branch (lines 73-74). OpenCV refuses to open a truly empty mp4, so we
    fake a VideoCapture that opens but returns FRAME_COUNT == 0."""
    _write_triplet(tmp_path, "empty")

    class FakeCap:
        def __init__(self, path):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            return 0  # zero frames / zero size

        def release(self):
            pass

    monkeypatch.setattr("scripts.data.ingest.cv2.VideoCapture", FakeCap)
    ok, report = _validate(tmp_path, "empty")
    assert not ok
    assert report["files"]["fisheye"]["error"] == "no frames"


def test_validate_csv_read_raises(tmp_path):
    """A non-readable mmwave path (a directory) makes pd.read_csv raise,
    exercising the except branch (lines 79-81)."""
    _write_triplet(tmp_path, "badcsv")
    # Replace the csv file with a directory so read_csv raises.
    mm = tmp_path / "mmwave_badcsv.csv"
    mm.unlink()
    mm.mkdir()
    ok, report = _validate(tmp_path, "badcsv")
    assert not ok
    assert report["files"]["mmwave"]["ok"] is False
    assert "error" in report["files"]["mmwave"]


def test_ingest_one_returns_not_ok_on_invalid(tmp_path):
    """ingest_one short-circuits when validation fails (line 96)."""
    _write_triplet(tmp_path, "empty", add_frames=False)
    dest = tmp_path / "out"
    r = ingest_one(tmp_path, dest, "empty", "cfg-hash", dry_run=False)
    assert r["ok"] is False
    assert not dest.exists()


def test_main_warns_when_dest_exists(tmp_path, monkeypatch, capsys):
    """An existing dest dir (not dry-run) prints the 'already exists'
    warning (line 135)."""
    src = tmp_path / "src"; src.mkdir()
    _write_triplet(src, "ts1")
    captures = tmp_path / "captures"
    monkeypatch.setattr("scripts.data.ingest.CAPTURES_DIR", captures)
    (captures / "mission_x").mkdir(parents=True)   # pre-existing dest
    with mock.patch.object(sys, "argv", ["ingest", "--source", str(src),
                                          "--mission", "mission_x"]):
        main()
    out = capsys.readouterr().out
    assert "already exists" in out


def test_main_dry_run_does_not_copy(tmp_path, monkeypatch, capsys):
    """Dry-run prints the no-copy notice (line 145) and writes nothing."""
    src = tmp_path / "src"; src.mkdir()
    _write_triplet(src, "ts1")
    captures = tmp_path / "captures"
    monkeypatch.setattr("scripts.data.ingest.CAPTURES_DIR", captures)
    with mock.patch.object(sys, "argv", ["ingest", "--source", str(src),
                                          "--mission", "m", "--dry-run"]):
        main()
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert not (captures / "m" / "fisheye_ts1.mp4").exists()


def test_main_happy_path(tmp_path, monkeypatch):
    src = tmp_path / "src"; src.mkdir()
    _write_triplet(src, "ts1")
    captures = tmp_path / "captures"
    monkeypatch.setattr("scripts.data.ingest.CAPTURES_DIR", captures)
    with mock.patch.object(sys, "argv", ["ingest", "--source", str(src),
                                          "--mission", "mission_x"]):
        main()
    assert (captures / "mission_x" / "fisheye_ts1.mp4").exists()


# --- fisheye-only captures --------------------------------------------------

def _write_fisheye_only(dir_: Path, ts: str, add_sidecars: bool = True):
    """What the capture service writes under ASVPROJECT_FISHEYE_ONLY=1:
    the fisheye mp4 + the per-frame timestamp sidecar, and nothing else."""
    fish = dir_ / f"fisheye_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fw = cv2.VideoWriter(str(fish), fourcc, 3.0, (40, 30))
    for _ in range(3):
        fw.write(np.full((30, 40, 3), 80, dtype=np.uint8))
    fw.release()
    if add_sidecars:
        (dir_ / f"frames_{ts}.csv").write_text(
            "frame_index,Date,Time\n0,2099-01-01,00:00:00.0\n")
    return fish


def test_validate_fisheye_only_passes(tmp_path):
    _write_fisheye_only(tmp_path, "ts1")
    ok, report = _validate(tmp_path, "ts1")
    assert ok
    assert report["files"]["fisheye"]["ok"] is True
    # Absent is recorded as absent: never as a failure.
    assert report["files"]["thermal"] == {"ok": True, "present": False}
    assert report["files"]["mmwave"] == {"ok": True, "present": False}
    assert report["streams"] == ["fisheye", "frames"]


def test_ingest_fisheye_only_copies_only_what_exists(tmp_path):
    _write_fisheye_only(tmp_path, "ts1")
    dest = tmp_path / "out"
    r = ingest_one(tmp_path, dest, "ts1", None, dry_run=False)
    assert r["ok"]
    assert r["streams"] == ["fisheye", "frames"]
    assert (dest / "fisheye_ts1.mp4").exists()
    assert (dest / "frames_ts1.csv").exists()
    assert not (dest / "thermal_ts1.mp4").exists()
    assert not (dest / "mmwave_ts1.csv").exists()
    meta = json.loads((dest / "ts1.metadata.json").read_text())
    assert meta["streams"] == ["fisheye", "frames"]


def test_missing_fisheye_fails_validation(tmp_path):
    ok, report = _validate(tmp_path, "nothing")
    assert not ok
    assert report["files"]["fisheye"]["error"] == "missing"


def test_present_but_broken_thermal_still_fails(tmp_path):
    """Absent != broken: a thermal file that IS there but won't open must
    still fail, or a truncated recording would ingest as 'fisheye-only'."""
    _write_fisheye_only(tmp_path, "ts1")
    (tmp_path / "thermal_ts1.mp4").write_bytes(b"not a video")
    ok, report = _validate(tmp_path, "ts1")
    assert not ok
    assert report["files"]["thermal"]["ok"] is False


def test_ingest_copies_imu_and_frames_sidecars(tmp_path):
    """The IMU + per-frame timestamp sidecars are part of the capture; before
    this they were left behind on the Pi by ingest."""
    _write_triplet(tmp_path, "ts1")
    (tmp_path / "imu_ts1.csv").write_text(
        "Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az\n"
        "2099-01-01,00:00:00.0,1,2,3,0,0,9.8\n")
    (tmp_path / "frames_ts1.csv").write_text(
        "frame_index,Date,Time\n0,2099-01-01,00:00:00.0\n")
    dest = tmp_path / "out"
    r = ingest_one(tmp_path, dest, "ts1", None, dry_run=False)
    assert r["ok"]
    assert (dest / "imu_ts1.csv").exists()
    assert (dest / "frames_ts1.csv").exists()
    assert r["streams"] == ["fisheye", "frames", "imu", "mmwave", "thermal"]


def test_main_reports_streams_per_capture(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src"; src.mkdir()
    _write_fisheye_only(src, "ts1")
    captures = tmp_path / "captures"
    monkeypatch.setattr("scripts.data.ingest.CAPTURES_DIR", captures)
    with mock.patch.object(sys, "argv", ["ingest", "--source", str(src),
                                          "--mission", "m"]):
        main()
    out = capsys.readouterr().out
    assert "[fisheye+frames]" in out
    assert "1/1 ingested cleanly" in out
