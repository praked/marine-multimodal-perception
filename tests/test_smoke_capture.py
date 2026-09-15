"""Verdict logic in scripts/data_collection/smoke_capture.py.

`smoke_capture` is excluded from the coverage report (it lives under
`data_collection/`, which needs the Pi), but `validate()` is pure logic over
files already on disk, and it is what an operator reads in the field to decide
whether the box is fit to leave the dock. The rule it encodes:

    a sensor that is switched OFF by configuration is not a failure;
    a sensor that was EXPECTED and did not deliver is.

Getting that backwards makes a correct fisheye-only box report FAIL, which is
exactly the situation these tests exist to prevent.
"""

import cv2
import numpy as np
import pytest

from scripts.data_collection.smoke_capture import validate

ALL_ON = (True, True, True, True)          # (fisheye, thermal, radar, imu)
FISHEYE_ONLY = (True, False, False, True)


def _write_mp4(path, n=5, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 3.0, size)
    for _ in range(n):
        frame = np.full((size[1], size[0], 3), 90, dtype=np.uint8)
        frame[10:30, 10:40] = 220            # some contrast for the thermal check
        w.write(frame)
    w.release()


@pytest.fixture
def fisheye_only_dir(tmp_path):
    """What a ASVPROJECT_FISHEYE_ONLY chunk leaves behind: the fisheye mp4 and
    the per-frame timestamp sidecar, and nothing else."""
    _write_mp4(tmp_path / "fisheye_ts.mp4")
    (tmp_path / "frames_ts.csv").write_text(
        "frame_index,Date,Time\n0,2099-01-01,00:00:00.0\n")
    return tmp_path


@pytest.fixture
def full_stack_dir(tmp_path):
    _write_mp4(tmp_path / "fisheye_ts.mp4")
    _write_mp4(tmp_path / "thermal_ts.mp4", size=(160, 120))
    (tmp_path / "mmwave_ts.csv").write_text(
        "Date,Time,X,Y,Z,V,SNR,NOISE\n"
        "2099-01-01,00:00:00.1,0.5,2.0,0.1,0.0,12.3,24.0\n")
    (tmp_path / "imu_ts.csv").write_text(
        "Date,Time,Yaw,Pitch,Roll,Ax,Ay,Az\n"
        "2099-01-01,00:00:00.1,1,2,3,0,0,9.8\n")
    (tmp_path / "frames_ts.csv").write_text(
        "frame_index,Date,Time\n0,2099-01-01,00:00:00.0\n")
    return tmp_path


def test_fisheye_only_passes_with_disabled_sensors(fisheye_only_dir, capsys):
    """The headline case: radar + thermal off by configuration, so the run is
    a PASS and they are reported OFF rather than FAIL."""
    ok = validate(str(fisheye_only_dir),
                  opened=(True, False, False, True),
                  enabled=FISHEYE_ONLY)
    out = capsys.readouterr().out
    assert ok is True
    assert "thermal : OFF" in out
    assert "radar   : OFF" in out
    assert "FAIL" not in out
    assert "OVERALL : PASS" in out


def test_absent_sensor_still_fails_when_expected(fisheye_only_dir, capsys):
    """Same files, but the run EXPECTED the full stack: that is a real
    failure, and must not be softened by the disabled-sensor handling."""
    ok = validate(str(fisheye_only_dir),
                  opened=(True, False, False, True),
                  enabled=ALL_ON)
    out = capsys.readouterr().out
    assert ok is False
    assert "thermal : FAIL" in out
    assert "radar   : FAIL" in out


def test_full_stack_passes(full_stack_dir, capsys):
    ok = validate(str(full_stack_dir), opened=ALL_ON, enabled=ALL_ON)
    out = capsys.readouterr().out
    assert ok is True
    assert "OVERALL : PASS" in out


def test_enabled_defaults_to_all_on(fisheye_only_dir, capsys):
    """Omitting `enabled` keeps the original strict behaviour, so any caller
    that has not been updated cannot silently start passing broken runs."""
    ok = validate(str(fisheye_only_dir), opened=(True, False, False, True))
    out = capsys.readouterr().out
    assert ok is False
    assert "thermal : FAIL" in out


def test_disabled_imu_reported_off_and_never_fails(fisheye_only_dir, capsys):
    ok = validate(str(fisheye_only_dir),
                  opened=(True, False, False, False),
                  enabled=(True, False, False, False))
    out = capsys.readouterr().out
    assert ok is True
    assert "imu     : OFF" in out


def test_fisheye_failure_fails_the_run(tmp_path, capsys):
    """The fisheye is the one sensor that can never be disabled, so a fisheye
    that produced no frames is always a failure."""
    (tmp_path / "fisheye_ts.mp4").write_bytes(b"not a video")
    ok = validate(str(tmp_path), opened=(True, False, False, False),
                  enabled=FISHEYE_ONLY)
    out = capsys.readouterr().out
    assert ok is False
    assert "fisheye : FAIL" in out


def test_frames_sidecar_reported(fisheye_only_dir, capsys):
    """The per-frame RTC sidecar is what lets a fisheye-only clip be aligned
    to other streams later; report it, but never fail on it."""
    validate(str(fisheye_only_dir), opened=(True, False, False, True),
             enabled=FISHEYE_ONLY)
    out = capsys.readouterr().out
    assert "frames  : PASS" in out


# --- GPS verdict ------------------------------------------------------------
#
# GPS is own-boat position metadata read from boat1's autopilot log. The rule
# it encodes, on top of the enabled/expected one above:
#
#     a row exists is NOT the same as a position was measured;
#     the fixed fallback in configs/gps.yaml must never read as a live fix.
#
# Getting that wrong means a mission's clips carry a plausible-looking harbour
# position that nobody ever measured, and nothing downstream can tell.

ALL_ON_GPS = (True, True, True, True, True)     # + gps

GPS_HEADER = "Date,Time,Lat,Lon,Fix,Source,FixTime\n"


def _write_gps(d, source="boat_log", fix="4"):
    (d / "gps_ts.csv").write_text(
        GPS_HEADER + f"2026-08-24,12:00:00.1,46.0000000,9.0000000,{fix},"
                     f"{source},2026-08-24T09:59:58+00:00\n")


def test_gps_live_fix_passes(full_stack_dir, capsys):
    _write_gps(full_stack_dir)
    assert validate(str(full_stack_dir), ALL_ON_GPS, ALL_ON_GPS)
    out = capsys.readouterr().out
    assert "gps     : PASS" in out and "source=boat_log" in out


def test_gps_fallback_warns_and_says_so(full_stack_dir, capsys):
    _write_gps(full_stack_dir, source="fallback", fix="")
    assert validate(str(full_stack_dir), ALL_ON_GPS, ALL_ON_GPS)   # not a failure
    out = capsys.readouterr().out
    assert "gps     : WARN" in out
    assert "FALLBACK position from configs/gps.yaml, NOT a live fix" in out


def test_gps_fallback_fails_under_require_gps(full_stack_dir):
    _write_gps(full_stack_dir, source="fallback", fix="")
    assert not validate(str(full_stack_dir), ALL_ON_GPS, ALL_ON_GPS,
                        require_gps=True)


def test_gps_live_fix_satisfies_require_gps(full_stack_dir):
    _write_gps(full_stack_dir)
    assert validate(str(full_stack_dir), ALL_ON_GPS, ALL_ON_GPS,
                    require_gps=True)


def test_gps_no_rows_warns_but_does_not_fail_the_smoke(full_stack_dir, capsys):
    """A missing position costs training metadata, not a single frame: it must
    not mask a real sensor failure by turning the overall verdict red."""
    (full_stack_dir / "gps_ts.csv").write_text(GPS_HEADER)
    assert validate(str(full_stack_dir), ALL_ON_GPS, ALL_ON_GPS)
    assert "gps     : WARN  0 row(s)" in capsys.readouterr().out


def test_gps_disabled_is_not_a_failure(full_stack_dir, capsys):
    enabled = (True, True, True, True, False)
    assert validate(str(full_stack_dir), (True, True, True, True, False), enabled)
    assert "gps     : OFF" in capsys.readouterr().out


def test_gps_status_detail_is_printed(full_stack_dir, capsys):
    """The reason a GPS read failed is the only actionable part of the line."""
    _write_gps(full_stack_dir, source="fallback", fix="")
    validate(str(full_stack_dir), ALL_ON_GPS, ALL_ON_GPS,
             gps_status="boat_log[boat-b@boat-b] no fix: no route to host "
                        "| fallback: +46.000000,+9.000000")
    out = capsys.readouterr().out
    assert "boat-b@boat-b" in out and "no route to host" in out


def test_pre_gps_four_tuples_still_validate(full_stack_dir, capsys):
    """Callers written before GPS existed pass 4-tuples; GPS then reads as
    absent, which is what it was."""
    assert validate(str(full_stack_dir), ALL_ON, ALL_ON)
    assert "gps     : OFF" in capsys.readouterr().out
