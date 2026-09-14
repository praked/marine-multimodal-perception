"""Tests for the boat1 GPS reader: boat-log / NMEA parsing, the source chain,
and offline sun position."""

import asyncio
import os
import struct
import types
from datetime import datetime, timezone

import pytest

import scripts.sensor_processing.gps_boat1 as gps
from scripts.sensor_processing.gps_boat1 import (
    GPSFix,
    GPSReaderBLE,
    load_gps_config,
    parse_gps_packet,
    sun_position,
)


# --- packet parsing ---------------------------------------------------------

def test_parse_nmea_gga():
    # InstitutionOne-ish: 47 39.6'N, 009 10.8'E, RTK-fixed (quality 4)
    s = b"$GNGGA,120000.00,4739.6000,N,00910.8000,E,4,12,0.8,400.0,M,48.0,M,,*XX"
    fix = parse_gps_packet(s, "nmea")
    assert fix is not None
    assert fix.lat_deg == pytest.approx(47 + 39.6 / 60, abs=1e-6)
    assert fix.lon_deg == pytest.approx(9 + 10.8 / 60, abs=1e-6)
    assert fix.fix_quality == 4


def test_parse_nmea_southern_western_hemisphere():
    s = b"$GPGGA,000000,3345.0000,S,07030.0000,W,1,08,1.0,10.0,M,,M,,*00"
    fix = parse_gps_packet(s, "nmea")
    assert fix.lat_deg == pytest.approx(-(33 + 45.0 / 60), abs=1e-6)
    assert fix.lon_deg == pytest.approx(-(70 + 30.0 / 60), abs=1e-6)


def test_parse_nmea_rmc_when_no_gga():
    s = b"$GNRMC,120000,A,4739.6000,N,00910.8000,E,0.0,0.0,140726,,,A*00"
    fix = parse_gps_packet(s, "nmea")
    assert fix is not None
    assert fix.lat_deg == pytest.approx(47 + 39.6 / 60, abs=1e-6)


def test_parse_nmea_no_fix_returns_none():
    s = b"$GNGGA,120000.00,,,,,0,00,,,M,,M,,*00"     # quality 0 = no fix
    assert parse_gps_packet(s, "nmea") is None


def test_parse_latlon_f64_le():
    data = struct.pack("<dd", 47.66, 9.18)
    fix = parse_gps_packet(data, "latlon_f64_le")
    assert fix.lat_deg == pytest.approx(47.66)
    assert fix.lon_deg == pytest.approx(9.18)
    assert fix.fix_quality is None


def test_parse_csv():
    fix = parse_gps_packet(b"47.66,9.18,4\n", "csv")
    assert (fix.lat_deg, fix.lon_deg, fix.fix_quality) == pytest.approx((47.66, 9.18, 4))


def test_parse_short_binary_returns_none():
    assert parse_gps_packet(b"\x00\x00", "latlon_f64_le") is None


def test_parse_unknown_fmt_raises():
    with pytest.raises(ValueError):
        parse_gps_packet(b"x", "bogus")


# --- sun position (offline; robust invariants + known cases) ----------------

def test_sun_overhead_at_equator_equinox_noon():
    # Equator, lon 0, equinox, ~solar noon UTC -> sun nearly overhead.
    el, az = sun_position(0.0, 0.0, datetime(2026, 3, 20, 12, 7, tzinfo=timezone.utc))
    assert el > 85.0
    assert 0.0 <= az <= 360.0


def test_sun_below_horizon_at_local_midnight():
    # InstitutionOne, ~local midnight (UTC ~23:00 in summer) -> sun well below horizon.
    el, _az = sun_position(47.66, 9.18, datetime(2026, 6, 21, 23, 0, tzinfo=timezone.utc))
    assert el < 0.0


def test_sun_higher_at_noon_than_morning():
    noon, _ = sun_position(47.66, 9.18, datetime(2026, 6, 21, 11, 30, tzinfo=timezone.utc))
    morn, _ = sun_position(47.66, 9.18, datetime(2026, 6, 21, 5, 0, tzinfo=timezone.utc))
    assert noon > morn > 0.0
    assert noon < 90.0


def test_sun_summer_noon_elevation_institutionone_reasonable():
    # Summer-solstice solar noon at ~47.66N: elevation ~ 90-(47.66-23.44) ~ 65.8 deg.
    el, az = sun_position(47.66, 9.18, datetime(2026, 6, 21, 11, 24, tzinfo=timezone.utc))
    assert el == pytest.approx(65.8, abs=2.0)
    assert 150.0 < az < 210.0        # roughly due south at local noon


def test_sun_position_naive_utc_ok():
    # A naive datetime is treated as UTC (no tzinfo): must not raise.
    el, az = sun_position(0.0, 0.0, datetime(2026, 3, 20, 12, 7))
    assert el > 85.0


# --- config -----------------------------------------------------------------

def test_load_gps_config_defaults_disabled(tmp_path):
    cfg = load_gps_config(tmp_path / "nope.yaml")
    assert cfg["enabled"] is False and cfg["packet_fmt"] == "nmea"


def test_load_gps_config_real_file():
    cfg = load_gps_config()      # repo configs/gps.yaml
    assert "ble_char_uuid" in cfg and "refresh_interval_s" in cfg


# --- additional parsing edge cases ------------------------------------------

def test_nmea_to_deg_empty_field_raises():
    from scripts.sensor_processing.gps_boat1 import _nmea_to_deg
    with pytest.raises(ValueError):
        _nmea_to_deg("", "N")


def test_parse_csv_too_few_fields_returns_none():
    assert parse_gps_packet(b"47.66", "csv") is None


def test_parse_csv_garbage_returns_none():
    # float() raises ValueError inside -> caught -> None
    assert parse_gps_packet(b"abc,def", "csv") is None


def test_parse_nmea_gga_quality_zero_with_coords_returns_none():
    # Coordinates present but fix quality 0: the GGA branch `continue`s and
    # the sentence loop falls through to None.
    s = b"$GNGGA,120000.00,4739.6000,N,00910.8000,E,0,04,1.0,400.0,M,,M,,*00"
    assert parse_gps_packet(s, "nmea") is None


# --- sun position: remaining seasonal / polar branches -----------------------

def test_sun_january_date_julian_rollover():
    # month <= 2 exercises the y-=1/m+=12 Julian-day branch; winter noon at
    # InstitutionOne is low but above the horizon.
    el, az = sun_position(47.66, 9.18,
                          datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc))
    assert 0.0 < el < 30.0
    assert 0.0 <= az <= 360.0


def test_sun_azimuth_at_poles_degenerate_denominator():
    # cos(lat) ~ 0 at the poles -> the azimuth denominator degenerates and
    # the convention fallback applies (180 north pole / 0 south pole).
    _el_n, az_n = sun_position(90.0, 0.0,
                               datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc))
    assert az_n == pytest.approx(180.0)
    _el_s, az_s = sun_position(-90.0, 0.0,
                               datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc))
    assert az_s == pytest.approx(0.0)


# --- GPSReaderBLE (fake bleak; no hardware) ----------------------------------

def _reader(**over):
    cfg = {"ble_name": "boat1-gps", "ble_address": "", "ble_char_uuid": "",
           "packet_fmt": "csv", "refresh_interval_s": 0.0,
           "stale_after_s": 3600.0}
    cfg.update(over)
    return GPSReaderBLE(cfg)


@pytest.fixture
def fast_asyncio(monkeypatch):
    """Shim gps.asyncio so backoff sleeps inside _loop are instant."""
    async def _sleep(_s):
        return None
    shim = types.SimpleNamespace(run=asyncio.run, sleep=_sleep)
    monkeypatch.setattr(gps, "asyncio", shim)
    return shim


def test_reader_get_none_before_any_fix():
    assert _reader().get() is None


def test_reader_publish_and_get_fresh_fix():
    r = _reader()
    r._publish(GPSFix(47.0, 9.0, fix_quality=4))
    fix = r.get()
    assert fix is not None
    assert (fix.lat_deg, fix.lon_deg) == (47.0, 9.0)
    assert fix.timestamp is not None and r.reads_ok == 1


def test_reader_get_stale_fix_returns_none():
    r = _reader(stale_after_s=-1.0)     # everything is instantly stale
    r._publish(GPSFix(47.0, 9.0))
    assert r.get() is None


def test_reader_loop_bleak_missing(monkeypatch, capsys):
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "bleak", None)   # import -> ImportError
    r = _reader(ble_char_uuid="abcd")
    asyncio.run(r._loop())
    assert "bleak missing" in capsys.readouterr().out


def test_reader_loop_no_char_uuid(monkeypatch, capsys):
    _fake_bleak(monkeypatch, scanner=object, client_cls=object)
    r = _reader(ble_char_uuid="")
    asyncio.run(r._loop())
    assert "no ble_char_uuid" in capsys.readouterr().out


def _fake_bleak(monkeypatch, scanner=None, client_cls=None):
    import sys as _sys
    mod = types.ModuleType("bleak")
    mod.BleakScanner = scanner
    mod.BleakClient = client_cls
    monkeypatch.setitem(_sys.modules, "bleak", mod)
    return mod


def test_reader_loop_scan_not_found_then_stop(monkeypatch, fast_asyncio):
    r = _reader(ble_char_uuid="abcd")

    class Scanner:
        @staticmethod
        async def find_device_by_name(name, timeout):
            r.stop()               # one pass, then exit the outer loop
            return None
    _fake_bleak(monkeypatch, scanner=Scanner)
    asyncio.run(r._loop())
    assert r.get() is None and not r.connected


def test_reader_loop_connect_read_publish(monkeypatch, fast_asyncio):
    r = _reader(ble_char_uuid="abcd", packet_fmt="csv")
    calls = {"n": 0}

    class Scanner:
        @staticmethod
        async def find_device_by_name(name, timeout):
            return types.SimpleNamespace(address="AA:BB")

    class Client:
        def __init__(self, addr):
            assert addr == "AA:BB"
            self.is_connected = True

        async def __aenter__(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("transient BLE failure")   # session-error branch
            return self

        async def __aexit__(self, *a):
            return False

        async def read_gatt_char(self, char):
            assert char == "abcd"
            r.stop()               # exit after the first successful read
            return bytearray(b"47.66,9.18,4")

    _fake_bleak(monkeypatch, scanner=Scanner, client_cls=Client)
    asyncio.run(r._loop())
    fix = r.get()
    assert fix is not None
    assert fix.lat_deg == pytest.approx(47.66)
    assert fix.fix_quality == 4
    assert r.reads_ok == 1
    assert calls["n"] == 2         # first session raised, second published


def test_reader_loop_uses_configured_address(monkeypatch, fast_asyncio):
    # With ble_address set the scanner must NOT be consulted.
    r = _reader(ble_char_uuid="abcd", ble_address="CC:DD")

    class Client:
        def __init__(self, addr):
            assert addr == "CC:DD"
            self.is_connected = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def read_gatt_char(self, char):
            r.stop()
            return bytearray(b"1.0,2.0")
    _fake_bleak(monkeypatch, scanner=None, client_cls=Client)
    asyncio.run(r._loop())
    assert r.get().lat_deg == pytest.approx(1.0)


def test_reader_refresh_sleep_chunked_and_interruptible(monkeypatch):
    # The ~30 min refresh sleep must be walked in <=1 s chunks so stop()
    # (close() joins with a 3 s timeout) interrupts it promptly.
    r = _reader(ble_char_uuid="abcd", ble_address="EE:FF",
                refresh_interval_s=2.5)
    chunks = []

    async def _sleep(s):
        chunks.append(s)
        if len(chunks) == 2:
            r.stop()               # stop mid-interval: loop must bail out

    monkeypatch.setattr(gps, "asyncio",
                        types.SimpleNamespace(run=asyncio.run, sleep=_sleep))

    class Client:
        def __init__(self, addr):
            self.is_connected = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def read_gatt_char(self, char):
            return bytearray(b"1.0,2.0")

    _fake_bleak(monkeypatch, scanner=None, client_cls=Client)
    asyncio.run(r._loop())
    assert chunks == [1.0, 1.0]    # 2.5 s walked in chunks, cut off by stop()
    assert r.reads_ok == 1


def test_reader_run_survives_loop_exception(capsys):
    r = _reader()

    async def _boom():
        raise RuntimeError("kaboom")
    r._loop = _boom
    r.run()                        # must not raise; capture is never killed
    assert not r.connected
    assert "reader stopped" in capsys.readouterr().out


def test_reader_start_and_close_thread(monkeypatch):
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "bleak", None)
    r = _reader(ble_char_uuid="")   # loop returns immediately
    r.start()
    r.close()                       # stop + join; thread already finished
    assert not r.is_alive()


# --- CLI --------------------------------------------------------------------

def test_main_scan_lists_devices(monkeypatch, capsys):
    class Scanner:
        @staticmethod
        async def discover(timeout):
            return [types.SimpleNamespace(address="AA:BB", name="boat1-gps"),
                    types.SimpleNamespace(address="CC:DD", name=None)]
    _fake_bleak(monkeypatch, scanner=Scanner)
    assert gps.main(["--scan", "--seconds", "0.1"]) == 0
    out = capsys.readouterr().out
    assert "AA:BB" in out and "boat1-gps" in out
    assert "(no name)" in out


def test_main_scan_without_bleak(monkeypatch, capsys):
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "bleak", None)
    assert gps.main(["--scan"]) == 1
    assert "need bleak" in capsys.readouterr().out


class _FakeReader:
    """Stands in for the whole GPSReader chain in the CLI: no BLE, no ssh, no
    threads."""

    fix = None
    closed = False

    def __init__(self, config=None):
        type(self).closed = False

    def start(self):
        pass

    def get(self, live_only=False):
        return type(self).fix

    def wait_for_fix(self, timeout_s=15.0):
        return type(self).fix

    def describe(self):
        return "fake reader"

    def close(self):
        type(self).closed = True


def _run_cli(monkeypatch, fix, argv):
    """Run a CLI mode with a canned fix; time.sleep raises KeyboardInterrupt so
    a streaming loop runs exactly once and the finally-close path executes."""
    _FakeReader.fix = fix
    monkeypatch.setattr(gps, "GPSReader", _FakeReader)

    def _sleep(_s):
        raise KeyboardInterrupt
    monkeypatch.setattr(gps, "time", types.SimpleNamespace(
        time=lambda: 0.0, sleep=_sleep, monotonic=lambda: 0.0))
    return gps.main(argv)


def test_main_monitor_prints_fix_and_sun(monkeypatch, capsys):
    rc = _run_cli(monkeypatch, GPSFix(47.66, 9.18, fix_quality=4), ["--monitor"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "lat=+47.66" in out and "sun elev=" in out
    assert _FakeReader.closed


def test_main_monitor_waiting_without_fix(monkeypatch, capsys):
    rc = _run_cli(monkeypatch, None, ["--monitor"])
    assert rc == 0
    assert "waiting for a fix" in capsys.readouterr().out
    assert _FakeReader.closed


def test_main_check_reports_a_live_fix(monkeypatch, capsys):
    """No args = --check: resolve one fix exactly the way capture init does."""
    rc = _run_cli(monkeypatch,
                  GPSFix(47.695639, 9.193917, fix_quality=4, source="boat_log"),
                  [])
    out = capsys.readouterr().out
    assert rc == 0
    assert "source=boat_log" in out and "sun elevation" in out


def test_main_check_exit_2_on_a_fallback_position(monkeypatch, capsys):
    """A distinct exit code, because a fallback IS a resolved position and
    would otherwise read as success in a deploy-day script."""
    rc = _run_cli(monkeypatch, GPSFix(47.6, 9.1, source="fallback"), [])
    assert rc == 2
    assert "source=fallback" in capsys.readouterr().out


def test_main_check_exit_1_when_nothing_answers(monkeypatch, capsys):
    rc = _run_cli(monkeypatch, None, [])
    assert rc == 1
    assert "no fix from any configured source" in capsys.readouterr().out


def test_main_log_only_bypasses_the_chain(monkeypatch, capsys):
    """--log-only is the sharpest test of the SSH/path setup: it reports the
    boat-log read itself, with no fallback to paper over a failure."""
    monkeypatch.setattr(gps, "read_boat_log_fix",
                        lambda *a, **k: (None, "boat-b@boat-b: no route"))
    assert gps.main(["--log-only"]) == 1
    assert "no route" in capsys.readouterr().out


# --- boat1's autopilot log: the default source ------------------------------
#
# The payload format these exercise is what `_local_log_payload` /
# `_ssh_log_payload` produce and `parse_boat_log_payload` consumes: a `#<path>`
# line, the header, then the tail rows. Column names and the ~10 Hz row cadence
# come from boatv1's utils/data_logger.py (branch auto-work).

from datetime import timedelta                                  # noqa: E402

from scripts.sensor_processing.gps_boat1 import (                # noqa: E402
    GPSReader,
    GPSReaderBoatLog,
    parse_boat_log_payload,
    read_boat_log_fix,
    _check_globs,
    _local_log_payload,
)

# The real header, trimmed to the columns that matter plus enough neighbours
# to keep the interesting ones off index 0 (a positional bug would pass on a
# three-column stub).
LOG_HEADER = ("ts_iso,gps_lat,gps_lon,gps_ts,awa_deg,wind_spd,"
              "mode,auto_mode,wp_left,dist_m,fix_q,sys_temp_c")


def _row(ts, lat="", lon="", fix=""):
    return f"{ts},{lat},{lon},,12.0,3.4,AUTO,SK,2,40.1,{fix},45.3"


def _payload(rows, path="/home/boat-b/boatv1/logs/boat_log_20260824_120000.csv"):
    return "\n".join([f"#{path}", LOG_HEADER, *rows]) + "\n"


def _iso(delta_s=0.0):
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_s)).isoformat()


def test_boat_log_takes_the_newest_row_carrying_a_fix():
    # Fixless warm-up rows, real fixes, then the autopilot logging on without a
    # fix again: the answer must be the LAST row with coordinates, not the last
    # row and not the first fix.
    rows = ([_row(_iso(-60))] * 3
            + [_row(_iso(-40), "47.6950", "9.1930", "4"),
               _row(_iso(-30), "47.695639", "9.193917", "4")]
            + [_row(_iso(-5))])
    fix, path, note = parse_boat_log_payload(_payload(rows))
    assert fix is not None
    assert (fix.lat_deg, fix.lon_deg) == pytest.approx((47.695639, 9.193917))
    assert fix.fix_quality == 4 and fix.source == "boat_log"
    assert path.endswith("boat_log_20260824_120000.csv")
    assert "47.695639" in note


def test_boat_log_records_the_rows_own_time_not_ours():
    stamp = _iso(-120)
    fix, _p, _n = parse_boat_log_payload(_payload([_row(stamp, "47.7", "9.2", "4")]))
    assert fix.fix_time_utc == datetime.fromisoformat(stamp)


def test_boat_log_skips_null_island():
    """Empty NMEA fields coerced to 0.0 somewhere upstream: a fixless receiver,
    not a position in the Gulf of Guinea."""
    rows = [_row(_iso(-30), "47.695639", "9.193917", "4"),
            _row(_iso(-1), "0.0", "0.0", "1")]
    fix, _p, _n = parse_boat_log_payload(_payload(rows))
    assert fix.lat_deg == pytest.approx(47.695639)


def test_boat_log_no_fix_anywhere():
    fix, _p, note = parse_boat_log_payload(_payload([_row(_iso(-2))] * 5))
    assert fix is None and "no GPS lock" in note


def test_boat_log_header_only():
    fix, _p, note = parse_boat_log_payload(_payload([]))
    assert fix is None and "header-only" in note


def test_boat_log_missing_file_is_reported_not_raised():
    fix, path, note = parse_boat_log_payload("")
    assert (fix, path) == (None, None) and "no boat log" in note


def test_boat_log_without_gps_columns():
    text = "#/x.csv\nts_iso,mode,rudder_deg\n" + f"{_iso()},AUTO,3.0\n"
    fix, _p, note = parse_boat_log_payload(text)
    assert fix is None and "no gps_lat" in note


def test_boat_log_garbage_coordinates_are_skipped():
    rows = [_row(_iso(-30), "47.7", "9.2", "4"), _row(_iso(-1), "nan-ish", "x")]
    fix, _p, _n = parse_boat_log_payload(_payload(rows))
    assert fix.lat_deg == pytest.approx(47.7)


def test_boat_log_missing_fix_quality_is_none_not_zero():
    """Empty fix_q means 'not reported', which is not the same as GGA 0 = no
    fix; recording it as 0 would mark a good RTK position as fixless."""
    fix, _p, _n = parse_boat_log_payload(
        _payload([_row(_iso(-1), "47.7", "9.2", "")]))
    assert fix.fix_quality is None


def test_boat_log_stale_fix_is_refused_with_a_reason():
    """A log file on disk says nothing about whether the autopilot still runs."""
    old = (datetime.now(timezone.utc) - timedelta(hours=9)).isoformat()
    fix, _p, note = parse_boat_log_payload(
        _payload([_row(old, "47.7", "9.2", "4")]), max_age_s=3600.0)
    assert fix is None
    assert "9.0 h old" in note and "autopilot stopped" in note


def test_boat_log_stale_fix_kept_when_no_age_limit():
    old = (datetime.now(timezone.utc) - timedelta(hours=9)).isoformat()
    fix, _p, _n = parse_boat_log_payload(_payload([_row(old, "47.7", "9.2")]))
    assert fix is not None


def test_boat_log_naive_timestamp_treated_as_utc():
    naive = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    fix, _p, _n = parse_boat_log_payload(
        _payload([_row(naive, "47.7", "9.2")]), max_age_s=3600.0)
    assert fix is not None and fix.fix_time_utc.tzinfo is timezone.utc


# --- glob safety: these patterns are pasted into a remote shell --------------

def test_check_globs_accepts_a_bare_string_and_a_list():
    assert _check_globs("~/boatv1/logs/*.csv") == ["~/boatv1/logs/*.csv"]
    assert _check_globs(["/a/*.csv", "", "/b/*.csv"]) == ["/a/*.csv", "/b/*.csv"]


@pytest.mark.parametrize("pattern", [
    "/tmp/x; rm -rf /",          # command separator
    "$(cat /etc/passwd)",        # substitution
    "/logs/*.csv && curl evil",  # chaining
    "`id`",                      # backticks
])
def test_check_globs_rejects_shell_metacharacters(pattern):
    """log_glob is deliberately NOT quoted in the remote command (the shell has
    to expand it), so the validation is the only thing between a config value
    and arbitrary execution on boat1."""
    with pytest.raises(ValueError):
        _check_globs(pattern)


def test_unsafe_glob_is_a_config_error_not_a_crash():
    fix, note = read_boat_log_fix(
        {"boat_log": {"ssh_host": "", "log_glob": "/tmp/x; rm -rf /"}})
    assert fix is None and "config error" in note


# --- local payload assembly -------------------------------------------------

def _write_log(path, rows):
    path.write_text(LOG_HEADER + "\n" + "\n".join(rows) + "\n")


def test_local_payload_picks_the_newest_file_by_mtime(tmp_path):
    """A new log file per autopilot run, so 'newest fix' starts with 'newest
    file'; the names sort the wrong way round here on purpose."""
    old = tmp_path / "boat_log_20991231_235959.csv"      # sorts last by name
    new = tmp_path / "boat_log_20260101_000000.csv"
    _write_log(old, [_row(_iso(-5), "1.0", "2.0", "1")])
    _write_log(new, [_row(_iso(-5), "47.7", "9.2", "4")])
    os.utime(old, (1, 1))
    fix, _note = read_boat_log_fix(
        {"boat_log": {"ssh_host": "", "log_glob": str(tmp_path / "boat_log_*.csv"),
                      "max_age_s": None}})
    assert fix.lat_deg == pytest.approx(47.7)


def test_local_payload_drops_a_half_read_first_row(tmp_path):
    """A byte-bounded tail starts mid-row, and a truncated row can still parse
    into plausible-looking columns; the first line is discarded for that reason
    (matching the remote `tail -c | tail -n +2`)."""
    log = tmp_path / "boat_log_20260824_120000.csv"
    _write_log(log, [_row(_iso(-i), f"{47.0 + i / 1000:.4f}", "9.2", "4")
                     for i in range(200, 0, -1)])
    payload = _local_log_payload([str(log)], tail_bytes=300)
    lines = payload.splitlines()
    assert lines[0] == f"#{log}"
    assert lines[1] == LOG_HEADER
    # Every surviving row must be whole: same field count as the header.
    n_cols = len(LOG_HEADER.split(","))
    assert all(len(ln.split(",")) == n_cols for ln in lines[2:] if ln.strip())


def test_local_payload_empty_when_nothing_matches(tmp_path):
    assert _local_log_payload([str(tmp_path / "nope_*.csv")], 1024) == ""


# --- ssh transport (fake subprocess; no boat) -------------------------------

def _fake_ssh(monkeypatch, *, stdout="", returncode=0, stderr="", capture=None):
    def _run(cmd, **kwargs):
        if capture is not None:
            capture.append((cmd, kwargs))
        return types.SimpleNamespace(args=cmd, returncode=returncode,
                                     stdout=stdout, stderr=stderr)
    monkeypatch.setattr(gps.subprocess, "run", _run)


def test_ssh_reads_the_log_in_one_round_trip(monkeypatch):
    """One command, not three (find / header / tail): halves the latency on a
    marginal boat link and removes the race where the log rotates mid-read."""
    calls = []
    _fake_ssh(monkeypatch,
              stdout=_payload([_row(_iso(-10), "47.695639", "9.193917", "4")]),
              capture=calls)
    fix, note = read_boat_log_fix({"boat_log": {
        "ssh_host": "boat-b@boat-b",
        "log_glob": ["~/boatv1/logs/boat_log_*.csv"],
        "ssh_options": ["BatchMode=yes"], "tail_bytes": 4096,
        "ssh_timeout_s": 5.0, "max_age_s": 3600.0}})
    assert fix.lat_deg == pytest.approx(47.695639)
    assert note.startswith("boat-b@boat-b:/home/boat-b/boatv1/logs/")
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd[:4] == ["ssh", "-o", "BatchMode=yes", "boat-b@boat-b"]
    remote = cmd[-1]
    assert "ls -1t ~/boatv1/logs/boat_log_*.csv" in remote
    assert "tail -c 4096" in remote and "tail -n +2" in remote
    assert kwargs["timeout"] == 5.0


def test_ssh_exit_3_means_connected_but_no_log_there(monkeypatch):
    """Distinct from an ssh failure: the link is fine, boat1 has just never run
    the autopilot (or BOAT_LOG_DIR points somewhere else)."""
    _fake_ssh(monkeypatch, returncode=3)
    fix, note = read_boat_log_fix({"boat_log": {
        "ssh_host": "boat-b@boat-b", "log_glob": "/x/*.csv",
        "ssh_options": [], "max_age_s": None}})
    assert fix is None and "no boat log found" in note


def test_ssh_failure_reports_what_ssh_said(monkeypatch):
    _fake_ssh(monkeypatch, returncode=255,
              stderr="ssh: connect to host boat-b port 22: No route to host")
    fix, note = read_boat_log_fix({"boat_log": {
        "ssh_host": "boat-b@boat-b", "log_glob": "/x/*.csv",
        "ssh_options": [], "max_age_s": None}})
    assert fix is None
    assert "rc=255" in note and "No route to host" in note


def test_ssh_timeout_is_reported_not_raised(monkeypatch):
    def _run(cmd, **kwargs):
        raise gps.subprocess.TimeoutExpired(cmd, 10.0)
    monkeypatch.setattr(gps.subprocess, "run", _run)
    fix, note = read_boat_log_fix({"boat_log": {
        "ssh_host": "boat-b@boat-b", "log_glob": "/x/*.csv",
        "ssh_options": []}})
    assert fix is None and "timed out" in note


def test_ssh_host_with_shell_metacharacters_is_refused(monkeypatch):
    _fake_ssh(monkeypatch, stdout="")
    fix, note = read_boat_log_fix({"boat_log": {
        "ssh_host": "boat-b@boat-b; rm -rf /", "log_glob": "/x/*.csv",
        "ssh_options": []}})
    assert fix is None and "unsafe" in note


# --- the poller -------------------------------------------------------------

def _log_cfg(tmp_path, **over):
    cfg = {"boat_log": {"ssh_host": "", "max_age_s": None,
                        "log_glob": str(tmp_path / "boat_log_*.csv")},
           "refresh_interval_s": 1800.0, "stale_after_s": 3600.0}
    cfg.update(over)
    return cfg


def test_poller_publishes_and_stamps_the_read_time(tmp_path):
    _write_log(tmp_path / "boat_log_20260824_120000.csv",
               [_row(_iso(-5), "47.7", "9.2", "4")])
    r = GPSReaderBoatLog(_log_cfg(tmp_path))
    fix = r.poll_once()
    assert fix.source == "boat_log" and fix.timestamp is not None
    assert r.reads_ok == 1 and r.get() is not None


def test_poller_failure_keeps_the_reason(tmp_path):
    r = GPSReaderBoatLog(_log_cfg(tmp_path))
    assert r.poll_once() is None
    assert r.reads_fail == 1 and "no boat log found" in r.last_note


def test_poller_get_expires_a_stale_fix(tmp_path):
    """Half an hour between polls is fine; a fix that has outlived
    stale_after_s means the source has gone quiet and must stop answering."""
    _write_log(tmp_path / "boat_log_20260824_120000.csv",
               [_row(_iso(-5), "47.7", "9.2", "4")])
    r = GPSReaderBoatLog(_log_cfg(tmp_path, stale_after_s=-1.0))
    assert r.poll_once() is not None
    assert r.get() is None


# --- the chain capture actually holds ---------------------------------------

def _chain(tmp_path, sources, **over):
    cfg = {"enabled": True, "sources": sources,
           "boat_log": {"ssh_host": "", "max_age_s": None,
                        "log_glob": str(tmp_path / "boat_log_*.csv")},
           "ble_char_uuid": "", "packet_fmt": "csv",
           "fallback": {"lat": 47.695639, "lon": 9.193917, "note": "launch"},
           "refresh_interval_s": 1800.0, "stale_after_s": 3600.0}
    cfg.update(over)
    return GPSReader(cfg)


def test_chain_prefers_the_boat_log_over_the_fallback(tmp_path):
    _write_log(tmp_path / "boat_log_20260824_120000.csv",
               [_row(_iso(-5), "47.111", "9.111", "4")])
    r = _chain(tmp_path, ["boat_log", "fallback"])
    r.boat_log.poll_once()
    assert r.get().source == "boat_log"


def test_chain_falls_back_when_the_boat_is_unreachable(tmp_path):
    r = _chain(tmp_path, ["boat_log", "fallback"])
    r.boat_log.poll_once()
    fix = r.get()
    assert fix.source == "fallback"
    assert (fix.lat_deg, fix.lon_deg) == pytest.approx((47.695639, 9.193917))


def test_chain_live_only_never_returns_the_fallback(tmp_path):
    r = _chain(tmp_path, ["boat_log", "fallback"])
    assert r.get() is not None
    assert r.get(live_only=True) is None


def test_wait_for_fix_does_not_let_the_fallback_win_the_race(tmp_path):
    """The fallback is a constant and so is available on the first instant,
    while the log read is an SSH round trip. A plain 'first non-None wins' loop
    would therefore record the fixed position on every mission start."""
    log = tmp_path / "boat_log_20260824_120000.csv"
    r = _chain(tmp_path, ["boat_log", "fallback"])

    polls = {"n": 0}
    real_poll = r.boat_log.poll_once

    def _slow_poll():
        polls["n"] += 1
        if polls["n"] >= 2:                      # the log appears mid-wait
            _write_log(log, [_row(_iso(-5), "47.222", "9.222", "4")])
        return real_poll()

    r.boat_log.poll_once = _slow_poll
    r.boat_log._interval_s = 0.05
    r.boat_log.start()
    try:
        fix = r.wait_for_fix(timeout_s=5.0)
    finally:
        r.close()
    assert fix.source == "boat_log" and fix.lat_deg == pytest.approx(47.222)


def test_wait_for_fix_gives_up_to_the_fallback(tmp_path):
    r = _chain(tmp_path, ["boat_log", "fallback"])
    fix = r.wait_for_fix(timeout_s=0.3)
    assert fix.source == "fallback"


def test_wait_for_fix_returns_none_with_no_fallback_configured(tmp_path):
    r = _chain(tmp_path, ["boat_log"])
    assert r.wait_for_fix(timeout_s=0.3) is None


def test_chain_sources_list_gates_which_readers_exist(tmp_path):
    r = _chain(tmp_path, ["boat_log"])
    assert r.boat_log is not None and r.ble is None and r.fallback is None


def test_chain_disabled_reports_nothing(tmp_path):
    r = _chain(tmp_path, ["boat_log", "fallback"], enabled=False)
    assert r.get() is None and r.source_names == []
    assert "disabled" in r.describe()


def test_take_new_writes_one_row_per_change_not_per_poll(tmp_path):
    """The capture loop calls this at frame rate; it must yield a row only when
    the position (or the source) actually changed."""
    log = tmp_path / "boat_log_20260824_120000.csv"
    _write_log(log, [_row(_iso(-5), "47.111", "9.111", "4")])
    r = _chain(tmp_path, ["boat_log", "fallback"])
    r.boat_log.poll_once()
    assert r.take_new() is not None
    assert r.take_new() is None                  # same fix: no second row
    _write_log(log, [_row(_iso(-1), "47.222", "9.222", "4")])
    r.boat_log.poll_once()
    assert r.take_new().lat_deg == pytest.approx(47.222)


def test_take_new_treats_a_failover_as_a_change(tmp_path):
    """Losing boat1 and dropping to the fixed position is itself worth a row:
    otherwise the sidecar shows the last live fix holding steady for hours."""
    _write_log(tmp_path / "boat_log_20260824_120000.csv",
               [_row(_iso(-5), "47.111", "9.111", "4")])
    r = _chain(tmp_path, ["boat_log", "fallback"], stale_after_s=-1.0)
    r.boat_log.poll_once()
    assert r.take_new().source == "fallback"     # the live fix is already stale


def test_describe_names_the_host_and_the_reason(tmp_path, monkeypatch):
    """What an operator reads when GPS is the broken piece."""
    _fake_ssh(monkeypatch, returncode=3)          # connected, no log there
    r = _chain(tmp_path, ["boat_log", "fallback"],
               boat_log={"ssh_host": "boat-b@boat-b", "max_age_s": None,
                         "ssh_options": [],
                         "log_glob": str(tmp_path / "nope_*.csv")})
    r.boat_log.poll_once()
    text = r.describe()
    assert "boat-b@boat-b" in text and "no boat log found" in text
    assert "fallback: +47.695639" in text


def test_fallback_without_coordinates_is_not_invented(tmp_path):
    r = _chain(tmp_path, ["fallback"], fallback={"lat": None, "lon": None})
    assert r.fallback is None and r.get() is None
    assert "no lat/lon set" in r.describe()


def test_ble_source_without_a_uuid_is_reported_not_silently_started(tmp_path):
    """The BLE bridge is a scaffold with no confirmed characteristic. It stays
    listed as a source, but must not spawn a thread whose whole life is to
    print 'no uuid set' on every capture boot."""
    r = _chain(tmp_path, ["boat_log", "ble", "fallback"])
    assert r.ble is None
    assert "no ble_char_uuid set" in r.describe()


def test_ble_source_is_built_once_a_uuid_is_configured(tmp_path):
    r = _chain(tmp_path, ["ble"], ble_char_uuid="0000-abcd")
    assert r.ble is not None
    assert "not connected" in r.describe()
