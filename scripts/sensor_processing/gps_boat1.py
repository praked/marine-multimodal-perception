"""boat1 GPS: position + offline sun geometry for capture metadata.

The GPS is an ArduSimple simpleRTK2B (u-blox ZED-F9P, GPS L1/L2, RTK rover,
sub-cm fix) wired into **boat1**, the boat's central Pi running the autopilot.
SensorBox reads `(lat, lon)` at capture init and on a slow cadence
(~30 min); this is *metadata for training* (sun elevation/azimuth, and weather
correlation done offline by timestamp), NOT navigation, so a coarse, occasional
fix is plenty.

Three sources, tried in the order given by `sources` in configs/gps.yaml:

  1. **`boat_log` (default).** boat1's autopilot already logs every fix to
     `<boatv1>/logs/boat_log_<stamp>.csv` (`utils/data_logger.py` on the
     `auto-work` branch: columns `ts_iso,gps_lat,gps_lon,gps_ts,...,fix_q`, one
     row per ~10 Hz control cycle, a new file per run). Reading the newest row
     out of that file needs NOTHING enabled or changed on the boatv1 side,
     which is why it replaced the BLE plan. Reached either over SSH
     (`ssh_host`, the deployed case: the two Pis are separate machines) or as a
     local path when something else — a mount, an rsync — already puts the logs
     on this box.
  2. **`ble`.** The original plan: boat1 bridges the position over a BLE
     characteristic exactly like the IMU used to. Still a SCAFFOLD — the
     characteristic UUID and packet layout are boat1-side and must be confirmed
     with `--scan` / `--sniff` — and now only a fallback for when SSH is down.
  3. **`fallback`.** A fixed lake position, so a clip always carries a
     defensible sun geometry. Reported as `source=fallback` everywhere, and the
     smoke test says so loudly: it is a last resort, not a measurement.

`sun_position` and `parse_gps_packet` are final and unit-tested (offline).

Run:
    python -m scripts.sensor_processing.gps_boat1 --check      # resolve one fix
    python -m scripts.sensor_processing.gps_boat1 --scan       # find boat1 (BLE)
    python -m scripts.sensor_processing.gps_boat1 --monitor    # live fix + sun
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import math
import os
import re
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

GPS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "gps.yaml"


# ---------------------------------------------------------------------------
# Fix value type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPSFix:
    """A single position fix. `fix_quality` follows the NMEA GGA convention
    (0 = no fix, 1 = GPS, 2 = DGPS, 4 = RTK fixed, 5 = RTK float) or None when
    the source doesn't report it."""
    lat_deg: float
    lon_deg: float
    fix_quality: int | None = None
    timestamp: float | None = None      # time.monotonic() at receipt
    source: str = "boat1-gps"
    # The SOURCE's own timestamp, not ours: the `ts_iso` of the boat-log row a
    # file-sourced fix came from. `timestamp` says when *we* read it, which for
    # a log file is no evidence at all that the fix is recent — the autopilot
    # may have stopped hours ago and left its last log behind. Keeping both is
    # what makes a stale-log check possible (and what a clip records, so the
    # staleness is still visible offline).
    fix_time_utc: datetime | None = None


# ---------------------------------------------------------------------------
# Sun geometry (offline; final + tested): NOAA solar-position algorithm
# ---------------------------------------------------------------------------

def _julian_day(dt: datetime) -> float:
    """UTC datetime -> Julian Day (fractional)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    y, m = dt.year, dt.month
    day = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6)
                               / 60.0) / 60.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return (int(365.25 * (y + 4716)) + int(30.6001 * (m + 1))
            + day + b - 1524.5)


def sun_position(lat_deg: float, lon_deg: float, when_utc: datetime
                 ) -> tuple[float, float]:
    """Solar (elevation, azimuth) in degrees for a lat/lon and UTC datetime.

    NOAA solar-position algorithm (good to ~0.01 deg: plenty for scheduling).
    Elevation is negative below the horizon; azimuth is measured clockwise from
    true north (0 = N, 90 = E, 180 = S, 270 = W). Feeds the day/night thermal
    role scheduler (capture_runbook §3.4).
    """
    r = math.radians
    jd = _julian_day(when_utc)
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + T * (36000.76983 + T * 0.0003032)) % 360.0
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    Mr = r(M)
    C = ((1.914602 - T * (0.004817 + 0.000014 * T)) * math.sin(Mr)
         + (0.019993 - 0.000101 * T) * math.sin(2 * Mr)
         + 0.000289 * math.sin(3 * Mr))
    true_long = L0 + C
    omega = 125.04 - 1934.136 * T
    lam = true_long - 0.00569 - 0.00478 * math.sin(r(omega))
    eps0 = 23.0 + (26.0 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813)))
                   / 60.0) / 60.0
    eps = eps0 + 0.00256 * math.cos(r(omega))
    decl = math.asin(math.sin(r(eps)) * math.sin(r(lam)))
    y = math.tan(r(eps / 2.0)) ** 2
    L0r = r(L0)
    eot = 4.0 * math.degrees(
        y * math.sin(2 * L0r) - 2 * e * math.sin(Mr)
        + 4 * e * y * math.sin(Mr) * math.cos(2 * L0r)
        - 0.5 * y * y * math.sin(4 * L0r)
        - 1.25 * e * e * math.sin(2 * Mr))
    u = when_utc.astimezone(timezone.utc) if when_utc.tzinfo else when_utc
    minutes = (u.hour * 60 + u.minute + u.second / 60.0
               + u.microsecond / 6e7)
    tst = (minutes + eot + 4.0 * lon_deg) % 1440.0
    ha = tst / 4.0 - 180.0
    if ha < -180.0:
        ha += 360.0
    har, latr = r(ha), r(lat_deg)
    cos_zen = (math.sin(latr) * math.sin(decl)
               + math.cos(latr) * math.cos(decl) * math.cos(har))
    cos_zen = max(-1.0, min(1.0, cos_zen))
    zen = math.acos(cos_zen)
    elevation = 90.0 - math.degrees(zen)
    denom = math.cos(latr) * math.sin(zen)
    if abs(denom) > 1e-9:
        ca = (math.sin(latr) * math.cos(zen) - math.sin(decl)) / denom
        az = math.degrees(math.acos(max(-1.0, min(1.0, ca))))
        azimuth = (az + 180.0) % 360.0 if ha > 0 else (540.0 - az) % 360.0
    else:
        azimuth = 180.0 if lat_deg >= 0 else 0.0
    return elevation, azimuth


def current_sun_elevation(reader_or_fix: Any,
                          when_utc: datetime | None = None) -> float | None:
    """Solar elevation (deg) at a GPS reader's current position, now.

    The live-capture counterpart of the replay-side computation in
    scripts/fusion_model/build_features.py: feeds
    ObstacleDetectionPipeline.set_sun_elevation for the thermal profile
    scheduler (detection.yaml thermal.profiles). `when_utc` defaults to
    the system clock (RTC-backed on the box, so correct offline); pass a
    value only in tests. Accepts a GPSReader (anything with .get() ->
    GPSFix|None) or a bare GPSFix. Returns None when no position is
    available — the scheduler then honestly falls back to the day
    profile. A `fallback`-source fix still yields an elevation: anywhere
    on the lake agrees to well under a degree.
    """
    fix = (reader_or_fix.get() if hasattr(reader_or_fix, "get")
           else reader_or_fix)
    if fix is None:
        return None
    when = when_utc if when_utc is not None else datetime.now(timezone.utc)
    elevation, _ = sun_position(fix.lat_deg, fix.lon_deg, when)
    return elevation


# ---------------------------------------------------------------------------
# Packet parsing (final + tested)
# ---------------------------------------------------------------------------

def _nmea_to_deg(field: str, hemi: str) -> float:
    """NMEA ddmm.mmmm / dddmm.mmmm + hemisphere -> signed decimal degrees."""
    if not field:
        raise ValueError("empty coordinate")
    v = float(field)
    deg = int(v // 100)
    minutes = v - deg * 100
    dec = deg + minutes / 60.0
    if hemi in ("S", "W"):
        dec = -dec
    return dec


def parse_gps_packet(data: bytes, fmt: str = "nmea") -> GPSFix | None:
    """Decode one boat1 GPS packet into a GPSFix. CONFIRM `fmt` against boat1.

      - "nmea": an ASCII NMEA sentence (GGA carries fix quality; RMC otherwise).
      - "latlon_f64_le": 16 bytes = lat, lon float64 LE (fix unknown).
      - "csv": ASCII "lat,lon[,fix]" in decimal degrees.

    Returns None on a malformed/short/unfixed packet so the reader resyncs.
    """
    try:
        if fmt == "latlon_f64_le":
            if len(data) < 16:
                return None
            lat, lon = struct.unpack_from("<dd", data, 0)
            return GPSFix(lat_deg=lat, lon_deg=lon)
        if fmt == "csv":
            parts = data.decode("ascii", "ignore").strip().split(",")
            if len(parts) < 2:
                return None
            fix = int(parts[2]) if len(parts) > 2 and parts[2] != "" else None
            return GPSFix(lat_deg=float(parts[0]), lon_deg=float(parts[1]),
                          fix_quality=fix)
        if fmt == "nmea":
            line = data.decode("ascii", "ignore").strip()
            for sentence in line.splitlines():
                body = sentence.split("*", 1)[0]
                f = body.split(",")
                tag = f[0][-3:] if f else ""
                if tag == "GGA" and len(f) >= 7 and f[2] and f[4]:
                    q = int(f[6]) if f[6] else 0
                    if q == 0:
                        continue
                    return GPSFix(_nmea_to_deg(f[2], f[3]),
                                  _nmea_to_deg(f[4], f[5]), fix_quality=q)
                if tag == "RMC" and len(f) >= 7 and f[2] == "A" and f[3] and f[5]:
                    return GPSFix(_nmea_to_deg(f[3], f[4]),
                                  _nmea_to_deg(f[5], f[6]))
            return None
    except (ValueError, struct.error, IndexError):
        return None
    raise ValueError(f"unknown gps packet fmt: {fmt!r}")


# ---------------------------------------------------------------------------
# boat1's own log file: the DEFAULT source (2026-08-24)
# ---------------------------------------------------------------------------
#
# boat1's autopilot writes `<boatv1>/logs/boat_log_<YYYYmmdd>_<HHMMSS>.csv`
# (utils/data_logger.py, branch auto-work) with a header row and then one row
# per control cycle (~10 Hz). The columns this cares about:
#
#     ts_iso    when the row was written, UTC ISO-8601   -> freshness
#     gps_lat   decimal degrees, EMPTY until a real fix  -> position
#     gps_lon   decimal degrees, EMPTY until a real fix
#     fix_q     GGA fix quality (4 = RTK fixed)          -> quality
#
# Everything else in the row is autopilot state we have no use for. A new file
# is opened per autopilot run, so "the newest fix" means: newest file by mtime,
# then the last row in it that actually carries coordinates. The boat's own
# GPSHandler only fills those cells once it has a genuine fix (it guards
# against empty NMEA fields coercing to 0.0), so an empty cell honestly means
# "no fix yet" and is skipped rather than read as a position.

BOAT_LOG_TIME_COL = "ts_iso"
BOAT_LOG_LAT_COL = "gps_lat"
BOAT_LOG_LON_COL = "gps_lon"
BOAT_LOG_FIX_COL = "fix_q"

# Glob patterns are pasted UNQUOTED into a remote shell command (they have to
# be, or the remote shell will not expand them), so they are restricted to
# path/glob characters. Without this a config value would be a command.
_SAFE_GLOB = re.compile(r"^[A-Za-z0-9._/~*?\[\]-]+$")
_SAFE_SSH_HOST = re.compile(r"^[A-Za-z0-9._@-]+$")


def _check_globs(patterns: Any) -> list[str]:
    """Validate + normalise `log_glob` (a string or a list) into a list."""
    if isinstance(patterns, str):
        patterns = [patterns]
    out: list[str] = []
    for raw in patterns or []:
        pat = str(raw).strip()
        if not pat:
            continue
        if not _SAFE_GLOB.match(pat):
            raise ValueError(f"unsafe gps log_glob pattern: {pat!r}")
        out.append(pat)
    return out


def _local_log_payload(patterns: list[str], tail_bytes: int) -> str:
    """Header + tail of the newest local log matching `patterns`.

    Returns "" when nothing matches. The payload format is shared with the SSH
    path so one parser serves both: a `#<path>` line, the header line, then the
    tail rows.
    """
    matches: list[str] = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.expanduser(pat)))
    if not matches:
        return ""
    path = max(matches, key=os.path.getmtime)
    with open(path, "rb") as fh:
        header = fh.readline()
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - int(tail_bytes)))
        tail = fh.read()
    # Drop through the first newline, exactly as the remote `tail -c | tail -n
    # +2` does: on a big file that discards a half-read row (which could
    # otherwise still parse into plausible-looking columns), and on a small one
    # it discards the header, which is sent separately anyway.
    body = tail.decode("utf-8", "replace")
    body = body.split("\n", 1)[1] if "\n" in body else ""
    head = header.decode("utf-8", "replace")
    if not head.endswith("\n"):
        head += "\n"
    return f"#{path}\n{head}{body}"


def _ssh_log_payload(host: str, patterns: list[str], tail_bytes: int,
                     timeout_s: float, options: list[str]) -> str:
    """Same payload, fetched from `host` in ONE ssh round trip.

    One command rather than three (find the file, read its header, read its
    tail) both halves the latency of a link that may be a marginal boat WiFi
    and removes the race where the autopilot rotates its log between calls.
    """
    if not _SAFE_SSH_HOST.match(host):
        raise ValueError(f"unsafe gps ssh_host: {host!r}")
    remote = (
        f"f=$(ls -1t {' '.join(patterns)} 2>/dev/null | head -n1); "
        f'[ -n "$f" ] || exit 3; '
        f"printf '#%s\\n' \"$f\"; "
        f'head -n1 "$f"; '
        f'tail -c {int(tail_bytes)} "$f" | tail -n +2'
    )
    cmd = ["ssh"]
    for opt in options:
        cmd += ["-o", str(opt)]
    cmd += [host, remote]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=float(timeout_s))
    if proc.returncode == 3:
        return ""                       # connected fine; no log file there
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(f"ssh {host} rc={proc.returncode}: "
                           f"{err[-1] if err else 'no stderr'}")
    return proc.stdout


def parse_boat_log_payload(text: str, max_age_s: float | None = None,
                           now_utc: datetime | None = None
                           ) -> tuple[GPSFix | None, str | None, str]:
    """Newest usable fix in a boat-log payload -> `(fix, log_path, note)`.

    Walks the rows backwards and takes the first that carries coordinates, so
    the answer is the freshest fix even though the tail is mostly fixless rows.
    `note` is a one-line human-readable status; it is what the smoke test
    prints, so a failure says *why* rather than just "no GPS".

    `max_age_s` rejects a fix whose own `ts_iso` is older than that (default:
    no age limit). A log file sitting on disk proves nothing about whether the
    autopilot is still running.
    """
    lines = text.splitlines()
    log_path = None
    if lines and lines[0].startswith("#"):
        log_path = lines[0][1:].strip()
        lines = lines[1:]
    if not lines:
        return None, log_path, "no boat log found"
    try:
        header = next(csv.reader([lines[0]]))
    except (csv.Error, StopIteration):
        return None, log_path, "unreadable log header"
    idx = {name.strip(): i for i, name in enumerate(header)}
    if BOAT_LOG_LAT_COL not in idx or BOAT_LOG_LON_COL not in idx:
        return None, log_path, (f"log has no {BOAT_LOG_LAT_COL}/"
                                f"{BOAT_LOG_LON_COL} columns")
    i_lat, i_lon = idx[BOAT_LOG_LAT_COL], idx[BOAT_LOG_LON_COL]
    i_fix = idx.get(BOAT_LOG_FIX_COL)
    i_time = idx.get(BOAT_LOG_TIME_COL)

    rows = lines[1:]
    if not rows:
        return None, log_path, "log is header-only (autopilot just started?)"
    for raw in reversed(rows):
        if not raw.strip():
            continue
        try:
            row = next(csv.reader([raw]))
        except (csv.Error, StopIteration):
            continue
        if len(row) <= max(i_lat, i_lon):
            continue
        if not row[i_lat].strip() or not row[i_lon].strip():
            continue                    # no fix on this cycle
        try:
            lat = float(row[i_lat])
            lon = float(row[i_lon])
        except ValueError:
            continue
        # Null island: a fixless receiver whose empty NMEA fields coerced to
        # 0.0 somewhere upstream. Never a position this boat sails from.
        if lat == 0.0 and lon == 0.0:
            continue
        fix_q = None
        if i_fix is not None and len(row) > i_fix and row[i_fix].strip():
            try:
                fix_q = int(float(row[i_fix]))
            except ValueError:
                fix_q = None
        row_time = None
        if i_time is not None and len(row) > i_time and row[i_time].strip():
            try:
                row_time = datetime.fromisoformat(row[i_time].strip())
                if row_time.tzinfo is None:
                    row_time = row_time.replace(tzinfo=timezone.utc)
            except ValueError:
                row_time = None
        if max_age_s is not None and row_time is not None:
            now = now_utc or datetime.now(timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            age = (now - row_time).total_seconds()
            if age > float(max_age_s):
                return None, log_path, (
                    f"newest fix is {age / 3600.0:.1f} h old "
                    f"(> max_age_s {float(max_age_s) / 3600.0:.1f} h): "
                    f"autopilot stopped, or a clock skew between the boxes")
        quality = "" if fix_q is None else f", fix_q={fix_q}"
        return (GPSFix(lat_deg=lat, lon_deg=lon, fix_quality=fix_q,
                       source="boat_log", fix_time_utc=row_time),
                log_path,
                f"lat={lat:+.6f} lon={lon:+.6f}{quality}")
    return None, log_path, "log has rows but none carries a fix (no GPS lock)"


def read_boat_log_fix(config: dict[str, Any] | None = None
                      ) -> tuple[GPSFix | None, str]:
    """One-shot read of boat1's log -> `(fix, note)`. Never raises.

    `note` always explains the outcome (an ssh failure, a stale log, the fix
    itself), because "GPS: nothing" in a field smoke test is not actionable
    and this is the thing most likely to be broken on deploy day.
    """
    cfg = dict((config or load_gps_config()).get("boat_log") or {})
    try:
        patterns = _check_globs(cfg.get("log_glob"))
    except ValueError as exc:
        return None, f"config error: {exc}"
    if not patterns:
        return None, "no log_glob configured"
    host = str(cfg.get("ssh_host", "")).strip()
    tail_bytes = int(cfg.get("tail_bytes", 65536))
    max_age_s = cfg.get("max_age_s")
    where = f"{host}:{patterns[0]}" if host else patterns[0]
    try:
        if host:
            text = _ssh_log_payload(
                host, patterns, tail_bytes,
                float(cfg.get("ssh_timeout_s", 10.0)),
                list(cfg.get("ssh_options") or []))
        else:
            text = _local_log_payload(patterns, tail_bytes)
    except subprocess.TimeoutExpired:
        return None, f"{where}: ssh timed out (boat1 unreachable?)"
    except FileNotFoundError:
        return None, "ssh not installed on this box"
    except (RuntimeError, ValueError, OSError) as exc:
        return None, f"{where}: {exc}"
    if not text.strip():
        return None, f"{where}: no boat log found"
    fix, log_path, note = parse_boat_log_payload(
        text, None if max_age_s in (None, "") else float(max_age_s))
    tag = f"{host}:{log_path}" if host and log_path else (log_path or where)
    return fix, f"{tag}: {note}"


class GPSReaderBoatLog(threading.Thread):
    """Poll boat1's autopilot log on a slow cadence, holding the latest fix.

    Deliberately a poller, not a tail: the point is a position every
    `refresh_interval_s` (~30 min) for training metadata, so one cheap round
    trip per half hour beats holding a session open across a boat WiFi that
    comes and goes. Reads once immediately on start so capture init has a fix.

    Never raises into the capture loop: every failure becomes `last_note` and
    `get()` returning the previous fix (until it goes stale) or None.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(daemon=True, name="gps-reader-boatlog")
        self.config = config or load_gps_config()
        self._interval_s = float(self.config.get("refresh_interval_s", 1800.0))
        self._stale_after_s = float(self.config.get("stale_after_s", 3600.0))
        self._lock = threading.Lock()
        self._latest: GPSFix | None = None
        self._stop_evt = threading.Event()
        self.reads_ok = 0
        self.reads_fail = 0
        self.last_note = "not polled yet"

    def poll_once(self) -> GPSFix | None:
        """Read the log now, publish on success. Also the CLI's one-shot path."""
        fix, note = read_boat_log_fix(self.config)
        self.last_note = note
        if fix is None:
            self.reads_fail += 1
            return None
        self.reads_ok += 1
        stamped = GPSFix(fix.lat_deg, fix.lon_deg, fix.fix_quality,
                         timestamp=time.monotonic(), source="boat_log",
                         fix_time_utc=fix.fix_time_utc)
        with self._lock:
            self._latest = stamped
        return stamped

    def get(self) -> GPSFix | None:
        """Latest fix, or None when it has gone stale / none ever arrived."""
        with self._lock:
            fix = self._latest
        if fix is None:
            return None
        if (fix.timestamp is not None
                and (time.monotonic() - fix.timestamp) > self._stale_after_s):
            return None
        return fix

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self.poll_once()
            except Exception as exc:            # noqa: BLE001 - never kill capture
                self.reads_fail += 1
                self.last_note = f"poll error: {exc!r}"
            # Sleep in <=1 s slices so close() joins promptly instead of
            # hanging for the rest of a 30 min interval.
            remaining = self._interval_s
            while remaining > 0 and not self._stop_evt.is_set():
                step = min(1.0, remaining)
                self._stop_evt.wait(step)
                remaining -= step

    def stop(self) -> None:
        self._stop_evt.set()

    def close(self) -> None:
        self.stop()
        self.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_gps_config(path: str | Path = GPS_CONFIG_PATH) -> dict[str, Any]:
    """Read configs/gps.yaml over these defaults.

    Defaults are DISABLED: an absent config file means the box was never set up
    for GPS, and silently shelling out to an unconfigured host on every capture
    is not a sensible thing to do by default. The shipped configs/gps.yaml is
    the thing that turns it on.

    The `boat_log`, `ble` and `fallback` sub-blocks merge key-by-key, so a
    config that overrides one key of a block keeps the rest of that block's
    defaults instead of blanking it.
    """
    defaults: dict[str, Any] = {
        "enabled": False,
        # Tried in order; the first that yields a fix wins. Dropping a name
        # disables that source entirely (e.g. `sources: [boat_log]` for a box
        # that must never silently fall back to a fixed position).
        "sources": ["boat_log", "ble", "fallback"],
        # --- 1. boat1's autopilot log (default source) ---
        "boat_log": {
            # Empty ssh_host = read `log_glob` as a LOCAL path (a mount, or an
            # rsync that already put the logs on this box).
            "ssh_host": "",
            "log_glob": [],
            "ssh_timeout_s": 10.0,
            "ssh_options": ["BatchMode=yes", "ConnectTimeout=5",
                            "StrictHostKeyChecking=accept-new"],
            "tail_bytes": 65536,
            "max_age_s": 3600.0,
        },
        # --- 2. BLE bridge (scaffold; flat keys, historical) ---
        "ble_name": "boat1-gps",
        "ble_address": "",
        "ble_char_uuid": "",
        "packet_fmt": "nmea",
        # --- 3. fixed fallback ---
        "fallback": {"lat": None, "lon": None, "note": ""},
        # GPS here is slow metadata: read at init, then on this cadence.
        "refresh_interval_s": 1800.0,   # ~30 min cadence (metadata, not nav)
        "stale_after_s": 3600.0,
    }
    if not Path(path).exists():
        return defaults
    with open(path) as fh:
        loaded = yaml.safe_load(fh) or {}
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(defaults.get(key), dict):
            merged = dict(defaults[key])
            merged.update(value)
            defaults[key] = merged
        else:
            defaults[key] = value
    return defaults


# ---------------------------------------------------------------------------
# BLE reader: SCAFFOLD (mirrors imu_bno085's BLE reader / continuous_capture
# IMUReader; confirm UUIDs + packet_fmt on land before trusting live fixes)
# ---------------------------------------------------------------------------

class GPSReaderBLE(threading.Thread):
    """Hold the latest GPSFix from boat1 over BLE; `get()` is what capture polls.

    Non-blocking + self-healing exactly like the BLE IMU reader: if `bleak` is
    missing or boat1 isn't advertising, `get()` simply returns None and capture
    proceeds with no GPS metadata. Auto-reconnects with backoff. Reads on a slow
    cadence (`refresh_interval_s`): GPS here is position metadata, not nav.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(daemon=True, name="gps-reader")
        self.config = config or load_gps_config()
        self._name = str(self.config.get("ble_name", "boat1-gps"))
        self._address = str(self.config.get("ble_address", ""))
        self._char = str(self.config.get("ble_char_uuid", ""))
        self._fmt = str(self.config.get("packet_fmt", "nmea"))
        self._stale_after_s = float(self.config.get("stale_after_s", 3600.0))
        self._lock = threading.Lock()
        self._latest: GPSFix | None = None
        # NB not `_stop`: Thread has an internal `_stop()` METHOD on
        # Python <= 3.12 that join() calls: shadowing it with an Event
        # makes join() raise TypeError.
        self._stop_evt = threading.Event()
        self.connected = False
        self.reads_ok = 0

    def _publish(self, fix: GPSFix) -> None:
        fix = GPSFix(fix.lat_deg, fix.lon_deg, fix.fix_quality,
                     timestamp=time.monotonic(), source="boat1-gps")
        self.reads_ok += 1
        with self._lock:
            self._latest = fix

    def get(self) -> GPSFix | None:
        """Latest fix, or None if stale / never received."""
        with self._lock:
            fix = self._latest
        if fix is None:
            return None
        if (fix.timestamp is not None
                and (time.monotonic() - fix.timestamp) > self._stale_after_s):
            return None
        return fix

    def run(self) -> None:
        try:
            asyncio.run(self._loop())
        except Exception as exc:      # noqa: BLE001 - surface, never kill capture
            print(f"[gps] reader stopped: {exc!r}")
            self.connected = False

    async def _loop(self) -> None:
        try:
            from bleak import BleakClient, BleakScanner  # type: ignore
        except ImportError:
            print("[gps] bleak missing: BLE source off "
                  "(other sources unaffected)")
            return
        if not self._char:
            print("[gps] no ble_char_uuid set (confirm against boat1): "
                  "BLE source off")
            return
        backoff = 2.0
        while not self._stop_evt.is_set():
            try:
                addr = self._address
                if not addr:
                    dev = await BleakScanner.find_device_by_name(
                        self._name, timeout=10.0)
                    if dev is None:
                        await asyncio.sleep(min(backoff, 30.0))
                        backoff = min(backoff * 1.5, 30.0)
                        continue
                    addr = dev.address
                async with BleakClient(addr) as client:
                    self.connected = True
                    backoff = 2.0
                    print(f"[gps] connected {addr} ({self._name})")
                    while not self._stop_evt.is_set() and client.is_connected:
                        data = await client.read_gatt_char(self._char)
                        fix = parse_gps_packet(bytes(data), self._fmt)
                        if fix is not None:
                            self._publish(fix)
                        # Sleep in <=1 s chunks: close() joins with a 3 s
                        # timeout, and a monolithic ~30 min sleep would keep
                        # the thread hanging long past it.
                        remaining = float(
                            self.config.get("refresh_interval_s", 1800.0))
                        while remaining > 0 and not self._stop_evt.is_set():
                            step = min(1.0, remaining)
                            await asyncio.sleep(step)
                            remaining -= step
            except Exception as exc:  # noqa: BLE001
                self.connected = False
                print(f"[gps] session error: {exc!r}")
                await asyncio.sleep(min(backoff, 30.0))
                backoff = min(backoff * 1.5, 30.0)

    def stop(self) -> None:
        self._stop_evt.set()

    def close(self) -> None:
        self.stop()
        self.join(timeout=3.0)


# ---------------------------------------------------------------------------
# The reader capture actually uses: the configured sources, in order
# ---------------------------------------------------------------------------

class GPSReader:
    """Resolve a position from the configured sources, best first.

    This is what `continuous_capture` holds: `start()` once, `get()` whenever a
    row is written, `close()` at shutdown. Each source runs (or doesn't) on its
    own terms — the log poller and the BLE reader are independent background
    threads, the fallback is a constant — and `get()` simply returns the first
    one with something to say. So a boat1 that goes off the air mid-mission
    degrades from a live fix to the previous one, to BLE, to the fixed
    fallback, and every row records WHICH, so the degradation is visible in the
    data rather than inferred later.

    Disabled (`enabled: false`, or an empty `sources`) is a first-class state:
    `get()` returns None and capture writes no GPS rows at all.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_gps_config()
        self.enabled = bool(self.config.get("enabled", False))
        names = [str(s).strip().lower()
                 for s in (self.config.get("sources") or [])]
        self.source_names = names if self.enabled else []
        self.boat_log = (GPSReaderBoatLog(self.config)
                         if "boat_log" in self.source_names else None)
        # Only built when it can actually do something: the BLE bridge is a
        # scaffold with no confirmed characteristic UUID, and starting a thread
        # whose entire life is to print "no uuid set" on every capture boot is
        # noise. `describe()` still reports the source as configured-but-unset.
        self.ble = (GPSReaderBLE(self.config)
                    if ("ble" in self.source_names
                        and self.config.get("ble_char_uuid")) else None)
        self.fallback = None
        if "fallback" in self.source_names:
            fb = self.config.get("fallback") or {}
            lat, lon = fb.get("lat"), fb.get("lon")
            if lat is not None and lon is not None:
                self.fallback = GPSFix(float(lat), float(lon),
                                       source="fallback")
        self._taken_key: tuple | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self.boat_log is not None:
            self.boat_log.start()
        if self.ble is not None:
            self.ble.start()

    def close(self) -> None:
        for reader in (self.boat_log, self.ble):
            if reader is not None:
                try:
                    reader.close()
                except Exception:       # noqa: BLE001 - shutdown is best-effort
                    pass

    # -- reading ------------------------------------------------------------

    def get(self, live_only: bool = False) -> GPSFix | None:
        """Best available fix right now, or None (disabled / nothing at all).

        `live_only` skips the fixed fallback, which is the difference between
        "what should this row record" (everything) and "has anything actually
        measured a position" (live only).
        """
        for name in self.source_names:
            if name == "boat_log" and self.boat_log is not None:
                fix = self.boat_log.get()
            elif name == "ble" and self.ble is not None:
                fix = self.ble.get()
            elif name == "fallback":
                if live_only:
                    continue
                fix = self.fallback
            else:
                continue
            if fix is not None:
                return fix
        return None

    def wait_for_fix(self, timeout_s: float = 15.0) -> GPSFix | None:
        """Block up to `timeout_s` for the first fix: the capture-INIT read.

        Waits on the LIVE sources only, and drops to the fallback just once
        they have run out of time. The fallback is a constant and is therefore
        available on the first instant of the first poll, so a plain "first
        non-None wins" loop would hand back the fixed position every time and
        the live sources would never get their SSH round trip in -- which is
        the whole point of waiting at all.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            fix = self.get(live_only=True)
            if fix is not None:
                return fix
            if time.monotonic() >= deadline:
                return self.get()       # fallback, or None if none configured
            time.sleep(0.25)

    def take_new(self) -> GPSFix | None:
        """The current fix, but only the first time it is seen.

        Lets the capture loop poll cheaply at frame rate and write a row only
        when the position actually changed (or the source did), instead of
        1800 identical rows per half hour. Source-agnostic on purpose: a
        failover from `boat_log` to `fallback` is itself a change worth a row.
        """
        fix = self.get()
        if fix is None:
            return None
        key = (fix.source, fix.lat_deg, fix.lon_deg, fix.fix_time_utc)
        if key == self._taken_key:
            return None
        self._taken_key = key
        return fix

    # -- diagnostics --------------------------------------------------------

    def describe(self) -> str:
        """One line per configured source, for the smoke test / field log.

        The failure detail lives here (which host, which glob, what ssh said),
        because "no GPS" alone is not something an operator on a pontoon can
        act on.
        """
        if not self.enabled:
            return "gps disabled (configs/gps.yaml enabled: false)"
        if not self.source_names:
            return "gps enabled but no sources configured"
        lines = []
        for name in self.source_names:
            if name == "boat_log" and self.boat_log is not None:
                cfg = self.config.get("boat_log") or {}
                host = str(cfg.get("ssh_host", "")).strip() or "(local path)"
                live = "fix" if self.boat_log.get() is not None else "no fix"
                lines.append(f"boat_log[{host}] {live}: {self.boat_log.last_note}")
            elif name == "ble":
                if self.ble is None:
                    lines.append("ble: no ble_char_uuid set (scaffold: confirm "
                                 "against boat1 before this can work)")
                else:
                    state = "connected" if self.ble.connected else "not connected"
                    lines.append(
                        f"ble[{self.config.get('ble_name', 'boat1-gps')}] "
                        f"{state}, reads={self.ble.reads_ok}")
            elif name == "fallback":
                if self.fallback is None:
                    lines.append("fallback: configured but no lat/lon set")
                else:
                    note = (self.config.get("fallback") or {}).get("note", "")
                    lines.append(
                        f"fallback: {self.fallback.lat_deg:+.6f},"
                        f"{self.fallback.lon_deg:+.6f}"
                        f"{f'  ({note})' if note else ''}")
        return " | ".join(lines)


# ---------------------------------------------------------------------------
# CLI (on-land confirmation + monitor)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="resolve one fix through the configured source chain "
                         "and print where it came from (default)")
    ap.add_argument("--log-only", action="store_true",
                    help="read boat1's log once, bypassing the chain: the "
                         "sharpest test of the SSH/path setup")
    ap.add_argument("--scan", action="store_true", help="list nearby BLE devices")
    ap.add_argument("--monitor", action="store_true",
                    help="stream live fix + sun position")
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args(argv)

    if args.scan:
        try:
            from bleak import BleakScanner
        except ImportError:
            print("need bleak: pip install bleak")
            return 1

        async def _scan():
            for d in await BleakScanner.discover(timeout=args.seconds):
                print(f"  {d.address}  {d.name or '(no name)'}")
        asyncio.run(_scan())
        return 0

    if args.log_only:
        fix, note = read_boat_log_fix()
        print(f"boat_log: {note}")
        return 0 if fix is not None else 1

    if args.monitor:
        reader = GPSReader()
        print(reader.describe())
        reader.start()
        try:
            t0 = time.time()
            while time.time() - t0 < args.seconds if args.seconds else True:
                fix = reader.get()
                if fix is None:
                    print("   ...waiting for a fix", end="\r")
                else:
                    el, az = sun_position(fix.lat_deg, fix.lon_deg,
                                          datetime.now(timezone.utc))
                    print(f"[{fix.source}] lat={fix.lat_deg:+.6f} "
                          f"lon={fix.lon_deg:+.6f} fix={fix.fix_quality}  "
                          f"sun elev={el:+.1f} az={az:.1f}")
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            reader.close()
        return 0

    # Default: one pass through the chain, exactly what capture init does.
    reader = GPSReader()
    reader.start()
    try:
        fix = reader.wait_for_fix(timeout_s=args.seconds)
        print(reader.describe())
        if fix is None:
            print("RESULT: no fix from any configured source")
            return 1
        el, az = sun_position(fix.lat_deg, fix.lon_deg,
                              datetime.now(timezone.utc))
        age = ("" if fix.fix_time_utc is None else
               f"  (fix logged {fix.fix_time_utc.isoformat()})")
        print(f"RESULT: {fix.lat_deg:+.6f}, {fix.lon_deg:+.6f}  "
              f"source={fix.source}  fix_quality={fix.fix_quality}{age}")
        print(f"        sun elevation {el:+.1f} deg, azimuth {az:.1f} deg")
        # A fallback position is not a measurement; never let it read as one.
        return 0 if fix.source != "fallback" else 2
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
