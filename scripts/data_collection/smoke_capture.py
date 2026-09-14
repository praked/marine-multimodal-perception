#!/usr/bin/env python3
"""Short capture smoke test for fisheye + thermal + radar (+ IMU + GPS).

Runs a few seconds of capture through the SAME code paths as
``continuous_capture.py`` (radar telemetry counters, sentinel/heartbeat
rows, monotonic timers, try/finally finalization) into a throwaway
directory, then reads the artefacts back and prints a per-sensor verdict.

It deliberately reuses ``continuous_capture``'s functions rather than
duplicating them, so a pass here means the real service paths work, including
the sensor-enable flags, so this is also the smoke test for a reduced sensor
set.

Run on the Pi with the capture service STOPPED (it owns the sensors):

    sudo systemctl stop asvproject-capture
    cd ~/ASVProject-ObstacleDetection
    python3 -m scripts.data_collection.smoke_capture --seconds 10
    sudo systemctl start asvproject-capture

Fisheye-only box (no radar/thermal fitted, or a supply that can't hold the
full stack): radar and thermal are then reported OFF instead of failing:

    python3 -m scripts.data_collection.smoke_capture --seconds 10 --fisheye-only

(For the *power* question specifically, i.e. does the lighter load stay under
the converter's trip threshold, use ``smoke_fisheye.py``, which additionally
reports the under-voltage/throttle flags before vs after the run.)

GPS is own-boat position metadata read from boat1's autopilot log (see
configs/gps.yaml). It is reported like the IMU -- never fails the smoke, since
a missing position costs training metadata and not a single frame of capture --
unless you ask for it explicitly, which is the check to run on deploy day once
boat1 is up:

    python3 -m scripts.data_collection.smoke_capture --seconds 10 --require-gps

The GPS line names the SOURCE, and a `fallback` source is called out: the
configured fixed position is not a measurement, and it is indistinguishable
from a real fix by looking at the coordinates alone.

Exit code is 0 only if every ENABLED sensor produced valid data.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def _open_chunk(seconds: int, out_dir: str, fisheye_only: bool = False):
    """Set env (read at import time) then drive one short chunk.

    Returns ``(opened, enabled)``: which sensors actually came up, and which
    ones this run was *configured* to use at all. A sensor that is disabled by
    configuration is not a failure; see ``validate``.
    """
    os.environ["ASVPROJECT_CHUNK_SECONDS"] = str(seconds)
    os.environ["ASVPROJECT_CAPTURE_DIR"] = out_dir
    if fisheye_only:
        os.environ["ASVPROJECT_FISHEYE_ONLY"] = "1"

    # Import only after env is set: continuous_capture reads CHUNK_SECONDS
    # and CAPTURE_DIR at module import time.
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts.data_collection import continuous_capture as cc

    enabled = (True, cc.THERMAL_ENABLE, cc.RADAR_ENABLE, cc.IMU_ENABLE,
               cc.GPS_ENABLE)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if cc.FISHEYE_ONLY:
        print("[smoke] FISHEYE-ONLY mode: radar + thermal disabled", flush=True)
    print(f"[smoke] capturing {seconds}s into {out_dir}", flush=True)
    # Record the radar profile beside the data. smoke_capture is the tool
    # field A/Bs are run with, and on 2026-08-19 both halves of the
    # clutterRemoval A/B silently ran the default profile; a clip that cannot
    # say how the radar was configured cannot be compared with another.
    if cc.RADAR_ENABLE:
        cc.write_radar_profile(Path(out_dir), cc.CONFIG_FILE)
    fish, therm, radar, imu, gps = cc.open_sensors()

    def _state(handle, is_enabled):
        if not is_enabled:
            return "off"
        return "ok" if handle else "MISSING"

    print(
        f"[smoke] opened: fisheye={_state(fish, True)} "
        f"thermal={_state(therm, cc.THERMAL_ENABLE)} "
        f"radar={_state(radar, cc.RADAR_ENABLE)} "
        f"imu={_state(imu, cc.IMU_ENABLE)} "
        f"gps={_state(gps, cc.GPS_ENABLE)}",
        flush=True,
    )
    try:
        cc.write_one_chunk(fish, therm, radar, imu, gps)
    finally:
        # Ask the reader what happened BEFORE closing it: per-source detail
        # (which host, what ssh said, whether the fix is the fallback) is the
        # only actionable thing when GPS is the piece that is broken, and it
        # dies with the reader threads.
        gps_status = gps.describe() if gps is not None else None
        cc.close_sensors(fish, therm, radar, imu, gps)
    opened = (fish is not None, therm is not None, radar is not None,
              imu is not None, gps is not None)
    return opened, enabled, gps_status


def _video_info(path: Path):
    """(frame_count, (w, h)) for an mp4, counting frames by decode."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    n, shape = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if shape is None:
            shape = (frame.shape[1], frame.shape[0])
        n += 1
    cap.release()
    return n, shape


def _thermal_contrast(path: Path, sample_every: int = 3):
    """Mean / std-dev (contrast) / dynamic range (p99-p1) averaged over
    sampled grayscale frames. Low std-dev + low dynamic range is the
    signature of an obscured cover (water film, fog) vs a clear scene;
    see docs/thermal_water_problem.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    means, stds, dyns = [], [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % sample_every == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            means.append(float(g.mean()))
            stds.append(float(g.std()))
            dyns.append(float(np.percentile(g, 99) - np.percentile(g, 1)))
        i += 1
    cap.release()
    if not means:
        return None
    n = len(means)
    return sum(means) / n, sum(stds) / n, sum(dyns) / n


def _radar_info(path: Path):
    """(total_rows, heartbeat_rows, point_rows, doppler_rows, snr_rows)
    from an mmwave csv.

    A heartbeat/sentinel row has a timestamp but empty X/Y/Z; it proves
    the radar was alive and honestly reported nothing. A point row carries
    real X/Y/Z floats. doppler_rows counts point rows that also carry the
    V (radial Doppler velocity) column added 2026-07-06; snr_rows counts
    point rows carrying the SNR column (TLV-7 side info) added
    2026-07-09: on a fresh capture both should equal point_rows; 0
    means an old parser is running.
    """
    total = beats = points = doppler = snr = 0
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if len(row) < 5:
                continue
            total += 1
            if row[2] == "" and row[3] == "" and row[4] == "":
                beats += 1
            else:
                points += 1
                if len(row) >= 6 and row[5] != "":
                    doppler += 1
                if len(row) >= 8 and row[6] != "":
                    snr += 1
    return total, beats, points, doppler, snr


def _imu_info(path: Path):
    """(total_rows, valued_rows, kind) from an imu csv. Handles both formats:
    native Euler (Date,Time,Yaw,Pitch,Roll[,Ax,Ay,Az], UART-RVC: default) and
    legacy quaternion (Date,Time,W,X,Y,Z). A valued row has non-empty first two
    value columns; total should equal valued rows on a live link (the reader
    only writes a row when a fresh sample arrived)."""
    total = valued = 0
    kind = "?"
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header:
            low = [c.strip().lower() for c in header]
            kind = "euler" if "yaw" in low else "quat" if "w" in low else "?"
        for row in r:
            if len(row) < 4:
                continue
            total += 1
            if row[2] != "" and row[3] != "":
                valued += 1
    return total, valued, kind


def _gps_info(path: Path):
    """(rows, lat, lon, fix, source, fix_time) from the newest gps_<ts>.csv row.

    `source` is the load-bearing field: `boat_log` is a live position from
    boat1's autopilot, `fallback` is the fixed coordinate in configs/gps.yaml.
    The two are indistinguishable by their numbers, so the verdict is written
    off this column, never off whether a row exists.
    """
    rows = 0
    last = None
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        idx = {name.strip(): i for i, name in enumerate(header or [])}
        for row in r:
            if len(row) < 4:
                continue
            rows += 1
            last = row

    def _cell(name):
        i = idx.get(name)
        if i is None or last is None or len(last) <= i:
            return ""
        return last[i].strip()

    return (rows, _cell("Lat"), _cell("Lon"), _cell("Fix"), _cell("Source"),
            _cell("FixTime"))


def _rtc_info():
    """(present, rtc_utc_str, skew_seconds) for a hardware RTC.

    Reads ``/sys/class/rtc/rtc0/{date,time}`` (UTC, world-readable, no sudo)
    and compares to system UTC. A hardware RTC that tracks system time is what
    lets the box stamp captures correctly **offline** (no NTP / no BLE time).
    present=False when there is no ``rtc0`` at all; skew is |RTC - system| in
    seconds (None if the nodes are unreadable)."""
    from datetime import datetime, timezone

    base = Path("/sys/class/rtc/rtc0")
    if not base.exists():
        return False, None, None
    try:
        date = (base / "date").read_text().strip()
        clock = (base / "time").read_text().strip()
        rtc = datetime.strptime(f"{date} {clock}", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc)
        skew = abs((datetime.now(timezone.utc) - rtc).total_seconds())
        return True, f"{date} {clock} UTC", skew
    except Exception:
        return True, None, None


def _newest(d: Path, pattern: str):
    files = sorted(d.glob(pattern))
    return files[-1] if files else None


def validate(out_dir: str, opened, enabled=None, gps_status: str | None = None,
             require_gps: bool = False) -> bool:
    """Read the artefacts back and print a per-sensor verdict.

    ``enabled`` (fisheye, thermal, radar, imu[, gps]) says which sensors this
    run was configured to use. A sensor that is switched OFF by configuration
    reports ``OFF (disabled)`` and does not fail the smoke, only a sensor that
    was *expected* and did not deliver does.

    ``gps_status`` is the reader's own per-source account (see
    ``GPSReader.describe``); it is printed under the GPS line so a failure
    names the host and the reason. ``require_gps`` promotes a live GPS fix to a
    hard requirement -- off by default because GPS is training metadata and a
    missing position costs no frames, on for the deploy-day check.

    The 4-element tuples the pre-GPS callers pass are still accepted: GPS then
    reads as absent, which is what it was.
    """
    d = Path(out_dir)
    fish_ok, therm_ok, radar_ok, imu_ok, *rest = opened
    gps_ok = bool(rest[0]) if rest else False
    fish_on, therm_on, radar_on, imu_on, *rest_on = (
        enabled or (True, True, True, True, False))
    gps_on = bool(rest_on[0]) if rest_on else False
    print("\n========== VALIDATION ==========", flush=True)
    ok = True

    fpath = _newest(d, "fisheye_*.mp4")
    if fish_ok:
        n, shape = _video_info(fpath) if fpath else (0, None)
        verdict = "PASS" if n > 0 else "FAIL"
        ok &= n > 0
        print(f"fisheye : {verdict}  {n} frames  {shape}  ({fpath.name if fpath else '-'})")
    else:
        print("fisheye : SKIP (did not open)")

    tpath = _newest(d, "thermal_*.mp4")
    if not therm_on:
        print("thermal : OFF (disabled: ASVPROJECT_FISHEYE_ONLY / "
              "ASVPROJECT_THERMAL_ENABLE=0)")
    elif therm_ok:
        n, shape = _video_info(tpath) if tpath else (0, None)
        verdict = "PASS" if n > 0 else "FAIL"
        ok &= n > 0
        line = f"thermal : {verdict}  {n} frames  {shape}"
        c = _thermal_contrast(tpath) if (tpath and n > 0) else None
        if c:
            mean, std, dyn = c
            # std-dev < ~25 or dynamic range < ~120 => the scene contrast
            # has collapsed, i.e. the cover is obscured (water film / fog)
            # rather than showing a clear thermal scene.
            flag = ("  <-- LOW CONTRAST (cover obscured?)"
                    if (std < 25 or dyn < 120) else "")
            line += f"  contrast: mean={mean:.0f} std={std:.0f} dyn={dyn:.0f}{flag}"
        line += f"  ({tpath.name if tpath else '-'})"
        print(line)
    else:
        print("thermal : FAIL (camera did not open)")
        ok = False

    mpath = _newest(d, "mmwave_*.csv")
    if not radar_on:
        print("radar   : OFF (disabled: ASVPROJECT_FISHEYE_ONLY / "
              "ASVPROJECT_RADAR_ENABLE=0)")
    elif radar_ok:
        total, beats, points, doppler, snr = (_radar_info(mpath) if mpath
                                              else (0, 0, 0, 0, 0))
        # Radar "works" = it was read and wrote rows (heartbeats and/or
        # points). Zero rows means no frames were decoded at all -> dead.
        verdict = "PASS" if total > 0 else "FAIL"
        ok &= total > 0
        note = "" if points else "  (no detections: expected indoors)"
        if points and not doppler:
            note += "  <-- NO DOPPLER (V column missing: old parser?)"
        if points and not snr:
            note += "  <-- NO SNR (TLV-7 side info missing: old parser?)"
        print(f"radar   : {verdict}  {total} rows = {beats} heartbeat + "
              f"{points} point ({doppler} with doppler, {snr} with snr)"
              f"{note}  ({mpath.name if mpath else '-'})")
        # Which profile produced this? An A/B whose two halves report the
        # same line here did not vary what it meant to vary.
        prof = d / "radar_profile.cfg"
        if prof.exists():
            summary = next((ln[len("% summary: "):].strip()
                            for ln in prof.read_text().splitlines()
                            if ln.startswith("% summary: ")), "?")
            src = next((ln[len("% source: "):].strip()
                        for ln in prof.read_text().splitlines()
                        if ln.startswith("% source: ")), "?")
            print(f"          profile: {src}")
            print(f"                   {summary}")
    else:
        print("radar   : FAIL (data port did not open)")
        ok = False

    # IMU is optional: reported but never fails the smoke.
    ipath = _newest(d, "imu_*.csv")
    if not imu_on:
        print("imu     : OFF (disabled: ASVPROJECT_IMU_ENABLE=0)")
    elif imu_ok:
        total, valued, kind = _imu_info(ipath) if ipath else (0, 0, "?")
        label = {"euler": "euler (UART-RVC)", "quat": "quaternion (BLE)"}.get(
            kind, "attitude")
        if valued > 0:
            print(f"imu     : PASS  {valued} {label} rows  "
                  f"({ipath.name if ipath else '-'})")
        else:
            print(f"imu     : WARN  0 rows: UART port absent / BLE link down "
                  f"(capture unaffected)  ({ipath.name if ipath else '-'})")
    else:
        print("imu     : SKIP (IMU reader off / dep missing)")

    # GPS: own-boat position metadata from boat1's autopilot log. Reported,
    # and only fails the smoke under --require-gps (see the docstring).
    gpath = _newest(d, "gps_*.csv")
    if not gps_on:
        print("gps     : OFF (disabled: ASVPROJECT_GPS_ENABLE=0)")
    elif gps_ok:
        rows, lat, lon, fixq, source, fix_time = (
            _gps_info(gpath) if gpath else (0, "", "", "", "", ""))
        live = source not in ("", "fallback")
        if rows and live:
            verdict = "PASS"
        elif rows:
            # A row exists, but it is the configured fixed position: the boat
            # was never actually asked where it is.
            verdict = "FAIL" if require_gps else "WARN"
        else:
            verdict = "FAIL" if require_gps else "WARN"
        if require_gps:
            ok &= (rows > 0 and live)
        detail = (f"{lat},{lon} source={source or '?'} fix_q={fixq or '-'}"
                  if rows else "no position from any source")
        age = f"  fix logged {fix_time}" if fix_time else ""
        print(f"gps     : {verdict}  {rows} row(s)  {detail}{age}  "
              f"({gpath.name if gpath else '-'})")
        if rows and not live:
            print("          ^ FALLBACK position from configs/gps.yaml, NOT a "
                  "live fix from boat1")
        if gps_status:
            for line in gps_status.split(" | "):
                print(f"          {line}")
    else:
        print("gps     : SKIP (GPS reader off / configs/gps.yaml disabled)")
        if require_gps:
            ok = False
        if gps_status:
            print(f"          {gps_status}")

    # Per-frame timestamp sidecar (frames_<ts>.csv, since 2026-07-16): reported,
    # never fails. Row count should track the fisheye frame count.
    frpath = _newest(d, "frames_*.csv")
    if frpath:
        try:
            with open(frpath, newline="") as f:
                fr_rows = max(0, sum(1 for _ in f) - 1)
        except Exception:
            fr_rows = 0
        print(f"frames  : {'PASS' if fr_rows > 0 else 'WARN'}  {fr_rows} per-frame "
              f"timestamps  ({frpath.name})")
    else:
        print("frames  : WARN  no frames_*.csv sidecar (old capture code deployed?)")

    # RTC (hardware clock): reported, never fails the smoke. A present + in-sync
    # RTC is what gives captures correct timestamps OFFLINE (no NTP/BLE).
    present, rtc_str, skew = _rtc_info()
    if not present:
        print("rtc     : WARN  no /dev/rtc0, no offline clock (timestamps rely "
              "on NTP/network)")
    elif skew is None:
        print("rtc     : WARN  rtc0 present but time nodes unreadable")
    elif skew <= 120:
        print(f"rtc     : PASS  {rtc_str}  (skew {skew:.0f}s vs system)")
    else:
        print(f"rtc     : WARN  {rtc_str}  (skew {skew:.0f}s vs system: RTC unset? "
              f"run: sudo hwclock -w)")

    print("================================", flush=True)
    print(f"OVERALL : {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=int, default=10,
                    help="capture duration (default 10)")
    ap.add_argument("--dir", default="/home/vesselauser/captures/_smoketest",
                    help="throwaway output directory")
    ap.add_argument("--fisheye-only", action="store_true",
                    help="capture the fisheye alone (radar + thermal disabled, "
                         "never opened or powered); equivalent to setting "
                         "ASVPROJECT_FISHEYE_ONLY=1")
    ap.add_argument("--require-gps", action="store_true",
                    help="fail the smoke unless a LIVE position was read from "
                         "boat1 (the configured fallback does not count). Use "
                         "on deploy day, with boat1 powered and reachable.")
    args = ap.parse_args()

    opened, enabled, gps_status = _open_chunk(
        args.seconds, args.dir, args.fisheye_only)
    ok = validate(args.dir, opened, enabled, gps_status=gps_status,
                  require_gps=args.require_gps)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
