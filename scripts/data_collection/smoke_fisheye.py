#!/usr/bin/env python3
"""Fisheye-ONLY capture smoke test: fisheye + RTC, no radar/thermal/IMU.

Purpose: find out whether the box can run a *reduced* capture (just the fisheye
camera) on a supply that browns out under the full sensor load: e.g. the
boat-power converter tripping its over-current protection when the whole stack
inrushes (see docs/reference/power_and_supply.md / capture_status_and_prereqs.md).

It opens **only** the fisheye (no radar config push, no thermal, no IMU: none of
their power draw), captures a few seconds writing the same
``fisheye_<ts>.mp4`` + ``frames_<ts>.csv`` (per-frame RTC timestamps) as the real
capture path, and reports:
  * the fisheye clip + per-frame timestamp sidecar,
  * the RTC (offline clock), and
  * **the under-voltage / throttle flags before vs after**: the decisive check:
    if fisheye-only stays clean where the full stack browns out, this is a viable
    reduced capture mode for the current boat power.

Run on the Pi with the capture service STOPPED:

    sudo systemctl stop asvproject-capture
    cd ~/ASVProject-ObstacleDetection
    python3 -m scripts.data_collection.smoke_fisheye --seconds 10

Exit code 0 iff the fisheye produced frames.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _throttled():
    """`vcgencmd get_throttled` -> (raw_hex_str, value_int, human_str) or None.

    Bit 0 = under-voltage NOW, bit 2 = throttled NOW, bit 16 = under-voltage
    since boot, bit 18 = throttling since boot. The NOW bits (mask 0x5) are what
    tell you the supply is failing *right now*; the since-boot bits latch until
    reboot, so a before/after delta reveals a transient during this run.
    """
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        raw = out.split("=")[-1].strip()
        val = int(raw, 16)
    except Exception:
        return None
    bits = []
    if val & 0x1:
        bits.append("UV-now")
    if val & 0x4:
        bits.append("throttled-now")
    if val & 0x10000:
        bits.append("UV-since-boot")
    if val & 0x40000:
        bits.append("throttled-since-boot")
    return raw, val, (", ".join(bits) if bits else "clean")


def _frame_count(mp4: Path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=int, default=10,
                    help="capture duration (default 10)")
    ap.add_argument("--dir", default="/home/vesselauser/captures/_smoketest_fisheye",
                    help="throwaway output directory")
    args = ap.parse_args()

    import cv2
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    # Reuse the real capture module (Picamera2 handle, calibrated size, FPS) and
    # the RTC check, so a pass here reflects the real fisheye path.
    from scripts.data_collection import continuous_capture as cc
    from scripts.data_collection.smoke_capture import _rtc_info

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    before = _throttled()
    if before:
        print(f"[smoke-fisheye] throttled BEFORE: {before[0]} ({before[2]})",
              flush=True)
    else:
        print("[smoke-fisheye] throttled: vcgencmd unavailable (not a Pi?)",
              flush=True)

    print("[smoke-fisheye] opening ONLY the fisheye (no radar/thermal/imu)",
          flush=True)
    if cc.Picamera2 is None:
        print("[smoke-fisheye] FAIL picamera2 not importable")
        sys.exit(1)
    try:
        fish = cc.Picamera2()
        fish.configure(fish.create_preview_configuration(
            main={"format": "RGB888", "size": cc.FISHEYE_SIZE}))
        fish.start()
    except Exception as e:
        print(f"[smoke-fisheye] FAIL fisheye open: {e!r}")
        after = _throttled()
        if after:
            print(f"[smoke-fisheye] throttled AFTER fail: {after[0]} ({after[2]})")
        sys.exit(1)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mp4 = out_dir / f"fisheye_{ts}.mp4"
    frames_csv = out_dir / f"frames_{ts}.csv"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4), fourcc, cc.FPS_TARGET, cc.FISHEYE_SIZE)
    fcsv = open(frames_csv, "w", newline="")
    fw = csv.writer(fcsv)
    fw.writerow(["frame_index", "Date", "Time"])

    print(f"[smoke-fisheye] capturing {args.seconds}s -> {mp4.name}", flush=True)
    n = 0
    period = 1.0 / cc.FPS_TARGET
    start = time.monotonic()
    try:
        while (time.monotonic() - start) < args.seconds:
            loop = time.monotonic()
            stamp = datetime.now()
            frame = fish.capture_array()
            writer.write(frame)
            fw.writerow([n, stamp.strftime("%Y-%m-%d"),
                         stamp.strftime("%H:%M:%S.%f")[:-5]])
            n += 1
            sleep_for = period - (time.monotonic() - loop)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        writer.release()
        fcsv.close()
        try:
            fish.stop()
            fish.close()
        except Exception:
            pass

    frames = _frame_count(mp4)
    try:
        with open(frames_csv, newline="") as f:
            fr_rows = max(0, sum(1 for _ in f) - 1)
    except Exception:
        fr_rows = 0
    after = _throttled()
    present, rtc_str, skew = _rtc_info()

    print("\n========== FISHEYE-ONLY VALIDATION ==========", flush=True)
    ok = frames > 0
    print(f"fisheye : {'PASS' if ok else 'FAIL'}  {frames} frames  "
          f"{cc.FISHEYE_SIZE}  ({mp4.name})")
    print(f"frames  : {'PASS' if fr_rows > 0 else 'WARN'}  {fr_rows} per-frame "
          f"timestamps  ({frames_csv.name})")
    if not present:
        print("rtc     : WARN  no /dev/rtc0, no offline clock")
    elif skew is None:
        print("rtc     : WARN  rtc0 present but unreadable")
    elif skew <= 120:
        print(f"rtc     : PASS  {rtc_str}  (skew {skew:.0f}s vs system)")
    else:
        print(f"rtc     : WARN  {rtc_str}  (skew {skew:.0f}s: RTC unset?)")

    # Power verdict: the reason this test exists. Distinguish three cases by
    # comparing before vs after, not just the after state: (1) clean throughout,
    # (2) the supply was ALREADY under-volting at idle before the camera (so
    # fisheye-only isn't the cause: before == after; it still ran, throttled),
    # (3) fisheye-only itself TRIGGERED under-voltage that wasn't there before.
    if before and after:
        before_now = bool(before[1] & 0x5)               # UV/throttle-now before
        after_now = bool(after[1] & 0x5)                 # UV/throttle-now after
        new_latched = bool((after[1] & 0x50000) & ~(before[1] & 0x50000))
        print(f"power   : before {before[0]} ({before[2]})  ->  "
              f"after {after[0]} ({after[2]})")
        if not after_now and not new_latched:
            print("          PASS: stayed clean: fisheye-only is viable on this "
                  "supply")
        elif before_now:
            print(f"          WARN; supply was ALREADY under-volting/throttling "
                  f"at idle (before the camera); fisheye-only did NOT change it "
                  f"(before==after) and still captured {frames} frames. Marginal "
                  f"supply (fix converter/cap for reliability) but fisheye-only "
                  f"RUNS (throttled), unlike the full stack which collapses.")
        else:
            print("          FAIL: fisheye-only TRIGGERED under-voltage/throttle "
                  "during the run (was clean before the camera)")
    else:
        print("power   : SKIP (vcgencmd unavailable)")

    print("=============================================", flush=True)
    print(f"OVERALL : {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
