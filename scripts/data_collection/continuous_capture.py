#!/usr/bin/env python3
"""Continuous capture loop: boot-on-power capture for the boat.

Runs as a systemd service from boot (see asvproject-capture.service).
Opens the enabled sensors once, then writes rotating chunks
(fisheye mp4 + thermal mp4 + mmwave csv + imu csv + frames csv + gps csv)
to /home/vesselauser/captures/ until the service is stopped or the disk fills.

Sensor subsets are supported: ASVPROJECT_FISHEYE_ONLY=1 (or the individual
ASVPROJECT_RADAR_ENABLE / ASVPROJECT_THERMAL_ENABLE flags) runs the fisheye
alone, which is both the reduced-power mode for a marginal supply and the
configuration used by downstream projects that only need RGB vision. A
disabled sensor is never opened and writes no file at all.

The chunked layout mirrors the existing data/<scene>/ convention so
captured triplets can be ingested by `scripts.data.ingest` and replayed
with the canonical pipeline without any rewriting.

Operator commands (SSH, when Tailscale is up):
    sudo systemctl status   asvproject-capture
    sudo systemctl stop     asvproject-capture
    sudo systemctl start    asvproject-capture
    sudo systemctl restart  asvproject-capture
    sudo systemctl disable  asvproject-capture   # stop boot-on-power
    sudo journalctl -u asvproject-capture -f     # follow live logs
    ls -la /home/vesselauser/captures/            # see what's been written

Design notes:
- Single Python process owns all three sensors, so no `/dev/ttyACM*`
  contention with the autopilot (which doesn't touch these).
- One chunk per CHUNK_SECONDS, which keeps individual mp4 files
  manageable and means a crash mid-chunk only loses that chunk.
- All errors are caught + logged + the loop continues. The systemd
  unit also restarts the process on hard exit. Goal: survive long
  enough to be useful in the field, not be pretty.
- Disk-free is checked between chunks; below MIN_FREE_GB the loop
  sleeps and waits for space (manual cleanup expected).
"""

from __future__ import annotations

import asyncio
import binascii
import csv
import os
import glob
import shutil
import signal
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial

try:
    from picamera2 import Picamera2
except ImportError:      # Pi-only dependency: absent on the laptop, where
    Picamera2 = None     # this module is imported for unit tests of the
                         # pure radar-parsing functions (tests/test_continuous_capture.py)

# === Config (operator-tunable) =============================================

def _env_flag(name: str, default: str = "1") -> bool:
    """Read a boolean environment flag. Anything in the falsy set turns the
    feature off; everything else (including "1", "true", "yes") turns it on."""
    return os.environ.get(name, default).strip().lower() not in (
        "0", "false", "no", "off", "")


# --- Per-box constants -----------------------------------------------------
# Values that describe THIS BOX rather than this run, and which must therefore
# be identical however the capture is launched. Putting one only in the systemd
# unit's Environment= is not enough: a manual run (smoke_capture, a calibration
# walk) does not inherit it, and on 2026-08-19 that silently recorded UNROTATED
# thermal by hand while the service recorded rotated. Resolution order is
# environment, then the box file, then the default, so the service and an
# operator at a terminal always agree.
BOX_CONFIG_PATH = os.environ.get("ASVPROJECT_BOX_CONFIG", "/etc/asvproject/box.env")


def _box_env(name: str, default: str) -> str:
    """Read a per-box constant: environment > BOX_CONFIG_PATH > default."""
    if name in os.environ:
        return os.environ[name]
    try:
        with open(BOX_CONFIG_PATH) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == name:
                    return value.strip().strip('"').strip("'")
    except OSError:
        pass
    return default


CAPTURE_DIR = Path(os.environ.get(
    "ASVPROJECT_CAPTURE_DIR", "/home/vesselauser/captures"))
# Per-boot session folders (staged 2026-08-25, first deployed AFTER the
# 2026-08-25 capture session): each service start writes its chunks into
# CAPTURE_DIR/<start-stamp>/ so one session = one folder, ready to copy or
# drop into the dashboard's /upload page as-is. The flat legacy layout
# needed a date-filtered rsync to separate sessions. Applied by main() only
# — smoke tools drive write_one_chunk directly and keep their own out_dir
# flat. Opt out with ASVPROJECT_SESSION_SUBDIR=0.
SESSION_SUBDIR = os.environ.get("ASVPROJECT_SESSION_SUBDIR", "1") != "0"
CHUNK_SECONDS = int(os.environ.get("ASVPROJECT_CHUNK_SECONDS", "300"))   # 5 min
FPS_TARGET = float(os.environ.get("ASVPROJECT_FPS", "3.0"))
MIN_FREE_GB = float(os.environ.get("ASVPROJECT_MIN_FREE_GB", "2.0"))
# Latest-frame snapshot for a process running beside capture (shadow_fusion
# --tail --snapshot-dir): the newest fisheye/thermal pair is published to a
# tmpfs directory once per loop (scripts/data_collection/frame_snapshot.py).
# Default: /run/asvproject when it exists (the systemd unit's RuntimeDirectory),
# else /dev/shm/asvproject where /dev/shm exists (manual runs on the Pi), off
# elsewhere; ASVPROJECT_SNAPSHOT_DIR="" disables it. Cost ~2-4 ms per loop.
# NB /dev/shm entries owned by a user are DELETED by systemd-logind
# (RemoveIPC=yes default) when that user's last login session ends -- which is
# why the service uses /run/asvproject; manual runs from an ssh shell are fine.
SNAPSHOT_DIR = os.environ.get(
    "ASVPROJECT_SNAPSHOT_DIR",
    "/run/asvproject" if os.path.isdir("/run/asvproject")
    else ("/dev/shm/asvproject" if os.path.isdir("/dev/shm") else ""))

# --- Per-sensor enables ----------------------------------------------------
# The box is used by more than one project, and not every deployment carries
# (or can power) the full sensor stack:
#   * fisheye-only vision: a downstream project uses the enclosure with the
#     RGB fisheye as its ONLY vision sensor;
#   * marginal supply: the boat DC-DC converter trips its over-current
#     protection at the full-stack camera inrush, while the fisheye alone stays
#     under the threshold (docs/reference/power_and_supply.md).
# ASVPROJECT_FISHEYE_ONLY=1 is the one-flag shorthand for "fisheye (+ IMU) only":
# it turns the radar and the thermal camera off, so neither is opened, powered,
# nor given an output file. The individual flags override nothing: they are
# simply ANDed with it, so you can also drop a single sensor
# (e.g. ASVPROJECT_THERMAL_ENABLE=0 with the radar still on).
# The IMU keeps its own independent ASVPROJECT_IMU_ENABLE: it is not a vision
# sensor and draws negligible power, so fisheye-only leaves it running.
FISHEYE_ONLY = _env_flag("ASVPROJECT_FISHEYE_ONLY", "0")
RADAR_ENABLE = _env_flag("ASVPROJECT_RADAR_ENABLE") and not FISHEYE_ONLY
THERMAL_ENABLE = _env_flag("ASVPROJECT_THERMAL_ENABLE") and not FISHEYE_ONLY

CLI_PORT = "/dev/ttyACM0"
DATA_PORT = "/dev/ttyACM1"
CLI_BAUD = 115200
DATA_BAUD = 921600
# Radar chirp/frame config pushed at startup. Overridable so the capture
# service can switch profiles (e.g. the clutterRemoval A/B in
# test_config_clutter_on.cfg) via a systemd drop-in / environment line
# without editing code.
CONFIG_FILE = os.environ.get(
    "ASVPROJECT_RADAR_CONFIG", "/home/vesselauser/test_config.cfg")
# Frames with more detected objects than this are logged as heartbeat
# only. Raised 100 -> 500 (2026-07-06 bench): with CFAR at 10 dB a
# person pacing at 2-3 m floods past 100 objects and the old cap
# discarded 128/135 frames: exactly the strong-nearby-target case the
# radar exists for. 500 keeps the pathological-corruption guard while
# logging real dense scenes; the reader-side mmwave.max_objects stays
# the pipeline's (sweepable) filtering policy.
MAX_LOG_OBJECTS = int(os.environ.get("ASVPROJECT_RADAR_MAX_OBJ", "500"))
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

FISHEYE_SIZE = (864, 648)     # (W, H): calibrated intrinsics shape
THERMAL_SIZE = (160, 120)

# The PureThermal does not reliably hand over a frame on the FIRST read after
# the node is opened. Measured 2026-08-17 on healthy hardware: a single read
# rejected the camera on roughly half of all attempts, while retrying 3x at
# 0.3 s scored 12/12 (with and without picamera2 running, so it is not
# contention with libcamera). A single probe therefore reports "no working
# thermal camera" on a perfectly good Lepton and the chunk records no thermal
# at all. Retries apply ONLY to the by-id node, which is definitely the
# PureThermal; scanned nodes keep a single cheap probe so an unresponsive one
# can never stall discovery (the wedge this function was written to avoid).
THERMAL_OPEN_ATTEMPTS = int(os.environ.get("ASVPROJECT_THERMAL_OPEN_ATTEMPTS", "4"))
THERMAL_OPEN_RETRY_S = float(os.environ.get("ASVPROJECT_THERMAL_OPEN_RETRY_S", "0.3"))

# Thermal mount orientation, applied at CAPTURE time so that everything written
# to disk is canonical and the processing side never needs to know how this
# particular box was built. The orientation is a property of the BOX, not of the
# pipeline: it has already varied across builds (the 2026-07-08 clips carry a
# per-clip thermal rotation_deg: 180 for exactly this reason, while the 2025 and
# day-1 corpora are unrotated), so a processing-side global default would
# silently re-orient the legacy corpus. Set this per box in the systemd unit;
# 0 means "record what the sensor gives".
THERMAL_ROTATE_DEG = int(_box_env("ASVPROJECT_THERMAL_ROTATE_DEG", "0")) % 360
if THERMAL_ROTATE_DEG not in (0, 90, 180, 270):
    raise SystemExit(
        f"ASVPROJECT_THERMAL_ROTATE_DEG must be 0, 90, 180 or 270 "
        f"(got {THERMAL_ROTATE_DEG})")

_THERMAL_ROTATE_CODE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}.get(THERMAL_ROTATE_DEG)


def _rotate_thermal(frame):
    """Apply the box's thermal mount rotation. No-op when unset.

    Applied to the probe frame too, so the writer is sized from the ROTATED
    frame and a 90/270 mount carries its swapped dimensions instead of having
    every frame silently dropped by VideoWriter.
    """
    if frame is None or _THERMAL_ROTATE_CODE is None:
        return frame
    return cv2.rotate(frame, _THERMAL_ROTATE_CODE)

# IMU (BNO085). Two sources, selected by ASVPROJECT_IMU_BACKEND:
#   uart_rvc (DEFAULT, since 2026-07-14): the BNO085 is wired directly into
#     SensorBox and read over UART in RVC mode (PS1=Low/PS0=High, sensor TX
#     = SDA pad -> pin 10/GPIO15, /dev/ttyAMA0 @115200). It free-runs 19-byte
#     yaw/pitch/roll frames at 100 Hz: no host commands, no BLE. Writes native
#     Euler CSV. See docs/reference/imu_uart_rvc.md (incl. the SPI post-mortem).
#   ble: legacy path: BNO08x on boat1, bridged over a BLE notify characteristic
#     (boat1:imu_ble_bridge.py); SensorBox is the central. Writes quaternion
#     CSV. Kept selectable + as the template for the planned GPS-over-BLE reader.
# Either way the IMU is fully optional: a missing port / absent boat degrades to
# no imu_*.csv rows, exactly like an absent thermal camera.
IMU_ENABLE = _env_flag("ASVPROJECT_IMU_ENABLE")
IMU_BACKEND = os.environ.get("ASVPROJECT_IMU_BACKEND", "uart_rvc").strip().lower()
# uart_rvc backend
IMU_UART_PORT = os.environ.get("ASVPROJECT_IMU_UART", "/dev/ttyAMA0")
IMU_UART_BAUD = int(os.environ.get("ASVPROJECT_IMU_BAUD", "115200"))
# ble backend (legacy)
IMU_BLE_NAME = os.environ.get("ASVPROJECT_IMU_BLE_NAME", "boat1-imu")
IMU_BLE_ADDR = os.environ.get("ASVPROJECT_IMU_BLE_ADDR", "")   # pin MAC to skip scan
IMU_CHAR_UUID = os.environ.get(
    "ASVPROJECT_IMU_CHAR_UUID", "9a8b7c6d-0002-4b5c-8d6e-1f2a3b4c5d6e")

# GPS (own-boat position, metadata only). Read at capture init and every
# ~30 min into a gps_<ts>.csv sidecar, for sun geometry and offline weather
# correlation during training -- NOT navigation, so a coarse occasional fix is
# plenty and nothing here is on the capture critical path.
#
# The position comes from boat1, the boat's central Pi, whose autopilot ALREADY
# logs every fix to <boatv1>/logs/boat_log_<stamp>.csv. We read the newest row
# of that log (over SSH by default), so nothing has to be enabled or changed on
# the boatv1 side -- which is why this replaced the planned BLE bridge. Sources,
# credentials and the fixed fallback position live in configs/gps.yaml; the
# whole thing degrades to "no gps_*.csv" if it is disabled or unreachable.
GPS_ENABLE = _env_flag("ASVPROJECT_GPS_ENABLE")
# How long capture init waits for that first fix. An SSH round trip over boat
# WiFi is a second or two; without a short wait the first chunk of every
# mission would record the fallback position when a live fix was moments away.
GPS_INIT_TIMEOUT_S = float(os.environ.get("ASVPROJECT_GPS_INIT_TIMEOUT_S", "15"))

# === Signal handling =======================================================

_stop = False


def _handle_signal(signum, _frame):
    global _stop
    _stop = True
    print(f"[signal] {signum} received: finishing chunk and exiting",
          flush=True)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# === Radar helpers (kept byte-equivalent to collect.py) ====================

def _u32(d: bytes) -> int:
    return d[0] + d[1] * 256 + d[2] * 65536 + d[3] * 16777216


# Settings worth echoing at startup: these are what field A/Bs vary, and
# what an analyst needs in order to trust a clip's radar data.
RADAR_PROFILE_KEYS = ("clutterRemoval", "cfarCfg", "channelCfg")


def radar_profile_summary(config_path: str) -> str:
    """One-line digest of the radar profile actually on disk.

    Printed at startup and written beside the data (see write_radar_profile),
    so a clip always records which chirp/detection settings produced it.
    """
    try:
        with open(config_path) as f:
            lines = [ln.strip() for ln in f
                     if ln.strip() and not ln.startswith("%")]
    except OSError as e:
        return f"unreadable: {e!r}"
    picked = [ln for ln in lines if ln.split()[:1]
              and ln.split()[0] in RADAR_PROFILE_KEYS]
    return "; ".join(picked) if picked else "no recognised profile keys"


def write_radar_profile(out_dir, config_path: str) -> None:
    """Drop the radar profile into the capture directory.

    A clip that cannot say which profile it used cannot be compared against
    another clip, which is exactly how the 2026-08-19 A/B was lost.
    """
    from pathlib import Path
    try:
        body = Path(config_path).read_text()
    except OSError as e:
        body = f"% could not read {config_path}: {e!r}\n"
    try:
        (Path(out_dir) / "radar_profile.cfg").write_text(
            f"% source: {config_path}\n"
            f"% summary: {radar_profile_summary(config_path)}\n{body}")
    except OSError as e:
        print(f"[setup] WARN could not write radar_profile.cfg: {e!r}",
              flush=True)


def send_radar_config(cli_path: str, config_path: str) -> None:
    """Push the chirp + frame config to the radar CLI. Stops the sensor
    first so we don't carry junk in the data buffer from a previous
    session (the existing collect.py doesn't do this)."""
    with serial.Serial(cli_path, CLI_BAUD, timeout=1) as cli:
        cli.write(b"sensorStop\n")
        time.sleep(0.1)
        cli.reset_input_buffer()
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                cli.write((line + "\n").encode())
                time.sleep(0.05)


def read_radar_frame(ser: serial.Serial):
    """Sync to magic word, read header + payload. Returns None on
    timeout or truncated read; caller must handle."""
    sync = b""
    # monotonic, not wall-clock: NTP snaps mid-boot mustn't expire the
    # sync window early (same reasoning as the chunk timer).
    deadline = time.monotonic() + 1.0
    while MAGIC_WORD not in sync:
        b = ser.read(1)
        if not b:
            if time.monotonic() > deadline:
                return None
            continue
        sync += b
        sync = sync[-8:]
    header = ser.read(32)
    if len(header) < 32:
        return None
    packet_len = _u32(header[4:8])
    num_det_obj = _u32(header[20:24])
    num_tlvs = _u32(header[24:28])
    payload = ser.read(packet_len - 40)
    if len(payload) < packet_len - 40:
        return None
    return payload, num_tlvs, num_det_obj


def parse_tlvs(data: bytes, num_tlvs: int, num_det_obj: int):
    """Extract the point cloud (TLV type 1) and, when present, the
    per-point side info (TLV type 7). Returns
    ``(xs, ys, zs, vs, snrs, noises)`` or None when the frame is
    structurally inconsistent. ``snrs``/``noises`` are lists of floats
    in dB, or None when the frame carried no (valid) side-info TLV;
    callers must treat side info as strictly optional.

    TLV 7 (MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO, SDK 3.x OOB
    demo): 4 bytes per detected object: int16 snr, int16 noise, both
    in 0.1 dB steps. It has been present in every frame we ever
    captured (guiMonitor detectedObjects=1 emits it alongside TLV 1);
    until 2026-07-09 the parser simply returned at TLV 1 and never
    read it. Parsed now so every logged point carries its detection
    confidence: the enabler for SNR-weighted processing offline.

    Consistency check (PLAN §I.4.7 parser hardening): a healthy frame's
    point TLV is exactly 16 bytes per detected object (x, y, z, radial
    Doppler v: the stride was always 16; earlier revisions just never
    read the 4th float). When the kernel UART buffer overflows mid-chunk
    the sync scan can splice one frame's header onto another's payload;
    the TLV walk then reads reinterpreted garbage (values of 1e30+ or
    denormals near 0). Such frames declare a TLV-1 length that
    disagrees with the header's num_det_obj; drop the whole frame
    rather than log fabricated points. A malformed TLV 7 (length
    disagreement) only discards the side info, not the points: TLV 1
    is the splice detector; a bad 7 alongside a good 1 most likely
    means a config emitting something unexpected, and points remain
    trustworthy on their own.
    """
    offset = 0
    points = None
    side = None
    for _ in range(num_tlvs):
        if offset + 8 > len(data):
            return None   # TLV walk ran off the payload: corrupt frame
        tlv_type = _u32(data[offset:offset + 4])
        tlv_length = _u32(data[offset + 4:offset + 8])
        offset += 8
        if tlv_type == 1:
            if (tlv_length != 16 * num_det_obj
                    or offset + tlv_length > len(data)):
                return None   # header/TLV disagreement: spliced frame
            xs, ys, zs, vs = [], [], [], []
            t = offset
            for _ in range(num_det_obj):
                xs.append(struct.unpack("<f", data[t:t + 4])[0])
                ys.append(struct.unpack("<f", data[t + 4:t + 8])[0])
                zs.append(struct.unpack("<f", data[t + 8:t + 12])[0])
                vs.append(struct.unpack("<f", data[t + 12:t + 16])[0])
                t += 16
            points = (xs, ys, zs, vs)
        elif tlv_type == 7:
            if (tlv_length == 4 * num_det_obj
                    and offset + tlv_length <= len(data)):
                snrs, noises = [], []
                t = offset
                for _ in range(num_det_obj):
                    snr_raw, noise_raw = struct.unpack(
                        "<hh", data[t:t + 4])
                    snrs.append(snr_raw * 0.1)     # 0.1 dB steps → dB
                    noises.append(noise_raw * 0.1)
                    t += 4
                side = (snrs, noises)
        offset += tlv_length
        if points is not None and side is not None:
            break
    if points is None:
        return None   # no point TLV found for a frame that declared objects
    xs, ys, zs, vs = points
    if side is None:
        return xs, ys, zs, vs, None, None
    return xs, ys, zs, vs, side[0], side[1]


class RadarReader(threading.Thread):
    """Continuously drain the radar UART, keeping only the freshest frame.

    Why this exists (2026-07-06 bench finding): the radar streams ~10 Hz
    (~1.4 KB/frame) while the chunk loop reads ~3 Hz, and the kernel tty
    buffer is a fixed 4 KB: more than one loop interval's worth of bytes
    arrives between reads, so the buffer overflowed *continuously* and
    the dropped-byte holes spliced frames. Measured effect: ~30-40% of
    logged points carried garbage coordinates/Doppler, most invisible to
    the 50 m sanity bound because reinterpreted bytes decode as
    denormals near zero. A dedicated reader that blocks on the port and
    parses frames back-to-back never lets the buffer fill; the chunk
    loop then samples the freshest complete frame. Side benefit: rows
    are now timestamped within one frame of the radar's reality instead
    of seconds of backlog.
    """

    def __init__(self, ser: serial.Serial):
        super().__init__(daemon=True, name="radar-reader")
        self.ser = ser
        self._lock = threading.Lock()
        self._latest = None          # (seq, payload, n_tlv, n_obj, stamp)
        self._seq = 0
        self._taken = 0
        self._stop_flag = False
        self.reads_ok = 0            # frames parsed since start
        self.reads_none = 0          # sync timeouts / truncated reads

    def run(self):
        while not self._stop_flag:
            rf = read_radar_frame(self.ser)
            if rf is None:
                self.reads_none += 1
                continue
            payload, n_tlv, n_obj = rf
            self.reads_ok += 1
            with self._lock:
                self._seq += 1
                self._latest = (self._seq, payload, n_tlv, n_obj,
                                datetime.now())

    def take_fresh(self):
        """(payload, n_tlv, n_obj, stamp) for the newest frame not yet
        taken, or None when nothing new arrived since the last take."""
        with self._lock:
            if self._latest is None or self._latest[0] == self._taken:
                return None
            self._taken = self._latest[0]
            _, payload, n_tlv, n_obj, stamp = self._latest
            return payload, n_tlv, n_obj, stamp

    def stop(self):
        self._stop_flag = True

    def close(self):
        self.stop()
        self.join(timeout=2.0)
        self.ser.close()


class ThermalReader(threading.Thread):
    """Drain the thermal camera on its own thread, latest-wins (the
    RadarReader pattern). The capture loop samples the freshest frame and
    NEVER blocks on the sensor.

    Why: on 2026-08-19 the PureThermal died ~90 min into the afloat session
    and the loop's blocking therm.read() then took ~10 s per tick — the
    fisheye collapsed from 3 fps to 0.1 fps for the rest of the outing. A
    dead thermal must cost the thermal stream only."""

    def __init__(self, cap):
        super().__init__(daemon=True, name="thermal-reader")
        self.cap = cap
        self._lock = threading.Lock()
        self._latest = None          # (frame, seq)
        self._taken_seq = 0
        self._seq = 0
        self.reads_ok = 0
        self.reads_fail = 0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                ret, frame = self.cap.read()
            except Exception:
                ret, frame = False, None
            if ret and frame is not None:
                self.reads_ok += 1
                with self._lock:
                    self._seq += 1
                    self._latest = (frame, self._seq)
            else:
                self.reads_fail += 1
                # Dead/stalled sensor: don't spin the CPU. The blocking
                # read() itself already paces us when the sensor is alive.
                self._stop_evt.wait(0.1)

    def peek(self, timeout: float = 0.0):
        """Newest frame WITHOUT consuming it (writer sizing); None if none
        arrives within `timeout` seconds."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._latest is not None:
                    return self._latest[0]
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def take_fresh(self):
        """Newest frame if it is newer than the previous take, else None
        (same contract as RadarReader.take_fresh)."""
        with self._lock:
            if self._latest is None:
                return None
            frame, seq = self._latest
            if seq == self._taken_seq:
                return None
            self._taken_seq = seq
            return frame

    def release(self):
        """Duck-types the cv2 capture handle for close_sensors()."""
        self._stop_evt.set()
        self.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:
            pass


class IMUReader(threading.Thread):
    """Subscribe to the boat's BNO08x quaternion over BLE, keeping the freshest
    sample (latest-wins: mirrors RadarReader).

    The BNO08x is on boat1's I2C bus; boat1 bridges its rotation-vector
    quaternion over a BLE GATT notify characteristic (boat1:imu_ble_bridge.py).
    SensorBox connects as the BLE central here. Each notification is a
    16-byte little-endian float32 quaternion (w, x, y, z).

    Fully optional and self-healing:
      * if `bleak` is missing or the boat is not advertising, the reader simply
        never yields samples and the capture loop proceeds (the imu_*.csv stays
        header-only), exactly like an absent thermal camera;
      * the BLE session auto-reconnects with backoff so a week-long mission
        survives the boat drifting out of range and back.
    """

    # CSV interface (shared shape with IMUReaderRVC so write_one_chunk is
    # format-agnostic): the columns after Date,Time and the values take_fresh
    # yields, in the same order.
    kind = "ble"
    csv_columns = ("W", "X", "Y", "Z")

    def __init__(self, name=IMU_BLE_NAME, address=IMU_BLE_ADDR,
                 char_uuid=IMU_CHAR_UUID):
        super().__init__(daemon=True, name="imu-reader")
        self._name = name
        self._address = address
        self._char = char_uuid
        self._lock = threading.Lock()
        self._latest = None          # (seq, (w, x, y, z), stamp)
        self._seq = 0
        self._taken = 0
        self._stop_flag = False
        self.reads_ok = 0            # notifications received since start
        self.connected = False

    def _on_notify(self, _handle, data):
        if data is None or len(data) < 16:
            return
        try:
            w, x, y, z = struct.unpack("<4f", bytes(data[:16]))
        except Exception:
            return
        self.reads_ok += 1
        with self._lock:
            self._seq += 1
            self._latest = (self._seq, (w, x, y, z), datetime.now())

    async def _session(self) -> bool:
        # Imported here so a missing optional dep never kills capture at import.
        from bleak import BleakClient, BleakScanner
        addr = self._address
        if not addr:
            dev = await BleakScanner.find_device_by_name(self._name, timeout=10.0)
            if dev is None:
                return False           # boat not in range this attempt
            addr = dev.address
        async with BleakClient(addr) as client:
            await client.start_notify(self._char, self._on_notify)
            self.connected = True
            print(f"[setup] IMU BLE connected {addr} ({self._name})", flush=True)
            try:
                while not self._stop_flag and client.is_connected:
                    await asyncio.sleep(0.5)
            finally:
                self.connected = False
                try:
                    await client.stop_notify(self._char)
                except Exception:
                    pass
        return True

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        backoff = 2.0
        while not self._stop_flag:
            try:
                ok = loop.run_until_complete(self._session())
                backoff = 2.0 if ok else min(backoff * 1.5, 30.0)
            except Exception as e:
                print(f"[imu] BLE session error: {e!r}", flush=True)
                backoff = min(backoff * 1.5, 30.0)
            # sleep before the next connect attempt, but stay responsive to stop
            waited = 0.0
            while not self._stop_flag and waited < backoff:
                time.sleep(0.25)
                waited += 0.25
        try:
            loop.close()
        except Exception:
            pass

    def take_fresh(self):
        """((w, x, y, z), stamp) for the newest sample not yet taken, or None
        when nothing new arrived since the last take (link stalled/disconnected)."""
        with self._lock:
            if self._latest is None or self._latest[0] == self._taken:
                return None
            self._taken = self._latest[0]
            _, quat, stamp = self._latest
            return quat, stamp

    def stop(self):
        self._stop_flag = True

    def close(self):
        self.stop()
        self.join(timeout=3.0)


class IMUReaderRVC(threading.Thread):
    """Read the directly-wired BNO085's UART-RVC stream, keeping the freshest
    sample (latest-wins: mirrors RadarReader / IMUReader).

    Since 2026-07-14 the BNO085 is wired into SensorBox and streams in RVC
    mode: /dev/ttyAMA0 @115200, fixed 19-byte frames at 100 Hz, NO host
    commands. Each frame is `0xAA 0xAA | index | yaw/pitch/roll int16 (0.01
    deg) | ax/ay/az int16 (mg) | MI | MR | RSVD | checksum`. Fully optional and
    self-healing: a missing (or reappearing) port just pauses/resumes the
    sample stream, with reconnect backoff, so the capture loop never blocks on
    the IMU. The canonical decoder is imu_bno085.decode_rvc_frame; the tiny copy
    here keeps this capture service dependency-light (cv2/numpy/serial only).
    """

    # CSV interface (shared shape with IMUReader): columns after Date,Time and
    # the values take_fresh yields, in order. Angles deg; acceleration m/s^2.
    kind = "rvc"
    csv_columns = ("Yaw", "Pitch", "Roll", "Ax", "Ay", "Az")

    _ACC_MS2 = 9.80665 / 1000.0   # 1 mg/LSB -> m/s^2

    def __init__(self, port=IMU_UART_PORT, baud=IMU_UART_BAUD):
        super().__init__(daemon=True, name="imu-reader-rvc")
        self._port = port
        self._baud = baud
        self._lock = threading.Lock()
        self._latest = None          # (seq, (yaw,pitch,roll,ax,ay,az), stamp)
        self._seq = 0
        self._taken = 0
        self._stop_flag = False
        self.reads_ok = 0            # frames decoded since start
        self.connected = False

    @staticmethod
    def _decode(frame):
        """(yaw_deg, pitch_deg, roll_deg, ax, ay, az) or None on a bad frame."""
        if (len(frame) < 19 or frame[0] != 0xAA or frame[1] != 0xAA
                or (sum(frame[2:18]) & 0xFF) != frame[18]):
            return None
        yaw, pitch, roll, ax, ay, az = struct.unpack_from("<hhhhhh", frame, 3)
        g = IMUReaderRVC._ACC_MS2
        return (yaw * 0.01, pitch * 0.01, roll * 0.01, ax * g, ay * g, az * g)

    def run(self):
        buf = bytearray()
        backoff = 2.0
        while not self._stop_flag:
            try:
                with serial.Serial(self._port, self._baud, timeout=1.0) as ser:
                    self.connected = True
                    print(f"[setup] IMU UART-RVC open {self._port} @{self._baud}",
                          flush=True)
                    backoff = 2.0
                    buf.clear()
                    while not self._stop_flag:
                        chunk = ser.read(64)
                        if not chunk:
                            continue
                        buf.extend(chunk)
                        while len(buf) >= 19:
                            if buf[0] != 0xAA or buf[1] != 0xAA:
                                del buf[0]          # realign to header
                                continue
                            decoded = self._decode(bytes(buf[:19]))
                            if decoded is None:
                                del buf[0]          # bad checksum/false header
                                continue
                            del buf[:19]
                            self.reads_ok += 1
                            with self._lock:
                                self._seq += 1
                                self._latest = (self._seq, decoded, datetime.now())
            except Exception as e:
                self.connected = False
                print(f"[imu] UART-RVC error: {e!r}", flush=True)
                waited = 0.0
                while not self._stop_flag and waited < backoff:
                    time.sleep(0.25)
                    waited += 0.25
                backoff = min(backoff * 1.5, 30.0)
        self.connected = False

    def take_fresh(self):
        """((yaw,pitch,roll,ax,ay,az), stamp) for the newest sample not yet
        taken, or None when nothing new arrived since the last take (port down
        / stream stalled)."""
        with self._lock:
            if self._latest is None or self._latest[0] == self._taken:
                return None
            self._taken = self._latest[0]
            _, vals, stamp = self._latest
            return vals, stamp

    def stop(self):
        self._stop_flag = True

    def close(self):
        self.stop()
        self.join(timeout=3.0)


# === Disk + housekeeping ===================================================

def disk_free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    s = shutil.disk_usage(path)
    return s.free / (1024 ** 3)


# === Sensor setup ==========================================================

def open_sensors():
    """Open the enabled sensors. Returns (fisheye_cam, thermal_cap, radar, imu,
    gps).
    Any individual sensor failure becomes a None so we can degrade gracefully
    instead of bailing the whole process out; a *disabled* sensor is likewise
    None, but is never opened or powered in the first place (see FISHEYE_ONLY /
    ASVPROJECT_{RADAR,THERMAL}_ENABLE)."""
    if FISHEYE_ONLY:
        print("[setup] FISHEYE-ONLY mode (ASVPROJECT_FISHEYE_ONLY=1): radar "
              "and thermal are disabled; no mmwave_*.csv / thermal_*.mp4 will "
              "be written", flush=True)

    radar = None
    if RADAR_ENABLE:
        # Name the profile. A mis-invoked ASVPROJECT_RADAR_CONFIG used to be
        # invisible: the 2026-08-19 clutterRemoval A/B ran both halves on the
        # default profile and only the analysis, days later, revealed it.
        print(f"[setup] pushing radar config: {CONFIG_FILE} "
              f"({radar_profile_summary(CONFIG_FILE)})", flush=True)
        try:
            send_radar_config(CLI_PORT, CONFIG_FILE)
        except Exception as e:
            print(f"[setup] WARN radar config push failed: {e!r}", flush=True)

        print("[setup] opening radar data port", flush=True)
        try:
            ser = serial.Serial(DATA_PORT, DATA_BAUD, timeout=1)
            ser.reset_input_buffer()
            radar = RadarReader(ser)
            radar.start()
        except Exception as e:
            print(f"[setup] WARN radar data port: {e!r}", flush=True)
            radar = None
    else:
        print("[setup] radar disabled (ASVPROJECT_RADAR_ENABLE=0)", flush=True)

    print("[setup] starting fisheye (picamera2)", flush=True)
    try:
        fish = Picamera2()
        fish.configure(fish.create_preview_configuration(
            main={"format": "RGB888", "size": FISHEYE_SIZE}))
        fish.start()
    except Exception as e:
        print(f"[setup] WARN fisheye: {e!r}", flush=True)
        fish = None

    therm = None
    if not THERMAL_ENABLE:
        print("[setup] thermal disabled (ASVPROJECT_THERMAL_ENABLE=0)",
              flush=True)
    else:
        therm = _open_thermal()
        if therm is not None:
            # Latest-wins reader thread: the chunk loop samples frames and
            # never blocks on the sensor (2026-08-19 incident).
            therm = ThermalReader(therm)
            therm.start()

    # IMU (optional; never blocks capture). Backend per ASVPROJECT_IMU_BACKEND:
    # uart_rvc (default, directly-wired box IMU) or ble (legacy boat1 bridge).
    imu = None
    if IMU_ENABLE:
        try:
            if IMU_BACKEND == "ble":
                import bleak  # noqa: F401 - probe the optional dep up front
                imu = IMUReader()
                imu.start()
                print(f"[setup] IMU BLE reader started "
                      f"(name={IMU_BLE_NAME}, char={IMU_CHAR_UUID})", flush=True)
            else:
                if IMU_BACKEND != "uart_rvc":
                    print(f"[setup] WARN unknown ASVPROJECT_IMU_BACKEND="
                          f"{IMU_BACKEND!r}: using uart_rvc", flush=True)
                imu = IMUReaderRVC()
                imu.start()
                print(f"[setup] IMU UART-RVC reader started "
                      f"(port={IMU_UART_PORT} @{IMU_UART_BAUD})", flush=True)
        except Exception as e:
            print(f"[setup] WARN IMU reader unavailable: {e!r}", flush=True)
            imu = None
    else:
        print("[setup] IMU disabled (ASVPROJECT_IMU_ENABLE=0)", flush=True)

    # GPS (optional metadata; never blocks capture beyond the init read).
    gps = None
    if GPS_ENABLE:
        try:
            # Imported lazily, and with the repo root put on the path: the
            # systemd unit runs this file BY PATH (not as -m), so sys.path[0]
            # is scripts/data_collection/ and `scripts.*` is not importable
            # without this. Lazy + guarded so a missing PyYAML or a malformed
            # gps.yaml costs the GPS metadata, never the capture.
            _repo = str(Path(__file__).resolve().parents[2])
            if _repo not in sys.path:
                sys.path.insert(0, _repo)
            from scripts.sensor_processing.gps_boat1 import GPSReader

            reader = GPSReader()
            if not reader.enabled:
                print("[setup] GPS disabled in configs/gps.yaml", flush=True)
            else:
                reader.start()
                gps = reader
                # The "read position on initialisation" half of the job. Print
                # the source as well as the fix: a fallback position looks
                # exactly like a real one in the data, and this line is where
                # the difference is meant to be caught.
                fix = gps.wait_for_fix(GPS_INIT_TIMEOUT_S)
                if fix is None:
                    print(f"[setup] WARN GPS: no fix at init "
                          f"({gps.describe()})", flush=True)
                else:
                    print(f"[setup] GPS init fix {fix.lat_deg:+.6f},"
                          f"{fix.lon_deg:+.6f} source={fix.source} "
                          f"quality={fix.fix_quality}", flush=True)
                    if fix.source == "fallback":
                        print(f"[setup] WARN GPS is the CONFIGURED FALLBACK, "
                              f"not a live fix: {gps.describe()}", flush=True)
        except Exception as e:
            print(f"[setup] WARN GPS reader unavailable: {e!r}", flush=True)
            gps = None
    else:
        print("[setup] GPS disabled (ASVPROJECT_GPS_ENABLE=0)", flush=True)

    return fish, therm, radar, imu, gps


# V4L2 nodes belonging to the SoC's own image/codec blocks rather than to a
# camera. They are NOT capture devices, but they open successfully, and a
# read() on one can block forever: /dev/video14 (bcm2835-isp) hung the capture
# service indefinitely on 2026-08-06 with the PureThermal unplugged. A missing
# sensor must degrade to "absent", never wedge the run, so these are skipped by
# name before anything tries to read them.
#
# Matched against /sys/class/video4linux/<node>/name. The PureThermal is reached
# via its by-id symlink regardless of this list, so a genuine thermal camera can
# never be excluded by it.
_NON_CAMERA_V4L2 = ("bcm2835", "unicam", "rpi-hevc", "codec", "isp",
                    "pispbe", "rp1-cfe")


def _v4l2_node_name(dev: str) -> str:
    """Driver-reported name of a /dev/videoN node ('' if unknown)."""
    try:
        with open(f"/sys/class/video4linux/{os.path.basename(dev)}/name") as f:
            return f.read().strip().lower()
    except OSError:
        return ""


def _open_thermal():
    """Find and open the PureThermal/Lepton V4L2 node. Returns the capture
    handle, or None when no genuine thermal node answers.

    Never blocks: on-SoC ISP/codec nodes are filtered out by driver name before
    being probed (see _NON_CAMERA_V4L2), because reading one can hang forever.
    """
    therm = None
    # The Lepton's /dev/video index is NOT fixed: when the CSI fisheye is
    # connected, the camera front-end claims the low indices and pushes the
    # PureThermal higher. A hard-coded (0,1,2,3) walk therefore misses it once
    # the fisheye is plugged in. Build an index-independent candidate list (the
    # stable by-id symlink first, then every plausible /dev/video* node in
    # numeric order) and let the frame-size guard below pick the real one.
    def _vidx(path):
        tail = path.rsplit("video", 1)[-1]
        return int(tail) if tail.isdigit() else 9999
    by_id = {os.path.realpath(link)
             for link in sorted(glob.glob("/dev/v4l/by-id/*PureThermal*index0"))}
    candidates = sorted(by_id)
    skipped = []
    for dev in sorted(glob.glob("/dev/video*"), key=_vidx):
        if dev in candidates:
            continue
        name = _v4l2_node_name(dev)
        if any(bad in name for bad in _NON_CAMERA_V4L2):
            skipped.append(f"{dev}({name})")
            continue
        candidates.append(dev)
    if skipped:
        print(f"[setup] thermal: skipping {len(skipped)} on-SoC node(s): "
              f"{', '.join(skipped[:4])}{' ...' if len(skipped) > 4 else ''}",
              flush=True)
    for dev in candidates:
        # Explicit V4L2 backend: avoids OpenCV's obsensor/GStreamer autoprobe
        # (noisy, and it can grab the wrong node). BUFFERSIZE=1 keeps the read
        # on the freshest frame so a slow loop can't stall the stream.
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # See THERMAL_OPEN_ATTEMPTS: the by-id node gets several tries because
        # the first read after opening is unreliable; anything found by the
        # generic scan gets one, so a dead node cannot stall discovery.
        attempts = THERMAL_OPEN_ATTEMPTS if dev in by_id else 1
        ret, frame = False, None
        for attempt in range(attempts):
            ret, frame = cap.read()
            if ret and frame is not None:
                if attempt:
                    print(f"[setup] thermal on {dev}: first frame took "
                          f"{attempt + 1} reads", flush=True)
                break
            if attempt + 1 < attempts:
                time.sleep(THERMAL_OPEN_RETRY_S)
        # Accept ONLY a genuine thermal-sized frame (<= 320x240). Otherwise we
        # can accidentally pick a large CSI/ISP /dev/video node whose frames
        # don't match the writer size: which silently drops every frame and
        # yields an empty (44-byte) thermal.mp4.
        if (ret and frame is not None
                and frame.shape[0] <= 240 and frame.shape[1] <= 320):
            therm = cap
            print(f"[setup] thermal on {dev} "
                  f"({frame.shape[1]}x{frame.shape[0]})", flush=True)
            break
        cap.release()
    if therm is None:
        print("[setup] WARN no working thermal camera", flush=True)
    return therm


def close_sensors(fish, therm, radar, imu, gps=None):
    print("[shutdown] closing sensors", flush=True)
    try:
        if fish is not None:
            fish.stop()
            fish.close()
    except Exception as e:
        print(f"[shutdown] fisheye close: {e!r}", flush=True)
    try:
        if therm is not None:
            therm.release()
    except Exception as e:
        print(f"[shutdown] thermal close: {e!r}", flush=True)
    try:
        if radar is not None:
            radar.close()   # stops the reader thread and closes the port
    except Exception as e:
        print(f"[shutdown] radar close: {e!r}", flush=True)
    try:
        if imu is not None:
            imu.close()     # stops the BLE reader thread
    except Exception as e:
        print(f"[shutdown] imu close: {e!r}", flush=True)
    try:
        if gps is not None:
            gps.close()     # stops the log-poller / BLE reader threads
    except Exception as e:
        print(f"[shutdown] gps close: {e!r}", flush=True)


# === Capture loop ==========================================================

GPS_CSV_COLUMNS = ("Date", "Time", "Lat", "Lon", "Fix", "Source", "FixTime")


def gps_row(fix, stamp):
    """One gps_<ts>.csv row for `fix`, written at wall-clock `stamp`.

    Date/Time are when the row was WRITTEN (this box's RTC, so the sidecar
    aligns with the radar/IMU/frames streams on RoundedTime); FixTime is the
    source's own UTC timestamp. They differ by up to the poll interval, and
    keeping both is what makes a repeated row recognisable as the same fix
    rather than a fresh one. Source is `boat_log` / `ble` / `fallback`: a
    fallback position is never allowed to look like a measurement.
    """
    return [
        stamp.strftime("%Y-%m-%d"),
        stamp.strftime("%H:%M:%S.%f")[:-5],
        f"{fix.lat_deg:.7f}",
        f"{fix.lon_deg:.7f}",
        "" if fix.fix_quality is None else fix.fix_quality,
        fix.source,
        "" if fix.fix_time_utc is None else fix.fix_time_utc.isoformat(),
    ]


#: frames_<ts>.csv exposure columns (picamera2 request metadata keys).
FRAMES_EXPOSURE_COLUMNS = ("ExposureTime", "AnalogueGain", "DigitalGain", "Lux")


def _grab_fisheye(fish):
    """One fisheye frame plus the camera's per-frame metadata.

    picamera2's request API hands back the frame AND the metadata the ISP
    used for it (ExposureTime in us, AnalogueGain, DigitalGain, Lux estimate,
    ...) in one shot; capture_array() alone throws the metadata away. Objects
    without capture_request (test stubs, other backends) fall back to
    capture_array() with empty metadata, so the sidecar columns are simply
    blank there.
    """
    capture_request = getattr(fish, "capture_request", None)
    if capture_request is None:
        return fish.capture_array(), {}
    req = capture_request()
    try:
        frame = req.make_array("main")
        meta = req.get_metadata() or {}
    finally:
        req.release()
    return frame, meta


def _exposure_cols(meta: dict) -> list:
    """FRAMES_EXPOSURE_COLUMNS values from picamera2 metadata, '' when absent."""
    out = []
    for key in FRAMES_EXPOSURE_COLUMNS:
        v = meta.get(key) if meta else None
        if v is None:
            out.append("")
        elif key == "ExposureTime":
            out.append(int(v))
        else:
            out.append(round(float(v), 4))
    return out


def write_one_chunk(fish, therm, radar, imu, gps=None) -> int:
    """Write a single CHUNK_SECONDS-long chunk (fisheye/thermal/radar + IMU +
    GPS). Returns frame count."""
    chunk_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fisheye_path = CAPTURE_DIR / f"fisheye_{chunk_ts}.mp4"
    thermal_path = CAPTURE_DIR / f"thermal_{chunk_ts}.mp4"
    mmwave_path = CAPTURE_DIR / f"mmwave_{chunk_ts}.csv"
    imu_path = CAPTURE_DIR / f"imu_{chunk_ts}.csv"
    frames_path = CAPTURE_DIR / f"frames_{chunk_ts}.csv"
    gps_path = CAPTURE_DIR / f"gps_{chunk_ts}.csv"
    snapshot = None
    if SNAPSHOT_DIR:
        try:
            from scripts.data_collection.frame_snapshot import SnapshotWriter
            snapshot = SnapshotWriter(SNAPSHOT_DIR)
        except Exception as e:
            print(f"[chunk] WARN snapshot dir {SNAPSHOT_DIR} unusable: {e!r}",
                  flush=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fwriter = cv2.VideoWriter(str(fisheye_path), fourcc, FPS_TARGET, FISHEYE_SIZE)
    # Radar CSV: written whenever the radar is ENABLED, even if the port failed
    # to open: a header-only mmwave_<ts>.csv is the on-disk record of "the
    # radar was expected here and said nothing", which is exactly the signal
    # post-hoc analysis needs. When the radar is *disabled* (fisheye-only /
    # ASVPROJECT_RADAR_ENABLE=0) no file is created at all, so the chunk is an
    # honest fisheye-only recording rather than a triplet with a dead leg.
    csvfile = open(mmwave_path, "w", newline="") if RADAR_ENABLE else None
    # IMU sidecar CSV, one row per loop when a fresh attitude sample is
    # available. Only created when the reader exists, mirroring the box sensors
    # (an absent IMU produces no file at all, rather than empty clutter).
    imu_csvfile = open(imu_path, "w", newline="") if imu is not None else None
    # Per-frame timestamp sidecar: one row per loop iteration mapping the mp4
    # frame_index to the RTC-backed wall-clock time it was captured. Lets
    # downstream align frames to radar/imu/gps by real time even when the loop
    # rate drifts (e.g. the Pi throttling under a marginal supply): chunk_start
    # + index/FPS silently desyncs there. Absent on pre-2026-07-16 clips, so
    # loaders fall back to index/FPS (see datasets.load_frames_csv).
    frames_csvfile = open(frames_path, "w", newline="")
    # GPS sidecar: a row at chunk start with the position in force, plus a row
    # whenever a new fix lands mid-chunk. Written per chunk rather than once
    # per mission so every clip is self-contained -- resolve_triplet hands the
    # pipeline a position without it having to know which chunk came first.
    gps_csvfile = open(gps_path, "w", newline="") if gps is not None else None

    # Size the thermal writer from a REAL probe frame, not the hardcoded
    # THERMAL_SIZE: cv2.VideoWriter.write() silently no-ops when the frame size
    # != the writer size, so any mismatch produces an empty (44-byte) mp4 with
    # no error. Sizing from the actual frame (and resizing defensively below)
    # makes that failure mode impossible.
    thermal_size = THERMAL_SIZE
    if therm is not None:
        # peek() does not consume: the same frame reaches the loop's first
        # take_fresh(), so no thermal frame is lost to sizing. The bounded
        # timeout means a dead sensor costs 2 s per chunk, not a blocked read.
        probe = therm.peek(timeout=2.0)
        if probe is not None:
            probe = _rotate_thermal(probe)
            thermal_size = (probe.shape[1], probe.shape[0])   # (W, H)
    # Same rule as the radar CSV above: enabled-but-dead still produces the
    # (empty) mp4 as evidence; disabled produces nothing.
    twriter = (cv2.VideoWriter(str(thermal_path), fourcc, FPS_TARGET, thermal_size)
               if THERMAL_ENABLE else None)

    # Radar telemetry counters: let on-disk artefacts distinguish
    # "radar is dead" from "radar is honestly seeing nothing".
    # reads_* deltas come from the RadarReader thread (its 10 Hz view of
    # the sensor); sampled/stale count this loop's 3 Hz takes.
    r_ok0 = radar.reads_ok if radar is not None else 0
    r_none0 = radar.reads_none if radar is not None else 0
    sampled = 0
    stale = 0
    obj_zero = 0
    obj_capped = 0
    obj_bad = 0
    obj_logged = 0
    n_frames = 0
    # Thermal telemetry: so an empty thermal.mp4 is never silent again.
    # ok/fail deltas come from the ThermalReader thread; written/stale count
    # this loop's takes (same split as the radar counters above).
    t_ok0 = therm.reads_ok if therm is not None else 0
    t_fail0 = therm.reads_fail if therm is not None else 0
    t_written = 0
    t_stale = 0
    # IMU telemetry: distinguish "link is dead" from "boat is level & steady".
    imu_sampled = 0
    imu_stale = 0
    i_ok0 = imu.reads_ok if imu is not None else 0
    gps_rows = 0
    gps_source = "-"

    try:
        csv_w = None
        if csvfile is not None:
            csv_w = csv.writer(csvfile)
            # V = per-point radial Doppler velocity (m/s) from TLV 1 (since
            # 2026-07-06). SNR/NOISE = per-point detection SNR and noise
            # floor in dB from TLV 7 (since 2026-07-09; empty when the frame
            # carried no side info). Readers treat every column past Z as
            # optional so older CSVs (and older readers seeing new CSVs)
            # keep working.
            csv_w.writerow(["Date", "Time", "X", "Y", "Z", "V", "SNR", "NOISE"])

        imu_csv_w = None
        if imu_csvfile is not None:
            imu_csv_w = csv.writer(imu_csvfile)
            # Columns depend on the backend (imu.csv_columns): the uart_rvc
            # reader writes native Euler Yaw,Pitch,Roll (deg) + Ax,Ay,Az
            # (m/s^2); the legacy BLE reader writes the rotation-vector unit
            # quaternion W,X,Y,Z. Downstream (datasets.load_imu_csv /
            # imu_replay) detects which by the header.
            imu_csv_w.writerow(["Date", "Time"] + list(imu.csv_columns))

        frames_csv_w = csv.writer(frames_csvfile)
        # Exposure columns (since 2026-08-28): the camera's own per-frame
        # metadata -- ExposureTime (us), AnalogueGain, DigitalGain, Lux
        # (picamera2's estimate). At 3 fps auto-exposure stretches to ~300 ms
        # and high gain at dusk, so a frame that LOOKS like daylight can be a
        # long, noisy, motion-blurred integration of near-darkness: the
        # 2026-08-26 outing's 20:40 frames read as evening while it was
        # nearly dark in person. Without these numbers the training corpus
        # cannot tell the two apart. Empty when the camera gives no metadata
        # (stub/legacy); readers treat every column past Time as optional.
        frames_csv_w.writerow(["frame_index", "Date", "Time",
                               *FRAMES_EXPOSURE_COLUMNS])

        gps_csv_w = None
        if gps_csvfile is not None:
            gps_csv_w = csv.writer(gps_csvfile)
            gps_csv_w.writerow(list(GPS_CSV_COLUMNS))
            # Chunk-start row: the position in force when this clip began. It
            # repeats the previous chunk's fix when the ~30 min poll has not
            # come round again, which is the intended behaviour -- the boat has
            # not moved far at sailing speed, and FixTime shows it is the same
            # reading rather than a new one.
            #
            # Guarded like the in-loop read below: this runs before the capture
            # loop starts, so an exception escaping here would cost the WHOLE
            # chunk (main() would catch it, sleep, and try again -- losing a
            # chunk per attempt) over a piece of metadata.
            try:
                start_fix = gps.get()
                if start_fix is not None:
                    gps_csv_w.writerow(gps_row(start_fix, datetime.now()))
                    gps_rows += 1
                    gps_source = start_fix.source
                    gps.take_new()  # don't re-write it as "new" in the loop
            except Exception as e:
                print(f"[chunk] WARN gps init read: {e!r}", flush=True)

        # No per-chunk buffer reset needed any more: the RadarReader
        # thread drains the port continuously, so there is no stale
        # backlog to discard at chunk boundaries.

        free = disk_free_gb(CAPTURE_DIR)
        print(f"[chunk] {chunk_ts} → {CAPTURE_DIR}  (free {free:.1f} GB)",
              flush=True)

        # Use monotonic for the chunk-length guard so an NTP clock-snap
        # (Pi 4 has no RTC and may sync several minutes post-boot) doesn't
        # end the chunk early. datetime.now() above is fine for the
        # filename: that's tagging wall-clock, which is what we want.
        chunk_start = time.monotonic()
        frame_period = 1.0 / FPS_TARGET

        while not _stop and (time.monotonic() - chunk_start) < CHUNK_SECONDS:
            loop_start = time.monotonic()
            frame_stamp = datetime.now()   # wall-clock (RTC) for this frame
            fish_meta: dict = {}
            frame = None
            tframe = None

            # Fisheye (frame + the camera's exposure metadata for the sidecar)
            try:
                if fish is not None:
                    frame, fish_meta = _grab_fisheye(fish)
                    fwriter.write(frame)
            except Exception as e:
                print(f"[chunk] WARN fisheye read: {e!r}", flush=True)

            # Thermal: sample the freshest frame the reader thread has.
            # NEVER a blocking sensor read here — see ThermalReader.
            try:
                if therm is not None and twriter is not None:
                    tframe = therm.take_fresh()
                    if tframe is None:
                        # No new frame since the last take: at the Lepton's
                        # ~8.7 Hz vs this 3 Hz loop that means the stream
                        # has stalled (USB dropout / dead module).
                        t_stale += 1
                    else:
                        tframe = _rotate_thermal(tframe)
                        # Defensive: guarantee the frame matches the writer size
                        # (a mismatch would be silently dropped -> empty mp4).
                        if (tframe.shape[1], tframe.shape[0]) != thermal_size:
                            tframe = cv2.resize(tframe, thermal_size)
                        twriter.write(tframe)
                        t_written += 1
            except Exception as e:
                print(f"[chunk] WARN thermal write: {e!r}", flush=True)

            # Snapshot for the shadow process (see SNAPSHOT_DIR). Keyed by
            # (chunk_ts, frame index) = this iteration's frames_<ts>.csv row.
            if snapshot is not None and fish is not None:
                snapshot.publish(chunk_ts, n_frames,
                                 frame_stamp.strftime("%H:%M:%S.%f")[:-5],
                                 frame, tframe if tframe is not None else None)

            # Radar: sample the freshest frame the reader thread has.
            try:
                if radar is not None and csv_w is not None:
                    frame = radar.take_fresh()
                    if frame is None:
                        # No new frame since the last take: at the 10 Hz
                        # radar vs 3 Hz loop this means the stream has
                        # stalled (dead sensor / unplugged cable).
                        stale += 1
                    else:
                        sampled += 1
                        payload, n_tlv, n_obj, stamp = frame
                        date_str = stamp.strftime("%Y-%m-%d")
                        time_str = stamp.strftime("%H:%M:%S.%f")[:-5]
                        if n_obj == 0:
                            # Sentinel row: timestamp + empty X/Y/Z. Tells
                            # post-hoc analysis "the radar was alive at
                            # this instant and reported nothing". Loader
                            # keeps the row so iterate_triplet can use it
                            # as a heartbeat.
                            obj_zero += 1
                            csv_w.writerow(
                                [date_str, time_str, "", "", "", "", "", ""])
                        elif n_obj > MAX_LOG_OBJECTS:
                            # Same idea for the upper sanity cap: log the
                            # heartbeat with no points so callers know the
                            # frame existed but was discarded.
                            obj_capped += 1
                            csv_w.writerow(
                                [date_str, time_str, "", "", "", "", "", ""])
                        else:
                            parsed = parse_tlvs(payload, n_tlv, n_obj)
                            if parsed is None:
                                # Structurally inconsistent (spliced)
                                # frame: heartbeat only, no fabricated
                                # points.
                                obj_bad += 1
                                csv_w.writerow(
                                    [date_str, time_str, "", "", "", "",
                                     "", ""])
                            else:
                                obj_logged += 1
                                xs, ys, zs, vs, snrs, noises = parsed
                                if snrs is None:
                                    snrs = [""] * len(xs)
                                    noises = [""] * len(xs)
                                for x, y, z, v, s, nz in zip(
                                        xs, ys, zs, vs, snrs, noises):
                                    csv_w.writerow(
                                        [date_str, time_str, x, y, z, v,
                                         s, nz])
            except Exception as e:
                print(f"[chunk] WARN radar read: {e!r}", flush=True)

            # IMU: sample the freshest attitude the reader has (uart_rvc Euler
            # or legacy BLE quaternion; write_one_chunk is format-agnostic).
            try:
                if imu is not None and imu_csv_w is not None:
                    sample = imu.take_fresh()
                    if sample is None:
                        # No new sample since last take: stream stalled (port
                        # down / boat out of range). Gap in the CSV marks it
                        # (same convention as radar's stale count).
                        imu_stale += 1
                    else:
                        imu_sampled += 1
                        vals, istamp = sample
                        imu_csv_w.writerow([
                            istamp.strftime("%Y-%m-%d"),
                            istamp.strftime("%H:%M:%S.%f")[:-5],
                            *vals])
            except Exception as e:
                print(f"[chunk] WARN imu read: {e!r}", flush=True)

            # GPS: a row only when the fix actually changed (take_new), so a
            # 30 min cadence writes ~1 row per chunk instead of one per loop.
            # Cheap enough to poll at frame rate: it is a lock and a tuple
            # compare against the reader's cached value, never any I/O.
            try:
                if gps is not None and gps_csv_w is not None:
                    new_fix = gps.take_new()
                    if new_fix is not None:
                        gps_csv_w.writerow(gps_row(new_fix, datetime.now()))
                        gps_rows += 1
                        gps_source = new_fix.source
            except Exception as e:
                print(f"[chunk] WARN gps read: {e!r}", flush=True)

            # Per-frame timestamp: frame_index is 0-based and matches the mp4
            # frame order (both cameras are written above in this iteration).
            frames_csv_w.writerow([
                n_frames,
                frame_stamp.strftime("%Y-%m-%d"),
                frame_stamp.strftime("%H:%M:%S.%f")[:-5],
                *_exposure_cols(fish_meta)])

            # Hand this iteration's rows to the kernel now (2026-09-03). The
            # default ~8 KB buffer released the ~70-byte frames/IMU rows only
            # every ~38 s, which is exactly the 12-50 s lag a reader tailing
            # these files (shadow_fusion --tail) measured on the Pi. One
            # flush() per file per loop is one syscall into the page cache --
            # the SD card still sees the kernel's coalesced writeback, so no
            # extra wear -- and a hard power cut now loses the rows since the
            # last loop instead of up to 8 KB per file.
            for f in (csvfile, imu_csvfile, frames_csvfile, gps_csvfile):
                if f is not None:
                    f.flush()

            n_frames += 1
            elapsed = time.monotonic() - loop_start
            sleep_for = frame_period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        # Always finalize mp4s and close the CSV, even on exception, so
        # the outer main()-loop retry doesn't leak file descriptors and
        # we don't accumulate unfinalized .mp4 files across crashes.
        fwriter.release()
        if twriter is not None:
            twriter.release()
        if csvfile is not None:
            csvfile.close()
        if imu_csvfile is not None:
            imu_csvfile.close()
        frames_csvfile.close()
        if gps_csvfile is not None:
            gps_csvfile.close()

    reads_ok = (radar.reads_ok - r_ok0) if radar is not None else 0
    reads_none = (radar.reads_none - r_none0) if radar is not None else 0
    radar_total = reads_ok + reads_none
    radar_pct = (reads_ok / radar_total * 100.0) if radar_total else 0.0
    t_read_ok = (therm.reads_ok - t_ok0) if therm is not None else 0
    t_read_fail = (therm.reads_fail - t_fail0) if therm is not None else 0
    if therm is not None and t_written == 0:
        print(f"[chunk] WARN thermal wrote 0 frames "
              f"(reader ok={t_read_ok} fail={t_read_fail}): sensor/stream issue",
              flush=True)
    # Loop-rate alarm: the 2026-08-19 collapse ran 90 min before anyone
    # noticed. A healthy chunk logs ~CHUNK_SECONDS*FPS_TARGET loops; under
    # half of that means the loop itself is degraded (blocking sensor,
    # throttling) and the whole chunk's fisheye is sparse.
    expected_loops = CHUNK_SECONDS * FPS_TARGET
    if n_frames < expected_loops * 0.5 and not _stop:
        print(f"[chunk] WARN loop rate degraded: {n_frames} loops of "
              f"~{expected_loops:.0f} expected — fisheye is sparse; check "
              f"sensor health", flush=True)
    imu_reads = (imu.reads_ok - i_ok0) if imu is not None else 0
    imu_conn = imu.connected if imu is not None else False
    if imu is not None and imu_sampled == 0:
        print(f"[chunk] WARN imu wrote 0 rows "
              f"(backend={imu.kind} connected={imu_conn}): stream down "
              f"(UART port absent / BLE link down / boat out of range)",
              flush=True)
    # Disabled sensors report "off" rather than a run of zeroes, so a
    # fisheye-only journal line can never be mistaken for a dead-sensor one.
    thermal_report = (
        f"thermal wrote={t_written} stale={t_stale} "
        f"({thermal_size[0]}x{thermal_size[1]}, "
        f"reader ok={t_read_ok} fail={t_read_fail})"
        if THERMAL_ENABLE else "thermal off")
    radar_report = (
        f"radar ok={reads_ok}/{radar_total} ({radar_pct:.0f}%) "
        f"none={reads_none} sampled={sampled} stale={stale} · "
        f"obj zero={obj_zero} capped={obj_capped} bad={obj_bad} "
        f"logged={obj_logged}"
        if RADAR_ENABLE else "radar off")
    if gps is not None and gps_rows == 0:
        print(f"[chunk] WARN gps wrote 0 rows: no position from any source "
              f"({gps.describe()})", flush=True)
    gps_report = (f"gps rows={gps_rows} src={gps_source}"
                  if gps is not None else "gps off")
    print(
        f"[chunk] done {chunk_ts}: {n_frames} loops · "
        f"{thermal_report} · {radar_report} · "
        f"imu logged={imu_sampled} stale={imu_stale} rx={imu_reads} "
        f"conn={imu_conn} · {gps_report}",
        flush=True,
    )
    return n_frames


def _log_time_source():
    """Record the wall-clock source at boot so chunk timestamps are auditable
    offline. On the Pi 5 the system clock is backed by the on-board RTC (local,
    no NTP/BLE dependency); this logs what we booted with and loudly flags a
    clearly-unset clock so a bad-time chunk is never silently mis-stamped."""
    now = datetime.now()
    rtc = "present" if os.path.exists("/dev/rtc0") else "ABSENT"
    print(f"[boot] wall-clock {now.strftime('%Y-%m-%d %H:%M:%S')} "
          f"(local RTC /dev/rtc0: {rtc})", flush=True)
    if now.year < 2024:
        print(f"[boot] WARN system clock looks UNSET (year {now.year}): chunk "
              f"timestamps will be wrong until the RTC/NTP sets it; check the "
              f"RTC coin cell and `timedatectl`.", flush=True)


def session_capture_dir(base, stamp=None):
    """The per-boot session folder under ``base``: one service start = one
    folder, named like the chunks it will contain (``%Y-%m-%d_%H-%M-%S``).
    Pure given ``stamp``; ``main()`` passes None to stamp with boot time."""
    if stamp is None:
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    return Path(base) / stamp


def main():
    global CAPTURE_DIR
    print(f"[boot] continuous_capture starting", flush=True)
    if SESSION_SUBDIR:
        CAPTURE_DIR = session_capture_dir(CAPTURE_DIR)
    print(f"[boot] capture dir: {CAPTURE_DIR}"
          f"{'' if SESSION_SUBDIR else '  [flat: ASVPROJECT_SESSION_SUBDIR=0]'}",
          flush=True)
    print(f"[boot] chunk seconds: {CHUNK_SECONDS}", flush=True)
    print(f"[boot] fps target: {FPS_TARGET}", flush=True)
    print(f"[boot] sensors: fisheye=on radar={'on' if RADAR_ENABLE else 'OFF'} "
          f"thermal={'on' if THERMAL_ENABLE else 'OFF'}"
          f"{f' rot{THERMAL_ROTATE_DEG}' if THERMAL_ROTATE_DEG else ''} "
          f"imu={IMU_BACKEND if IMU_ENABLE else 'OFF'} "
          f"gps={'on' if GPS_ENABLE else 'OFF'}"
          f"{'  [FISHEYE-ONLY]' if FISHEYE_ONLY else ''}", flush=True)
    print(f"[boot] snapshot dir: {SNAPSHOT_DIR or 'OFF'}", flush=True)
    _log_time_source()
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    if RADAR_ENABLE:
        write_radar_profile(CAPTURE_DIR, CONFIG_FILE)

    fish, therm, radar, imu, gps = open_sensors()
    try:
        while not _stop:
            free = disk_free_gb(CAPTURE_DIR)
            if free < MIN_FREE_GB:
                print(f"[loop] WARN disk free {free:.1f} GB < "
                      f"{MIN_FREE_GB} GB: sleeping 60 s", flush=True)
                time.sleep(60)
                continue
            try:
                write_one_chunk(fish, therm, radar, imu, gps)
            except Exception as e:
                print(f"[loop] chunk failed: {e!r}: sleeping 5 s "
                      f"before retry", flush=True)
                time.sleep(5)
    finally:
        close_sensors(fish, therm, radar, imu, gps)
        print("[done] exiting cleanly", flush=True)


if __name__ == "__main__":
    main()
