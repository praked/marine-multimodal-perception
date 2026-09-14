"""BNO085 attitude source. Its pitch/roll feed the monocular water-plane range
estimator (geometry.range_from_water_plane_up), which otherwise falls back to
the detected horizon / level assumption; yaw/heading is exposed for later use.

As of 2026-07-14 the BNO085 is **wired directly into SensorBox (Pi 5)** and
read over **UART in RVC mode**; that is the default. The older BLE path (IMU on
boat1, bridged over Bluetooth) is kept as a selectable backend and doubles as the
template for the planned GPS-over-BLE reader. Backends behind one interface:

  - `uart_rvc`: **default.** Directly-wired BNO085 in UART-RVC mode: PS1=Low/
                   PS0=High, sensor TX (SDA pad) -> Pi pin 10 (GPIO15/RXD),
                   /dev/ttyAMA0 @115200. Sensor free-runs 19-byte frames at
                   100 Hz (no host commands). Needs `pyserial`. See
                   docs/reference/imu_uart_rvc.md (incl. the abandoned-SPI
                   post-mortem; SPI's reset-gated write window was unopenable).
  - `simulation`: synthesizes gentle roll/pitch; needs no hardware.
  - `ble`: BNO085 on boat1 bridged over a BLE notify characteristic via
                   `bleak`; address/UUIDs + packet layout from configs/imu.yaml.
  - `serial`: packets over a serial (e.g. RFCOMM) port via `pyserial`.

Self-test:
    python -m scripts.sensor_processing.imu_bno085 --backend uart_rvc --monitor   # on the Pi
    python -m scripts.sensor_processing.imu_bno085 --backend simulation --monitor # anywhere
"""

from __future__ import annotations

import argparse
import math
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import yaml

from scripts.utils.geometry import up_from_pitch_roll

IMU_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "imu.yaml"


# ---------------------------------------------------------------------------
# Attitude value type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Attitude:
    """A single attitude sample, in radians, in the BOAT body frame.

    Mount alignment: the BNO085 is not the camera. `mount_*_offset_rad` in
    configs/imu.yaml are added so the reported angles are the CAMERA's
    pitch/roll (measure the residual on land with the boat level).
    """
    pitch_rad: float
    roll_rad: float
    yaw_rad: float = 0.0
    timestamp: float | None = None
    source: str = "bno085"

    @property
    def pitch_deg(self) -> float:
        return math.degrees(self.pitch_rad)

    @property
    def roll_deg(self) -> float:
        return math.degrees(self.roll_rad)

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw_rad)

    def up_vector_camera(self) -> np.ndarray:
        """World-up in camera coords (feeds geometry.range_from_water_plane_up)."""
        return up_from_pitch_roll(self.pitch_rad, self.roll_rad)


# ---------------------------------------------------------------------------
# Pure conversions (final + tested; transport-independent)
# ---------------------------------------------------------------------------

def quaternion_to_euler(w: float, x: float, y: float, z: float
                        ) -> tuple[float, float, float]:
    """Quaternion (w, x, y, z) -> (roll, pitch, yaw) in radians (ZYX / aero).

    The BNO085 rotation-vector report is a unit quaternion; this is the
    standard Tait-Bryan extraction with pitch clamped at the +/-90 deg
    singularity (gimbal lock).
    """
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return 0.0, 0.0, 0.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis), clamped at the poles
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def parse_packet(data: bytes, fmt: str = "quat_f32_le") -> Attitude | None:
    """Decode one Bluetooth packet into an Attitude.

    Supported `fmt` values (selectable in configs/imu.yaml). CONFIRM which one
    the BNO085 bridge firmware actually emits before the lake test:

      - "quat_f32_le": 16 bytes = w,x,y,z float32 little-endian (e.g. a
        bridge forwarding the rotation-vector quaternion verbatim).
      - "euler_f32_le_deg": 12 bytes = roll,pitch,yaw float32 LE in degrees.
      - "csv": ASCII "roll,pitch,yaw\\n" in degrees (easy to eyeball / from an
        Arduino Serial.print bridge).

    Returns None on a malformed/short packet so the reader can resync rather
    than crash mid-mission.
    """
    try:
        if fmt == "quat_f32_le":
            if len(data) < 16:
                return None
            w, x, y, z = struct.unpack_from("<ffff", data, 0)
            roll, pitch, yaw = quaternion_to_euler(w, x, y, z)
            return Attitude(pitch_rad=pitch, roll_rad=roll, yaw_rad=yaw)
        if fmt == "euler_f32_le_deg":
            if len(data) < 12:
                return None
            roll, pitch, yaw = struct.unpack_from("<fff", data, 0)
            return Attitude(pitch_rad=math.radians(pitch),
                            roll_rad=math.radians(roll),
                            yaw_rad=math.radians(yaw))
        if fmt == "csv":
            parts = data.decode("ascii", "ignore").strip().split(",")
            if len(parts) < 3:
                return None
            roll, pitch, yaw = (float(parts[0]), float(parts[1]), float(parts[2]))
            return Attitude(pitch_rad=math.radians(pitch),
                            roll_rad=math.radians(roll),
                            yaw_rad=math.radians(yaw))
    except (struct.error, ValueError):
        return None
    raise ValueError(f"unknown packet fmt: {fmt!r}")


# ---------------------------------------------------------------------------
# UART-RVC frame decode (the box-mounted BNO085's default-since-2026-07-14 path)
# ---------------------------------------------------------------------------
#
# The IMU is now wired directly to SensorBox over UART in "RVC" (Robot
# Vacuum Cleaner) mode: PS1=Low/PS0=High, sensor TX (the SDA pad) -> Pi
# pin 10 (GPIO15/RXD), read from /dev/ttyAMA0 @115200 8N1. In this mode the
# sensor free-runs a fixed 19-byte frame at 100 Hz with NO host commands: the
# whole reason we use it (SPI needed a reset-gated write window we could never
# open; see docs/reference/imu_uart_rvc.md for the full post-mortem).
#
# Frame (little-endian): 0xAA 0xAA | index | yaw i16 | pitch i16 | roll i16
# | ax i16 | ay i16 | az i16 | MI | MR | RSVD | checksum. Angles are 0.01 deg
# / LSB; acceleration is 1 mg / LSB. Checksum = sum(bytes[2:18]) & 0xFF.

RVC_FRAME_LEN = 19
RVC_HEADER = b"\xaa\xaa"
_RVC_ACC_MS2 = 9.80665 / 1000.0   # 1 mg/LSB -> m/s^2


def decode_rvc_frame(frame: bytes) -> tuple[float, float, float, float, float, float] | None:
    """Decode one 19-byte UART-RVC frame.

    Returns ``(yaw_deg, pitch_deg, roll_deg, ax, ay, az)``: angles in degrees,
    acceleration in m/s^2, or None on a bad header/short frame/checksum
    mismatch so a stream reader can drop the byte and resync.
    """
    if (len(frame) < RVC_FRAME_LEN
            or frame[0] != 0xAA or frame[1] != 0xAA
            or (sum(frame[2:18]) & 0xFF) != frame[18]):
        return None
    yaw, pitch, roll, ax, ay, az = struct.unpack_from("<hhhhhh", frame, 3)
    return (yaw * 0.01, pitch * 0.01, roll * 0.01,
            ax * _RVC_ACC_MS2, ay * _RVC_ACC_MS2, az * _RVC_ACC_MS2)


def parse_rvc_frame(frame: bytes) -> Attitude | None:
    """Decode a UART-RVC frame into an Attitude (angles only; accel dropped).

    RVC already emits fused Tait-Bryan angles, so this bypasses
    ``quaternion_to_euler`` entirely. Returns None on a malformed frame.
    """
    decoded = decode_rvc_frame(frame)
    if decoded is None:
        return None
    yaw, pitch, roll, _ax, _ay, _az = decoded
    return Attitude(pitch_rad=math.radians(pitch),
                    roll_rad=math.radians(roll),
                    yaw_rad=math.radians(yaw),
                    source="bno085:uart_rvc")


# ---------------------------------------------------------------------------
# Reader: maintains the latest Attitude on a background thread
# ---------------------------------------------------------------------------

class BNO085Reader:
    """Holds the latest Attitude; `get()` is what the pipeline polls.

    Thread-safe and non-blocking: the pipeline never waits on Bluetooth. If no
    sample has arrived within `stale_after_s`, `get()` returns None and range
    estimation falls back to the detected horizon / level assumption.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_imu_config()
        self.backend = str(self.config.get("backend", "simulation"))
        self.packet_fmt = str(self.config.get("packet_fmt", "quat_f32_le"))
        self.stale_after_s = float(self.config.get("stale_after_s", 0.5))
        # uart_rvc backend (the default, directly-wired box IMU)
        self.uart_port = str(self.config.get("uart_port", "/dev/ttyAMA0"))
        self.uart_baud = int(self.config.get("uart_baud", 115200))
        mount = self.config.get("mount_offset_rad", {}) or {}
        self._mount_pitch = float(mount.get("pitch", 0.0))
        self._mount_roll = float(mount.get("roll", 0.0))
        # "euler" publishes the backend's raw angles (+ offsets); "rotation"
        # rebuilds the rotation and re-extracts pitch/roll/heading in the
        # CAMERA frame -- required for the box mount, whose raw Euler angles
        # sit beside gimbal lock (same knobs, and the same composition code,
        # as the replay provider in imu_replay: live/replay parity).
        self._attitude_source = str(self.config.get("attitude_source", "euler"))
        self._invert_lateral = bool(self.config.get("mount_invert_lateral", True))

        self._lock = threading.Lock()
        self._latest: Attitude | None = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "BNO085Reader":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "BNO085Reader":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def connected(self) -> bool:
        return self._connected

    # -- polling interface (used by the pipeline) ---------------------------

    def get(self) -> Attitude | None:
        """Latest attitude, or None if stale / never received."""
        with self._lock:
            att = self._latest
        if att is None:
            return None
        if att.timestamp is not None and (time.monotonic() - att.timestamp) > self.stale_after_s:
            return None
        return att

    def _publish(self, att: Attitude) -> None:
        pitch, roll, yaw = att.pitch_rad, att.roll_rad, att.yaw_rad
        # The simulation backend synthesizes CAMERA-frame attitude directly,
        # so composition (which expects raw box-mount sensor angles) must
        # not touch it.
        if self._attitude_source == "rotation" and self.backend != "simulation":
            # Deferred import: imu_replay imports this module at top level.
            from scripts.sensor_processing.imu_replay import (
                camera_heading_from_rvc,
                camera_pitch_roll_from_rvc,
            )
            yaw = float(camera_heading_from_rvc(yaw, pitch, roll))
            pitch, roll = camera_pitch_roll_from_rvc(
                pitch, roll, invert_lateral=self._invert_lateral)
        att = Attitude(
            pitch_rad=float(pitch) + self._mount_pitch,
            roll_rad=float(roll) + self._mount_roll,
            yaw_rad=float(yaw),
            timestamp=time.monotonic(),
            source=f"bno085:{self.backend}",
        )
        with self._lock:
            self._latest = att

    # -- backends -----------------------------------------------------------

    def _run(self) -> None:
        try:
            if self.backend == "simulation":
                self._run_simulation()
            elif self.backend == "uart_rvc":
                self._run_uart_rvc()
            elif self.backend == "ble":
                self._run_ble()
            elif self.backend == "serial":
                self._run_serial()
            else:
                raise ValueError(f"unknown IMU backend: {self.backend!r}")
        except Exception as exc:  # noqa: BLE001 - surface, don't kill the boat
            print(f"[imu] backend {self.backend!r} stopped: {exc}")
            self._connected = False

    def _run_simulation(self) -> None:
        """Synthesize a gentle seaway: ~5 deg roll @ 0.2 Hz, ~2 deg pitch."""
        hz = float(self.config.get("simulation_hz", 50.0))
        roll_amp = math.radians(float(self.config.get("simulation_roll_deg", 5.0)))
        pitch_amp = math.radians(float(self.config.get("simulation_pitch_deg", 2.0)))
        period = float(self.config.get("simulation_period_s", 5.0))
        self._connected = True
        t0 = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic() - t0
            phase = 2.0 * math.pi * t / period
            self._publish(Attitude(
                pitch_rad=pitch_amp * math.sin(phase * 1.3),
                roll_rad=roll_amp * math.sin(phase),
                yaw_rad=0.0,
            ))
            time.sleep(1.0 / hz)

    def _run_uart_rvc(self) -> None:
        """Read the directly-wired BNO085's UART-RVC stream (the default).

        Needs `pyserial`. Syncs to the ``0xAA 0xAA`` header and decodes fixed
        19-byte frames back-to-back; each valid frame publishes an Attitude.
        A malformed frame just shifts one byte and resyncs, so line noise can
        never wedge the reader.
        """
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "uart_rvc backend needs `pyserial` (pip install pyserial). Use "
                "--backend simulation to test without hardware."
            ) from exc

        with serial.Serial(self.uart_port, self.uart_baud, timeout=1.0) as ser:
            self._connected = True
            buf = bytearray()
            while not self._stop.is_set():
                chunk = ser.read(64)
                if not chunk:
                    continue
                buf.extend(chunk)
                # Parse every complete frame currently buffered.
                while len(buf) >= RVC_FRAME_LEN:
                    if buf[0] != 0xAA or buf[1] != 0xAA:
                        del buf[0]          # not a header: realign
                        continue
                    att = parse_rvc_frame(bytes(buf[:RVC_FRAME_LEN]))
                    if att is None:
                        del buf[0]          # bad checksum/false header: realign
                        continue
                    del buf[:RVC_FRAME_LEN]
                    self._publish(att)

    def _run_ble(self) -> None:
        """Subscribe to the BNO085 bridge's BLE notify characteristic.

        Needs `bleak`. The address + UUIDs live in configs/imu.yaml and MUST be
        confirmed on land (use a BLE scanner / `bleak` discovery, then check the
        packet_fmt by eyeballing `--monitor`).
        """
        try:
            import asyncio

            from bleak import BleakClient  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ble backend needs `bleak` (pip install bleak). Use "
                "--backend simulation to test without hardware."
            ) from exc

        address = self.config.get("ble_address")
        char_uuid = self.config.get("ble_char_uuid")
        if not address or not char_uuid:
            raise RuntimeError(
                "set ble_address and ble_char_uuid in configs/imu.yaml "
                "(discover them on land with a BLE scanner)."
            )

        def on_notify(_handle: int, data: bytearray) -> None:
            att = parse_packet(bytes(data), self.packet_fmt)
            if att is not None:
                self._publish(att)

        async def loop() -> None:
            async with BleakClient(address) as client:
                self._connected = client.is_connected
                await client.start_notify(char_uuid, on_notify)
                while not self._stop.is_set():
                    await asyncio.sleep(0.1)
                await client.stop_notify(char_uuid)

        asyncio.run(loop())

    def _run_serial(self) -> None:
        """Read length-delimited packets over a Bluetooth serial port.

        Needs `pyserial`. Assumes one packet per line for `csv`, or fixed-width
        frames for the binary formats (adjust `serial_frame_bytes`).
        """
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "serial backend needs `pyserial` (pip install pyserial)."
            ) from exc

        port = self.config.get("serial_port")
        baud = int(self.config.get("serial_baud", 115200))
        if not port:
            raise RuntimeError("set serial_port in configs/imu.yaml")
        frame_bytes = int(self.config.get("serial_frame_bytes", 0))

        with serial.Serial(port, baud, timeout=1.0) as ser:
            self._connected = True
            while not self._stop.is_set():
                data = ser.readline() if frame_bytes == 0 else ser.read(frame_bytes)
                if not data:
                    continue
                att = parse_packet(bytes(data), self.packet_fmt)
                if att is not None:
                    self._publish(att)


# ---------------------------------------------------------------------------
# Config + CLI self-test
# ---------------------------------------------------------------------------

def load_imu_config(path: str | Path = IMU_CONFIG_PATH) -> dict[str, Any]:
    """Load configs/imu.yaml, with safe defaults if it is absent."""
    defaults: dict[str, Any] = {
        "enabled": False,
        "backend": "simulation",
        "packet_fmt": "quat_f32_le",
        "stale_after_s": 0.5,
        "uart_port": "/dev/ttyAMA0",
        "uart_baud": 115200,
        "mount_offset_rad": {"pitch": 0.0, "roll": 0.0},
    }
    if not Path(path).exists():
        return defaults
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    defaults.update(raw)
    return defaults


def _monitor(reader: BNO085Reader, hz: float = 5.0) -> None:
    print(f"[imu] backend={reader.backend} fmt={reader.packet_fmt}: Ctrl-C to stop")
    print(f"{'pitch°':>8} {'roll°':>8} {'yaw°':>8}  up_camera (x,y,z)  source")
    try:
        with reader:
            while True:
                att = reader.get()
                if att is None:
                    status = "connected" if reader.connected else "connecting"
                    print(f"   ...waiting for samples ({status})", end="\r")
                else:
                    up = att.up_vector_camera()
                    print(f"{att.pitch_deg:8.2f} {att.roll_deg:8.2f} {att.yaw_deg:8.2f}  "
                          f"({up[0]:+.2f},{up[1]:+.2f},{up[2]:+.2f})  {att.source}")
                time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        print("\n[imu] stopped.")


def _ble_scan(seconds: float = 8.0) -> int:
    """List nearby BLE devices (find the BNO085 bridge's address)."""
    try:
        import asyncio
        from bleak import BleakScanner
    except ImportError as exc:
        print(f"need bleak: pip install bleak ({exc})")
        return 1

    async def run():
        print(f"scanning {seconds:.0f}s for BLE devices...")
        devs = await BleakScanner.discover(timeout=seconds)
        for d in sorted(devs, key=lambda x: -(x.rssi or -999)):
            print(f"  {d.address}  rssi={getattr(d, 'rssi', '?'):>4}  {d.name or '(no name)'}")
        print(f"\n{len(devs)} devices. Use --inspect --address <addr> to list its characteristics.")
    asyncio.run(run())
    return 0


def _ble_inspect(address: str) -> int:
    """List a device's services/characteristics (find the notify UUID)."""
    try:
        import asyncio
        from bleak import BleakClient
    except ImportError as exc:
        print(f"need bleak: pip install bleak ({exc})")
        return 1

    async def run():
        async with BleakClient(address) as c:
            print(f"connected to {address}\n")
            for s in c.services:
                print(f"service {s.uuid}")
                for ch in s.characteristics:
                    print(f"  char {ch.uuid}  props={','.join(ch.properties)}")
    asyncio.run(run())
    return 0


def _ble_sniff(address: str, char_uuid: str, seconds: float = 10.0) -> int:
    """Dump raw notification bytes from a characteristic + guess the format.

    Use this to confirm/adjust `packet_fmt` against the real BNO085 bridge:
    watch the byte length and whether a quat/euler parse yields sane angles.
    """
    try:
        import asyncio
        from bleak import BleakClient
    except ImportError as exc:
        print(f"need bleak: pip install bleak ({exc})")
        return 1

    def on_notify(_h, data: bytearray):
        b = bytes(data)
        guesses = []
        for fmt in ("quat_f32_le", "euler_f32_le_deg", "csv"):
            att = parse_packet(b, fmt)
            if att is not None:
                guesses.append(f"{fmt}->p{att.pitch_deg:.0f} r{att.roll_deg:.0f} y{att.yaw_deg:.0f}")
        print(f"  {len(b):3d}B  {b[:20].hex(' ')}{'...' if len(b) > 20 else ''}"
              f"   [{' | '.join(guesses) or 'no fmt matched'}]")

    async def run():
        async with BleakClient(address) as c:
            print(f"sniffing {char_uuid} on {address} for {seconds:.0f}s "
                  f"(move the boat/board to see angles change)...")
            await c.start_notify(char_uuid, on_notify)
            await asyncio.sleep(seconds)
            await c.stop_notify(char_uuid)
    asyncio.run(run())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["simulation", "uart_rvc", "ble", "serial"],
                    default=None, help="Override configs/imu.yaml backend.")
    ap.add_argument("--monitor", action="store_true",
                    help="Stream live attitude to the terminal (on-land self-test).")
    ap.add_argument("--hz", type=float, default=5.0, help="Monitor print rate.")
    # --- BLE discovery (find + characterise the boat's BNO085 stream) ---
    ap.add_argument("--scan", action="store_true", help="List nearby BLE devices.")
    ap.add_argument("--inspect", action="store_true",
                    help="List services/characteristics of --address.")
    ap.add_argument("--sniff", action="store_true",
                    help="Dump raw bytes from --address/--char + guess packet_fmt.")
    ap.add_argument("--address", default=None, help="BLE device address (for inspect/sniff).")
    ap.add_argument("--char", default=None, help="characteristic UUID (for sniff).")
    ap.add_argument("--seconds", type=float, default=10.0, help="scan/sniff duration.")
    args = ap.parse_args(argv)

    if args.scan:
        return _ble_scan(args.seconds)
    if args.inspect:
        if not args.address:
            raise SystemExit("--inspect needs --address")
        return _ble_inspect(args.address)
    if args.sniff:
        if not (args.address and args.char):
            raise SystemExit("--sniff needs --address and --char")
        return _ble_sniff(args.address, args.char, args.seconds)

    cfg = load_imu_config()
    if args.backend:
        cfg["backend"] = args.backend
    reader = BNO085Reader(cfg)

    if args.monitor:
        _monitor(reader, hz=args.hz)
        return 0

    # Default: a brief connectivity check.
    with reader:
        time.sleep(1.0)
        att = reader.get()
    if att is None:
        print(f"[imu] no samples from backend {reader.backend!r} "
              f"(connected={reader.connected}). Run with --monitor to debug.")
        return 1
    print(f"[imu] OK: pitch={att.pitch_deg:.1f} roll={att.roll_deg:.1f} "
          f"yaw={att.yaw_deg:.1f} source={att.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
