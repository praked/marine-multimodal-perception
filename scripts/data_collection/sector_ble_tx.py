"""BLE sector-stream transmitter: tail a shadow/sectors JSONL, notify boat1.

Closes the box->autopilot loop at the TRANSPORT level: the obstacle box
(SensorBox) advertises a GATT service and streams each new sectors record
(written by `scripts/data_collection/shadow_fusion.py`, or any `fusion.py
--out` JSONL) as compact packets over a notify characteristic. boat1 runs the
matching central (`sector_rx.py` on the boatv1-boat-b `sector-rx` branch)
and logs what it receives. Processing on the autopilot side is out of scope.

Design rules (mirroring shadow's own constraints):

* READ-ONLY tail of the JSONL, never blocking the writer — the same
  seek/tell + partial-line pattern as shadow_fusion's `_CsvTail`, applied to
  lines. Shadow flushes per record, so tail latency is one poll interval.
* Malformed or unencodable lines are SKIPPED and counted, never fatal.
* The BLE layer failing (no adapter, no subscriber, disconnects) never stops
  the tailer: we advertise forever, notifications to nobody are dropped by
  BlueZ, and a returning subscriber picks the stream up mid-flight (each
  packet is self-contained; `seq` exposes the gap).
* Heartbeat line every ~30 s: records sent / skipped, subscriber yes/no.

Library choice — `bluezero` (peripheral role), imported LAZILY:

* The peripheral/GATT-server role on Linux means driving BlueZ's D-Bus
  GATT-manager API. `bluezero` is the de-facto Raspberry Pi wrapper for
  exactly that (python3-dbus + python3-gi under the hood, both plain apt
  packages on Raspberry Pi OS trixie; bluezero itself is `pip install
  bluezero` or apt `python3-bluezero` where packaged).
* `bless` (the async cross-platform alternative) was considered and passed
  over: its Linux backend is younger, pulls its own dbus-fast event loop,
  and we have no cross-platform need — the sender only ever runs on the Pi.
* `bleak` is central-only, so it cannot serve this side; it IS the right
  tool for the receiver (boatv1 already uses it in
  `sensors/wind_sensor_ble.py`), and for the box's existing central-role
  scaffolds (`imu_bno085 --scan`, gps_boat1's ble fallback).

Threading: bluezero's `publish()` owns the process main thread (GLib main
loop). The tailer runs on a daemon thread and only ever appends fragments to
a deque; a 50 ms GLib timer drains the deque and calls `set_value` from the
main-loop thread, so every D-Bus call happens on the GLib thread.

Wire format: `scripts/data_collection/sector_codec.py` (the source of truth,
vendored by the receiver) — documented in `docs/reference/sector_protocol.md`
under "BLE transport". Default notification size is the 20-byte ATT floor
(3 notifications per 46-byte record at 3 fps = 9/s, trivial for BLE);
`--notify-bytes 180` once the BlueZ<->BlueZ MTU (typically 517) is confirmed
on the boat makes it one notification per record.

Usage::

    # on the box, beside shadow (auto-picks the newest shadow_*.jsonl):
    python3 -m scripts.data_collection.sector_ble_tx \
        --session-dir ~/captures/<session>

    # explicit file:
    python3 -m scripts.data_collection.sector_ble_tx --tail <path>.jsonl

    # laptop, no bluetooth: encoder + fake transport end-to-end:
    python3 -m scripts.data_collection.sector_ble_tx \
        --tail results/shadow/shadow_x.jsonl --loopback --idle-timeout 2
"""

from __future__ import annotations

import argparse
import collections
import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:      # runnable by path, like the capture unit
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_collection.sector_codec import (  # noqa: E402
    CHAR_UUID,
    LOCAL_NAME,
    SAFE_NOTIFY_BYTES,
    SERVICE_UUID,
    CodecError,
    Reassembler,
    decode_packet,
    encode_record,
    fragment,
)

HEARTBEAT_S = 30.0
POLL_S = 0.1
RESOLVE_S = 2.0          # session-dir mode: re-check for a newer shadow file


def _log(msg: str) -> None:
    print(f"[ble-tx] {msg}", flush=True)


# ---------------------------------------------------------------------------
# JSONL tail (read-only; complete lines only)
# ---------------------------------------------------------------------------

class JsonlTail:
    """Incremental read-only tail of a growing JSONL file.

    Same contract as shadow_fusion._CsvTail: seek to the last position, read
    what is new, hold any trailing partial line until its newline arrives.
    ``from_end=True`` starts at the current end of file (skip history).
    """

    def __init__(self, path, from_end: bool = False):
        self.path = Path(path)
        self._pos = 0
        self._partial = ""
        if from_end:
            try:
                self._pos = self.path.stat().st_size
            except OSError:
                self._pos = 0

    def read_new_lines(self) -> list[str]:
        try:
            with open(self.path, "r") as fh:
                fh.seek(self._pos)
                chunk = fh.read()
                self._pos = fh.tell()
        except OSError:
            return []
        if not chunk:
            return []
        text = self._partial + chunk
        lines = text.split("\n")
        self._partial = lines.pop()
        return [ln for ln in (l.strip("\r") for l in lines) if ln]


def resolve_shadow_jsonl(session_dir) -> Path | None:
    """Newest shadow_*.jsonl in a session dir (one subdir level, like
    shadow's own TailSources._newest)."""
    d = Path(session_dir)
    hits = sorted(d.glob("shadow_*.jsonl")) + sorted(d.glob("*/shadow_*.jsonl"))
    return max(hits, key=lambda p: p.name) if hits else None


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------

class LoopbackTransport:
    """Fake in-process transport: fragments go straight into a Reassembler
    and decoded records are collected (and printed). Tests + laptop demo."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.reassembler = Reassembler()
        self.records: list[dict] = []
        self.decode_errors = 0

    @property
    def subscribed(self) -> bool:
        return True

    def notify(self, data: bytes) -> None:
        packet = self.reassembler.feed(data)
        if packet is None:
            return
        try:
            rec = decode_packet(packet)
        except CodecError as e:
            self.decode_errors += 1
            _log(f"loopback decode error: {e}")
            return
        self.records.append(rec)
        if self.verbose:
            _log(f"loopback rx seq={rec['seq']} ts={rec['timestamp']} "
                 f"scores={rec['scores']}")

    def serve_forever(self) -> None:      # symmetry with the real transport
        while True:
            time.sleep(1)


def _mgmt_is_advertising() -> bool:
    """True when the controller is advertising OR already has a central
    connected (a legacy peripheral stops advertising the moment a central
    connects — boat1 was in within the first second, which read as
    "not advertising" until this accepted connections as success)."""
    for line in _btmgmt("info", timeout_s=5.0).splitlines():
        if "current settings" in line:
            if "advertising" in line.split(":", 1)[-1].split():
                return True
            break
    con = _btmgmt("con", timeout_s=5.0)
    return any(ln.strip() and "type" in ln for ln in con.splitlines())


def _btmgmt(*args: str, timeout_s: float = 10.0) -> str:
    """Run one `btmgmt` command (passwordless sudo) and return its output.

    btmgmt 5.82 hangs when its stdin is at EOF (/dev/null under systemd or
    nohup) and exits promptly when stdin is an open pipe — measured on the
    box 2026-09-03. So hold a pipe open, read stdout until it exits or the
    deadline passes, then reap it. Errors come back as text, never raise.
    """
    try:
        proc = subprocess.Popen(["sudo", "-n", "btmgmt", *args],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
    except OSError as e:
        return f"spawn failed: {e!r}"
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        out = (out or "") + " [btmgmt timed out]"
    return out or ""


class BluezeroTransport:
    """GATT peripheral over BlueZ via bluezero. All D-Bus work happens on
    the GLib main-loop thread (serve_forever); notify() only queues."""

    QUEUE_MAX = 64        # ~2 s of records at 3 fps x 3 fragments

    def __init__(self, adapter_address: str | None = None,
                 local_name: str = LOCAL_NAME):
        try:
            from bluezero import adapter, peripheral  # noqa: PLC0415
        except ImportError as e:
            raise SystemExit(
                "sector_ble_tx needs bluezero for the real BLE transport "
                "(sudo apt install python3-dbus python3-gi && pip install "
                "bluezero). Use --loopback on a laptop. "
                f"({e})") from e
        self._pending: collections.deque[bytes] = collections.deque(
            maxlen=self.QUEUE_MAX)
        self.subscribed = False
        self._char = None
        if adapter_address is None:
            adapters = list(adapter.Adapter.available())
            if not adapters:
                raise SystemExit("no bluetooth adapter found")
            adapter_address = adapters[0].address
        self._peripheral = peripheral.Peripheral(adapter_address,
                                                 local_name=local_name)
        self._peripheral.add_service(srv_id=1, uuid=SERVICE_UUID, primary=True)
        self._peripheral.add_characteristic(
            srv_id=1, chr_id=1, uuid=CHAR_UUID, value=[],
            notifying=False, flags=["notify"],
            notify_callback=self._on_notify_state,
            read_callback=None, write_callback=None)
        _log(f"peripheral on {adapter_address} as {local_name!r} "
             f"service {SERVICE_UUID}")
        # bluezero reports a failed LEAdvertisingManager1 registration by
        # PRINTING from a module-level callback (looked up at call time) and
        # keeps the GLib loop + GATT app alive. Hook it so we can fall back.
        from bluezero import advertisement  # noqa: PLC0415
        advertisement.register_ad_error_cb = self._advert_failed
        self._mgmt_instance: int | None = None

    #: Legacy-MGMT advertising instance used by the fallback (bluetoothd
    #: takes low numbers for its own; the BCM4345 offers 5).
    MGMT_ADV_INSTANCE = 4

    def _advert_failed(self, error) -> None:
        """Main-loop thread. Raspberry Pi kernel 6.18.34 rejects bluetoothd's
        `Add Extended Advertising Data` (MGMT length check, raspberrypi/linux
        #7473; measured 2026-09-03: bluetoothd sends plen 14 for a 3-byte
        payload, btmgmt's plen 6 is accepted) so nothing goes on air while the
        GATT service itself registers fine. The legacy `Add Advertising` path
        still works on that kernel, so register the same service UUID through
        `btmgmt` (needs the box's passwordless sudo). The kernel re-enables
        MGMT instances after a central disconnects, exactly as it does for
        bluetoothd's own."""
        _log(f"WARN D-Bus advertisement registration failed ({error}); "
             f"falling back to legacy MGMT advertising via btmgmt")
        # No pre-emptive rm-adv: removing an absent instance just errors, and
        # add-adv on an existing instance number replaces it anyway. Verify
        # that the controller is REALLY advertising afterwards and retry: a
        # previous tx still tearing down (its rm-adv can land after our
        # add-adv — measured 2026-09-03: "Instance added" yet no
        # `advertising` in the current settings, boat1 scanning into the
        # void) must not leave us silent.
        for attempt in range(4):
            out = _btmgmt("add-adv", "-c", "-g", "-u", SERVICE_UUID,
                          str(self.MGMT_ADV_INSTANCE))
            if "Instance added" not in out:
                _log(f"WARN btmgmt add-adv: {out.strip()!r}")
            time.sleep(1.0)
            if _mgmt_is_advertising():
                self._mgmt_instance = self.MGMT_ADV_INSTANCE
                _log(f"advertising via legacy MGMT instance "
                     f"{self._mgmt_instance} (service UUID only; the "
                     f"receiver matches on UUID; attempt {attempt + 1})")
                return
            time.sleep(2.0)
        _log("WARN legacy MGMT advertising did not come up after 4 "
             "attempts — not advertising")

    def close(self) -> None:
        if self._mgmt_instance is not None:
            _btmgmt("rm-adv", str(self._mgmt_instance), timeout_s=5.0)
            self._mgmt_instance = None

    def _on_notify_state(self, notifying, characteristic) -> None:
        self.subscribed = bool(notifying)
        self._char = characteristic
        _log("subscriber connected" if notifying else "subscriber gone")

    def _flush(self, *_args) -> bool:
        """GLib timer body (main-loop thread): drain queued fragments."""
        while self._pending:
            data = self._pending.popleft()
            if self.subscribed and self._char is not None:
                try:
                    self._char.set_value(list(data))
                except Exception as e:      # noqa: BLE001 — keep serving
                    _log(f"WARN notify failed: {e!r}")
        return True                          # keep the timer alive

    def notify(self, data: bytes) -> None:
        if self.subscribed:                  # nobody listening -> drop cheaply
            self._pending.append(data)

    def serve_forever(self) -> None:
        from bluezero import async_tools  # noqa: PLC0415
        from gi.repository import GLib  # noqa: PLC0415
        async_tools.add_timer_ms(50, self._flush)
        # GLib's loop does not deliver Python's KeyboardInterrupt: without
        # this, Ctrl-C / SIGTERM leave the process alive with its GATT app
        # registered, and the NEXT tx adds a second copy of the characteristic
        # ("Multiple Characteristics with this UUID" on every central; three
        # zombie transmitters found on the box 2026-09-03).
        for sig in (signal.SIGINT, signal.SIGTERM):
            GLib.unix_signal_add(GLib.PRIORITY_HIGH, sig, self._stop, sig)
        self._peripheral.publish()           # blocks in the GLib main loop

    def _stop(self, sig) -> bool:
        _log(f"signal {signal.Signals(sig).name}: stopping")
        # Advertisement first, THEN the centrals: Device1.Disconnect blocks
        # until the link is down, and a successor tx started meanwhile would
        # have its fresh instance deleted by our late rm-adv.
        self.close()
        self._disconnect_centrals()
        self._peripheral.mainloop.quit()
        return False

    def _disconnect_centrals(self) -> None:
        """Drop every connected central before the GATT app goes away.

        bluetoothd keeps the LE link up when the process owning the GATT
        application exits; a subscribed central then sits "connected" to a
        characteristic that no longer exists and never re-scans (measured
        2026-09-03 with the boat1 receiver: records frozen, no disconnect
        event, until the link was dropped by hand). Disconnecting here turns
        a restart into the disconnect the receiver already handles."""
        try:
            import dbus  # noqa: PLC0415
            bus = dbus.SystemBus()
            om = dbus.Interface(bus.get_object("org.bluez", "/"),
                                "org.freedesktop.DBus.ObjectManager")
            for path, ifaces in om.GetManagedObjects().items():
                dev = ifaces.get("org.bluez.Device1")
                if dev and dev.get("Connected"):
                    _log(f"disconnecting central {dev.get('Address')}")
                    dbus.Interface(bus.get_object("org.bluez", path),
                                   "org.bluez.Device1").Disconnect()
        except Exception as e:                # noqa: BLE001 — best effort
            _log(f"WARN could not disconnect centrals: {e!r}")


# ---------------------------------------------------------------------------
# TX loop
# ---------------------------------------------------------------------------

@dataclass
class TxStats:
    sent: int = 0
    skipped: int = 0          # malformed JSON or unencodable record
    notifications: int = 0
    started: float = field(default_factory=time.monotonic)

    def line(self, transport, tail_path) -> str:
        sub = "yes" if getattr(transport, "subscribed", False) else "no"
        return (f"sent={self.sent} skipped={self.skipped} "
                f"notifications={self.notifications} subscriber={sub} "
                f"tailing={Path(tail_path).name}")


def run_tx(tail: JsonlTail, transport, notify_bytes: int = SAFE_NOTIFY_BYTES,
           session_dir=None, idle_timeout_s: float | None = None,
           max_records: int | None = None, heartbeat_s: float = HEARTBEAT_S,
           poll_s: float = POLL_S, stats: TxStats | None = None) -> TxStats:
    """Tail -> encode -> fragment -> notify, forever (or until idle/max)."""
    stats = stats or TxStats()
    seq = 0
    last_new = time.monotonic()
    next_beat = time.monotonic() + heartbeat_s
    next_resolve = time.monotonic() + RESOLVE_S
    while True:
        now = time.monotonic()
        if session_dir is not None and now >= next_resolve:
            next_resolve = now + RESOLVE_S
            newest = resolve_shadow_jsonl(session_dir)
            if newest is not None and newest != tail.path:
                _log(f"newer shadow file: tailing {newest.name}")
                tail = JsonlTail(newest)
        lines = tail.read_new_lines()
        if lines:
            last_new = time.monotonic()
        for line in lines:
            try:
                rec = json.loads(line)
                packet = encode_record(rec, seq)
            except (json.JSONDecodeError, CodecError) as e:
                stats.skipped += 1
                _log(f"skip unencodable line: {e}")
                continue
            for frag in fragment(packet, seq, notify_bytes):
                transport.notify(frag)
                stats.notifications += 1
            seq = (seq + 1) & 0xFFFF
            stats.sent += 1
            if max_records is not None and stats.sent >= max_records:
                _log("max records reached: " + stats.line(transport, tail.path))
                return stats
        if time.monotonic() >= next_beat:
            next_beat = time.monotonic() + heartbeat_s
            _log(stats.line(transport, tail.path))
        if (idle_timeout_s is not None
                and (time.monotonic() - last_new) > idle_timeout_s):
            _log(f"tail idle > {idle_timeout_s:.1f}s: stopping — "
                 + stats.line(transport, tail.path))
            return stats
        if not lines:
            time.sleep(poll_s)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tail", metavar="JSONL",
                     help="sectors/shadow JSONL to tail (read-only)")
    src.add_argument("--session-dir", metavar="DIR",
                     help="auto-pick (and follow) the newest shadow_*.jsonl "
                          "in this capture session dir")
    ap.add_argument("--loopback", action="store_true",
                    help="no bluetooth: run the encoder + a fake in-process "
                         "transport end-to-end (decodes + prints records)")
    ap.add_argument("--notify-bytes", type=int, default=SAFE_NOTIFY_BYTES,
                    help="max bytes per notification (default %(default)s = "
                         "the ATT-MTU-23 floor; raise once the negotiated "
                         "MTU is confirmed, e.g. 180)")
    ap.add_argument("--from-end", action="store_true",
                    help="skip existing content, stream new records only")
    ap.add_argument("--adapter", default=None,
                    help="bluetooth adapter address (default: first)")
    ap.add_argument("--name", default=LOCAL_NAME,
                    help="advertised local name (default %(default)s)")
    ap.add_argument("--idle-timeout", type=float, default=None,
                    help="stop after this many seconds without new records "
                         "(default: run forever)")
    ap.add_argument("--max-records", type=int, default=None)
    args = ap.parse_args(argv)

    session_dir = None
    if args.session_dir:
        session_dir = Path(args.session_dir)
        path = resolve_shadow_jsonl(session_dir)
        if path is None:
            _log(f"no shadow_*.jsonl in {session_dir} yet — waiting")
            while path is None:
                time.sleep(1.0)
                path = resolve_shadow_jsonl(session_dir)
        _log(f"tailing {path}")
    else:
        path = Path(args.tail)
        if not path.exists():
            _log(f"WARN {path} does not exist yet — will tail once it does")
    tail = JsonlTail(path, from_end=args.from_end)

    if args.loopback:
        transport = LoopbackTransport()
        run_tx(tail, transport, notify_bytes=args.notify_bytes,
               session_dir=session_dir, idle_timeout_s=args.idle_timeout,
               max_records=args.max_records)
        _log(f"loopback: {len(transport.records)} records decoded, "
             f"{transport.decode_errors} decode errors, "
             f"{transport.reassembler.dropped_partials} dropped partials")
        return 0 if transport.decode_errors == 0 else 1

    transport = BluezeroTransport(adapter_address=args.adapter,
                                  local_name=args.name)
    tx = threading.Thread(
        target=run_tx, args=(tail, transport),
        kwargs=dict(notify_bytes=args.notify_bytes, session_dir=session_dir,
                    idle_timeout_s=args.idle_timeout,
                    max_records=args.max_records),
        daemon=True, name="sector-ble-tx")
    tx.start()
    try:
        transport.serve_forever()            # GLib main loop (blocks)
    except KeyboardInterrupt:
        _log("interrupted")
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
