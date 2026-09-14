"""Wire codec for the BLE sector link (pure python, no dependencies).

SOURCE OF TRUTH. The receiving side (boatv1-boat-b, `sector_rx.py`) carries
a byte-for-byte vendored copy of this file — if you change the format here,
bump ``VERSION``, update `docs/reference/sector_protocol.md` ("BLE transport"
section) and re-copy the file into the boatv1 repo. Do NOT import across the
two repos.

What travels: the ESSENTIALS of one shadow/sectors JSONL record (see
`scripts/data_collection/shadow_fusion.py` and
`docs/reference/sector_protocol.md`): timestamp, the (uniform) bin grid,
per-bin fused score, per-bin learned p_obstacle when present, per-bin
min_range_m, and the shadow health bits (scorer_ok / seg_fresh / tick_ms /
source_latency_s). Everything else in the record (heading, targets, attitude,
…) deliberately stays on disk — reception is the goal, not the full protocol.

Packet layout (little-endian), ``VERSION = 1``::

    off  size  field
      0     2  magic  b"SB"
      2     1  version (1)
      3     2  seq            uint16, wraps; monotonic per tx process
      5     1  flags          bit0 HAS_P_OBSTACLE, bit1 SCORER_OK,
                              bit2 SEG_FRESH,     bit3 HAS_LATENCY,
                              bit4 YOLO_FRESH (typed boxes fresh this tick;
                              added 2026-09-07, VERSION unchanged: older
                              decoders ignore the bit)
      6     4  timestamp      uint32 deciseconds since midnight (100 ms —
                              exactly the repo's HH:MM:SS.f RoundedTime)
     10     1  n_bins         uint8
     11     1  first_center   int8 degrees   (bin grid is uniform: D.2 default
     12     1  step           uint8 degrees   is first=-45, step=15, n=7)
     13     2  tick_ms        uint16, pipeline tick time (clamped 65535)
     15     2  latency_cs     uint16 centiseconds source latency,
                              0xFFFF = absent (also flag bit3 clear)
     17     n  scores         uint8 each = round(score * 200)   (0..200,
                              i.e. 1/200 = 0.005 resolution)
    17+n   2n  min_range      uint16 each, centimetres; 0xFFFF = null,
                              values are clamped to 0xFFFE (655.34 m)
    ...     n  p_obstacle     uint8 each = round(p * 200); present only when
                              flag bit0 is set
    end     1  checksum       two's complement of the byte sum: the whole
                              packet sums to 0 mod 256

Sizes: 7 bins = 46 bytes with p_obstacle, 39 without. Fits one notification
whenever the negotiated ATT MTU ≥ packet + 2 (fragment header) + 3 (ATT).

Fragmentation (the notification layer): BLE 4.x's default ATT MTU of 23
allows only 20 notification bytes, so every packet travels as one or more
fragments, each prefixed with 2 bytes::

    byte 0  tag        = seq & 0xFF  (ties fragments to their packet)
    byte 1  index      low 7 bits = fragment index; 0x80 = final fragment
    2..     payload    packet bytes

Notifications on a single BLE connection are ordered and link-layer-checked,
so the reassembler only handles *loss across reconnects*: a fragment whose
tag or index does not continue the current buffer drops the partial packet
(a fresh index 0 always starts a new one). The trailing checksum then guards
against any pathological splice.

Quantisation tolerances (asserted by tests/test_sector_ble.py in the
obstacle-detection repo): scores and p_obstacle survive within ±0.0025
(=1/400); min_range within ±0.005 m up to 655.33 m.
"""

from __future__ import annotations

import struct

MAGIC = b"SB"
VERSION = 1

FLAG_HAS_P = 0x01
FLAG_SCORER_OK = 0x02
FLAG_SEG_FRESH = 0x04
FLAG_HAS_LATENCY = 0x08
FLAG_YOLO_FRESH = 0x10

#: 128-bit UUIDs for the GATT service/characteristic ("5a11" ~ SAIL). The box
#: (SensorBox) is the PERIPHERAL advertising SERVICE_UUID with one
#: notify-only characteristic; boat1 is the central that subscribes.
SERVICE_UUID = "b5ec70a0-5a11-4a5a-8d2a-6f1e0c9a0001"
CHAR_UUID = "b5ec70a0-5a11-4a5a-8d2a-6f1e0c9a0002"
LOCAL_NAME = "ASVProjectSector"

#: Safe notification payload floor (ATT MTU 23 - 3). Raise per-run once the
#: link is verified to negotiate a larger MTU (BlueZ<->BlueZ typically 517).
SAFE_NOTIFY_BYTES = 20

_HEADER = struct.Struct("<2sBHBIBbBHH")   # through latency_cs (17 bytes)

RANGE_NULL = 0xFFFF
RANGE_MAX = 0xFFFE          # 655.34 m
LATENCY_NULL = 0xFFFF
_QSCALE = 200               # scores / p_obstacle fixed-point scale


class CodecError(ValueError):
    """Malformed, foreign, or corrupt packet."""


# ---------------------------------------------------------------------------
# timestamp <-> deciseconds-of-day
# ---------------------------------------------------------------------------

def ts_to_deciseconds(ts: str) -> int:
    """'HH:MM:SS.f' (100 ms RoundedTime) -> deciseconds since midnight."""
    h, m, s = ts.split(":")
    ds = int(h) * 36000 + int(m) * 600 + int(round(float(s) * 10))
    if not 0 <= ds < 864000:
        raise CodecError(f"timestamp out of range: {ts!r}")
    return ds


def deciseconds_to_ts(ds: int) -> str:
    s, tenth = divmod(int(ds), 10)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}.{tenth}"


# ---------------------------------------------------------------------------
# quantisation helpers
# ---------------------------------------------------------------------------

def _q_unit(value) -> int:
    """Score/probability in [0, 1] -> uint8 (clamped)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.0
    if v != v:                       # NaN
        v = 0.0
    return max(0, min(_QSCALE, int(round(v * _QSCALE))))


def _q_range(value) -> int:
    if value is None:
        return RANGE_NULL
    try:
        v = float(value)
    except (TypeError, ValueError):
        return RANGE_NULL
    if v != v or v < 0:
        return RANGE_NULL
    return min(RANGE_MAX, int(round(v * 100.0)))


def _grid(bin_centers) -> tuple[int, int, int]:
    """Uniform integer-degree grid -> (n, first, step). Raises CodecError."""
    centers = [int(round(float(c))) for c in bin_centers]
    n = len(centers)
    if n == 0 or n > 255:
        raise CodecError(f"unencodable bin count: {n}")
    step = centers[1] - centers[0] if n > 1 else 1
    if n > 1 and (step <= 0 or step > 255
                  or any(b - a != step for a, b in zip(centers, centers[1:]))):
        raise CodecError(f"non-uniform bin grid: {centers}")
    if not -128 <= centers[0] <= 127:
        raise CodecError(f"first bin centre out of int8: {centers[0]}")
    return n, centers[0], step


# ---------------------------------------------------------------------------
# packet encode / decode
# ---------------------------------------------------------------------------

def encode_record(rec: dict, seq: int) -> bytes:
    """One shadow/sectors JSONL record (dict) -> packet bytes.

    Raises CodecError on records that cannot be represented (missing
    timestamp/bins, non-uniform grid) — callers skip those and move on.
    """
    if not isinstance(rec, dict):
        raise CodecError("record is not an object")
    try:
        ds = ts_to_deciseconds(str(rec["timestamp"]))
        n, first, step = _grid(rec["bin_centers_deg"])
        scores = list(rec["scores"])
    except (KeyError, TypeError, ValueError) as e:
        raise CodecError(f"unencodable record: {e!r}") from e
    if len(scores) != n:
        raise CodecError(f"scores length {len(scores)} != {n} bins")
    ranges = list(rec.get("min_range_m") or [None] * n)
    if len(ranges) != n:
        raise CodecError(f"min_range_m length {len(ranges)} != {n} bins")
    p = rec.get("p_obstacle")
    if p is not None and len(p) != n:
        raise CodecError(f"p_obstacle length {len(p)} != {n} bins")

    shadow = rec.get("shadow") or {}
    flags = 0
    if p is not None:
        flags |= FLAG_HAS_P
    if shadow.get("scorer_ok"):
        flags |= FLAG_SCORER_OK
    if shadow.get("seg_fresh"):
        flags |= FLAG_SEG_FRESH
    if shadow.get("yolo_fresh"):
        flags |= FLAG_YOLO_FRESH
    lat = shadow.get("source_latency_s")
    lat_cs = LATENCY_NULL
    if lat is not None:
        try:
            lat_cs = min(0xFFFE, max(0, int(round(float(lat) * 100.0))))
            flags |= FLAG_HAS_LATENCY
        except (TypeError, ValueError):
            lat_cs = LATENCY_NULL
    tick = shadow.get("tick_ms")
    try:
        tick_ms = min(0xFFFF, max(0, int(round(float(tick)))))
    except (TypeError, ValueError):
        tick_ms = 0

    out = bytearray(_HEADER.pack(MAGIC, VERSION, seq & 0xFFFF, flags, ds,
                                 n, first, step, tick_ms, lat_cs))
    out += bytes(_q_unit(s) for s in scores)
    for r in ranges:
        out += struct.pack("<H", _q_range(r))
    if p is not None:
        out += bytes(_q_unit(v) for v in p)
    out.append((-sum(out)) & 0xFF)          # whole packet sums to 0 mod 256
    return bytes(out)


def decode_packet(buf: bytes) -> dict:
    """Packet bytes -> record dict (the essentials; see module docstring)."""
    if len(buf) < _HEADER.size + 1:
        raise CodecError(f"short packet ({len(buf)} bytes)")
    magic, ver, seq, flags, ds, n, first, step, tick_ms, lat_cs = \
        _HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise CodecError(f"bad magic {magic!r}")
    if ver != VERSION:
        raise CodecError(f"unsupported version {ver} (want {VERSION})")
    has_p = bool(flags & FLAG_HAS_P)
    expect = _HEADER.size + n + 2 * n + (n if has_p else 0) + 1
    if len(buf) != expect:
        raise CodecError(f"length {len(buf)} != expected {expect}")
    if sum(buf) & 0xFF:
        raise CodecError("checksum mismatch")

    off = _HEADER.size
    scores = [b / _QSCALE for b in buf[off:off + n]]
    off += n
    ranges: list[float | None] = []
    for i in range(n):
        (r,) = struct.unpack_from("<H", buf, off + 2 * i)
        ranges.append(None if r == RANGE_NULL else r / 100.0)
    off += 2 * n
    p_obstacle = None
    if has_p:
        p_obstacle = [b / _QSCALE for b in buf[off:off + n]]

    rec = {
        "seq": seq,
        "timestamp": deciseconds_to_ts(ds),
        "bin_centers_deg": [first + i * step for i in range(n)],
        "scores": scores,
        "min_range_m": ranges,
        "shadow": {
            "tick_ms": tick_ms,
            "scorer_ok": bool(flags & FLAG_SCORER_OK),
            "seg_fresh": bool(flags & FLAG_SEG_FRESH),
            "yolo_fresh": bool(flags & FLAG_YOLO_FRESH),
            "source_latency_s": (lat_cs / 100.0
                                 if (flags & FLAG_HAS_LATENCY)
                                 and lat_cs != LATENCY_NULL else None),
        },
    }
    if p_obstacle is not None:
        rec["p_obstacle"] = p_obstacle
    return rec


# ---------------------------------------------------------------------------
# fragmentation (notification layer)
# ---------------------------------------------------------------------------

FRAG_OVERHEAD = 2
FRAG_FINAL = 0x80


def fragment(packet: bytes, seq: int,
             notify_bytes: int = SAFE_NOTIFY_BYTES) -> list[bytes]:
    """Split a packet into notification-sized fragments (see docstring)."""
    payload = notify_bytes - FRAG_OVERHEAD
    if payload < 1:
        raise CodecError(f"notify_bytes too small: {notify_bytes}")
    tag = seq & 0xFF
    chunks = [packet[i:i + payload] for i in range(0, len(packet), payload)]
    if len(chunks) > 0x7F:
        raise CodecError(f"packet needs {len(chunks)} fragments (> 127)")
    out = []
    for idx, chunk in enumerate(chunks):
        hdr = idx | (FRAG_FINAL if idx == len(chunks) - 1 else 0)
        out.append(bytes((tag, hdr)) + chunk)
    return out


class Reassembler:
    """Feed notification payloads; get completed packets back.

    In-order delivery is the BLE guarantee; this only survives *gaps*
    (reconnects, dropped partials): any discontinuity throws the partial
    packet away and waits for the next index-0 fragment.
    """

    def __init__(self):
        self._tag: int | None = None
        self._next_idx = 0
        self._buf = bytearray()
        self.dropped_partials = 0

    def feed(self, data: bytes) -> bytes | None:
        if len(data) < FRAG_OVERHEAD + 1:
            return None
        tag, hdr = data[0], data[1]
        idx, final = hdr & 0x7F, bool(hdr & FRAG_FINAL)
        if idx == 0:
            if self._tag is not None:
                self.dropped_partials += 1
            self._tag, self._next_idx, self._buf = tag, 0, bytearray()
        elif tag != self._tag or idx != self._next_idx:
            if self._tag is not None:
                self.dropped_partials += 1
            self._tag = None
            return None
        self._buf += data[FRAG_OVERHEAD:]
        self._next_idx = idx + 1
        if not final:
            return None
        packet = bytes(self._buf)
        self._tag = None
        return packet
