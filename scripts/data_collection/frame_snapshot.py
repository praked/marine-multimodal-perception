"""Latest-frame snapshot channel between the capture service and a reader.

The capture service owns the CSI camera and the PureThermal, and the mp4 it
writes is moov-less until the chunk closes, so a process running beside it
(`shadow_fusion --tail`) had no camera pixels at all. This module publishes
the newest fisheye/thermal frame to a tmpfs directory (``/dev/shm`` on the
Pi) once per capture loop, and lets the reader pick it up by (chunk, frame
index) -- the same key as the ``frames_<ts>.csv`` sidecar row, so the join
is by index, never by wall-clock.

Wire format (all files in one directory):

    fisheye.npy   raw array exactly as handed to the mp4 writer (BGR uint8)
    thermal.npy   rotated thermal frame (absent when the thermal is off)
    latest.json   {"chunk_ts", "frame_index", "stamp", "has_thermal", "seq"}

Every file is written to ``<name>.tmp`` and renamed into place, and the
meta file is written LAST, so a reader that sees a meta describing frame
N can rely on the arrays being frame N (or newer, never older: a reader
that races a publish gets a consistent, newer pair on its next read).
Cost on the capture loop: one ``np.save`` of 1.7 MB into page cache
(~2-4 ms on a Pi 4). Never raises into the capture loop: failures are
counted and reported through ``SnapshotWriter.errors``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

META_NAME = "latest.json"
FISHEYE_NAME = "fisheye.npy"
THERMAL_NAME = "thermal.npy"


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


class SnapshotWriter:
    """Publish the newest frame pair into ``directory`` (created if absent)."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seq = 0
        self.errors = 0
        self.last_error: str | None = None

    def publish(self, chunk_ts: str, frame_index: int, stamp: str,
                fisheye: np.ndarray | None,
                thermal: np.ndarray | None = None) -> bool:
        """Write fisheye (+ thermal) then the meta. Returns True on success."""
        if fisheye is None:
            return False
        try:
            _atomic_save(self.directory / FISHEYE_NAME, np.ascontiguousarray(fisheye))
            has_thermal = thermal is not None
            if has_thermal:
                _atomic_save(self.directory / THERMAL_NAME,
                             np.ascontiguousarray(thermal))
            self.seq += 1
            meta = {"chunk_ts": chunk_ts, "frame_index": int(frame_index),
                    "stamp": stamp, "has_thermal": has_thermal,
                    "seq": self.seq, "published_at": time.time()}
            _atomic_write_text(self.directory / META_NAME, json.dumps(meta))
            return True
        except Exception as e:      # never into the capture loop
            self.errors += 1
            self.last_error = repr(e)
            return False


@dataclass
class Snapshot:
    chunk_ts: str
    frame_index: int
    stamp: str
    seq: int
    published_at: float
    fisheye: np.ndarray
    thermal: np.ndarray | None

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.published_at)


class SnapshotReader:
    """Read the newest published frame pair; ``None`` when nothing new."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._last_seq: int | None = None

    def peek_meta(self) -> dict | None:
        try:
            with open(self.directory / META_NAME) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def read(self, want: tuple[str, int] | None = None,
             allow_repeat: bool = False, tolerance: int = 0) -> Snapshot | None:
        """Return the snapshot, or None.

        ``want`` = (chunk_ts, frame_index) restricts the read to that frame
        (the frames-sidecar row being processed) or, with ``tolerance`` = k,
        to a snapshot at most k frames NEWER in the same chunk: capture
        publishes the frame before it writes the row, so a reader whose tick
        is slower than the 3 Hz loop usually finds the snapshot one frame
        ahead of the newest row it can see (measured 2026-09-07: 1 tick in 3
        matched exactly). Older snapshots and other chunks are always
        refused, so pixels never lag the radar. Without ``want`` the newest
        snapshot is returned once per ``seq`` (``allow_repeat`` re-reads the
        same one).
        """
        meta = self.peek_meta()
        if meta is None:
            return None
        if want is not None:
            same_chunk = meta.get("chunk_ts") == want[0]
            ahead = int(meta.get("frame_index", -1)) - int(want[1])
            if not same_chunk or ahead < 0 or ahead > int(tolerance):
                return None
        seq = int(meta.get("seq", 0))
        if not allow_repeat and self._last_seq == seq:
            return None
        try:
            fisheye = np.load(self.directory / FISHEYE_NAME, allow_pickle=False)
            thermal = (np.load(self.directory / THERMAL_NAME, allow_pickle=False)
                       if meta.get("has_thermal") else None)
        except (OSError, ValueError):
            return None
        # A publish may have landed between the meta read and the array
        # reads; the arrays are then NEWER than meta says. Re-check and
        # report the meta that matches what was loaded.
        meta2 = self.peek_meta()
        if meta2 is not None and int(meta2.get("seq", 0)) != seq:
            if want is not None:
                ahead2 = int(meta2.get("frame_index", -1)) - int(want[1])
                if meta2.get("chunk_ts") != want[0] or ahead2 < 0 or ahead2 > int(tolerance):
                    return None      # the wanted frame is gone
            meta, seq = meta2, int(meta2.get("seq", 0))
        self._last_seq = seq
        return Snapshot(chunk_ts=str(meta.get("chunk_ts", "")),
                        frame_index=int(meta.get("frame_index", -1)),
                        stamp=str(meta.get("stamp", "")), seq=seq,
                        published_at=float(meta.get("published_at", 0.0)),
                        fisheye=fisheye, thermal=thermal)
