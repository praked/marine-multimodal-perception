# ****************************************************************************
# *  Phase-4 shadow mode: a live fusion service for the Pi 4 that runs
# *  beside the capture service and emits one sectors record per fisheye
# *  frame, in real time, to shadow_<start-stamp>.jsonl.
# ****************************************************************************
"""Shadow fusion service (Phase 4): per-frame sector records at capture rate.

Process model: SEPARATE PROCESS (decision + rationale)
------------------------------------------------------
Shadow does NOT share the capture service's process. The alternatives were
weighed by reading `continuous_capture.py`:

* `write_one_chunk`'s inner 3 fps loop is a monolith with no hook point, and
  every one of its sensor hand-offs (`RadarReader.take_fresh`,
  `ThermalReader.take_fresh`, `IMUReaderRVC.take_fresh`) is a single-consumer
  latest-wins contract — a second in-process consumer would either steal
  frames from the CSV/mp4 writers or require a fan-out refactor of the
  field-verified capture path.
* Capture has absolute priority (CLAUDE.md §9 + the 2026-08-19 loop-collapse
  incident): inserting per-tick shadow work (frame copy, queue, pipeline) into
  the capture loop is exactly the risk profile we must not take days before a
  capture campaign.

So shadow is its own process with three interchangeable frame sources:

* ``ReplaySources`` — walk a recorded triplet (``resolve_triplet`` /
  ``iterate_triplet``), optionally at real-time pace. The bench harness and
  the tests; runs anywhere.
* ``TailSources`` — run BESIDE the live capture service: tail its growing
  ``frames_<ts>.csv`` (the tick: one record per fisheye frame, capture's own
  RTC-backed timestamps) and ``mmwave_<ts>.csv`` / ``imu_<ts>.csv``
  (read-only, never the serial ports — the radar UART cannot be opened
  twice). Camera pixels come from capture's latest-frame SNAPSHOT when
  ``--snapshot-dir`` is given (``frame_snapshot.py``: the capture loop
  publishes its newest fisheye/thermal pair to tmpfs, keyed by the frames
  sidecar row, since 2026-09-06) -- then the seg/YOLO workers and the full
  scorer run beside capture, latest-wins (only the newest sidecar row is
  processed per tick; ``shadow.skipped_rows`` counts the rest). Without a
  snapshot dir the growing mp4s are unreadable (no moov atom until the
  chunk closes) and the record honestly carries radar(+IMU)-only evidence
  with the camera sensors absent. Capture flushes its CSVs per loop since
  2026-09-03, so ``shadow.source_latency_s`` is ~0.1 s.
* ``LiveSources`` — own the full sensor set (fisheye + thermal + radar +
  IMU) via `continuous_capture`'s own primitives. Pi-only, and ONLY when the
  capture service is stopped (the sensors are exclusive); refuses cleanly on
  a laptop. This is the full-stack shadow demo path (segmentation worker
  included) for on-Pi verification.

Reused, not reimplemented: `ObstacleDetectionPipeline` (one instance for the
service lifetime — the thermal background is stateful), `SegWorker`
(latest-wins async CNN; a stale mask degrades to the classical path exactly
like an absent offline mask), `load_any_scorer` +
`build_features.bin_rows_for_frame` (the same feature construction as
`fusion.py --scorer`, one code path, no drift). Scorer failure or a missing
artefact degrades to the legacy n/3 scores with a logged warning — never a
crash.

Timeline discipline: replay records carry the triplet's own timestamps;
tail records carry capture's RTC-backed per-frame stamps; live records carry
the wall clock at frame grab. All ``HH:MM:SS.f`` rounded to 100 ms, matching
every other stream in the repo. The two timelines are never mixed.

Usage
-----
Bench (laptop, no hardware)::

    python -m scripts.data_collection.shadow_fusion \
        --replay data/Boats/2025-06-23_16-21-07 --max-frames 60

Beside capture (Pi; read-only on capture's files)::

    python -m scripts.data_collection.shadow_fusion \
        --tail ~/captures/<session>

Standalone live (Pi; capture service STOPPED — sensors are exclusive)::

    python -m scripts.data_collection.shadow_fusion --live \
        --seg-onnx ~/ewasr/student_512x384.int8.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:      # runnable by path, like the capture unit
    sys.path.insert(0, str(REPO_ROOT))

#: The live scorer artefact (v1b GBT bundle, promoted 2026-09-02). Missing /
#: unloadable (e.g. no scikit-learn on the Pi) -> legacy scores only, warned.
DEFAULT_SCORER = REPO_ROOT / "models" / "fusion_scorer_live_v1b.joblib"

#: Reactive-tick budget (everything except the async seg CNN): one camera
#: frame period at the 3 fps capture cadence.
TICK_BUDGET_MS = 333.0

#: Consumers treat a seg mask older than this as absent (same default as
#: pi_seg_worker.DEFAULT_MAX_AGE_S; kept literal so this module imports
#: without onnxruntime).
DEFAULT_SEG_MAX_AGE_S = 2.0

PROTOCOL_VERSION = 0     # legacy core schema + additive extensions


def _log(msg: str) -> None:
    print(f"[shadow] {msg}", flush=True)


def _stamp_str(dt: datetime) -> str:
    """Wall-clock -> the repo's 100 ms 'HH:MM:SS.f' RoundedTime format."""
    return dt.strftime("%H:%M:%S.%f")[:-5]


def _seconds_of_day(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

@dataclass
class ShadowFrame:
    """One tick's worth of sensor input for the pipeline."""

    ts: str                                  # HH:MM:SS.f (100 ms)
    fisheye: np.ndarray | None               # BGR (capture convention) or None
    thermal: np.ndarray | None
    points: np.ndarray                       # Nx{3,4,6} radar cloud (may be empty)
    source_latency_s: float | None = None    # tail mode: now - frame stamp


class ReplaySources:
    """Walk a recorded triplet; the bench harness and what the tests use."""

    def __init__(self, prefix, detection: dict, realtime: bool = False,
                 fps: float = 3.0, clip_overrides: dict | None = None):
        from scripts.sensor_processing.imu_bno085 import load_imu_config
        from scripts.sensor_processing.imu_replay import ImuLogAttitudeProvider
        from scripts.utils.datasets import resolve_triplet

        self.triplet = resolve_triplet(prefix)
        self.detection = detection
        self.realtime = realtime
        self.fps = fps
        self._clip_overrides = clip_overrides
        self.clip_id = self.triplet.clip_id
        self.clip_date = self.triplet.timestamp[:10]
        self.chunk_hms = self.triplet.timestamp.split("_")[-1]
        self.attitude_provider = ImuLogAttitudeProvider.for_triplet(
            self.triplet, (load_imu_config().get("replay", {}) or {}))

    def frame_id_for(self, ts: str) -> str:
        from scripts.sensor_processing.fusion import _frame_id_for
        return _frame_id_for(self.triplet.scene, self.triplet.timestamp, ts)

    def frames(self):
        from scripts.sensor_processing.pipeline import iterate_triplet

        period = 1.0 / self.fps
        next_t = time.monotonic()
        for ts, fish, therm, pts in iterate_triplet(
                self.triplet, self.detection,
                clip_overrides=self._clip_overrides):
            if self.realtime:
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield ShadowFrame(ts=ts, fisheye=fish, thermal=therm, points=pts)

    def close(self) -> None:
        pass


class _CsvTail:
    """Incremental read-only tail of a growing CSV: complete lines only."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._pos = 0
        self._partial = ""
        self.header: list[str] | None = None

    def read_new_rows(self) -> list[list[str]]:
        try:
            with open(self.path, "r", newline="") as fh:
                fh.seek(self._pos)
                chunk = fh.read()
                self._pos = fh.tell()
        except OSError:
            return []
        if not chunk:
            return []
        text = self._partial + chunk
        lines = text.split("\n")
        self._partial = lines.pop()      # last piece may be mid-write
        rows = []
        for line in lines:
            line = line.strip("\r")
            if not line:
                continue
            cells = line.split(",")
            if self.header is None:
                self.header = cells
                continue
            rows.append(cells)
        return rows


def _to_float(cell: str) -> float:
    try:
        return float(cell)
    except (TypeError, ValueError):
        return float("nan")


class _TailAttitudeProvider:
    """Attitude from capture's growing imu_<ts>.csv, via the canonical
    ImuLogAttitudeProvider (same axis mapping / offsets as replay). Rebuilt
    from the file every ``rebuild_s`` — cheap (numpy over a few hundred rows)
    and torn final lines are dropped by load_imu_csv's NaN filter."""

    def __init__(self, path: Path, rebuild_s: float = 2.0):
        self._path = Path(path)
        self._rebuild_s = rebuild_s
        self._built = 0.0
        self._inner = None

    def _maybe_rebuild(self) -> None:
        now = time.monotonic()
        if self._inner is not None and (now - self._built) < self._rebuild_s:
            return
        self._built = now
        try:
            from scripts.sensor_processing.imu_replay import (
                ImuLogAttitudeProvider,
            )
            self._inner = ImuLogAttitudeProvider.from_csv(self._path)
        except Exception:
            self._inner = None

    def set_time(self, ts) -> None:
        self._maybe_rebuild()
        if self._inner is not None:
            self._inner.set_time(ts)

    def get(self):
        return self._inner.get() if self._inner is not None else None


class TailSources:
    """Run beside the capture service: tail its growing CSVs, read-only.

    Ticks once per new ``frames_<ts>.csv`` row (one record per fisheye frame,
    at capture's own cadence and RTC-backed timestamps); radar points come
    from the newest complete timestamp group in ``mmwave_<ts>.csv``; attitude
    from ``imu_<ts>.csv``. Camera pixels are unavailable (growing mp4s are
    moov-less), so fisheye/thermal are honestly None. Follows chunk rolls
    (a newer frames_<ts>.csv supersedes the current one).
    """

    def __init__(self, session_dir, poll_s: float = 0.05,
                 idle_timeout_s: float | None = None,
                 snapshot_dir: str | Path | None = None,
                 latest_wins: bool | None = None):
        self.session_dir = Path(session_dir)
        self.poll_s = poll_s
        self.idle_timeout_s = idle_timeout_s
        # Camera pixels beside capture: the capture service publishes its
        # newest frame pair to a tmpfs dir (ASVPROJECT_SNAPSHOT_DIR, see
        # frame_snapshot.py), keyed by (chunk_ts, frame_index) = the frames
        # sidecar row. A full-stack tick (seg + YOLO + scorer) is slower
        # than the 3 Hz sidecar, so with snapshots the source is
        # LATEST-WINS: of the rows that arrived since the last tick only the
        # newest is processed (`skipped_rows` counts the rest) instead of
        # falling ever further behind. Without a snapshot dir every row is
        # processed, as before.
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.latest_wins = (latest_wins if latest_wins is not None
                            else self.snapshot_dir is not None)
        self._snap_reader = None
        if self.snapshot_dir is not None:
            from scripts.data_collection.frame_snapshot import SnapshotReader
            self._snap_reader = SnapshotReader(self.snapshot_dir)
        # Accept a snapshot up to this many frames NEWER than the sidecar row
        # (see SnapshotReader.read): 1 = one camera period (333 ms), the same
        # order as the tail's own latency; the radar cloud is the newest
        # complete group either way.
        self.snapshot_tolerance = 1
        self.camera_offset_frames: int | None = None
        self.skipped_rows = 0
        self.camera_frames = 0       # ticks that carried camera pixels
        self.camera_misses = 0       # ticks whose snapshot was another frame
        self.chunk_ts: str | None = None
        self.clip_id = f"live/{self.session_dir.name}"
        self.clip_date = datetime.now().strftime("%Y-%m-%d")
        self.chunk_hms = None
        self._frames_tail: _CsvTail | None = None
        self._mm_tail: _CsvTail | None = None
        self._mm_groups: dict[str, list[list[str]]] = {}
        self._mm_order: list[str] = []
        self.attitude_provider = None
        self._resolve()

    # -- file resolution ---------------------------------------------------

    def _newest(self, pattern: str) -> Path | None:
        hits = sorted(self.session_dir.glob(pattern))
        hits += sorted(self.session_dir.glob(f"*/{pattern}"))
        return max(hits, key=lambda p: p.name) if hits else None

    def _resolve(self) -> None:
        frames = self._newest("frames_*.csv")
        if frames is None:
            return
        if self._frames_tail is not None and self._frames_tail.path == frames:
            return
        self._frames_tail = _CsvTail(frames)
        # Tailing the captures ROOT (the systemd unit): name the record after
        # the per-boot session folder the file lives in, not the root.
        self.clip_id = f"live/{frames.parent.name}"
        ts = frames.stem[len("frames_"):]
        self.chunk_ts = ts
        self.chunk_hms = ts.split("_")[-1]
        self.clip_date = ts[:10]
        mm = frames.parent / f"mmwave_{ts}.csv"
        self._mm_tail = _CsvTail(mm) if mm.exists() else None
        self._mm_groups, self._mm_order = {}, []
        imu = frames.parent / f"imu_{ts}.csv"
        self.attitude_provider = (_TailAttitudeProvider(imu)
                                  if imu.exists() else None)
        _log(f"tailing {frames.name}"
             f"{' + ' + mm.name if self._mm_tail else ' (no radar csv)'}")

    # -- radar grouping ----------------------------------------------------

    def _drain_radar(self) -> None:
        if self._mm_tail is None:
            return
        for cells in self._mm_tail.read_new_rows():
            if len(cells) < 5:
                continue
            key = f"{cells[0]} {cells[1]}"
            if key not in self._mm_groups:
                self._mm_groups[key] = []
                self._mm_order.append(key)
                # keep a short history only
                while len(self._mm_order) > 8:
                    self._mm_groups.pop(self._mm_order.pop(0), None)
            self._mm_groups[key].append(cells)

    def _latest_points(self) -> np.ndarray:
        """Points of the newest COMPLETE radar timestamp group (the newest
        group may still be mid-write; heartbeat rows -> empty cloud)."""
        empty = np.empty((0, 4), dtype=np.float64)
        if len(self._mm_order) < 2:
            return empty
        key = self._mm_order[-2]
        rows = self._mm_groups.get(key, [])
        pts = []
        for cells in rows:
            x, y, z = (_to_float(c) for c in cells[2:5])
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue      # sentinel heartbeat row
            v = _to_float(cells[5]) if len(cells) > 5 else float("nan")
            row = [x, y, z, v]
            if len(cells) > 7:
                row += [_to_float(cells[6]), _to_float(cells[7])]
            pts.append(row)
        if not pts:
            return empty
        width = max(len(r) for r in pts)
        return np.array([r + [float("nan")] * (width - len(r)) for r in pts],
                        dtype=np.float64)

    # -- frame stream ------------------------------------------------------

    def frames(self):
        last_new = time.monotonic()
        while True:
            self._resolve()
            self._drain_radar()
            rows = (self._frames_tail.read_new_rows()
                    if self._frames_tail is not None else [])
            if rows:
                last_new = time.monotonic()
                rows = [c for c in rows if len(c) >= 3]
                if self.latest_wins and len(rows) > 1:
                    self.skipped_rows += len(rows) - 1
                    rows = rows[-1:]
                for cells in rows:
                    ts = cells[2]
                    latency = None
                    try:
                        stamp = datetime.strptime(
                            f"{cells[1]} {ts}", "%Y-%m-%d %H:%M:%S.%f")
                        latency = max(0.0,
                                      (datetime.now() - stamp).total_seconds())
                    except ValueError:
                        pass
                    fisheye = thermal = None
                    if self._snap_reader is not None:
                        snap = None
                        try:
                            snap = self._snap_reader.read(
                                want=(self.chunk_ts or "", int(cells[0])),
                                allow_repeat=True,
                                tolerance=self.snapshot_tolerance)
                        except (ValueError, TypeError):
                            snap = None
                        if snap is not None:
                            fisheye, thermal = snap.fisheye, snap.thermal
                            self.camera_frames += 1
                            self.camera_offset_frames = snap.frame_index - int(cells[0])
                        else:
                            self.camera_misses += 1
                            self.camera_offset_frames = None
                    yield ShadowFrame(ts=ts, fisheye=fisheye, thermal=thermal,
                                      points=self._latest_points(),
                                      source_latency_s=latency)
                continue
            if (self.idle_timeout_s is not None
                    and (time.monotonic() - last_new) > self.idle_timeout_s):
                _log(f"tail idle > {self.idle_timeout_s:.1f}s: stopping")
                return
            time.sleep(self.poll_s)

    def close(self) -> None:
        pass


class LiveSources:
    """Own the full sensor set via continuous_capture's primitives. Pi-only;
    ONLY when the capture service is stopped (sensors are exclusive)."""

    def __init__(self, fps: float = 3.0):
        from scripts.data_collection import continuous_capture as cc

        if cc.Picamera2 is None:
            raise SystemExit(
                "shadow --live needs the Pi camera stack (picamera2). "
                "Run on SensorBox with the capture service STOPPED "
                "(sudo systemctl stop asvproject-capture); on a laptop use "
                "--replay or --tail.")
        self._cc = cc
        self.fps = fps
        self.clip_id = "live/shadow"
        self.clip_date = datetime.now().strftime("%Y-%m-%d")
        self.chunk_hms = datetime.now().strftime("%H-%M-%S")

        # Radar: config push + drain thread (capture's own reader class).
        self.radar = None
        try:
            cc.send_radar_config(cc.CLI_PORT, cc.CONFIG_FILE)
            import serial
            ser = serial.Serial(cc.DATA_PORT, cc.DATA_BAUD, timeout=1)
            ser.reset_input_buffer()
            self.radar = cc.RadarReader(ser)
            self.radar.start()
        except Exception as e:
            _log(f"WARN radar unavailable: {e!r}")

        # Fisheye (exclusive CSI camera: this is why capture must be stopped).
        self.fish = None
        try:
            fish = cc.Picamera2()
            fish.configure(fish.create_preview_configuration(
                main={"format": "RGB888", "size": cc.FISHEYE_SIZE}))
            fish.start()
            self.fish = fish
        except Exception as e:
            _log(f"WARN fisheye unavailable: {e!r}")

        # Thermal: by-id discovery + latest-wins reader.
        self.therm = None
        try:
            cap = cc._open_thermal()
            if cap is not None:
                self.therm = cc.ThermalReader(cap)
                self.therm.start()
        except Exception as e:
            _log(f"WARN thermal unavailable: {e!r}")

        # Attitude: the canonical live reader (uart_rvc backend + the same
        # rotation-composition mapping the replay provider applies).
        self.attitude_provider = None
        try:
            from scripts.sensor_processing.imu_bno085 import (
                BNO085Reader,
                load_imu_config,
            )
            self.attitude_provider = BNO085Reader(load_imu_config()).start()
        except Exception as e:
            _log(f"WARN IMU unavailable: {e!r}")

    def _radar_points(self) -> np.ndarray:
        cc = self._cc
        empty = np.empty((0, 4), dtype=np.float64)
        if self.radar is None:
            return empty
        frame = self.radar.take_fresh()
        if frame is None:
            return empty
        payload, n_tlv, n_obj, _stamp = frame
        if n_obj == 0 or n_obj > cc.MAX_LOG_OBJECTS:
            return empty
        parsed = cc.parse_tlvs(payload, n_tlv, n_obj)
        if parsed is None:
            return empty
        xs, ys, zs, vs, snrs, noises = parsed
        cols = [xs, ys, zs, vs]
        if snrs is not None:
            cols += [snrs, noises]
        return np.array(list(zip(*cols)), dtype=np.float64)

    def frames(self):
        cc = self._cc
        period = 1.0 / self.fps
        next_t = time.monotonic()
        while True:
            next_t += period
            stamp = datetime.now()
            fish_frame = None
            if self.fish is not None:
                try:
                    fish_frame, _meta = cc._grab_fisheye(self.fish)
                except Exception as e:
                    _log(f"WARN fisheye grab: {e!r}")
            therm_frame = None
            if self.therm is not None:
                t = self.therm.take_fresh()
                if t is not None:
                    therm_frame = cc._rotate_thermal(t)
            yield ShadowFrame(ts=_stamp_str(stamp), fisheye=fish_frame,
                              thermal=therm_frame,
                              points=self._radar_points())
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)

    def close(self) -> None:
        for closer in (
                lambda: self.radar and self.radar.close(),
                lambda: self.fish and self.fish.stop(),
                lambda: self.therm and self.therm.release(),
                lambda: self.attitude_provider
                and self.attitude_provider.stop()):
            try:
                closer()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Segmentation worker adapter (latest-wins mask -> pipeline seg provider)
# ---------------------------------------------------------------------------

class SegWorkerProvider:
    """Duck-types the pipeline's SegProvider (.get(frame_id) -> mask|None)
    over a live SegWorker: returns the freshest mask inside the staleness
    gate, or None -> the pipeline degrades to the classical path exactly as
    it does for an absent offline mask (the existing contract)."""

    def __init__(self, worker, max_age_s: float = DEFAULT_SEG_MAX_AGE_S):
        self.worker = worker
        self.max_age_s = max_age_s
        self.last_age_s: float | None = None    # telemetry for the record

    def get(self, _frame_id) -> np.ndarray | None:
        res = self.worker.latest(max_age_s=None)
        self.last_age_s = res.age_s() if res is not None else None
        if res is None or res.age_s() > self.max_age_s:
            return None
        return res.mask

    def available(self) -> bool:
        return True

    def submit(self, frame_bgr: np.ndarray) -> None:
        """Non-blocking: offer the (undistorted) frame for the NEXT ticks'
        mask. BGR in (pipeline convention) -> RGB for the worker."""
        self.worker.submit(frame_bgr[:, :, ::-1])


def build_seg_worker(onnx_path: str, threads: int = 2):
    """SegWorker from pi_seg_worker, or None (warned) when onnxruntime is
    absent — pi_seg_worker raises SystemExit at import on the laptop."""
    try:
        from scripts.gpu_seg.pi_seg_worker import SegWorker
    except (ImportError, SystemExit) as e:
        _log(f"WARN seg worker unavailable (onnxruntime?): {e}")
        return None
    try:
        worker = SegWorker(str(onnx_path), threads=threads)
        worker.start()
        return worker
    except Exception as e:
        _log(f"WARN seg worker failed to start: {e!r}")
        return None


class YoloWorker:
    """Latest-wins async typed detector (YOLOv8 ONNX export, no NMS in the
    graph) — the SegWorker pattern: `submit` never blocks or queues, `latest`
    returns the freshest box list with its age. Output boxes use the
    DetProvider schema ({cls, xyxy normalised, confidence}) so
    `bin_rows_for_frame(det_boxes=...)` fills the yolo_* columns exactly as
    the offline `data/det` files do. Measured 2026-09-03: v8n-640 675 ms on
    the Pi 4 at 2 threads -> ~0.5 Hz beside the tick and the seg worker."""

    CLASSES = ("boat", "buoy", "duck", "other", "person", "structure")  # audited data.yaml order

    def __init__(self, onnx_path: str, threads: int = 2, conf: float = 0.25,
                 iou: float = 0.5):
        import threading
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), so,
                                         providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        self.size = int(inp.shape[2]), int(inp.shape[3])      # (h, w)
        self.conf, self.iou = conf, iou
        self._lock = threading.Lock()
        self._pending = None
        self._latest: tuple[list, float] | None = None       # (boxes, t_mono)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="yolo-worker",
                                        daemon=True)
        self.n_infer = 0
        self.infer_ms: list[float] = []

    def start(self):
        self._thread.start()
        return self

    def stop(self, join_timeout_s: float = 10.0):
        self._stop.set()
        self._thread.join(join_timeout_s)

    def submit(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._pending = frame_bgr

    def latest(self, max_age_s: float | None = None):
        if self._latest is None:
            return None, None
        boxes, t = self._latest
        age = time.monotonic() - t
        if max_age_s is not None and age > max_age_s:
            return None, age
        return boxes, age

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is None:
                time.sleep(0.02)
                continue
            t0 = time.perf_counter()
            boxes = self._infer(frame)
            self.infer_ms.append((time.perf_counter() - t0) * 1000.0)
            self.n_infer += 1
            self._latest = (boxes, time.monotonic())

    def _infer(self, frame_bgr: np.ndarray) -> list:
        import cv2
        h, w = frame_bgr.shape[:2]
        ih, iw = self.size
        img = cv2.resize(frame_bgr, (iw, ih))[:, :, ::-1].astype(np.float32) / 255.0
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
        out = self.sess.run(None, {self.name: x})[0]           # (1, 4+nc, N)
        pred = out[0].T                                        # (N, 4+nc)
        cls_scores = pred[:, 4:]
        cls_id = cls_scores.argmax(1)
        score = cls_scores[np.arange(len(pred)), cls_id]
        keep = score >= self.conf
        if not keep.any():
            return []
        cx, cy, bw, bh = pred[keep, 0], pred[keep, 1], pred[keep, 2], pred[keep, 3]
        x0, y0 = cx - bw / 2, cy - bh / 2
        rects = [[float(a), float(b), float(c), float(d)]
                 for a, b, c, d in zip(x0, y0, bw, bh)]
        idx = cv2.dnn.NMSBoxes(rects, score[keep].astype(float).tolist(),
                               self.conf, self.iou)
        boxes = []
        for i in np.array(idx).reshape(-1):
            boxes.append({"cls": self.CLASSES[int(cls_id[keep][i])]
                          if int(cls_id[keep][i]) < len(self.CLASSES)
                          else str(int(cls_id[keep][i])),
                          "xyxy": [x0[i] / iw, y0[i] / ih,
                                   (x0[i] + bw[i]) / iw, (y0[i] + bh[i]) / ih],
                          "confidence": float(score[keep][i])})
        return boxes


def build_yolo_worker(onnx_path: str, threads: int = 2):
    try:
        w = YoloWorker(str(onnx_path), threads=threads)
    except Exception as e:                    # noqa: BLE001 — degrade, never crash
        _log(f"WARN yolo worker unavailable: {e!r}")
        return None
    w.start()
    _log(f"yolo worker: {Path(onnx_path).name} @{w.size[1]}x{w.size[0]} "
         f"({threads} threads, async)")
    return w


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------

@dataclass
class ShadowStats:
    n_records: int = 0
    tick_ms: list[float] = field(default_factory=list)
    source_latency_s: list[float] = field(default_factory=list)
    scorer_ok_frames: int = 0
    seg_fresh_frames: int = 0

    def pct(self, q: float) -> float:
        return float(np.percentile(self.tick_ms, q)) if self.tick_ms else float("nan")

    def summary(self) -> str:
        over = sum(1 for t in self.tick_ms if t > TICK_BUDGET_MS)
        lat = (f" · source latency median "
               f"{float(np.median(self.source_latency_s)):.2f}s"
               if self.source_latency_s else "")
        return (f"{self.n_records} records · tick p50 {self.pct(50):.1f} ms / "
                f"p95 {self.pct(95):.1f} ms (budget {TICK_BUDGET_MS:.0f} ms, "
                f"over {over}/{len(self.tick_ms)}) · scorer on "
                f"{self.scorer_ok_frames}/{self.n_records} · seg fresh on "
                f"{self.seg_fresh_frames}/{self.n_records}{lat}")


class BoardProbe:
    """Firmware ARM clock, SoC temperature and throttle flags, sampled every
    ``period_s`` (vcgencmd is a ~10 ms subprocess). The benchmark lesson of
    2026-09-03 applies to the field logs too: `scaling_cur_freq` misses the
    under-voltage clock drops, so the record carries the firmware clock. All
    None off the Pi."""

    def __init__(self, period_s: float = 3.0):
        self.period_s = period_s
        self._last = 0.0
        self._cache = {"clock_mhz": None, "temp_c": None, "throttled": None}
        self._enabled = os.path.exists("/sys/class/thermal/thermal_zone0/temp")

    def sample(self) -> dict:
        if not self._enabled:
            return self._cache
        now = time.monotonic()
        if now - self._last < self.period_s:
            return self._cache
        self._last = now
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as fh:
                self._cache["temp_c"] = round(int(fh.read().strip()) / 1000.0, 1)
        except (OSError, ValueError):
            pass
        try:
            import subprocess
            out = subprocess.run(["vcgencmd", "measure_clock", "arm"],
                                 capture_output=True, text=True, timeout=1.0).stdout
            self._cache["clock_mhz"] = int(int(out.strip().split("=")[1]) / 1e6)
            out = subprocess.run(["vcgencmd", "get_throttled"],
                                 capture_output=True, text=True, timeout=1.0).stdout
            self._cache["throttled"] = out.strip().split("=")[1]
        except Exception:
            pass
        return self._cache


class ShadowService:
    """One pipeline instance for the service lifetime (the thermal running
    average is stateful); one JSONL record per source frame."""

    #: Consecutive per-frame scorer failures before the scorer is disabled
    #: for the rest of the run (a broken artefact must not spam the journal).
    SCORER_MAX_FAILURES = 3

    def __init__(self, sources, out_path, intrinsics: dict | None = None,
                 detection: dict | None = None,
                 scorer_path: str | Path | None = None,
                 seg_provider: SegWorkerProvider | None = None,
                 yolo_worker=None, yolo_max_age_s: float = 3.0):
        from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
        from scripts.utils.calibration import load_detection, load_intrinsics
        from scripts.utils.geometry import load_extrinsics

        self.sources = sources
        self.out_path = Path(out_path)
        self.intrinsics = intrinsics if intrinsics is not None else load_intrinsics()
        self.detection = detection if detection is not None else load_detection()
        self.attitude = getattr(sources, "attitude_provider", None)
        self.seg_provider = seg_provider
        self.yolo_worker = yolo_worker
        self.yolo_max_age_s = yolo_max_age_s
        self._yolo_age: float | None = None
        self.pipeline = ObstacleDetectionPipeline(
            self.intrinsics, self.detection,
            attitude_provider=self.attitude, seg_provider=seg_provider)
        self.fisheye_height = float(
            (load_extrinsics().get("camera_height_m", {}) or {})
            .get("fisheye", 0.27))
        self._sun_cache: tuple[float, float] | None = None   # (sod, elev)
        self._board = BoardProbe()
        self._scorer = None
        self._scorer_failures = 0
        if scorer_path:
            self._init_scorer(scorer_path)

    # -- scorer ------------------------------------------------------------

    def _init_scorer(self, path) -> None:
        """Mirror fusion.py's --scorer block; any failure -> legacy-only."""
        try:
            import pandas as pd  # noqa: F401  (needed by the per-frame path)

            from scripts.fusion_model.build_features import (
                DEFAULT_LAT,
                DEFAULT_LON,
                DEFAULT_TZ,
                bin_rows_for_frame,
                frame_datetime_utc,
            )
            from scripts.fusion_model.models import (
                load_any_scorer,
                load_fusion_model_config,
                prepare_matrix,
                threat_from_score,
            )
            from scripts.sensor_processing.gps_boat1 import sun_position
            from scripts.utils.segmentation import SegDetectParams

            scorer = load_any_scorer(path)
            self._pin_scorer_threads()
            _seg_det = ((self.detection.get("segmentation", {}) or {})
                        .get("detect", {}) or {})
            self._sc = {
                "pd": pd, "scorer": scorer,
                "fm_cfg": load_fusion_model_config(),
                "bin_rows_for_frame": bin_rows_for_frame,
                "frame_datetime_utc": frame_datetime_utc,
                "prepare_matrix": prepare_matrix,
                "threat_from_score": threat_from_score,
                "sun_position": sun_position,
                "lat": DEFAULT_LAT, "lon": DEFAULT_LON, "tz": DEFAULT_TZ,
                "seg_params": SegDetectParams(
                    min_area=int(_seg_det.get("min_area", 25)),
                    water_edge_margin_px=int(
                        _seg_det.get("water_edge_margin_px", 10)),
                    max_components=int(_seg_det.get("max_components", 100))),
            }
            self._scorer = scorer
            _log(f"scoring with {path} ({type(scorer).__name__})")
        except Exception as e:
            self._scorer = None
            _log(f"WARN scorer unavailable ({path}): {e!r} — "
                 f"emitting legacy n/3 scores only")

    #: OpenMP threads for the scorer's per-frame predict. Measured on the Pi
    #: 4 (2026-09-03, replay + seg worker, 90 frames): scikit-learn's HistGBT
    #: predict spawns an OpenMP pool over all four cores for a 7-row batch,
    #: and that pool spin-waits against onnxruntime's two CNN threads —
    #: scorer cost 26 ms -> 440 ms per frame once the SegWorker is alive, tick
    #: p50 669 -> 516 ms with the pool pinned to one thread. Predictions do
    #: not depend on the thread count. None = leave the runtime default.
    SCORER_OMP_THREADS: int | None = 1

    def _pin_scorer_threads(self) -> None:
        if self.SCORER_OMP_THREADS is None:
            return
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return   # ships with scikit-learn; without it the scorer still runs
        # Keep the controller alive for the service lifetime (its __del__
        # would restore the previous limits).
        self._omp_limits = threadpool_limits(
            limits=self.SCORER_OMP_THREADS, user_api="openmp")

    def _sun_elevation(self, ts: str) -> float:
        """Solar elevation for the frame, cached per ~60 s of frame time."""
        if self._scorer is None:
            return float("nan")
        sod = _seconds_of_day(ts)
        if self._sun_cache is not None and abs(sod - self._sun_cache[0]) < 60.0:
            return self._sun_cache[1]
        sc = self._sc
        try:
            when = sc["frame_datetime_utc"](
                self.sources.clip_date, ts, sc["tz"],
                getattr(self.sources, "chunk_hms", None))
            elev, _ = sc["sun_position"](sc["lat"], sc["lon"], when)
        except ValueError:
            elev = float("nan")
        self._sun_cache = (sod, elev)
        return elev

    def _score(self, result, frame_index: int, att, elev: float
               ) -> tuple[list[float], list[float]] | None:
        """(p_obstacle, threat) or None. Mirrors fusion.py's --scorer block
        (same bin_rows_for_frame feature construction: one code path)."""
        if self._scorer is None:
            return None
        sc = self._sc
        try:
            det_boxes = None
            self._yolo_age = None
            if self.yolo_worker is not None:
                det_boxes, self._yolo_age = self.yolo_worker.latest(
                    max_age_s=self.yolo_max_age_s)
            sdf = sc["pd"].DataFrame(sc["bin_rows_for_frame"](
                result, frame_index, self.sources.clip_id, self.detection,
                self.intrinsics, sc["seg_params"], self.fisheye_height,
                attitude=att, det_boxes=det_boxes))
            f_res = result.fisheye
            sdf["sun_elevation_deg"] = elev
            sdf["luminance_mean"] = (f_res.luminance_mean
                                     if f_res else float("nan"))
            sdf["is_dark"] = bool(f_res.is_dark) if f_res else False
            sdf["thermal_quality_ok"] = (bool(result.thermal.quality_ok)
                                         if result.thermal else False)
            sdf["imu_available"] = att is not None
            scorer = sc["scorer"]
            if hasattr(scorer, "predict_proba_calibrated_df"):
                p = scorer.predict_proba_calibrated_df(sdf)
            else:
                e_m, c_m, _ = sc["prepare_matrix"](
                    sdf, sc["fm_cfg"],
                    columns=tuple(scorer.context_columns),
                    sensors=tuple(scorer.sensors))
                p = scorer.predict_proba_calibrated(e_m, c_m)
            threat = sc["threat_from_score"](
                p, sdf["min_range_m"].to_numpy(dtype=float),
                sdf["per_bin_ttc_s"].to_numpy(dtype=float),
                severity=1.0, cfg=sc["fm_cfg"])
            self._scorer_failures = 0
            return ([round(float(v), 4) for v in p],
                    [round(float(v), 4) for v in threat])
        except Exception as e:
            self._scorer_failures += 1
            _log(f"WARN scorer failed on frame {frame_index}: {e!r}")
            if self._scorer_failures >= self.SCORER_MAX_FAILURES:
                _log(f"WARN scorer disabled after "
                     f"{self._scorer_failures} consecutive failures")
                self._scorer = None
            return None

    # -- main loop ---------------------------------------------------------

    def run(self, max_frames: int | None = None) -> ShadowStats:
        stats = ShadowStats()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        _log(f"writing {self.out_path}")
        with open(self.out_path, "a") as out:
            for frame in self.sources.frames():
                t0 = time.perf_counter()
                if self.attitude is not None and hasattr(self.attitude,
                                                         "set_time"):
                    self.attitude.set_time(frame.ts)
                elev = self._sun_elevation(frame.ts)
                if np.isfinite(elev):
                    self.pipeline.set_sun_elevation(elev)
                fid = (self.sources.frame_id_for(frame.ts)
                       if hasattr(self.sources, "frame_id_for") else None)
                result = self.pipeline.process_frame(
                    frame.fisheye, frame.thermal, frame.points,
                    timestamp=frame.ts, frame_id=fid)
                # Offer the undistorted frame to the async seg worker for the
                # NEXT ticks' mask (one-frame pipelining; submit never blocks).
                if (self.seg_provider is not None
                        and result.fisheye is not None):
                    self.seg_provider.submit(result.fisheye.undistorted)
                if self.yolo_worker is not None and result.fisheye is not None:
                    self.yolo_worker.submit(result.fisheye.undistorted)
                att = (self.attitude.get()
                       if self.attitude is not None else None)
                rec = self._record(result, frame, att, elev,
                                   stats.n_records, t0)
                out.write(json.dumps(rec) + "\n")
                out.flush()      # tail -f / crash-safe: a service, not a batch
                stats.n_records += 1
                stats.tick_ms.append(rec["shadow"]["tick_ms"])
                if rec["shadow"]["scorer_ok"]:
                    stats.scorer_ok_frames += 1
                if rec["shadow"]["seg_fresh"]:
                    stats.seg_fresh_frames += 1
                if frame.source_latency_s is not None:
                    stats.source_latency_s.append(frame.source_latency_s)
                if max_frames is not None and stats.n_records >= max_frames:
                    break
        if getattr(self.sources, "snapshot_dir", None) is not None:
            _log(f"camera frames {self.sources.camera_frames} / misses "
                 f"{self.sources.camera_misses} / rows skipped "
                 f"{self.sources.skipped_rows} (latest-wins)")
        _log(stats.summary())
        return stats

    def _record(self, result, frame: ShadowFrame, att, elev: float,
                frame_index: int, t0: float) -> dict:
        fr = result.fusion
        rec = {
            "protocol": PROTOCOL_VERSION,
            "timestamp": frame.ts,
            "clip_id": self.sources.clip_id,
            "bin_centers_deg": fr.bin_centers.tolist(),
            "scores": fr.scores,
            "min_range_m": [float(r) if r is not None else None
                            for r in fr.min_ranges],
            "sensor_hits": fr.sensor_hit_mask.astype(int).tolist(),
            "tracked": False,
            "attitude": ({"roll_deg": att.roll_deg,
                          "pitch_deg": att.pitch_deg,
                          "yaw_deg": att.yaw_deg} if att is not None else None),
            "per_bin_velocity_mps": [float(v) if v is not None else None
                                     for v in (fr.per_bin_velocity_mps or [])],
            "per_bin_ttc_s": [float(t) if t is not None else None
                              for t in (fr.per_bin_ttc_s or [])],
        }
        scored = self._score(result, frame_index, att, elev)
        if scored is not None:
            rec["p_obstacle"], rec["threat"] = scored
        seg_age = seg_fresh = None
        if self.seg_provider is not None:
            seg_age = self.seg_provider.last_age_s
            seg_fresh = (seg_age is not None
                         and seg_age <= self.seg_provider.max_age_s)
        rec["shadow"] = {
            "tick_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "seg_age_s": (round(seg_age, 2) if seg_age is not None else None),
            "seg_fresh": bool(seg_fresh) if seg_fresh is not None else False,
            "scorer_ok": scored is not None,
        }
        if self.yolo_worker is not None:
            rec["shadow"]["yolo_age_s"] = (round(self._yolo_age, 2)
                                           if self._yolo_age is not None else None)
            rec["shadow"]["yolo_fresh"] = (self._yolo_age is not None
                                           and self._yolo_age <= self.yolo_max_age_s)
        if frame.source_latency_s is not None:
            rec["shadow"]["source_latency_s"] = round(frame.source_latency_s, 2)
        if getattr(self.sources, "snapshot_dir", None) is not None:
            rec["shadow"]["camera"] = frame.fisheye is not None
            rec["shadow"]["camera_offset_frames"] = self.sources.camera_offset_frames
            rec["shadow"]["skipped_rows"] = int(self.sources.skipped_rows)
        board = self._board.sample()
        if board.get("temp_c") is not None:
            rec["shadow"]["board"] = dict(board)
        return rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_out(args, sources) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.replay:
        return (REPO_ROOT / "results" / "shadow"
                / f"shadow_{sources.triplet.timestamp}.jsonl")
    if args.tail:
        return Path(args.tail) / f"shadow_{stamp}.jsonl"
    # --live: same per-boot session-folder convention as continuous_capture.
    from scripts.data_collection import continuous_capture as cc
    return cc.session_capture_dir(cc.CAPTURE_DIR) / f"shadow_{stamp}.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", metavar="TRIPLET",
                      help="bench: replay a recorded triplet prefix")
    mode.add_argument("--tail", metavar="DIR",
                      help="run beside capture: tail the session folder's "
                           "growing CSVs (read-only; no camera pixels)")
    mode.add_argument("--live", action="store_true",
                      help="own the sensors (Pi only; capture service must "
                           "be STOPPED)")
    ap.add_argument("--out", default=None,
                    help="output JSONL (appended); a DIRECTORY (existing, or "
                         "written with a trailing /) gets a fresh "
                         "shadow_<stamp>.jsonl per start")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--scorer", default=str(DEFAULT_SCORER),
                    help="scorer artefact ('none' disables; default the "
                         "live v1b bundle, degrades to legacy scores if "
                         "unloadable)")
    ap.add_argument("--seg-onnx", default=None,
                    help="eWaSR/student ONNX for the async SegWorker "
                         "(replay/live modes; needs onnxruntime)")
    ap.add_argument("--seg-max-age", type=float, default=DEFAULT_SEG_MAX_AGE_S)
    ap.add_argument("--yolo-onnx", default=None,
                    help="YOLOv8 ONNX (no NMS) for an async typed-detection "
                         "worker; its boxes fill the scorer's yolo_* columns "
                         "(replay/live modes). Pi 4: v8n-640 ~0.5 Hz beside "
                         "the tick — a Pi 5 feature in practice.")
    ap.add_argument("--yolo-max-age", type=float, default=3.0,
                    help="boxes older than this are not fed to the scorer")
    ap.add_argument("--realtime", action="store_true",
                    help="replay at the 3 fps camera cadence instead of "
                         "as-fast-as-possible")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--snapshot-dir", default=None, metavar="DIR",
                    help="tail mode: read capture's latest-frame snapshot "
                         "(ASVPROJECT_SNAPSHOT_DIR: /run/asvproject under the "
                         "capture unit, /dev/shm/asvproject for manual runs) so the tick has camera pixels -> the "
                         "full stack (--seg-onnx/--yolo-onnx) runs BESIDE "
                         "capture, latest-wins")
    ap.add_argument("--idle-timeout", type=float, default=None,
                    help="tail mode: stop after this many seconds without "
                         "new rows (default: run until interrupted)")
    ap.add_argument("--detection", default=None,
                    help="override path to detection.yaml")
    ap.add_argument("--intrinsics", default=None,
                    help="override path to intrinsics.yaml")
    args = ap.parse_args(argv)

    from scripts.utils.calibration import load_detection, load_intrinsics
    detection = (load_detection(args.detection) if args.detection
                 else load_detection())
    intrinsics = (load_intrinsics(args.intrinsics) if args.intrinsics
                  else load_intrinsics())

    if args.replay:
        sources = ReplaySources(args.replay, detection,
                                realtime=args.realtime, fps=args.fps)
    elif args.tail:
        sources = TailSources(args.tail, idle_timeout_s=args.idle_timeout,
                              snapshot_dir=args.snapshot_dir)
    else:
        sources = LiveSources(fps=args.fps)

    tail_blind = bool(args.tail) and not args.snapshot_dir
    seg_provider = None
    if args.seg_onnx:
        if tail_blind:
            _log("WARN --seg-onnx ignored in --tail mode without "
                 "--snapshot-dir (no camera pixels)")
        else:
            worker = build_seg_worker(args.seg_onnx)
            if worker is not None:
                seg_provider = SegWorkerProvider(worker,
                                                 max_age_s=args.seg_max_age)

    yolo_worker = None
    if args.yolo_onnx and not tail_blind:
        yolo_worker = build_yolo_worker(args.yolo_onnx)
    elif args.yolo_onnx:
        _log("WARN --yolo-onnx ignored in --tail mode without "
             "--snapshot-dir (no camera pixels)")

    scorer_path = None if args.scorer in (None, "", "none") else args.scorer
    out = Path(args.out) if args.out else _default_out(args, sources)
    if args.out and (out.is_dir() or str(args.out).endswith("/")):
        # A directory means "a fresh file per start in here" -- what the
        # systemd unit wants (one JSONL per service start, like capture's
        # per-boot session folders), without a shell wrapper for the stamp.
        out.mkdir(parents=True, exist_ok=True)
        out = out / f"shadow_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
    service = ShadowService(sources, out, intrinsics=intrinsics,
                            detection=detection, scorer_path=scorer_path,
                            seg_provider=seg_provider, yolo_worker=yolo_worker,
                            yolo_max_age_s=args.yolo_max_age)
    try:
        service.run(max_frames=args.max_frames)
    except KeyboardInterrupt:
        _log("interrupted — records up to here are on disk")
    finally:
        if seg_provider is not None:
            try:
                seg_provider.worker.stop()
            except Exception:
                pass
        sources.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
