"""Latest-wins async segmentation worker for the Pi 4 (onnxruntime).

The two-tier runtime architecture (docs/history/2026-07-06_pi_quantisation.md): the
*reactive* tier (radar Doppler/TTC, fusion binning, heading, classical
detectors) runs per sensor frame and must never block on the CNN; the
*deliberative* tier (this worker) segments at whatever rate the model
sustains (~1.1 Hz at 512x384 int8, 2 threads) and consumers always take the
freshest mask, gated by staleness. Same design as continuous_capture's
RadarReader: a thread owns the slow resource, nobody queues behind it.

Contract:
  - `submit(frame_rgb, ts)` never blocks and never queues: a frame submitted
    while another is pending *replaces* it (the dropped one is counted).
  - `latest(max_age_s)` returns the freshest SegResult or None when nothing
    fresh enough exists: the caller then degrades to the classical path,
    exactly like an absent mask in the offline pipeline.
  - Staleness is measured from the *frame's* timestamp, not inference end:
    what matters is how old the world in the mask is.

Standalone (numpy + cv2 + onnxruntime, no torch, no repo imports in the
class). The __main__ is a two-tier timing demo that replays a captured
fisheye mp4 at camera rate, runs the repo's classical reactive path per tick
alongside this worker, and reports tick latency, seg update rate, and mask
staleness: the measured version of the architecture, not the estimated one.

    python3 pi_seg_worker.py --onnx ewasr_lars_512x384.int8.onnx \
        --repo ~/ASVProject-ObstacleDetection \
        --video ~/captures/fisheye_<ts>.mp4 --seconds 30
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - Pi runtime dependency
    raise SystemExit("pip install onnxruntime numpy opencv-python  (run on the Pi)")

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

#: Consumers treat a mask older than this as absent (degrade to classical).
DEFAULT_MAX_AGE_S = 2.0


@dataclass
class SegResult:
    mask: np.ndarray          # HxW uint8 class ids (0=obstacle,1=water,2=sky),
                              # upsampled to the submitted frame's resolution
    frame_ts: float           # time.monotonic() when the frame was submitted
    done_ts: float            # time.monotonic() when inference finished
    seq: int                  # increments per completed inference
    infer_ms: float           # preprocess + CNN + argmax + upsample
    derived: object = None    # postprocess(mask) output (e.g. the free-space
                              # profile), computed in the worker thread so the
                              # reactive tick only reads it. None if no
                              # postprocess was configured or it raised.

    def age_s(self, now: float | None = None) -> float:
        """How old the *world in this mask* is (measured from frame capture)."""
        return (time.monotonic() if now is None else now) - self.frame_ts


class SegWorker(threading.Thread):
    """Run eWaSR ONNX inference on its own threads; keep only the freshest
    result. `submit` and `latest` are safe from any thread and never block
    on inference.

    `postprocess(mask) -> object` (optional) runs in the worker thread right
    after inference and its result is published as `SegResult.derived`:
    measured on-device, computing the free-space profile from a full-res mask
    costs ~200+ ms, which would blow the reactive tick budget if done by the
    consumer. Deriving in the worker keeps the reactive side to a dict read.
    """

    #: Frames darker than this mean-luminance are not segmented: RGB seg on a
    #: near-black night frame produces confident garbage (measured 2026-07-07:
    #: the night OpenWater clip segments as 86% "obstacle"). Consumers see no
    #: fresh mask and degrade to thermal/radar: the same fail-safe family as
    #: the thermal wet-cover guard. ~25/255 keeps dawn/dusk usable.
    DARKNESS_FLOOR = 25.0

    def __init__(self, onnx_path: str, threads: int = 2, postprocess=None,
                 darkness_floor: float | None = DARKNESS_FLOOR):
        super().__init__(daemon=True, name="seg-worker")
        self._postprocess = postprocess
        self._darkness_floor = darkness_floor
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads   # 2 = the Pi-4 sweet spot
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self._iname = inp.name
        self._mh, self._mw = int(inp.shape[2]), int(inp.shape[3])

        self._cond = threading.Condition()
        self._pending: tuple[np.ndarray, float] | None = None
        self._latest: SegResult | None = None
        self._stop_flag = False
        self._seq = 0
        # Telemetry
        self.submitted = 0
        self.replaced = 0      # frames dropped because a newer one arrived
        self.completed = 0
        self.too_dark = 0      # frames refused by the darkness gate
        self.infer_ms_all: list[float] = []

    # -- producer side -------------------------------------------------------

    def submit(self, frame_rgb: np.ndarray, ts: float | None = None) -> None:
        """Offer a frame. Never blocks; replaces any not-yet-started frame."""
        with self._cond:
            if self._pending is not None:
                self.replaced += 1
            self._pending = (frame_rgb, time.monotonic() if ts is None else ts)
            self.submitted += 1
            self._cond.notify()

    # -- consumer side -------------------------------------------------------

    def latest(self, max_age_s: float | None = DEFAULT_MAX_AGE_S
               ) -> SegResult | None:
        """Freshest completed result, or None if absent / older than
        `max_age_s` (pass None to disable the gate)."""
        with self._cond:
            r = self._latest
        if r is None:
            return None
        if max_age_s is not None and r.age_s() > max_age_s:
            return None
        return r

    # -- lifecycle ------------------------------------------------------------

    def run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop_flag:
                    self._cond.wait(timeout=0.5)
                if self._stop_flag:
                    return
                frame, frame_ts = self._pending
                self._pending = None
            if (self._darkness_floor is not None
                    and float(frame.mean()) < self._darkness_floor):
                # Too dark to segment meaningfully: publish nothing; the
                # last daylight mask ages out via the staleness gate.
                with self._cond:
                    self.too_dark += 1
                continue
            t0 = time.perf_counter()
            mask = self._infer(frame)
            derived = None
            if self._postprocess is not None:
                try:
                    derived = self._postprocess(mask)
                except Exception:
                    derived = None   # a broken derivation must not kill the worker
            infer_ms = (time.perf_counter() - t0) * 1000.0
            with self._cond:
                self._seq += 1
                self.completed += 1
                self.infer_ms_all.append(infer_ms)
                self._latest = SegResult(mask=mask, frame_ts=frame_ts,
                                         done_ts=time.monotonic(),
                                         seq=self._seq, infer_ms=infer_ms,
                                         derived=derived)

    def _infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        fh, fw = frame_rgb.shape[:2]
        small = (frame_rgb if (fw, fh) == (self._mw, self._mh)
                 else cv2.resize(frame_rgb, (self._mw, self._mh)))
        x = ((small.astype(np.float32) / 255.0 - _MEAN) / _STD)
        x = x.transpose(2, 0, 1)[None]
        logits = self.sess.run(None, {self._iname: x})[0][0]   # C,h',w'
        cls = logits.argmax(0).astype(np.uint8)
        return cv2.resize(cls, (fw, fh), interpolation=cv2.INTER_NEAREST)

    def stop(self, join_timeout_s: float = 10.0) -> None:
        """Signal the thread and wait for it: tearing down the ORT session
        while an inference is mid-flight aborts the process at exit."""
        with self._cond:
            self._stop_flag = True
            self._cond.notify()
        if self.is_alive():
            self.join(timeout=join_timeout_s)


# ---------------------------------------------------------------------------
# Two-tier timing demo: reactive tier @ camera rate + this worker async
# ---------------------------------------------------------------------------

def _med(xs):
    return float(np.median(xs)) if xs else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="eWaSR int8 .onnx")
    ap.add_argument("--repo", default=None,
                    help="repo root (scripts/ + configs/); enables the real "
                         "reactive tier + free-space consumption")
    ap.add_argument("--video", default=None, help="captured fisheye mp4 to replay")
    ap.add_argument("--image", default=None, help="single frame instead of --video")
    ap.add_argument("--fps", type=float, default=3.0, help="camera tick rate")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=2, help="ORT intra-op threads")
    ap.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE_S)
    args = ap.parse_args(argv)
    if not args.video and not args.image:
        ap.error("give --video or --image")

    # Frame source: replay the mp4 in a loop (or repeat one image).
    frames: list[np.ndarray] = []
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while len(frames) < 90:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f[:, :, ::-1].copy())      # BGR -> RGB
        cap.release()
    else:
        frames = [cv2.imread(args.image)[:, :, ::-1].copy()]
    if not frames:
        raise SystemExit("no frames decoded")

    # Reactive tier: the repo's classical fisheye path (undistort + horizon +
    # gradient blobs + fusion-ready bearings): the per-tick work that must
    # never wait on the CNN. Without --repo we fall back to a stub so the
    # worker half of the demo still measures.
    reactive = None
    consume_mask = None
    if args.repo:
        sys.path.insert(0, args.repo)
        from scripts.sensor_processing.pipeline import ObstacleDetectionPipeline
        from scripts.utils.calibration import load_detection, load_intrinsics
        from scripts.utils.geometry import (
            UP_LEVEL, load_extrinsics, up_from_horizon_line)
        from scripts.utils.segmentation import free_space_profile, horizon_from_water
        intr_all, det = load_intrinsics(), load_detection()
        pl = ObstacleDetectionPipeline(intr_all, det)
        intr = intr_all["fisheye"]
        K = np.asarray(intr["K"], float)
        height = float((load_extrinsics().get("camera_height_m", {}) or {})
                       .get("fisheye", 0.27))
        fcfg = det["fusion"]
        edges = np.arange(fcfg["bin_min_deg"],
                          fcfg["bin_max_deg"] + fcfg["bin_step_deg"],
                          fcfg["bin_step_deg"])
        max_range = float((det.get("range", {}) or {}).get("max_range_m", 15.0))

        def reactive(frame_rgb):
            # process_fisheye expects BGR (capture convention)
            return pl.process_fisheye(frame_rgb[:, :, ::-1])

        def consume_mask(seg):
            s, b, conf = horizon_from_water(seg)
            up = up_from_horizon_line(s, b, K) if conf > 0.3 else UP_LEVEL
            return free_space_profile(seg, K, up, height, float(intr["cx"]),
                                      float(intr["pix_deg_ratio"]), edges,
                                      max_range_m=max_range)

    # Free-space derivation runs INSIDE the worker (measured: ~200+ ms on a
    # full-res mask: too heavy for the reactive tick, whose job is only to
    # read the precomputed profile).
    worker = SegWorker(args.onnx, threads=args.threads, postprocess=consume_mask)
    worker.start()

    period = 1.0 / args.fps
    n_ticks = int(args.seconds * args.fps)
    tick_ms, ages, fresh_ticks, stale_ticks, derived_ticks = [], [], 0, 0, 0
    t_start = time.monotonic()
    for i in range(n_ticks):
        tick_deadline = t_start + (i + 1) * period
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()

        worker.submit(frame)                       # never blocks
        if reactive is not None:
            reactive(frame)                        # classical per-tick work
        res = worker.latest(max_age_s=args.max_age)
        if res is not None:
            fresh_ticks += 1
            ages.append(res.age_s())
            if res.derived is not None:            # free-space, precomputed
                derived_ticks += 1
        else:
            stale_ticks += 1

        tick_ms.append((time.perf_counter() - t0) * 1000.0)
        sleep = tick_deadline - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
    elapsed = time.monotonic() - t_start
    worker.stop()

    seg_hz = worker.completed / elapsed
    over = sum(1 for t in tick_ms if t > period * 1000.0)
    print(f"\n==== two-tier timing ({args.seconds:.0f}s @ {args.fps:g} fps ticks, "
          f"ORT {args.threads} threads) ====")
    print(f"reactive tick   : median {_med(tick_ms):6.1f} ms  "
          f"p95 {float(np.percentile(tick_ms, 95)):6.1f} ms  "
          f"deadline misses {over}/{n_ticks}"
          f"{'' if reactive else '   (stub, no --repo)'}")
    print(f"seg worker      : {worker.completed} masks in {elapsed:.1f}s = "
          f"{seg_hz:.2f} Hz   (infer+derive median {_med(worker.infer_ms_all):.0f} ms, "
          f"submitted {worker.submitted}, superseded {worker.replaced}; "
          f"free-space precomputed on {derived_ticks} fresh ticks)")
    print(f"mask freshness  : fresh on {fresh_ticks}/{n_ticks} ticks "
          f"(stale/absent {stale_ticks}) · age median "
          f"{_med(ages):.2f}s  max {max(ages) if ages else float('nan'):.2f}s "
          f"(gate {args.max_age:.1f}s)")
    verdict = "OK" if over == 0 and stale_ticks <= args.fps * args.max_age else "CHECK"
    print(f"verdict         : {verdict}: reactive tier "
          f"{'meets' if over == 0 else 'MISSES'} the "
          f"{period*1000:.0f} ms tick budget; seg refreshes every "
          f"{1.0/seg_hz if seg_hz else float('inf'):.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
