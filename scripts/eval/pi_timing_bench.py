"""On-device timing benchmark: every model and pipeline stage, alone and in
combination, with the CPU clock recorded per iteration.

Why the clock matters: on the boat feed the Pi 4 drops to 600 MHz for seconds
at a time under sustained load (under-voltage, 2026-09-03). A latency
distribution taken across such episodes is bimodal and its p50 means nothing.
Every iteration here reads the ARM clock from the FIRMWARE (`vcgencmd
measure_clock arm`; `scaling_cur_freq` is only the governor's set-point and
kept reporting 1.8 GHz through 600 MHz under-voltage episodes -- measured
2026-09-03, process_fisheye 245 ms vs 640 ms on the same frame) AND the SoC
temperature before and after, and is tagged `full` only when the clock is at
its maximum and the SoC is under the Pi 4's 80 °C throttle point (the sealed
box idles at 73-78 °C in a warm lab); the report shows the full-clock
percentiles, the all-samples percentiles, and the throttled fraction side by
side. Each model/combination starts only once the SoC has cooled below
--cool-c (sealed box: this is what makes runs comparable), and a model's
iteration count is capped so one run stays under ~--run-budget seconds.
`vcgencmd get_throttled` + temperature are logged before and after.

Sections (pick with --only; default all):

  models    onnxruntime inference latency of every deployable model file
            (segmentation students/teachers, thermal student, YOLO exports),
            at 2 and 4 intra-op threads, on a real frame from the triplet.
            YOLO numbers are the graph only -- ultralytics exports without
            NMS, so decode+NMS (a few ms) is not included.
  pipeline  the fusion pipeline over the triplet, as the shadow service runs
            it (ReplaySources -> ObstacleDetectionPipeline -> scorer), with
            per-stage timers: fisheye classical (and its horizon RANSAC),
            thermal classical, radar, scorer, fusion remainder; the async
            SegWorker's inference latency + achieved rate when a seg model
            is configured. Combinations are named configs (see COMBOS).
  codec     sector_codec encode + fragment cost per record.

Usage (on the Pi, capture service STOPPED for clean numbers):

    python3 -m scripts.eval.pi_timing_bench \
        --triplet ~/captures/<session>/<ts> --out /tmp/pi_timing.json

Laptop: the models section needs onnxruntime; the pipeline section runs
anywhere (numbers are then laptop numbers, tagged by platform).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CPUFREQ = Path("/sys/devices/system/cpu/cpu0/cpufreq")
EWASR = Path.home() / "ewasr"

#: Model files to time. (label, path, family). Missing files are skipped and
#: reported, never fatal -- the ladder differs per box.
MODEL_FILES = [
    ("seg student 512×384 int8", EWASR / "student_512x384.int8.onnx", "seg"),
    ("seg student 640×480 int8", EWASR / "student_640x480.int8.onnx", "seg"),
    ("seg student 864×648 int8", EWASR / "student864t_864x648.int8.onnx", "seg"),
    ("seg student 512×384 fp32", EWASR / "student_512x384.onnx", "seg"),
    ("seg student 864×648 fp32", EWASR / "student_864x648.onnx", "seg"),
    ("seg teacher eWaSR 256×192 int8", EWASR / "ewasr_lars_256x192.int8.onnx", "seg"),
    ("seg teacher eWaSR 512×384 int8", EWASR / "ewasr_lars_512x384.int8.onnx", "seg"),
    ("seg teacher eWaSR 864×648 int8", EWASR / "ewasr_lars_864x648.pc_int8.onnx", "seg"),
    ("thermal seg student 160×120 fp32",
     EWASR / "thermal_student_night_v3_s1_160x120.fp32.onnx", "thermal"),
    ("YOLOv8n audited 640 fp32", EWASR / "yolo" / "yolov8n_audited_640.onnx", "yolo"),
    ("YOLOv8n audited 864 fp32", EWASR / "yolo" / "yolov8n_audited_864.onnx", "yolo"),
    ("YOLOv8s audited 640 fp32", EWASR / "yolo" / "yolov8s_audited_640.onnx", "yolo"),
    ("YOLOv8s audited 864 fp32", EWASR / "yolo" / "yolov8s_audited_864.onnx", "yolo"),
]

#: Pipeline combinations. Each is (label, scorer, seg model, detection
#: overrides). Scorer: None / "v1a" / "v1b". Overrides are merged over
#: configs/detection.yaml (one level deep).
COMBOS = [
    ("classical, no scorer", None, None, {}),
    ("classical + v1a scorer", "v1a", None, {}),
    ("classical + v1b scorer", "v1b", None, {}),
    ("classical + v1b, association off", "v1b", None,
     {"fusion": {"association": {"enabled": False}}}),
    ("+ seg student 512 int8", "v1b", "student_512x384.int8.onnx", {}),
    ("+ seg student 864 int8", "v1b", "student864t_864x648.int8.onnx", {}),
    ("+ seg 512 + target motion", "v1b", "student_512x384.int8.onnx",
     {"motion": {"targets": {"enabled": True}}}),
    # 2026-09-04: the learned primaries (AuthorTwo) — seg replaces the classical
    # fisheye (lazy horizon + seg detector = configs/detection_shadow_seg.yaml)
    ("SEG-PRIMARY: seg 512 + lazy horizon + seg detector", "v1b",
     "student_512x384.int8.onnx",
     {"horizon": {"lazy": True}, "fisheye": {"detector": "segmentation"},
      "motion": {"targets": {"enabled": True}}}),
    ("SEG-PRIMARY + YOLOv8n-640 worker", "v1b", "student_512x384.int8.onnx",
     {"horizon": {"lazy": True}, "fisheye": {"detector": "segmentation"},
      "motion": {"targets": {"enabled": True}},
      "_yolo": "yolo/yolov8n_audited_640.onnx"}),
]


# ---------------------------------------------------------------------------
# platform + clock helpers
# ---------------------------------------------------------------------------

def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def cur_khz() -> int | None:
    """Actual ARM clock in kHz from the firmware (falls back to cpufreq's
    set-point off-Pi). ~10 ms per call: keep it outside timed regions."""
    v = vcgencmd("measure_clock", "arm")
    if v and "=" in v:
        try:
            return int(v.split("=")[1]) // 1000
        except ValueError:
            pass
    v = _read(CPUFREQ / "scaling_cur_freq")
    return int(v) if v else None


THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
THROTTLE_C = 80.0          # Pi 4 firmware starts capping the ARM clock here


def soc_temp_c() -> float | None:
    v = _read(THERMAL)
    return int(v) / 1000.0 if v else None


def cool_down(limit_c: float, max_wait_s: float, log) -> float | None:
    """Block until the SoC is below limit_c (or max_wait_s passed)."""
    t0 = time.time()
    t = soc_temp_c()
    if t is None:
        return None
    if t > limit_c:
        log(f"cool-down: {t:.1f} °C > {limit_c:.0f} °C, waiting")
    while t is not None and t > limit_c and time.time() - t0 < max_wait_s:
        time.sleep(5)
        t = soc_temp_c()
    if time.time() - t0 > 1:
        log(f"cool-down: {t:.1f} °C after {time.time() - t0:.0f} s")
    return t


def max_khz() -> int | None:
    v = _read(CPUFREQ / "scaling_max_freq")
    return int(v) if v else None


def vcgencmd(*args: str) -> str | None:
    if shutil.which("vcgencmd") is None:
        return None
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def power_state() -> dict:
    return {"throttled": vcgencmd("get_throttled"),
            "temp_c": soc_temp_c(),
            "arm_khz": cur_khz()}


class ClockTagger:
    """Per-iteration latency samples tagged full-clock or not."""

    def __init__(self):
        self.max = max_khz()
        self.ms: list[float] = []
        self.full: list[bool] = []

    def add(self, dt_s: float, k0: int | None, k1: int | None,
            t0_c: float | None = None, t1_c: float | None = None) -> None:
        self.ms.append(dt_s * 1000.0)
        ok = (self.max is None
              or (k0 is not None and k1 is not None
                  and k0 >= self.max * 0.98 and k1 >= self.max * 0.98))
        cool = all(t is None or t < THROTTLE_C for t in (t0_c, t1_c))
        self.full.append(bool(ok and cool))

    def stats(self) -> dict:
        a = np.array(self.ms, dtype=float)
        f = np.array(self.full, dtype=bool)
        out = {"n": int(a.size),
               "p50_all_ms": float(np.median(a)) if a.size else None,
               "p95_all_ms": float(np.percentile(a, 95)) if a.size else None,
               "throttled_frac": float(1.0 - f.mean()) if a.size else None,
               "n_full": int(f.sum())}
        if f.sum() >= 3:
            ff = a[f]
            out["p50_full_ms"] = float(np.median(ff))
            out["p95_full_ms"] = float(np.percentile(ff, 95))
            out["fps_full"] = 1000.0 / float(np.median(ff))
        else:
            out["p50_full_ms"] = out["p95_full_ms"] = out["fps_full"] = None
        return out


def timed(fn, tagger: ClockTagger):
    k0, c0 = cur_khz(), soc_temp_c()
    t0 = time.perf_counter()
    r = fn()
    dt = time.perf_counter() - t0
    tagger.add(dt, k0, cur_khz(), c0, soc_temp_c())
    return r


# ---------------------------------------------------------------------------
# section: models
# ---------------------------------------------------------------------------

def _first_frames(triplet: str, n: int = 1):
    from scripts.sensor_processing.pipeline import iterate_triplet
    from scripts.utils.calibration import load_detection
    from scripts.utils.datasets import resolve_triplet
    trip = resolve_triplet(triplet)
    det = load_detection()
    frames = []
    for ts, fish, therm, pts in iterate_triplet(trip, det):
        frames.append((ts, fish, therm, pts))
        if len(frames) >= n:
            break
    return frames


def _model_input(sess, fish_bgr, therm_bgr):
    """Build one real input for the model from a fisheye/thermal frame."""
    import cv2
    inp = sess.get_inputs()[0]
    shape = [1 if (d is None or isinstance(d, str)) else int(d) for d in inp.shape]
    if len(shape) != 4:
        return inp.name, np.random.rand(*shape).astype(np.float32)
    n, c, h, w = shape
    src = therm_bgr if (h <= 160 and w <= 200 and therm_bgr is not None) else fish_bgr
    img = cv2.resize(src, (w, h))[:, :, ::-1].astype(np.float32) / 255.0
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
    if c != 3:
        x = np.random.rand(*shape).astype(np.float32)
    return inp.name, x


def bench_models(frames, threads_list, iters: int, log, cool_c: float = 70.0,
                 run_budget_s: float = 20.0) -> list[dict]:
    try:
        import onnxruntime as ort
    except ImportError:
        log("models: onnxruntime not installed here — skipped")
        return []
    _, fish, therm, _ = frames[0]
    rows = []
    for label, path, family in MODEL_FILES:
        if not path.exists():
            rows.append({"label": label, "family": family, "path": str(path),
                         "missing": True})
            log(f"models: missing {path.name}")
            continue
        for threads in threads_list:
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(path), so,
                                        providers=["CPUExecutionProvider"])
            name, x = _model_input(sess, fish, therm)
            t_start = cool_down(cool_c, 300, log)
            t0 = time.perf_counter()
            for _ in range(2):
                sess.run(None, {name: x})
            per = (time.perf_counter() - t0) / 2
            n_iter = int(max(5, min(iters, run_budget_s / max(per, 1e-3))))
            tag = ClockTagger()
            for _ in range(n_iter):
                timed(lambda s=sess, n=name, xx=x: s.run(None, {n: xx}), tag)
            st = tag.stats()
            rows.append({"label": label, "family": family, "path": str(path),
                         "threads": threads, "input": list(x.shape),
                         "size_mb": round(path.stat().st_size / 1e6, 1),
                         "temp_start_c": t_start, "temp_end_c": soc_temp_c(),
                         **st})
            log(f"models: {label} @{threads}t  p50_full "
                f"{st['p50_full_ms'] and round(st['p50_full_ms'], 1)} ms  "
                f"(all p50 {st['p50_all_ms']:.1f}, throttled "
                f"{st['throttled_frac']:.0%})")
            del sess
    return rows


# ---------------------------------------------------------------------------
# section: pipeline combos
# ---------------------------------------------------------------------------

def _merge(base: dict, over: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def bench_pipeline(triplet: str, combos, n_frames: int, seg_threads: int,
                   log, cool_c: float = 70.0) -> list[dict]:
    import scripts.data_collection.shadow_fusion as sf
    import scripts.sensor_processing.pipeline as pl
    from scripts.utils.calibration import load_detection, load_intrinsics

    base_det = load_detection()
    intrinsics = load_intrinsics()
    rows = []
    for label, scorer, seg_model, over in combos:
        det = _merge(base_det, over)
        sources = sf.ReplaySources(triplet, det, realtime=False)
        seg_provider = worker = None
        seg_tag = ClockTagger()
        if seg_model:
            worker = sf.build_seg_worker(str(EWASR / seg_model), threads=seg_threads)
            if worker is None:
                rows.append({"label": label, "skipped": "seg worker unavailable"})
                log(f"pipeline: {label}: seg worker unavailable — skipped")
                continue
            orig_infer = worker._infer

            def _infer(frame, _o=orig_infer, _t=seg_tag):
                return timed(lambda: _o(frame), _t)
            worker._infer = _infer
            seg_provider = sf.SegWorkerProvider(worker)
        yolo_rel = over.pop("_yolo", None) if isinstance(over, dict) else None
        yolo_worker = None
        yolo_tag = ClockTagger()
        if yolo_rel:
            yolo_worker = sf.build_yolo_worker(str(EWASR / yolo_rel), threads=seg_threads)
            if yolo_worker is None:
                rows.append({"label": label, "skipped": "yolo worker unavailable"})
                log(f"pipeline: {label}: yolo worker unavailable — skipped")
                continue
            orig_yinfer = yolo_worker._infer

            def _yinfer(frame, _o=orig_yinfer, _t=yolo_tag):
                return timed(lambda: _o(frame), _t)
            yolo_worker._infer = _yinfer
        scorer_path = {None: None,
                       "v1a": REPO_ROOT / "models" / "fusion_scorer_live.json",
                       "v1b": REPO_ROOT / "models" / "fusion_scorer_live_v1b.joblib"}[scorer]
        out_path = Path(tempfile.gettempdir()) / f"pi_timing_{abs(hash(label))}.jsonl"
        if out_path.exists():
            out_path.unlink()
        service = sf.ShadowService(sources, out_path, intrinsics=intrinsics,
                                   detection=det, scorer_path=scorer_path,
                                   seg_provider=seg_provider,
                                   yolo_worker=yolo_worker)
        # -- stage timers (bound-method wrappers; one pipeline instance) ----
        # Raw ms per call; the full-clock tag is decided per FRAME from the
        # firmware clock read before process_frame and after _record (which
        # holds the scorer), then applied to every stage sample of the frame.
        stages = {k: ClockTagger() for k in
                  ("fisheye", "thermal", "mmwave", "horizon", "score", "tick")}
        pipe = service.pipeline
        frame_clk = {"k0": None, "c0": None}

        def raw(fn, tag):
            t0 = time.perf_counter()
            r = fn()
            tag.ms.append((time.perf_counter() - t0) * 1000.0)
            return r
        for meth in ("process_fisheye", "process_thermal", "process_mmwave"):
            orig = getattr(pipe, meth)
            key = meth.split("_")[1]

            def wrap(*a, _o=orig, _k=key, **kw):
                return raw(lambda: _o(*a, **kw), stages[_k])
            setattr(pipe, meth, wrap)
        orig_h = pl.detect_horizon

        def wrap_h(*a, **kw):
            return raw(lambda: orig_h(*a, **kw), stages["horizon"])
        pl.detect_horizon = wrap_h
        orig_score = service._score

        def wrap_s(*a, **kw):
            return raw(lambda: orig_score(*a, **kw), stages["score"])
        service._score = wrap_s
        orig_pf = pipe.process_frame

        def wrap_pf(*a, **kw):
            frame_clk["k0"], frame_clk["c0"] = cur_khz(), soc_temp_c()
            return raw(lambda: orig_pf(*a, **kw), stages["tick"])
        pipe.process_frame = wrap_pf
        orig_rec = service._record

        def wrap_rec(*a, **kw):
            r = orig_rec(*a, **kw)
            k1, c1 = cur_khz(), soc_temp_c()
            probe = ClockTagger()
            probe.add(0.0, frame_clk["k0"], k1, frame_clk["c0"], c1)
            flag = probe.full[0]
            for tg in stages.values():
                tg.full.extend([flag] * (len(tg.ms) - len(tg.full)))
            return r
        service._record = wrap_rec

        cool_down(cool_c, 300, log)
        p_before = power_state()
        t_wall0 = time.perf_counter()
        st = service.run(max_frames=n_frames)
        wall = time.perf_counter() - t_wall0
        p_after = power_state()
        pl.detect_horizon = orig_h
        if worker is not None:
            worker.stop()
        if yolo_worker is not None:
            yolo_worker.stop()
        n_records = st.n_records
        row = {"label": label, "scorer": scorer, "seg_model": seg_model,
               "overrides": over, "n_frames": n_records,
               "wall_s": wall, "fps_wall": n_records / wall if wall else None,
               "tick_service_p50_ms": st.pct(50), "tick_service_p95_ms": st.pct(95),
               "seg_fresh_frames": st.seg_fresh_frames,
               "scorer_ok_frames": st.scorer_ok_frames,
               "power_before": p_before, "power_after": p_after,
               "stages": {k: v.stats() for k, v in stages.items()}}
        if seg_model:
            row["seg_infer"] = seg_tag.stats()
            row["seg_rate_hz"] = (seg_tag.stats()["n"] / wall) if wall else None
        if yolo_rel:
            row["yolo_model"] = yolo_rel
            row["yolo_infer"] = yolo_tag.stats()
            row["yolo_rate_hz"] = (yolo_tag.stats()["n"] / wall) if wall else None
        # horizon is called twice per frame (fisheye + thermal): per-frame sum
        h = row["stages"]["horizon"]
        if h["p50_full_ms"] is not None:
            h["per_frame_full_ms"] = h["p50_full_ms"] * (h["n"] / max(1, n_records))
        rows.append(row)
        tk = row["stages"]["tick"]
        log(f"pipeline: {label}: tick p50_full {tk['p50_full_ms'] and round(tk['p50_full_ms'])} ms "
            f"(all p50 {tk['p50_all_ms']:.0f}, throttled {tk['throttled_frac']:.0%}), "
            f"{row['fps_wall']:.2f} fps wall"
            + (f", seg {row['seg_rate_hz']:.2f} Hz" if seg_model else ""))
    return rows


# ---------------------------------------------------------------------------
# section: codec
# ---------------------------------------------------------------------------

def bench_codec(iters: int, log) -> dict:
    from scripts.data_collection.sector_codec import (SAFE_NOTIFY_BYTES,
                                                      decode_packet,
                                                      encode_record,
                                                      fragment)
    rec = {"timestamp": "12:00:00.0", "bin_centers_deg": [-45, -30, -15, 0, 15, 30, 45],
           "scores": [0.33, 0.0, 0.66, 1.0, 0.33, 0.0, 0.33],
           "min_range_m": [5.5, None, 2.7, 3.4, 1.6, 0.8, 0.9],
           "p_obstacle": [0.1, 0.3, 0.7, 0.9, 0.8, 0.6, 0.7],
           "shadow": {"tick_ms": 66, "scorer_ok": True, "seg_fresh": False,
                      "source_latency_s": 0.1}}
    enc = ClockTagger()
    frag = ClockTagger()
    dec = ClockTagger()
    pkt = None
    for i in range(iters):
        pkt = timed(lambda: encode_record(rec, i), enc)
        frs = timed(lambda: fragment(pkt, i, SAFE_NOTIFY_BYTES), frag)
        timed(lambda: decode_packet(pkt), dec)
    out = {"packet_bytes": len(pkt), "fragments_at_20B": len(frs),
           "encode_us": enc.stats()["p50_all_ms"] * 1000,
           "fragment_us": frag.stats()["p50_all_ms"] * 1000,
           "decode_us": dec.stats()["p50_all_ms"] * 1000}
    log(f"codec: {out['packet_bytes']} B packet -> {out['fragments_at_20B']} "
        f"fragments; encode {out['encode_us']:.0f} µs, fragment "
        f"{out['fragment_us']:.0f} µs, decode {out['decode_us']:.0f} µs")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--triplet", required=True, help="triplet prefix (a real clip)")
    ap.add_argument("--out", required=True, help="JSON report path")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["models", "pipeline", "codec"])
    ap.add_argument("--frames", type=int, default=60, help="frames per pipeline combo")
    ap.add_argument("--iters", type=int, default=30, help="iterations per model")
    ap.add_argument("--threads", type=int, nargs="*", default=[2, 4],
                    help="onnxruntime intra-op thread counts for the models section")
    ap.add_argument("--seg-threads", type=int, default=2,
                    help="SegWorker threads in the pipeline section (deployment: 2)")
    ap.add_argument("--combos", nargs="*", default=None,
                    help="pipeline combo labels to run (default all)")
    ap.add_argument("--cool-c", type=float, default=70.0,
                    help="wait for the SoC to be below this before each run")
    ap.add_argument("--run-budget", type=float, default=20.0,
                    help="cap a model's timed loop at about this many seconds")
    args = ap.parse_args(argv)

    sections = set(args.only or ["models", "pipeline", "codec"])
    report = {"platform": {"node": platform.node(), "machine": platform.machine(),
                           "python": platform.python_version(),
                           "arm_max_khz": max_khz(),
                           "model": _read(Path("/proc/device-tree/model"))},
              "triplet": args.triplet, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "power_start": power_state()}
    log = lambda s: print(f"[bench] {s}", flush=True)  # noqa: E731
    log(f"platform {report['platform']}")

    frames = _first_frames(args.triplet, 1)
    if "models" in sections:
        report["models"] = bench_models(frames, args.threads, args.iters, log,
                                        args.cool_c, args.run_budget)
    if "pipeline" in sections:
        combos = COMBOS if args.combos is None else [
            c for c in COMBOS if c[0] in args.combos]
        report["pipeline"] = bench_pipeline(args.triplet, combos, args.frames,
                                            args.seg_threads, log, args.cool_c)
    if "codec" in sections:
        report["codec"] = bench_codec(2000, log)
    report["power_end"] = power_state()
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
