"""End-to-end seg->nav per-frame timing on the Raspberry Pi 4.

`pi_benchmark.py` / `pi_tflite_benchmark.py` time only the eWaSR CNN. This
harness times the FULL per-frame path the navigation layer actually runs:
decode, preprocess, infer, argmax + upsample to frame resolution, then the
real numpy nav functions (`horizon_from_water`, `free_space_profile`,
`seg_obstacle_detections`): to confirm the nav layer is cheap relative to
the CNN and the whole thing clears the operational cadence on-device.

Two backends (2026-07-06: ONNX added for the 64-bit deployment runtime; the
TFLite path is kept for the 32-bit history):
  --onnx  ewasr_lars_512x384.int8.onnx      # onnxruntime (the deployed path)
  --model ewasr_..._dynint8_tf214.tflite    # tflite_runtime (legacy 32-bit)

Runs against the repo's actual `scripts.utils.segmentation` / `geometry`
(no torch). Point --repo at a checkout that has scripts/ + configs/.

    python pi_seg_nav_timing.py --repo ~/ASVProject-ObstacleDetection \
        --onnx ewasr_lars_512x384.int8.onnx \
        --image institutionone_frame.jpg --threads 2 --iters 25

For the async two-tier loop (reactive tick + latest-wins seg worker), see
`pi_seg_worker.py`; this harness measures the synchronous per-frame cost.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _make_infer(args):
    """Return (infer(x_hwc_float) -> logits_hwc, model_w, model_h, label)."""
    if args.onnx:
        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit("pip install onnxruntime  (64-bit Pi)")
        so = ort.SessionOptions()
        if args.threads:
            so.intra_op_num_threads = args.threads
            so.inter_op_num_threads = 1
        sess = ort.InferenceSession(args.onnx, sess_options=so,
                                    providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        name, mh, mw = inp.name, int(inp.shape[2]), int(inp.shape[3])

        def infer(x_hwc):
            x = x_hwc.transpose(2, 0, 1)[None]          # NCHW
            out = sess.run(None, {name: x})[0][0]        # C,h',w'
            return out.transpose(1, 2, 0)                # back to HWC
        return infer, mw, mh, f"onnxruntime {args.onnx}"

    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        raise SystemExit("pip install --break-system-packages tflite-runtime "
                         "(32-bit legacy path), or use --onnx on 64-bit")
    it = Interpreter(model_path=args.model, num_threads=args.threads or 4)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    _, mh, mw, _ = inp["shape"]

    def infer(x_hwc):
        it.set_tensor(inp["index"], x_hwc[None])         # NHWC
        it.invoke()
        return it.get_tensor(out["index"])[0]
    return infer, int(mw), int(mh), f"tflite {args.model}"


def _med(ts):
    return float(np.median(ts)) if ts else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="checkout root with scripts/ + configs/")
    ap.add_argument("--model", default=None, help="dynamic-range-int8 .tflite (legacy 32-bit)")
    ap.add_argument("--onnx", default=None, help="int8 .onnx (onnxruntime, 64-bit)")
    ap.add_argument("--image", required=True, help="an undistorted fisheye frame (864x648)")
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = backend default (ONNX: use 2 on the Pi 4)")
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args(argv)
    if bool(args.model) == bool(args.onnx):
        ap.error("give exactly one of --model (tflite) or --onnx")

    sys.path.insert(0, args.repo)
    import cv2
    from scripts.utils.calibration import load_detection, load_intrinsics
    from scripts.utils.geometry import UP_LEVEL, load_extrinsics, up_from_horizon_line
    from scripts.utils.segmentation import (
        SegDetectParams, WATER, free_space_profile, horizon_from_water,
        seg_obstacle_detections,
    )

    intr = load_intrinsics()["fisheye"]
    det = load_detection()
    K = np.asarray(intr["K"], float)
    cx, pix_deg = float(intr["cx"]), float(intr["pix_deg_ratio"])
    height = float((load_extrinsics().get("camera_height_m", {}) or {}).get("fisheye", 0.27))
    fcfg = det["fusion"]
    edges = np.arange(fcfg["bin_min_deg"], fcfg["bin_max_deg"] + fcfg["bin_step_deg"],
                      fcfg["bin_step_deg"])
    max_range = float((det.get("range", {}) or {}).get("max_range_m", 15.0))
    seg_params = SegDetectParams()

    infer, mw, mh, label = _make_infer(args)

    img_full = cv2.imread(args.image)[:, :, ::-1]            # RGB, frame resolution
    fh, fw = img_full.shape[:2]
    print(f"{label}\nmodel {mw}x{mh}  frame {fw}x{fh}  threads "
          f"{args.threads or 'default'}  iters {args.iters}")

    stages = {k: [] for k in
              ["preprocess", "infer", "argmax+upsample", "horizon_from_water",
               "free_space_profile", "seg_obstacle_detections", "TOTAL"]}

    for _ in range(args.iters + 3):
        warm = _ < 3
        t0 = time.perf_counter()

        small = cv2.resize(img_full, (mw, mh))
        x = (small.astype(np.float32) / 255.0 - _MEAN) / _STD
        t1 = time.perf_counter()

        logits = infer(x)                                    # h',w',3 (HWC)
        t2 = time.perf_counter()

        cls = logits.argmax(-1).astype(np.uint8)
        seg = cv2.resize(cls, (fw, fh), interpolation=cv2.INTER_NEAREST)
        t3 = time.perf_counter()

        s, b, conf = horizon_from_water(seg)
        up = up_from_horizon_line(s, b, K) if conf > 0.3 else UP_LEVEL
        t4 = time.perf_counter()

        free_space_profile(seg, K, up, height, cx, pix_deg, edges, max_range_m=max_range)
        t5 = time.perf_counter()

        seg_obstacle_detections(seg, seg_params)
        t6 = time.perf_counter()

        if warm:
            continue
        stages["preprocess"].append((t1 - t0) * 1000)
        stages["infer"].append((t2 - t1) * 1000)
        stages["argmax+upsample"].append((t3 - t2) * 1000)
        stages["horizon_from_water"].append((t4 - t3) * 1000)
        stages["free_space_profile"].append((t5 - t4) * 1000)
        stages["seg_obstacle_detections"].append((t6 - t5) * 1000)
        stages["TOTAL"].append((t6 - t0) * 1000)

    print(f"\n{'stage':>26} {'median ms':>10}")
    for k in ["preprocess", "infer", "argmax+upsample", "horizon_from_water",
              "free_space_profile", "seg_obstacle_detections", "TOTAL"]:
        print(f"{k:>26} {_med(stages[k]):10.1f}")
    tot = _med(stages["TOTAL"])
    inf = _med(stages["infer"])
    print(f"\nend-to-end: {1000.0/tot:.2f} fps   (CNN alone {1000.0/inf:.2f} fps)")
    print(f"nav-layer overhead over CNN: {tot - inf:.1f} ms "
          f"({100.0*(tot-inf)/tot:.0f}% of frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
