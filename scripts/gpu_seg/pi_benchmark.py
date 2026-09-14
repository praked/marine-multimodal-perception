"""Benchmark the eWaSR ONNX model on the Raspberry Pi 4 (CPU).

Pure onnxruntime + numpy + Pillow (NO torch) so it runs on the Pi.
Sweeps input sizes and thread counts, reports latency + fps, to decide whether
segmentation can stream on-device (and at what resolution / cadence).

Each ONNX is fixed-size (the eWaSR decoder can't export dynamic spatial dims),
so pass one .onnx per resolution; the size is read from each model.

On the Pi:
    pip install onnxruntime numpy pillow
    python pi_benchmark.py \
        --onnx ewasr_lars_512x384.onnx,ewasr_lars_256x192.onnx \
        --threads 4 --iters 30

Optionally pass --image <frame.jpg> to time real preprocessing too.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise SystemExit("pip install onnxruntime numpy pillow  (run this on the Pi)")

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _model_hw(sess, fallback):
    """(W, H) from the model's fixed input shape, else `fallback`."""
    shp = sess.get_inputs()[0].shape          # [N, C, H, W]
    h, w = shp[2], shp[3]
    if isinstance(h, int) and isinstance(w, int):
        return w, h
    return fallback


def _input(size, image):
    tw, th = size
    if image:
        from PIL import Image
        im = Image.open(image).convert("RGB").resize((tw, th))
        arr = np.asarray(im, np.float32) / 255.0
    else:
        arr = np.random.rand(th, tw, 3).astype(np.float32)
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="comma-separated .onnx files (one per size)")
    ap.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--image", default=None)
    ap.add_argument("--fallback-size", default="512x384",
                    help="used only if a model's input shape is dynamic")
    args = ap.parse_args(argv)

    fw, fh = (int(v) for v in args.fallback_size.lower().split("x"))
    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1

    print(f"threads={args.threads or 'default'}  iters={args.iters}")
    print(f"{'model':>34} {'size':>9} {'mean ms':>9} {'median ms':>10} {'fps':>7}")
    for path in args.onnx.split(","):
        path = path.strip()
        sess = ort.InferenceSession(path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        size = _model_hw(sess, (fw, fh))
        x = _input(size, args.image)
        for _ in range(3):                       # warmup
            sess.run(None, {iname: x})
        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            sess.run(None, {iname: x})
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        med = float(np.median(ts))
        name = path.split("/")[-1][-34:]
        print(f"{name:>34} {size[0]}x{size[1]:>5} {ts.mean():9.1f} {med:10.1f} {1000.0/med:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
