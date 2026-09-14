"""Benchmark the eWaSR TFLite model on the Raspberry Pi 4 (CPU).

TFLite sibling of pi_benchmark.py: used because the deployment Pi runs a
32-bit (armhf) userland, for which onnxruntime ships no wheels; tflite-runtime
does. Pure tflite_runtime + numpy + Pillow (no torch, no tensorflow).

Each TFLite is fixed-size NHWC (the eWaSR decoder can't export dynamic spatial
dims), so pass one .tflite per resolution; size is read from each model.

On the Pi:
    pip install --break-system-packages tflite-runtime numpy pillow
    python pi_tflite_benchmark.py \
        --tflite ewasr_lars_512x384_float32.tflite,ewasr_lars_512x384_int8.tflite \
        --threads 4 --iters 30

Optionally pass --image <frame.jpg> to use a real frame instead of noise.

NOTE: these are conservative (32-bit) lower-bound numbers. The authoritative
deployment benchmark is onnxruntime on the 64-bit image (re-flash pre-deploy).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # full tensorflow also provides this
    try:
        from tensorflow.lite import Interpreter  # type: ignore
    except ImportError:
        raise SystemExit("pip install --break-system-packages tflite-runtime numpy pillow")

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _input_nhwc(w, h, image):
    if image:
        from PIL import Image
        arr = np.asarray(Image.open(image).convert("RGB").resize((w, h)), np.float32) / 255.0
    else:
        arr = np.random.rand(h, w, 3).astype(np.float32)
    arr = (arr - _MEAN) / _STD
    return arr[None].astype(np.float32)            # 1,H,W,3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite", required=True, help="comma-separated .tflite files (one per size)")
    ap.add_argument("--threads", type=int, default=0, help="0 = interpreter default")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--image", default=None)
    args = ap.parse_args(argv)

    print(f"threads={args.threads or 'default'}  iters={args.iters}")
    print(f"{'model':>40} {'size':>9} {'mean ms':>9} {'median ms':>10} {'fps':>7}")
    for path in args.tflite.split(","):
        path = path.strip()
        kw = {"model_path": path}
        if args.threads:
            kw["num_threads"] = args.threads
        it = Interpreter(**kw)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        out = it.get_output_details()[0]
        _, h, w, _ = inp["shape"]
        x = _input_nhwc(int(w), int(h), args.image)
        for _ in range(3):                          # warmup
            it.set_tensor(inp["index"], x); it.invoke()
        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            it.set_tensor(inp["index"], x); it.invoke()
            _ = it.get_tensor(out["index"])
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        med = float(np.median(ts))
        name = path.split("/")[-1][-40:]
        print(f"{name:>40} {int(w)}x{int(h):>5} {ts.mean():9.1f} {med:10.1f} {1000.0/med:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
