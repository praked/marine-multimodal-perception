"""Run the eWaSR TFLite model on the Raspberry Pi and save a colour mask overlay.

TFLite sibling of pi_predict.py (the Pi is 32-bit armhf -> no onnxruntime wheel).
Pure tflite_runtime + numpy + Pillow. Confirms the deployed model produces a SANE
segmentation on-device, not just that it's fast.

    python pi_tflite_predict.py --tflite ewasr_lars_512x384_int8.tflite \
        --image frame.jpg --out frame_seg.png
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
    except ImportError:
        raise SystemExit("pip install --break-system-packages tflite-runtime numpy pillow")

SEG_COLORS = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], np.uint8)  # obs/water/sky
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="seg_overlay.png")
    ap.add_argument("--mask-out", default=None, help="optional raw class-id PNG")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)

    kw = {"model_path": args.tflite}
    if args.threads:
        kw["num_threads"] = args.threads
    it = Interpreter(**kw)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    _, h, w, _ = inp["shape"]
    w, h = int(w), int(h)

    im = Image.open(args.image).convert("RGB")
    ow, oh = im.size
    arr = (np.asarray(im.resize((w, h)), np.float32) / 255.0 - _MEAN) / _STD
    x = arr[None].astype(np.float32)                 # 1,H,W,3

    it.set_tensor(inp["index"], x); it.invoke()
    logits = it.get_tensor(out["index"])[0]          # h',w',3  (NHWC)
    cls = logits.argmax(-1).astype(np.uint8)
    frac = {c: float((cls == i).mean()) for i, c in enumerate(["obstacle", "water", "sky"])}
    print(f"{w}x{h}  class fractions: " + "  ".join(f"{k}={v:.2f}" for k, v in frac.items()))

    mask = Image.fromarray(cls).resize((ow, oh), Image.NEAREST)
    overlay = (np.asarray(im) * 0.6 + SEG_COLORS[np.asarray(mask)] * 0.4).astype(np.uint8)
    Image.fromarray(overlay).save(args.out)
    print(f"overlay -> {args.out}")
    if args.mask_out:
        mask.save(args.mask_out)
        print(f"mask -> {args.mask_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
