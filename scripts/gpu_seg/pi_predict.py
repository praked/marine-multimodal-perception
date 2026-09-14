"""Run the eWaSR ONNX model on the Raspberry Pi and save a colour mask overlay.

Pure onnxruntime + numpy + Pillow (no torch): to confirm the deployed model
produces a SANE segmentation on-device, not just that it's fast. Pairs with
pi_benchmark.py (which only times).

    python pi_predict.py --onnx ewasr_lars_512x384.int8.onnx \
        --image frame.jpg --out frame_seg.png
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image

SEG_COLORS = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], np.uint8)  # obs/water/sky
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="seg_overlay.png")
    ap.add_argument("--mask-out", default=None, help="optional raw class-id PNG")
    args = ap.parse_args(argv)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    shp = sess.get_inputs()[0].shape
    w, h = int(shp[3]), int(shp[2])

    im = Image.open(args.image).convert("RGB")
    ow, oh = im.size
    arr = (np.asarray(im.resize((w, h)), np.float32) / 255.0 - _MEAN) / _STD
    x = arr.transpose(2, 0, 1)[None].astype(np.float32)

    logits = sess.run(None, {name: x})[0][0]           # [C, hh, ww]
    cls = logits.argmax(0).astype(np.uint8)
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
