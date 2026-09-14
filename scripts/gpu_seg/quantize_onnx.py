"""Static INT8-quantize the eWaSR ONNX models + check accuracy (GPU node or any
box with onnxruntime). Pure onnxruntime + numpy + Pillow, no torch.

240 MB fp32 is too big for the Pi; static (QDQ) int8 quantization with real
InstitutionOne frames as calibration shrinks it ~4x and speeds CPU conv inference. We
then verify the quantized model still agrees with fp32 (per-pixel argmax) so we
don't ship a fast-but-broken model.

    python quantize_onnx.py --onnx ewasr_lars_512x384.onnx \
        --frames-root /scratch0/$USER/asvproject_seg/pred_frames \
        --out ewasr_lars_512x384.int8.onnx --calib 100 --check 30
"""

from __future__ import annotations

import argparse
import glob
import random

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _hw(onnx_path):
    s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    shp = s.get_inputs()[0].shape
    return s.get_inputs()[0].name, int(shp[3]), int(shp[2])   # name, W, H


def _preprocess(path, w, h):
    arr = np.asarray(Image.open(path).convert("RGB").resize((w, h)), np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


class _Calib(CalibrationDataReader):
    def __init__(self, frames, name, w, h):
        self.frames, self.name, self.w, self.h = frames, name, w, h
        self._it = iter(frames)

    def get_next(self):
        p = next(self._it, None)
        return None if p is None else {self.name: _preprocess(p, self.w, self.h)}

    def rewind(self):
        self._it = iter(self.frames)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", type=int, default=100)
    ap.add_argument("--check", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    # Per-channel weight quant is more accurate but emits DequantizeLinear(axis=),
    # which needs opset>=13; the original ONNX was exported at opset 12, so the
    # default stays per-tensor. The *.op13.onnx re-exports enable --per-channel.
    ap.add_argument("--per-channel", action="store_true")
    # Graph format + activation dtype knobs (2026-07-06 Pi recipe race):
    # QDQ keeps Q/DQ pairs for the optimizer to fuse; QOperator emits
    # QLinearConv directly. u8 activations pair with s8 weights (u8s8),
    # the layout some ARM64 MLAS kernels prefer. Defaults = the shipped
    # recipe (QDQ, s8s8).
    ap.add_argument("--format", choices=["qdq", "qoperator"], default="qdq")
    ap.add_argument("--act-type", choices=["int8", "uint8"], default="int8")
    ap.add_argument("--no-pre-process", action="store_true",
                    help="skip quant_pre_process (needed for some "
                         "dynamo-exported graphs it chokes on)")
    args = ap.parse_args(argv)

    name, w, h = _hw(args.onnx)
    frames = sorted(glob.glob(f"{args.frames_root}/*/*.jpg")
                    + glob.glob(f"{args.frames_root}/*/*.png"))
    if not frames:
        raise SystemExit(
            f"No calibration frames found under {args.frames_root} "
            f"(expected {args.frames_root}/*/*.jpg or *.png)")
    random.Random(args.seed).shuffle(frames)
    calib, check = frames[:args.calib], frames[args.calib:args.calib + args.check]
    print(f"{args.onnx}  {w}x{h}  calib={len(calib)} check={len(check)}")

    if args.no_pre_process:
        pre = args.onnx
    else:
        pre = args.out + ".pre.onnx"
        quant_pre_process(args.onnx, pre)
    fmt = QuantFormat.QDQ if args.format == "qdq" else QuantFormat.QOperator
    act = QuantType.QInt8 if args.act_type == "int8" else QuantType.QUInt8
    quantize_static(pre, args.out, _Calib(calib, name, w, h),
                    quant_format=fmt, per_channel=args.per_channel,
                    weight_type=QuantType.QInt8, activation_type=act)

    import os
    print(f"size: fp32 {os.path.getsize(args.onnx)//(1<<20)} MB -> "
          f"int8 {os.path.getsize(args.out)//(1<<20)} MB")

    # accuracy: per-pixel argmax agreement fp32 vs int8
    s32 = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    agree = []
    for p in check:
        x = _preprocess(p, w, h)
        a = s32.run(None, {name: x})[0][0].argmax(0)
        b = s8.run(None, {name: x})[0][0].argmax(0)
        agree.append(float((a == b).mean()))
    if agree:
        print(f"int8-vs-fp32 pixel agreement: mean {np.mean(agree)*100:.1f}%  "
              f"min {np.min(agree)*100:.1f}%  (over {len(agree)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
