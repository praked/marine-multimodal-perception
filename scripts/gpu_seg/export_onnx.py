"""Export the trained eWaSR model to ONNX (runs ON THE GPU NODE, in the venv).

Wraps the model so it takes a single image tensor [1,3,H,W] and feeds an internal
all-zeros imu_mask (the non-IMU model ignores it; see prepare.py), so the ONNX
graph has one clean `image` input. Spatial dims are dynamic, so the same .onnx
serves any input size the Pi benchmark sweeps.

    python export_onnx.py --ewasr-dir /scratch0/$USER/asvproject_seg/eWaSR \
        --weights /scratch0/$USER/asvproject_seg/runs/.../weights.pth \
        --model ewasr_resnet18 --out ewasr_resnet18_lars.onnx --size 512x384

NOTE on sizes: the decoder's pyramid pooling exports only when each pool grid
size is a *factor* of the input dims, so use sizes like 512x384 or 256x192;
e.g. 384x288 fails ("adaptive_avg_pool2d, output size not a factor of input").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _parse_size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


class _ImageOnly(nn.Module):
    """Adapt the eWaSR forward (dict in, dict out) to image-tensor in, logits out."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        b, _, h, w = image.shape
        imu = torch.zeros((b, h, w), dtype=image.dtype, device=image.device)
        return self.model({"image": image, "imu_mask": imu})["out"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ewasr-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="ewasr_resnet18")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=_parse_size, default=(512, 384))
    ap.add_argument("--opset", type=int, default=13)  # 13+ enables per-channel int8 quant
    ap.add_argument("--simplify", action="store_true")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.ewasr_dir).resolve()))
    import wasr.models as models
    from wasr.utils import load_weights

    model = models.get_model(args.model, num_classes=3, pretrained=False)
    model.load_state_dict(load_weights(args.weights))
    wrap = _ImageOnly(model).eval()

    tw, th = args.size
    dummy = torch.randn(1, 3, th, tw)
    # FIXED spatial dims: the eWaSR decoder's pyramid pooling uses adaptive
    # pooling with an input-derived output size, which ONNX cannot export with
    # dynamic H/W. One .onnx per resolution; the Pi benchmark reads the size
    # from each model. (Batch stays fixed at 1 too.)
    torch.onnx.export(
        wrap, dummy, args.out, opset_version=args.opset,
        input_names=["image"], output_names=["logits"],
    )
    print(f"exported -> {args.out}  (fixed {tw}x{th})")

    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
            m, ok = simplify(onnx.load(args.out))
            if ok:
                onnx.save(m, args.out)
                print("simplified OK")
        except Exception as e:  # noqa: BLE001
            print(f"simplify skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
