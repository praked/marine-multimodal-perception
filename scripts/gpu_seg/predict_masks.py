"""Predict CLEAN label-encoded eWaSR masks (runs ON THE GPU NODE, in the venv).

eWaSR's stock predict.py writes a blended colour overlay (orig*0.7 + colour*0.3),
which is no good for geometry. This script instead saves the raw class map
(0=obstacle, 1=water, 2=sky) as an 8-bit grayscale PNG per input frame, at the
ORIGINAL frame resolution, exactly what scripts/utils/segmentation.py consumes
from data/seg/<clip>/<frame_id>.png.

Inference resizes each frame to the training size (default 512x384), runs the
fully-convolutional net, then NEAREST-upsamples the argmax back to the frame's
native size so mask pixels line up with the undistorted-image coordinates the
range geometry uses.

    python predict_masks.py \
        --ewasr-dir /scratch0/$USER/asvproject_seg/eWaSR \
        --image-dir /scratch0/$USER/asvproject_seg/pred_frames/Boats__2025-06-23_16-21-07 \
        --weights /scratch0/$USER/asvproject_seg/runs/.../weights.pth \
        --model ewasr_resnet18 \
        --out-dir /scratch0/$USER/asvproject_seg/pred_masks/Boats__2025-06-23_16-21-07 \
        --size 512x384
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

# Colours for the optional eyeball overlay (obstacle, water, sky): matches the
# eWaSR convention so our overlays look like the paper's.
SEG_COLORS = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], np.uint8)
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ewasr-dir", required=True, help="path to the cloned eWaSR repo")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="ewasr_resnet18")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=_parse_size, default=(512, 384), help="WxH train size")
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overlay", action="store_true", help="also write *_overlay.png")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.ewasr_dir).resolve()))
    import wasr.models as models  # noqa: E402
    from datasets.transforms import PytorchHubNormalization  # noqa: E402
    from wasr.utils import load_weights  # noqa: E402

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Build with the SAME structural defaults train.py/predict.py use
    # (mixer=CCCCSS, enricher=SS, project=False) so the state_dict loads cleanly.
    model = models.get_model(args.model, num_classes=args.num_classes, pretrained=False)
    model.load_state_dict(load_weights(args.weights))
    model = model.eval().to(device)

    normalise = PytorchHubNormalization()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if not img_paths:
        print(f"No images in {args.image_dir}", file=sys.stderr)
        return 1
    print(f"{len(img_paths)} frames -> {out_dir}")

    tw, th = args.size
    n = 0
    for start in range(0, len(img_paths), args.batch_size):
        chunk = img_paths[start:start + args.batch_size]
        imgs, orig_sizes = [], []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            orig_sizes.append(im.size)  # (W, H)
            im_r = im.resize((tw, th), Image.BILINEAR)
            imgs.append(normalise(np.array(im_r)))
        img_t = torch.stack(imgs).to(device)
        # eWaSR's forward reads x['imu_mask'] unconditionally; the non-IMU model
        # ignores its contents, so a zeros mask is correct (see prepare.py).
        batch = {"image": img_t,
                 "imu_mask": torch.zeros((img_t.size(0), th, tw), device=device)}
        with torch.no_grad():
            out = model(batch)["out"].cpu()
        # InterpolationMode, not a PIL int: torchvision deprecated int
        # interpolation args (warns on the pinned 0.15, removed later).
        out = TF.resize(out, (th, tw), interpolation=InterpolationMode.BILINEAR)
        cls = out.argmax(1).byte().numpy()  # [B, th, tw]

        for i, p in enumerate(chunk):
            w0, h0 = orig_sizes[i]
            m = Image.fromarray(cls[i]).resize((w0, h0), Image.NEAREST)
            m.save(out_dir / f"{p.stem}.png")
            if args.overlay:
                ov = (np.array(im_to_rgb(p)) * 0.6
                      + SEG_COLORS[np.array(m)] * 0.4).astype(np.uint8)
                Image.fromarray(ov).save(out_dir / f"{p.stem}_overlay.png")
            n += 1
        print(f"  {n}/{len(img_paths)}", file=sys.stderr, flush=True)

    print(f"Done. {n} masks written to {out_dir}")
    return 0


def im_to_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


if __name__ == "__main__":
    raise SystemExit(main())
