"""Distil a fast student segmenter for the Pi 4 (GPU node; CPU-smoke-testable).

Why: the Pi-4 int8 floor for eWaSR-ResNet18 is ~3.9 µs/pixel; quantisation is
exhausted (docs/history/2026-07-06_pi_quantisation.md). Real-time needs a smaller
model: this trains torchvision's LRASPP-MobileNetV3-Large (~3.2 M params,
designed for fast semantic seg) on the same 3-class water/sky/obstacle task,
using (a) LaRS ground truth and (b) the 10,731 InstitutionOne frames labelled by the
eWaSR teacher's masks (hard-label distillation: the teacher masks ARE the
training target on our domain).

Data comes as image/mask directory pairs, matched by filename stem
(recursively), so both sources plug in unchanged:
  --pairs <staged_lars>/images:<staged_lars>/masks     # LaRS (MaSTr tree, prepare.py)
  --pairs data/seg_frames:data/seg                     # InstitutionOne teacher masks
Mask pixels: 0=obstacle, 1=water, 2=sky, 4=ignore (MaSTr convention).

After training it exports int8-ready ONNX at the requested sizes (the same
torch dynamo path as export_onnx.py, so arbitrary sizes work); feed those to
quantize_onnx.py + pi_benchmark.py, the 30-minute Pi-evaluation loop.

GPU node:
    python train_student.py --pairs .../lars/images:.../lars/masks \
        --pairs .../institutionone/frames:.../institutionone/masks \
        --out runs/student --epochs 40 --batch 24 --size 512x384 \
        --export-sizes 512x384,640x480,864x648

Laptop CPU smoke test (validates the whole loop on a handful of images):
    python train_student.py --pairs data/seg_frames:data/seg --out /tmp/stu \
        --epochs 1 --batch 2 --size 256x192 --limit 8 --device cpu \
        --export-sizes 256x192
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

IGNORE_INDEX = 4          # MaSTr convention (prepare.py remaps LaRS 255 -> 4)
NUM_CLASSES = 3           # 0=obstacle, 1=water, 2=sky
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _index_pairs(img_root: Path, mask_root: Path) -> list[tuple[Path, Path]]:
    """Match images to masks by stem, searching both trees recursively.
    InstitutionOne layouts nest per clip; stems are unique per clip, so we key on
    <parent-dir-name>/<stem> first and fall back to bare stem."""
    def key(p: Path):
        return f"{p.parent.name}/{p.stem}"
    masks = {}
    for m in mask_root.rglob("*.png"):
        masks.setdefault(key(m), m)
        masks.setdefault(m.stem, m)
        # MaSTr/LaRS convention: mask "xxxm.png" belongs to image "xxx.jpg".
        if m.stem.endswith("m"):
            masks.setdefault(m.stem[:-1], m)
    out = []
    for img in sorted(img_root.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS or not img.is_file():
            continue
        m = masks.get(key(img)) or masks.get(img.stem)
        if m is not None and m != img:
            out.append((img, m))
    return out


class PairDataset(Dataset):
    def __init__(self, pairs, size_wh, train: bool, aug: dict | None = None,
                 gray: bool = False):
        self.pairs = pairs
        self.w, self.h = size_wh
        self.train = train
        # Optional extra augmentations (all default-off; added 2026-07-10 for
        # the thermal student's layout-memorization problem:
        # docs/history/2026-07-09_thermal_tuning.md "thermal-student v0"):
        #   polarity_p: probability of full gray inversion (255 - img).
        #                  Simulates the day/night thermal polarity crossover
        #                  (water cooler than scene by day, warmer at night).
        #   vshift_px: random vertical shift in [-N, +N] px, image edge-
        #                  replicated, mask filled with IGNORE (fabricated
        #                  rows teach nothing): breaks the fixed sky/strip/
        #                  water row-layout prior.
        #   contrast: random contrast scale in [1-c, 1+c] about the
        #                  frame mean + brightness offset (simulates the
        #                  Lepton AGC's scene-dependent stretch).
        self.aug = aug or {}
        # gray: collapse RGB to luminance replicated x3: trains an
        # RGB-shaped net on colourless input (LaRS-as-structure pretraining
        # for the thermal student, 2026-07-10).
        self.gray = gray

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        import cv2
        img_p, mask_p = self.pairs[i]
        img = cv2.imread(str(img_p))[:, :, ::-1]                    # RGB
        if self.gray:
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack([g, g, g], axis=-1)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.w, self.h))
        mask = cv2.resize(mask, (self.w, self.h),
                          interpolation=cv2.INTER_NEAREST)
        if self.train and random.random() < 0.5:                    # h-flip
            img, mask = img[:, ::-1], mask[:, ::-1]
        if self.train:
            if random.random() < float(self.aug.get("polarity_p", 0.0)):
                img = 255 - img
            c = float(self.aug.get("contrast", 0.0))
            if c > 0:
                f = 1.0 + random.uniform(-c, c)
                b = random.uniform(-25.0, 25.0) * c
                m = float(img.mean())
                img = np.clip(m + (img.astype(np.float32) - m) * f + b,
                              0, 255).astype(np.uint8)
            vs = int(self.aug.get("vshift_px", 0))
            if vs > 0:
                dy = random.randint(-vs, vs)
                if dy != 0:
                    img2 = np.empty_like(img)
                    mask2 = np.full_like(mask, IGNORE_INDEX)
                    if dy > 0:      # scene moves down
                        img2[dy:] = img[:-dy]
                        img2[:dy] = img[0]
                        mask2[dy:] = mask[:-dy]
                    else:
                        img2[:dy] = img[-dy:]
                        img2[dy:] = img[-1]
                        mask2[:dy] = mask[-dy:]
                    img, mask = img2, mask2
        x = ((img.astype(np.float32) / 255.0 - _MEAN) / _STD)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(mask)).long()
        y[y >= NUM_CLASSES] = IGNORE_INDEX                          # safety
        return x, y


def build_student(pretrained: bool = True) -> nn.Module:
    from torchvision.models.segmentation import (
        LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large)
    weights = (LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
    model = lraspp_mobilenet_v3_large(weights=weights)
    # Swap both classifier convs to 3 classes.
    low = model.classifier.low_classifier
    high = model.classifier.high_classifier
    model.classifier.low_classifier = nn.Conv2d(low.in_channels, NUM_CLASSES, 1)
    model.classifier.high_classifier = nn.Conv2d(high.in_channels, NUM_CLASSES, 1)
    return model


class _LogitsOnly(nn.Module):
    """Unwrap the torchvision dict output for a clean single-output ONNX."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def export_onnx(model, size_wh, out_path: Path) -> None:
    w, h = size_wh
    wrapped = _LogitsOnly(model).eval().cpu()
    dummy = torch.zeros(1, 3, h, w)
    torch.onnx.export(wrapped, (dummy,), str(out_path),
                      input_names=["image"], output_names=["logits"],
                      opset_version=17)
    print(f"exported -> {out_path}  (fixed {w}x{h})")


def evaluate(model, loader, device) -> tuple[float, list[float]]:
    """(mean pixel acc over valid px, per-class IoU)."""
    model.eval()
    inter = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)["out"].argmax(1)
            valid = y != IGNORE_INDEX
            correct += int((pred[valid] == y[valid]).sum())
            total += int(valid.sum())
            for c in range(NUM_CLASSES):
                pc, yc = (pred == c) & valid, (y == c) & valid
                inter[c] += int((pc & yc).sum())
                union[c] += int((pc | yc).sum())
    ious = [float(inter[c] / union[c]) if union[c] else float("nan")
            for c in range(NUM_CLASSES)]
    return (correct / max(total, 1)), ious


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", action="append", required=True,
                    metavar="IMG_DIR:MASK_DIR",
                    help="image/mask directory pair, repeatable")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--size", type=_parse_size, default=(512, 384))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total pairs (smoke tests)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip ImageNet backbone weights (offline smoke tests)")
    ap.add_argument("--export-sizes", default="512x384",
                    help="comma list of WxH ONNX exports of the best model")
    # --- optional extras (2026-07-10, thermal student; all default-off) ---
    ap.add_argument("--holdout-pairs", action="append", default=[],
                    metavar="IMG_DIR:MASK_DIR",
                    help="clip-level holdout pair(s): NEVER trained on, "
                         "evaluated with the best model after training "
                         "(the honest generalization number: random "
                         "val-frac splits are temporally correlated)")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch/numpy/random seed (A/B reproducibility)")
    ap.add_argument("--aug-polarity", type=float, default=0.0,
                    help="P(gray inversion): thermal day/night polarity")
    ap.add_argument("--aug-vshift", type=int, default=0,
                    help="max random vertical shift px (mask fill = ignore)")
    ap.add_argument("--aug-contrast", type=float, default=0.0,
                    help="random contrast scale +-c about the frame mean")
    ap.add_argument("--gray", action="store_true",
                    help="collapse inputs to grayscale x3 (thermal-style "
                         "colourless input; applies to train/val/holdout)")
    ap.add_argument("--init-weights", default=None,
                    help="warm-start from a prior student_best.pth "
                         "(e.g. grayscale-LaRS pretraining)")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path]] = []
    for spec in args.pairs:
        img_d, mask_d = spec.split(":")
        found = _index_pairs(Path(img_d), Path(mask_d))
        print(f"{spec}: {len(found)} pairs")
        pairs.extend(found)
    if not pairs:
        raise SystemExit("no image/mask pairs found")
    random.Random(0).shuffle(pairs)
    if args.limit:
        pairs = pairs[:args.limit]
    n_val = max(1, int(len(pairs) * args.val_frac))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"train {len(train_pairs)}  val {n_val}  size {args.size[0]}x{args.size[1]}  "
          f"device {device}")

    aug = {"polarity_p": args.aug_polarity, "vshift_px": args.aug_vshift,
           "contrast": args.aug_contrast}
    train_ds = PairDataset(train_pairs, args.size, train=True, aug=aug,
                           gray=args.gray)
    val_ds = PairDataset(val_pairs, args.size, train=False, gray=args.gray)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=(device == "cuda"),
                          drop_last=len(train_ds) > args.batch)
    val_ld = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)

    model = build_student(pretrained=not args.no_pretrained)
    if args.init_weights:
        state = torch.load(args.init_weights, map_location="cpu")
        model.load_state_dict(state)
        print(f"warm-started from {args.init_weights}")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_acc = -1.0
    best_path = out_dir / "student_best.pth"
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for bi, (x, y) in enumerate(train_ld):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = loss_fn(model(x)["out"], y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss)
        sched.step()
        acc, ious = evaluate(model, val_ld, device)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            marker = "  <- best"
        print(f"epoch {epoch + 1:3d}/{args.epochs}  "
              f"loss {running / max(len(train_ld), 1):.4f}  "
              f"val acc {acc:.4f}  IoU obst/water/sky "
              f"{ious[0]:.3f}/{ious[1]:.3f}/{ious[2]:.3f}  "
              f"{time.time() - t0:5.1f}s{marker}", flush=True)

    print(f"best val acc {best_acc:.4f} -> {best_path}")
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model = model.to(device)

    # Clip-level holdout: the honest generalization number (random val-frac
    # frames are temporally correlated with training frames).
    for spec in args.holdout_pairs:
        img_d, mask_d = spec.split(":")
        ho_pairs = _index_pairs(Path(img_d), Path(mask_d))
        if not ho_pairs:
            print(f"holdout {spec}: no pairs found")
            continue
        ho_ld = DataLoader(PairDataset(ho_pairs, args.size, train=False,
                                       gray=args.gray),
                           batch_size=args.batch, num_workers=args.workers)
        acc, ious = evaluate(model, ho_ld, device)
        print(f"HOLDOUT {spec}: n={len(ho_pairs)}  acc {acc:.4f}  "
              f"IoU obst/water/sky {ious[0]:.3f}/{ious[1]:.3f}/{ious[2]:.3f}",
              flush=True)

    for s in args.export_sizes.split(","):
        w, h = _parse_size(s.strip())
        export_onnx(model, (w, h), out_dir / f"student_{w}x{h}.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
