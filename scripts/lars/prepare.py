"""Stage LaRS into the MaSTr1325 layout eWaSR's dataloader expects.

eWaSR (datasets/mastr.py) reads, per stem:
    <image_dir>/<stem>.jpg          (RGB image)
    <mask_dir>/<stem>m.png          (label-encoded mask: 0/1/2, others ignored)
and selects the subset for a split from a plain text <stem>-per-line list.
Crucially it applies **no resize**; images are fed at native resolution and
batched, so a multi-resolution source like LaRS (2208x1242 and others) would
both break collation and blow past 16 GB. We therefore resize everything to a
fixed training size here (default 512x384, the WaSR/eWaSR convention; 4:3, the
same aspect as our 864x648 undistorted fisheye frames, so no anamorphic
distortion vs deployment).

Output tree (eWaSR config paths are relative to the config file's parent, so
the configs sit at the tree root):

    <out>/
      images/<stem>.jpg            resized (bilinear)
      masks/<stem>m.png            resized (nearest), LaRS 255 -> MaSTr 4
      train_images.txt             one stem per line
      val_images.txt
      mastr_train.yaml             image_dir/mask_dir/image_list for eWaSR
      mastr_val.yaml

Runs with PIL + numpy only (no torch/opencv), so it works on the laptop
(Python 3.14) and inside the eWaSR venv on the GPU node alike.

    python -m scripts.lars.prepare \
        --lars-root /Volumes/ROS2_SSD/LaRS_v1.0.0 \
        --out data/lars_mastr --size 512x384
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.lars import LARS_IGNORE, MASTR_IGNORE, OBSTACLE, SKY, WATER

DEFAULT_LARS_ROOT = Path("/Volumes/ROS2_SSD/LaRS_v1.0.0")
VALID_CLASS_IDS = (OBSTACLE, WATER, SKY)


def _parse_size(s: str) -> tuple[int, int]:
    """'512x384' -> (512, 384) as (width, height)."""
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            f"--size must look like WxH (e.g. 512x384), got {s!r}"
        ) from exc


def lars_paths(root: Path, split: str) -> tuple[Path, Path]:
    """Return (images_dir, semantic_masks_dir) for a LaRS split."""
    images_dir = root / "lars_v1.0.0_images" / split / "images"
    masks_dir = root / "lars_v1.0.0_annotations" / split / "semantic_masks"
    return images_dir, masks_dir


def list_stems(images_dir: Path) -> list[str]:
    """Sorted stems of the .jpg images in a split (deterministic)."""
    return sorted(p.stem for p in images_dir.glob("*.jpg"))


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """LaRS ignore (255) -> MaSTr ignore (4); leave 0/1/2 as-is.

    Any unexpected value is also folded into the ignore label so it never
    masquerades as a real class downstream.
    """
    out = mask.copy()
    valid = np.isin(out, VALID_CLASS_IDS)
    out[~valid] = MASTR_IGNORE
    return out


def _resize_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.convert("RGB").resize(size, Image.BILINEAR)


def _resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    # NEAREST so class ids are never interpolated into bogus intermediate values.
    return mask.resize(size, Image.NEAREST)


def prepare_split(
    root: Path,
    out: Path,
    split: str,
    size: tuple[int, int],
    limit: int | None = None,
) -> list[str]:
    """Stage one split. Returns the list of stems written."""
    images_dir, masks_dir = lars_paths(root, split)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"LaRS images dir not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"LaRS semantic_masks dir not found: {masks_dir}")

    out_images = out / "images"
    out_masks = out / "masks"
    out_imus = out / "imus"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_imus.mkdir(parents=True, exist_ok=True)
    # eWaSR's WaSR.forward reads x['imu_mask'] UNCONDITIONALLY (even for the
    # non-IMU ewasr_resnet18: the SIM block just ignores it when imu=False).
    # So we ship an all-zeros imu mask per stem; the dataloader needs the file
    # to exist and the model treats zeros as "no prior". See plan §issues.
    imu_zero = Image.fromarray(np.zeros((size[1], size[0]), dtype=np.uint8))

    stems = list_stems(images_dir)
    if limit is not None:
        stems = stems[:limit]

    written: list[str] = []
    missing: list[str] = []
    n = len(stems)
    for i, stem in enumerate(stems):
        img_path = images_dir / f"{stem}.jpg"
        mask_path = masks_dir / f"{stem}.png"
        if not mask_path.exists():
            missing.append(stem)
            continue

        _resize_image(Image.open(img_path), size).save(
            out_images / f"{stem}.jpg", quality=95
        )
        mask_arr = np.array(_resize_mask(Image.open(mask_path), size))
        remapped = remap_mask(mask_arr)
        Image.fromarray(remapped.astype(np.uint8)).save(out_masks / f"{stem}m.png")
        imu_zero.save(out_imus / f"{stem}.png")
        written.append(stem)

        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"  [{split}] {i + 1}/{n}", file=sys.stderr, flush=True)

    if missing:
        print(
            f"  [{split}] WARNING: {len(missing)} images had no mask, skipped "
            f"(e.g. {missing[:3]})",
            file=sys.stderr,
        )
    return written


def write_image_list(out: Path, split: str, stems: list[str]) -> Path:
    list_path = out / f"{split}_images.txt"
    list_path.write_text("\n".join(stems) + "\n")
    return list_path


def write_config(out: Path, split: str) -> Path:
    """Write the eWaSR dataset config yaml for a split (paths relative to root)."""
    cfg_path = out / f"mastr_{split}.yaml"
    cfg_path.write_text(
        "# Auto-generated by scripts/lars/prepare.py: LaRS staged as MaSTr1325.\n"
        "# Paths are relative to this file's directory (eWaSR convention).\n"
        "image_dir: images\n"
        "mask_dir: masks\n"
        "imu_dir: imus\n"
        f"image_list: {split}_images.txt\n"
    )
    return cfg_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lars-root", type=Path, default=DEFAULT_LARS_ROOT)
    ap.add_argument("--out", type=Path, required=True, help="staging tree root")
    ap.add_argument("--size", type=_parse_size, default=(512, 384), help="WxH, default 512x384")
    ap.add_argument("--splits", default="train,val", help="comma list, default train,val")
    ap.add_argument("--limit", type=int, default=None, help="cap stems per split (smoke test)")
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"LaRS root : {args.lars_root}")
    print(f"Out tree  : {args.out}")
    print(f"Size      : {args.size[0]}x{args.size[1]} (WxH)")
    print(f"Splits    : {splits}")

    for split in splits:
        print(f"Staging split: {split}")
        stems = prepare_split(args.lars_root, args.out, split, args.size, args.limit)
        write_image_list(args.out, split, stems)
        write_config(args.out, split)
        print(f"  -> {len(stems)} stems, list + config written")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
