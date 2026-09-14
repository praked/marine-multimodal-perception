"""Pure-numpy readers for LaRS (sanity checks, previews, tests).

No torch/opencv: PIL + numpy only, so it runs on the laptop (Python 3.14).
Reads the LaRS-on-disk layout (not the staged MaSTr tree); for the staged tree
the masks are already label-encoded 0/1/2/4 and need no special handling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def read_semantic_mask(path: str | Path) -> np.ndarray:
    """Read a LaRS semantic mask as an HxW uint8 array (0/1/2, 255=ignore)."""
    return np.array(Image.open(path))


def class_fractions(mask: np.ndarray) -> dict[str, float]:
    """Fraction of valid (non-ignore) pixels per class: handy for sanity."""
    from scripts.lars import OBSTACLE, SKY, WATER

    valid = mask != 255
    total = int(valid.sum()) or 1
    return {
        "obstacle": float((mask == OBSTACLE).sum()) / total,
        "water": float((mask == WATER).sum()) / total,
        "sky": float((mask == SKY).sum()) / total,
    }


def read_image(path: str | Path) -> np.ndarray:
    """Read an RGB image as an HxWx3 uint8 array."""
    return np.array(Image.open(path).convert("RGB"))
