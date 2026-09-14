"""Per-clip metadata overrides applied at frame-read time.

Loads `configs/clip_overrides.yaml` and exposes a small API for applying
resize + rotation to fisheye / thermal frames as iterate_triplet reads
them. Intended only for one-off capture quirks (camera mounted rotated,
resolution mismatch with the calibrated intrinsics).

The longer-term answer for any clip listed here is to recapture or
recalibrate; the override is a workaround so the existing detection
pipeline still produces useful numbers in the meantime.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OVERRIDES_PATH = REPO_ROOT / "configs" / "clip_overrides.yaml"

_ROTATION_CODE = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def load_overrides(path: str | Path = DEFAULT_OVERRIDES_PATH) -> dict[str, dict]:
    """Return {clip_id: {sensor: {field: value}}}. Empty dict if file absent."""
    if not Path(path).exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("clips", {}) or {}


def get_for_clip(overrides: dict, clip_id: str) -> dict:
    """Return the override dict for one clip, or an empty dict."""
    return overrides.get(clip_id, {}) or {}


def apply_overrides(frame: np.ndarray | None, sensor_overrides: dict
                    ) -> np.ndarray | None:
    """Resize and/or rotate a frame according to per-sensor overrides.

    Returns the frame (possibly modified). No-op if `sensor_overrides`
    is empty/None or the frame is None.
    """
    if frame is None or not sensor_overrides:
        return frame

    target_size = sensor_overrides.get("target_image_size")
    if target_size:
        w, h = int(target_size[0]), int(target_size[1])
        if (frame.shape[1], frame.shape[0]) != (w, h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    rotation = int(sensor_overrides.get("rotation_deg", 0)) % 360
    if rotation in _ROTATION_CODE:
        frame = cv2.rotate(frame, _ROTATION_CODE[rotation])

    return frame
