"""Loaders for YAML-backed configs (intrinsics, detection, sweep)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRINSICS = REPO_ROOT / "configs" / "intrinsics.yaml"
DEFAULT_DETECTION = REPO_ROOT / "configs" / "detection.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_intrinsics(path: str | Path = DEFAULT_INTRINSICS) -> dict[str, Any]:
    """Load intrinsics.yaml and convert K/D to numpy arrays in-place."""
    cfg = load_yaml(path)
    for sensor in ("fisheye", "thermal"):
        if sensor not in cfg:
            continue
        cfg[sensor]["K"] = np.array(cfg[sensor]["K"], dtype=np.float64)
        cfg[sensor]["D"] = np.array(cfg[sensor]["D"], dtype=np.float64)
    return cfg


def load_detection(path: str | Path = DEFAULT_DETECTION) -> dict[str, Any]:
    return load_yaml(path)


def get_dotted(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Resolve 'fisheye.blob.minArea' -> cfg['fisheye']['blob']['minArea']."""
    cur: Any = cfg
    for part in dotted_key.split("."):
        cur = cur[part]
    return cur


def set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


if __name__ == "__main__":
    intr = load_intrinsics()
    det = load_detection()
    print("fisheye K shape:", intr["fisheye"]["K"].shape)
    print("thermal D shape:", intr["thermal"]["D"].shape)
    print("fusion bins:", det["fusion"]["bin_min_deg"], "..", det["fusion"]["bin_max_deg"], "step", det["fusion"]["bin_step_deg"])
