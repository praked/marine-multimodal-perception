"""Typed-detection provider; read precomputed YOLO detections per frame_id.

Like scripts/utils/segmentation.SegProvider, but for the LaRS-trained typed
object detector (boat/swimmer/buoy/...). Because there is no local torch (the
laptop runs Python 3.14), YOLO inference happens on the GPU
(scripts/gpu_finetune/yolo_predict_frames.py) and writes one dashboard-schema
JSONL per clip:

    <det_root>/<scene>__<triplet_ts>.jsonl
    {"frame_id": "<scene>/<triplet_ts>/ts=<HH-MM-SS.f>",
     "fisheye_bboxes": [{"cls": "boat_ship", "xyxy": [x0,y0,x1,y1], "confidence": 0.83}]}

`xyxy` is NORMALISED to [0,1] in the undistorted-fisheye frame (matching
scripts/eval/yolo_infer.py), so the consumer scales by the image size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Stable display colours per LaRS thing class (RGB). Order matches
# scripts/lars/export_yolo_det.THING_CLASSES.
CLASS_COLOURS = {
    "boat_ship": (255, 64, 64),
    "row_boats": (255, 140, 0),
    "paddle_board": (255, 215, 0),
    "buoy": (0, 220, 120),
    "swimmer": (0, 200, 255),
    "animal": (180, 100, 255),
    "float": (255, 0, 200),
    "other": (180, 180, 180),
}
DEFAULT_COLOUR = (255, 255, 0)


def _clip_dirname(scene: str, triplet_ts: str) -> str:
    return f"{scene}__{triplet_ts}"


def det_file_for_frame_id(det_root: str | Path, frame_id: str) -> Path | None:
    """Map a frame_id to its per-clip detections JSONL on disk."""
    parts = frame_id.split("/")
    if len(parts) < 3:
        return None
    scene, triplet_ts = parts[0], parts[1]
    return Path(det_root) / f"{_clip_dirname(scene, triplet_ts)}.jsonl"


@dataclass
class DetProvider:
    """Lazy, cached reader of per-clip typed detections keyed by frame_id."""

    det_root: Path
    _clips: dict[str, dict[str, list]] = field(default_factory=dict)

    def __init__(self, det_root: str | Path):
        self.det_root = Path(det_root)
        self._clips = {}

    def available(self) -> bool:
        return self.det_root.is_dir()

    def _load_clip(self, path: Path) -> dict[str, list]:
        key = str(path)
        if key in self._clips:
            return self._clips[key]
        table: dict[str, list] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fid = rec.get("frame_id")
                if fid:
                    table[fid] = rec.get("fisheye_bboxes", [])
        self._clips[key] = table
        return table

    def get(self, frame_id: str | None) -> list | None:
        """Return the list of typed bboxes for a frame, or None if unavailable."""
        if not frame_id:
            return None
        path = det_file_for_frame_id(self.det_root, frame_id)
        if path is None:
            return None
        return self._load_clip(path).get(frame_id)


@dataclass
class InstanceSegProvider:
    """Per-clip typed INSTANCE-segmentation reader (YOLOv8-seg).

    Same on-disk layout as DetProvider but records carry `instances`, each with
    a normalised `polygon` (flat [x1,y1,x2,y2,...] in [0,1]) plus `cls`/`conf`:
        <root>/<scene>__<triplet_ts>.jsonl
        {"frame_id": "...", "instances": [{"cls": "boat_ship",
          "confidence": 0.8, "polygon": [x1,y1,...]}]}
    """

    root: Path
    _clips: dict[str, dict[str, list]] = field(default_factory=dict)

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._clips = {}

    def available(self) -> bool:
        return self.root.is_dir()

    def _load_clip(self, path: Path) -> dict[str, list]:
        key = str(path)
        if key in self._clips:
            return self._clips[key]
        table: dict[str, list] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fid = rec.get("frame_id")
                if fid:
                    table[fid] = rec.get("instances", [])
        self._clips[key] = table
        return table

    def get(self, frame_id: str | None) -> list | None:
        if not frame_id:
            return None
        path = det_file_for_frame_id(self.root, frame_id)
        if path is None:
            return None
        return self._load_clip(path).get(frame_id)
