"""Debug dashboard: pick any triplet from a dropdown, play it as video,
inspect every signal that drives fusion in real time.

Phase I.4.2b of PLAN.md. Built on top of `viewer.py`'s rendering, with
Tk chrome (Comboboxes, sidebar tables, toggles) so we can hop between
clips and watch which sensors fire in which bins without restarting.

Usage:
    python -m scripts.eval.dashboard
    python -m scripts.eval.dashboard --triplet data/Boats/2025-06-23_16-21-07

Keys (parity with viewer.py):
    left / right            step one frame
    shift+left / shift+right  jump -5s / +5s
    space                   play / pause
    s                       screenshot (outside the Labelling tab) / set the
                            structure class (inside the Labelling tab)
    r                       hot-reload detection.yaml (rebuilds pipeline)
    q                       quit

The "Export video…" button writes the main figure: current layout,
overlay toggles and threshold, exactly as on screen: to an mp4 under
results/exports/, at the transport's playback speed (or a custom
multiplier; 1x = real-time = 3 fps of footage).

Labelling-tab keys (parity with label_tool.py, only active while the
Labelling tab is selected and focus is not on a text-entry widget):
    b / d / B / p / m / o   set active class (and reclass selected bbox
                            if one is selected) to boat / duck / buoy /
                            person / structure / other
    s                       also sets structure (alias for m)
    u                       delete the most recent bbox on this frame
    cmd+z / ctrl+z          undo the last labelling action on this frame
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

# Tk is stdlib but Homebrew ships Python without it by default: keep the
# rest of the module importable so rendering helpers can be unit-tested.
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _TK_AVAILABLE = True
    _TK_ERROR: Optional[str] = None
except ImportError as _err:
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    simpledialog = None  # type: ignore[assignment]
    FigureCanvasTkAgg = None  # type: ignore[assignment]
    _TK_AVAILABLE = False
    _TK_ERROR = str(_err)

from scripts.sensor_processing.heading import (  # noqa: E402
    HeadingRecommendation,
    HeadingSmoother,
    recommend_heading,
)
from scripts.sensor_processing.pipeline import (  # noqa: E402
    FrameResult,
    ObstacleDetectionPipeline,
    angle_to_bin,
    iterate_triplet,
    make_bins,
)
from scripts.sensor_processing.tracker import SectorTracker  # noqa: E402
from scripts.utils.calibration import (  # noqa: E402
    DEFAULT_DETECTION,
    load_detection,
    load_intrinsics,
)
from scripts.utils.clip_overrides import get_for_clip, load_overrides  # noqa: E402
from scripts.utils.datasets import (  # noqa: E402
    Triplet,
    load_mmwave_csv,
    resolve_triplet,
)
from scripts.utils.geometry import (  # noqa: E402
    load_extrinsics,
    project_radar_to_fisheye,
    project_radar_to_thermal,
    range_from_water_plane_up,
)
from scripts.utils.segmentation import FreeSpaceSmoother, SegProvider  # noqa: E402
from scripts.utils.detections import DetProvider, InstanceSegProvider  # noqa: E402
# Share the class taxonomy with the Qwen labeller + label_tool so all three
# labelling paths agree (incl. the `structure` class). See
# configs/qwen_label_prompt.yaml.
from scripts.eval.qwen_batch_labeler import PROMPT_CFG  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCREEN_DIR = REPO_ROOT / "results" / "screens"
EXPORT_DIR = REPO_ROOT / "results" / "exports"

SPEEDS = (("0.5x", 666), ("1x", 333), ("2x", 166), ("4x", 83))
# Video-export frame rate at 1x. The clips are captured at ~3 fps and the
# 1x transport tick is 333 ms, so 3.0 keeps "1x" exports real-time.
EXPORT_BASE_FPS = 3.0
SENSOR_COLOURS = {0: "#888", 1: "#f6b26b", 2: "#f1c232", 3: "#6aa84f"}

# --- Customisable main-figure layout -----------------------------------------
#
# The main figure is a fixed LAYOUT_ROWS×LAYOUT_COLS gridspec; each panel
# (fisheye, thermal, radar, fusion bars, heading cost curve, heading history)
# claims a (row, col, rowspan, colspan, visible) slot. Users can swap between
# built-in presets via the Layout combobox or fine-tune cell-by-cell via the
# Customize... dialog. Custom presets persist to LAYOUTS_FILE.

LAYOUT_ROWS = 3
LAYOUT_COLS = 2

PANEL_KEYS: tuple[str, ...] = (
    "fisheye", "thermal", "radar", "bars", "heading_curve", "heading_hist",
)
PANEL_LABELS: dict[str, str] = {
    "fisheye": "Fisheye RGB",
    "thermal": "Thermal",
    "radar": "Radar",
    "bars": "Fusion bars",
    "heading_curve": "Heading cost",
    "heading_hist": "Heading history",
}
# Single-letter glyphs used by the dialog's ASCII layout preview.
PANEL_GLYPHS: dict[str, str] = {
    "fisheye": "F",
    "thermal": "T",
    "radar": "R",
    "bars": "B",
    "heading_curve": "H",
    "heading_hist": "h",
}
# Pastel fills for each panel in the visual grid picker: distinct enough
# to identify on a small canvas while still letting black text read on top.
PANEL_COLORS: dict[str, str] = {
    "fisheye": "#fcd5b4",
    "thermal": "#fff2cc",
    "radar": "#d9ead3",
    "bars": "#cfe2f3",
    "heading_curve": "#d9d2e9",
    "heading_hist": "#f4cccc",
}

DEFAULT_LAYOUT: dict[str, dict] = {
    "fisheye":       {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": True},
    "thermal":       {"row": 0, "col": 1, "rowspan": 1, "colspan": 1, "visible": True},
    "radar":         {"row": 1, "col": 0, "rowspan": 1, "colspan": 1, "visible": True},
    "bars":          {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "visible": True},
    "heading_curve": {"row": 2, "col": 0, "rowspan": 1, "colspan": 1, "visible": True},
    "heading_hist":  {"row": 2, "col": 1, "rowspan": 1, "colspan": 1, "visible": True},
}

BUILTIN_PRESETS: dict[str, dict[str, dict]] = {
    "Default 2×2 + heading": DEFAULT_LAYOUT,
    "Fisheye focus": {
        "fisheye":       {"row": 0, "col": 0, "rowspan": 2, "colspan": 2, "visible": True},
        "thermal":       {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
        "radar":         {"row": 2, "col": 0, "rowspan": 1, "colspan": 1, "visible": True},
        "bars":          {"row": 2, "col": 1, "rowspan": 1, "colspan": 1, "visible": True},
        "heading_curve": {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
        "heading_hist":  {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
    },
    "Thermal focus": {
        "fisheye":       {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
        "thermal":       {"row": 0, "col": 0, "rowspan": 2, "colspan": 2, "visible": True},
        "radar":         {"row": 2, "col": 0, "rowspan": 1, "colspan": 1, "visible": True},
        "bars":          {"row": 2, "col": 1, "rowspan": 1, "colspan": 1, "visible": True},
        "heading_curve": {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
        "heading_hist":  {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
    },
    "Radar focus": {
        "fisheye":       {"row": 0, "col": 1, "rowspan": 1, "colspan": 1, "visible": True},
        "thermal":       {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "visible": True},
        "radar":         {"row": 0, "col": 0, "rowspan": 2, "colspan": 1, "visible": True},
        "bars":          {"row": 2, "col": 0, "rowspan": 1, "colspan": 2, "visible": True},
        "heading_curve": {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
        "heading_hist":  {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
    },
    "Cameras only": {
        "fisheye":       {"row": 0, "col": 0, "rowspan": 3, "colspan": 1, "visible": True},
        "thermal":       {"row": 0, "col": 1, "rowspan": 3, "colspan": 1, "visible": True},
        "radar":         {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
        "bars":          {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
        "heading_curve": {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
        "heading_hist":  {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
    },
    "Heading focus": {
        "fisheye":       {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": True},
        "thermal":       {"row": 0, "col": 1, "rowspan": 1, "colspan": 1, "visible": True},
        "radar":         {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
        "bars":          {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "visible": False},
        "heading_curve": {"row": 1, "col": 0, "rowspan": 2, "colspan": 1, "visible": True},
        "heading_hist":  {"row": 1, "col": 1, "rowspan": 2, "colspan": 1, "visible": True},
    },
}

CUSTOM_LABEL = "(custom)"

LAYOUTS_FILE = Path.home() / ".config" / "asvproject-dashboard" / "layouts.json"

# --- Labelling -----------------------------------------------------------------
#
# Labels are written into labels/manual.jsonl, one record per frame, with the
# schema below. The dashboard uses a timestamp-based frame_id so dashboard
# labels never collide with the cli `label_tool.py` scheme (which is video-
# frame-index based). When you revisit a frame the LabelStore looks the record
# up and the labelling tab re-displays its bboxes + horizon line.

REPO_ROOT_FOR_LABELS = Path(__file__).resolve().parents[2]
LABELS_DIR = REPO_ROOT_FOR_LABELS / "labels"
DASHBOARD_LABELS_PATH = LABELS_DIR / "manual.jsonl"

# Class palette: BGR triples for OpenCV overlays don't apply here (matplotlib),
# so we keep hex strings that read on both light and dark frames.
# Taxonomy comes from the shared Qwen prompt config so the dashboard,
# label_tool, and the Qwen labeller never drift.
LABEL_CLASSES: tuple[str, ...] = tuple(PROMPT_CFG.get(
    "classes", ("boat", "duck", "buoy", "person", "structure", "other")))
CLASS_COLORS: dict[str, str] = {
    "boat":      "#34a853",  # green
    "duck":      "#fbbc05",  # amber
    "buoy":      "#ea4335",  # red
    "person":    "#4285f4",  # blue
    "structure": "#ff6d00",  # deep orange: mooring posts / piers / platforms
    "other":     "#9c27b0",  # purple
    "unset":     "#9e9e9e",  # grey: drawn but not yet classed
}
HORIZON_COLOR = "#00bcd4"  # cyan: distinct from every class colour above

LABEL_TOOL_NONE = "none"
LABEL_TOOL_BBOX = "bbox"
LABEL_TOOL_HORIZON = "horizon"


def _frame_id_for(scene: str, triplet_ts: str, frame_ts: str) -> str:
    """Build a stable frame identifier from the radar timestamp.

    Colons in the radar timestamp would clash with frame_id parsing, so we
    swap them for dashes. The `ts=` prefix flags the new (timestamp-based)
    scheme so it never collides with `label_tool.py`'s video-frame-index
    IDs in the same labels/manual.jsonl file.
    """
    safe_ts = frame_ts.replace(":", "-")
    return f"{scene}/{triplet_ts}/ts={safe_ts}"


class LabelStore:
    """Tiny load/save layer over labels/manual.jsonl with dedup by frame_id.

    The dashboard wants to revisit a frame and *replace* the existing label
    rather than append duplicates, which is why we hold the whole file in
    memory keyed by frame_id and rewrite on save. The file is small enough
    (one record per labelled frame; clips have a few hundred frames) that
    full-file rewrites are negligible.
    """

    def __init__(self, path: Optional[Path] = None):
        # Resolve the path at call time so tests can monkey-patch the
        # module-level DASHBOARD_LABELS_PATH before constructing.
        self.path = path if path is not None else DASHBOARD_LABELS_PATH
        self.records: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fid = obj.get("frame_id")
                    if isinstance(fid, str):
                        # Last record wins: matches the existing
                        # label_tool.py convention of append-only writes,
                        # so reopening after a partial session reflects
                        # the most recent state per frame_id.
                        self.records[fid] = obj
        except OSError:
            pass

    def write_all(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp sibling first so an interrupted save can't leave
        # us with a half-written jsonl.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w") as f:
            for fid in sorted(self.records.keys()):
                f.write(json.dumps(self.records[fid]) + "\n")
        tmp.replace(self.path)

    def get(self, frame_id: str) -> Optional[dict]:
        return self.records.get(frame_id)

    def put(self, record: dict) -> None:
        fid = record.get("frame_id")
        if not isinstance(fid, str):
            raise ValueError("record missing frame_id")
        # Drop empty records on save: an empty bbox list AND no horizon
        # line means the frame has been visited but isn't actually
        # labelled, so we don't want to clutter manual.jsonl with it.
        bboxes = record.get("fisheye_bboxes") or []
        horizon = record.get("horizon_line_label")
        if not bboxes and not horizon:
            self.records.pop(fid, None)
        else:
            self.records[fid] = record
        self.write_all()

    def frame_ids_for_triplet(self, scene: str, triplet_ts: str) -> set[str]:
        """All labelled frame IDs that belong to a given triplet: used by
        the prev/next-labelled-frame nav buttons."""
        prefix = f"{scene}/{triplet_ts}/"
        return {fid for fid in self.records if fid.startswith(prefix)}


def _clamp_placement(p: dict) -> dict:
    """Clamp a (row, col, rowspan, colspan, visible) tuple to the fixed grid."""
    try:
        r = max(0, min(int(p.get("row", 0)), LAYOUT_ROWS - 1))
        c = max(0, min(int(p.get("col", 0)), LAYOUT_COLS - 1))
        rs = max(1, min(int(p.get("rowspan", 1)), LAYOUT_ROWS - r))
        cs = max(1, min(int(p.get("colspan", 1)), LAYOUT_COLS - c))
    except (TypeError, ValueError):
        r, c, rs, cs = 0, 0, 1, 1
    return {"row": r, "col": c, "rowspan": rs, "colspan": cs,
            "visible": bool(p.get("visible", True))}


def _normalize_layout(layout: dict) -> dict[str, dict]:
    """Fill in any missing panels with defaults and clamp every placement."""
    out: dict[str, dict] = {}
    for key in PANEL_KEYS:
        out[key] = _clamp_placement(layout.get(key, DEFAULT_LAYOUT[key]))
    return out


def _layouts_equal(a: dict, b: dict) -> bool:
    a = _normalize_layout(a)
    b = _normalize_layout(b)
    for key in PANEL_KEYS:
        if a[key] != b[key]:
            return False
    return True


def _load_layouts_file() -> dict:
    """Return {'current': layout|None, 'presets': {name: layout}}."""
    empty = {"current": None, "presets": {}}
    if not LAYOUTS_FILE.exists():
        return empty
    try:
        data = json.loads(LAYOUTS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(data, dict):
        return empty
    cur_raw = data.get("current")
    cur = _normalize_layout(cur_raw) if isinstance(cur_raw, dict) else None
    presets_raw = data.get("presets") or {}
    presets = {
        name: _normalize_layout(layout)
        for name, layout in presets_raw.items()
        if isinstance(name, str) and isinstance(layout, dict)
    }
    return {"current": cur, "presets": presets}


def _save_layouts_file(current: Optional[dict], presets: dict) -> None:
    LAYOUTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"current": current, "presets": presets}
    LAYOUTS_FILE.write_text(json.dumps(payload, indent=2))


def _short_hash(path: Path) -> str:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()[:8]
    except OSError:
        return "unknown"


def _overlay_camera(sensor_result, *, projected=None, ranges=None,
                    show_horizon=True, show_projection=True,
                    show_detections=True, show_seg=False,
                    typed_dets=None, instance_dets=None) -> np.ndarray:
    """Same overlay routine as viewer.py / snapshot.py. Returns an RGB image."""
    if sensor_result is None:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    img = cv2.cvtColor(sensor_result.undistorted, cv2.COLOR_BGR2RGB).copy()
    h, w = img.shape[:2]

    # Water segmentation layer (fisheye only; drawn first so markers sit on top).
    # Blue tint = water, green line = the water-edge "horizon" used as the range
    # up-vector, green dots = the true waterline-contact point that replaces the
    # bbox bottom for monocular range. None of these appear when no mask is
    # present for the frame (the range path silently falls back to bbox bottom).
    seg = getattr(sensor_result, "seg_mask", None) if show_seg else None
    if seg is not None and seg.shape[:2] == img.shape[:2]:
        from scripts.utils.segmentation import (
            OBSTACLE,
            SKY,
            WATER,
            ContactParams,
            horizon_from_water,
            water_edge_contact,
        )
        # Tint ALL three classes (eWaSR palette, RGB): obstacle=amber,
        # water=blue, sky=purple: so the full segmentation is visible, not
        # just water. Obstacle is the "anything non-water/non-sky" class
        # (static + dynamic), semantic (no per-object instances).
        palette = {OBSTACLE: (247, 195, 37),
                   WATER: (41, 167, 224),
                   SKY: (90, 75, 164)}
        colour_map = np.zeros_like(img)
        tinted = np.zeros(seg.shape[:2], dtype=bool)
        for cls, colour in palette.items():
            m = seg == cls
            colour_map[m] = colour
            tinted |= m
        if tinted.any():
            img[tinted] = (0.6 * img[tinted]
                           + 0.4 * colour_map[tinted]).astype(np.uint8)
        s_slope, s_int, s_conf = horizon_from_water(seg)
        if s_conf > 0:
            y0 = int(np.clip(s_slope * 0 + s_int, 0, h - 1))
            y1 = int(np.clip(s_slope * (w - 1) + s_int, 0, h - 1))
            cv2.line(img, (0, y0), (w - 1, y1), (0, 230, 0), 1)
        cp = ContactParams()
        for (x, y), size in zip(sensor_result.coords, sensor_result.sizes):
            r = size / 2.0
            contact = water_edge_contact(seg, (x - r, y - r, x + r, y + r), cp)
            if contact is not None:
                cv2.circle(img, (int(contact[0]), int(contact[1])), 4,
                           (0, 230, 0), -1)

    if show_horizon:
        slope, intercept, conf = sensor_result.horizon_line
        if conf > 0:
            x0, x1 = 0, w - 1
            y0 = int(np.clip(slope * x0 + intercept, 0, h - 1))
            y1 = int(np.clip(slope * x1 + intercept, 0, h - 1))
            cv2.line(img, (x0, y0), (x1, y1), (0, 255, 255), 1)

    if show_projection and projected is not None and ranges is not None:
        r_min, r_max = 0.0, 9.0
        for (px, py), r in zip(projected, ranges):
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            if not (0 <= px < w and 0 <= py < h):
                continue
            norm = float(np.clip((r - r_min) / (r_max - r_min), 0, 1))
            colour = (int(255 * (1 - norm)),
                      int(255 * norm * 0.5),
                      int(255 * norm))
            cv2.circle(img, (int(px), int(py)), 3, colour, -1)

    if show_detections:
        # Blobs are plain markers: distance is shown on labelled bboxes only
        # (their bottom edge is the true waterline contact). See
        # Dashboard._draw_label_overlay.
        for (x, y) in sensor_result.coords:
            cv2.circle(img, (x, y), 6, (255, 0, 0), 2)

    # Typed object detections (LaRS-trained YOLO: boat/swimmer/buoy/...), drawn
    # last so they sit on top. Boxes are normalised xyxy in [0,1]; colour per
    # class. These are precomputed on the GPU (no local torch) and read from a
    # DetProvider: see scripts/utils/detections.py.
    if typed_dets:
        from scripts.utils.detections import CLASS_COLOURS, DEFAULT_COLOUR
        for d in typed_dets:
            box = d.get("xyxy")
            if not box or len(box) != 4:
                continue
            x0, y0, x1, y1 = box
            # normalised -> pixels (heuristic: values <=1.5 are normalised)
            if max(box) <= 1.5:
                x0, x1 = x0 * w, x1 * w
                y0, y1 = y0 * h, y1 * h
            name = str(d.get("cls", "?"))
            conf = d.get("confidence")
            colour = CLASS_COLOURS.get(name, DEFAULT_COLOUR)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), colour, 2)
            label = name if conf is None else f"{name} {float(conf):.2f}"
            ytxt = int(y0) - 4 if y0 > 12 else int(y1) + 12
            cv2.putText(img, label, (int(x0), ytxt), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, colour, 1, cv2.LINE_AA)

    # Typed INSTANCE masks (YOLOv8-seg): per-object filled polygons + label:
    # the panoptic-style view. Polygons are normalised; colour per class.
    if instance_dets:
        from scripts.utils.detections import CLASS_COLOURS, DEFAULT_COLOUR
        fill = img.copy()
        for d in instance_dets:
            poly = d.get("polygon")
            if not poly or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            if pts.max() <= 1.5:                 # normalised -> pixels
                pts[:, 0] *= w
                pts[:, 1] *= h
            pts = pts.astype(np.int32)
            name = str(d.get("cls", "?"))
            colour = CLASS_COLOURS.get(name, DEFAULT_COLOUR)
            cv2.fillPoly(fill, [pts], colour)
            cv2.polylines(img, [pts], True, colour, 2)
        img = cv2.addWeighted(fill, 0.45, img, 0.55, 0)
        # labels on top after blend
        for d in instance_dets:
            poly = d.get("polygon")
            if not poly or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            if pts.max() <= 1.5:
                pts[:, 0] *= w
                pts[:, 1] *= h
            name = str(d.get("cls", "?"))
            conf = d.get("confidence")
            colour = CLASS_COLOURS.get(name, DEFAULT_COLOUR)
            x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
            label = name if conf is None else f"{name} {float(conf):.2f}"
            cv2.putText(img, label, (x0, max(10, y0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Per-frame heading computation
#
# Lives at module scope (a) so it can be unit-tested without spinning up
# a Tk root, and (b) so the dashboard's _ensure_cached_up_to and any
# follow-on tooling share the exact same code path.
# ---------------------------------------------------------------------------

def _compute_heading_for_frame(
    res: "FrameResult",
    confirmed: list[bool],
    bin_centers: np.ndarray,
    detection_cfg: dict,
    *,
    use_tracker_gate: bool,
    current_heading_deg: Optional[float] = 0.0,
    free_dist_m: Optional[list] = None,
):
    """Return a `HeadingRecommendation` for one frame, applying the
    tracker gate the same way the dashboard's checkbox does.

    Pulled out of `_render` so producing a frame and rendering a frame
    don't share mutable state; see PLAN.md §I.4.12 for why.
    """
    scores = list(res.fusion.scores)
    if use_tracker_gate:
        for i, conf in enumerate(confirmed):
            if not conf:
                scores[i] = 0.0
    n_bins = len(bin_centers)
    v_list = list(res.fusion.per_bin_velocity_mps or [])
    t_list = list(res.fusion.per_bin_ttc_s or [])
    if len(v_list) != n_bins:
        v_list = [None] * n_bins
    if len(t_list) != n_bins:
        t_list = [None] * n_bins
    return recommend_heading(
        bin_centers_deg=list(bin_centers),
        scores=scores,
        min_range_m=list(res.fusion.min_ranges),
        per_bin_velocity_mps=v_list,
        per_bin_ttc_s=t_list,
        config=detection_cfg.get("heading", {}) or {},
        current_heading_deg=current_heading_deg,
        free_dist_m=free_dist_m,
    )


# ---------------------------------------------------------------------------
# Clip metadata helper
# ---------------------------------------------------------------------------

def _summarise_overrides(clip_overrides: dict) -> str:
    if not clip_overrides:
        return "(none)"
    bits: list[str] = []
    for sensor in ("fisheye", "thermal"):
        s = clip_overrides.get(sensor, {})
        if not s:
            continue
        parts = []
        if "target_image_size" in s:
            w, h = s["target_image_size"]
            parts.append(f"resize {w}x{h}")
        if "rotation_deg" in s:
            parts.append(f"rot {s['rotation_deg']}°")
        if parts:
            bits.append(f"{sensor[0].upper()}: {', '.join(parts)}")
    ta = clip_overrides.get("time_alignment")
    if ta:
        bits.append(f"time={ta}")
    return " · ".join(bits) if bits else "(none)"


def _count_frames(triplet: Triplet) -> int:
    try:
        df = load_mmwave_csv(triplet.mmwave)
        n_radar = int(df["RoundedTime"].nunique()) if not df.empty else 0
    except Exception:
        n_radar = -1
    if n_radar > 0:
        return n_radar
    # Empty radar CSV: iterate_triplet falls back to fisheye frames,
    # so the scrub bar should reflect the video length.
    cap = cv2.VideoCapture(str(triplet.fisheye))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return n if n > 0 else n_radar


def _parse_ts_seconds(ts: str) -> Optional[float]:
    """Parse 'HH:MM:SS' or 'HH:MM:SS.f' into seconds-since-midnight."""
    if not ts:
        return None
    parts = ts.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class Dashboard:
    def __init__(self, root, initial_triplet: Optional[str] = None,
                 detection_path: Optional[str] = None,
                 pseudo_path: Optional[str] = None,
                 exclude_path: Optional[str] = None, target: int = 500,
                 train_out: Optional[str] = None):
        if not _TK_AVAILABLE:
            raise RuntimeError(
                f"tkinter is required for the dashboard but is not available "
                f"({_TK_ERROR}). On macOS with Homebrew Python, install with: "
                f"brew install python-tk@<major>.<minor>  "
                f"(e.g. `brew install python-tk@3.14`)."
            )
        self.root = root
        self.root.title("Obstacle-detection debug dashboard")
        self.detection_path = detection_path

        # --- Audit mode: pre-fill frames with detector suggestions, accept a
        # curated subset into the manual set with a n/target counter. Gated on
        # --pseudo so the normal dashboard is untouched without it.
        self._target = int(target)
        self._frame_committed = False
        self._pseudo: dict[str, list] = {}
        self._exclude: set[str] = set()
        # Accepted audit frames persist to their OWN file (separate from the
        # eval annotations in manual.jsonl), so you can quit and resume across
        # sessions and the n/target counter picks up where you left off.
        self._train_out: Optional[Path] = Path(train_out) if train_out else None
        if pseudo_path:
            for line in Path(pseudo_path).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fid = r.get("frame_id")
                if isinstance(fid, str):
                    self._pseudo[fid] = [
                        {"cls": b["cls"], "xyxy": list(b["xyxy"])}
                        for b in r.get("fisheye_bboxes", [])]
        self._pseudo_mode = bool(self._pseudo)
        if pseudo_path:
            n_with = sum(1 for v in self._pseudo.values() if v)
            print(f"[dashboard] audit mode: {len(self._pseudo)} suggested "
                  f"frames loaded from {pseudo_path} ({n_with} with boxes: "
                  f"empty frames are normal on open water)")
            if not self._pseudo:
                print("[dashboard] WARN --pseudo file yielded no records: "
                      "check the path/JSONL")
        if exclude_path:
            # Exclude the ORIGINAL frame_ids that back the eval clip so the
            # held-out test images never enter the training set (no leakage).
            m = json.loads(Path(exclude_path).read_text())
            self._exclude = set(m.get("fake_to_original", {}).values())

        # Core configs: loaded once, mutable on hot-reload.
        self.intrinsics = load_intrinsics()
        self.detection = (load_detection(detection_path)
                          if detection_path else load_detection())
        self.extrinsics = load_extrinsics()
        self.all_overrides = load_overrides()

        # Per-clip state: set by _load_triplet().
        self.triplet: Optional[Triplet] = None
        self.pipeline: Optional[ObstacleDetectionPipeline] = None
        self.tracker: Optional[SectorTracker] = None
        self.frame_total: int = 0
        self._gen = None
        # Cached frames carry every per-frame derived quantity so a
        # re-render after stepping back / scrubbing is a pure read:
        # the smoother and heading recommender never advance twice on
        # the same frame.
        self.cached: list[tuple[str, np.ndarray, np.ndarray, np.ndarray,
                                FrameResult, list[bool],
                                HeadingRecommendation,
                                Optional[float]]] = []
        self.idx = 0
        self.playing = False
        self.speed_ms = 333
        self._tick_handle: Optional[str] = None
        self._scrub_user = False
        self._scrub_resume = False
        self._exporting = False
        self.scrub_var = tk.DoubleVar(value=0.0)

        # Display toggles.
        self.var_projection = tk.BooleanVar(value=True)
        self.var_horizon = tk.BooleanVar(value=True)
        self.var_wedges = tk.BooleanVar(value=True)
        self.var_tracker = tk.BooleanVar(value=True)
        self.var_heading = tk.BooleanVar(value=True)
        # Camera detection blobs (the red circles). Turn off for a clean
        # canvas while annotating.
        self.var_detections = tk.BooleanVar(value=True)
        # Water-segmentation overlay (eWaSR/LaRS). Only meaningful when masks
        # exist under the seg root for the loaded clip; the provider is None
        # otherwise and the toggle is a no-op. See scripts/utils/segmentation.py.
        seg_root = (self.detection.get("segmentation", {}) or {}).get(
            "seg_root", "data/seg")
        seg_root_path = Path(seg_root)
        if not seg_root_path.is_absolute():
            seg_root_path = REPO_ROOT_FOR_LABELS / seg_root
        self._seg_provider = (
            SegProvider(seg_root_path) if seg_root_path.is_dir() else None
        )
        # Default the overlay ON when masks are available, else OFF.
        self.var_seg = tk.BooleanVar(value=self._seg_provider is not None)
        # Typed object detections (LaRS-trained YOLO). Precomputed on the GPU
        # (no local torch) and read per-frame from data/det. See
        # scripts/utils/detections.py + scripts/gpu_finetune/yolo_predict_frames.py.
        det_root = Path("data/det")
        if not det_root.is_absolute():
            det_root = REPO_ROOT_FOR_LABELS / "data" / "det"
        self._det_provider = DetProvider(det_root) if det_root.is_dir() else None
        self.var_typed = tk.BooleanVar(value=self._det_provider is not None)
        # Typed instance masks (YOLOv8-seg): the panoptic-style per-object view.
        inst_root = REPO_ROOT_FOR_LABELS / "data" / "det_seg"
        self._inst_provider = InstanceSegProvider(inst_root) if inst_root.is_dir() else None
        self.var_instances = tk.BooleanVar(value=self._inst_provider is not None)
        # Manual label bboxes overlaid on the fisheye. Hoisted here (shared with
        # the labelling tab's own checkbox) so it's also toggleable from the main
        # show/no-show panel: they shouldn't have to be on all the time.
        self.var_show_labels = tk.BooleanVar(value=True)
        # Live toggle: feed the segmentation free-space into the heading
        # recommender (Phase 1). Initialised from detection.yaml
        # heading.use_free_space; flipping it recomputes the cached headings in
        # place (same panel, updated computation) for a one-click A/B.
        self.var_freespace_heading = tk.BooleanVar(
            value=bool((self.detection.get("heading", {}) or {}).get("use_free_space", False)))
        # Monocular water-plane range labels, drawn on labelled bboxes only.
        # ON as a BASELINE for now: the numbers are still distortion-biased at
        # the 0.27 m mount height until the fisheye is recalibrated (see
        # fisheye_recalibrate.py): useful to eyeball and to compare against the
        # same frames after recalibration. The toggle turns them off; when the
        # BNO085 is live (configs/imu.yaml) the attitude side also improves.
        self._imu = self._maybe_start_imu()
        self.var_range_labels = tk.BooleanVar(value=True)
        self.var_threshold = tk.DoubleVar(
            value=float(self.detection["fusion"].get("hit_threshold", 0.33))
        )
        # Rolling history of recommended heading for the sparkline.
        # ~30 s at the radar's 10 Hz nominal cadence ≈ 300 frames; clips
        # in data/ are short so 300 is also "the whole clip" usually.
        self.heading_history: deque[Optional[float]] = deque(maxlen=300)
        self.smoothed_history: deque[Optional[float]] = deque(maxlen=300)
        self.last_heading: Optional[HeadingRecommendation] = None
        head_cfg = self.detection.get("heading", {}) or {}
        self.heading_smoother = HeadingSmoother(
            window_n=int(head_cfg.get("smoothing_window", 9)),
            min_samples=int(head_cfg.get("smoothing_min_samples", 3)),
        )
        # Temporal EMA on the segmentation free-space (Phase 1) so the seg-driven
        # heading doesn't jitter frame-to-frame. alpha=1.0 disables smoothing.
        self.freespace_smoother = FreeSpaceSmoother(
            alpha=float(head_cfg.get("free_space_smoothing_alpha", 0.5)),
            max_range_m=float((self.detection.get("range", {}) or {}).get("max_range_m", 15.0)),
        )

        # Layout state: built-in presets are merged with anything the user
        # has saved to ~/.config/asvproject-dashboard/layouts.json. "current"
        # in that file is the layout used at startup; updated every Apply.
        stored = _load_layouts_file()
        self.custom_presets: dict[str, dict] = stored["presets"]
        self.layout: dict[str, dict] = _normalize_layout(
            stored["current"] or DEFAULT_LAYOUT
        )

        # Labelling state. `label_store` is shared across triplets: it
        # holds every record in labels/manual.jsonl keyed by frame_id and
        # rewrites the file on every put(). `current_label` is the in-
        # progress record for the playhead's frame; it gets written to
        # the store every time the user navigates away from the frame.
        # In audit mode, the "manual set" is the dedicated training file
        # (--train-out); every accept + edit persists there and reloads on
        # restart. Outside audit mode it's the usual labels/manual.jsonl.
        self.label_store = LabelStore(self._train_out)
        self.current_label: dict = self._empty_label_record()
        # Per-frame undo: snapshots of current_label taken before each edit so
        # Cmd/Ctrl+Z can step back. Reset on every frame change.
        self._undo_stack: list[dict] = []
        self._undo_snapshot: Optional[dict] = None
        self._undoing: bool = False
        self.label_tool: str = LABEL_TOOL_NONE
        self.label_class: str = LABEL_CLASSES[0]
        self.label_show_existing: bool = True
        # Mouse drag state for bbox drawing.
        self._bbox_drag_start: Optional[tuple[float, float]] = None
        self._bbox_drag_current: Optional[tuple[float, float]] = None
        # First click for the 2-point horizon picker.
        self._horizon_pending: Optional[tuple[float, float]] = None
        # mpl artists currently on the fisheye axis from labels: cleared
        # at the start of every _render so we don't leak ghosts.
        self.label_artists: list = []
        self._mpl_cids: list[int] = []

        self._build_chrome()
        self._build_figure()
        self._populate_triplets()
        self._bind_keys()
        self._connect_mpl_mouse_events()

        # Auto-load whatever the user passed in on the CLI.
        if initial_triplet:
            self._preload_triplet(initial_triplet)

    # -- UI construction ----------------------------------------------------

    def _build_chrome(self):
        root = self.root
        root.columnconfigure(0, weight=1)
        # Row 0 = top bar, row 1 = layout controls, row 2 = scrub bar,
        # row 3 = main figure + sidebar (expandable), row 4 = status.
        root.rowconfigure(3, weight=1)

        # Top bar: current clip + the global utility buttons. Clip
        # selection itself moved to the sidebar's Clips tab, where the
        # full data/ folder tree lives: we just surface the loaded clip
        # path here so the transport controls have context.
        top = ttk.Frame(root, padding=(8, 6))
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Clip:").pack(side="left")
        self.lbl_clip = ttk.Label(
            top, text="(no clip loaded: pick one from the Clips tab)",
            foreground="#555", font=("TkFixedFont", 10),
        )
        self.lbl_clip.pack(side="left", padx=(4, 12))

        ttk.Button(top, text="Reload cfg",
                   command=self._hot_reload).pack(side="left", padx=2)
        ttk.Button(top, text="Screenshot",
                   command=self._screenshot).pack(side="left", padx=2)
        ttk.Button(top, text="Export video…",
                   command=self._open_export_dialog).pack(side="left", padx=2)

        # Transport row continues inside the same top bar.
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y",
                                                    padx=8)
        ttk.Button(top, text="◀", width=3,
                   command=lambda: self._step(-1)).pack(side="left")
        ttk.Button(top, text="-5s", width=4,
                   command=lambda: self._jump_seconds(-5.0)).pack(
                       side="left", padx=(2, 0))
        self.btn_play = ttk.Button(top, text="▶", width=3,
                                   command=self._toggle_play)
        self.btn_play.pack(side="left", padx=2)
        ttk.Button(top, text="+5s", width=4,
                   command=lambda: self._jump_seconds(+5.0)).pack(
                       side="left", padx=(0, 2))
        ttk.Button(top, text="▶|", width=3,
                   command=lambda: self._step(+1)).pack(side="left")
        self.lbl_frame = ttk.Label(top, text="frame -/-")
        self.lbl_frame.pack(side="left", padx=(12, 6))
        self.lbl_ts = ttk.Label(top, text="ts -", width=18)
        self.lbl_ts.pack(side="left")
        ttk.Label(top, text="speed").pack(side="left", padx=(12, 4))
        self.cb_speed = ttk.Combobox(top, values=[s[0] for s in SPEEDS],
                                     state="readonly", width=5)
        self.cb_speed.set("1x")
        self.cb_speed.pack(side="left")
        self.cb_speed.bind("<<ComboboxSelected>>", self._on_speed_change)

        # Layout controls row: preset combobox + custom editor entry.
        # Sits between the top transport bar and the scrub bar so the
        # layout selector lives next to the rest of the global controls
        # rather than being buried in the sidebar.
        layout_row = ttk.Frame(root, padding=(8, 2))
        layout_row.grid(row=1, column=0, sticky="ew")
        ttk.Label(layout_row, text="Layout").pack(side="left")
        self.cb_layout = ttk.Combobox(layout_row, values=[],
                                      state="readonly", width=26)
        self.cb_layout.pack(side="left", padx=(4, 8))
        self.cb_layout.bind("<<ComboboxSelected>>", self._on_layout_preset)
        ttk.Button(layout_row, text="Customize…",
                   command=self._open_layout_dialog).pack(side="left")
        ttk.Button(layout_row, text="Save as preset…",
                   command=self._save_current_as_preset).pack(side="left",
                                                              padx=(4, 0))
        self._refresh_layout_combo()

        # Scrub bar: drag through footage like a YouTube seek bar.
        scrub_row = ttk.Frame(root, padding=(8, 2))
        scrub_row.grid(row=2, column=0, sticky="ew")
        scrub_row.columnconfigure(0, weight=1)
        self.scrub = ttk.Scale(scrub_row, from_=0, to=0, orient="horizontal",
                               variable=self.scrub_var,
                               command=self._on_scrub_command)
        self.scrub.grid(row=0, column=0, sticky="ew")
        self.scrub.bind("<ButtonPress-1>", self._on_scrub_press)
        self.scrub.bind("<ButtonRelease-1>", self._on_scrub_release)

        # Main split: matplotlib left, sidebar right.
        main = ttk.Frame(root)
        main.grid(row=3, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        self.fig_frame = ttk.Frame(main)
        self.fig_frame.grid(row=0, column=0, sticky="nsew")
        self.side = ttk.Frame(main, padding=(4, 6), width=420)
        self.side.grid(row=0, column=1, sticky="ns")
        self.side.grid_propagate(False)

        self._build_sidebar()

        # Status bar.
        self.lbl_status = ttk.Label(root, anchor="w",
                                    text="ready: pick a scene + clip",
                                    padding=(8, 4))
        self.lbl_status.grid(row=4, column=0, sticky="ew")

    def _make_scrollable(self, parent: "ttk.Frame") -> "ttk.Frame":
        """Wrap `parent` in a vertically-scrolling canvas and return the inner
        frame to build content into, so a tall sidebar scrolls with the
        wheel/trackpad. Call _activate_scroll_wheels() once content is built."""
        if not hasattr(self, "_scrollables"):
            self._scrollables = []
        canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win, width=e.width))
        self._scrollables.append((canvas, inner))
        return inner

    def _activate_scroll_wheels(self) -> None:
        """Give the canvas-backed sidebar tabs the SAME trackpad physics as the
        Clips Treeview.

        Tk 9.0 routes precise trackpad scrolling to <TouchpadScroll>; the canvas
        has no built-in handler (ttk Treeview does, which is why only Clips
        scrolled, and smoothly). We mirror the Treeview's own binding exactly:
        throttle to every 5th event (Tk's `%# %% 5 == 0`: touchpads fire a
        flood, and acting on all of them is what made the feel hyper-sensitive),
        decode the packed delta with tk::PreciseScrollDeltas, and yview-scroll by
        -deltaY units (proportional to the gesture, not a fixed step). Mouse
        wheels use Tk's divisor convention. Scrolls the selected tab's canvas."""
        def _active_canvas():
            try:
                sel = self.nb_sidebar.select()
            except Exception:
                return None
            for canvas, _inner in getattr(self, "_scrollables", []):
                if str(canvas.master) == sel:
                    return canvas
            return None  # e.g. Clips (Treeview scrolls itself)

        self._touchpad_serial = 0

        def _on_touchpad(e):
            self._touchpad_serial += 1
            if self._touchpad_serial % 5 != 0:   # mirror Tk's `%# %% 5 == 0`
                return None
            canvas = _active_canvas()
            if canvas is None:
                return None
            try:
                dx, dy = self.root.tk.call("tk::PreciseScrollDeltas", e.delta)
            except (tk.TclError, ValueError):
                return None
            if int(dy):
                canvas.yview_scroll(-int(dy), "units")
            return "break"

        def _on_wheel(e):
            canvas = _active_canvas()
            if canvas is None:
                return None
            num = getattr(e, "num", None)
            if num == 4:
                canvas.yview_scroll(-1, "units"); return "break"
            if num == 5:
                canvas.yview_scroll(1, "units"); return "break"
            d = getattr(e, "delta", 0)
            if not d:
                return None
            units = int(round(-d / 40.0)) or (-1 if d > 0 else 1)  # Tk aqua divisor
            canvas.yview_scroll(units, "units")
            return "break"

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(seq, _on_wheel, add="+")
        try:                                  # Tk >= 8.7/9.0 only
            self.root.bind_all("<TouchpadScroll>", _on_touchpad, add="+")
        except tk.TclError:
            pass

    def _build_sidebar(self):
        # Sidebar lives in a Notebook so the same screen real-estate
        # serves the Clips browser (data/ folder tree), the detection
        # view (sensor health, fusion bins, heading), and the labelling
        # view (annotation tools, bbox list, horizon picker). Mouse
        # events on the fisheye axis behave differently depending on
        # which tab is active: see `_on_canvas_button_press`.
        self.nb_sidebar = ttk.Notebook(self.side)
        self.nb_sidebar.pack(fill="both", expand=True)
        self.tab_clips = ttk.Frame(self.nb_sidebar, padding=(4, 4))
        self.tab_detect = ttk.Frame(self.nb_sidebar, padding=(4, 4))
        self.tab_label = ttk.Frame(self.nb_sidebar, padding=(4, 4))
        self.nb_sidebar.add(self.tab_clips, text="Clips")
        self.nb_sidebar.add(self.tab_detect, text="Detection")
        self.nb_sidebar.add(self.tab_label, text="Labelling")
        self.nb_sidebar.bind("<<NotebookTabChanged>>",
                             self._on_sidebar_tab_changed)
        self._build_clips_tab(self.tab_clips)

        # --- Detection tab -----------------------------------------------
        # Scrollable body so the (tall) detection sidebar can be panned.
        detect_body = self._make_scrollable(self.tab_detect)
        # Per-sensor health.
        health = ttk.LabelFrame(detect_body, text="Per-sensor health",
                                padding=(6, 4))
        health.pack(fill="x", pady=(0, 6))
        self.lbl_health = {
            "fisheye": ttk.Label(health, text="Fisheye: ", anchor="w",
                                 font=("TkFixedFont", 10)),
            "thermal": ttk.Label(health, text="Thermal: ", anchor="w",
                                 font=("TkFixedFont", 10)),
            "mmwave":  ttk.Label(health, text="mmWave: ", anchor="w",
                                 font=("TkFixedFont", 10)),
        }
        for w in self.lbl_health.values():
            w.pack(fill="x")

        # Heading recommendation.
        head_lf = ttk.LabelFrame(detect_body, text="Heading",
                                 padding=(6, 4))
        head_lf.pack(fill="x", pady=(0, 6))
        self.lbl_heading = ttk.Label(head_lf, text="recommended:: ", anchor="w",
                                     font=("TkFixedFont", 10))
        self.lbl_heading.pack(fill="x")
        self.lbl_heading_reason = ttk.Label(
            head_lf, text="reason:: ", anchor="w",
            font=("TkFixedFont", 10),
        )
        self.lbl_heading_reason.pack(fill="x")
        self.lbl_heading_smoothed = ttk.Label(
            head_lf, text="smoothed:: ", anchor="w",
            font=("TkFixedFont", 10), foreground="#1c4587",
        )
        self.lbl_heading_smoothed.pack(fill="x")
        self.lbl_heading_current = ttk.Label(
            head_lf, text="current bow: 0° (IMU stub)", anchor="w",
            font=("TkFixedFont", 9), foreground="#666",
        )
        self.lbl_heading_current.pack(fill="x")

        # Per-bin table. `v` (closing speed m/s) and `TTC` (s) are the
        # v1 fields from PLAN.md §I.4.9; both fall back to "—" when
        # the radar produced no velocity in that bin.
        bins_lf = ttk.LabelFrame(detect_body, text="Per-bin",
                                 padding=(4, 4))
        bins_lf.pack(fill="both", expand=True, pady=(0, 6))
        cols = ("score", "f", "t", "m", "range", "v", "ttc", "track")
        self.tree = ttk.Treeview(bins_lf, columns=cols, show="tree headings",
                                 height=11)
        self.tree.heading("#0", text="bin")
        self.tree.column("#0", width=44, anchor="center")
        widths = {"score": 44, "f": 24, "t": 24, "m": 24, "range": 50,
                  "v": 50, "ttc": 50, "track": 60}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths[c], anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("blocked", background="#d9ead3",
                                font=("TkDefaultFont", 9, "bold"))
        self.tree.tag_configure("confirmed", background="#fff2cc")
        self.tree.tag_configure("urgent", background="#f4cccc")

        # Clip metadata.
        meta_lf = ttk.LabelFrame(detect_body, text="Clip metadata",
                                 padding=(6, 4))
        meta_lf.pack(fill="x", pady=(0, 6))
        self.lbl_meta = ttk.Label(meta_lf, text="—", anchor="w", justify="left",
                                  font=("TkFixedFont", 9))
        self.lbl_meta.pack(fill="x")

        # Toggles.
        tog = ttk.LabelFrame(detect_body, text="Toggles", padding=(6, 4))
        tog.pack(fill="x")
        ttk.Checkbutton(tog, text="Camera detections (red blobs)",
                        variable=self.var_detections,
                        command=self._rerender).pack(anchor="w")
        seg_txt = ("Segmentation (amber=obstacle, blue=water, purple=sky)"
                   if self._seg_provider is not None
                   else "Segmentation (no masks in data/seg)")
        seg_cb = ttk.Checkbutton(tog, text=seg_txt, variable=self.var_seg,
                                 command=self._rerender)
        seg_cb.pack(anchor="w")
        if self._seg_provider is None:
            seg_cb.configure(state="disabled")
        typed_txt = ("Typed detections (LaRS YOLO: boat/buoy/swimmer/...)"
                     if self._det_provider is not None
                     else "Typed detections (none in data/det)")
        typed_cb = ttk.Checkbutton(tog, text=typed_txt, variable=self.var_typed,
                                   command=self._rerender)
        typed_cb.pack(anchor="w")
        if self._det_provider is None:
            typed_cb.configure(state="disabled")
        inst_txt = ("Instance masks (YOLOv8-seg: per-object typed masks)"
                    if self._inst_provider is not None
                    else "Instance masks (none in data/det_seg)")
        inst_cb = ttk.Checkbutton(tog, text=inst_txt, variable=self.var_instances,
                                  command=self._rerender)
        inst_cb.pack(anchor="w")
        if self._inst_provider is None:
            inst_cb.configure(state="disabled")
        ttk.Checkbutton(tog, text="Manual label boxes",
                        variable=self.var_show_labels,
                        command=self._on_show_labels_toggle).pack(anchor="w")
        fsh_txt = ("Free-space heading (seg drives the recommendation)"
                   if self._seg_provider is not None
                   else "Free-space heading (no masks in data/seg)")
        fsh_cb = ttk.Checkbutton(tog, text=fsh_txt,
                                 variable=self.var_freespace_heading,
                                 command=self._on_freespace_heading_toggle)
        fsh_cb.pack(anchor="w")
        if self._seg_provider is None:
            fsh_cb.configure(state="disabled")
        ttk.Checkbutton(tog, text="Range labels (monocular, m)",
                        variable=self.var_range_labels,
                        command=self._rerender).pack(anchor="w")
        ttk.Checkbutton(tog, text="Radar projection on cameras",
                        variable=self.var_projection,
                        command=self._rerender).pack(anchor="w")
        ttk.Checkbutton(tog, text="Horizon line",
                        variable=self.var_horizon,
                        command=self._rerender).pack(anchor="w")
        ttk.Checkbutton(tog, text="Radar bin wedges",
                        variable=self.var_wedges,
                        command=self._rerender).pack(anchor="w")
        ttk.Checkbutton(tog, text="Tracker overlay on bars",
                        variable=self.var_tracker,
                        command=self._rerender).pack(anchor="w")
        ttk.Checkbutton(tog, text="Heading curve + recommendation",
                        variable=self.var_heading,
                        command=self._rerender).pack(anchor="w")
        thr_row = ttk.Frame(tog)
        thr_row.pack(fill="x", pady=(4, 0))
        ttk.Label(thr_row, text="Hit threshold").pack(side="left")
        self.lbl_thr = ttk.Label(thr_row, text=f"{self.var_threshold.get():.2f}",
                                 width=5)
        self.lbl_thr.pack(side="right")
        ttk.Scale(tog, from_=0.0, to=1.0, orient="horizontal",
                  variable=self.var_threshold,
                  command=self._on_threshold).pack(fill="x")

        # --- Labelling tab ----------------------------------------------
        self._build_label_tab(self._make_scrollable(self.tab_label))

        # Enable wheel/trackpad scrolling now that all sidebar content exists.
        self._activate_scroll_wheels()

    def _build_figure(self):
        """Create the Figure + Canvas once and lay axes out via _apply_layout.

        Bar/wedge bin definitions are needed by panel-setup methods, so
        compute them up front (hot-reload recomputes them too).
        """
        self.fig = Figure(figsize=(11, 10), dpi=100)
        edges, centers = make_bins(self.detection["fusion"])
        self.bin_edges = edges
        self.bin_centers = centers
        self.bar_x = np.arange(len(centers))

        # Nulled here so _apply_layout's `if visible` branches always have
        # something to overwrite, and the renderer can guard with `is None`
        # when a panel is hidden.
        self.ax_fish = None
        self.ax_therm = None
        self.ax_radar = None
        self.ax_bar = None
        self.ax_head_curve = None
        self.ax_head_hist = None
        self.im_fish = None
        self.im_therm = None
        self.radar_scat = None
        self.wedges: list = []
        self.bars: list = []
        self.threshold_line = None
        self.line_head_curve = None
        self.head_choice_line = None
        self.line_head_hist = None
        self.line_head_hist_smoothed = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._apply_layout()

    # -- Layout application ------------------------------------------------

    def _apply_layout(self) -> None:
        """Clear the figure and rebuild axes from self.layout.

        Cheap enough that we call it for every preset switch + Apply in
        the dialog; cached frame data is independent of figure state, so
        we just re-render the current playhead at the end.
        """
        self.fig.clf()
        # Reset all panel references so partial layouts leave None where
        # a panel was hidden: the render path then skips it.
        self.ax_fish = None
        self.ax_therm = None
        self.ax_radar = None
        self.ax_bar = None
        self.ax_head_curve = None
        self.ax_head_hist = None
        self.im_fish = None
        self.im_therm = None
        self.radar_scat = None
        self.wedges = []
        self.bars = []
        self.threshold_line = None
        self.line_head_curve = None
        self.head_choice_line = None
        self.line_head_hist = None
        self.line_head_hist_smoothed = None

        # Tight rows for the heading panels so the cost curve title doesn't
        # clip into the row above; matches what the original 3×2 used.
        gs = self.fig.add_gridspec(
            LAYOUT_ROWS, LAYOUT_COLS,
            hspace=0.6, wspace=0.18,
            height_ratios=[1.0, 1.0, 0.55],
        )

        setup_funcs = {
            "fisheye": self._setup_fisheye_axis,
            "thermal": self._setup_thermal_axis,
            "radar": self._setup_radar_axis,
            "bars": self._setup_bars_axis,
            "heading_curve": self._setup_heading_curve_axis,
            "heading_hist": self._setup_heading_hist_axis,
        }
        for key in PANEL_KEYS:
            placement = _clamp_placement(
                self.layout.get(key, DEFAULT_LAYOUT[key])
            )
            if not placement["visible"]:
                continue
            r = placement["row"]
            c = placement["col"]
            rs = placement["rowspan"]
            cs = placement["colspan"]
            ax = self.fig.add_subplot(gs[r:r + rs, c:c + cs])
            setup_funcs[key](ax)

        self.canvas.draw_idle()

    # -- Per-panel axis setup ----------------------------------------------

    def _setup_fisheye_axis(self, ax) -> None:
        self.ax_fish = ax
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Fisheye RGB")
        self.im_fish = ax.imshow(np.zeros((10, 10, 3), dtype=np.uint8))

    def _setup_thermal_axis(self, ax) -> None:
        self.ax_therm = ax
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Thermal")
        self.im_therm = ax.imshow(np.zeros((10, 10, 3), dtype=np.uint8))

    def _setup_radar_axis(self, ax) -> None:
        self.ax_radar = ax
        ax.set_title("Radar (bird's eye)")
        self.radar_scat = ax.scatter([], [], c=[], cmap="turbo",
                                     s=30, vmin=0, vmax=8)
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 9)
        ax.set_xlabel("X lateral (m)")
        ax.set_ylabel("Y forward (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        self._draw_wedges()

    def _setup_bars_axis(self, ax) -> None:
        self.ax_bar = ax
        ax.set_title("Fusion bin scores")
        self.bars = ax.bar(self.bar_x,
                           np.zeros_like(self.bin_centers, dtype=float))
        ax.set_ylim(0, 1.05)
        ax.set_xticks(self.bar_x)
        ax.set_xticklabels([f"{int(c):+d}°" for c in self.bin_centers])
        self.threshold_line = ax.axhline(
            self.var_threshold.get(), color="orange", linestyle="--",
            linewidth=0.8,
        )
        ax.set_ylabel("fused score")

    def _setup_heading_curve_axis(self, ax) -> None:
        self.ax_head_curve = ax
        ax.set_title("Heading cost (recommended = green)")
        ax.set_xlabel("candidate heading (°)")
        ax.set_ylabel("cost")
        ax.grid(True, alpha=0.3)
        (self.line_head_curve,) = ax.plot([], [], color="#3c78d8",
                                          linewidth=1.4)
        self.head_choice_line = ax.axvline(
            0.0, color="#6aa84f", linestyle="--", linewidth=1.4, alpha=0.0,
        )

    def _setup_heading_hist_axis(self, ax) -> None:
        self.ax_head_hist = ax
        ax.set_title("Recommended heading (rolling)")
        ax.set_xlabel("frame")
        ax.set_ylim(-60, 60)
        ax.axhline(0, color="#888", linewidth=0.5, alpha=0.6)
        ax.grid(True, alpha=0.3)
        # Two-trace sparkline: raw recommendation (thin grey) plus the
        # smoothed steering signal the autopilot would actually follow
        # (thick green). Smoothed trace holds across ambiguous frames.
        (self.line_head_hist,) = ax.plot(
            [], [], color="#aaaaaa", linewidth=0.8, label="raw",
        )
        (self.line_head_hist_smoothed,) = ax.plot(
            [], [], color="#6aa84f", linewidth=1.8, label="smoothed",
        )
        ax.legend(loc="upper left", fontsize=7, frameon=False)

    def _draw_wedges(self):
        for w in self.wedges:
            try:
                w.remove()
            except (ValueError, NotImplementedError):
                pass
        self.wedges = []
        if self.ax_radar is None or not self.var_wedges.get():
            return
        radius = self.ax_radar.get_ylim()[1]
        for e0, e1 in zip(self.bin_edges[:-1], self.bin_edges[1:]):
            wedge = Wedge((0, 0), radius, 90 - float(e1), 90 - float(e0),
                          facecolor="none", edgecolor="gray",
                          linewidth=0.3, alpha=0.6)
            self.ax_radar.add_patch(wedge)
            self.wedges.append(wedge)

    # -- Clips tab + triplet loading ---------------------------------------

    def _build_clips_tab(self, parent: ttk.Frame) -> None:
        """Sidebar tree mirroring data/. Folders + triplet prefixes only;
        individual .mp4/.csv triplet members are summarised by the
        timestamp leaf so the tree doesn't show three lines per triplet."""
        header = ttk.Frame(parent)
        header.pack(fill="x")
        ttk.Label(header, text="data/ folder",
                  font=("TkDefaultFont", 9, "bold")).pack(side="left")
        ttk.Button(header, text="Refresh",
                   command=self._populate_clip_tree,
                   width=8).pack(side="right")

        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.tree_clips = ttk.Treeview(
            tree_frame, show="tree", selectmode="browse",
        )
        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self.tree_clips.yview)
        self.tree_clips.configure(yscrollcommand=vsb.set)
        self.tree_clips.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        # Triplet leaves vs folder nodes get distinct visual treatment so
        # the operator can tell at a glance what's clickable.
        self.tree_clips.tag_configure("triplet", foreground="#1c4587")
        self.tree_clips.tag_configure("folder", foreground="#222")
        self.tree_clips.bind("<Double-1>", self._on_clip_tree_activate)
        self.tree_clips.bind("<Return>", self._on_clip_tree_activate)

        ttk.Label(
            parent,
            text=("double-click or press Enter to load.\n"
                  "Triplet prefixes (fisheye+thermal+mmwave) appear in blue."),
            foreground="#666", font=("TkDefaultFont", 9),
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        # Lookup table from tree iid → Triplet for activation.
        self._clip_tree_triplets: dict[str, Triplet] = {}

    def _populate_clip_tree(self) -> None:
        """Walk data/ and rebuild the tree to match the current filesystem."""
        if not hasattr(self, "tree_clips"):
            return
        self.tree_clips.delete(*self.tree_clips.get_children())
        self._clip_tree_triplets = {}
        data_root = REPO_ROOT_FOR_LABELS / "data"
        if not data_root.is_dir():
            self._status(f"no data/ directory at {data_root}")
            return
        self._populate_clip_tree_node(data_root, parent_iid="")
        # Auto-expand the top level so scenes are visible without an
        # extra click. Deeper levels stay collapsed to keep the tree
        # tidy until the operator drills in.
        for iid in self.tree_clips.get_children(""):
            self.tree_clips.item(iid, open=True)
        # Keep the currently loaded clip visible + selected if there is one.
        if self.triplet is not None:
            self._reveal_triplet_in_tree(self.triplet)

    def _populate_clip_tree_node(self, directory: Path,
                                  parent_iid: str) -> bool:
        """Insert triplet leaves and recurse into subdirs. Returns True if
        any triplet exists at or below this node (lets callers prune
        completely empty branches if they want; we currently keep them
        because the user asked for the full folder layout)."""
        any_triplet = False
        seen_ts: set[str] = set()
        for fish in sorted(directory.glob("fisheye_*.mp4")):
            ts = fish.stem[len("fisheye_"):]
            if ts in seen_ts:
                continue
            seen_ts.add(ts)
            try:
                triplet = resolve_triplet(directory / ts)
            except FileNotFoundError:
                # Incomplete triplet (e.g. missing mmwave csv): surface
                # as a folder-coloured leaf so the operator sees what's
                # there but can't load it.
                self.tree_clips.insert(
                    parent_iid, "end", text=f"{ts}  (incomplete)",
                    tags=("folder",),
                )
                continue
            iid = self.tree_clips.insert(
                parent_iid, "end", text=ts, tags=("triplet",),
            )
            self._clip_tree_triplets[iid] = triplet
            any_triplet = True
        for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
            if sub.name.startswith("."):
                continue
            sub_iid = self.tree_clips.insert(
                parent_iid, "end", text=f"{sub.name}/", tags=("folder",),
            )
            child_has = self._populate_clip_tree_node(sub, sub_iid)
            any_triplet = any_triplet or child_has
        return any_triplet

    def _reveal_triplet_in_tree(self, triplet: Triplet) -> None:
        for iid, t in self._clip_tree_triplets.items():
            if t.fisheye == triplet.fisheye:
                # Walk up to open all ancestors so the leaf is visible.
                parent = self.tree_clips.parent(iid)
                while parent:
                    self.tree_clips.item(parent, open=True)
                    parent = self.tree_clips.parent(parent)
                try:
                    self.tree_clips.selection_set(iid)
                    self.tree_clips.see(iid)
                except tk.TclError:
                    pass
                return

    def _on_clip_tree_activate(self, _event=None) -> None:
        sel = self.tree_clips.selection()
        if not sel:
            return
        iid = sel[0]
        triplet = self._clip_tree_triplets.get(iid)
        if triplet is None:
            # Folder node: toggle expand/collapse so the tree double-
            # click behaves the way file browsers do.
            is_open = bool(self.tree_clips.item(iid, "open"))
            self.tree_clips.item(iid, open=not is_open)
            return
        self._load_triplet(triplet)

    def _populate_triplets(self) -> None:
        """Backwards-compat shim: build the tree the chrome relies on."""
        self._populate_clip_tree()

    def _preload_triplet(self, prefix: str) -> None:
        """Resolve `data/<...>/<timestamp>` and load it.

        Used by the `--triplet` CLI argument. Accepts both absolute paths
        and paths relative to the repo root.
        """
        p = Path(prefix)
        if not p.is_absolute():
            p = REPO_ROOT_FOR_LABELS / p
        try:
            triplet = resolve_triplet(p)
        except FileNotFoundError as exc:
            self._status(f"failed to preload triplet: {exc}")
            return
        self._load_triplet(triplet)

    def _on_load(self) -> None:
        """Reload the current triplet: used by `_hot_reload` to re-run
        the pipeline against the same clip with the freshly reloaded
        detection.yaml. First-time loading goes via `_load_triplet`."""
        if self.triplet is None:
            return
        self._load_triplet(self.triplet)

    def _maybe_start_imu(self):
        """Start the BNO085 attitude reader iff configs/imu.yaml enables it.

        Returns a started reader (polled by the pipeline for range attitude) or
        None. Vision-only monocular range is unreliable at the low mount height,
        so range labels stay hidden until a real attitude source is live; this
        is what re-enables them. Never lets IMU setup break the dashboard.
        """
        try:
            from scripts.sensor_processing.imu_bno085 import (
                BNO085Reader,
                load_imu_config,
            )
            cfg = load_imu_config()
            if not cfg.get("enabled", False):
                return None
            reader = BNO085Reader(cfg).start()
            print(f"[dashboard] IMU enabled (backend={reader.backend}): "
                  f"range labels on.")
            return reader
        except Exception as exc:  # noqa: BLE001
            print(f"[dashboard] IMU not started ({exc}); range labels stay off.")
            return None

    def _load_triplet(self, target: Triplet) -> None:
        self._cancel_tick()
        self.playing = False
        self.btn_play.configure(text="▶")
        self.triplet = target
        # Recorded-IMU replay: a clip with an imu_<ts>.csv sidecar (quad
        # captures since 2026-07-08) replays its attitude log, preferred over
        # the live reader: the replayed attitude matches the footage. Falls
        # back to the live reader (or None) exactly as before.
        self._imu_replay = None
        try:
            from scripts.sensor_processing.imu_bno085 import load_imu_config
            from scripts.sensor_processing.imu_replay import (
                ImuLogAttitudeProvider,
            )
            self._imu_replay = ImuLogAttitudeProvider.for_triplet(
                target, (load_imu_config().get("replay", {}) or {}))
        except Exception as exc:  # noqa: BLE001 - replay must never break loading
            print(f"[dashboard] IMU replay unavailable ({exc})")
        attitude = self._imu_replay if self._imu_replay is not None else self._imu
        self.pipeline = ObstacleDetectionPipeline(self.intrinsics,
                                                  self.detection,
                                                  attitude_provider=attitude,
                                                  seg_provider=self._seg_provider)
        n_bins = len(self.bin_centers)
        track_cfg = self.detection.get("tracker", {})
        self.tracker = SectorTracker(
            n_bins=n_bins,
            min_hits=int(track_cfg.get("min_hits", 3)),
            max_age=int(track_cfg.get("max_age", 5)),
        )
        self.frame_total = _count_frames(target)
        self.cached = []
        self.heading_history.clear()
        self.smoothed_history.clear()
        self.heading_smoother.reset()
        self.freespace_smoother.reset()
        self.last_heading = None
        self.idx = 0
        if self._gen is not None:
            self._gen.close()
        self._gen = iterate_triplet(target, self.detection)
        max_idx = max((self.frame_total if self.frame_total > 0 else 1) - 1, 0)
        self.scrub.configure(to=max_idx)
        self.scrub_var.set(0.0)
        self._refresh_metadata(target)
        # Top-bar label mirrors the path relative to data/ so the user
        # always knows what clip the transport controls operate on.
        rel = target.fisheye.parent.relative_to(REPO_ROOT_FOR_LABELS / "data")
        self.lbl_clip.configure(
            text=f"data/{rel}/{target.timestamp}", foreground="#222",
        )
        # Highlight the loaded leaf in the Clips tree so the operator
        # can see where they are when they next come back to browse.
        if hasattr(self, "tree_clips"):
            self._reveal_triplet_in_tree(target)
        if self._advance_to(0):
            self._status(f"loaded {target.clip_id} ({self.frame_total} frames)")
        else:
            self._status(f"{target.clip_id} produced no frames")

    def _refresh_metadata(self, triplet: Triplet):
        overrides = get_for_clip(self.all_overrides, triplet.clip_id)
        lines = [
            f"scene   {triplet.scene}",
            f"clip    {triplet.timestamp}",
            f"frames  {self.frame_total if self.frame_total >= 0 else '?'}",
            f"over    {_summarise_overrides(overrides)}",
            f"extr    measured={self.extrinsics.get('measured', False)}",
            f"det.yml {_short_hash(Path(self.detection_path or DEFAULT_DETECTION))}",
        ]
        if getattr(self, "_imu_replay", None) is not None:
            lines.append(f"imu     replay ({len(self._imu_replay)} samples)")
        elif self._imu is not None:
            lines.append("imu     live")
        self.lbl_meta.configure(text="\n".join(lines))

    def _free_space_dist(self, res) -> Optional[list]:
        """Per-bin segmentation free-space profile for the heading recommender,
        or None when disabled / no mask. Behind detection.yaml
        `heading.use_free_space` (default off): Phase 1 segmentation-driven nav.

        Bins are the fusion bins, so the returned list aligns with
        `self.bin_centers` that the recommender consumes.
        """
        # Live checkbox (seeded from detection.yaml heading.use_free_space).
        if not self.var_freespace_heading.get():
            return None
        fish = getattr(res, "fisheye", None)
        seg = getattr(fish, "seg_mask", None) if fish is not None else None
        if seg is None:
            return None
        from scripts.utils.geometry import UP_LEVEL, up_from_horizon_line
        from scripts.utils.segmentation import free_space_profile, horizon_from_water

        intr = self.intrinsics["fisheye"]
        K = np.asarray(intr["K"], float)
        cx, pix_deg = float(intr["cx"]), float(intr["pix_deg_ratio"])
        height = float((self.extrinsics.get("camera_height_m", {}) or {})
                       .get("fisheye", 0.27))
        f = self.detection["fusion"]
        edges = np.arange(f["bin_min_deg"], f["bin_max_deg"] + f["bin_step_deg"],
                          f["bin_step_deg"])
        max_range = float((self.detection.get("range", {}) or {}).get("max_range_m", 15.0))
        s, b, conf = horizon_from_water(seg)
        up = up_from_horizon_line(s, b, K) if conf > 0.3 else UP_LEVEL
        prof = free_space_profile(seg, K, up, height, cx, pix_deg, edges,
                                  max_range_m=max_range)
        return prof.free_dist_m

    def _smoothed_free_dist(self, res) -> Optional[list]:
        """Per-frame free-space distance, temporally smoothed (sequential).
        None when free-space heading is off / no mask (smoother not advanced)."""
        fd = self._free_space_dist(res)
        if fd is None:
            return None
        return self.freespace_smoother.update(fd)

    # -- Playback -----------------------------------------------------------

    def _ensure_cached_up_to(self, target_idx: int) -> bool:
        while len(self.cached) <= target_idx:
            if self._gen is None:
                return False
            try:
                ts, fish, therm, mm_pts = next(self._gen)
            except StopIteration:
                self._gen = None
                return False
            fid = _frame_id_for(self.triplet.scene, self.triplet.timestamp, ts)
            if self._imu_replay is not None:
                self._imu_replay.set_time(ts)
            res = self.pipeline.process_frame(fish, therm, mm_pts,
                                              timestamp=ts, frame_id=fid)
            bin_hits = [bool(res.fusion.sensor_hit_mask[i].any())
                        for i in range(len(self.bin_centers))]
            confirmed = list(self.tracker.update(bin_hits))
            recommendation = _compute_heading_for_frame(
                res, confirmed, self.bin_centers, self.detection,
                use_tracker_gate=self.var_tracker.get(),
                current_heading_deg=0.0,   # IMU stub: PLAN.md §IV.1 #14
                free_dist_m=self._smoothed_free_dist(res),
            )
            smoothed = self.heading_smoother.update(
                recommendation.recommended_heading_deg
            )
            self.cached.append((ts, fish, therm, mm_pts, res, confirmed,
                                recommendation, smoothed))
        return True

    def _advance_to(self, idx: int) -> bool:
        if idx < 0:
            idx = 0
        if not self._ensure_cached_up_to(idx):
            if not self.cached:
                return False
            idx = len(self.cached) - 1
        self.idx = idx
        # Histories follow the playhead: scrubbing backward shrinks the
        # sparkline to "as of this frame", not the longest-seen history.
        # Cheap to rebuild: the deques cap at 300 entries anyway.
        self._rebuild_heading_histories_from_cache(idx)
        # Swap the labelling tab over to this frame's saved record (or a
        # blank seed). The record is what `_render` will overlay onto the
        # fisheye, so this has to happen *before* the render call.
        self._activate_label_for_current_frame()
        self._render(*self.cached[idx])
        return True

    def _on_freespace_heading_toggle(self) -> None:
        """Live A/B of segmentation free-space in the heading: recompute the
        cached recommendations in place (no pipeline re-run) and redraw."""
        self._recompute_headings()
        self._status(f"free-space heading: {'ON' if self.var_freespace_heading.get() else 'OFF'}")

    def _recompute_headings(self) -> None:
        """Recompute every cached frame's heading from its cached pipeline
        result (cheap: reuses `res` + tracker `confirmed`), replaying the
        smoother so the sparkline stays consistent. Used by the free-space
        toggle so flipping it is instant."""
        if not self.cached:
            return
        self.heading_smoother.reset()
        self.freespace_smoother.reset()
        new_cached = []
        for (ts, fish, therm, mm, res, confirmed, _rec, _sm) in self.cached:
            rec = _compute_heading_for_frame(
                res, confirmed, self.bin_centers, self.detection,
                use_tracker_gate=self.var_tracker.get(),
                current_heading_deg=0.0,
                free_dist_m=self._smoothed_free_dist(res),
            )
            sm = self.heading_smoother.update(rec.recommended_heading_deg)
            new_cached.append((ts, fish, therm, mm, res, confirmed, rec, sm))
        self.cached = new_cached
        self._rebuild_heading_histories_from_cache(self.idx)
        self._render(*self.cached[self.idx])

    def _rebuild_heading_histories_from_cache(self, idx: int) -> None:
        self.heading_history.clear()
        self.smoothed_history.clear()
        lo = max(0, idx + 1 - self.heading_history.maxlen)
        for entry in self.cached[lo: idx + 1]:
            _, _, _, _, _, _, head, smoothed = entry
            self.heading_history.append(head.recommended_heading_deg)
            self.smoothed_history.append(smoothed)
        # Keep last_heading in sync with the playhead too.
        if self.cached:
            self.last_heading = self.cached[idx][6]
        else:
            self.last_heading = None

    def _step(self, delta: int):
        self._cancel_tick()
        self.playing = False
        self.btn_play.configure(text="▶")
        if self.triplet is None:
            return
        self._advance_to(self.idx + delta)

    def _jump_seconds(self, delta_sec: float):
        """Seek forward/back by wall-clock seconds, using radar timestamps."""
        self._cancel_tick()
        self.playing = False
        self.btn_play.configure(text="▶")
        if self.triplet is None or not self.cached:
            return
        cur_sec = _parse_ts_seconds(self.cached[self.idx][0])
        if cur_sec is None:
            return
        target_sec = cur_sec + delta_sec
        if delta_sec < 0:
            best = 0
            for i in range(self.idx, -1, -1):
                ts_i = _parse_ts_seconds(self.cached[i][0])
                if ts_i is not None and ts_i <= target_sec:
                    best = i
                    break
            self._advance_to(best)
        else:
            i = self.idx
            while True:
                ts_i = _parse_ts_seconds(self.cached[i][0])
                if ts_i is not None and ts_i >= target_sec:
                    break
                if not self._ensure_cached_up_to(i + 1):
                    break
                if i + 1 >= len(self.cached):
                    break
                i += 1
            self._advance_to(i)

    def _on_scrub_press(self, _event):
        self._scrub_user = True
        self._scrub_resume = self.playing
        if self.playing:
            self.playing = False
            self.btn_play.configure(text="▶")
            self._cancel_tick()

    def _on_scrub_command(self, _val):
        if not self._scrub_user:
            return
        v = int(round(float(self.scrub_var.get())))
        total = (self.frame_total if self.frame_total > 0
                 else max(len(self.cached), 1))
        self.lbl_frame.configure(text=f"frame {v + 1}/{total}")

    def _on_scrub_release(self, _event):
        if not self._scrub_user:
            return
        self._scrub_user = False
        if self.triplet is None:
            return
        target = int(round(float(self.scrub_var.get())))
        self._advance_to(target)
        if self._scrub_resume:
            self._scrub_resume = False
            self.playing = True
            self.btn_play.configure(text="⏸")
            self._schedule_tick()

    def _toggle_play(self):
        if self.triplet is None:
            return
        self.playing = not self.playing
        self.btn_play.configure(text="⏸" if self.playing else "▶")
        if self.playing:
            self._schedule_tick()
        else:
            self._cancel_tick()

    def _schedule_tick(self):
        self._tick_handle = self.root.after(self.speed_ms, self._tick)

    def _cancel_tick(self):
        if self._tick_handle is not None:
            try:
                self.root.after_cancel(self._tick_handle)
            except tk.TclError:
                pass
            self._tick_handle = None

    def _tick(self):
        self._tick_handle = None
        if not self.playing:
            return
        target = self.idx + 1
        if not self._advance_to(target) or self.idx < target:
            # End of clip: clamp + stop playback.
            self.playing = False
            self.btn_play.configure(text="▶")
            return
        self._schedule_tick()

    def _on_speed_change(self, event=None):
        label = self.cb_speed.get()
        for tag, ms in SPEEDS:
            if tag == label:
                self.speed_ms = ms
                break
        if self.playing:
            self._cancel_tick()
            self._schedule_tick()

    def _on_threshold(self, _value):
        v = float(self.var_threshold.get())
        self.lbl_thr.configure(text=f"{v:.2f}")
        if self.threshold_line is not None:
            self.threshold_line.set_ydata([v, v])
        self.canvas.draw_idle()

    # -- Rendering ----------------------------------------------------------

    def _rerender(self):
        if not self.cached:
            return
        self._draw_wedges()
        self._render(*self.cached[self.idx])

    @staticmethod
    def _set_image(ax, im, rgb: np.ndarray) -> None:
        """Update an imshow artist with a new frame and resync its extent.

        Necessary because matplotlib keeps the original imshow extent
        (set from the placeholder size at construction) when set_data is
        called with a differently sized image. Without this, the new
        frame is drawn at the old extent and our set_xlim/set_ylim
        push the view far outside the image; it renders as a smudge.
        """
        h, w = rgb.shape[:2]
        im.set_data(rgb)
        im.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal", adjustable="box")

    def _render(self, ts, fish_bgr, therm_bgr, mm_pts, res: FrameResult,
                confirmed: list[bool],
                recommendation: HeadingRecommendation,
                smoothed: Optional[float]):
        radar_pts = res.mmwave.points_xyz if (
            res.mmwave is not None and len(res.mmwave.points_xyz)
        ) else np.empty((0, 3))
        ranges = radar_pts[:, 1] if len(radar_pts) else None

        # Per-frame attitude readout in the fisheye panel title (replay
        # attitude follows the playhead; live attitude is whatever's newest).
        if self.ax_fish is not None:
            att = None
            if getattr(self, "_imu_replay", None) is not None:
                self._imu_replay.set_time(ts)
                att = self._imu_replay.get()
            elif self._imu is not None:
                att = self._imu.get()
            title = "Fisheye RGB"
            if att is not None:
                title += (f"   ·   IMU roll {att.roll_deg:+.1f}°  "
                          f"pitch {att.pitch_deg:+.1f}°  yaw {att.yaw_deg:+.0f}°")
            self.ax_fish.set_title(title, fontsize=10)

        # Fisheye + thermal overlays: skip the projection compute entirely
        # if neither camera panel is currently visible.
        fish_proj = thermal_proj = None
        cameras_visible = (self.ax_fish is not None
                           or self.ax_therm is not None)
        if cameras_visible and len(radar_pts) and self.var_projection.get():
            fish_proj = project_radar_to_fisheye(
                radar_pts,
                self.intrinsics["fisheye"]["K"],
                self.intrinsics["fisheye"]["D"],
                self.extrinsics["T_radar_to_fisheye"],
            )
            thermal_proj = project_radar_to_thermal(
                radar_pts,
                self.intrinsics["thermal"]["K"],
                self.intrinsics["thermal"]["D"],
                self.extrinsics["T_radar_to_thermal"],
            )
        # Label patches sit on the same axis as the fisheye image, so they
        # have to be cleared (and redrawn against the new image extent)
        # every render: otherwise stale artists survive scrubbing.
        self._clear_label_artists()
        if self.ax_fish is not None and self.im_fish is not None:
            typed_dets = instance_dets = None
            if ((self._det_provider is not None and self.var_typed.get())
                    or (self._inst_provider is not None and self.var_instances.get())):
                fid = _frame_id_for(self.triplet.scene, self.triplet.timestamp, ts)
                if self._det_provider is not None and self.var_typed.get():
                    typed_dets = self._det_provider.get(fid)
                if self._inst_provider is not None and self.var_instances.get():
                    instance_dets = self._inst_provider.get(fid)
            fish_rgb = _overlay_camera(
                res.fisheye, projected=fish_proj, ranges=ranges,
                show_horizon=self.var_horizon.get(),
                show_projection=self.var_projection.get(),
                show_detections=self.var_detections.get(),
                show_seg=self.var_seg.get(),
                typed_dets=typed_dets,
                instance_dets=instance_dets,
            )
            self._set_image(self.ax_fish, self.im_fish, fish_rgb)
            self._draw_label_overlay()
        if self.ax_therm is not None and self.im_therm is not None:
            therm_rgb = _overlay_camera(
                res.thermal, projected=thermal_proj, ranges=ranges,
                show_horizon=self.var_horizon.get(),
                show_projection=self.var_projection.get(),
                show_detections=self.var_detections.get(),
            )
            self._set_image(self.ax_therm, self.im_therm, therm_rgb)

        # Radar.
        if self.radar_scat is not None:
            if len(radar_pts):
                self.radar_scat.set_offsets(
                    np.column_stack((radar_pts[:, 0], radar_pts[:, 1]))
                )
                self.radar_scat.set_array(radar_pts[:, 1])
            else:
                self.radar_scat.set_offsets(np.empty((0, 2)))
                self.radar_scat.set_array(np.array([]))

        # Bars.
        threshold = float(self.var_threshold.get())
        for bar, score, hits, conf in zip(self.bars, res.fusion.scores,
                                          res.fusion.sensor_hit_mask,
                                          confirmed):
            bar.set_height(score)
            n_hits = int(hits.sum())
            base = SENSOR_COLOURS[n_hits]
            bar.set_color(base)
            if self.var_tracker.get() and conf:
                bar.set_edgecolor("#1c4587")
                bar.set_linewidth(2.0)
            else:
                bar.set_edgecolor("none")
                bar.set_linewidth(0.0)

        self.canvas.draw_idle()

        # Heading recommendation + smoothed value were computed once
        # at production time and live on the cached tuple: see
        # _ensure_cached_up_to. _render is a pure read: no smoother
        # advances, no history appends.
        self.last_heading = recommendation

        # Sidebar widgets.
        self._update_health(res)
        self._update_bins_table(res, confirmed, threshold)
        self._update_heading(recommendation)
        self._update_transport(ts)

    def _update_health(self, res: FrameResult):
        fish = res.fisheye
        therm = res.thermal
        mm = res.mmwave

        if fish is None:
            self.lbl_health["fisheye"].configure(text="Fisheye  ● DOWN",
                                                 foreground="#cc0000")
        else:
            conf = fish.horizon_line[2]
            self.lbl_health["fisheye"].configure(
                text=f"Fisheye  ● horizon {conf:.2f} · {len(fish.coords)} blobs",
                foreground="#274e13",
            )
        if therm is None:
            self.lbl_health["thermal"].configure(text="Thermal  ● DOWN",
                                                 foreground="#cc0000")
        else:
            avg = therm.new_average
            self.lbl_health["thermal"].configure(
                text=f"Thermal  ● avg {avg:5.1f} · {len(therm.coords)} blobs",
                foreground="#274e13",
            )
        if mm is None:
            self.lbl_health["mmwave"].configure(text="mmWave   ● DOWN",
                                                foreground="#cc0000")
        else:
            kept = len(mm.points_xyz)
            self.lbl_health["mmwave"].configure(
                text=f"mmWave   ● {kept} pts kept (after Y filter)",
                foreground="#274e13",
            )

    def _update_bins_table(self, res: FrameResult, confirmed: list[bool],
                           threshold: float):
        self.tree.delete(*self.tree.get_children())
        velocities = res.fusion.per_bin_velocity_mps or [None] * len(self.bin_centers)
        ttcs = res.fusion.per_bin_ttc_s or [None] * len(self.bin_centers)
        for i, (c, score, hits, conf) in enumerate(
            zip(self.bin_centers, res.fusion.scores,
                res.fusion.sensor_hit_mask, confirmed)
        ):
            rng = res.fusion.min_ranges[i]
            rng_s = f"{rng:.1f} m" if rng is not None else "—"
            v = velocities[i]
            v_s = f"{v:+.1f}" if v is not None else "—"
            ttc = ttcs[i]
            ttc_s = f"{ttc:.1f}s" if ttc is not None else "—"
            track_s = "✓" if conf else "—"
            tags: list[str] = []
            if ttc is not None and ttc <= 3.0 and score >= 0.33:
                tags.append("urgent")
            elif score >= threshold * 2 - 1e-6 or score >= 0.66:
                tags.append("blocked")
            elif conf:
                tags.append("confirmed")
            self.tree.insert(
                "", "end",
                text=f"{int(c):+d}°",
                values=(
                    f"{score:.2f}",
                    "✓" if hits[0] else "",
                    "✓" if hits[1] else "",
                    "✓" if hits[2] else "",
                    rng_s,
                    v_s,
                    ttc_s,
                    track_s,
                ),
                tags=tags,
            )

    def _update_heading(self, recommendation: HeadingRecommendation):
        if recommendation.recommended_heading_deg is None:
            self.lbl_heading.configure(
                text=f"recommended:: ({recommendation.reason})",
                foreground="#a00",
            )
        else:
            self.lbl_heading.configure(
                text=f"recommended: {recommendation.recommended_heading_deg:+.0f}°",
                foreground="#274e13",
            )
        terms = recommendation.per_term_at_choice
        if terms:
            top = sorted(terms.items(), key=lambda kv: -kv[1])[:2]
            reason_s = "  ".join(f"{k}={v:.2f}" for k, v in top)
        else:
            reason_s = recommendation.reason
        self.lbl_heading_reason.configure(text=f"reason: {reason_s}")

        # Show the smoothed (steering) value next to the raw one so the
        # operator can see how much the rolling mean is tempering the
        # per-frame jitter.
        smoothed = (self.smoothed_history[-1] if self.smoothed_history
                    else None)
        if smoothed is None:
            self.lbl_heading_smoothed.configure(text="smoothed:: ")
        else:
            self.lbl_heading_smoothed.configure(
                text=f"smoothed (steering): {smoothed:+.0f}°"
            )

        if not self.var_heading.get():
            if self.line_head_curve is not None:
                self.line_head_curve.set_data([], [])
            if self.head_choice_line is not None:
                self.head_choice_line.set_alpha(0.0)
            if self.line_head_hist is not None:
                self.line_head_hist.set_data([], [])
            if self.line_head_hist_smoothed is not None:
                self.line_head_hist_smoothed.set_data([], [])
            return
        if (recommendation.candidate_headings_deg
                and self.line_head_curve is not None
                and self.ax_head_curve is not None):
            xs = recommendation.candidate_headings_deg
            ys = recommendation.cost_curve
            self.line_head_curve.set_data(xs, ys)
            self.ax_head_curve.set_xlim(xs[0], xs[-1])
            if ys:
                self.ax_head_curve.set_ylim(0, max(0.05, max(ys) * 1.15))
            if self.head_choice_line is not None:
                if recommendation.recommended_heading_deg is not None:
                    self.head_choice_line.set_xdata(
                        [recommendation.recommended_heading_deg,
                         recommendation.recommended_heading_deg]
                    )
                    self.head_choice_line.set_alpha(0.9)
                else:
                    self.head_choice_line.set_alpha(0.0)

        raw = [h if h is not None else float("nan")
               for h in self.heading_history]
        smoothed_series = [h if h is not None else float("nan")
                           for h in self.smoothed_history]
        if raw and self.ax_head_hist is not None:
            if self.line_head_hist is not None:
                self.line_head_hist.set_data(range(len(raw)), raw)
            if self.line_head_hist_smoothed is not None:
                self.line_head_hist_smoothed.set_data(
                    range(len(smoothed_series)), smoothed_series,
                )
            self.ax_head_hist.set_xlim(0, max(len(raw), 30))

    def _update_transport(self, ts: str):
        total = (self.frame_total if self.frame_total > 0
                 else len(self.cached))
        self.lbl_frame.configure(text=f"frame {self.idx + 1}/{total}")
        self.lbl_ts.configure(text=f"ts {ts}")
        if not self._scrub_user:
            max_idx = max(total - 1, 0)
            try:
                current_to = int(float(self.scrub.cget("to")))
            except (tk.TclError, ValueError):
                current_to = 0
            if current_to != max_idx:
                self.scrub.configure(to=max_idx)
            self.scrub_var.set(float(self.idx))

    # -- Layout selection + customisation ----------------------------------

    def _all_presets(self) -> dict[str, dict]:
        """Built-in presets first, then user presets. User presets shadow
        built-ins with the same name (rare, but well-defined)."""
        merged: dict[str, dict] = {}
        merged.update(BUILTIN_PRESETS)
        merged.update(self.custom_presets)
        return merged

    def _refresh_layout_combo(self) -> None:
        names = list(self._all_presets().keys())
        # Append CUSTOM_LABEL only when the live layout doesn't match any
        # known preset, so the combobox shows "(custom)" rather than a
        # stale preset name after the user tweaks something.
        self.cb_layout["values"] = names
        match = self._matching_preset_name()
        self.cb_layout.set(match or CUSTOM_LABEL)
        if match is None:
            # Add the synthetic entry so the user can re-select it after
            # picking a real preset without losing their tweaks. We append
            # it to the visible list each time so it always appears last.
            self.cb_layout["values"] = names + [CUSTOM_LABEL]

    def _matching_preset_name(self) -> Optional[str]:
        for name, layout in self._all_presets().items():
            if _layouts_equal(self.layout, layout):
                return name
        return None

    def _on_layout_preset(self, _event=None):
        name = self.cb_layout.get()
        if name == CUSTOM_LABEL:
            return
        layout = self._all_presets().get(name)
        if layout is None:
            return
        self.layout = _normalize_layout(layout)
        self._apply_layout()
        self._persist_current_layout()
        if self.cached:
            self._rerender()
        self._status(f"layout: {name}")

    def _persist_current_layout(self) -> None:
        try:
            _save_layouts_file(self.layout, self.custom_presets)
        except OSError as exc:
            self._status(f"failed to persist layout: {exc}")

    def _save_current_as_preset(self) -> None:
        """Save the live layout (not the dialog draft) under a new name.

        Bound to the chrome button: useful after a sequence of preset
        switches + tweaks where the user wants to bookmark the result
        without re-opening the dialog.
        """
        self._save_preset_from_layout(self.layout)

    def _save_preset_from_layout(self, layout: dict) -> Optional[str]:
        name = simpledialog.askstring(
            "Save layout preset",
            "Preset name:",
            parent=self.root,
        )
        if not name:
            return None
        name = name.strip()
        if not name or name == CUSTOM_LABEL:
            return None
        if name in BUILTIN_PRESETS:
            self._status(f"'{name}' is a built-in preset name: pick another")
            return None
        self.custom_presets[name] = _normalize_layout(layout)
        try:
            _save_layouts_file(self.layout, self.custom_presets)
        except OSError as exc:
            self._status(f"failed to save preset: {exc}")
            return None
        self._refresh_layout_combo()
        self.cb_layout.set(name)
        self._status(f"saved layout preset '{name}'")
        return name

    def _open_layout_dialog(self):
        """Visual layout editor.

        Two-pane Toplevel: the left pane lists the panels (radio selector +
        per-panel visibility toggle), the right pane is a Tk Canvas painted
        with the live LAYOUT_ROWS×LAYOUT_COLS grid. Click+drag on the
        canvas to assign a rectangle to the currently selected panel; any
        previously-visible panel that would overlap the new rectangle is
        auto-hidden so the picker stays conflict-free (each rectangle is
        unique). Toggling a hidden panel back on restores its last position.
        """
        dlg = tk.Toplevel(self.root)
        dlg.title("Customize layout")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        # Working copy: every interaction in the dialog mutates this dict
        # and `_apply_layout` is only called from the Apply button. Cancel
        # leaves self.layout untouched.
        working: dict[str, dict] = {
            k: dict(_clamp_placement(self.layout.get(k, DEFAULT_LAYOUT[k])))
            for k in PANEL_KEYS
        }
        # `last_rect[panel]` remembers the placement so flipping the
        # visibility checkbox off-then-on restores the previous footprint
        # rather than dumping the panel back into (0,0,1,1).
        last_rect: dict[str, dict] = {k: dict(working[k]) for k in PANEL_KEYS}

        outer = ttk.Frame(dlg, padding=(10, 8))
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text=("Pick a panel on the left, then click and drag across "
                  "the cells you want it to occupy.\n"
                  "Drag a 1×1 rectangle for a single cell; drag across "
                  "multiple cells to span them."),
            justify="left",
            foreground="#555",
        ).pack(anchor="w", pady=(0, 8))

        body = ttk.Frame(outer)
        body.pack(fill="x")

        # --- Left: panel selector ---------------------------------------
        sel_frame = ttk.LabelFrame(body, text="Panel", padding=(8, 6))
        sel_frame.pack(side="left", anchor="n", padx=(0, 12))

        selected_var = tk.StringVar(value=PANEL_KEYS[0])
        vis_vars: dict[str, tk.BooleanVar] = {}
        rect_labels: dict[str, ttk.Label] = {}

        def _rect_summary(p: dict) -> str:
            if not p["visible"]:
                return "hidden"
            r0, c0 = p["row"], p["col"]
            r1, c1 = r0 + p["rowspan"] - 1, c0 + p["colspan"] - 1
            if r0 == r1 and c0 == c1:
                return f"cell r{r0}c{c0}"
            return f"r{r0}c{c0} → r{r1}c{c1}"

        for i, key in enumerate(PANEL_KEYS):
            row = ttk.Frame(sel_frame)
            row.grid(row=i, column=0, sticky="ew", pady=1)
            # Coloured swatch so the same colour appears on the canvas.
            sw = tk.Frame(row, width=16, height=16,
                          background=PANEL_COLORS[key],
                          highlightthickness=1,
                          highlightbackground="#888")
            sw.pack(side="left", padx=(0, 6))
            sw.pack_propagate(False)
            ttk.Radiobutton(row, text=PANEL_LABELS[key],
                            value=key,
                            variable=selected_var,
                            command=lambda: _redraw_canvas()).pack(side="left")
            vis = tk.BooleanVar(value=bool(working[key]["visible"]))
            vis_vars[key] = vis

            def _on_vis_toggle(k=key):
                want = vis_vars[k].get()
                if want:
                    # Restore last footprint, evicting any overlap with
                    # the now-revealed rect to keep the picker conflict-free.
                    restored = dict(last_rect[k])
                    restored["visible"] = True
                    working[k] = restored
                    _evict_overlaps(k)
                else:
                    last_rect[k] = dict(working[k])
                    working[k]["visible"] = False
                _redraw_canvas()
                _refresh_rect_labels()

            ttk.Checkbutton(row, text="visible", variable=vis,
                            command=_on_vis_toggle).pack(side="left",
                                                         padx=(8, 0))
            lbl = ttk.Label(row, text="", foreground="#666",
                            font=("TkFixedFont", 9))
            lbl.pack(side="left", padx=(8, 0))
            rect_labels[key] = lbl

        def _refresh_rect_labels() -> None:
            for k, lbl in rect_labels.items():
                lbl.configure(text=_rect_summary(working[k]))

        _refresh_rect_labels()

        # --- Right: visual grid canvas ----------------------------------
        canvas_frame = ttk.LabelFrame(body, text="Layout grid",
                                      padding=(8, 6))
        canvas_frame.pack(side="left", anchor="n")

        cell_w, cell_h = 110, 80
        margin = 6
        canvas_w = LAYOUT_COLS * cell_w + 2 * margin
        canvas_h = LAYOUT_ROWS * cell_h + 2 * margin
        canvas = tk.Canvas(canvas_frame,
                           width=canvas_w, height=canvas_h,
                           background="#fafafa",
                           highlightthickness=0,
                           cursor="crosshair")
        canvas.grid(row=0, column=0)

        drag_state: dict = {"anchor": None, "current": None}

        def _xy_to_cell(x: int, y: int) -> tuple[int, int]:
            c = (x - margin) // cell_w
            r = (y - margin) // cell_h
            c = max(0, min(LAYOUT_COLS - 1, int(c)))
            r = max(0, min(LAYOUT_ROWS - 1, int(r)))
            return r, c

        def _rects_overlap(a: dict, b: dict) -> bool:
            a_r0, a_c0 = a["row"], a["col"]
            a_r1, a_c1 = a_r0 + a["rowspan"], a_c0 + a["colspan"]
            b_r0, b_c0 = b["row"], b["col"]
            b_r1, b_c1 = b_r0 + b["rowspan"], b_c0 + b["colspan"]
            return not (a_r1 <= b_r0 or b_r1 <= a_r0
                        or a_c1 <= b_c0 or b_c1 <= a_c0)

        def _evict_overlaps(claimer: str) -> None:
            c_rect = working[claimer]
            if not c_rect["visible"]:
                return
            for k in PANEL_KEYS:
                if k == claimer:
                    continue
                p = working[k]
                if not p["visible"]:
                    continue
                if _rects_overlap(p, c_rect):
                    last_rect[k] = dict(p)
                    working[k]["visible"] = False
                    vis_vars[k].set(False)

        def _redraw_canvas() -> None:
            canvas.delete("all")
            # Light grid lines first so empty cells still read as a grid.
            # Each cell carries faint r/c coordinates so the user can refer
            # to "row 0 col 1" while dragging without external labels.
            for r in range(LAYOUT_ROWS):
                for c in range(LAYOUT_COLS):
                    x0 = margin + c * cell_w
                    y0 = margin + r * cell_h
                    canvas.create_rectangle(
                        x0, y0, x0 + cell_w, y0 + cell_h,
                        outline="#d0d0d0", width=1, fill="#ffffff",
                    )
                    canvas.create_text(x0 + 6, y0 + 6,
                                       text=f"r{r}c{c}",
                                       anchor="nw",
                                       font=("TkFixedFont", 8),
                                       fill="#bbb")
            # Filled rectangles for each visible panel, drawn in PANEL_KEYS
            # order so later panels overlay earlier ones (overlap is
            # prevented by _evict_overlaps so this only matters if the
            # working layout was loaded externally).
            for key in PANEL_KEYS:
                p = working[key]
                if not p["visible"]:
                    continue
                x0 = margin + p["col"] * cell_w
                y0 = margin + p["row"] * cell_h
                x1 = x0 + p["colspan"] * cell_w
                y1 = y0 + p["rowspan"] * cell_h
                fill = PANEL_COLORS[key]
                is_selected = (key == selected_var.get())
                outline = "#1c4587" if is_selected else "#666"
                width = 3 if is_selected else 1
                canvas.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2,
                                        fill=fill, outline=outline,
                                        width=width)
                canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2,
                                   text=PANEL_LABELS[key],
                                   font=("TkDefaultFont", 10, "bold"),
                                   fill="#222")
            # In-flight drag highlight on top of everything else.
            a = drag_state["anchor"]
            b = drag_state["current"]
            if a is not None and b is not None:
                r0, r1 = sorted((a[0], b[0]))
                c0, c1 = sorted((a[1], b[1]))
                x0 = margin + c0 * cell_w
                y0 = margin + r0 * cell_h
                x1 = margin + (c1 + 1) * cell_w
                y1 = margin + (r1 + 1) * cell_h
                canvas.create_rectangle(
                    x0 + 1, y0 + 1, x1 - 1, y1 - 1,
                    outline="#0b5394", width=3, dash=(4, 3),
                )

        def _on_press(event):
            cell = _xy_to_cell(event.x, event.y)
            drag_state["anchor"] = cell
            drag_state["current"] = cell
            _redraw_canvas()

        def _on_motion(event):
            if drag_state["anchor"] is None:
                return
            drag_state["current"] = _xy_to_cell(event.x, event.y)
            _redraw_canvas()

        def _on_release(event):
            a = drag_state["anchor"]
            if a is None:
                return
            b = _xy_to_cell(event.x, event.y)
            r0, r1 = sorted((a[0], b[0]))
            c0, c1 = sorted((a[1], b[1]))
            panel = selected_var.get()
            working[panel] = {
                "row": r0, "col": c0,
                "rowspan": r1 - r0 + 1,
                "colspan": c1 - c0 + 1,
                "visible": True,
            }
            last_rect[panel] = dict(working[panel])
            vis_vars[panel].set(True)
            _evict_overlaps(panel)
            drag_state["anchor"] = None
            drag_state["current"] = None
            _redraw_canvas()
            _refresh_rect_labels()

        canvas.bind("<ButtonPress-1>", _on_press)
        canvas.bind("<B1-Motion>", _on_motion)
        canvas.bind("<ButtonRelease-1>", _on_release)

        _redraw_canvas()

        # --- Preset load row --------------------------------------------
        load_row = ttk.Frame(outer)
        load_row.pack(fill="x", pady=(10, 0))
        ttk.Label(load_row, text="Load from preset:").pack(side="left")
        preset_picker = ttk.Combobox(
            load_row, values=list(self._all_presets().keys()),
            state="readonly", width=24,
        )
        preset_picker.pack(side="left", padx=(4, 4))

        def _do_load(_event=None) -> None:
            name = preset_picker.get()
            layout = self._all_presets().get(name)
            if layout is None:
                return
            for key in PANEL_KEYS:
                p = _clamp_placement(layout.get(key, DEFAULT_LAYOUT[key]))
                working[key] = dict(p)
                last_rect[key] = dict(p)
                vis_vars[key].set(bool(p["visible"]))
            _redraw_canvas()
            _refresh_rect_labels()

        ttk.Button(load_row, text="Load",
                   command=_do_load).pack(side="left")
        preset_picker.bind("<<ComboboxSelected>>", _do_load)

        # --- Action buttons ---------------------------------------------
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill="x", pady=(10, 0))

        def _do_apply(close: bool) -> None:
            self.layout = _normalize_layout(working)
            self._apply_layout()
            self._persist_current_layout()
            self._refresh_layout_combo()
            if self.cached:
                self._rerender()
            self._status("layout applied")
            if close:
                dlg.destroy()

        def _do_reset() -> None:
            for key in PANEL_KEYS:
                p = dict(DEFAULT_LAYOUT[key])
                working[key] = dict(p)
                last_rect[key] = dict(p)
                vis_vars[key].set(bool(p["visible"]))
            _redraw_canvas()
            _refresh_rect_labels()

        def _do_save_preset() -> None:
            saved = self._save_preset_from_layout(_normalize_layout(working))
            if saved is not None:
                _do_apply(close=False)

        ttk.Button(btn_row, text="Reset to default",
                   command=_do_reset).pack(side="left")
        ttk.Button(btn_row, text="Cancel",
                   command=dlg.destroy).pack(side="right", padx=(4, 0))
        ttk.Button(btn_row, text="Apply",
                   command=lambda: _do_apply(close=True)).pack(
                       side="right",
                   )
        ttk.Button(btn_row, text="Save as preset…",
                   command=_do_save_preset).pack(side="right", padx=(0, 4))

    # -- Labelling ---------------------------------------------------------

    @staticmethod
    def _empty_label_record() -> dict:
        return {
            "frame_id": None,
            "scene": None,
            "triplet_ts": None,
            "frame_ts": None,
            "source": "dashboard-manual",
            "audited": True,
            "fisheye_bboxes": [],
            "obstacle_bins_fisheye": [],
            "width": 0,
            "height": 0,
            # horizon_line_label optional: only present when set.
        }

    def _build_label_tab(self, parent: ttk.Frame) -> None:
        """Sidebar contents for the Labelling tab.

        Mirrors the existing detection tab's vertical layout: each section
        is its own LabelFrame, top-to-bottom in importance order (tool /
        class / bboxes / horizon / nav / save status). All edits autosave
        via `_label_autosave`.
        """
        tool_lf = ttk.LabelFrame(parent, text="Annotation tool",
                                 padding=(6, 4))
        tool_lf.pack(fill="x", pady=(0, 6))
        self.var_label_tool = tk.StringVar(value=self.label_tool)
        for label, value in (
            ("Off (no editing)", LABEL_TOOL_NONE),
            ("Bbox: click-drag on fisheye", LABEL_TOOL_BBOX),
            ("Horizon: 2 clicks", LABEL_TOOL_HORIZON),
        ):
            ttk.Radiobutton(tool_lf, text=label, value=value,
                            variable=self.var_label_tool,
                            command=self._on_label_tool_changed).pack(
                anchor="w",
            )

        cls_lf = ttk.LabelFrame(parent, text="New bbox class",
                                padding=(6, 4))
        cls_lf.pack(fill="x", pady=(0, 6))
        cls_row = ttk.Frame(cls_lf)
        cls_row.pack(fill="x")
        self.var_label_class = tk.StringVar(value=self.label_class)
        self.cb_label_class = ttk.Combobox(
            cls_row, values=list(LABEL_CLASSES),
            textvariable=self.var_label_class,
            state="readonly", width=14,
        )
        self.cb_label_class.pack(side="left")
        self.cb_label_class.bind(
            "<<ComboboxSelected>>",
            lambda _e: setattr(self, "label_class", self.var_label_class.get()),
        )
        ttk.Label(cls_row, text="Applied to newly drawn boxes.",
                  foreground="#666", font=("TkDefaultFont", 9)).pack(
            side="left", padx=(8, 0),
        )
        # Keyboard shortcuts from label_tool.py, reproduced here so the
        # operator doesn't have to remember them. The keys apply to the
        # selected bbox (or the most recent one) AND update the dropdown
        # above so the next draw uses the same class.
        ttk.Label(
            cls_lf,
            text=("Keys:  b=boat   d=duck   B=buoy   p=person\n"
                  "       s=structure   o=other   u=delete last bbox"),
            foreground="#666", font=("TkFixedFont", 9),
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        box_lf = ttk.LabelFrame(parent, text="Bboxes on this frame",
                                padding=(4, 4))
        box_lf.pack(fill="both", expand=True, pady=(0, 6))
        cols = ("cls", "x", "y", "w", "h")
        self.tree_bboxes = ttk.Treeview(
            box_lf, columns=cols, show="headings", height=6, selectmode="browse",
        )
        widths = {"cls": 80, "x": 50, "y": 50, "w": 50, "h": 50}
        for c in cols:
            self.tree_bboxes.heading(c, text=c)
            self.tree_bboxes.column(c, width=widths[c], anchor="center")
        self.tree_bboxes.pack(fill="both", expand=True, side="top")
        self.tree_bboxes.bind("<<TreeviewSelect>>", self._on_bbox_selected)
        self.tree_bboxes.bind("<Delete>", self._on_bbox_delete)
        self.tree_bboxes.bind("<BackSpace>", self._on_bbox_delete)

        # Inline class editor for the selected bbox. The Combobox stays
        # disabled when nothing is selected so it can't be set to a stale
        # class; selecting a bbox enables it and copies in the current
        # class so changing the value is a single click.
        edit_row = ttk.Frame(box_lf)
        edit_row.pack(fill="x", pady=(4, 0))
        ttk.Label(edit_row, text="Selected class:").pack(side="left")
        self.var_bbox_class = tk.StringVar(value="")
        self.cb_bbox_class = ttk.Combobox(
            edit_row, values=list(LABEL_CLASSES),
            textvariable=self.var_bbox_class,
            state="disabled", width=12,
        )
        self.cb_bbox_class.pack(side="left", padx=(4, 0))
        self.cb_bbox_class.bind(
            "<<ComboboxSelected>>", self._on_bbox_class_changed,
        )

        bbox_btns = ttk.Frame(box_lf)
        bbox_btns.pack(fill="x", pady=(4, 0))
        ttk.Button(bbox_btns, text="Delete selected",
                   command=self._delete_selected_bbox).pack(side="left")
        ttk.Button(bbox_btns, text="Clear all",
                   command=self._clear_all_bboxes).pack(side="right")

        hor_lf = ttk.LabelFrame(parent, text="Horizon line label",
                                padding=(6, 4))
        hor_lf.pack(fill="x", pady=(0, 6))
        self.lbl_horizon_status = ttk.Label(
            hor_lf, text="not set", anchor="w",
            font=("TkFixedFont", 10),
        )
        self.lbl_horizon_status.pack(fill="x")
        hor_btns = ttk.Frame(hor_lf)
        hor_btns.pack(fill="x", pady=(2, 0))
        ttk.Button(hor_btns, text="Clear",
                   command=self._clear_horizon_label).pack(side="left")
        ttk.Button(hor_btns, text="Cancel pending click",
                   command=self._cancel_horizon_pending).pack(side="left",
                                                               padx=(4, 0))

        nav_lf = ttk.LabelFrame(parent, text="Quick nav",
                                padding=(6, 4))
        nav_lf.pack(fill="x", pady=(0, 6))

        # Forward/backward step-by-N buttons, listed first because this is
        # the most common navigation action while labelling. Common stride
        # values + a freeform spinbox cover the long tail.
        ttk.Label(nav_lf, text="Step by N frames",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        # Grid with weighted columns so the buttons share the sidebar width
        # and never clip (they used to overflow at the narrow sidebar width).
        steps = ttk.Frame(nav_lf)
        steps.pack(fill="x", pady=(2, 0))
        for c in range(6):
            steps.columnconfigure(c, weight=1, uniform="nav")
        ttk.Label(steps, text="fwd", foreground="#666").grid(
            row=0, column=0, sticky="w")
        for i, n in enumerate((1, 5, 10, 30, 100), start=1):
            ttk.Button(steps, text=f"+{n}",
                       command=lambda n=n: self._step(+n)).grid(
                row=0, column=i, sticky="ew", padx=1, pady=1)
        ttk.Label(steps, text="back", foreground="#666").grid(
            row=1, column=0, sticky="w")
        for i, n in enumerate((1, 5, 10, 30, 100), start=1):
            ttk.Button(steps, text=f"-{n}",
                       command=lambda n=n: self._step(-n)).grid(
                row=1, column=i, sticky="ew", padx=1, pady=1)
        custom_row = ttk.Frame(nav_lf)
        custom_row.pack(fill="x", pady=(4, 6))
        custom_row.columnconfigure(1, weight=1)
        ttk.Label(custom_row, text="custom", foreground="#666").grid(
            row=0, column=0, sticky="w", padx=(0, 4))
        self.var_step_n = tk.IntVar(value=1)
        ttk.Spinbox(custom_row, from_=1, to=10000, width=6,
                    textvariable=self.var_step_n).grid(
            row=0, column=1, sticky="ew")
        ttk.Button(custom_row, text="◀ N", width=4,
                   command=lambda: self._step_custom(-1)).grid(
            row=0, column=2, padx=2)
        ttk.Button(custom_row, text="N ▶", width=4,
                   command=lambda: self._step_custom(+1)).grid(
            row=0, column=3)

        # Existing label-aware nav lives below the step buttons since it's
        # used less often during heads-down annotation work.
        ttk.Separator(nav_lf, orient="horizontal").pack(fill="x",
                                                        pady=(4, 4))
        ttk.Label(nav_lf, text="Label-aware nav",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        nav_row = ttk.Frame(nav_lf)
        nav_row.pack(fill="x", pady=(2, 0))
        ttk.Button(nav_row, text="◀ prev labelled",
                   command=lambda: self._jump_labelled(-1)).pack(side="left")
        ttk.Button(nav_row, text="next labelled ▶",
                   command=lambda: self._jump_labelled(+1)).pack(side="left",
                                                                  padx=(4, 0))
        ttk.Button(nav_row, text="next unlabelled →",
                   command=self._jump_next_unlabelled).pack(side="left",
                                                            padx=(8, 0))

        show_lf = ttk.LabelFrame(parent, text="Display",
                                 padding=(6, 4))
        show_lf.pack(fill="x", pady=(0, 6))
        # var_show_labels is created in __init__ (shared with the main
        # show/no-show panel); just reflect its initial state here.
        self.var_show_labels.set(self.label_show_existing)
        ttk.Checkbutton(
            show_lf, text="Show labels overlaid on fisheye",
            variable=self.var_show_labels,
            command=self._on_show_labels_toggle,
        ).pack(anchor="w")

        save_lf = ttk.LabelFrame(parent, text="Save status",
                                 padding=(6, 4))
        save_lf.pack(fill="x")
        self.lbl_save_status = ttk.Label(
            save_lf, text=f"file: {self.label_store.path}",
            foreground="#666",
            font=("TkFixedFont", 9),
        )
        self.lbl_save_status.pack(fill="x")
        self.lbl_save_count = ttk.Label(
            save_lf, text="—", anchor="w",
            font=("TkFixedFont", 10),
        )
        self.lbl_save_count.pack(fill="x")
        self._refresh_label_summary()

        # Audit mode: accept the (reviewed) detector suggestion on this frame
        # into the manual set, with a n/target counter.
        if self._pseudo_mode:
            self.btn_add_manual = ttk.Button(
                save_lf, text="✓ Add to manual set (a)",
                command=self._on_add_to_manual)
            self.btn_add_manual.pack(fill="x", pady=(6, 0))
            ttk.Label(
                save_lf,
                text=("Detector boxes are pre-filled + editable. Review, fix, "
                      "then Add (key 'a') to accept into the manual set and "
                      "advance. Eval-clip frames are skipped."),
                foreground="#666", font=("TkFixedFont", 8),
                wraplength=240, justify="left").pack(fill="x", pady=(2, 0))
            self._refresh_added_counter()

    # -- Tab change ---------------------------------------------------------

    def _on_sidebar_tab_changed(self, _event=None) -> None:
        """Trigger a re-render whenever the user flips tabs.

        The labelling tab implies showing existing labels on the fisheye
        even when the user isn't actively drawing, and the detection tab
        wants them gone (or at least not steering attention), so a quick
        re-render keeps the overlays in sync.
        """
        if not self.cached:
            return
        self._rerender()

    def _label_tab_active(self) -> bool:
        # Compare the selected widget against self.tab_label rather than
        # an index: the index broke when the Clips tab was added in
        # front of Detection, so the gates silently disabled bbox/horizon
        # drawing and the b/d/B/p/o shortcuts.
        try:
            return self.nb_sidebar.select() == str(self.tab_label)
        except (tk.TclError, AttributeError):
            return False

    # -- State refreshers / autosave --------------------------------------

    def _refresh_label_summary(self) -> None:
        n = len(self.label_store.records)
        self.lbl_save_count.configure(
            text=f"{n} frame{'s' if n != 1 else ''} labelled across all clips"
        )

    def _accepted_count(self) -> int:
        """Manual-set frames for the current mission (excludes eval-clip ids)."""
        triplet = getattr(self, "triplet", None)
        scene = triplet.scene if triplet else None
        return sum(1 for fid, r in self.label_store.records.items()
                   if r.get("scene") == scene and fid not in self._exclude)

    def _refresh_added_counter(self) -> None:
        if not getattr(self, "_pseudo_mode", False):
            return
        if not hasattr(self, "btn_add_manual"):
            return
        n = self._accepted_count()
        mark = " ✓ added" if self._frame_committed else ""
        self.btn_add_manual.configure(
            text=f"✓ Add to manual set: {n}/{self._target} (a){mark}")

    def _on_add_to_manual(self) -> None:
        """Accept the current (reviewed) frame into the manual set + advance."""
        fid = self.current_label.get("frame_id")
        if not fid:
            return
        if fid in self._exclude:
            self._status("frame is in the eval set, not added (avoids leakage)")
            return
        if not self.current_label.get("fisheye_bboxes"):
            self._status("no boxes on this frame: nothing to add (delete-free)")
            return
        self.current_label["source"] = "dashboard-manual"
        self.current_label["audited"] = True
        self._frame_committed = True
        self._label_autosave()          # now persists to manual.jsonl
        self._refresh_added_counter()
        n = self._accepted_count()
        self._status(f"added to manual set: {n}/{self._target}")
        self._step(+1)                  # advance to the next frame to review

    def _refresh_bbox_tree(self) -> None:
        self.tree_bboxes.delete(*self.tree_bboxes.get_children())
        w = max(self.current_label.get("width", 1), 1)
        h = max(self.current_label.get("height", 1), 1)
        for i, bb in enumerate(self.current_label.get("fisheye_bboxes", [])):
            x0, y0, x1, y1 = bb["xyxy"]
            self.tree_bboxes.insert(
                "", "end", iid=str(i),
                values=(
                    bb.get("cls", "unset"),
                    f"{int(x0 * w)}",
                    f"{int(y0 * h)}",
                    f"{int((x1 - x0) * w)}",
                    f"{int((y1 - y0) * h)}",
                ),
            )

    def _refresh_horizon_status(self) -> None:
        hl = self.current_label.get("horizon_line_label")
        if hl:
            p0, p1 = hl["p0"], hl["p1"]
            self.lbl_horizon_status.configure(
                text=f"set: ({p0[0]:.2f},{p0[1]:.2f}) → "
                     f"({p1[0]:.2f},{p1[1]:.2f})",
                foreground="#274e13",
            )
        elif self._horizon_pending is not None:
            self.lbl_horizon_status.configure(
                text="awaiting 2nd click: click on fisheye to finish",
                foreground="#b45f06",
            )
        else:
            self.lbl_horizon_status.configure(text="not set",
                                              foreground="#666")

    def _label_autosave(self) -> None:
        if not self.current_label.get("frame_id"):
            return
        # Push the pre-edit state onto the undo stack (unless we're mid-undo).
        # `_undo_snapshot` holds the state as of the previous save / frame load;
        # current_label already carries this edit, so the snapshot IS the
        # "before" state Cmd+Z should restore.
        if not self._undoing and self._undo_snapshot is not None:
            self._undo_stack.append(self._undo_snapshot)
            if len(self._undo_stack) > 200:
                self._undo_stack.pop(0)
        self._undo_snapshot = copy.deepcopy(self.current_label)
        # The Treeview / horizon status / save count all reflect the
        # in-memory `current_label`; rebuild them after every edit so the
        # operator sees the change land before the file write happens.
        self._refresh_bbox_tree()
        self._refresh_horizon_status()
        # Recompute fusion bins from the current bboxes so downstream
        # metrics tooling has them ready without re-deriving offline.
        self._recompute_label_bins()
        # In audit mode, an un-accepted detector suggestion is shown + editable
        # but NOT persisted to manual.jsonl until the operator accepts it.
        if self._pseudo_mode and not self._frame_committed:
            if self.cached:
                self._redraw_label_overlay()
            return
        try:
            self.label_store.put(dict(self.current_label))
        except OSError as exc:
            self._status(f"save failed: {exc}")
            return
        self._refresh_label_summary()
        # Repaint the fisheye overlay artists from the updated record.
        if self.cached:
            self._redraw_label_overlay()

    def _recompute_label_bins(self) -> None:
        """Derive obstacle_bins_fisheye the same way `label_tool._derive_bins`
        does, so dashboard records stay shape-compatible with CLI ones."""
        fish_intr = self.intrinsics.get("fisheye", {})
        cx = fish_intr.get("cx")
        pix_deg = fish_intr.get("pix_deg_ratio")
        if cx is None or pix_deg is None or not self.bin_edges.size:
            self.current_label["obstacle_bins_fisheye"] = []
            return
        bboxes = [bb for bb in self.current_label.get("fisheye_bboxes", [])
                  if bb.get("cls") not in (None, "unset")]
        w = max(self.current_label.get("width", 1), 1)
        bins: set[int] = set()
        for bb in bboxes:
            x0, _, x1, _ = bb["xyxy"]
            cx_px = (x0 + x1) / 2.0 * w
            angle = (cx_px - cx) / pix_deg
            idx = angle_to_bin(angle, self.bin_edges)
            if idx is not None:
                bins.add(int(self.bin_centers[idx]))
        self.current_label["obstacle_bins_fisheye"] = sorted(bins)

    # -- Tool / class / show toggles --------------------------------------

    def _focus_is_text_entry(self) -> bool:
        """True when a typing widget owns focus: class shortcuts skip in
        this case so typing `b` into the class dropdown's filter doesn't
        also reclassify the active bbox."""
        try:
            focused = self.root.focus_get()
        except KeyError:
            return False
        if focused is None:
            return False
        cls = focused.winfo_class()
        return cls in (
            "Entry", "TEntry",
            "Spinbox", "TSpinbox",
            "Combobox", "TCombobox",
            "Text",
        )

    def _on_class_shortcut(self, cls_name: str) -> None:
        """Switch the active class via keyboard.

        Designed for the "draw a batch of one class, switch, draw the next
        batch" workflow: press `b` once and every subsequent click-drag
        creates a boat; press `d` and the next ones are ducks. The key
        does NOT retroactively reclass existing boxes; that surprised
        users when nothing was selected.

        The one exception is when a bbox is explicitly selected in the
        tree: that's clearly intentional ("fix this one's class"), so we
        reclass it in place as well.

        Gated to the Labelling tab + away from text-entry focus so the
        keys don't fire while navigating the Detection tab or typing in
        the class combobox.
        """
        if not self._label_tab_active():
            return
        if self._focus_is_text_entry():
            return
        self.label_class = cls_name
        if hasattr(self, "var_label_class"):
            self.var_label_class.set(cls_name)
        bboxes = self.current_label.get("fisheye_bboxes", [])
        try:
            sel = self.tree_bboxes.selection()
        except (tk.TclError, AttributeError):
            sel = ()
        target_idx: Optional[int] = None
        if sel:
            try:
                target_idx = int(sel[0])
            except ValueError:
                target_idx = None
        if (target_idx is not None
                and 0 <= target_idx < len(bboxes)
                and bboxes[target_idx].get("cls") != cls_name):
            old = bboxes[target_idx].get("cls", "unset")
            bboxes[target_idx]["cls"] = cls_name
            self._label_autosave()
            try:
                self.tree_bboxes.selection_set(str(target_idx))
            except tk.TclError:
                pass
            # Tk's programmatic selection_set doesn't fire
            # <<TreeviewSelect>>, so sync the inline class dropdown by
            # hand: otherwise it would still show the old class.
            self._on_bbox_selected(None)
            self._status(
                f"bbox {target_idx}: {old} → {cls_name}"
                f"  · new bbox class = {cls_name}"
            )
        else:
            self._status(f"new bbox class = {cls_name}")

    def _on_delete_last_shortcut(self) -> None:
        """Pop the most recent bbox on the current frame.

        Mirrors `label_tool.py`'s `u` (undo) key. Gated to the Labelling
        tab + away from text-entry focus so it can't fire while typing.
        """
        if not self._label_tab_active():
            return
        if self._focus_is_text_entry():
            return
        bboxes = self.current_label.get("fisheye_bboxes", [])
        if not bboxes:
            self._status("nothing to undo on this frame")
            return
        removed = bboxes.pop()
        self._label_autosave()
        self._status(f"deleted last bbox ({removed.get('cls', '?')})")

    def _on_label_tool_changed(self) -> None:
        self.label_tool = self.var_label_tool.get()
        # Cancel any in-flight bbox drag / pending horizon click whenever
        # the tool changes, so flipping tools never leaves half-state.
        self._bbox_drag_start = None
        self._bbox_drag_current = None
        if self.label_tool != LABEL_TOOL_HORIZON:
            self._horizon_pending = None
        self._refresh_horizon_status()
        if self.cached:
            self._redraw_label_overlay()

    def _on_show_labels_toggle(self) -> None:
        self.label_show_existing = self.var_show_labels.get()
        if self.cached:
            self._redraw_label_overlay()

    def _on_bbox_selected(self, _event=None) -> None:
        # Keep the inline class dropdown synced with selection: filled +
        # enabled when a bbox is selected, blank + disabled otherwise.
        sel = self.tree_bboxes.selection()
        target_idx: Optional[int] = None
        if sel:
            try:
                target_idx = int(sel[0])
            except ValueError:
                target_idx = None
        bboxes = self.current_label.get("fisheye_bboxes", [])
        if (target_idx is not None
                and 0 <= target_idx < len(bboxes)
                and hasattr(self, "cb_bbox_class")):
            self.cb_bbox_class.configure(state="readonly")
            self.var_bbox_class.set(bboxes[target_idx].get("cls", ""))
        elif hasattr(self, "cb_bbox_class"):
            self.var_bbox_class.set("")
            self.cb_bbox_class.configure(state="disabled")
        if self.cached:
            self._redraw_label_overlay()

    def _on_bbox_class_changed(self, _event=None) -> None:
        """Reclass the currently selected bbox to the dropdown's value."""
        sel = self.tree_bboxes.selection()
        if not sel:
            return
        try:
            i = int(sel[0])
        except ValueError:
            return
        bboxes = self.current_label.get("fisheye_bboxes", [])
        if not (0 <= i < len(bboxes)):
            return
        new_cls = self.var_bbox_class.get()
        if new_cls not in LABEL_CLASSES:
            return
        if bboxes[i].get("cls") == new_cls:
            return
        bboxes[i]["cls"] = new_cls
        self._label_autosave()
        # _refresh_bbox_tree (called by _label_autosave) wipes the
        # selection; restore it so the operator can keep working on
        # the same row without re-clicking.
        try:
            self.tree_bboxes.selection_set(str(i))
        except tk.TclError:
            pass

    def _on_bbox_delete(self, _event=None) -> None:
        self._delete_selected_bbox()

    def _delete_selected_bbox(self) -> None:
        sel = self.tree_bboxes.selection()
        if not sel:
            return
        try:
            i = int(sel[0])
        except ValueError:
            return
        bbs = self.current_label.get("fisheye_bboxes", [])
        if 0 <= i < len(bbs):
            del bbs[i]
            self._label_autosave()

    def _clear_all_bboxes(self) -> None:
        if not self.current_label.get("fisheye_bboxes"):
            return
        self.current_label["fisheye_bboxes"] = []
        self._label_autosave()

    def _clear_horizon_label(self) -> None:
        if "horizon_line_label" in self.current_label:
            del self.current_label["horizon_line_label"]
        self._horizon_pending = None
        self._label_autosave()

    def _cancel_horizon_pending(self) -> None:
        if self._horizon_pending is None:
            return
        self._horizon_pending = None
        self._refresh_horizon_status()
        if self.cached:
            self._redraw_label_overlay()

    # -- Quick navigation --------------------------------------------------

    def _step_custom(self, direction: int) -> None:
        """Step by the user-entered N in the Quick nav spinbox."""
        try:
            n = int(self.var_step_n.get())
        except (tk.TclError, ValueError):
            n = 1
        n = max(1, n)
        self._step(direction * n)

    def _jump_labelled(self, direction: int) -> None:
        """Step forward / backward to the nearest labelled frame in this clip."""
        if not self.cached or self.triplet is None:
            return
        present = self.label_store.frame_ids_for_triplet(
            self.triplet.scene, self.triplet.timestamp,
        )
        if not present:
            self._status("no labelled frames in this clip yet")
            return
        # Cached frames may not yet cover the entire clip; walk forward
        # via _ensure_cached_up_to so the prev/next search sees every
        # frame the operator has ever loaded.
        max_known = len(self.cached) - 1
        candidates: list[int] = []
        for i, entry in enumerate(self.cached):
            ts = entry[0]
            fid = _frame_id_for(self.triplet.scene,
                                self.triplet.timestamp, ts)
            if fid in present:
                candidates.append(i)
        if not candidates:
            self._status("labelled frames not yet visited: scrub forward")
            return
        if direction < 0:
            target = max((i for i in candidates if i < self.idx),
                         default=None)
        else:
            target = min((i for i in candidates if i > self.idx),
                         default=None)
        if target is None:
            self._status("no labelled frames in that direction")
            return
        self._advance_to(target)

    def _jump_next_unlabelled(self) -> None:
        if not self.cached or self.triplet is None:
            return
        present = self.label_store.frame_ids_for_triplet(
            self.triplet.scene, self.triplet.timestamp,
        )
        for i in range(self.idx + 1, len(self.cached)):
            ts = self.cached[i][0]
            fid = _frame_id_for(self.triplet.scene,
                                self.triplet.timestamp, ts)
            if fid not in present:
                self._advance_to(i)
                return
        # Fallthrough: ensure-forward one frame at a time. Cheap because
        # _ensure_cached_up_to stops on EOF.
        i = len(self.cached)
        while self._ensure_cached_up_to(i):
            ts = self.cached[i][0]
            fid = _frame_id_for(self.triplet.scene,
                                self.triplet.timestamp, ts)
            if fid not in present:
                self._advance_to(i)
                return
            i += 1
        self._status("no further unlabelled frames in this clip")

    # -- Matplotlib mouse events -----------------------------------------

    def _connect_mpl_mouse_events(self) -> None:
        # Disconnect first so hot-reload doesn't leak handlers when it
        # rebuilds the figure (which can re-call _connect_mpl_mouse_events).
        for cid in self._mpl_cids:
            try:
                self.fig.canvas.mpl_disconnect(cid)
            except Exception:  # noqa: BLE001 - backend-dependent
                pass
        self._mpl_cids = [
            self.fig.canvas.mpl_connect(
                "button_press_event", self._on_canvas_button_press,
            ),
            self.fig.canvas.mpl_connect(
                "motion_notify_event", self._on_canvas_motion,
            ),
            self.fig.canvas.mpl_connect(
                "button_release_event", self._on_canvas_button_release,
            ),
        ]

    def _label_active(self) -> bool:
        """True if mouse interactions on the fisheye should edit labels."""
        return (self._label_tab_active()
                and self.label_tool != LABEL_TOOL_NONE
                and self.ax_fish is not None
                and self.current_label.get("frame_id") is not None)

    def _image_coords_from_event(
        self, event,
    ) -> Optional[tuple[float, float]]:
        """Image-pixel (x, y) under the cursor, clamped to image bounds.

        Used by motion / release so the rubber-band keeps tracking and
        the bbox can be committed even when the cursor strays off the
        fisheye axis. `event.xdata`/`event.ydata` are only populated
        inside the axes; off-axis we fall back to inverting `transData`
        on the raw canvas pixel position so we can still tell where the
        cursor is. The result is clamped to [0, w] × [0, h] so the saved
        bbox edge sits on the image boundary rather than a synthetic
        point far past the frame.
        """
        if self.ax_fish is None:
            return None
        if event.x is None or event.y is None:
            return None
        if event.xdata is not None and event.ydata is not None:
            x, y = float(event.xdata), float(event.ydata)
        else:
            try:
                x, y = self.ax_fish.transData.inverted().transform(
                    (event.x, event.y)
                )
            except Exception:  # noqa: BLE001 - backend-dependent failure
                return None
            x, y = float(x), float(y)
        w = max(float(self.current_label.get("width", 1)), 1.0)
        h = max(float(self.current_label.get("height", 1)), 1.0)
        x = max(0.0, min(w, x))
        y = max(0.0, min(h, y))
        return x, y

    def _on_canvas_button_press(self, event) -> None:
        if not self._label_active():
            return
        if event.button != 1:
            return
        # The first click of any tool must land inside the fisheye image:
        # without that gate, random clicks on the figure (toolbar, other
        # axes) would start a labelling action. The horizon picker's
        # *second* click is allowed off-image, since the first click
        # already anchored the line inside the frame and the operator
        # may want to point at an off-frame target.
        is_horizon_second = (
            self.label_tool == LABEL_TOOL_HORIZON
            and self._horizon_pending is not None
        )
        if not is_horizon_second:
            if event.inaxes is not self.ax_fish:
                return
            if event.xdata is None or event.ydata is None:
                return
        if self.label_tool == LABEL_TOOL_BBOX:
            self._bbox_drag_start = (event.xdata, event.ydata)
            self._bbox_drag_current = (event.xdata, event.ydata)
            self._redraw_label_overlay()
        elif self.label_tool == LABEL_TOOL_HORIZON:
            if self._horizon_pending is None:
                self._horizon_pending = (event.xdata, event.ydata)
                self._refresh_horizon_status()
                self._redraw_label_overlay()
            else:
                x0, y0 = self._horizon_pending
                pt = self._image_coords_from_event(event)
                if pt is None:
                    return
                self._horizon_pending = None
                self._commit_horizon_line((x0, y0), pt)

    def _on_canvas_motion(self, event) -> None:
        if not self._label_active():
            return
        # If a drag is in flight, follow the cursor wherever it goes: the
        # buffer is what lets the operator finish a box that ends past the
        # image edge. If no drag is active we still require the cursor to
        # be in the fisheye axes so unrelated motion events don't waste
        # work re-drawing the overlay.
        drag_active = (
            (self.label_tool == LABEL_TOOL_BBOX
             and self._bbox_drag_start is not None)
            or (self.label_tool == LABEL_TOOL_HORIZON
                and self._horizon_pending is not None)
        )
        if not drag_active and event.inaxes is not self.ax_fish:
            return
        pt = self._image_coords_from_event(event)
        if pt is None:
            return
        if (self.label_tool == LABEL_TOOL_BBOX
                and self._bbox_drag_start is not None):
            self._bbox_drag_current = pt
            self._redraw_label_overlay()
        elif (self.label_tool == LABEL_TOOL_HORIZON
                and self._horizon_pending is not None):
            self._horizon_cursor = pt
            self._redraw_label_overlay()

    def _on_canvas_button_release(self, event) -> None:
        if not self._label_active():
            return
        if event.button != 1:
            return
        # Release commits even when the cursor has wandered off the
        # axes: coordinates are clamped to the image edge, so an object
        # that pokes out of the frame can still get a bbox that sits on
        # the boundary.
        if (self.label_tool == LABEL_TOOL_BBOX
                and self._bbox_drag_start is not None):
            pt = self._image_coords_from_event(event)
            x0, y0 = self._bbox_drag_start
            self._bbox_drag_start = None
            self._bbox_drag_current = None
            if pt is None:
                # Couldn't recover coords: abandon the drag silently
                # rather than create a degenerate 0-area box.
                self._redraw_label_overlay()
                return
            self._commit_bbox((x0, y0), pt)

    def _commit_bbox(self, p0_px: tuple[float, float],
                     p1_px: tuple[float, float]) -> None:
        if self.ax_fish is None or self.im_fish is None:
            return
        # imshow extents track the current undistorted image; pull width
        # and height from the cached label record (set on frame load).
        w = max(self.current_label.get("width", 1), 1)
        h = max(self.current_label.get("height", 1), 1)
        x0, y0 = p0_px
        x1, y1 = p1_px
        x0n = max(0.0, min(1.0, min(x0, x1) / w))
        y0n = max(0.0, min(1.0, min(y0, y1) / h))
        x1n = max(0.0, min(1.0, max(x0, x1) / w))
        y1n = max(0.0, min(1.0, max(y0, y1) / h))
        # Reject sub-pixel drags so a stray click doesn't create a 0-area
        # bbox that downstream metrics tooling would silently ignore.
        if (x1n - x0n) * w < 5 or (y1n - y0n) * h < 5:
            return
        self.current_label.setdefault("fisheye_bboxes", []).append({
            "cls": self.label_class,
            "xyxy": [x0n, y0n, x1n, y1n],
        })
        self._label_autosave()

    def _commit_horizon_line(self, p0_px: tuple[float, float],
                              p1_px: tuple[float, float]) -> None:
        w = max(self.current_label.get("width", 1), 1)
        h = max(self.current_label.get("height", 1), 1)
        x0, y0 = p0_px
        x1, y1 = p1_px
        self.current_label["horizon_line_label"] = {
            "p0": [max(0.0, min(1.0, x0 / w)),
                   max(0.0, min(1.0, y0 / h))],
            "p1": [max(0.0, min(1.0, x1 / w)),
                   max(0.0, min(1.0, y1 / h))],
        }
        # Drop the rubber-band cursor so it doesn't linger.
        if hasattr(self, "_horizon_cursor"):
            delattr(self, "_horizon_cursor")
        self._label_autosave()

    # -- Label overlay drawing -------------------------------------------

    def _clear_label_artists(self) -> None:
        for a in self.label_artists:
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        self.label_artists = []

    def _redraw_label_overlay(self) -> None:
        """Refresh just the label overlay artists without re-running detection.

        Called whenever annotations or label-visibility toggles change but
        the underlying frame is the same: much cheaper than a full
        `_rerender` (which re-projects radar points etc).
        """
        if self.ax_fish is None:
            return
        self._clear_label_artists()
        self._draw_label_overlay()
        self.canvas.draw_idle()

    def _draw_label_overlay(self) -> None:
        if self.ax_fish is None:
            return
        from matplotlib.patches import Rectangle  # local; keeps top imports lean

        w = max(self.current_label.get("width", 1), 1)
        h = max(self.current_label.get("height", 1), 1)
        label_visible = (self.label_show_existing or self._label_tab_active())
        if not label_visible:
            return

        sel_idx = -1
        if self._label_tab_active():
            try:
                sel = self.tree_bboxes.selection()
                sel_idx = int(sel[0]) if sel else -1
            except (tk.TclError, ValueError):
                sel_idx = -1

        # Monocular range for labelled boxes: a label's bottom edge is the true
        # waterline contact, so this is more accurate than the blob estimate.
        # Uses the same attitude reference the pipeline uses (IMU > confident
        # horizon > level). Computed once per redraw, reused for every box.
        rng_up = rng_K = None
        rng_height = 0.27
        rng_max = None
        rng_seg = None
        show_ranges = (getattr(self, "var_range_labels", None) is not None
                       and self.var_range_labels.get())
        if show_ranges and self.pipeline is not None:
            res = (self.cached[self.idx][4]
                   if self.cached and 0 <= self.idx < len(self.cached) else None)
            fish_res = res.fisheye if res is not None else None
            if fish_res is not None:
                rng_K = self.pipeline.intrinsics["fisheye"]["K"]
                rng_height = self.pipeline._camera_height["fisheye"]
                # Same attitude priority as the pipeline (IMU > water edge >
                # horizon > level): pass the mask so water-edge applies.
                rng_seg = fish_res.seg_mask
                rng_up = self.pipeline._range_up_vector(
                    rng_K, fish_res.horizon_line, seg_mask=rng_seg)
                mr = self.pipeline._range_cfg.get("max_range_m")
                rng_max = float(mr) if mr is not None else None
                # Radar arbitration for label boxes (same rules as the
                # pipeline): radar points projected into the undistorted
                # image; a box the radar confirms takes the radar range, a
                # near camera reading the radar contradicts is not announced.
                rng_rad_px = rng_rad_y = None
                mm = (self.cached[self.idx][4].mmwave
                      if self.cached and 0 <= self.idx < len(self.cached)
                      else None)
                if mm is not None and len(mm.points_xyz):
                    from scripts.utils.geometry import (
                        project_radar_to_undistorted,
                    )
                    rng_rad_px = project_radar_to_undistorted(
                        mm.points_xyz, rng_K,
                        self.pipeline.extrinsics["T_radar_to_fisheye"])
                    rng_rad_y = mm.points_xyz[:, 1]

        # Saved bboxes for the current frame.
        for i, bb in enumerate(self.current_label.get("fisheye_bboxes", [])):
            x0n, y0n, x1n, y1n = bb["xyxy"]
            x0, y0 = x0n * w, y0n * h
            bw, bh = (x1n - x0n) * w, (y1n - y0n) * h
            colour = CLASS_COLORS.get(bb.get("cls", "unset"),
                                      CLASS_COLORS["unset"])
            lw = 3.0 if i == sel_idx else 2.0
            rect = Rectangle((x0, y0), bw, bh,
                             fill=False, edgecolor=colour,
                             linewidth=lw)
            self.ax_fish.add_patch(rect)
            self.label_artists.append(rect)
            txt = self.ax_fish.text(
                x0 + 4, max(y0 - 4, 4), bb.get("cls", "?"),
                color=colour, fontsize=8,
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.4, pad=1,
                          edgecolor="none"),
            )
            self.label_artists.append(txt)
            # Range label at the box's bottom-right (its waterline contact).
            # A specific distance is only announced when RELIABLE (within
            # range.max_range_m). At the 0.27 m mount height an object near the
            # horizon sits only a few pixels below it, so the estimate there is
            # hypersensitive (1 px ~ several metres) and a precise number would
            # mislead: those show as ">Nm". None = bbox bottom above the
            # horizon (elevated / onshore object, not on the water plane) -> no
            # distance announced.
            if rng_up is not None:
                # Waterline-contact range when a seg mask exists: the range
                # ray goes through where the obstacle actually meets water
                # inside the box, not the box bottom: a loose/tilted box
                # over-captures water and its bottom edge reads falsely NEAR
                # (AuthorTwo, 2026-07-09). Falls back to bbox-bottom when no
                # contact is found (or no mask for the clip).
                from scripts.utils.geometry import range_from_contact_point
                rng = None
                point_uv = (x0 + bw / 2.0, y0 + bh)
                if rng_seg is not None:
                    from scripts.utils.segmentation import water_edge_contact
                    contact = water_edge_contact(
                        rng_seg, (x0, y0, x0 + bw, y0 + bh),
                        self.pipeline._contact_params)
                    if contact is not None:
                        rng = range_from_contact_point(
                            contact, rng_K, rng_up, rng_height,
                            max_range_m=None)
                        point_uv = contact
                if rng is None:
                    rng = range_from_water_plane_up(
                        (x0, y0, x0 + bw, y0 + bh), rng_K, rng_up,
                        rng_height, max_range_m=None)
                # Radar match for THIS box: min radar Y among points that
                # project inside the (padded) box.
                radar_rng = None
                box_had_radar = False
                if rng_rad_px is not None:
                    import numpy as _np
                    pad = 16.0
                    inside = ((rng_rad_px[:, 0] >= x0 - pad)
                              & (rng_rad_px[:, 0] <= x0 + bw + pad)
                              & (rng_rad_px[:, 1] >= y0 - pad)
                              & (rng_rad_px[:, 1] <= y0 + bh + pad)
                              & _np.isfinite(rng_rad_px).all(axis=1))
                    box_had_radar = bool(inside.any())
                    if box_had_radar:
                        radar_rng = float(rng_rad_y[inside].min())
                # Foreground arbitration (mirrors the pipeline): a matched
                # return on this object's OBSTACLE pixels is trusted outright
                # (silhouette rule); otherwise a return far NEARER than the
                # camera's own estimate is a foreground object's -> rejected.
                if radar_rng is not None:
                    on_sil = []
                    if rng_seg is not None:
                        import numpy as _np
                        h_s, w_s = rng_seg.shape[:2]
                        for j in _np.where(inside)[0]:
                            u, v = rng_rad_px[j]
                            ui, vi = int(round(u)), int(round(v))
                            if (0 <= vi < h_s and 0 <= ui < w_s
                                    and rng_seg[vi, ui] == 0):  # OBSTACLE id
                                on_sil.append(float(rng_rad_y[j]))
                    if on_sil:
                        radar_rng = float(min(on_sil))
                    else:
                        ratio = float(
                            (self.pipeline.detection.get("fusion", {}) or {})
                            .get("association", {})
                            .get("range_compat_ratio", 0.5) or 0.0)
                        if ratio > 0 and (rng is None
                                          or radar_rng < ratio * rng):
                            radar_rng = None
                if radar_rng is not None:
                    rng = radar_rng           # radar owns the near field
                elif (rng is not None and rng_rad_px is not None
                      and not box_had_radar):
                    # Contradiction veto: near camera reading, radar alive,
                    # nothing in the box -> don't announce (mirror pipeline;
                    # a box that HAD radar points, even incompatible ones,
                    # is exempt: matches radar_hits semantics).
                    veto = self.pipeline._range_cfg.get("radar_veto", {}) or {}
                    if veto.get("enabled", False):
                        import math as _math
                        fx0, cx0 = float(rng_K[0, 0]), float(rng_K[0, 2])
                        bearing = _math.degrees(_math.atan2(
                            (x0 + bw / 2.0) - cx0, fx0))
                        if (float(veto.get("min_m", 1.0)) <= rng
                                <= float(veto.get("max_m", 8.0))
                                and abs(bearing)
                                <= float(veto.get("max_bearing_deg", 50.0))):
                            rng = None
                if rng is not None:
                    # Reliability mirrors pipeline._detection_ranges: the
                    # pixel-noise-shifted bound must stay inside the trust
                    # envelope, else the number is noise-dominated -> ">Nm".
                    # Radar-sourced ranges are exact -> always reliable.
                    reliable = rng_max is None or rng <= rng_max
                    unc = float(self.pipeline._range_cfg.get(
                        "uncertainty_px", 0.0) or 0.0)
                    if (radar_rng is None and reliable and unc > 0
                            and rng_max is not None):
                        hi = range_from_contact_point(
                            (point_uv[0], point_uv[1] - unc), rng_K, rng_up,
                            rng_height, max_range_m=None)
                        reliable = hi is not None and hi <= rng_max
                    rtxt = self.ax_fish.text(
                        x0 + bw + 2, y0 + bh,
                        f"{rng:.1f}m" if reliable else f">{rng_max:.0f}m",
                        color="yellow" if reliable else "#ff9900",
                        fontsize=8, weight="bold",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1,
                                  edgecolor="none"),
                    )
                    self.label_artists.append(rtxt)

        # Horizon line saved on this frame. The 2 clicks define a line:
        # we extend it across the full image width so the operator sees
        # the implied horizon, not just the segment between their clicks.
        # The original anchor points stay rendered as small markers so
        # they can tell which clicks anchored the extrapolation.
        hl = self.current_label.get("horizon_line_label")
        if hl:
            p0 = hl["p0"]; p1 = hl["p1"]
            x0_px, y0_px = p0[0] * w, p0[1] * h
            x1_px, y1_px = p1[0] * w, p1[1] * h
            # Vertical-line case: extrapolation is a vertical segment at
            # the shared x. Anything else extends y = m·x + b across [0, w].
            if abs(x1_px - x0_px) < 1e-6:
                xs = [x0_px, x0_px]
                ys = [0.0, float(h)]
            else:
                slope = (y1_px - y0_px) / (x1_px - x0_px)
                intercept = y0_px - slope * x0_px
                xs = [0.0, float(w)]
                ys = [intercept, slope * w + intercept]
            line = self.ax_fish.plot(
                xs, ys,
                color=HORIZON_COLOR, linewidth=2.0,
            )[0]
            self.label_artists.append(line)
            for px, py in (p0, p1):
                marker = self.ax_fish.plot(
                    [px * w], [py * h],
                    marker="o", color=HORIZON_COLOR,
                    markersize=5,
                )[0]
                self.label_artists.append(marker)

        # In-flight drag rubber-band rectangle for the bbox tool.
        if (self.label_tool == LABEL_TOOL_BBOX
                and self._bbox_drag_start is not None
                and self._bbox_drag_current is not None
                and self._label_tab_active()):
            x0, y0 = self._bbox_drag_start
            x1, y1 = self._bbox_drag_current
            colour = CLASS_COLORS.get(self.label_class,
                                      CLASS_COLORS["unset"])
            rect = Rectangle((min(x0, x1), min(y0, y1)),
                             abs(x1 - x0), abs(y1 - y0),
                             fill=False, edgecolor=colour,
                             linewidth=1.5, linestyle="--")
            self.ax_fish.add_patch(rect)
            self.label_artists.append(rect)

        # Pending first click of the horizon picker.
        if (self.label_tool == LABEL_TOOL_HORIZON
                and self._horizon_pending is not None
                and self._label_tab_active()):
            px, py = self._horizon_pending
            marker = self.ax_fish.plot(
                [px], [py], marker="o",
                color=HORIZON_COLOR, markersize=6,
            )[0]
            self.label_artists.append(marker)
            cursor = getattr(self, "_horizon_cursor", None)
            if cursor is not None:
                line = self.ax_fish.plot(
                    [px, cursor[0]], [py, cursor[1]],
                    color=HORIZON_COLOR, linewidth=1.2,
                    linestyle="--", alpha=0.75,
                )[0]
                self.label_artists.append(line)

    # -- Per-frame load/save ---------------------------------------------

    def _label_record_for_frame(self, ts: str, fish_bgr) -> dict:
        """Return the saved record for this frame, or a blank seed dict."""
        scene = self.triplet.scene if self.triplet else ""
        triplet_ts = self.triplet.timestamp if self.triplet else ""
        fid = _frame_id_for(scene, triplet_ts, ts)
        existing = self.label_store.get(fid)
        if existing is not None:
            # Ensure the in-memory record carries the latest image size in
            # case the clip overrides resized it since the record was saved.
            existing = dict(existing)
            existing.setdefault("scene", scene)
            existing.setdefault("triplet_ts", triplet_ts)
            existing.setdefault("frame_ts", ts)
            existing.setdefault("fisheye_bboxes", [])
            h, w = (fish_bgr.shape[:2] if fish_bgr is not None
                    else (existing.get("height", 0),
                          existing.get("width", 0)))
            existing["width"] = w
            existing["height"] = h
            existing.setdefault("source", "dashboard-manual")
            existing.setdefault("audited", True)
            return existing
        # Audit mode: seed the frame with the detector's suggestions (editable,
        # not yet persisted): unless it's an excluded eval-clip frame.
        if (self._pseudo_mode and fid in self._pseudo
                and fid not in self._exclude):
            record = self._empty_label_record()
            record["frame_id"] = fid
            record["scene"] = scene
            record["triplet_ts"] = triplet_ts
            record["frame_ts"] = ts
            if fish_bgr is not None:
                record["height"], record["width"] = fish_bgr.shape[:2]
            record["fisheye_bboxes"] = copy.deepcopy(self._pseudo[fid])
            record["source"] = "detector-suggested"
            return record
        record = self._empty_label_record()
        record["frame_id"] = fid
        record["scene"] = scene
        record["triplet_ts"] = triplet_ts
        record["frame_ts"] = ts
        if fish_bgr is not None:
            record["height"], record["width"] = fish_bgr.shape[:2]
        return record

    def _activate_label_for_current_frame(self) -> None:
        if not self.cached:
            self.current_label = self._empty_label_record()
        else:
            ts, fish_bgr, *_ = self.cached[self.idx]
            self.current_label = self._label_record_for_frame(ts, fish_bgr)
        # A frame is "committed" if it's already in the manual store; a
        # detector-seeded suggestion is not, so its edits don't autosave until
        # the operator accepts it (Add to manual set).
        fid = self.current_label.get("frame_id")
        self._frame_committed = bool(fid and fid in self.label_store.records)
        self._refresh_added_counter()
        # Undo history is per-frame: reset the baseline to the loaded state so
        # Cmd+Z can't reach across a frame change.
        self._undo_stack = []
        self._undo_snapshot = copy.deepcopy(self.current_label)
        # Drag / pending state is per-frame: reset whenever we change frame
        # so an unfinished drag doesn't survive a scrub.
        self._bbox_drag_start = None
        self._bbox_drag_current = None
        self._horizon_pending = None
        if hasattr(self, "_horizon_cursor"):
            delattr(self, "_horizon_cursor")
        self._refresh_bbox_tree()
        self._refresh_horizon_status()
        self._refresh_label_summary()

    # -- Misc ---------------------------------------------------------------

    def _hot_reload(self):
        try:
            self.detection = (load_detection(self.detection_path)
                              if self.detection_path else load_detection())
            self.extrinsics = load_extrinsics()
            self.all_overrides = load_overrides()
            edges, centers = make_bins(self.detection["fusion"])
            self.bin_edges = edges
            self.bin_centers = centers
            self.bar_x = np.arange(len(centers))
            # Re-seed the heading smoother in case the window changed.
            head_cfg = self.detection.get("heading", {}) or {}
            self.heading_smoother = HeadingSmoother(
                window_n=int(head_cfg.get("smoothing_window", 9)),
                min_samples=int(head_cfg.get("smoothing_min_samples", 3)),
            )
            self.smoothed_history.clear()
            # _apply_layout rebuilds bars + wedges + every other panel
            # respecting the current visibility settings, so the reload
            # works even when (e.g.) the bars panel is hidden.
            self._apply_layout()
            # Reload pipeline + replay current clip.
            if self.triplet is not None:
                current_idx = self.idx
                self._on_load()
                self._advance_to(current_idx)
            else:
                self.canvas.draw_idle()
            self._status("reloaded detection.yaml / extrinsics / overrides")
        except Exception as exc:  # noqa: BLE001
            self._status(f"reload failed: {exc}")

    def _screenshot(self):
        if not self.cached:
            self._status("nothing to screenshot yet")
            return
        SCREEN_DIR.mkdir(parents=True, exist_ok=True)
        name = (f"{self.triplet.scene}_{self.triplet.timestamp}"
                f"_frame{self.idx:05d}_"
                f"{datetime.now().strftime('%H%M%S')}.png")
        path = SCREEN_DIR / name
        self.fig.savefig(path, dpi=140, bbox_inches="tight")
        self._status(f"saved {path}")

    # -- Video export --------------------------------------------------------

    @staticmethod
    def _speed_tag_to_multiplier(tag: str) -> Optional[float]:
        """'0.5x' -> 0.5, '2x' -> 2.0; None when the tag doesn't parse."""
        try:
            mult = float(tag.strip().rstrip("xX"))
        except ValueError:
            return None
        return mult if mult > 0 else None

    def _open_export_dialog(self) -> None:
        """Export the main figure: current layout, overlay toggles and
        threshold, exactly as rendered on screen: to an mp4.

        The playback speed defaults to the transport's current speed
        selection; picking "custom" takes a free-form multiplier (1x =
        real-time = EXPORT_BASE_FPS frames/s of footage).
        """
        if self.triplet is None:
            self._status("load a clip before exporting")
            return
        if self._exporting:
            return

        win = tk.Toplevel(self.root)
        win.title("Export main panel as video")
        win.transient(self.root)
        win.resizable(False, False)
        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)

        # Speed row: seeded from the transport's current selection.
        ttk.Label(frm, text="Speed").grid(row=0, column=0, sticky="w")
        speed_values = [s[0] for s in SPEEDS] + ["custom"]
        var_speed = tk.StringVar(value=self.cb_speed.get())
        cb = ttk.Combobox(frm, values=speed_values, textvariable=var_speed,
                          state="readonly", width=8)
        cb.grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Label(frm, text="Custom multiplier").grid(row=1, column=0,
                                                      sticky="w", pady=(6, 0))
        var_custom = tk.StringVar(value="1.5")
        ent_custom = ttk.Entry(frm, textvariable=var_custom, width=8,
                               state="disabled")
        ent_custom.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        lbl_fps = ttk.Label(frm, text="", foreground="#555")
        lbl_fps.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        def _current_fps() -> Optional[float]:
            tag = var_speed.get()
            mult = (self._speed_tag_to_multiplier(var_custom.get())
                    if tag == "custom"
                    else self._speed_tag_to_multiplier(tag))
            if mult is None:
                return None
            return EXPORT_BASE_FPS * mult

        def _refresh_fps(*_args) -> None:
            ent_custom.configure(
                state="normal" if var_speed.get() == "custom" else "disabled")
            fps = _current_fps()
            lbl_fps.configure(
                text=(f"→ {fps:.2f} fps (1x = {EXPORT_BASE_FPS:g} fps)"
                      if fps else "enter a positive multiplier, e.g. 1.5"))

        cb.bind("<<ComboboxSelected>>", _refresh_fps)
        var_custom.trace_add("write", _refresh_fps)
        _refresh_fps()

        # Output path row.
        ttk.Label(frm, text="Output").grid(row=3, column=0, sticky="w",
                                           pady=(10, 0))
        default_name = (f"{self.triplet.scene}_{self.triplet.timestamp}_"
                        f"{datetime.now().strftime('%H%M%S')}.mp4")
        var_path = tk.StringVar(value=str(EXPORT_DIR / default_name))
        ent_path = ttk.Entry(frm, textvariable=var_path, width=48)
        ent_path.grid(row=3, column=1, sticky="ew", padx=(6, 0), pady=(10, 0))

        def _browse() -> None:
            chosen = filedialog.asksaveasfilename(
                parent=win, defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4")],
                initialdir=str(Path(var_path.get()).parent),
                initialfile=Path(var_path.get()).name)
            if chosen:
                var_path.set(chosen)

        ttk.Button(frm, text="Browse…", command=_browse).grid(
            row=3, column=2, padx=(6, 0), pady=(10, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=3, sticky="e", pady=(12, 0))

        def _do_export() -> None:
            fps = _current_fps()
            if fps is None:
                self._status("export: invalid custom speed multiplier")
                return
            out = Path(var_path.get()).expanduser()
            win.destroy()
            self._export_video(out, fps)

        ttk.Button(btns, text="Cancel",
                   command=win.destroy).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Export",
                   command=_do_export).pack(side="left")
        win.grab_set()

    def _export_video(self, out_path: Path, fps: float) -> None:
        """Walk the whole clip through the normal playback/render path and
        write each on-screen figure frame to `out_path` at `fps`.

        Runs synchronously on the Tk thread; a small progress window with a
        Cancel button stays responsive because we pump the event loop once
        per frame. The playhead is restored afterwards.
        """
        # Freeze the transport while we drive the playhead ourselves.
        self._cancel_tick()
        self.playing = False
        self.btn_play.configure(text="▶")
        start_idx = self.idx
        self._exporting = True

        prog = tk.Toplevel(self.root)
        prog.title("Exporting…")
        prog.transient(self.root)
        prog.resizable(False, False)
        lbl_prog = ttk.Label(prog, text="starting…", padding=(12, 8),
                             width=36, anchor="w")
        lbl_prog.pack()
        cancelled = {"flag": False}
        ttk.Button(prog, text="Cancel",
                   command=lambda: cancelled.update(flag=True)).pack(
                       pady=(0, 8))
        prog.protocol("WM_DELETE_WINDOW",
                      lambda: cancelled.update(flag=True))

        writer = None
        frames_written = 0
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            total = self.frame_total if self.frame_total > 0 else None
            idx = 0
            while True:
                if cancelled["flag"]:
                    break
                if not self._advance_to(idx) or self.idx < idx:
                    break  # end of clip (same clamp check as _tick)
                # _render finished with draw_idle; force the draw so the
                # Agg buffer holds this frame before we grab it.
                self.canvas.draw()
                buf = np.asarray(self.canvas.buffer_rgba())
                frame = cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)
                if writer is None:
                    # mp4v wants even dimensions; crop a stray row/col.
                    h = frame.shape[0] - (frame.shape[0] % 2)
                    w = frame.shape[1] - (frame.shape[1] % 2)
                    size = (w, h)
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, size)
                    if not writer.isOpened():
                        self._status(f"export failed: cannot open {out_path}")
                        return
                if (frame.shape[1], frame.shape[0]) != size:
                    # Window resized mid-export: keep the writer size.
                    frame = cv2.resize(frame, size)
                else:
                    frame = frame[:size[1], :size[0]]
                writer.write(frame)
                frames_written += 1
                lbl_prog.configure(
                    text=f"frame {idx + 1}/{total if total else '?'}")
                self.root.update()  # keep UI alive + service Cancel
                idx += 1
        except Exception as exc:  # noqa: BLE001
            self._status(f"export failed: {exc}")
            return
        finally:
            if writer is not None:
                writer.release()
            self._exporting = False
            try:
                prog.destroy()
            except tk.TclError:
                pass
            # Put the playhead back where the user left it.
            if self.cached:
                self._advance_to(min(start_idx, len(self.cached) - 1))

        if frames_written == 0:
            self._status("export: no frames written")
        elif cancelled["flag"]:
            self._status(f"export cancelled: {frames_written} frames "
                         f"written to {out_path}")
        else:
            self._status(f"exported {frames_written} frames @ {fps:.2f} fps "
                         f"→ {out_path}")

    def _on_s_key(self) -> None:
        """`s`: set the active class to `structure` while the Labelling tab is
        active (parity with `m`), otherwise the original screenshot binding."""
        if self._label_tab_active() and not self._focus_is_text_entry():
            self._on_class_shortcut("structure")
        else:
            self._screenshot()

    def _on_add_key(self) -> None:
        """`a`: accept the current frame into the manual set (audit mode)."""
        if (self._pseudo_mode and self._label_tab_active()
                and not self._focus_is_text_entry()):
            self._on_add_to_manual()

    def _on_undo(self):
        """Cmd/Ctrl+Z: restore the previous label state on the current frame."""
        if not self._label_tab_active() or self._focus_is_text_entry():
            return None  # outside labelling: let any text widget undo itself
        if not self._undo_stack:
            self._status("nothing to undo")
            return "break"
        prev = self._undo_stack.pop()
        self.current_label = copy.deepcopy(prev)
        self._bbox_drag_start = None
        self._bbox_drag_current = None
        self._undoing = True
        try:
            self._label_autosave()  # persists prev + rebuilds tree/overlay
        finally:
            self._undoing = False
        self._status("undo")
        return "break"

    def _bind_keys(self):
        self.root.bind("<Left>", lambda _e: self._step(-1))
        self.root.bind("<Right>", lambda _e: self._step(+1))
        self.root.bind("<Shift-Left>", lambda _e: self._jump_seconds(-5.0))
        self.root.bind("<Shift-Right>", lambda _e: self._jump_seconds(+5.0))
        self.root.bind("<space>", lambda _e: self._toggle_play())
        # `s` sets the structure class while labelling, screenshots elsewhere.
        self.root.bind("s", lambda _e: self._on_s_key())
        # `a` accepts the current frame into the manual set (audit mode).
        self.root.bind("a", lambda _e: self._on_add_key())
        # Undo the last labelling action on this frame (Cmd+Z mac / Ctrl+Z).
        self.root.bind("<Command-z>", lambda _e: self._on_undo())
        self.root.bind("<Control-z>", lambda _e: self._on_undo())
        self.root.bind("r", lambda _e: self._hot_reload())
        self.root.bind("q", lambda _e: self.root.destroy())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())
        # `label_tool.py` class shortcuts: gated so they only fire when
        # the Labelling tab is active and focus isn't on a text-entry
        # widget. See _on_class_shortcut.
        for keysym, cls in (("b", "boat"), ("d", "duck"), ("B", "buoy"),
                            ("p", "person"), ("m", "structure"), ("o", "other")):
            if cls not in LABEL_CLASSES:
                continue
            self.root.bind(
                keysym,
                lambda _e, cls=cls: self._on_class_shortcut(cls),
            )
        # `u` pops the most recent bbox: same gates as the class keys.
        self.root.bind("u", lambda _e: self._on_delete_last_shortcut())

    def _status(self, msg: str):
        self.lbl_status.configure(text=msg)


def main():
    if not _TK_AVAILABLE:
        raise RuntimeError(
            f"tkinter is required for the dashboard but is not available "
            f"({_TK_ERROR}). On macOS with Homebrew Python, install with: "
            f"brew install python-tk@<major>.<minor>  "
            f"(e.g. `brew install python-tk@3.14`)."
        )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", default=None,
                    help="Preload data/<scene>/<timestamp>.")
    ap.add_argument("--detection", default=None,
                    help="Override path to detection.yaml.")
    ap.add_argument("--pseudo", default=None,
                    help="Detector labels JSONL: pre-fill frames with editable "
                         "suggestions + enable the Add-to-manual-set audit mode "
                         "(key 'a', n/target counter).")
    ap.add_argument("--exclude", default=None,
                    help="eval mapping JSON (e.g. labels/eval_smoke36_mapping.json); "
                         "its frames are excluded from the audit set to avoid "
                         "train/test leakage.")
    ap.add_argument("--target", type=int, default=500,
                    help="Manual-set size target shown in the n/target counter.")
    ap.add_argument("--train-out", default=None,
                    help="Where accepted audit frames persist (default "
                         "labels/training_frames.jsonl when --pseudo is set). "
                         "Reloaded on restart so progress resumes across "
                         "sessions; kept separate from labels/manual.jsonl.")
    args = ap.parse_args()

    train_out = args.train_out
    if args.pseudo and not train_out:
        train_out = "labels/training_frames.jsonl"

    root = tk.Tk()
    try:
        root.geometry("1500x900")
    except tk.TclError:
        pass
    Dashboard(root, initial_triplet=args.triplet,
              detection_path=args.detection, pseudo_path=args.pseudo,
              exclude_path=args.exclude, target=args.target,
              train_out=train_out)
    root.mainloop()


if __name__ == "__main__":
    main()
