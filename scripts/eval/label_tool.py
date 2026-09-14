"""Manual labelling + Qwen-audit UI for fisheye frames.

Phase I.4.5 Track A + §I.4.11 heading extension. Step through a triplet
(or a folder of pre-extracted frames), click-drag bounding boxes, assign
class with one key, mark a "safe heading" with the keyboard. Or audit a
provisional Qwen JSONL frame-by-frame (accept / reject / edit).

This tool is kept **in lockstep with the Qwen flow** (scripts/eval/
qwen_batch_labeler.py):

  * the class taxonomy is the same set (loaded from
    configs/qwen_label_prompt.yaml), including `structure`;
  * frames are pulled through the same `iter_clip_frames` (undistorted,
    dashboard `ts=` frame_ids), so a label here shares its `frame_id`
    with the Qwen label and the dashboard label of the same frame;
  * `data/captures/<mission>/` clips resolve the same way the Qwen
    labeller resolves them, so the audit can render Qwen output directly.

Usage:
    # manual labelling of a clip (mission clips work too):
    python -m scripts.eval.label_tool \\
        --triplet data/captures/2026-06-17_institutionone_day1/2026-06-17_12-41-23 --every 10
    # audit a provisional Qwen JSONL into labels/master.jsonl:
    python -m scripts.eval.label_tool --audit labels/qwen/qwen_2026-06-17_institutionone_day1.jsonl

Controls (labelling):
    left-click+drag    draw a bounding box
    b d B p m o        set class on the *most recent* bbox
                       (b=boat, d=duck, B=buoy, p=person, m=structure, o=other)
    , / .              nudge the safe-heading caret left/right by 5°
                       (< / > for 1° fine adjustment)
    h                  toggle "no safe direction" (safe_heading_deg = null)
    n                  next frame (saves all bboxes + safe heading)
    backspace          previous frame
    u                  undo last bbox
    space              mark "none" (no obstacle on this frame)
    s                  skip without saving
    q                  quit

Controls (audit):
    a accept · r reject · e edit · n skip · q quit

Heading semantics: see PLAN.md §I.4.11 + §I.7. `safe_heading_deg` is a
single per-frame number in degrees from the bow (positive = starboard).
When the whole FOV is blocked, press `h` so the saved label is
explicit-null instead of an arbitrary number.
"""

from __future__ import annotations

import argparse
import json
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from scripts.sensor_processing.pipeline import angle_to_bin, make_bins
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.cv_common import undistort_fisheye
from scripts.utils.datasets import (
    CAPTURES_DIR,
    DATA_DIR,
    REPO_ROOT,
    resolve_triplet,
)
# Share the exact taxonomy + frame iteration with the Qwen flow so labels and
# audited frames line up 1:1 with qwen_batch_labeler output.
from scripts.eval.qwen_batch_labeler import PROMPT_CFG, iter_clip_frames

LABELS_DIR = REPO_ROOT / "labels"
MANUAL_PATH = LABELS_DIR / "manual.jsonl"
MASTER_PATH = LABELS_DIR / "master.jsonl"

#: Class taxonomy comes from the Qwen prompt config so the two never drift.
LABEL_CLASSES = list(PROMPT_CFG.get(
    "classes", ["boat", "duck", "buoy", "person", "structure", "other"]))

#: Fixed key bindings; only those whose class is in the taxonomy are active.
_CLASS_KEYS = {
    "boat": "b", "duck": "d", "buoy": "B", "person": "p",
    "structure": "m", "other": "o",
}
KEY_TO_CLASS = {ord(k): cls for cls, k in _CLASS_KEYS.items()
                if cls in LABEL_CLASSES}

#: Overlay colours (BGR), matching scripts/eval/render_label_overlays.py.
CLASS_COLORS = {
    "boat": (0, 200, 0), "duck": (0, 200, 255), "buoy": (0, 0, 230),
    "person": (255, 80, 0), "structure": (0, 165, 255), "other": (160, 0, 160),
    "unset": (128, 128, 128),
}

#: Frame record passed around the UI.
Frame = namedtuple("Frame", "frame_id scene triplet_ts frame_ts img")


def _scene_dir(scene: str) -> Path | None:
    """Directory holding a scene's clips: data/<scene> or data/captures/<scene>."""
    for base in (DATA_DIR / scene, CAPTURES_DIR / scene):
        if base.is_dir():
            return base
    return None


def _frame_ts_from_id(frame_id: str) -> tuple[str | None, str | None]:
    """(triplet_ts, frame_ts) from a `<scene>/<ts>/ts=HH-MM-SS.f` frame_id.

    Returns (triplet_ts, None) for the legacy `<scene>/<ts>/<idx>` scheme.
    """
    parts = frame_id.split("/")
    if len(parts) != 3:
        return None, None
    triplet_ts, sel = parts[1], parts[2]
    if sel.startswith("ts="):
        return triplet_ts, sel[3:].replace("-", ":", 2)
    return triplet_ts, None


@dataclass
class FrameLabel:
    frame_id: str
    scene: str
    source: str = "manual"
    audited: bool = True
    fisheye_bboxes: list[dict] = field(default_factory=list)
    obstacle_bins_fisheye: list[int] = field(default_factory=list)
    width: int = 0
    height: int = 0
    triplet_ts: str | None = None
    frame_ts: str | None = None
    # PLAN.md §I.4.11: safe heading the operator would steer to right now.
    # `None` means the scene has no safe direction in our FOV.
    safe_heading_deg: float | None = None
    safe_heading_set: bool = False


def _derive_bins(bboxes: list[dict], image_w: int, image_h: int,
                 cx: float, pix_deg_ratio: float, fusion_params: dict
                 ) -> list[int]:
    edges, centers = make_bins(fusion_params)
    bins: set[int] = set()
    for bb in bboxes:
        x0, y0, x1, y1 = bb["xyxy"]
        cx_px = (x0 + x1) / 2.0 * image_w
        angle = (cx_px - cx) / pix_deg_ratio
        idx = angle_to_bin(angle, edges)
        if idx is not None:
            bins.add(int(centers[idx]))
    return sorted(bins)


def _append_label(lbl: FrameLabel, path: Path | None = None):
    if path is None:
        path = MANUAL_PATH
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "frame_id": lbl.frame_id,
        "scene": lbl.scene,
    }
    # triplet_ts / frame_ts carried when known, matching the dashboard + Qwen
    # record shape exactly.
    if lbl.triplet_ts is not None:
        record["triplet_ts"] = lbl.triplet_ts
    if lbl.frame_ts is not None:
        record["frame_ts"] = lbl.frame_ts
    record.update({
        "source": lbl.source,
        "audited": lbl.audited,
        "fisheye_bboxes": lbl.fisheye_bboxes,
        "obstacle_bins_fisheye": lbl.obstacle_bins_fisheye,
        "width": lbl.width,
        "height": lbl.height,
    })
    if lbl.safe_heading_set:
        record["safe_heading_deg"] = lbl.safe_heading_deg
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


class LabelSession:
    def __init__(self, frames: list[Frame], intrinsics: dict, detection: dict):
        self.frames = frames
        self.intrinsics = intrinsics["fisheye"]
        self.fusion_params = detection["fusion"]
        self.idx = 0
        self.active_cls = LABEL_CLASSES[0] if LABEL_CLASSES else "boat"
        self.bboxes: list[dict] = []
        self.drag_start: tuple[int, int] | None = None
        self.preview: tuple[int, int] | None = None
        self.safe_heading_deg: float | None = 0.0
        self.safe_heading_set: bool = False
        self.window = "labeller"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.preview = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x0, y0 = self.drag_start
            x1, y1 = x, y
            self.drag_start = None
            self.preview = None
            if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
                return
            frame_h, frame_w = self.frames[self.idx].img.shape[:2]
            self.bboxes.append({
                "cls": self.active_cls,
                "xyxy": [
                    min(x0, x1) / frame_w,
                    min(y0, y1) / frame_h,
                    max(x0, x1) / frame_w,
                    max(y0, y1) / frame_h,
                ],
            })

    def _draw(self):
        fr = self.frames[self.idx]
        canvas = fr.img.copy()
        h, w = canvas.shape[:2]
        for bb in self.bboxes:
            x0, y0, x1, y1 = bb["xyxy"]
            p0 = (int(x0 * w), int(y0 * h))
            p1 = (int(x1 * w), int(y1 * h))
            color = CLASS_COLORS.get(bb["cls"], CLASS_COLORS["unset"])
            cv2.rectangle(canvas, p0, p1, color, 2)
            cv2.putText(canvas, bb["cls"], (p0[0], max(0, p0[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if self.drag_start and self.preview:
            cv2.rectangle(canvas, self.drag_start, self.preview, (255, 0, 0), 1)

        cx_px = float(self.intrinsics.get("cx", w / 2))
        pix_deg = float(self.intrinsics.get("pix_deg_ratio", 7.2))
        if self.safe_heading_set and self.safe_heading_deg is not None:
            head_x = int(round(cx_px + self.safe_heading_deg * pix_deg))
            if 0 <= head_x < w:
                cv2.arrowedLine(canvas, (head_x, h - 24), (head_x, h - 4),
                                (0, 255, 0), 2, tipLength=0.4)
            head_label = f"safe_heading={self.safe_heading_deg:+.0f}°"
        elif self.safe_heading_set and self.safe_heading_deg is None:
            head_label = "safe_heading=NULL (no safe direction)"
        else:
            head_label = "safe_heading=unset (, . to nudge, h to null)"
        cv2.putText(canvas, head_label, (5, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        status = (f"[{self.active_cls}]  {fr.frame_id}  "
                  f"frame {self.idx+1}/{len(self.frames)}  bboxes={len(self.bboxes)}")
        cv2.putText(canvas, status, (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.window, canvas)

    def _save_current(self, none: bool = False):
        fr = self.frames[self.idx]
        h, w = fr.img.shape[:2]
        bboxes = [] if none else [bb for bb in self.bboxes if bb["cls"] != "unset"]
        if not none and len(bboxes) < len(self.bboxes):
            print(f"!! dropped {len(self.bboxes) - len(bboxes)} bboxes without a class")
        bins = _derive_bins(
            bboxes, w, h,
            self.intrinsics["cx"], self.intrinsics["pix_deg_ratio"],
            self.fusion_params,
        )
        lbl = FrameLabel(
            frame_id=fr.frame_id, scene=fr.scene, source="manual", audited=True,
            fisheye_bboxes=bboxes, obstacle_bins_fisheye=bins, width=w, height=h,
            triplet_ts=fr.triplet_ts, frame_ts=fr.frame_ts,
            safe_heading_deg=self.safe_heading_deg if self.safe_heading_set else None,
            safe_heading_set=self.safe_heading_set,
        )
        _append_label(lbl)
        head_s = "—"
        if self.safe_heading_set:
            head_s = (f"{self.safe_heading_deg:+.0f}°"
                      if self.safe_heading_deg is not None else "null")
        print(f"saved {fr.frame_id}  bboxes={len(bboxes)}  bins={bins}  head={head_s}")

    def _reset_per_frame(self):
        self.bboxes = []
        self.safe_heading_deg = 0.0
        self.safe_heading_set = False

    def _nudge_heading(self, delta_deg: float):
        if self.safe_heading_deg is None:
            self.safe_heading_deg = 0.0
        self.safe_heading_deg = float(max(-55.0, min(55.0,
                                          self.safe_heading_deg + delta_deg)))
        self.safe_heading_set = True

    def _apply_class_key(self, key: int):
        """Class keys set the active class AND reclass the most recent bbox."""
        cls = KEY_TO_CLASS[key]
        self.active_cls = cls
        if self.bboxes:
            self.bboxes[-1]["cls"] = cls

    def run(self):
        while 0 <= self.idx < len(self.frames):
            self._draw()
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in KEY_TO_CLASS:
                self._apply_class_key(key)
            elif key == ord(","):
                self._nudge_heading(-5.0)
            elif key == ord("."):
                self._nudge_heading(+5.0)
            elif key == ord("<"):
                self._nudge_heading(-1.0)
            elif key == ord(">"):
                self._nudge_heading(+1.0)
            elif key == ord("h"):
                if self.safe_heading_set and self.safe_heading_deg is None:
                    self.safe_heading_deg = 0.0
                else:
                    self.safe_heading_deg = None
                self.safe_heading_set = True
            elif key == ord("n"):
                self._save_current()
                self._reset_per_frame()
                self.idx += 1
            elif key == ord(" "):
                self._save_current(none=True)
                self._reset_per_frame()
                self.idx += 1
            elif key in (8, 127):   # backspace / delete
                if self.idx > 0:
                    self.idx -= 1
                    self._reset_per_frame()
            elif key == ord("u") and self.bboxes:
                self.bboxes.pop()
            elif key == ord("s"):
                self._reset_per_frame()
                self.idx += 1
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()


# Note: class keys (b/d/B/p/m/o) collide with the labelling shortcuts, so the
# active-class default ('boat') is fine; 'm' is structure. 's' stays "skip".


def _iter_triplet_frames(triplet_prefix: str, every: int,
                         intrinsics: dict, detection: dict) -> list[Frame]:
    """Step a clip via the Qwen flow's iterate path → ``ts=`` frame_ids.

    Works for both canonical scenes (data/<scene>/<ts>) and mission clips
    (data/captures/<mission>/<ts>).
    """
    prefix = Path(triplet_prefix)
    captures_dir, clip_ts = prefix.parent, prefix.name
    out: list[Frame] = []
    for frame_id, scene, triplet_ts, frame_ts, und in iter_clip_frames(
            captures_dir, clip_ts, every, detection, intrinsics):
        out.append(Frame(frame_id, scene, triplet_ts, frame_ts, und))
    return out


def _iter_dir_frames(frames_dir: str) -> list[Frame]:
    p = Path(frames_dir)
    scene = p.parent.name if p.parent.name else "Unknown"
    out: list[Frame] = []
    for f in sorted(p.glob("*.jpg")):
        img = cv2.imread(str(f))
        if img is None:
            continue
        out.append(Frame(f"{scene}/{p.name}/{f.stem}", scene, None, None, img))
    return out


def _seek_frame(video_path: Path, frame_idx: int):
    """Seek a video to a specific frame index (legacy index scheme)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _load_provisional(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("audited"):
                continue
            out.append(obj)
    # Group by clip so the per-clip frame cache stays warm.
    out.sort(key=lambda o: o["frame_id"])
    return out


class AuditSession:
    """Walk a list of provisional labels; accept / reject / edit each.

    Resolves frames through the same path the Qwen labeller used, so the
    rendered frame matches the one the model saw byte-for-byte. Handles both
    the dashboard/Qwen `ts=` frame_id scheme and the legacy index scheme.
    """

    def __init__(self, provisional: list[dict], intrinsics: dict, detection: dict):
        self.items = provisional
        self.idx = 0
        self.intrinsics_full = intrinsics
        self.intrinsics = intrinsics["fisheye"]
        self.detection = detection
        self.fusion_params = detection["fusion"]
        self.K = intrinsics["fisheye"]["K"]
        self.D = intrinsics["fisheye"]["D"]
        self.window = "audit"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self.counts = {"accept": 0, "reject": 0, "edit": 0, "skip": 0}
        self._clip_key: tuple[str, str] | None = None
        self._clip_cache: dict[str, np.ndarray] = {}

    def _load_clip(self, scene: str, triplet_ts: str) -> dict[str, np.ndarray]:
        """Cache every wanted `ts=` frame of one clip in a single decode pass."""
        cdir = _scene_dir(scene)
        if cdir is None:
            return {}
        wanted = {it["frame_id"] for it in self.items
                  if it["frame_id"].startswith(f"{scene}/{triplet_ts}/ts=")}
        cache: dict[str, np.ndarray] = {}
        if not wanted:
            return cache
        try:
            for fid, _sc, _tts, _fts, und in iter_clip_frames(
                    cdir, triplet_ts, 1, self.detection, self.intrinsics_full):
                if fid in wanted:
                    cache[fid] = und
                    if len(cache) >= len(wanted):
                        break
        except Exception as e:  # noqa: BLE001 - bad clip: leave cache empty
            print(f"!! could not load clip {scene}/{triplet_ts}: {e}")
        return cache

    def _resolve_frame(self, frame_id: str):
        parts = frame_id.split("/")
        if len(parts) != 3:
            return None
        scene, triplet_ts, sel = parts
        if sel.startswith("ts="):
            key = (scene, triplet_ts)
            if self._clip_key != key:
                self._clip_cache = self._load_clip(scene, triplet_ts)
                self._clip_key = key
            return self._clip_cache.get(frame_id)
        # Legacy index scheme: <scene>/<ts>/<idx>.
        cdir = _scene_dir(scene)
        if cdir is None:
            return None
        try:
            triplet = resolve_triplet(cdir / triplet_ts)
        except FileNotFoundError:
            return None
        raw = _seek_frame(triplet.fisheye, int(sel.lstrip("0") or "0"))
        if raw is None:
            return None
        return undistort_fisheye(raw, self.K, self.D)

    def _draw(self, item: dict, img):
        canvas = img.copy()
        h, w = canvas.shape[:2]
        for bb in item.get("fisheye_bboxes", []):
            x0, y0, x1, y1 = bb["xyxy"]
            p0 = (int(x0 * w), int(y0 * h))
            p1 = (int(x1 * w), int(y1 * h))
            color = CLASS_COLORS.get(bb.get("cls"), CLASS_COLORS["unset"])
            cv2.rectangle(canvas, p0, p1, color, 2)
            cv2.putText(canvas, bb.get("cls", "?"), (p0[0], max(0, p0[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        status = (
            f"{item['frame_id']}  src={item.get('source','?')}  "
            f"n_bboxes={len(item.get('fisheye_bboxes', []))}  "
            f"{self.idx+1}/{len(self.items)}  "
            f"a=accept r=reject e=edit n=skip q=quit"
        )
        cv2.putText(canvas, status, (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.window, canvas)

    def _label_from_item(self, item: dict, img, *, bboxes, source, bins=None):
        h, w = img.shape[:2]
        triplet_ts, frame_ts = _frame_ts_from_id(item["frame_id"])
        return FrameLabel(
            frame_id=item["frame_id"], scene=item["scene"], source=source,
            audited=True, fisheye_bboxes=bboxes,
            obstacle_bins_fisheye=(bins if bins is not None
                                   else item.get("obstacle_bins_fisheye", [])),
            width=w, height=h,
            triplet_ts=item.get("triplet_ts", triplet_ts),
            frame_ts=item.get("frame_ts", frame_ts),
        )

    def _accept(self, item: dict, img):
        _append_label(self._label_from_item(
            item, img, bboxes=item.get("fisheye_bboxes", []),
            source=item.get("source", "qwen3-vl")), MASTER_PATH)
        self.counts["accept"] += 1

    def _reject(self, item: dict, img):
        _append_label(self._label_from_item(
            item, img, bboxes=[], bins=[],
            source=item.get("source", "qwen3-vl")), MASTER_PATH)
        self.counts["reject"] += 1

    def _edit(self, item: dict, img):
        triplet_ts, frame_ts = _frame_ts_from_id(item["frame_id"])
        frame = Frame(item["frame_id"], item["scene"],
                      item.get("triplet_ts", triplet_ts),
                      item.get("frame_ts", frame_ts), img)
        sess = LabelSession([frame], self.intrinsics_full, self.detection)
        sess.bboxes = list(item.get("fisheye_bboxes", []))
        while sess.idx == 0:
            sess._draw()
            key = cv2.waitKey(20) & 0xFF
            if key in KEY_TO_CLASS:
                sess._apply_class_key(key)
            elif key == ord("u") and sess.bboxes:
                sess.bboxes.pop()
            elif key == ord("n"):
                h, w = img.shape[:2]
                kept = [bb for bb in sess.bboxes if bb["cls"] != "unset"]
                bins = _derive_bins(
                    kept, w, h, self.intrinsics["cx"],
                    self.intrinsics["pix_deg_ratio"], self.fusion_params)
                _append_label(self._label_from_item(
                    item, img, bboxes=kept, bins=bins,
                    source=item.get("source", "qwen3-vl") + "+edited"),
                    MASTER_PATH)
                self.counts["edit"] += 1
                break
            elif key == ord("q"):
                break
        cv2.destroyWindow(sess.window)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

    def run(self):
        while 0 <= self.idx < len(self.items):
            item = self.items[self.idx]
            img = self._resolve_frame(item["frame_id"])
            if img is None:
                print(f"!! could not resolve {item['frame_id']}; skipping")
                self.idx += 1
                continue
            self._draw(item, img)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key == ord("a"):
                self._accept(item, img); self.idx += 1
            elif key == ord("r"):
                self._reject(item, img); self.idx += 1
            elif key == ord("e"):
                self._edit(item, img); self.idx += 1
            elif key == ord("n"):
                self.counts["skip"] += 1; self.idx += 1
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()
        print(f"audit summary: {self.counts}")
        n_total = sum(self.counts.values())
        if n_total:
            ratio = self.counts["accept"] / n_total
            print(f"acceptance ratio: {ratio:.2f}  -> drop the source if < 0.8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--triplet", help="Triplet prefix to step through.")
    src.add_argument("--frames", help="Directory of pre-extracted .jpg frames.")
    src.add_argument("--audit", help="JSONL of provisional labels to review.")
    ap.add_argument("--every", type=int, default=30,
                    help="Stride for --triplet mode (1 = every frame).")
    args = ap.parse_args()

    intrinsics = load_intrinsics()
    detection = load_detection()

    if args.audit:
        provisional = _load_provisional(Path(args.audit))
        if not provisional:
            raise SystemExit(f"no un-audited entries in {args.audit}")
        print(f"auditing {len(provisional)} entries, writing audited labels to {MASTER_PATH}")
        AuditSession(provisional, intrinsics, detection).run()
        return

    if args.triplet:
        frames = _iter_triplet_frames(args.triplet, args.every, intrinsics, detection)
    else:
        frames = _iter_dir_frames(args.frames)

    if not frames:
        raise SystemExit("no frames to label")
    print(f"opening {len(frames)} frames, writing to {MANUAL_PATH}")
    LabelSession(frames, intrinsics, detection).run()


if __name__ == "__main__":
    main()
