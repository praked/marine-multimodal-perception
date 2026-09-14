"""Bake a small self-contained demo bundle for the web dashboard.

Reads the SAME artefacts the Tkinter dashboard consumes (via the repo's own
pipeline: iterate_triplet -> clip_overrides -> undistort_fisheye) and writes a
static bundle under dashboard/public/demo/. The bundle format is identical to
what dashboard/tools/ingest uploads to Supabase Storage, so the web app has a
single loader for both.

Bundle layout (per clip):
    <out>/<scene>__<triplet_ts>/
        meta.json        clip metadata + ordered frame index
        frames/ts=<safe>.jpg      undistorted fisheye, scaled
        thermal/ts=<safe>.jpg     thermal frame (clip_overrides applied)
        seg/ts=<safe>.png         label-encoded mask (0 obstacle/1 water/2 sky)
        sectors.json     {ts: sector-protocol record}    (kept frames)
        radar.json       {ts: [[x,y,z,v], ...]}          (ALL radar frames)
        boxes.json       {ts: [{cls,xyxy,confidence,source}]}
        instances.json   {ts: [{cls,confidence,polygon}]}
        labels.json      {ts: label record}              (annotation source)
plus a top-level clips.json catalogue.

Run from the repo root with the project venv:
    .venv/bin/python dashboard/tools/bake_demo.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.sensor_processing.pipeline import iterate_triplet  # noqa: E402
from scripts.utils.calibration import load_detection, load_intrinsics  # noqa: E402
from scripts.utils.cv_common import undistort_fisheye  # noqa: E402
from scripts.utils.datasets import resolve_triplet  # noqa: E402
from scripts.utils.segmentation import load_seg_mask  # noqa: E402

OUT_ROOT = REPO_ROOT / "dashboard" / "public" / "demo"
FISHEYE_SCALE = 0.5  # 864x648 -> 432x324
JPEG_Q = 68


def safe_ts(ts: str) -> str:
    return ts.replace(":", "-")


def round_pts(rows, ndigits=2):
    return [[round(float(v), ndigits) for v in row] for row in rows]


def load_jsonl_by_ts(path: Path, ts_of) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = ts_of(rec)
            if key is not None:
                out[key] = rec
    return out


def frame_ts_of(rec: dict) -> str | None:
    ts = rec.get("frame_ts")
    if ts:
        return ts
    fid = rec.get("frame_id", "")
    if "ts=" in fid:
        return fid.split("ts=")[-1].replace("-", ":", 2)
    return None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))


def bake_quad_clip(every: int = 6) -> dict:
    """The 2026-07-08 quad clip: fisheye+thermal+radar+IMU+seg+scorer."""
    scene, ts = "2026-07-08", "2026-07-08_16-37-01"
    clip_dir = OUT_ROOT / f"{scene}__{ts}"
    if clip_dir.exists():
        shutil.rmtree(clip_dir)

    triplet = resolve_triplet(REPO_ROOT / "data" / "captures" / scene / ts)
    intr = load_intrinsics()
    detection = load_detection()
    K, D = intr["fisheye"]["K"], intr["fisheye"]["D"]

    sectors_all = load_jsonl_by_ts(
        REPO_ROOT / "poster" / "render" / "quad_scorer.jsonl",
        lambda r: r.get("timestamp"),
    )
    boxes_all = load_jsonl_by_ts(
        REPO_ROOT / "labels" / "qwen" / "det_2026-07-08.jsonl", frame_ts_of
    )
    seg_dir = REPO_ROOT / "data" / "seg" / f"{scene}__{ts}"

    frames_index: list[dict] = []
    sectors: dict[str, dict] = {}
    radar: dict[str, list] = {}
    boxes: dict[str, list] = {}
    labels: dict[str, dict] = {}

    for i, (fts, fish, therm, pts) in enumerate(iterate_triplet(triplet, detection)):
        if pts is not None and len(pts):
            radar[fts] = round_pts(pts.tolist())
        if i % every:
            continue
        if fish is None:
            continue
        safe = safe_ts(fts)
        und = undistort_fisheye(fish, K, D)
        small = cv2.resize(und, None, fx=FISHEYE_SCALE, fy=FISHEYE_SCALE,
                           interpolation=cv2.INTER_AREA)
        (clip_dir / "frames").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(clip_dir / "frames" / f"ts={safe}.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])

        entry = {"ts": fts, "fisheye": True, "thermal": False, "seg": False}
        if therm is not None:
            (clip_dir / "thermal").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clip_dir / "thermal" / f"ts={safe}.jpg"), therm,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            entry["thermal"] = True

        mask_path = seg_dir / f"ts={safe}.png"
        if mask_path.exists():
            seg = load_seg_mask(mask_path)
            seg_small = cv2.resize(
                seg, (small.shape[1], small.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            (clip_dir / "seg").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clip_dir / "seg" / f"ts={safe}.png"), seg_small)
            entry["seg"] = True

        if fts in sectors_all:
            sectors[fts] = sectors_all[fts]
        rec = boxes_all.get(fts)
        if rec:
            boxes[fts] = [
                {"cls": b["cls"], "xyxy": [round(v, 4) for v in b["xyxy"]],
                 "confidence": b.get("confidence"),
                 "source": rec.get("source", "grounding-dino")}
                for b in rec.get("fisheye_bboxes", [])
            ]
            labels[fts] = rec
        frames_index.append(entry)

    meta = {
        "clip_id": f"{scene}/{ts}",
        "scene": scene,
        "triplet_ts": ts,
        "title": "Quad clip — dockside full stack",
        "description": ("First 4-sensor capture (fisheye + thermal + radar Doppler "
                        "+ IMU) with learned-scorer sectors, 2026-07-08 dockside."),
        "image_size": [432, 324],
        "native_size": [864, 648],
        "thermal_size": [160, 120],
        "streams": {"fisheye": True, "thermal": True, "radar": True,
                    "imu": True, "seg": True, "sectors": True},
        "bin_centers_deg": [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
        "n_frames": len(frames_index),
        "n_labelled": len(labels),
        "thumb_ts": frames_index[min(44, len(frames_index) - 1)]["ts"],
        "frames": frames_index,
    }
    write_json(clip_dir / "meta.json", meta)
    write_json(clip_dir / "sectors.json", sectors)
    write_json(clip_dir / "radar.json", radar)
    write_json(clip_dir / "boxes.json", boxes)
    write_json(clip_dir / "labels.json", labels)
    write_json(clip_dir / "instances.json", {})
    print(f"quad clip: {len(frames_index)} frames, {len(radar)} radar groups")
    return meta


def bake_day1_clip(every: int = 20) -> dict:
    """Day-1 swimmer clip: pre-extracted undistorted frames + typed dets +
    instance masks. Radar-dead outing -> exercises the DOWN/degraded path."""
    scene, ts = "2026-06-17_institutionone_day1", "2026-06-17_13-16-25"
    key = f"{scene}__{ts}"
    clip_dir = OUT_ROOT / key
    if clip_dir.exists():
        shutil.rmtree(clip_dir)

    src_frames = sorted((REPO_ROOT / "data" / "seg_frames" / key).glob("ts=*.jpg"))
    seg_dir = REPO_ROOT / "data" / "seg" / key
    typed_all = load_jsonl_by_ts(
        REPO_ROOT / "data" / "det" / f"{key}.jsonl", frame_ts_of)
    inst_all = load_jsonl_by_ts(
        REPO_ROOT / "data" / "det_seg" / f"{key}.jsonl", frame_ts_of)
    qwen_all = load_jsonl_by_ts(
        REPO_ROOT / "labels" / "qwen" / "qwen_2026-06-17_institutionone_day1.jsonl",
        frame_ts_of)

    frames_index: list[dict] = []
    boxes: dict[str, list] = {}
    instances: dict[str, list] = {}
    labels: dict[str, dict] = {}

    for path in src_frames[::every]:
        safe = path.stem.split("ts=")[-1]
        fts = safe.replace("-", ":", 2)
        img = cv2.imread(str(path))
        if img is None:
            continue
        small = cv2.resize(img, None, fx=FISHEYE_SCALE, fy=FISHEYE_SCALE,
                           interpolation=cv2.INTER_AREA)
        (clip_dir / "frames").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(clip_dir / "frames" / f"ts={safe}.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        entry = {"ts": fts, "fisheye": True, "thermal": False, "seg": False}

        mask_path = seg_dir / f"ts={safe}.png"
        if mask_path.exists():
            seg = load_seg_mask(mask_path)
            seg_small = cv2.resize(seg, (small.shape[1], small.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            (clip_dir / "seg").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clip_dir / "seg" / f"ts={safe}.png"), seg_small)
            entry["seg"] = True

        rec = typed_all.get(fts)
        if rec:
            boxes[fts] = [
                {"cls": b["cls"], "xyxy": [round(v, 4) for v in b["xyxy"]],
                 "confidence": b.get("confidence"), "source": "yolo"}
                for b in rec.get("fisheye_bboxes", [])
            ]
        irec = inst_all.get(fts)
        if irec:
            instances[fts] = [
                {"cls": p["cls"], "confidence": p.get("confidence"),
                 "polygon": [round(v, 4) for v in p.get("polygon", [])]}
                for p in irec.get("instances", [])
            ]
        qrec = qwen_all.get(fts)
        if qrec:
            labels[fts] = qrec
        frames_index.append(entry)

    meta = {
        "clip_id": f"{scene}/{ts}",
        "scene": scene,
        "triplet_ts": ts,
        "title": "Day 1 — swimmer + typed detections",
        "description": ("InstitutionOne day-1 clip with YOLO typed boxes, YOLOv8-seg "
                        "instances and Qwen labels. Radar-dead outing: the radar "
                        "panel shows the honest DOWN state."),
        "image_size": [432, 324],
        "native_size": [864, 648],
        "thermal_size": None,
        "streams": {"fisheye": True, "thermal": False, "radar": False,
                    "imu": False, "seg": True, "sectors": False},
        "bin_centers_deg": [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
        "n_frames": len(frames_index),
        "n_labelled": len(labels),
        "thumb_ts": frames_index[min(18, len(frames_index) - 1)]["ts"],
        "frames": frames_index,
    }
    write_json(clip_dir / "meta.json", meta)
    write_json(clip_dir / "sectors.json", {})
    write_json(clip_dir / "radar.json", {})
    write_json(clip_dir / "boxes.json", boxes)
    write_json(clip_dir / "instances.json", instances)
    write_json(clip_dir / "labels.json", labels)
    print(f"day-1 clip: {len(frames_index)} frames, "
          f"{len(boxes)} typed, {len(instances)} instance frames")
    return meta


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    metas = [bake_quad_clip(), bake_day1_clip()]
    catalogue = [
        {k: m[k] for k in ("clip_id", "scene", "triplet_ts", "title",
                           "description", "streams", "n_frames",
                           "n_labelled", "thumb_ts")}
        for m in metas
    ]
    write_json(OUT_ROOT / "clips.json", {"clips": catalogue})
    total = sum(p.stat().st_size for p in OUT_ROOT.rglob("*") if p.is_file())
    print(f"bundle total: {total / 1e6:.1f} MB -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
