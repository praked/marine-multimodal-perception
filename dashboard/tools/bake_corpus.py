"""Bake the FULL clip corpus into a dashboard bundle (for the private R2
bucket). Same per-clip layout as bake_demo.py, but:

- walks every triplet list_triplets() knows (2025 scenes + all capture
  missions, including the SSD-native 2026-08-* sessions),
- CONCATENATES successive capture chunks into one "activity": chunks of the
  same scene whose end-to-start gap is <= GAP_S (the capture service rolls
  chunks back-to-back) merge into a single clip with a seamless timeline.
  Per-frame `chunk` attribution keeps frame_ids round-tripping to the right
  capture files. Separately-started captures (calibration stations, takes)
  stay separate,
- every frame at half resolution by default (EVERY_BY_SCENE overrides the
  sampling for low-value footage like the bench soak),
- attaches whatever artefacts exist per chunk: seg masks, typed detections,
  instance masks, qwen/GroundingDINO labels, scorer sectors (quad clip).

Output goes to the SSD (NOT the repo): the bundle is a derived, regenerable
serving copy.

    .venv/bin/python dashboard/tools/bake_corpus.py [--out DIR] [--only PREFIX]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bake_demo import (  # noqa: E402
    FISHEYE_SCALE,
    JPEG_Q,
    frame_ts_of,
    load_jsonl_by_ts,
    round_pts,
    safe_ts,
    write_json,
)

from scripts.sensor_processing.pipeline import iterate_triplet  # noqa: E402
from scripts.utils.calibration import load_detection, load_intrinsics  # noqa: E402
from scripts.utils.curation import Curation, load_curation  # noqa: E402
from scripts.utils.cv_common import undistort_fisheye  # noqa: E402
from scripts.utils.datasets import Triplet, list_triplets  # noqa: E402
from scripts.utils.segmentation import load_seg_mask  # noqa: E402

DEFAULT_OUT = Path("/Volumes/ROS2_SSD/asvproject/dashboard_bundle")

SKIP_SCENES = ("eval_smoke36", "pi_card_backup",
               "2026-07-14_stability",  # bench soak: not dashboard material
               "2026-08-21_bench")  # indoor bench (ingest-pipeline test)
MIN_YEAR = 2026  # dashboard shows current-campaign data only (AuthorTwo's call)

# Chunk/set exclusions and trims are DATA, not code: `configs/curation.yaml`
# (exported from the dashboard's sail_curation table by `pnpm
# curation:export`). The former hard-coded EXCLUDE_CHUNK_IDS list (AuthorTwo's
# 2026-08-21 per-chunk first/last-frame screening) was migrated into that
# table verbatim, so nothing excluded before reappears. A set deleted in the
# dashboard is not baked (its member chunks are dropped BEFORE activity
# grouping); a set's cut ranges are not baked either. Frame ids never change.
EVERY_BY_SCENE: dict[str, int] = {}
# Derived from configs/detection.yaml (15-degree bins since D.2 2026-08-24);
# per-record bin_centers_deg in sectors.json is authoritative for consumers.
def _bin_centers():
    from scripts.sensor_processing.pipeline import make_bins
    from scripts.utils.calibration import load_detection
    _, centers = make_bins(load_detection()["fusion"])
    return [float(c) for c in centers]

BIN_CENTERS = _bin_centers()
GAP_S = 45.0  # max end-to-start gap for chunks to count as one activity
# (covers the capture service's chunk-roll overhead, measured up to ~21 s on
#  the degraded 2026-08-19 tail; separately-started captures are minutes apart)
NOMINAL_FPS = 3.0

SECTOR_SOURCES = {
    # (2026-09-02) the poster-era quad_scorer.jsonl override is retired: the
    # v1b re-emission covers the quad clip with the LIVE scorer like every
    # other clip, so the directory convention applies uniformly.
}
# Directory-convention fallback: per-chunk scored sector JSONLs emitted by the
# fusion pipeline (gpu-node night run + reruns) as <scene>__<ts>.jsonl. The
# explicit SECTOR_SOURCES entries above win. Since 2026-09-05 the files here
# are copies of results/sectors_v1b_oofcal_v2/ (fixed self-clutter zone +
# out-of-fold-calibrated v1b bundle; see docs/history/
# 2026-09-05_sector_reemission.md); results/sectors_v1b/ keeps the 09-02/03
# generation (leaky zone, train-fit calibrator) for comparison.
SECTOR_DIR = REPO_ROOT / "results" / "sectors_night"
TITLE_OVERRIDES = {
    "2026-07-08/2026-07-08_16-37-01": "Quad clip — dockside full stack",
}
# Default label source = labels/union (DART/SAM3 ∪ GroundingDINO, per-box
# `source`, SAM3 polygons): the published suggestion set since 2026-08-31 and
# the teacher verdict of 2026-09-02. --labels-dir swaps it (labels/qwen is
# the frozen DINO set). The default was labels/qwen until 2026-09-03: the
# 09-02 sector rebake ran without the flag and silently reverted every
# activity it touched to DINO-only suggestions on the dashboard.
LABELS_DIR = REPO_ROOT / "labels" / "union"
LABEL_FILES = sorted(LABELS_DIR.glob("*.jsonl"))


def clip_key(scene: str, ts: str) -> str:
    return f"{scene}__{ts}"


# Missions whose chunks live in sub-session directories (list_triplets only
# walks flat mission dirs). Sub-dirs listed are on-water material; the
# indoor/bench diagnostics (calibration poses, fx_walk stations, lab
# variants, water_test) are deliberately not baked.
NESTED_MISSIONS: dict[str, tuple[str, ...]] = {
    "2026-08-19_afloat": ("session", "canoe_fx", "imu_rock", "clutter_ab"),
    # 2026-09-08: per-boot session folders (capture writes captures/<stamp>/);
    # only the evening trials session is corpus material (morning = street/lab).
    "2026-09-08_afloat": ("2026-09-08_17-10-55",),
}


def discover_nested() -> list[Triplet]:
    out: list[Triplet] = []
    for mission, subs in NESTED_MISSIONS.items():
        mdir = REPO_ROOT / "data" / "captures" / mission
        for sub in subs:
            sdir = mdir / sub
            if not sdir.is_dir():
                continue
            for f in sorted(sdir.glob("fisheye_*.mp4")):
                ts = f.stem.removeprefix("fisheye_")

                def sib(kind: str, ext: str) -> Path | None:
                    p = sdir / f"{kind}_{ts}.{ext}"
                    return p if p.exists() else None

                out.append(Triplet(
                    scene=f"{mission}_{sub}",
                    timestamp=ts,
                    fisheye=f,
                    thermal=sib("thermal", "mp4"),
                    mmwave=sib("mmwave", "csv"),
                    imu=sib("imu", "csv"),
                    frames=sib("frames", "csv"),
                ))
    return out


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")


def _span_from_frames_csv(path: Path) -> float | None:
    """Wall-clock span of the frames sidecar. Preferred duration source: on
    the degraded 2026-08-19 tail the video held 30 frames (10 s at nominal
    fps) while the chunk really spanned 5 minutes — container metadata lies
    whenever the capture loop slows down."""
    try:
        lines = path.read_text().strip().splitlines()
        if len(lines) < 3:
            return None
        header = lines[0].split(",")
        t = header.index("Time")

        def secs(row: str) -> float:
            hh, mm, ss = row.split(",")[t].split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(ss)

        span = secs(lines[-1]) - secs(lines[1])
        return span if span > 0 else None
    except Exception:
        return None


def chunk_duration_s(triplet: Triplet) -> float:
    if triplet.frames is not None:
        span = _span_from_frames_csv(triplet.frames)
        if span is not None:
            return span
    cap = cv2.VideoCapture(str(triplet.fisheye))
    try:
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or NOMINAL_FPS
        if fps <= 0:
            fps = NOMINAL_FPS
        return float(n) / fps
    finally:
        cap.release()


def group_activities(triplets: list[Triplet], gap_s: float = GAP_S) -> list[list[Triplet]]:
    """Group same-scene chunks whose end-to-start gap is <= gap_s."""
    ordered = sorted(triplets, key=lambda t: (t.scene, t.timestamp))
    groups: list[list[Triplet]] = []
    current: list[Triplet] = []
    prev_end: datetime | None = None
    for t in ordered:
        start = parse_ts(t.timestamp)
        if (
            current
            and t.scene == current[-1].scene
            and prev_end is not None
            and (start - prev_end).total_seconds() <= gap_s
        ):
            current.append(t)
        else:
            if current:
                groups.append(current)
            current = [t]
        prev_end = start + timedelta(seconds=max(chunk_duration_s(t), 0.0))
    if current:
        groups.append(current)
    return groups


def labels_for_chunk(triplet_ts: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in LABEL_FILES:
        for ts, rec in load_jsonl_by_ts(path, frame_ts_of).items():
            if rec.get("triplet_ts") == triplet_ts:
                out[ts] = rec
    return out


def _bake_chunk_into(
    triplet: Triplet,
    first_ts: str,
    clip_dir: Path,
    every: int,
    K,
    D,
    detection,
    frames_index: list[dict],
    sectors: dict[str, dict],
    radar: dict[str, list],
    boxes: dict[str, list],
    instances: dict[str, list],
    label_out: dict[str, dict],
    curation: Curation | None = None,
) -> dict[str, bool]:
    """Bake one chunk's frames + artefacts into the shared activity dicts.
    Returns which optional streams this chunk contributed.

    Curation cuts are applied HERE (not by iterate_triplet) so the
    `every`-th sampling stays anchored to the raw video index; a cut frame
    contributes nothing — no image, no radar group, no artefacts."""
    if curation is None:
        curation = load_curation()
    mkey = clip_key(triplet.scene, triplet.timestamp)
    seg_dir = REPO_ROOT / "data" / "seg" / mkey
    typed_all = load_jsonl_by_ts(
        REPO_ROOT / "data" / "det" / f"{mkey}.jsonl", frame_ts_of)
    inst_all = load_jsonl_by_ts(
        REPO_ROOT / "data" / "det_seg" / f"{mkey}.jsonl", frame_ts_of)
    labels = labels_for_chunk(triplet.timestamp)
    sectors_all: dict[str, dict] = {}
    src = SECTOR_SOURCES.get(triplet.clip_id)
    if not (src and src.exists()):
        src = SECTOR_DIR / f"{clip_key(triplet.scene, triplet.timestamp)}.jsonl"
    if src and src.exists() and src.stat().st_size > 0:
        sectors_all = load_jsonl_by_ts(src, lambda r: r.get("timestamp"))
    is_first = triplet.timestamp == first_ts
    has_thermal = False
    has_radar = False

    for i, (fts, fish, therm, pts) in enumerate(
            iterate_triplet(triplet, detection, respect_curation=False)):
        if not curation.keep_frame(triplet.clip_id, fts):
            continue
        if pts is not None and len(pts):
            has_radar = True
            radar[fts] = round_pts(pts.tolist())
        if i % every or fish is None:
            continue
        safe = safe_ts(fts)
        und = undistort_fisheye(fish, K, D)
        small = cv2.resize(und, None, fx=FISHEYE_SCALE, fy=FISHEYE_SCALE,
                           interpolation=cv2.INTER_AREA)
        (clip_dir / "frames").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(clip_dir / "frames" / f"ts={safe}.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        entry: dict = {"ts": fts, "fisheye": True, "thermal": False,
                       "seg": False}
        if not is_first:
            entry["chunk"] = triplet.timestamp

        if therm is not None:
            (clip_dir / "thermal").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clip_dir / "thermal" / f"ts={safe}.jpg"),
                        therm, [cv2.IMWRITE_JPEG_QUALITY, 80])
            entry["thermal"] = True
            has_thermal = True
            # Degraded-but-present gate: the quality-guard contrast floors
            # (detection.yaml thermal.quality_guard: std 25 / p99-p1 120).
            # Day-1's misted cover reads std ~24 / dyn ~109; healthy water
            # scenes read 55-65 / 200+. Flag only the degraded case.
            tg = (cv2.cvtColor(therm, cv2.COLOR_BGR2GRAY)
                  if therm.ndim == 3 else therm)
            if float(tg.std()) < 25.0 or                     float(np.percentile(tg, 99) - np.percentile(tg, 1)) < 120.0:
                entry["thermal_q"] = "low"

        mask_path = seg_dir / f"ts={safe}.png"
        if mask_path.exists():
            seg = load_seg_mask(mask_path)
            seg_small = cv2.resize(seg, (small.shape[1], small.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            (clip_dir / "seg").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clip_dir / "seg" / f"ts={safe}.png"), seg_small)
            entry["seg"] = True

        if fts in sectors_all:
            sectors[fts] = sectors_all[fts]
        rec = typed_all.get(fts)
        if rec:
            boxes[fts] = [
                {"cls": b["cls"], "xyxy": [round(v, 4) for v in b["xyxy"]],
                 "confidence": b.get("confidence"),
                 "source": rec.get("source", "yolo")}
                for b in rec.get("fisheye_bboxes", [])
            ]
        irec = inst_all.get(fts)
        if irec:
            instances[fts] = [
                {"cls": p["cls"], "confidence": p.get("confidence"),
                 "polygon": [round(v, 4) for v in p.get("polygon", [])]}
                for p in irec.get("instances", [])
            ]
        lrec = labels.get(fts)
        if lrec:
            label_out[fts] = lrec
            if fts not in boxes and lrec.get("fisheye_bboxes"):
                boxes[fts] = [
                    {"cls": b["cls"], "xyxy": [round(v, 4) for v in b["xyxy"]],
                     "confidence": b.get("confidence"),
                     "source": lrec.get("source", "label")}
                    for b in lrec["fisheye_bboxes"]
                ]
        frames_index.append(entry)
    return {"thermal": has_thermal, "radar": has_radar}


def select_chunks(triplets: list[Triplet], curation: Curation,
                  skip_scenes: tuple[str, ...] = SKIP_SCENES,
                  min_year: int = MIN_YEAR) -> list[Triplet]:
    """The chunks the corpus bake considers: dashboard-policy scenes only
    (bench/smoke scenes and pre-campaign years stay code policy) minus
    every chunk that curation marks deleted — directly or through the
    activity it belongs to — BEFORE activity grouping."""
    return [t for t in triplets
            if not any(t.scene.startswith(s) for s in skip_scenes)
            and parse_ts(t.timestamp).year >= min_year
            and not curation.is_deleted(t.clip_id)]


def bake_activity(members: list[Triplet], out_root: Path, intr, detection,
                  force: bool = False,
                  curation: Curation | None = None) -> dict | None:
    if curation is None:
        curation = load_curation()
    first = members[0]
    scene = first.scene
    key = clip_key(scene, first.timestamp)
    clip_dir = out_root / key
    every = EVERY_BY_SCENE.get(scene, 1)

    # Incremental: an existing bake with the same chunk set is reused.
    meta_path = clip_dir / "meta.json"
    if meta_path.exists() and not force:
        try:
            import json
            old = json.loads(meta_path.read_text())
            if old.get("chunks") == [m.timestamp for m in members]:
                print(f"  reuse {first.clip_id}: {old['n_frames']} frames")
                return old
        except Exception:
            pass
    K, D = intr["fisheye"]["K"], intr["fisheye"]["D"]

    frames_index: list[dict] = []
    sectors: dict[str, dict] = {}
    radar: dict[str, list] = {}
    boxes: dict[str, list] = {}
    instances: dict[str, list] = {}
    label_out: dict[str, dict] = {}
    has_thermal = False
    has_radar = False
    has_imu = False

    for member in members:
        # A corrupt thermal must not cost the whole chunk: retry fisheye-only
        # (the dashboard then shows the honest THERMAL DOWN state).
        for triplet in (member, dataclasses.replace(member, thermal=None)):
            try:
                baked = _bake_chunk_into(
                    triplet, first.timestamp, clip_dir, every, K, D, detection,
                    frames_index, sectors, radar, boxes, instances, label_out,
                    curation=curation)
                has_thermal = has_thermal or baked["thermal"]
                has_radar = has_radar or baked["radar"]
                has_imu = has_imu or triplet.imu is not None
                break
            except Exception as exc:
                if triplet.thermal is None:
                    print(f"  skip chunk {member.clip_id}: {exc}")

    if not frames_index:
        print(f"  SKIP {first.clip_id}: no readable frames")
        return None

    chunks = [m.timestamp for m in members]
    streams = {"fisheye": True, "thermal": has_thermal, "radar": has_radar,
               "imu": has_imu, "seg": any(f["seg"] for f in frames_index),
               "sectors": bool(sectors)}
    n_chunks = len(chunks)
    meta = {
        "clip_id": first.clip_id,
        "scene": scene,
        "triplet_ts": first.timestamp,
        "title": TITLE_OVERRIDES.get(first.clip_id, f"{scene} · {first.timestamp}"),
        "description": (f"{scene} activity"
                        + (f" of {n_chunks} back-to-back chunks" if n_chunks > 1 else "")
                        + (f", every {every}th frame" if every > 1 else "")
                        + ". Streams: "
                        + ", ".join(k for k, v in streams.items() if v) + "."),
        "image_size": [432, 324],
        "native_size": [864, 648],
        "thermal_size": [160, 120] if has_thermal else None,
        "streams": streams,
        "bin_centers_deg": BIN_CENTERS,
        "n_frames": len(frames_index),
        "n_labelled": len(label_out),
        "thumb_ts": frames_index[len(frames_index) // 2]["ts"],
        "chunks": chunks,
        "end_ts": chunks[-1],
        "frames": frames_index,
    }
    write_json(clip_dir / "meta.json", meta)
    write_json(clip_dir / "sectors.json", sectors)
    write_json(clip_dir / "radar.json", radar)
    write_json(clip_dir / "boxes.json", boxes)
    write_json(clip_dir / "instances.json", instances)
    write_json(clip_dir / "labels.json", label_out)
    print(f"  {first.clip_id}: {n_chunks} chunk(s), {len(frames_index)} frames"
          f" (radar {len(radar)}, boxes {len(boxes)}, labels {len(label_out)})")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true",
                    help="rebake even when the chunk set is unchanged "
                         "(needed when seg/det/labels/sectors artefacts changed)")
    ap.add_argument("--only", default=None,
                    help="bake only activities whose first clip_id contains this")
    ap.add_argument("--labels-dir", default=None,
                    help="label JSONL dir to bake (default labels/union)")
    ap.add_argument("--ignore-curation", action="store_true",
                    help="bake deleted sets and cut frames anyway (debugging; "
                         "the published bundle must honour configs/curation.yaml)")
    args = ap.parse_args()
    if args.labels_dir:
        global LABELS_DIR, LABEL_FILES
        LABELS_DIR = Path(args.labels_dir)
        LABEL_FILES = sorted(LABELS_DIR.glob("*.jsonl"))
    args.out.mkdir(parents=True, exist_ok=True)

    intr = load_intrinsics()
    detection = load_detection()
    curation = Curation.empty() if args.ignore_curation else load_curation()
    all_chunks = list_triplets(respect_curation=False) + discover_nested()
    triplets = select_chunks(all_chunks, curation)
    n_deleted = sum(1 for t in all_chunks if curation.is_deleted(t.clip_id))
    groups = group_activities(triplets)
    print(f"{len(triplets)} chunks -> {len(groups)} activities"
          f" (curation: {n_deleted} deleted chunk(s) skipped"
          f"{'' if curation.source else ', no configs/curation.yaml'})")

    metas: list[dict] = []
    for group in groups:
        if args.only and args.only not in group[0].clip_id:
            # Not this run's target, but the catalogue must stay COMPLETE:
            # reuse the existing bake if one exists (a --only session ingest
            # must never clobber clips.json down to one mission).
            existing = args.out / clip_key(group[0].scene, group[0].timestamp) / "meta.json"
            if existing.exists():
                import json
                old = json.loads(existing.read_text())
                # A meta from the per-chunk era (no `chunks`) is not a usable
                # catalogue row: bake the activity instead of crashing the
                # catalogue write on it (2026-09-03).
                if "chunks" in old:
                    metas.append(old)
                    continue
                print(f"  stale meta for {group[0].clip_id} (pre-activity bake): rebaking")
            else:
                continue
        meta = bake_activity(group, args.out, intr, detection, force=args.force,
                             curation=curation)
        if meta:
            metas.append(meta)

    # reverse chronological: newest activity first
    metas.sort(key=lambda m: m["triplet_ts"], reverse=True)
    catalogue = [
        {k: m.get(k) for k in ("clip_id", "scene", "triplet_ts", "title",
                               "description", "streams", "n_frames",
                               "n_labelled", "thumb_ts", "chunks", "end_ts")}
        for m in metas
    ]
    write_json(args.out / "clips.json", {"clips": catalogue})
    total = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"{len(metas)} activities baked, {total / 1e9:.2f} GB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
