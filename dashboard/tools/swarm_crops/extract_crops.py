"""Extract boat-detection crops for the swarm self-recognition review gallery.

Feeds the D.1 crop classifier (docs/plans/swarm_self_recognition.md, decision
D2: classify crops on top of existing detections). For every `boat` box in the
union teacher labels of one scene, cut a padded crop out of the already
exported undistorted 864x648 fisheye frames and write a manifest the /crops
dashboard page reads.

Join discipline: the crop's frame is looked up by EXACT frame_id ts — never
nearest-timestamp. A label frame with no exported jpg is counted and skipped
(curation cuts produce exactly this: `iterate_triplet` does not export cut
frames). Chunks whose curation set is deleted are skipped entirely.

Ordering: manifest rows are chunk-chronological, and LARGEST crop first
within each chunk — big/near crops are the decidable ones; the hopeless
specks come last.

    .venv/bin/python dashboard/tools/swarm_crops/extract_crops.py \
        --scene 2026-08-26_afloat \
        --frames-root <scratch>/dart_ab_export \
        --out <scratch>/swarm_crops/2026-08-26
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

IMG_W, IMG_H = 864, 648
PAD_FRAC = 0.15
MIN_SIDE_PX = 16


def crop_rect(
    xyxy: list[float],
    img_w: int = IMG_W,
    img_h: int = IMG_H,
    pad_frac: float = PAD_FRAC,
    min_side_px: int = MIN_SIDE_PX,
) -> tuple[int, int, int, int] | None:
    """Normalised xyxy box -> padded integer pixel rect, or None to skip.

    Pads each side by `pad_frac` of the box's own dimension, clamps to the
    image, and refuses crops whose longest side lands under `min_side_px`
    (below the plan's ~16 px information limit nothing is decidable).
    Pure — unit-tested in tests/test_swarm_crops.py.
    """
    x0, y0, x1, y1 = xyxy
    if not (x1 > x0 and y1 > y0):
        return None
    bx0, by0, bx1, by1 = x0 * img_w, y0 * img_h, x1 * img_w, y1 * img_h
    pw, ph = (bx1 - bx0) * pad_frac, (by1 - by0) * pad_frac
    px0 = max(0, int(round(bx0 - pw)))
    py0 = max(0, int(round(by0 - ph)))
    px1 = min(img_w, int(round(bx1 + pw)))
    py1 = min(img_h, int(round(by1 + ph)))
    if px1 - px0 < 1 or py1 - py0 < 1:
        return None
    if max(px1 - px0, py1 - py0) < min_side_px:
        return None
    return px0, py0, px1, py1


def crop_id_for(frame_id: str, box_idx: int) -> str:
    """Stable id: sha1 of frame_id + index of the box in the label record."""
    return hashlib.sha1(f"{frame_id}|{box_idx}".encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="2026-08-26_afloat")
    ap.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "labels" / "union")
    ap.add_argument("--frames-root", type=Path, required=True,
                    help="root holding <scene>__<chunk>/ts=*.jpg exports")
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir: crops/ + manifest.jsonl")
    ap.add_argument("--cls", default="boat")
    ap.add_argument("--manifest-only", action="store_true",
                    help="regenerate the manifest without rewriting crop JPEGs")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT))
    import cv2  # noqa: PLC0415

    from scripts.utils.curation import in_cut, load_curation  # noqa: PLC0415

    curation = load_curation()

    crops_dir = args.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(args.labels_dir.glob(f"det_{args.scene}__*.jsonl"))
    if not label_files:
        print(f"no label files for scene {args.scene} in {args.labels_dir}")
        return 1

    rows = []  # manifest rows, ordered later
    n_boxes = n_small = n_noframe = n_cut = n_deleted = 0
    frame_cache: dict[str, object] = {}

    for lf in label_files:
        chunk = lf.stem.split("__", 1)[1]
        clip_id = f"{args.scene}/{chunk}"
        if curation.is_deleted(clip_id):
            n_del_chunk = sum(1 for _ in open(lf))
            n_deleted += n_del_chunk
            print(f"{chunk}: curation-deleted set — skipped ({n_del_chunk} records)")
            continue
        cuts = curation.cuts_for(clip_id)
        # frame dir: exports use <scene>__<chunk>; tolerate historical
        # <scene>_recovered__<chunk> dirs as a fallback
        frame_dir = args.frames_root / f"{args.scene}__{chunk}"
        alt_dir = args.frames_root / f"{args.scene}_recovered__{chunk}"
        chunk_rows = []
        for line in open(lf):
            rec = json.loads(line)
            frame_id = rec["frame_id"]
            ts_token = frame_id.split("/")[-1]  # "ts=HH-MM-SS.f"
            frame_ts = rec.get("frame_ts") or ts_token[3:].replace("-", ":")
            boats = [
                (i, b)
                for i, b in enumerate(rec.get("fisheye_bboxes", []))
                if b.get("cls") == args.cls
            ]
            if not boats:
                continue
            if in_cut(cuts, frame_ts):
                n_cut += len(boats)
                continue
            jpg = frame_dir / f"{ts_token}.jpg"
            if not jpg.exists():
                jpg = alt_dir / f"{ts_token}.jpg"
            if not jpg.exists():
                n_noframe += len(boats)
                continue
            img = None
            for i, b in boats:
                n_boxes += 1
                rect = crop_rect(b["xyxy"])
                if rect is None:
                    n_small += 1
                    continue
                x0, y0, x1, y1 = rect
                cid = crop_id_for(frame_id, i)
                if not args.manifest_only:
                    if img is None:
                        img = cv2.imread(str(jpg))
                        if img is None:
                            n_noframe += 1
                            break
                    crop = img[y0:y1, x0:x1]
                    cv2.imwrite(
                        str(crops_dir / f"{cid}.jpg"),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 90],
                    )
                chunk_rows.append(
                    {
                        "crop_id": cid,
                        "frame_id": frame_id,
                        "chunk": chunk,
                        "ts": frame_ts,
                        "xyxy": b["xyxy"],
                        # the padded crop window in frame pixels: the saved
                        # jpg is exactly frame[y0:y1, x0:x1] — the /crops
                        # overlay places the source box inside it with this
                        "window": [x0, y0, x1, y1],
                        "crop_w": x1 - x0,
                        "crop_h": y1 - y0,
                        "area_px": (x1 - x0) * (y1 - y0),
                        "confidence": b.get("confidence"),
                        "source": b.get("source"),
                    }
                )
            del img
        # largest first within the chunk; chunks stay chronological
        chunk_rows.sort(key=lambda r: -r["area_px"])
        rows.extend(chunk_rows)
        print(f"{chunk}: {len(chunk_rows)} crops")
        frame_cache.clear()

    manifest = args.out / "manifest.jsonl"
    with open(manifest, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    sizes = sorted(max(r["crop_w"], r["crop_h"]) for r in rows)

    def pct(p):
        return sizes[min(len(sizes) - 1, int(p * len(sizes)))] if sizes else 0

    print(
        f"\n{len(rows)} crops from {n_boxes} eligible boxes "
        f"({n_small} below {MIN_SIDE_PX}px, {n_noframe} without an exported "
        f"frame, {n_cut} in curation cuts, {n_deleted} records in deleted sets)"
    )
    print(
        "longest-side px: "
        f"min {sizes[0] if sizes else 0} / p25 {pct(0.25)} / p50 {pct(0.5)} / "
        f"p75 {pct(0.75)} / p95 {pct(0.95)} / max {sizes[-1] if sizes else 0}"
    )
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
