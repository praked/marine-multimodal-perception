"""Swarm self-recognition crop classifier: frozen CLIP embeddings + a
calibrated logistic-regression head over AuthorTwo's /crops labels.

Pipeline (all CPU, no torch — onnxruntime only):
  1. Snapshot sail_crop_labels (PostgREST, service role from
     dashboard/.env.local), fold latest-per-crop, drop `skip`.
  2. Embed every crop JPEG (SSD copy) with the CLIP ViT-B/32 image tower
     (ONNX, Xenova/clip-vit-base-patch32 `onnx/vision_model.onnx`,
     sha256 fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40,
     cached at /Volumes/ROS2_SSD/asvproject/models/onnx_encoders/).
     Crops are padded to square (black) before the 224x224 resize so
     extreme aspect ratios keep the whole detection. Embeddings are
     cached (clip_embeds.npz beside the crops) so retrains are minutes.
  3. Stratified 80/20 held-out metrics (per-class precision/recall;
     ACCEPT precision/recall is the number that matters), then refit on
     ALL labels and predict the full 70,458-crop corpus.
  4. Write predictions.json ({crop_id: {label, p}}) beside the manifest.

The deployed embeddings were computed ONCE on a GPU node with the
stronger CLIP ViT-L/14 (embed_gpu.py on gpu-node; 768-d image_embeds,
identical preprocessing) and live at
/Volumes/ROS2_SSD/asvproject/swarm_crops/2026-08-26/clip_vitl14_embeds.npz.
The crop corpus is frozen, so they never need recomputing.

RETRAIN PATH (as labels accumulate — fully local, ~1 min):
    .venv/bin/python dashboard/tools/swarm_crops/train_classifier.py \
        --embeddings /Volumes/ROS2_SSD/asvproject/swarm_crops/2026-08-26/clip_vitl14_embeds.npz
    cd dashboard && node --env-file=.env.local \
        tools/swarm_crops/upload_predictions.mjs   # -> R2
The /crops page picks the new predictions.json up on refresh. Without
--embeddings the script falls back to embedding locally with the ONNX
ViT-B/32 tower (slow: ~2.5 h for the full corpus on this laptop).
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

SSD = Path("/Volumes/ROS2_SSD/asvproject")
CROPS_DIR = SSD / "swarm_crops" / "2026-08-26"
ENCODER = SSD / "models" / "onnx_encoders" / "clip_vit_b32_vision.onnx"
ENV_LOCAL = Path(__file__).resolve().parents[2] / ".env.local"

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
CLASSES = ["accept", "deny", "junk"]  # `skip` = abstain, excluded


def read_env(path: Path) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def snapshot_labels(env: dict) -> dict[str, str]:
    """Latest label per crop_id from sail_crop_labels (paged PostgREST)."""
    base = env["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/")
    key = env["SUPABASE_SERVICE_ROLE_KEY"]
    latest: dict[str, tuple[str, str]] = {}
    offset, page = 0, 1000
    while True:
        url = (
            f"{base}/rest/v1/sail_crop_labels"
            f"?select=crop_id,label,created_at&order=created_at.asc"
            f"&offset={offset}&limit={page}"
        )
        req = urllib.request.Request(
            url, headers={"apikey": key, "Authorization": f"Bearer {key}"}
        )
        rows = json.load(urllib.request.urlopen(req))
        for r in rows:
            prev = latest.get(r["crop_id"])
            if not prev or r["created_at"] >= prev[1]:
                latest[r["crop_id"]] = (r["label"], r["created_at"])
        if len(rows) < page:
            break
        offset += page
    return {cid: lab for cid, (lab, _) in latest.items()}


def preprocess(paths: list[Path]) -> np.ndarray:
    """JPEGs -> CLIP pixel batch. Pad to square (black), resize 224."""
    from PIL import Image

    out = np.zeros((len(paths), 3, 224, 224), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        side = max(img.size)
        sq = Image.new("RGB", (side, side))
        sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        arr = np.asarray(sq.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
        out[i] = ((arr - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)
    return out


def embed_all(crop_ids: list[str], batch: int = 64) -> np.ndarray:
    """CLIP image_embeds for every crop id, through the npz cache."""
    import onnxruntime as ort

    cache_path = CROPS_DIR / "clip_embeds.npz"
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        cached = dict(zip(z["ids"].tolist(), z["embeds"]))
    todo = [c for c in crop_ids if c not in cached]
    if todo:
        print(f"embedding {len(todo)} crops ({len(cached)} cached)…")
        sess = ort.InferenceSession(str(ENCODER), providers=["CPUExecutionProvider"])
        for s in range(0, len(todo), batch):
            ids = todo[s : s + batch]
            x = preprocess([CROPS_DIR / "crops" / f"{c}.jpg" for c in ids])
            emb = sess.run(None, {"pixel_values": x})[0]
            for c, e in zip(ids, emb):
                cached[c] = e.astype(np.float16)
            if (s // batch) % 50 == 0:
                print(f"  {s}/{len(todo)}")
        ids_arr = np.array(list(cached.keys()))
        np.savez_compressed(cache_path, ids=ids_arr,
                            embeds=np.stack([cached[c] for c in ids_arr.tolist()]))
        print(f"cache -> {cache_path}")
    X = np.stack([cached[c] for c in crop_ids]).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)  # unit-normalised


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=CROPS_DIR / "predictions.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embeddings", type=Path, default=None,
                    help="precomputed npz (ids, embeds) — e.g. the GPU "
                         "CLIP ViT-L/14 pass; skips local ONNX embedding")
    args = ap.parse_args(argv)

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    env = read_env(ENV_LOCAL)
    labels = snapshot_labels(env)
    manifest_ids = [
        json.loads(l)["crop_id"]
        for l in open(CROPS_DIR / "manifest.jsonl")
    ]
    in_corpus = set(manifest_ids)
    used = {c: l for c, l in labels.items() if l in CLASSES and c in in_corpus}
    n_skip = sum(1 for l in labels.values() if l == "skip")
    print(f"label snapshot: {len(labels)} crops "
          f"({ {c: sum(1 for l in used.values() if l == c) for c in CLASSES} }, skip {n_skip} excluded)")
    if len(used) < 50:
        print("not enough labels to train")
        return 1

    ids = list(used)
    y = np.array([CLASSES.index(used[c]) for c in ids])
    if args.embeddings:
        z = np.load(args.embeddings, allow_pickle=False)
        bank = dict(zip(z["ids"].tolist(), z["embeds"]))
        missing = [c for c in manifest_ids if c not in bank]
        if missing:
            print(f"WARNING: {len(missing)} manifest crops missing from "
                  f"{args.embeddings} — they get no prediction")
        def embed(cids):
            E = np.stack([bank[c] for c in cids]).astype(np.float32)
            return E / np.linalg.norm(E, axis=1, keepdims=True)
        ids = [c for c in ids if c in bank]
        y = np.array([CLASSES.index(used[c]) for c in ids])
        manifest_ids = [c for c in manifest_ids if c in bank]
        print(f"embeddings: {args.embeddings.name} "
              f"({next(iter(bank.values())).shape[0]}-d, {len(bank)} crops)")
        X = embed(ids)
    else:
        embed = embed_all
        X = embed(ids)

    def make_head():
        return CalibratedClassifierCV(
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
            cv=3, method="sigmoid",
        )

    # held-out metrics. Classes with <5 examples can't be stratified (a
    # single `f` press crashed this once): pin their rows into TRAIN and
    # stratify the rest — they still train the refit-on-all model below,
    # they just don't get held-out numbers until they grow.
    counts = np.bincount(y, minlength=len(CLASSES))
    rare = {c for c in range(len(CLASSES)) if 0 < counts[c] < 5}
    if rare:
        print(f"EXCLUDED from this retrain (n<5, need >=5 labels): "
              f"{ {CLASSES[c]: int(counts[c]) for c in sorted(rare)} }")
        keep = np.array([c not in rare for c in y])
        X, y = X[keep], y[keep]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    head = make_head().fit(Xtr, ytr)
    print("\nheld-out (20%, stratified):")
    print(classification_report(
        yte, head.predict(Xte),
        labels=list(range(len(CLASSES))), target_names=CLASSES,
        digits=3, zero_division=0,
    ))

    # deploy head: refit on ALL labels, predict the whole corpus
    head = make_head().fit(X, y)
    X_all = embed(manifest_ids)
    proba = head.predict_proba(X_all)
    col = proba.argmax(axis=1)
    # proba columns follow head.classes_ (a class absent from the labels so
    # far — e.g. junk before AuthorTwo uses f — has no column): map through it
    pred = np.asarray(head.classes_)[col]
    preds = {
        cid: {"label": CLASSES[int(k)], "p": round(float(proba[i, c]), 4)}
        for i, (cid, k, c) in enumerate(zip(manifest_ids, pred, col))
    }
    args.out.write_text(json.dumps(preds))
    dist = {c: int((pred == CLASSES.index(c)).sum()) for c in CLASSES}

    band = int((proba.max(axis=1) < 0.8).sum())
    agree = sum(1 for c, l in used.items() if preds[c]["label"] == l)
    print(f"predictions -> {args.out}")
    print(f"distribution over {len(manifest_ids)}: {dist}; max-p<0.8 band {band}")
    print(f"agreement with the training labels (in-sample): {agree}/{len(used)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
