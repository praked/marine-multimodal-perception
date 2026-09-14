"""GPU embedding pass for the swarm-crop classifier (runs ON a GPU node).

Embeds every crop JPEG with CLIP ViT-L/14 (openai/clip-vit-large-patch14,
768-d image_embeds) and writes ids + fp16 embeddings to one npz. The crop
corpus is frozen (70,458 crops of the 2026-08-26 outing), so this runs
ONCE per corpus; every retrain of the head afterwards is local + fast
(train_classifier.py --embeddings <npz>).

Preprocessing matches train_classifier.py exactly: pad to square (black),
resize 224 bicubic, CLIP mean/std.

Node-side (gpu-node, corpus venv):
    env HF_HOME=$S/hf_cache $S/corpus_venv/bin/python embed_gpu.py \
        --crops $S/swarm_crops/2026-08-26/crops \
        --out   $S/swarm_crops/clip_vitl14_embeds.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import CLIPVisionModelWithProjection

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPVisionModelWithProjection.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(dev).eval()

    paths = sorted(args.crops.glob("*.jpg"))
    print(f"{len(paths)} crops, device {dev}, model {args.model}")
    ids, embeds = [], []
    t0 = time.time()
    for s in range(0, len(paths), args.batch):
        chunk = paths[s : s + args.batch]
        x = np.zeros((len(chunk), 3, 224, 224), dtype=np.float32)
        for i, p in enumerate(chunk):
            img = Image.open(p).convert("RGB")
            side = max(img.size)
            sq = Image.new("RGB", (side, side))
            sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
            arr = np.asarray(sq.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
            x[i] = ((arr - MEAN) / STD).transpose(2, 0, 1)
        with torch.no_grad():
            out = model(pixel_values=torch.from_numpy(x).half().to(dev))
        embeds.append(out.image_embeds.float().cpu().numpy().astype(np.float16))
        ids.extend(p.stem for p in chunk)
        if (s // args.batch) % 20 == 0:
            done = s + len(chunk)
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(paths)} ({rate:.0f}/s)", flush=True)
    np.savez_compressed(args.out, ids=np.array(ids), embeds=np.concatenate(embeds))
    print(f"{len(ids)} embeddings -> {args.out} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
