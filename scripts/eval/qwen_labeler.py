"""Qwen-VL assisted labelling: Phase I.4.5 Track B.

Runs a vision-language model over fisheye frames and writes provisional
bounding-box labels. Audit the output with label_tool.py before merging
into labels/master.jsonl.

Two backends, picked at the CLI:
  - api          OpenAI-compatible chat-completions endpoint
                 (DashScope / OpenRouter / vLLM / etc.).
                 Reads OPENAI_API_KEY and OPENAI_BASE_URL from env.
  - transformers Local Qwen2.5-VL via Hugging Face transformers
                 (~15 GB model download on first use).

Usage:
    # API mode (Qwen via OpenRouter, say):
    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1
    python -m scripts.eval.qwen_labeler \\
        --triplet data/Boats/2025-06-23_16-21-07 \\
        --every 30 --backend api --model qwen/qwen-2.5-vl-72b-instruct

    # Local mode:
    python -m scripts.eval.qwen_labeler \\
        --triplet data/Boats/2025-06-23_16-21-07 \\
        --every 30 --backend transformers --model Qwen/Qwen2.5-VL-7B-Instruct

Output:
    labels/qwen.jsonl       provisional labels (audited=False)
    labels/cache/<hash>.json  raw responses (cached for cheap regen)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

from scripts.sensor_processing.pipeline import angle_to_bin, make_bins
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.cv_common import undistort_fisheye
from scripts.utils.datasets import REPO_ROOT, resolve_triplet

LABELS_DIR = REPO_ROOT / "labels"
QWEN_PATH = LABELS_DIR / "qwen.jsonl"
CACHE_DIR = LABELS_DIR / "cache"

PROMPT = (
    "You are labelling images for an obstacle-detection dataset taken from "
    "a small autonomous sailboat on a lake (the lake). Identify all "
    "obstacles ON THE WATER SURFACE: boats, people in the water, ducks, "
    "swans, buoys, floating debris. For each, return a JSON object with "
    "{\"class\": one of [boat, duck, buoy, person, debris, other], "
    "\"bbox_xyxy_normalised\": [x0,y0,x1,y1] in [0,1], \"confidence\": float}. "
    "Return a single JSON object: {\"obstacles\": [...]}. "
    "If there are NO obstacles below the horizon, return {\"obstacles\": []}. "
    "Be conservative: do NOT label ripples, sun glint on water, distant land, "
    "clouds, the horizon itself, or shoreline trees. Respond with JSON only."
)


# ---------------------------------------------------------------------------
# Frame iteration (shared with label_tool)
# ---------------------------------------------------------------------------

def iter_fisheye_frames(triplet_prefix: str, every: int, intrinsics):
    triplet = resolve_triplet(triplet_prefix)
    cap = cv2.VideoCapture(str(triplet.fisheye))
    K = intrinsics["fisheye"]["K"]
    D = intrinsics["fisheye"]["D"]
    if not cap.isOpened():
        raise SystemExit(f"cannot open {triplet.fisheye}")
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every == 0:
                und = undistort_fisheye(frame, K, D)
                frame_id = f"{triplet.scene}/{triplet.timestamp}/{idx:06d}"
                yield frame_id, triplet.scene, und
            idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Encode for VLM
# ---------------------------------------------------------------------------

def _frame_to_jpeg_bytes(img_bgr: np.ndarray, max_long_side: int = 1280) -> bytes:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_long_side / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _frame_hash(jpeg_bytes: bytes, model: str, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(prompt.encode())
    h.update(jpeg_bytes)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def call_api(jpeg_bytes: bytes, model: str) -> str:
    """OpenAI-compatible vision chat. Reads OPENAI_API_KEY / OPENAI_BASE_URL."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("pip install openai for --backend api") from e

    client = OpenAI()  # picks up env vars
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=800,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


_TF_MODEL = None
_TF_PROCESSOR = None


def call_transformers(jpeg_bytes: bytes, model: str) -> str:
    """Local Qwen2.5-VL via transformers. Lazy-loads model on first call."""
    global _TF_MODEL, _TF_PROCESSOR
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as e:
        raise SystemExit("pip install transformers torch pillow for --backend transformers") from e

    if _TF_MODEL is None:
        print(f"loading {model} (first call only)...")
        _TF_PROCESSOR = AutoProcessor.from_pretrained(model)
        _TF_MODEL = AutoModelForVision2Seq.from_pretrained(
            model, torch_dtype="auto", device_map="auto",
        )

    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}],
    }]
    text = _TF_PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _TF_PROCESSOR(text=[text], images=[image], return_tensors="pt").to(_TF_MODEL.device)
    out = _TF_MODEL.generate(**inputs, max_new_tokens=800, do_sample=False)
    decoded = _TF_PROCESSOR.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return decoded[0]


# ---------------------------------------------------------------------------
# Retry + cost guard
# ---------------------------------------------------------------------------

def call_with_retry(fn, jpeg_bytes: bytes, model: str,
                    max_attempts: int = 3, base_delay: float = 1.0) -> str:
    """Exponential backoff. Retries any exception except SystemExit."""
    import time
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(jpeg_bytes, model)
        except SystemExit:
            raise
        except Exception as e:
            last_err = e
            if attempt == max_attempts:
                break
            sleep = base_delay * (2 ** (attempt - 1))
            print(f"  attempt {attempt} failed: {e}; retrying in {sleep:.1f}s")
            time.sleep(sleep)
    raise RuntimeError(f"backend failed after {max_attempts} attempts: {last_err}")


def estimate_cost_usd(n_calls: int, model: str, price_per_call_usd: float | None) -> float:
    """Rough cost estimate. Caller supplies the per-call price; we just multiply.
    The model is logged so cost discussions are reproducible.
    """
    if price_per_call_usd is None:
        return 0.0
    return n_calls * price_per_call_usd


# ---------------------------------------------------------------------------
# Parse + persist
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract the first {...} block; tolerate code-fenced or chatty wrappers."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not brace:
        raise ValueError(f"no JSON object in response: {text[:200]}...")
    return json.loads(brace.group(0))


def _qwen_to_record(frame_id: str, scene: str, img_shape: tuple[int, int, int],
                    parsed: dict, model: str, fusion_params: dict,
                    intrinsics_fisheye: dict) -> dict:
    h, w = img_shape[:2]
    bboxes = []
    for ob in parsed.get("obstacles", []):
        cls = str(ob.get("class", "other"))
        xyxy = ob.get("bbox_xyxy_normalised") or ob.get("bbox_xyxy") or ob.get("bbox")
        if not xyxy or len(xyxy) != 4:
            continue
        # Tolerate either normalised or pixel coords.
        x0, y0, x1, y1 = xyxy
        if max(x0, y0, x1, y1) > 1.5:
            x0, x1 = x0 / w, x1 / w
            y0, y1 = y0 / h, y1 / h
        bboxes.append({"cls": cls, "xyxy": [float(x0), float(y0), float(x1), float(y1)],
                       "confidence": float(ob.get("confidence", 0.0))})

    edges, centers = make_bins(fusion_params)
    bins = set()
    for bb in bboxes:
        x0, y0, x1, y1 = bb["xyxy"]
        cx_px = (x0 + x1) / 2.0 * w
        angle = (cx_px - intrinsics_fisheye["cx"]) / intrinsics_fisheye["pix_deg_ratio"]
        idx = angle_to_bin(angle, edges)
        if idx is not None:
            bins.add(int(centers[idx]))

    return {
        "frame_id": frame_id,
        "scene": scene,
        "source": "qwen-vl",
        "model_version": model,
        "audited": False,
        "fisheye_bboxes": bboxes,
        "obstacle_bins_fisheye": sorted(bins),
        "width": w,
        "height": h,
    }


def _process_frame(item, backend_fn, model, max_attempts, dry_run, intr_fish, fusion_params):
    """Per-frame worker. Returns (frame_id, record_dict_or_None, error_or_None).

    Cached frames are loaded from disk; uncached calls go through retry.
    """
    frame_id, scene, img = item
    jpeg = _frame_to_jpeg_bytes(img)
    h = _frame_hash(jpeg, model, PROMPT)
    cache_path = CACHE_DIR / f"{h}.json"

    from_cache = False
    if cache_path.exists():
        response_text = json.loads(cache_path.read_text())["raw"]
        from_cache = True
    elif dry_run:
        return frame_id, None, "dry-run"
    else:
        try:
            response_text = call_with_retry(backend_fn, jpeg, model, max_attempts)
        except Exception as e:
            return frame_id, None, str(e)
        cache_path.write_text(json.dumps({
            "frame_id": frame_id, "model": model, "raw": response_text,
        }))

    try:
        parsed = _extract_json(response_text)
    except Exception as e:
        return frame_id, None, f"parse failed: {e}"

    rec = _qwen_to_record(frame_id, scene, img.shape, parsed, model,
                          fusion_params, intr_fish)
    rec["_from_cache"] = from_cache
    return frame_id, rec, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triplet", required=True)
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--backend", choices=("api", "transformers"), default="api")
    ap.add_argument("--model", default="qwen/qwen-2.5-vl-72b-instruct")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=None,
                    help="Hard cap on actual API/model calls (excludes cache hits).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers (api backend only).")
    ap.add_argument("--retries", type=int, default=3,
                    help="Per-call retry budget with exponential backoff.")
    ap.add_argument("--price-per-call-usd", type=float, default=None,
                    help="Optional per-call cost; if set, prints a running total.")
    ap.add_argument("--cost-cap-usd", type=float, default=None,
                    help="Abort if the projected total exceeds this.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Encode and hash frames but don't call the model.")
    args = ap.parse_args()

    intrinsics = load_intrinsics()
    detection = load_detection()
    fusion_params = detection["fusion"]
    intr_fish = intrinsics["fisheye"]

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = open(QWEN_PATH, "a")

    if args.backend == "transformers" and args.workers > 1:
        print("warning: --workers > 1 with --backend transformers is ignored "
              "(model can't be shared across threads); using 1")
        args.workers = 1

    backend_fn = call_api if args.backend == "api" else call_transformers

    # Collect frames into a list so we can fan out.
    frames = list(iter_fisheye_frames(args.triplet, args.every, intrinsics))
    if args.max_frames:
        frames = frames[:args.max_frames]

    n_written = 0
    n_calls = 0   # actual model calls (not cache hits)
    n_failed = 0

    def _consume(result):
        nonlocal n_written, n_calls, n_failed
        frame_id, rec, err = result
        if err:
            if err != "dry-run":
                print(f"!! {frame_id}: {err}")
                n_failed += 1
            return
        is_cached = rec.pop("_from_cache", False)
        if not is_cached:
            n_calls += 1
        out.write(json.dumps(rec) + "\n")
        out.flush()
        n_written += 1
        tag = "cache" if is_cached else "call"
        print(f"  [{tag}] {frame_id}  n_bboxes={len(rec['fisheye_bboxes'])}  bins={rec['obstacle_bins_fisheye']}")

        if args.cost_cap_usd is not None and args.price_per_call_usd:
            cost = estimate_cost_usd(n_calls, args.model, args.price_per_call_usd)
            if cost >= args.cost_cap_usd:
                raise SystemExit(f"cost cap hit: ${cost:.4f} >= ${args.cost_cap_usd:.4f}")

    try:
        if args.workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = []
                for item in frames:
                    # Unreachable: n_calls is only bumped in _consume(), which
                    # runs after the submit loop, so it is always 0 here. The
                    # real cap is enforced in the as_completed loop below.
                    if args.max_calls and n_calls >= args.max_calls:  # pragma: no cover
                        break
                    futures.append(ex.submit(
                        _process_frame, item, backend_fn, args.model,
                        args.retries, args.dry_run, intr_fish, fusion_params,
                    ))
                for f in as_completed(futures):
                    _consume(f.result())
                    if args.max_calls and n_calls >= args.max_calls:
                        # Best-effort cancel of any futures not yet running.
                        for pending in futures:
                            pending.cancel()
                        break
        else:
            for item in frames:
                if args.max_calls and n_calls >= args.max_calls:
                    break
                _consume(_process_frame(
                    item, backend_fn, args.model, args.retries,
                    args.dry_run, intr_fish, fusion_params,
                ))
    finally:
        out.close()

    print(f"\nwrote {n_written} labels  (calls={n_calls}  failed={n_failed})")
    if args.price_per_call_usd:
        cost = estimate_cost_usd(n_calls, args.model, args.price_per_call_usd)
        print(f"estimated cost: ${cost:.4f} ({n_calls} calls @ ${args.price_per_call_usd:.4f})")
    print(f"output -> {QWEN_PATH}")


if __name__ == "__main__":
    main()
