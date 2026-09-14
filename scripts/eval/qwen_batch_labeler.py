"""Autonomous Qwen3-VL batch labeller for InstTwo-GPU runs: Phase I.4.5 Track B.

This is the *scaled-up* sibling of ``scripts/eval/qwen_labeler.py``. Where that
tool labels a single triplet through a hosted API or a per-frame transformers
call, this module is built to run unattended on a InstTwo GPU node (gpu-node /
bluestreak, mirroring the SmolVLA-Testing workflow) over a whole mission's worth
of fisheye clips, using a locally loaded Qwen3-VL via **vLLM** (loaded once,
continuous-batched) for throughput.

What it does that matters for the labelling pipeline:

  * **Dashboard-compatible output.** Frames are iterated through the exact same
    ``iterate_triplet`` generator the dashboard uses, undistorted with the same
    intrinsics the dashboard labels on, and the ``frame_id`` is built with the
    dashboard's timestamp scheme (``<scene>/<triplet_ts>/ts=<HH-MM-SS.f>``).
    A Qwen label for frame *i* of a clip therefore shares its ``frame_id`` with
    a hand-drawn dashboard label of the same frame, so the audit
    (``label_tool.py --audit``) and ``metrics.py`` line them up automatically.
  * **Annotation-manual prompt.** The instruction encodes the five-class
    taxonomy and the "do not label" edge cases from
    ``docs/annotation_manual.tex`` plus the fisheye scene context from
    ``docs/fisheye_context.tex``.
  * **Resumable.** Frames already present in the output JSONL are skipped, and
    raw model responses are cached per-frame, so a killed / re-queued run picks
    up where it left off without re-spending GPU time.

Output schema (one JSON object per line, written to ``--out``)::

    {"frame_id": "<scene>/<triplet_ts>/ts=13-21-25.0",
     "scene": "<scene>", "triplet_ts": "...", "frame_ts": "13:21:25.0",
     "source": "qwen3-vl", "model_version": "Qwen/Qwen3-VL-30B-A3B-Instruct",
     "audited": false,
     "fisheye_bboxes": [{"cls": "boat", "xyxy": [..], "confidence": 0.9}],
     "obstacle_bins_fisheye": [-15, -5], "width": 864, "height": 648}

Typical GPU-node invocation (see scripts/gpu_labeling/04_run_labeling.sh)::

    python -m scripts.eval.qwen_batch_labeler \\
        --captures-dir /scratch0/$USER/asvproject/data/captures/2026-06-17_institutionone_day1 \\
        --from 2026-06-17_12-41-23 --to 2026-06-17_13-41-26 \\
        --every 1 --backend vllm \\
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \\
        --out /scratch0/$USER/asvproject/out/qwen_master.jsonl \\
        --cache-dir /scratch0/$USER/asvproject/out/cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from scripts.sensor_processing.pipeline import (
    angle_to_bin,
    iterate_triplet,
    make_bins,
)
from scripts.utils.calibration import load_detection, load_intrinsics
from scripts.utils.cv_common import undistort_fisheye
from scripts.utils.datasets import REPO_ROOT, resolve_triplet

# Reuse the pure encode/parse helpers from the single-clip labeller so the two
# tools stay in lockstep on JPEG encoding, hashing, and tolerant JSON parsing.
from scripts.eval.qwen_labeler import (
    _extract_json,
    _frame_to_jpeg_bytes,
)

# ---------------------------------------------------------------------------
# Editable prompt + taxonomy (configs/qwen_label_prompt.yaml)
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_FILE = REPO_ROOT / "configs" / "qwen_label_prompt.yaml"

#: Minimal fallback used only if the YAML is missing (bare checkout).
_FALLBACK_PROMPT = {
    "coords": "auto",
    "classes": ["boat", "duck", "buoy", "person", "structure", "other"],
    "synonyms": {},
    "system": "You label water obstacles for ASVProject. Respond with JSON only.",
    "user": ('Return {"obstacles": [{"class": ..., "bbox": [x0,y0,x1,y1], '
             '"confidence": ...}]} with absolute pixel coords in an 864x648 image.'),
}

#: Active prompt/taxonomy, populated from the YAML at import. Overridable via
#: load_prompt_config() (CLI --prompt-file). The backends, the class
#: normaliser, and the cache key all read from this so a prompt edit takes
#: effect everywhere and busts the cache.
PROMPT_CFG: dict = {}


def load_prompt_config(path: str | Path | None = None) -> dict:
    """Load the editable prompt YAML into the module-global ``PROMPT_CFG``."""
    p = Path(path) if path else DEFAULT_PROMPT_FILE
    try:
        import yaml
        cfg = yaml.safe_load(p.read_text())
    except FileNotFoundError:
        cfg = dict(_FALLBACK_PROMPT)
    cfg.setdefault("coords", "auto")
    cfg["classes"] = [str(c).lower() for c in cfg.get("classes", [])]
    cfg["synonyms"] = {str(k).strip().lower(): str(v).strip().lower()
                       for k, v in (cfg.get("synonyms") or {}).items()}
    PROMPT_CFG.clear()
    PROMPT_CFG.update(cfg)
    return PROMPT_CFG


def prompt_fingerprint() -> str:
    """Cache-key material: any system/user prompt edit busts the frame cache."""
    return f"{PROMPT_CFG.get('system', '')}\n{PROMPT_CFG.get('user', '')}"


# Load the default prompt at import so the helpers below work standalone.
load_prompt_config()


# ---------------------------------------------------------------------------
# Frame iteration over a clip range
# ---------------------------------------------------------------------------

def list_clip_timestamps(captures_dir: Path, lo: str | None, hi: str | None) -> list[str]:
    """Return sorted clip timestamps under ``captures_dir`` within [lo, hi].

    Bounds are inclusive and compared lexicographically: safe because the
    ``YYYY-MM-DD_HH-MM-SS`` timestamp format sorts chronologically as a string.
    """
    stamps = sorted(
        p.stem[len("fisheye_"):]
        for p in captures_dir.glob("fisheye_*.mp4")
    )
    if lo is not None:
        stamps = [s for s in stamps if s >= lo]
    if hi is not None:
        stamps = [s for s in stamps if s <= hi]
    return stamps


def iter_clip_frames(captures_dir: Path, clip_ts: str, every: int,
                     detection_cfg: dict, intrinsics: dict):
    """Yield ``(frame_id, scene, triplet_ts, frame_ts, undistorted_bgr)`` for a clip.

    Uses the dashboard's ``iterate_triplet`` so the frames and their synthesized
    timestamps are identical to what a human labels in the dashboard, then
    undistorts with the loaded fisheye intrinsics (matching the undistorted
    canvas the dashboard draws boxes on).
    """
    triplet = resolve_triplet(captures_dir / clip_ts)
    K = intrinsics["fisheye"]["K"]
    D = intrinsics["fisheye"]["D"]
    for i, (frame_ts, fisheye_bgr, _thermal, _pts) in enumerate(
            iterate_triplet(triplet, detection_cfg)):
        if i % every != 0:
            continue
        und = undistort_fisheye(fisheye_bgr, K, D)
        safe_ts = frame_ts.replace(":", "-")
        frame_id = f"{triplet.scene}/{triplet.timestamp}/ts={safe_ts}"
        yield frame_id, triplet.scene, triplet.timestamp, frame_ts, und


# ---------------------------------------------------------------------------
# Parse model output -> dashboard-compatible record
# ---------------------------------------------------------------------------

def _normalise_class(raw: str) -> str:
    key = str(raw).strip().lower()
    syn = PROMPT_CFG.get("synonyms", {})
    if key in syn:
        return syn[key]
    if key in PROMPT_CFG.get("classes", []):
        return key
    return "other"


def response_to_record(frame_id: str, scene: str, triplet_ts: str, frame_ts: str,
                       img_shape: tuple[int, ...], response_text: str, model: str,
                       fusion_params: dict, intr_fish: dict,
                       parse_wh: tuple[int, int] | None = None) -> dict:
    """Turn one raw model response into a dashboard-schema label record.

    ``parse_wh`` is the (width, height) the model actually grounded against:
    the image size AFTER qwen-vl-utils' smart_resize. Qwen3-VL reports absolute
    pixels in that processed resolution, not the original frame, so dividing by
    it (rather than the stored 864x648) removes a small systematic offset. Falls
    back to the original frame size when unknown.
    """
    parsed = _extract_json(response_text)
    h, w = img_shape[:2]
    pw, ph = parse_wh if parse_wh else (w, h)
    coords = PROMPT_CFG.get("coords", "auto")

    bboxes: list[dict] = []
    for ob in parsed.get("obstacles", []):
        xyxy = (ob.get("bbox") or ob.get("bbox_xyxy_normalised")
                or ob.get("bbox_xyxy"))
        if not xyxy or len(xyxy) != 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in xyxy)
        # Pixel -> normalised against the PROCESSED size. `absolute` always
        # divides; `auto` divides only when a value looks like pixels (>1.5);
        # `normalised` leaves as-is.
        if coords == "absolute" or (coords != "normalized"
                                    and max(x0, y0, x1, y1) > 1.5):
            x0, x1 = x0 / pw, x1 / pw
            y0, y1 = y0 / ph, y1 / ph
        # Order + clamp to [0,1].
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in (x0, y0, x1, y1))
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        bboxes.append({
            "cls": _normalise_class(ob.get("class", "other")),
            "xyxy": [x0, y0, x1, y1],
            "confidence": float(ob.get("confidence", 0.0)),
        })

    # obstacle_bins_fisheye: identical derivation to label_tool / dashboard.
    edges, centers = make_bins(fusion_params)
    bins: set[int] = set()
    for bb in bboxes:
        x0, _y0, x1, _y1 = bb["xyxy"]
        cx_px = (x0 + x1) / 2.0 * w
        angle = (cx_px - intr_fish["cx"]) / intr_fish["pix_deg_ratio"]
        idx = angle_to_bin(angle, edges)
        if idx is not None:
            bins.add(int(centers[idx]))

    return {
        "frame_id": frame_id,
        "scene": scene,
        "triplet_ts": triplet_ts,
        "frame_ts": frame_ts,
        "source": "qwen3-vl",
        "model_version": model,
        "audited": False,
        "fisheye_bboxes": bboxes,
        "obstacle_bins_fisheye": sorted(bins),
        "width": w,
        "height": h,
    }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class VLLMBackend:
    """Qwen3-VL via vLLM. Model loaded once; frames batched per ``generate``."""

    _AWQ = "Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ"

    def __init__(self, model: str | None, gpu_mem_util: float, max_model_len: int,
                 tensor_parallel_size: int, max_tokens: int,
                 min_pixels: int | None = None, max_pixels: int | None = None):
        self.model = model or self._resolve_default()
        self.max_tokens = max_tokens
        # Vision resolution knobs (pixels). Raising min_pixels makes qwen-vl-utils
        # upscale small frames -> a finer patch grid -> better small-object
        # localisation. None = library defaults.
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        from vllm import LLM, SamplingParams  # noqa: F401  (import error surfaced early)
        from vllm import LLM
        print(f"[vllm] loading {self.model} "
              f"(gpu_mem_util={gpu_mem_util}, max_model_len={max_model_len}, "
              f"tp={tensor_parallel_size}) ...", flush=True)
        t0 = time.time()
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 1},
            # 16 GB cards are tight once FP8 weights + the vision tower are
            # resident; skipping CUDA-graph capture frees ~1-2 GiB for the KV
            # cache. Offline batch labelling barely benefits from graphs anyway.
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print(f"[vllm] loaded in {time.time() - t0:.0f}s", flush=True)

    def _resolve_default(self) -> str:
        try:
            from huggingface_hub import model_info
            model_info(self._AWQ)
            return self._AWQ
        except Exception:
            return "Qwen/Qwen3-VL-30B-A3B-Instruct"

    def _build_request(self, img_bgr: np.ndarray):
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        img_dict: dict = {"type": "image", "image": pil}
        if self.min_pixels:
            img_dict["min_pixels"] = self.min_pixels
        if self.max_pixels:
            img_dict["max_pixels"] = self.max_pixels
        messages = [
            {"role": "system", "content": PROMPT_CFG["system"]},
            {"role": "user", "content": [
                img_dict,
                {"type": "text", "text": PROMPT_CFG["user"]},
            ]},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _video_inputs = process_vision_info(messages)
        # The processed (smart_resized) size the model actually grounds against.
        proc = image_inputs[0]
        proc_wh = (int(proc.width), int(proc.height))
        return {"prompt": text, "multi_modal_data": {"image": image_inputs}}, proc_wh

    def generate_batch(self, imgs_bgr: list[np.ndarray]) -> list[tuple[str, tuple[int, int]]]:
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=self.max_tokens, temperature=0.0,
                            repetition_penalty=1.05)
        built = [self._build_request(im) for im in imgs_bgr]
        reqs = [b[0] for b in built]
        sizes = [b[1] for b in built]
        outputs = self.llm.generate(reqs, sp)
        return [(o.outputs[0].text, sz) for o, sz in zip(outputs, sizes)]


class TransformersBackend:
    """Fallback: Qwen-VL via HF transformers, one frame at a time (slow)."""

    def __init__(self, model: str, max_tokens: int):
        import torch  # noqa: F401
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.model_id = model
        self.max_tokens = max_tokens
        print(f"[transformers] loading {model} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model, torch_dtype="auto", device_map="auto")
        print("[transformers] loaded", flush=True)

    def _generate_one(self, img_bgr: np.ndarray) -> str:
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        messages = [
            {"role": "system", "content": PROMPT_CFG["system"]},
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": PROMPT_CFG["user"]},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil],
                                return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                                  do_sample=False)
        decoded = self.processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return decoded[0]

    def generate_batch(self, imgs_bgr: list[np.ndarray]
                       ) -> list[tuple[str, tuple[int, int] | None]]:
        # None proc size -> response_to_record falls back to the frame size.
        return [(self._generate_one(im), None) for im in imgs_bgr]


# ---------------------------------------------------------------------------
# Cache + resume
# ---------------------------------------------------------------------------

def _frame_hash(jpeg_bytes: bytes, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(prompt_fingerprint().encode())
    h.update(jpeg_bytes)
    return h.hexdigest()[:16]


def _load_done_frame_ids(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["frame_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures-dir", required=True,
                    help="Mission folder holding fisheye_*.mp4 (absolute path "
                         "on the GPU node, e.g. /scratch0/$USER/.../institutionone_day1).")
    ap.add_argument("--from", dest="clip_from", default=None,
                    help="First clip timestamp to include (inclusive), e.g. "
                         "2026-06-17_12-41-23. Default: earliest clip.")
    ap.add_argument("--to", dest="clip_to", default=None,
                    help="Last clip timestamp to include (inclusive). Default: latest.")
    ap.add_argument("--every", type=int, default=1,
                    help="Frame stride within each clip (1 = every frame).")
    ap.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                    help="HF model id. For vllm, an empty/auto value resolves to "
                         "the AWQ build if reachable.")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--cache-dir", default=None,
                    help="Directory for raw per-frame response cache "
                         "(default: <out_dir>/cache).")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Frames per vLLM generate() call / write checkpoint.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Global cap on frames processed this run (for smoke tests).")
    ap.add_argument("--prompt-file", default=None,
                    help="Prompt/taxonomy YAML to use (default: "
                         "configs/qwen_label_prompt.yaml).")
    ap.add_argument("--frames-file", default=None,
                    help="Restrict to the frame_ids listed in this file (one per "
                         "line; '#' comments ok). Used by the prompt-iteration "
                         "smoke test to label a fixed comparison set.")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--min-pixels", type=int, default=None,
                    help="Min vision pixels (vllm). Raise to upscale small "
                         "frames for finer small-object localisation, e.g. 1200000.")
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="Max vision pixels (vllm).")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-label frames even if already present in --out.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Iterate + encode + cache-check but never load a model "
                         "or call it. Reports how many frames WOULD be labelled.")
    args = ap.parse_args()

    if args.prompt_file:
        load_prompt_config(args.prompt_file)
    print(f"prompt: coords={PROMPT_CFG.get('coords')} "
          f"classes={PROMPT_CFG.get('classes')}", flush=True)

    captures_dir = Path(args.captures_dir).expanduser().resolve()
    if not captures_dir.is_dir():
        raise SystemExit(f"--captures-dir not found: {captures_dir}")

    # Optional fixed frame set (prompt-iteration smoke test).
    frames_filter: set[str] | None = None
    if args.frames_file:
        frames_filter = set()
        for line in Path(args.frames_file).expanduser().read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                frames_filter.add(line)
        if not frames_filter:
            raise SystemExit(f"--frames-file is empty: {args.frames_file}")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = (Path(args.cache_dir).expanduser().resolve()
                 if args.cache_dir else out_path.parent / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = load_intrinsics()
    detection = load_detection()
    fusion_params = detection["fusion"]
    intr_fish = intrinsics["fisheye"]

    clips = list_clip_timestamps(captures_dir, args.clip_from, args.clip_to)
    if frames_filter is not None:
        # Only the clips that actually contain a requested frame_id. frame_id is
        # "<scene>/<triplet_ts>/ts=...", so the triplet_ts is the 2nd segment.
        want_clips = {fid.split("/")[1] for fid in frames_filter if "/" in fid}
        clips = [c for c in clips if c in want_clips]
        print(f"frames-file: {len(frames_filter)} frame_ids across "
              f"{len(clips)} clips", flush=True)
    if not clips:
        raise SystemExit(f"No fisheye clips in {captures_dir} for the given range.")
    print(f"clips in range ({len(clips)}): {clips[0]} .. {clips[-1]}", flush=True)

    done = set() if args.no_resume else _load_done_frame_ids(out_path)
    if done:
        print(f"resume: {len(done)} frame_ids already in {out_path.name}; skipping them",
              flush=True)

    backend = None
    if not args.dry_run:
        if args.backend == "vllm":
            backend = VLLMBackend(
                args.model or None, args.gpu_mem_util, args.max_model_len,
                args.tensor_parallel_size, args.max_tokens,
                min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            model_name = backend.model
        else:
            backend = TransformersBackend(args.model, args.max_tokens)
            model_name = args.model
    else:
        model_name = args.model

    out = open(out_path, "a")
    n_seen = n_written = n_calls = n_cache = n_failed = 0
    t_start = time.time()

    # Pending = frames awaiting a model call this batch.
    pending: list[tuple] = []  # (frame_id, scene, triplet_ts, frame_ts, img, jpeg, hash)

    def flush_pending() -> None:
        nonlocal n_written, n_calls, n_failed
        if not pending:
            return
        imgs = [p[4] for p in pending]
        if args.dry_run:
            results = [("{\"obstacles\": []}", None)] * len(pending)
        else:
            results = backend.generate_batch(imgs)
        for (frame_id, scene, triplet_ts, frame_ts, img, _jpeg, fhash), (resp, proc_wh) in zip(
                pending, results):
            if not args.dry_run:
                (cache_dir / f"{fhash}.json").write_text(json.dumps(
                    {"frame_id": frame_id, "model": model_name, "raw": resp,
                     "proc_wh": proc_wh}))
                n_calls += 1
            try:
                rec = response_to_record(frame_id, scene, triplet_ts, frame_ts,
                                         img.shape, resp, model_name,
                                         fusion_params, intr_fish, parse_wh=proc_wh)
            except Exception as e:  # noqa: BLE001 - never let one frame kill the run
                print(f"!! {frame_id}: parse failed: {e}", flush=True)
                n_failed += 1
                continue
            out.write(json.dumps(rec) + "\n")
            out.flush()
            done.add(frame_id)
            n_written += 1
            print(f"  [{frame_id}] n_bboxes={len(rec['fisheye_bboxes'])} "
                  f"bins={rec['obstacle_bins_fisheye']}", flush=True)
        pending.clear()

    n_bad_clips = 0
    try:
        for clip_ts in clips:
            print(f"\n=== clip {clip_ts} ===", flush=True)
            # A corrupt/truncated mp4 (e.g. an interrupted capture chunk:
            # "moov atom not found") makes iterate_triplet raise on open. One
            # bad clip must not abort the whole multi-clip run: log it, drain
            # any buffered frames, and move on to the next clip.
            try:
                for frame_id, scene, triplet_ts, frame_ts, img in iter_clip_frames(
                        captures_dir, clip_ts, args.every, detection, intrinsics):
                    if frames_filter is not None and frame_id not in frames_filter:
                        continue
                    if args.max_frames is not None and n_seen >= args.max_frames:
                        break
                    n_seen += 1
                    if frame_id in done:
                        continue

                    jpeg = _frame_to_jpeg_bytes(img)
                    fhash = _frame_hash(jpeg, model_name)
                    cache_path = cache_dir / f"{fhash}.json"

                    if cache_path.exists():
                        # Cached raw response: reparse without a model call.
                        cached = json.loads(cache_path.read_text())
                        resp = cached["raw"]
                        proc_wh = cached.get("proc_wh")
                        if proc_wh is not None:
                            proc_wh = tuple(proc_wh)
                        try:
                            rec = response_to_record(
                                frame_id, scene, triplet_ts, frame_ts, img.shape,
                                resp, model_name, fusion_params, intr_fish,
                                parse_wh=proc_wh)
                        except Exception as e:  # noqa: BLE001
                            print(f"!! {frame_id}: cached parse failed: {e}", flush=True)
                            n_failed += 1
                            continue
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
                        done.add(frame_id)
                        n_written += 1
                        n_cache += 1
                        continue

                    pending.append((frame_id, scene, triplet_ts, frame_ts, img, jpeg, fhash))
                    if len(pending) >= args.batch_size:
                        flush_pending()
            except Exception as e:  # noqa: BLE001 - skip unreadable clips, keep going
                flush_pending()
                print(f"!! clip {clip_ts}: SKIPPED ({type(e).__name__}: {e})", flush=True)
                n_bad_clips += 1
                continue
            if args.max_frames is not None and n_seen >= args.max_frames:
                break
        flush_pending()
    finally:
        flush_pending()  # drain anything left if the loop broke mid-batch
        out.close()

    dt = time.time() - t_start
    rate = n_written / dt if dt > 0 else 0.0
    print(f"\ndone: seen={n_seen} written={n_written} "
          f"(model_calls={n_calls} from_cache={n_cache} failed={n_failed} "
          f"bad_clips={n_bad_clips}) "
          f"in {dt:.0f}s ({rate:.2f} frames/s)", flush=True)
    print(f"output -> {out_path}", flush=True)
    if args.dry_run:
        print("dry-run: no model was loaded or called.", flush=True)


if __name__ == "__main__":
    main()
