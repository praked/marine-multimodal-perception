# InstTwo-GPU labelling pipeline

GPU-side labelling of the InstitutionOne day-1 fisheye captures on a InstTwo TSG GPU node
(gpu-node), mirroring the SmolVLA-Testing local → jump-host → GPU-node workflow.

> **Current best labeller is the GroundingDINO detector**, not the VLM; it
> localises far tighter. See **[Detector pipeline](#detector-pipeline)** below.
> The Qwen3-VL flow (rest of this README) gives correct classes but loose boxes;
> kept for class reasoning. Both write the same dashboard schema + `frame_id`s.

## Detector pipeline

GroundingDINO (`scripts/eval/detector_labeler.py`), tuned to **F1 0.702** on the
hand-labelled eval clip. Config: `configs/detector_labeler.yaml` (multi-pass:
objects@0.30 + structure/large@0.20, `se=1000`, `boat_gate=0.8`).

```bash
# iterate the detector config on the fixed smoke set → contact sheet
bash scripts/gpu_labeling/iterate_detector.sh my-tag
open results/qwen_audit/det_my-tag/contact_01.png

# parameter grid search (objects_bt × struct_bt × boat_gate); leave running
bash scripts/gpu_labeling/run_gridsearch.sh          # -> out/grid_fine on node
bash scripts/gpu_labeling/pull_gridsearch.sh         # -> results/grid_fine

# label the WHOLE mission with the best config (nohup, resumable)
STRIDE=2 bash scripts/gpu_labeling/run_detector_full.sh
```

### Build the fine-tuning set: dashboard audit mode

The full-mission detector labels are a *suggestion layer*. Turn them into a
clean training set by reviewing them in the dashboard's **audit mode**: every
frame is pre-filled with the detector's boxes, you correct what's wrong, and
press `a` to accept the frame into the training set.

```bash
python -m scripts.eval.dashboard \
    --pseudo labels/qwen/det_2026-06-17_institutionone_day1.jsonl \
    --exclude labels/eval_smoke36_mapping.json \
    --triplet data/captures/2026-06-17_institutionone_day1/2026-06-17_12-41-23
```

- Suggestions are editable but **not saved by browsing**, only `a` commits a
  frame, advances, and bumps the `n/500` counter (`--target` changes the goal).
- Accepted frames persist to **`labels/training_frames.jsonl`** (`--train-out`),
  reloaded on restart: **quit and resume across sessions**; the counter picks
  up where you left off. The file is git-tracked and kept separate from the
  eval annotations in `manual.jsonl`.
- `--exclude` blocks the 36 held-out eval-clip frames from the training set (no
  leakage). Full annotator guide: `docs/annotation_manual.pdf` §"Fast path".

### Evaluation / metrics loop (no per-config visual check)

```bash
# 1. build the dashboard-loadable eval clip from the smoke set (once)
python -m scripts.eval.make_eval_clip \
    --captures-dir data/captures/2026-06-17_institutionone_day1 \
    --frames-file labels/qwen/smoke_set.txt --out-mission eval_smoke36

# 2. (optional) pre-seed the dashboard from a detector run, then correct it
python -m scripts.eval.seed_labels_from_detector \
    --pred results/grid_fine/obt03_sbt02_gate08/labels.jsonl \
    --mapping labels/eval_smoke36_mapping.json
python -m scripts.eval.dashboard --triplet data/captures/eval_smoke36/2026-06-17_18-00-00

# 3. score / rank any detector run(s) vs your annotation
python -m scripts.eval.score_detections --truth labels/manual.jsonl \
    --grid-dir results/grid_fine --mapping labels/eval_smoke36_mapping.json
```

**Findings + roadmap (fine-tuning to 0.8+):** see
`docs/fisheye_context.pdf` §"Detector pipeline, evaluation, and fine-tuning
roadmap".

---

## Qwen3-VL pipeline (class reasoning)

Autonomous Qwen3-VL bounding-box labelling on a InstTwo TSG GPU node, mirroring the
SmolVLA-Testing local → jump-host → GPU-node workflow.

The model labels each fisheye frame the way the annotation manual
(`docs/annotation_manual.tex`) says we would by hand, and writes records in the
**dashboard schema** with **matching `frame_id`s**, so the output drops straight
into the existing audit (`label_tool.py --audit`) and `metrics.py` paths.

```
local Mac ──ssh──► gpu-node.cluster.example.org ──ssh──► <gpu-node>.cluster.example.org
 (this repo)        (jump host)                    (vLLM + Qwen3-VL on scratch)
```

## One-time prerequisites

1. **InstTwo VPN** active (`vpn.example.org`). gpu-node + GPU nodes are unreachable
   off-campus without it.
2. **SSH key** at `~/.ssh/cluster_key`:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/cluster_key -N ""
   ```
   Install it on the GPU node (from an active booking):
   ```bash
   cat ~/.ssh/cluster_key.pub | ssh -J user@gpu-node.cluster.example.org \
       user@gpu-node.cluster.example.org \
       "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys \
        && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
   ```
3. **`~/.ssh/config`** stanzas (avoids "Too many authentication failures"):
   ```
   Host gpu-node.cluster.example.org
       User user
       IdentityFile ~/.ssh/cluster_key
       IdentitiesOnly yes
   Host gpu-node.cluster.example.org
       User user
       IdentityFile ~/.ssh/cluster_key
       IdentitiesOnly yes
   ```

## Configure

Edit `scripts/gpu_labeling/00_run_params.sh` (or a gitignored
`00_run_params.local.sh` override) for **each booking**:

- `GPU_NODE`: the node you booked (changes each time).
- `WORKFLOW_USER`: your InstTwo username.
- `MODEL`; see GPU memory note below.
- `CLIP_FROM` / `CLIP_TO`: clip range (defaults cover 12:41:23 → 13:41:26).
- `EVERY`: frame stride (`1` = every frame ≈ 11k frames; `30` matches the
  manual eval stride ≈ 380 frames for a fast first pass).
- `MAX_FRAMES`: set for a smoke test, leave empty for the full range.

## Run

```bash
cd scripts/gpu_labeling
bash 01_sync_to_gpu.sh        # code -> home, clip range -> scratch
bash 02_preflight_gpu.sh      # SSH + layout + GPU sanity (prints nvidia-smi)
bash 03_setup_gpu.sh          # venv + deps + model download on the node (slow, once per booking)
bash 04_run_labeling.sh       # nohup launch; resumable
# ... monitor (commands printed by 04) ...
bash 05_extract_from_scratch.sh   # pull labels back BEFORE the booking ends
```

Monitor:
```bash
ssh -i ~/.ssh/cluster_key -J user@gpu-node.cluster.example.org user@gpu-node.cluster.example.org \
    "tail -f /scratch/user/asvproject/out/run.log"
```

## GPU memory → model choice

`02_preflight_gpu.sh` prints the node's GPU. Pick `MODEL` accordingly:

| VRAM  | Suggested `MODEL`                          | Notes                                  |
|-------|--------------------------------------------|----------------------------------------|
| 24 GB | `Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ`       | MoE, 3B active, AWQ                     |
| 16 GB | `Qwen/Qwen3-VL-8B-Instruct-FP8`            | **default (gpu-node)**; ~9 GB weights |
| ≤12 GB| `Qwen/Qwen3-VL-4B-Instruct`                | last resort, lower quality             |

`Qwen3-VL-8B-Instruct` in bf16 (~16 GB weights) does **not** fit 16 GB once the
KV cache is added; use the FP8 build (the 4070 Ti SUPER is Ada / FP8-capable).

If a model OOMs at load, lower `GPU_MEM_UTIL` / `MAX_MODEL_LEN`, or drop a tier.

## Output & resume

- Provisional labels: `out/qwen_<mission>.jsonl` on scratch, pulled to
  `labels/qwen/` locally by `05_extract`. Schema: dashboard-compatible, with
  `source: "qwen3-vl"`, `audited: false`.
- Raw responses cached per-frame under `out/cache/`. Re-running `04` skips
  frames already in the JSONL and reparses cache hits without GPU calls, so a
  killed or re-queued run resumes cleanly.

## After extraction

Audit a sample and watch the per-scene acceptance ratio against the 0.8 floor
(annotation manual §"Acceptance-ratio sanity"):
```bash
python -m scripts.eval.label_tool --audit labels/qwen/qwen_2026-06-17_institutionone_day1.jsonl
```
Accepted rows land in `labels/master.jsonl` with `audited: true`.

## Prompt iteration loop

The master prompt + class taxonomy live in **`configs/qwen_label_prompt.yaml`**
(system/user text, allowed `classes`, a `synonyms` map, and the `coords`
convention). Edit that file, then smoke-test it on a fixed comparison set:

```bash
# 1. edit configs/qwen_label_prompt.yaml
# 2. run the prompt on the fixed 36-frame smoke set (labels/qwen/smoke_set.txt)
bash scripts/gpu_labeling/iterate_prompt.sh my-tag
# 3. eyeball the result
open results/qwen_audit/prompt_my-tag/contact_01.png
```

Each run gets its own output, cache, and a copy of the prompt under
`results/qwen_audit/prompt_<tag>/`, so prompts never mask each other and you can
compare sheets side by side. Notes:

- **`coords: absolute`** asks the model for pixel coordinates in the 864×648
  image (Qwen3-VL's trained grounding convention). This localises much better
  than normalised floats; it fixed the "right-size box, wrong place" offset.
- **`structure`** is a class for mooring posts / pilings / piers / swim
  platforms, so they stop being mislabelled as buoy/boat. ⚠️ The dashboard,
  `label_tool`, and `metrics` still use the 5-class manual taxonomy; treat
  `structure` rows as a review bucket until that taxonomy is extended (or map
  `structure → other` in the synonyms if you want manual-compatible output).
- The smoke set is curated to span the hard cases (posts/pier, swim platform,
  shoreline, marina) and easy ones (clean boats, empty water). Regenerate or
  hand-edit `labels/qwen/smoke_set.txt` (one frame_id per line) to change it.

To render a contact sheet from any labels JSONL directly:

```bash
python -m scripts.eval.render_label_overlays \
    --labels labels/qwen/qwen_2026-06-17_institutionone_day1.jsonl \
    --captures-dir data/captures/2026-06-17_institutionone_day1 \
    --out-dir results/qwen_audit/full --detections-only
```

## Local dry-run (no GPU)

Validate clip resolution, frame_id synthesis, and the record schema without a
model:
```bash
python -m scripts.eval.qwen_batch_labeler \
    --captures-dir data/captures/2026-06-17_institutionone_day1 \
    --from 2026-06-17_12-41-23 --to 2026-06-17_13-41-26 \
    --every 30 --out /tmp/dry.jsonl --dry-run
```
