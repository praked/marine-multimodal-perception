# gpu_dart — DART (SAM3 multi-class) labeller on a GPU node, express edition

Everything needed to run the open-vocabulary teacher candidate on a fresh
booking, driven from the laptop with one script. Plan and rationale:
`docs/plans/teacher_comparison.md`; driver: `scripts/gpu_corpus/label_bundle_dart.py`;
prompts/thresholds: `configs/dart_labeler.yaml`.

```
scripts/gpu_dart/
├── dart_params.sh        ALL knobs (node layout, pins, data selection, compile, placement)
├── dart_params.local.sh  per-booking overrides (gitignored, optional)
├── run_dart.sh           LOCAL entry point: --prepare --stage --launch --status --pull --all
├── remote_setup.sh       node: uv Python 3.12 + venv (torch cu126, DART -e .) + weights check
├── remote_export.sh      node: undistorted 864x648 frame export for the staged triplets
└── remote_queue.sh       node (nohup): smoke -> label every exported clip, resumable
```

## New node, from zero (≈ 15 min + transfer)

```bash
# 0. node name for this booking
echo 'GPU_NODE="<node>.cluster.example.org"' > scripts/gpu_seg/00_run_params.local.sh
# 1. env + weights (gated facebook/sam3: needs `hf auth login` here, or the printed node one-liner)
bash scripts/gpu_dart/run_dart.sh --prepare
# 2. videos -> node, frames exported (SSD mounted)
bash scripts/gpu_dart/run_dart.sh --stage
# 3. go; then poll
bash scripts/gpu_dart/run_dart.sh --launch
bash scripts/gpu_dart/run_dart.sh --status
# 4. when STATUS_dart says "DART DONE": archive + labels/dart (SSD mounted)
bash scripts/gpu_dart/run_dart.sh --pull
```

`--all` = prepare + stage + launch. Every step is idempotent; the queue resumes
at the frame level, so a crash or a `--launch` after a node restart just continues.

## What is pinned (dart_params.sh)

- DART commit `16fada39` (2026-08-30), Python 3.12 via uv on scratch (nodes ship
  3.9), torch cu126 (resolved 2.13.0 on 2026-08-31), `numpy<2`.
- `sam3.pt` md5 `2615191b…` (3.45 GB). The repo is **gated** (Meta SAM License):
  `--prepare` downloads on the laptop when `hf auth whoami` works and rsyncs it
  up (no token ever reaches the node); otherwise it prints the node one-liner
  (`--token` on the command line, `HF_HOME` on scratch → dies with the booking).
- `--compile max-autotune-no-cudagraphs`: 99.8 % box-identical to eager and
  ~25 % faster (3.36 vs 2.6 rec/s on a 4070 Ti SUPER). Plain `max-autotune`
  **crashes** DART's encoder (CUDA-graph output reuse). `default` untested.

## Data selection

`DART_MISSIONS` (every fisheye chunk), `DART_EXTRA_TRIPLETS` (explicit
`data/captures/<mission>[/<sub>]/<ts>` — the way to reach nested missions),
`DART_PLAN_CSV` (every clip an audit plan names; nested clip names are resolved
back to their on-disk path). Only fisheye mp4 + `frames_*.csv` + recovered
copies are staged (~1.7 GB for 08-26 + day-1 + 07-08).

## Gotchas this kit already encodes

- macOS rsync 2.6.9: `--rsh=` not `-e`; `--files-from` not `--include` chains;
  it will not create nested destination parents (pre-created).
- Remote login shell is tcsh: everything remote goes through `bash -s` with
  `env VAR=x`; never `2>/dev/null` in a one-liner.
- AFS home quota: caches on scratch, `pip --no-cache-dir`; a dangling
  `~/.local/bin/python3.12` from a previous booking made pip fall through to
  the home user-site once (2026-08-31) — `remote_setup.sh` resolves the
  interpreter through uv every time.
- The public `mehmetkeremturkcan/DART` "pruned_NNblocks.pt" files are
  trunk-only block-skip fine-tunes and the EfficientSAM3 files are backbones
  only: none of them run without the gated base checkpoint.
- Output goes to `labels/dart/` locally — **never `labels/qwen`** (the audit's
  label set). Union for teacher-blind audits: `scripts.eval.merge_teacher_suggestions`.

## Other clusters

Topology and layout come from `scripts/gpu_seg/00_run_params.local.sh`
(`GPU_NODE`, `JUMP_HOST=""` for direct ssh, `REMOTE_USER`, `SSH_KEY_FILE`,
`REMOTE_SCRATCH_BASE`) and `TORCH_INDEX` / `REMOTE_CODE_REL` (see
`scripts/gpu_express/README.md` § Portability); nothing here hard-codes InstTwo.
