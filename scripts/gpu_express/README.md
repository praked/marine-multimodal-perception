# gpu_express — from nothing on a GPU node to everything running

One parameter file, numbered steps, each idempotent and resumable. Drives the
proven pieces (Track B corpus queue, thermal ladder, fusion scorer, sectors,
native GroundingDINO, the DART/SAM3 labeller kit, the YOLO fine-tune) instead
of re-implementing them — the queue scripts that ran the 2026-08-27/31
bookings are the ones this launches.

```
scripts/gpu_express/
├── 00_params.sh          THE knob file (node layout, SSD sources, data scope, which queues/stages, pull policy)
├── 00_params.local.sh    per-booking overrides (gitignored)           GPU_NODE lives in gpu_seg/00_run_params.local.sh
├── 01_prepare.sh         code sync · corpus venv · published weights · rclone · DART env+checkpoint · YOLO env
├── 02_preflight.sh       read-only: GPU, scratch, home quota, envs, staged inputs, running queues
├── 03_stage.sh           SSD -> node: raw triplets (scoped), LaRS-as-MaSTr, thermal inits, labels/, audit plan, DINO scope
├── 04_launch.sh          nohup express_queue.sh on the node: Track B (stages) -> DART -> YOLO
├── 05_status.sh          one screen for everything (--watch = refresh every 60 s)
├── 06_pull.sh            scratch -> ${SSD}/<node>_<date>/ (+ optional placement); run BEFORE the booking ends
├── run_all.sh            01 + 03 + 04
├── express_queue.sh      node: sequences the sub-queues so they never share the GPU
└── yolo_queue.sh         node: on-domain YOLO fine-tune (RUN_YOLO=1)
```

## Cluster profiles

Per-cluster topology (user / node / jump host / key / scratch / torch index)
lives in `profiles/<name>.sh`, so moving between shared GPUs never means
editing params — select once and every numbered step resolves it:

```bash
./use_profile.sh cluster-a     # persisted in profiles/ACTIVE (gitignored)
GPU_PROFILE=institutionone ./run_all.sh   # or per-run, overrides ACTIVE
```

- `profiles/cluster-a.sh` — the InstTwo setup that ran the gpu-node/gpu-node
  campaigns (per-booking node name is the one thing that still changes).
- `profiles/institutionone.sh.template` — copy to `institutionone.sh`, fill the TODOs
  on first contact (username, host, scratch path, torch index per driver),
  run `./02_preflight.sh` to verify, then commit `institutionone.sh` so the
  whole team shares it.
- No profile selected → the legacy `gpu_seg/00_run_params.sh` values
  (backwards compatible).
- Run-scope knobs (missions, stages, seeds) are NOT topology and stay in
  `00_params.sh` / `00_params.local.sh`.

## The whole thing

```bash
echo 'GPU_NODE="<node>.cluster.example.org"' > scripts/gpu_seg/00_run_params.local.sh   # new booking
vim scripts/gpu_express/00_params.local.sh    # optional: MISSIONS, RUN_*, TRACKB_STAGES, DINO_ONLY ...
bash scripts/gpu_express/run_all.sh           # = 01_prepare + 03_stage + 04_launch   (SSD mounted, HF logged in for SAM3)
bash scripts/gpu_express/05_status.sh --watch
bash scripts/gpu_express/06_pull.sh           # before the booking ends
```

Typical timings (gpu-node, RTX 4070 Ti SUPER): prepare 5–15 min · stage 10–60 min
(lab uplink ~2 MB/s; the whole 2026 corpus is ~4 GB of videos) · Track B stages
4+5 ≈ 10 min · thermal ladder ≈ 5 h (3 seeds) · fusion ≈ 1.3 h · sectors ≈ 1.5 h ·
native DINO ≈ 3 h per outing at stride 2 · DART ≈ 4.5 h for 54k frames.

## Knobs you actually change

| Knob | Meaning |
|---|---|
| `MISSIONS` | capture missions staged in full (nested ones are walked automatically) |
| `AUDIT_PLAN` | clips named in an audit plan are staged too (for labeller comparisons) |
| `RUN_TRACKB` / `TRACKB_STAGES` | which Track-B stages: 4 export · 5 native predict · 7 thermal ladder · 8 fusion · 9 sectors · 6 native DINO |
| `DINO_ONLY` | scope of the native relabel (a whole-corpus relabel is ~7 h and forks the label set) |
| `THERMAL_SEEDS` / `THERMAL_EPOCHS` / `SCORER_TAG` | ladder + scorer run identity |
| `RUN_DART` (+ `gpu_dart/dart_params.sh`) | the SAM3 labeller: missions/plan clips, prompt config, compile mode |
| `RUN_YOLO` + `YOLO_*` | on-domain fine-tune from `labels/training_frames.jsonl` (audited) via `export_yolo` |
| `PULL_PLACE_*` | what 06_pull copies into the repo tree (labels/dart yes; native masks opt-in; **never labels/qwen**) |

## Rules the kit enforces (so you don't have to remember them)

- AFS home = code only; every venv/cache on scratch (`setup_env.sh`, `remote_setup.sh`).
- Remote shell is tcsh: everything goes through `bash -s` + `env VAR=x`.
- macOS rsync 2.6.9: `--rsh=` and `--files-from`; nested destination parents pre-created.
- Gated `facebook/sam3`: downloaded on the laptop when logged in, rsynced up; never a token on the node.
- Sub-queues run strictly in sequence on the one GPU; each keeps its own `STATUS*` file; the
  Track-B stage gate is `TRACKB_STAGES` (defaults = the historical order).
- `06_pull.sh` re-checks the SSD after every directory (the cable drops) and archives the
  as-run node scripts alongside the outputs.

## History it replaces

`gpu_corpus/stage_raw.sh` + `run_night.sh` (Track A/B), `gpu_seg/0x_*` (eWaSR/LRASPP),
`gpu_seg/thermal_restage.sh`, `gpu_corpus/rescore_restage.sh`, the session-scratchpad
one-offs of 2026-08-28/31 (`pull_trackb.sh`, `fill_0819.sh`, `thermal_0819.sh`,
`stage_dart.sh`). Those still work and remain the reference for what each stage does.

## Portability: a non-InstTwo cluster (e.g. a InstitutionOne GPU)

Nothing in the numbered steps is InstTwo-specific; the cluster shape lives in
`scripts/gpu_seg/00_run_params.local.sh` (topology) and
`scripts/gpu_express/00_params.local.sh` (layout), e.g.:

```bash
# scripts/gpu_seg/00_run_params.local.sh
GPU_NODE="gpu01.example.institution-one.example"
JUMP_HOST=""                          # direct ssh: no -J is emitted anywhere
REMOTE_USER="authortwo"
SSH_KEY_FILE="$HOME/.ssh/id_ed25519"
REMOTE_SCRATCH_BASE="/data/authortwo/asvproject"   # the node's big/fast working root (was /scratch0/$USER)
# scripts/gpu_express/00_params.local.sh
TORCH_INDEX="https://download.pytorch.org/whl/cu121"   # match `nvidia-smi`'s driver CUDA version
REMOTE_CODE_REL="work/ASVProject-ObstacleDetection"     # code dir, relative to the node's $HOME
```

What the node must have: Linux + bash (the login shell may be anything — every
remote step runs `bash -s`), `rsync`, `git`, `curl`, a system `python3` ≥ 3.9
(the corpus venv) — DART's Python 3.12 is fetched by `uv` into `$S`, so no
module system is needed — an NVIDIA driver matching `TORCH_INDEX`, and
outbound HTTPS (GitHub, PyPI, HuggingFace). The home-quota discipline (caches
on `$S`) is harmless where home is large. Not covered: SLURM/PBS schedulers —
the queues assume they own the GPU on an interactive/allocated node; wrap
`04_launch`'s nohup line in an `srun`/`sbatch` if the cluster requires it.
