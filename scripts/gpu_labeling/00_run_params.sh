#!/usr/bin/env bash
# Shared parameters for the InstTwo-GPU Qwen labelling workflow.
#
# Mirrors the SmolVLA-Testing scripts/00_run_params.sh pattern: a single
# sourced config consumed by 01_sync / 02_preflight / 03_setup / 04_run /
# 05_extract. Personal overrides go in 00_run_params.local.sh (gitignored),
# which is sourced last and can override anything below.

PARAMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LOCAL_PROJECT_ROOT="$(cd "${PARAMS_DIR}/../.." && pwd)"

# -------- Workflow identity --------
# InstTwo username (set explicitly; do not rely on local $USER).
WORKFLOW_USER="user"

# -------- SSH topology --------
# GPU_NODE changes per booking; JUMP_HOST is always gpu-node.
REMOTE_USER="${WORKFLOW_USER}"
GPU_NODE="gpu-node.cluster.example.org"
JUMP_HOST="gpu-node.cluster.example.org"
# Passwordless key. Generate once:  ssh-keygen -t ed25519 -f ~/.ssh/cluster_key -N ""
# Install on a new node (from an active session):
#   cat ~/.ssh/cluster_key.pub | ssh -J ${REMOTE_USER}@${JUMP_HOST} \
#       ${REMOTE_USER}@${GPU_NODE} \
#       "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys \
#        && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
SSH_KEY_FILE="${HOME}/.ssh/cluster_key"

# -------- Remote layout --------
# Code lives in persistent home (small: scripts + configs, no data/.git).
REMOTE_PROJECT_DIRNAME="asvproject_project"
REMOTE_REPO_DIRNAME="ASVProject-ObstacleDetection"
# Scratch is local NVMe, large, but WIPED when the booking ends: extract first.
REMOTE_SCRATCH_BASE="/scratch0/${REMOTE_USER}"
LABEL_SCRATCH="${REMOTE_SCRATCH_BASE}/asvproject"

# -------- Dataset / clip range --------
MISSION="2026-06-17_institutionone_day1"
# Inclusive clip-timestamp range (lexicographic == chronological here).
CLIP_FROM="2026-06-17_12-41-23"
CLIP_TO="2026-06-17_13-41-26"

# -------- Model / inference --------
# gpu-node is a 16 GB RTX 4070 Ti SUPER (Ada, FP8-capable). FP8 8B weights are
# ~9 GB and leave room for the KV cache: best quality that fits. On a 24 GB node
# you could step up to Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ; on <12 GB drop to
# Qwen/Qwen3-VL-4B-Instruct. 02_preflight prints nvidia-smi so you can confirm.
MODEL="Qwen/Qwen3-VL-8B-Instruct-FP8"
BACKEND="vllm"            # vllm | transformers
EVERY=10                  # frame stride within each clip (10 = ~1 label / 3.3 s)
BATCH_SIZE=16             # frames per generate() call / write checkpoint
MAX_TOKENS=1024
GPU_MEM_UTIL=0.92
MAX_MODEL_LEN=8192       # one image + prompt + output fits well under this
# Vision resolution (pixels). Upscaling the 864x648 frame gives the model a
# finer patch grid, which fixed the small-object (boat/person) localisation
# drift: boxes that landed beside the object at native res now sit on it.
# ~30% slower per frame; fits comfortably under MAX_MODEL_LEN on 16 GB.
# Empty = library default (native res, drifts on small objects).
MIN_PIXELS="1200000"
MAX_PIXELS="2000000"
# Linear brightness gain applied to every frame before detection (1.0 = off).
# ~1.4 brightens the dark/green water scenes. Used by the detector labeller +
# the contact-sheet renderer so what you eyeball matches what the model saw.
EXPOSURE="1.0"
TENSOR_PARALLEL_SIZE=1
# Smoke-test cap; leave empty for the full range.
MAX_FRAMES=""

# -------- Derived remote paths (used by 03/04/05) --------
REMOTE_DATA_DIR="${LABEL_SCRATCH}/data/captures/${MISSION}"
REMOTE_OUT_DIR="${LABEL_SCRATCH}/out"
REMOTE_OUT_JSONL="${REMOTE_OUT_DIR}/qwen_${MISSION}.jsonl"
REMOTE_CACHE_DIR="${REMOTE_OUT_DIR}/cache"
REMOTE_RUN_LOG="${REMOTE_OUT_DIR}/run.log"
ACTIVATE_SHIM="${LABEL_SCRATCH}/activate_labeling.sh"

# -------- Local paths --------
LOCAL_PROJECT_ROOT="${DEFAULT_LOCAL_PROJECT_ROOT}"
LOCAL_DATA_DIR="${LOCAL_PROJECT_ROOT}/data/captures/${MISSION}"
# Where 05_extract drops the labels pulled back from scratch.
LOCAL_EXTRACT_DIR="${LOCAL_PROJECT_ROOT}/labels/qwen"

# -------- Local override (gitignored) --------
if [[ -f "${PARAMS_DIR}/00_run_params.local.sh" ]]; then
	# shellcheck source=/dev/null
	source "${PARAMS_DIR}/00_run_params.local.sh"
fi
