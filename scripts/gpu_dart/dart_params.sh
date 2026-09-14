#!/usr/bin/env bash
# Single knob file for the DART (SAM3 multi-class) labeller on a GPU node.
# Sourced by run_dart.sh (local) and exported into the remote scripts.
# Per-booking overrides (node name etc.) go in dart_params.local.sh (gitignored),
# and the SSH topology is inherited from scripts/gpu_seg/00_run_params.sh
# (+ its .local.sh, where GPU_NODE changes per booking).
#
# Everything on the node lives on /scratch0 (wiped at booking end); AFS home
# holds code only — every cache is exported to scratch (home-quota rule).

PARAMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${PARAMS_DIR}/../gpu_seg/00_run_params.sh"
# When invoked from gpu_express, the resolved cluster profile is
# authoritative (EXPRESS_* exports; note JUMP_HOST may legitimately be
# EMPTY for direct-ssh clusters, hence the dash — not :- — expansions).
REMOTE_USER="${EXPRESS_REMOTE_USER:-${REMOTE_USER}}"
GPU_NODE="${EXPRESS_GPU_NODE:-${GPU_NODE}}"
JUMP_HOST="${EXPRESS_JUMP_HOST-${JUMP_HOST}}"
SSH_KEY_FILE="${EXPRESS_SSH_KEY_FILE:-${SSH_KEY_FILE}}"
REMOTE_SCRATCH_BASE="${EXPRESS_SCRATCH:-${REMOTE_SCRATCH_BASE}}"
TORCH_INDEX="${EXPRESS_TORCH_INDEX:-${TORCH_INDEX:-}}"

# -------- Remote layout (all under /scratch0/$REMOTE_USER = $S on the node) --------
DART_S="${REMOTE_SCRATCH_BASE}"                 # gpu_seg params; override per cluster
DART_CODE='$HOME/asvproject_project/ASVProject-ObstacleDetection'   # AFS home, expanded remotely
DART_REPO_URL="https://github.com/mkturkcan/DART.git"
DART_COMMIT="${DART_COMMIT:-16fada39054ac5058f6e7c1e8748cb9cc288f90c}"   # 2026-08-30 main, the 2026-08-31 run; empty = tip of main
DART_PYTHON_VERSION="3.12"              # DART needs >=3.11; nodes ship 3.9 -> uv-managed interpreter on scratch
DART_TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
DART_TORCH_SPEC="torch torchvision"     # 2026-08-31 run resolved torch 2.13.0+cu126 / torchvision 0.28.0 / numpy 1.26.4; pin here if a resolve breaks

# -------- Weights --------
# facebook/sam3 is GATED (SAM License). Two routes, chosen by run_dart.sh --prepare:
#   (a) laptop logged into HF (`hf auth whoami` works): download here, rsync up — no token on the node;
#   (b) otherwise it prints the one-line `hf download ... --token` for you to run yourself.
SAM3_REPO="facebook/sam3"
SAM3_FILE="sam3.pt"
SAM3_MD5="${SAM3_MD5:-2615191b18293b447020cf26f4800e51}"   # sam3.pt as downloaded 2026-08-31 (3,450,062,241 bytes)

# -------- Data to label --------
# Missions whose fisheye videos (+ frames CSVs, recovered copies) are staged
# from the SSD, and which of their clips get exported/labelled:
#   DART_MISSIONS      -> every fisheye chunk of these missions
#   DART_EXTRA_TRIPLETS -> explicit data/captures/<mission>[/<sub>]/<ts> paths (nested missions!)
#   DART_PLAN_CSV      -> if set, every clip named in this audit plan is added too
DART_MISSIONS="2026-08-26_afloat"
DART_EXTRA_TRIPLETS=""
DART_PLAN_CSV="${LOCAL_PROJECT_ROOT}/results/audit_plan.csv"
DART_EVERY=1                             # frame stride for export + labelling

# -------- Labeller --------
DART_CONFIG="configs/dart_labeler.yaml"  # prompt set + thresholds (relative to repo)
DART_COMPILE="max-autotune-no-cudagraphs"  # validated 99.8 % box-identical, +25 %; plain max-autotune CRASHES the encoder
DART_SMOKE_CLIP=""                        # substring of a clip to smoke on first (empty = first exported clip)
DART_OUT_NAME="labels_dart"               # $S/corpus_out/<name>/det_<clip>.jsonl

# -------- Local placement on --pull --------
DART_ARCHIVE_ROOT="/Volumes/ROS2_SSD/asvproject"          # archive: <root>/<node>_<date>/labels_dart/
DART_LOCAL_LABELS="${LOCAL_PROJECT_ROOT}/labels/dart"    # NEVER labels/qwen (the audit's set)

# -------- Local override (gitignored) --------
if [[ -f "${PARAMS_DIR}/dart_params.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PARAMS_DIR}/dart_params.local.sh"
fi
