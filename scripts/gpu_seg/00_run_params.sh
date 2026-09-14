#!/usr/bin/env bash
# Shared parameters for the InstTwo-GPU eWaSR water-segmentation workflow.
#
# Mirrors scripts/gpu_labeling/00_run_params.sh: a single sourced config used by
# 01_sync / 02_preflight / 03_setup / 04_train / 05_predict / 06_extract.
# Personal/per-booking overrides go in 00_run_params.local.sh (gitignored),
# sourced last.
#
# Goal: train eWaSR (tersekmatija/eWaSR, ResNet-18, 3-class water/sky/obstacle)
# on LaRS, then predict masks for our undistorted fisheye clips so the local
# range pipeline can use the true waterline-contact point. See
# docs/reference/segmentation.md (or the internal docs/lars_segmentation_plan.md).

PARAMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LOCAL_PROJECT_ROOT="$(cd "${PARAMS_DIR}/../.." && pwd)"

# -------- Workflow identity --------
WORKFLOW_USER="user"

# -------- SSH topology --------
# GPU_NODE changes per booking; JUMP_HOST is gpu-node at InstTwo. On a cluster with
# direct ssh (e.g. InstitutionOne) set JUMP_HOST="" in 00_run_params.local.sh — every
# kit builds its -J option only when JUMP_HOST is non-empty. REMOTE_SCRATCH_BASE
# (below) is the node-side working root; override it to the cluster's scratch.
REMOTE_USER="${WORKFLOW_USER}"
GPU_NODE="gpu-node.cluster.example.org"
JUMP_HOST="gpu-node.cluster.example.org"
SSH_KEY_FILE="${HOME}/.ssh/cluster_key"
# The GPU nodes' login shell is tcsh; all remote command execution pipes a bash
# script to `bash -s` rather than relying on `VAR=val cmd` prefixes.

# -------- Remote layout --------
# Code lives in persistent home (small AFS quota: scripts + configs, no data/.git).
REMOTE_PROJECT_DIRNAME="asvproject_project"
REMOTE_REPO_DIRNAME="ASVProject-ObstacleDetection"
# Scratch is local NVMe, large, but WIPED when the booking ends: extract first.
REMOTE_SCRATCH_BASE="/scratch0/${REMOTE_USER}"
SEG_SCRATCH="${REMOTE_SCRATCH_BASE}/asvproject_seg"

# -------- Model / training --------
EWASR_REPO="https://github.com/tersekmatija/eWaSR.git"
# Non-IMU variant: we do not ship eWaSR "IMU" horizon-prior masks (a future
# improvement now that the BNO085 is coming online). 3 classes: obstacle/water/sky.
MODEL="ewasr_resnet18"
NUM_CLASSES=3
EPOCHS=50
BATCH_SIZE=8
PATIENCE=15
# Pretrained MaSTr1325 weights (release 0.1.0): used for an IMMEDIATE inference
# baseline while LaRS training runs, and as a sanity check of the predict path.
PRETRAINED_URL="https://github.com/tersekmatija/eWaSR/releases/download/0.1.0/ewasr_resnet18.pth"

# -------- Dataset staging (built locally, rsynced to scratch) --------
# Raw LaRS lives on the SSD; prepare.py resizes it to a tiny MaSTr-format tree.
LARS_LOCAL_ROOT="/Volumes/ROS2_SSD/LaRS_v1.0.0"
TRAIN_SIZE="512x384"           # WxH; 4:3 like our 864x648 undistorted frames
LOCAL_MASTR_DIR="${DEFAULT_LOCAL_PROJECT_ROOT}/data/lars_mastr"

# -------- Clips to predict masks for (undistorted fisheye frames) --------
# We segment/test on the real InstitutionOne day-1 mission (our domain), not the old
# Boats/Ducks/OpenWater/Rain test footage. If MISSION is set, 01_sync expands it
# to every triplet under data/captures/<MISSION>/; otherwise PREDICT_CLIPS (a
# space-separated list of <path-under-data> ids) is used verbatim.
MISSION="2026-06-17_institutionone_day1"
PREDICT_CLIPS=""              # leave empty to use MISSION; else e.g. "Boats/2025-06-23_16-21-07"
# Inclusive triplet-timestamp range within MISSION (lexicographic == chrono).
# Empty = all clips in the mission.
CLIP_FROM="2026-06-17_12-41-23"
CLIP_TO="2026-06-17_13-41-26"
PREDICT_EVERY=1              # 1 = every frame (full annotation)
LOCAL_PRED_FRAMES_DIR="${DEFAULT_LOCAL_PROJECT_ROOT}/data/seg_frames"
LOCAL_SEG_OUT_DIR="${DEFAULT_LOCAL_PROJECT_ROOT}/data/seg"
LOCAL_MODELS_DIR="${DEFAULT_LOCAL_PROJECT_ROOT}/models"

# -------- Derived remote paths --------
SEG_VENV="${SEG_SCRATCH}/seg_venv"
SEG_CACHE="${SEG_SCRATCH}/.cache"
SEG_EWASR="${SEG_SCRATCH}/eWaSR"
SEG_MASTR="${SEG_SCRATCH}/lars_mastr"
SEG_RUNS="${SEG_SCRATCH}/runs"
SEG_PRED_IN="${SEG_SCRATCH}/pred_frames"
SEG_PRED_OUT="${SEG_SCRATCH}/pred_masks"
SEG_WEIGHTS_DIR="${SEG_SCRATCH}/weights"
SEG_SHIM="${SEG_SCRATCH}/activate_seg.sh"

# -------- Local paths --------
LOCAL_PROJECT_ROOT="${DEFAULT_LOCAL_PROJECT_ROOT}"

# -------- Local override (gitignored) --------
if [[ -f "${PARAMS_DIR}/00_run_params.local.sh" ]]; then
	# shellcheck source=/dev/null
	source "${PARAMS_DIR}/00_run_params.local.sh"
fi
