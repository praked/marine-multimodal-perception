#!/usr/bin/env bash
# =============================================================================
# 00_params.sh — THE parameter file for "nothing on the GPU node -> everything
# running". Every numbered step sources this and nothing else. Per-booking
# overrides (node name, scope, which stages) go in 00_params.local.sh
# (gitignored, sourced last). SSH topology + key come from
# scripts/gpu_seg/00_run_params.sh (+ its .local.sh where GPU_NODE lives).
#
# Sequence:  01_prepare -> 02_preflight -> 03_stage -> 04_launch -> 05_status -> 06_pull
#            (run_all.sh = 01 + 03 + 04)
# =============================================================================
PARAMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# gpu_seg/00_run_params.sh REASSIGNS PARAMS_DIR to its own directory when
# sourced below — keep our own path in a uniquely named variable.
EXPRESS_DIR="${PARAMS_DIR}"
# shellcheck source=/dev/null
source "${PARAMS_DIR}/../gpu_seg/00_run_params.sh"      # REMOTE_USER JUMP_HOST GPU_NODE SSH_KEY_FILE LOCAL_PROJECT_ROOT
LOCAL="${LOCAL_PROJECT_ROOT}"

# -------- Cluster profile (profiles/<name>.sh) --------
# Saved per-cluster topology (user/node/jump/key/scratch/torch index) so
# switching GPUs never means editing params: select once with
# ./use_profile.sh <name> (writes profiles/ACTIVE, gitignored) or per-run
# with GPU_PROFILE=<name>. No profile selected -> legacy gpu_seg values.
GPU_PROFILE="${GPU_PROFILE:-$(cat "${EXPRESS_DIR}/profiles/ACTIVE" 2>/dev/null || true)}"
if [[ -n "${GPU_PROFILE}" ]]; then
  if [[ ! -f "${EXPRESS_DIR}/profiles/${GPU_PROFILE}.sh" ]]; then
    echo "unknown GPU profile '${GPU_PROFILE}' — have: $(ls "${EXPRESS_DIR}/profiles" 2>/dev/null | sed -n 's/\.sh$//p' | grep -v '\.template$' | tr '\n' ' ')" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${EXPRESS_DIR}/profiles/${GPU_PROFILE}.sh"
fi
# Child kits (gpu_dart's run_dart.sh) re-source the legacy gpu_seg topology;
# export the express-resolved values so they follow the active profile.
export EXPRESS_REMOTE_USER="${REMOTE_USER}" EXPRESS_GPU_NODE="${GPU_NODE}" \
       EXPRESS_JUMP_HOST="${JUMP_HOST}" EXPRESS_SSH_KEY_FILE="${SSH_KEY_FILE}" \
       EXPRESS_SCRATCH="${REMOTE_SCRATCH_BASE}" EXPRESS_TORCH_INDEX="${TORCH_INDEX:-}"

# -------- Node layout (scratch is wiped at booking end; AFS home = code only) --------
S="${REMOTE_SCRATCH_BASE}"                                   # gpu_seg params: /scratch0/$USER at InstTwo; override per cluster
REMOTE_CODE_REL="asvproject_project/ASVProject-ObstacleDetection"   # code dir, relative to the node's $HOME
REMOTE_CODE='$HOME/'"${REMOTE_CODE_REL}"                    # expanded on the node
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"   # match the node's driver (cu121/cu118 for older drivers)
RAW="${S}/asvproject_raw"                 # staged inputs: captures/, lars_mastr/, thermal_inits/, labels/

# -------- Local sources (SSD) --------
SSD="/Volumes/ROS2_SSD/asvproject"
CAPTURES="${SSD}/captures"
LARS_MASTR_LOCAL="${SSD}/data/lars_mastr"             # built by scripts.lars.prepare if missing
LARS_RAW_LOCAL="/Volumes/ROS2_SSD/LaRS_v1.0.0"
THERMAL_INITS_LOCAL="${SSD}/models/thermal_student_v0"
LABELS_LOCAL="${LOCAL}/labels"                        # labels/qwen seeds the node's label set
AUDIT_PLAN="${LOCAL}/results/audit_plan.csv"          # clips named here are staged too ("" = none)

# -------- Data scope --------
# MISSIONS: capture missions to stage in full (all streams, all chunks). Empty = every 2026-*
# mission except stability/bench/lab/eval (stage_raw.sh's rule). Nested missions
# (2026-08-19_afloat/{session,canoe_fx,...}) are walked by the queue since 1508db5.
MISSIONS="2026-06-17_institutionone_day1 2026-07-08 2026-08-18_pontoon 2026-08-19_afloat 2026-08-26_afloat"
EXTRA_TRIPLETS=""                                     # explicit data/captures/<m>[/<sub>]/<ts> paths
STAGE_STREAMS="fisheye thermal mmwave imu frames gps" # which file kinds to stage (+ recovered copies)

# -------- What to run on the node (04_launch) --------
RUN_TRACKB=1        # native frames+masks+det, thermal ladder, fusion scorer, sectors, native DINO
TRACKB_STAGES="4 5 7 8 9 6"   # subset/order of night_queue_trackb.sh stages (4 export, 5 native predict,
                              # 7 thermal ladder, 8 fusion features/targets/train, 9 sectors, 6 native DINO)
DINO_ONLY="2026-08-26_afloat" # scope of the native GroundingDINO relabel (stage 6); "" = whole corpus (~7 h)
THERMAL_SEEDS="0 1 2"; THERMAL_EPOCHS=60
SCORER_TAG="night_v1"         # fusion_model.train --tag
RUN_DART=1          # DART/SAM3 labeller over the staged frames (scripts/gpu_dart, its own params)
RUN_YOLO=0          # on-domain YOLO fine-tune (needs an audited/exported dataset, see YOLO_* below)
YOLO_LABELS="${LOCAL}/labels/training_frames.jsonl"   # export_yolo input (audited frames)
YOLO_PSEUDO="";  YOLO_PSEUDO_COUNT=0                   # optional pseudo-label mix-in
YOLO_BASE="yolov8n.pt"; YOLO_EPOCHS=100; YOLO_IMGSZ=864; YOLO_BATCH=16

# -------- Pull / archive (06_pull) --------
ARCHIVE_ROOT="${SSD}"                                  # -> ${SSD}/<node>_<YYYY-MM-DD>/
PULL_PLACE_LABELS_DART=1                               # -> labels/dart/ (never labels/qwen)
PULL_PLACE_NATIVE_SEG=0                                # -> data/seg/ (supersedes bundle-res masks; opt-in)
PULL_NATIVE_FRAMES=0                                   # 7 GB of re-exportable JPEGs; off by default

# -------- Derived (do not edit) --------
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=25"
JUMPOPT="${JUMP_HOST:+-J ${REMOTE_USER}@${JUMP_HOST}}"; REMOTE="${REMOTE_USER}@${GPU_NODE}"
RSH="ssh ${SSH_OPTS} ${JUMPOPT}"
RENV_BASE="S=${S} CODE=${REMOTE_CODE} TORCH_INDEX=${TORCH_INDEX}${RENV_EXTRA:+ ${RENV_EXTRA}}"   # every node-side script gets these; profiles may append via RENV_EXTRA (e.g. CUDA_VISIBLE_DEVICES on shared boxes)
NOISE="${NOISE:-VBoxManage\\|VirtualBox}"                  # login-noise filter (InstTwo VirtualBox spew; profiles override)
rs() { local a; for a in 1 2 3 4 5 6; do rsync -az --partial --rsh="ssh ${SSH_OPTS} ${JUMPOPT}" "$@" && return 0; echo "  rsync attempt $a failed; 20 s" >&2; sleep 20; done; return 1; }
remote() { $RSH "$REMOTE" "env ${RENV_BASE} bash -s" 2>&1 | grep -v "$NOISE"; }             # pipe a script: remote < file
remote_env() { local e="$1"; shift; $RSH "$REMOTE" "env ${RENV_BASE} $e bash -s" 2>&1 | grep -v "$NOISE"; }
ts() { date +%H:%M:%S; }
need_ssd() { [[ -d "${CAPTURES}" ]] || { echo "SSD not mounted (${CAPTURES})" >&2; exit 1; }; }
sync_code() { echo "[$(ts)] code -> node home"; $RSH "$REMOTE" "mkdir -p ${REMOTE_CODE_REL}" 2>&1 | grep -v "$NOISE" || true; rsync -az --rsh="ssh ${SSH_OPTS} ${JUMPOPT}" --exclude='__pycache__' "${LOCAL}/scripts" "${LOCAL}/configs" "${REMOTE}:${REMOTE_CODE_REL}/"; [[ -f "${LOCAL}/requirements.txt" ]] && rsync -az --rsh="ssh ${SSH_OPTS} ${JUMPOPT}" "${LOCAL}/requirements.txt" "${REMOTE}:${REMOTE_CODE_REL}/" || true; }

# -------- Local override (gitignored) --------
if [[ -f "${EXPRESS_DIR}/00_params.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "${EXPRESS_DIR}/00_params.local.sh"
fi
