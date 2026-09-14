#!/usr/bin/env bash
# Label ONE mission (or one clip) on the GPU node with the current best
# detector config: the generalised sibling of run_detector_full.sh, for
# pre-seeding labels on new field captures (e.g. the 2026-07-08 quad clip)
# without editing 00_run_params.sh.
#
#   MISSION=2026-07-08 CLIP=2026-07-08_16-37-01 STRIDE=6 \
#       bash scripts/gpu_labeling/run_detector_mission.sh
#
# MISSION   capture folder under data/captures/ (required)
# CLIP      single clip timestamp; empty = every clip in the mission
# STRIDE    frame stride (default 6 ≈ one label / 2 s at 3 fps)
# SETUP=1   chain setup_scratch_labeling first (idempotent; needed after a
#           node reimage/booking change: detects automatically if unset)
#
# Data synced per-clip (recovered/ layout preserved so the updated
# datasets.resolve_triplet picks the trimmed copies). Job runs under nohup;
# resumable. Pull results with:
#   rsync -e "ssh -i ~/.ssh/cluster_key -J <user>@gpu-node..." \
#     <user>@<node>:/scratch0/<user>/asvproject/out/detector_<MISSION>/det_<MISSION>.jsonl labels/qwen/
set -euo pipefail

# Capture the caller's env BEFORE sourcing params: 00_run_params.sh sets
# MISSION unconditionally and would clobber it.
CALLER_MISSION="${MISSION:?set MISSION=<capture folder name>}"
CLIP="${CLIP:-}"
STRIDE="${STRIDE:-6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"
MISSION="${CALLER_MISSION}"

# Re-derive the remote paths 00_run_params computed from its own MISSION.
REMOTE_DATA_DIR="${LABEL_SCRATCH}/data/captures/${MISSION}"
REMOTE_OUT_DIR="${LABEL_SCRATCH}/out"
LOCAL_DATA_DIR="${LOCAL_PROJECT_ROOT}/data/captures/${MISSION}"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new"
RSH="ssh ${SSH_OPTS} -J ${SSH_JUMP}"

REMOTE_HOME_DIR="$($RSH "${SSH_REMOTE}" 'printf %s "$HOME"')"
REMOTE_CODE_DIR="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}"
REMOTE_JOB_DIR="${REMOTE_OUT_DIR}/detector_${MISSION}"
REMOTE_OUT="${REMOTE_JOB_DIR}/det_${MISSION}.jsonl"
REMOTE_LOG="${REMOTE_JOB_DIR}/run.log"

echo "node=${GPU_NODE}  mission=${MISSION}  clip=${CLIP:-<all>}  stride=${STRIDE}"

echo "[1/3] Syncing code to persistent home ..."
$RSH "${SSH_REMOTE}" "mkdir -p '${REMOTE_CODE_DIR}' '${REMOTE_DATA_DIR}/recovered' '${REMOTE_JOB_DIR}'"
rsync -az -e "${RSH}" \
  --exclude '.git' --exclude 'data' --exclude 'labels/cache' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'images' \
  --exclude 'CAD' --exclude '*.pdf' --exclude '*.mp4' --exclude 'results' \
  --exclude 'models' \
  "${LOCAL_PROJECT_ROOT}/" "${SSH_REMOTE}:${REMOTE_CODE_DIR}/"

echo "[2/3] Syncing clip data (recovered/ layout preserved) ..."
if [[ -n "${CLIP}" ]]; then
  PATTERNS=(--include "*_${CLIP}*.mp4" --include "*_${CLIP}*.csv")
else
  PATTERNS=(--include '*.mp4' --include '*.csv')
fi
rsync -az -e "${RSH}" "${PATTERNS[@]}" --exclude '*' \
  "${LOCAL_DATA_DIR}/" "${SSH_REMOTE}:${REMOTE_DATA_DIR}/"
if [[ -d "${LOCAL_DATA_DIR}/recovered" ]]; then
  rsync -az -e "${RSH}" "${PATTERNS[@]}" --exclude '*' \
    "${LOCAL_DATA_DIR}/recovered/" "${SSH_REMOTE}:${REMOTE_DATA_DIR}/recovered/"
fi

echo "[3/3] Launching (setup chained if scratch env is absent) ..."
CLIP_ARGS=""
if [[ -n "${CLIP}" ]]; then
  CLIP_ARGS="--from '${CLIP}' --to '${CLIP}'"
fi
$RSH "${SSH_REMOTE}" bash -s <<EOF
set -e
cat > '${REMOTE_JOB_DIR}/launch.sh' <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
export LABEL_SCRATCH='${LABEL_SCRATCH}'
if [[ ! -f '${ACTIVATE_SHIM}' ]]; then
  echo "[launch] scratch env absent -> running setup_scratch_labeling"
  bash '${REMOTE_CODE_DIR}/scripts/gpu_labeling/setup_scratch_labeling.sh'
fi
source '${ACTIVATE_SHIM}'
cd '${REMOTE_CODE_DIR}'
export PYTHONUNBUFFERED=1
exec python -m scripts.eval.detector_labeler \
  --captures-dir '${REMOTE_DATA_DIR}' \
  --every ${STRIDE} ${CLIP_ARGS} \
  --out '${REMOTE_OUT}' --cache-dir '${REMOTE_JOB_DIR}/cache' \
  --exposure ${EXPOSURE:-1.0} --batch-size 8
LAUNCH
chmod +x '${REMOTE_JOB_DIR}/launch.sh'
: > '${REMOTE_LOG}'
nohup bash '${REMOTE_JOB_DIR}/launch.sh' >> '${REMOTE_LOG}' 2>&1 < /dev/null &
echo "started PID \$!  log: ${REMOTE_LOG}"
EOF

echo ""
echo "monitor : ${RSH} ${SSH_REMOTE} 'tail -5 ${REMOTE_LOG}'"
echo "pull    : rsync -az -e \"${RSH}\" ${SSH_REMOTE}:${REMOTE_OUT} ${LOCAL_PROJECT_ROOT}/labels/qwen/"
