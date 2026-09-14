#!/usr/bin/env bash
# Detector-iteration loop: edit configs/detector_labeler.yaml (model, queries,
# thresholds), then run this to smoke-test GroundingDINO on the fixed frame set
# and get a contact sheet to eyeball. Mirror of iterate_prompt.sh.
#
#   1. edit  configs/detector_labeler.yaml
#   2. bash scripts/gpu_labeling/iterate_detector.sh [tag]
#   3. open results/qwen_audit/det_<tag>/contact_01.png
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

TAG="${1:-$(date +%Y%m%d-%H%M%S)}"
SMOKE_SET_REL="labels/qwen/smoke_set.txt"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new"
RSH="ssh ${SSH_OPTS} -J ${SSH_JUMP}"

REMOTE_HOME_DIR="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
REMOTE_CODE_DIR="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}"
REMOTE_OUT="${REMOTE_OUT_DIR}/det_${TAG}.jsonl"
REMOTE_CACHE="${REMOTE_OUT_DIR}/det_cache_${TAG}"
REMOTE_LOG="${REMOTE_OUT_DIR}/det_${TAG}.log"
LOCAL_OUT_DIR="${LOCAL_PROJECT_ROOT}/results/qwen_audit/det_${TAG}"

echo "=== detector smoke test: tag=${TAG} ==="

echo "[1/4] syncing detector config + code + frame set ..."
$RSH "${SSH_REMOTE}" "mkdir -p '${REMOTE_CODE_DIR}/configs' '${REMOTE_CODE_DIR}/labels/qwen' '${REMOTE_OUT_DIR}'"
rsync -az -e "${RSH}" "${LOCAL_PROJECT_ROOT}/configs/detector_labeler.yaml" \
  "${SSH_REMOTE}:${REMOTE_CODE_DIR}/configs/detector_labeler.yaml"
rsync -az -e "${RSH}" "${LOCAL_PROJECT_ROOT}/scripts/eval/detector_labeler.py" \
  "${SSH_REMOTE}:${REMOTE_CODE_DIR}/scripts/eval/detector_labeler.py"
rsync -az -e "${RSH}" "${LOCAL_PROJECT_ROOT}/scripts/eval/qwen_batch_labeler.py" \
  "${SSH_REMOTE}:${REMOTE_CODE_DIR}/scripts/eval/qwen_batch_labeler.py"
rsync -az -e "${RSH}" "${LOCAL_PROJECT_ROOT}/${SMOKE_SET_REL}" \
  "${SSH_REMOTE}:${REMOTE_CODE_DIR}/${SMOKE_SET_REL}"

echo "[2/4] launching detector run on ${GPU_NODE} ..."
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -s <<EOF
set -e
cat > '${REMOTE_OUT_DIR}/launch_det_${TAG}.sh' <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
source '${ACTIVATE_SHIM}'
cd '${REMOTE_CODE_DIR}'
export PYTHONUNBUFFERED=1
exec python -m scripts.eval.detector_labeler \
  --captures-dir '${REMOTE_DATA_DIR}' \
  --frames-file '${REMOTE_CODE_DIR}/${SMOKE_SET_REL}' \
  --out '${REMOTE_OUT}' --cache-dir '${REMOTE_CACHE}' \
  --batch-size 8 --exposure ${EXPOSURE:-1.0} --no-resume
LAUNCH
chmod +x '${REMOTE_OUT_DIR}/launch_det_${TAG}.sh'
: > '${REMOTE_LOG}'
nohup bash '${REMOTE_OUT_DIR}/launch_det_${TAG}.sh' >> '${REMOTE_LOG}' 2>&1 < /dev/null &
echo "started PID \$!"
EOF

echo "[3/4] waiting for detector run ..."
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -c \
  "'for i in \$(seq 1 120); do grep -qE \"^done:|Traceback|Error\" \"${REMOTE_LOG}\" 2>/dev/null && break; sleep 5; done; grep -E \"^done:|Traceback|Error|prompt:|detector:\" \"${REMOTE_LOG}\" | head'" \
  2>&1 | grep -vE "VBoxManage|NS_ERROR|VirtualBox|tinderbox|Document is empty|Location:|Details:" || true

echo "[4/4] pulling results + rendering contact sheet ..."
mkdir -p "${LOCAL_OUT_DIR}"
rsync -az -e "${RSH}" "${SSH_REMOTE}:${REMOTE_OUT}" "${LOCAL_OUT_DIR}/det_${TAG}.jsonl"
rsync -az -e "${RSH}" "${SSH_REMOTE}:${REMOTE_LOG}" "${LOCAL_OUT_DIR}/det_${TAG}.log" || true
cp "${LOCAL_PROJECT_ROOT}/configs/detector_labeler.yaml" "${LOCAL_OUT_DIR}/detector_labeler.yaml"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
( cd "${LOCAL_PROJECT_ROOT}" && "${PYTHON}" -m scripts.eval.render_label_overlays \
    --labels "${LOCAL_OUT_DIR}/det_${TAG}.jsonl" \
    --captures-dir "${LOCAL_DATA_DIR}" --out-dir "${LOCAL_OUT_DIR}" \
    --exposure "${EXPOSURE:-1.0}" )

echo ""
echo "Done. Review:  ${LOCAL_OUT_DIR}/contact_01.png"
command -v open >/dev/null && open "${LOCAL_OUT_DIR}/contact_01.png" || true
