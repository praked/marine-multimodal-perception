#!/usr/bin/env bash
# Label the whole mission on the GPU with the current best detector config
# (configs/detector_labeler.yaml). Runs under nohup; resumable (skip-done +
# per-frame cache), so it is safe to leave running and pull later.
#
#   STRIDE=3 bash scripts/gpu_labeling/run_detector_full.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

STRIDE="${STRIDE:-3}"
SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new"
RSH="ssh ${SSH_OPTS} -J ${SSH_JUMP}"

REMOTE_HOME_DIR="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
REMOTE_CODE_DIR="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}"
REMOTE_OUT="${REMOTE_OUT_DIR}/detector_full/det_${MISSION}.jsonl"
REMOTE_CACHE="${REMOTE_OUT_DIR}/detector_full/cache"
REMOTE_LOG="${REMOTE_OUT_DIR}/detector_full.log"

echo "Syncing detector code + config ..."
$RSH "${SSH_REMOTE}" "mkdir -p '${REMOTE_CODE_DIR}/configs' '${REMOTE_CODE_DIR}/scripts/eval' '${REMOTE_OUT_DIR}/detector_full'"
for f in scripts/eval/detector_labeler.py scripts/eval/qwen_batch_labeler.py \
         configs/detector_labeler.yaml; do
  rsync -az -e "${RSH}" "${LOCAL_PROJECT_ROOT}/${f}" "${SSH_REMOTE}:${REMOTE_CODE_DIR}/${f}"
done

echo "Launching full detector labelling (stride ${STRIDE}) under nohup ..."
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -s <<EOF
set -e
cat > '${REMOTE_OUT_DIR}/launch_detector_full.sh' <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
source '${ACTIVATE_SHIM}'
cd '${REMOTE_CODE_DIR}'
export PYTHONUNBUFFERED=1
exec python -m scripts.eval.detector_labeler \
  --captures-dir '${REMOTE_DATA_DIR}' \
  --every ${STRIDE} \
  --out '${REMOTE_OUT}' --cache-dir '${REMOTE_CACHE}' \
  --exposure ${EXPOSURE:-1.0} --batch-size 8
LAUNCH
chmod +x '${REMOTE_OUT_DIR}/launch_detector_full.sh'
: > '${REMOTE_LOG}'
nohup bash '${REMOTE_OUT_DIR}/launch_detector_full.sh' >> '${REMOTE_LOG}' 2>&1 < /dev/null &
echo "started PID \$!"
EOF

echo ""
echo "Running. Monitor:"
echo "  ssh -i ${SSH_KEY_FILE} -J ${SSH_JUMP} ${SSH_REMOTE} \"tail -f '${REMOTE_LOG}'\""
echo "  ssh -i ${SSH_KEY_FILE} -J ${SSH_JUMP} ${SSH_REMOTE} \"wc -l '${REMOTE_OUT}'\""
echo "Pull when done:"
echo "  rsync -avzP -e \"ssh ${SSH_OPTS} -J ${SSH_JUMP}\" ${SSH_REMOTE}:${REMOTE_OUT} labels/qwen/det_${MISSION}.jsonl"
