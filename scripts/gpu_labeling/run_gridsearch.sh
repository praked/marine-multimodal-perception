#!/usr/bin/env bash
# Launch the GroundingDINO parameter grid search on the GPU node under nohup and
# leave it running. Sweeps exposure x resolution x structure thresholds
# (configs/detector_gridsearch.yaml) over the smoke set; writes a contact sheet
# + summary row per config. Pull results with pull_gridsearch.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new"
RSH="ssh ${SSH_OPTS} -J ${SSH_JUMP}"

REMOTE_HOME_DIR="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
REMOTE_CODE_DIR="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}"
# GRID_NAME picks the output subdir so successive sweeps don't clobber each
# other (override: GRID_NAME=grid2 bash run_gridsearch.sh).
GRID_NAME="${GRID_NAME:-grid_fine}"
REMOTE_GRID_OUT="${REMOTE_OUT_DIR}/${GRID_NAME}"
REMOTE_GRID_LOG="${REMOTE_OUT_DIR}/${GRID_NAME}.log"

echo "Syncing detector code + grid spec + smoke set ..."
$RSH "${SSH_REMOTE}" "mkdir -p '${REMOTE_CODE_DIR}/configs' '${REMOTE_CODE_DIR}/scripts/eval' '${REMOTE_CODE_DIR}/labels/qwen' '${REMOTE_GRID_OUT}'"
for f in scripts/eval/detector_labeler.py scripts/eval/detector_gridsearch.py \
         scripts/eval/qwen_batch_labeler.py configs/detector_gridsearch.yaml \
         labels/qwen/smoke_set.txt; do
  rsync -az -e "${RSH}" "${LOCAL_PROJECT_ROOT}/${f}" "${SSH_REMOTE}:${REMOTE_CODE_DIR}/${f}"
done

echo "Launching grid search under nohup ..."
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -s <<EOF
set -e
cat > '${REMOTE_OUT_DIR}/launch_gridsearch.sh' <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
source '${ACTIVATE_SHIM}'
cd '${REMOTE_CODE_DIR}'
export PYTHONUNBUFFERED=1
exec python -m scripts.eval.detector_gridsearch \
  --captures-dir '${REMOTE_DATA_DIR}' \
  --frames-file '${REMOTE_CODE_DIR}/labels/qwen/smoke_set.txt' \
  --grid '${REMOTE_CODE_DIR}/configs/detector_gridsearch.yaml' \
  --out-dir '${REMOTE_GRID_OUT}'
LAUNCH
chmod +x '${REMOTE_OUT_DIR}/launch_gridsearch.sh'
: > '${REMOTE_GRID_LOG}'
nohup bash '${REMOTE_OUT_DIR}/launch_gridsearch.sh' >> '${REMOTE_GRID_LOG}' 2>&1 < /dev/null &
echo "started PID \$!"
EOF

echo ""
echo "Grid search running. Monitor:"
echo "  ssh -i ${SSH_KEY_FILE} -J ${SSH_JUMP} ${SSH_REMOTE} \"tail -f '${REMOTE_GRID_LOG}'\""
echo "Progress (summary rows so far):"
echo "  ssh -i ${SSH_KEY_FILE} -J ${SSH_JUMP} ${SSH_REMOTE} \"cat '${REMOTE_GRID_OUT}/summary.csv'\""
echo "When done, pull results:"
echo "  bash scripts/gpu_labeling/pull_gridsearch.sh"
