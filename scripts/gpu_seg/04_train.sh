#!/usr/bin/env bash
# Launch eWaSR training on LaRS (nohup, resumable, survives SSH disconnect).
# Prints the pid + log path; poll with:  bash 04_train.sh --status
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

LOG="${SEG_SCRATCH}/train.log"

if [[ "${1:-}" == "--status" ]]; then
  ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'bash -s' <<EOF
set +e
echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- python procs ---"; pgrep -af "train.py" || echo "no train.py running"
echo "--- tail train.log ---"; tail -n 25 '${LOG}' 2>/dev/null || echo "no log yet"
echo "--- weights produced? ---"; ls -la ${SEG_RUNS}/logs/ewasr_lars/version_*/weights.pth 2>/dev/null || echo "none yet"
EOF
  exit 0
fi

echo "Launching eWaSR training (model=${MODEL}, epochs=${EPOCHS}, batch=${BATCH_SIZE}) ..."
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'bash -s' <<EOF
set -e
source '${SEG_SHIM}'
cd '${SEG_EWASR}'
mkdir -p '${SEG_RUNS}'
nohup python train.py \
  --train_config '${SEG_MASTR}/mastr_train.yaml' \
  --val_config '${SEG_MASTR}/mastr_val.yaml' \
  --model '${MODEL}' --model_name ewasr_lars \
  --num_classes ${NUM_CLASSES} --validation \
  --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --patience ${PATIENCE} \
  --output_dir '${SEG_RUNS}' --workers 4 \
  > '${LOG}' 2>&1 &
echo "train pid: \$!"
echo "log: ${LOG}"
sleep 8
echo "--- first lines ---"; head -n 15 '${LOG}' 2>/dev/null || true
EOF

echo ""
echo "Training launched. Poll with:  bash ${SCRIPT_DIR}/04_train.sh --status"
