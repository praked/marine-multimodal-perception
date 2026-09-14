#!/usr/bin/env bash
# Print the GPU node's state (nvidia-smi, scratch free space, staged inputs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'bash -s' <<EOF
set +e
echo "host: \$(hostname)"
echo "--- gpu ---"; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
echo "--- scratch ---"; df -h /scratch0 | tail -1
echo "--- staged MaSTr tree ---"
ls -la '${SEG_MASTR}' 2>/dev/null | head -8
echo "train list: \$(wc -l < '${SEG_MASTR}/train_images.txt' 2>/dev/null) val list: \$(wc -l < '${SEG_MASTR}/val_images.txt' 2>/dev/null)"
echo "images: \$(ls '${SEG_MASTR}/images' 2>/dev/null | wc -l) masks: \$(ls '${SEG_MASTR}/masks' 2>/dev/null | wc -l)"
echo "--- predict frame clips ---"; ls -la '${SEG_PRED_IN}' 2>/dev/null
echo "--- venv present? ---"; ls '${SEG_VENV}/bin/python' 2>/dev/null && echo yes || echo "no (run 03_setup)"
EOF
