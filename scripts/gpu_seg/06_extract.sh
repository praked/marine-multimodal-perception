#!/usr/bin/env bash
# Pull results back from scratch BEFORE the booking ends (scratch is wiped):
#   - predicted masks  -> data/seg/<clip>/
#   - trained weights  -> models/
#   - train log        -> models/train.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

mkdir -p "${LOCAL_SEG_OUT_DIR}" "${LOCAL_MODELS_DIR}"

echo "[1/3] Pulling predicted masks -> ${LOCAL_SEG_OUT_DIR} ..."
rsync -avzP \
  -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
  "${SSH_REMOTE}:${SEG_PRED_OUT}/" \
  "${LOCAL_SEG_OUT_DIR}/" || echo "  (no masks yet)"

echo ""
echo "[2/3] Pulling trained weights -> ${LOCAL_MODELS_DIR} ..."
# weights.pth from the latest training version dir, renamed for clarity.
# Remote login shell is tcsh, which rejects `2>/dev/null`: run the glob in bash.
LARS_W="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" \
  bash -lc "'ls -t ${SEG_RUNS}/logs/ewasr_lars/version_*/weights.pth 2>/dev/null | head -1'" || true)"
if [[ -n "${LARS_W}" ]]; then
  rsync -avzP -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
    "${SSH_REMOTE}:${LARS_W}" "${LOCAL_MODELS_DIR}/ewasr_resnet18_lars.pth"
else
  echo "  (no LaRS weights yet)"
fi

echo ""
echo "[3/3] Pulling train log ..."
rsync -avz -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
  "${SSH_REMOTE}:${SEG_SCRATCH}/train.log" "${LOCAL_MODELS_DIR}/train.log" || true

echo ""
echo "Extract complete."
echo "  masks  : ${LOCAL_SEG_OUT_DIR}"
echo "  weights: ${LOCAL_MODELS_DIR}"
