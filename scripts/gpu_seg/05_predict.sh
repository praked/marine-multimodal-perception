#!/usr/bin/env bash
# Predict clean label masks for every uploaded clip dir under SEG_PRED_IN.
# Uses the LaRS-trained weights if present, else the pretrained MaSTr weights.
# Override the weights with:  WEIGHTS=/path/to/weights.pth bash 05_predict.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

REMOTE_HOME_DIR="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
PREDICT_PY="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}/scripts/gpu_seg/predict_masks.py"
WEIGHTS_OVERRIDE="${WEIGHTS:-}"

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'bash -s' <<EOF
set -e
source '${SEG_SHIM}'

# Pick weights: explicit override > LaRS-trained > pretrained MaSTr.
WEIGHTS='${WEIGHTS_OVERRIDE}'
if [[ -z "\$WEIGHTS" ]]; then
  LARS_W=\$(ls -t ${SEG_RUNS}/logs/ewasr_lars/version_*/weights.pth 2>/dev/null | head -1 || true)
  if [[ -n "\$LARS_W" ]]; then
    WEIGHTS="\$LARS_W"
  else
    WEIGHTS='${SEG_WEIGHTS_DIR}/ewasr_resnet18_mastr.pth'
  fi
fi
echo "Using weights: \$WEIGHTS"

mkdir -p '${SEG_PRED_OUT}'
for clip_dir in ${SEG_PRED_IN}/*/; do
  [[ -d "\$clip_dir" ]] || continue
  name=\$(basename "\$clip_dir")
  echo "=== predicting \$name ==="
  python '${PREDICT_PY}' \
    --ewasr-dir '${SEG_EWASR}' \
    --image-dir "\$clip_dir" \
    --weights "\$WEIGHTS" \
    --model '${MODEL}' \
    --out-dir "${SEG_PRED_OUT}/\$name" \
    --size '${TRAIN_SIZE}' --overlay
done
echo "Prediction complete -> ${SEG_PRED_OUT}"
EOF
