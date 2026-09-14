#!/usr/bin/env bash
# Run the scratch setup remotely on the GPU node (creates venv, downloads model).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

REMOTE_HOME_DIR="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
REMOTE_SETUP="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}/scripts/gpu_labeling/setup_scratch_labeling.sh"

echo "Running remote setup (venv + model download). This can take 10-30 min."
echo "  model: ${MODEL}"
# GPU nodes default to tcsh, so pipe a bash script to `bash -s` rather than
# relying on a `VAR=val cmd` prefix (which tcsh does not support).
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -s <<EOF
export LABEL_SCRATCH='${LABEL_SCRATCH}'
bash '${REMOTE_SETUP}' '${MODEL}'
EOF

echo "Remote setup finished."
