#!/usr/bin/env bash
# Run the scratch setup remotely (venv + torch cu118 + eWaSR deps, clone eWaSR,
# download pretrained MaSTr weights). Slow: 10-30 min for the torch install.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

REMOTE_HOME_DIR="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
REMOTE_SETUP="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}/scripts/gpu_seg/setup_scratch_seg.sh"

echo "Running remote setup (venv + eWaSR). This can take 10-30 min."
# tcsh login shell -> pipe a bash script to `bash -s` and export vars inside it.
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'bash -s' <<EOF
set -e
export SEG_SCRATCH='${SEG_SCRATCH}'
export EWASR_REPO='${EWASR_REPO}'
export PRETRAINED_URL='${PRETRAINED_URL}'
bash '${REMOTE_SETUP}'
EOF

echo "Remote setup finished."
