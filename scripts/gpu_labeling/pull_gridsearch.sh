#!/usr/bin/env bash
# Pull the grid-search results (summary.csv + per-config contact sheets +
# labels) from the GPU node into results/grid/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new"

GRID_NAME="${GRID_NAME:-grid_fine}"
LOCAL_GRID="${LOCAL_PROJECT_ROOT}/results/${GRID_NAME}"
mkdir -p "${LOCAL_GRID}"

echo "Pulling grid results -> ${LOCAL_GRID} ..."
rsync -avzP -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
  "${SSH_REMOTE}:${REMOTE_OUT_DIR}/${GRID_NAME}/" "${LOCAL_GRID}/"

echo ""
echo "Done. Review:"
echo "  ${LOCAL_GRID}/summary.csv"
echo "  open ${LOCAL_GRID}/*/contact.png"
