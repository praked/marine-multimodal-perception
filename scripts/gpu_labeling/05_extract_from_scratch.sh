#!/usr/bin/env bash
# Pull the Qwen labels + cache + log back from scratch (do this before the
# booking ends: scratch is wiped). Merges into labels/qwen/ locally.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

mkdir -p "${LOCAL_EXTRACT_DIR}/cache" "${LOCAL_EXTRACT_DIR}/logs"

echo "Pulling labels from scratch ..."
rsync -avP --timeout=120 \
  -e "ssh ${SSH_OPTS} -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -J ${SSH_JUMP}" \
  "${SSH_REMOTE}:${REMOTE_OUT_JSONL}" \
  "${LOCAL_EXTRACT_DIR}/$(basename "${REMOTE_OUT_JSONL}")"

echo "Pulling raw response cache ..."
rsync -avP \
  -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
  "${SSH_REMOTE}:${REMOTE_CACHE_DIR}/" \
  "${LOCAL_EXTRACT_DIR}/cache/" || true

echo "Pulling run log ..."
rsync -avP \
  -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
  "${SSH_REMOTE}:${REMOTE_RUN_LOG}" \
  "${LOCAL_EXTRACT_DIR}/logs/" || true

n=$(wc -l < "${LOCAL_EXTRACT_DIR}/$(basename "${REMOTE_OUT_JSONL}")" 2>/dev/null || echo 0)
echo ""
echo "Extraction complete: ${LOCAL_EXTRACT_DIR}/$(basename "${REMOTE_OUT_JSONL}") (${n} labels)"
echo ""
echo "Next: audit the provisional labels, e.g."
echo "  python -m scripts.eval.label_tool --audit ${LOCAL_EXTRACT_DIR}/$(basename "${REMOTE_OUT_JSONL}")"
