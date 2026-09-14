#!/usr/bin/env bash
# Select the cluster profile once: ./use_profile.sh cluster-a | institutionone
# Writes profiles/ACTIVE (gitignored, machine-local); every numbered step
# then resolves it. One-off override: GPU_PROFILE=<name> ./<step>.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
have() { ls "${HERE}/profiles" | sed -n 's/\.sh$//p' | grep -v '\.template$' | tr '\n' ' '; }
if [[ $# -ne 1 ]]; then
  echo "usage: $0 <profile>   (available: $(have))" >&2
  echo "active: $(cat "${HERE}/profiles/ACTIVE" 2>/dev/null || echo '<none — legacy gpu_seg params>')" >&2
  exit 1
fi
[[ -f "${HERE}/profiles/$1.sh" ]] || { echo "no profiles/$1.sh (available: $(have))" >&2; exit 1; }
echo "$1" > "${HERE}/profiles/ACTIVE"
# shellcheck source=/dev/null
source "${HERE}/00_params.sh"
echo "profile: $1"
echo "  node:    ${REMOTE_USER}@${GPU_NODE}${JUMP_HOST:+ (via ${JUMP_HOST})}"
echo "  scratch: ${REMOTE_SCRATCH_BASE}"
echo "  torch:   ${TORCH_INDEX}"
echo "next: ./02_preflight.sh to verify the node end-to-end"
