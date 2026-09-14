#!/usr/bin/env bash
# From nothing on the node to everything running:  01_prepare -> 03_stage -> 04_launch.
#   bash scripts/gpu_express/run_all.sh            # all three
#   bash scripts/gpu_express/run_all.sh --from 3   # skip prepare
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROM=1; [[ "${1:-}" == "--from" ]] && FROM="$2"
[[ $FROM -le 1 ]] && bash "${SCRIPT_DIR}/01_prepare.sh"
[[ $FROM -le 3 ]] && bash "${SCRIPT_DIR}/03_stage.sh"
bash "${SCRIPT_DIR}/04_launch.sh"
echo "everything launched — poll: bash scripts/gpu_express/05_status.sh --watch ; extract before the booking ends: bash scripts/gpu_express/06_pull.sh"
