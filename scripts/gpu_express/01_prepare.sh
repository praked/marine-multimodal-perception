#!/usr/bin/env bash
# 01: fresh booking -> code on the node, corpus venv, published weights, rclone
#     (if dashboard creds exist), DART env + gated checkpoint (RUN_DART), YOLO env (RUN_YOLO).
# Idempotent: re-run any time. ~5 min when the venv already exists, ~15 min from zero.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_params.sh"
G="${SCRIPT_DIR}/../gpu_corpus"

sync_code
echo "[$(ts)] corpus venv (torch cu126 + inference deps)"
remote < "${G}/setup_env.sh" | tail -4
echo "[$(ts)] published weights (HF hf-handle collection)"
remote < "${G}/fetch_weights.sh" | tail -8
if grep -qE '^R2_ACCOUNT_ID=' "${LOCAL}/dashboard/.env.local" 2>/dev/null; then
  echo "[$(ts)] rclone (R2 creds from dashboard/.env.local; conf on scratch only)"
  R2_ENV=$(grep -E '^R2_(ACCOUNT_ID|ACCESS_KEY_ID|SECRET_ACCESS_KEY)=' "${LOCAL}/dashboard/.env.local" | tr '\n' ' ')
  remote_env "$R2_ENV" < "${G}/setup_rclone.sh" | tail -2
else
  echo "[$(ts)] rclone skipped (no dashboard/.env.local R2 creds)"
fi
if [[ "${RUN_DART}" == 1 ]]; then
  echo "[$(ts)] DART env + checkpoint"
  bash "${SCRIPT_DIR}/../gpu_dart/run_dart.sh" --prepare || echo "  (DART checkpoint step needs your action — see above)"
fi
if [[ "${RUN_YOLO}" == 1 ]]; then
  echo "[$(ts)] YOLO env (cu118 venv on scratch)"
  remote_env "YOLO_SCRATCH=${S}/asvproject_yolo" < "${SCRIPT_DIR}/../gpu_finetune/setup_scratch_yolo.sh" | tail -3
fi
echo "[$(ts)] 01 PREPARE DONE"
