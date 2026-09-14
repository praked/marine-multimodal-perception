#!/usr/bin/env bash
# LOCAL driver: sync code, build the env, fetch weights, launch the Track-A
# night queue on the GPU node under nohup. Safe to re-run (everything is
# idempotent / resumable).
#
#   bash scripts/gpu_corpus/run_night.sh            # full launch
#   ONLY=2026-08-26_afloat bash scripts/gpu_corpus/run_night.sh   # scope stages 2-3
#   EVERY=1 ONLY=... bash scripts/gpu_corpus/run_night.sh          # DINO on every frame (default stride 2)
#   bash scripts/gpu_corpus/run_night.sh --prepare  # steps 1-2 only (fresh booking, bundle not yet published)
#   bash scripts/gpu_corpus/run_night.sh --status   # poll
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../gpu_seg/00_run_params.sh"   # SSH topology + node
LOCAL="${LOCAL_PROJECT_ROOT}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
JUMP="${REMOTE_USER}@${JUMP_HOST}"
REMOTE="${REMOTE_USER}@${GPU_NODE}"
RSH="ssh ${SSH_OPTS} -J ${JUMP}"

if [[ "${1:-}" == "--status" ]]; then
  $RSH "$REMOTE" 'bash -s' <<'EOF' 2>&1 | grep -v VBoxManage
S=/scratch0/$USER
echo "=== STATUS ==="; tail -n 12 $S/logs/STATUS 2>/dev/null
echo "=== last log lines ==="
for f in $S/logs/predict_bundle.log $S/logs/label_dino.log; do
  [[ -f $f ]] && { echo "--- $(basename $f)"; tail -n 3 "$f"; }
done
pgrep -fa "night_queue|predict_bundle|label_bundle" | grep -v pgrep || echo "(queue idle)"
EOF
  exit 0
fi

echo "[1/3] code -> node home"
rsync -az -e "ssh ${SSH_OPTS} -J ${JUMP}" \
  --include='scripts/***' --include='configs/***' --include='requirements*.txt' \
  --exclude='*' --exclude='__pycache__' \
  "$LOCAL/" "$REMOTE:asvproject_project/ASVProject-ObstacleDetection/"

echo "[2/3] env + weights + rclone (idempotent)"
$RSH "$REMOTE" 'bash -s' < "${SCRIPT_DIR}/setup_env.sh" 2>&1 | grep -v VBoxManage | tail -4
$RSH "$REMOTE" 'bash -s' < "${SCRIPT_DIR}/fetch_weights.sh" 2>&1 | grep -v VBoxManage | tail -12
# R2 creds from the dashboard env (never stored on the node's home; the conf
# lives on scratch, 0600, and dies with the booking)
R2_ENV=$(grep -E '^R2_(ACCOUNT_ID|ACCESS_KEY_ID|SECRET_ACCESS_KEY)=' "$LOCAL/dashboard/.env.local" | tr '\n' ' ')
$RSH "$REMOTE" "env $R2_ENV bash -s" < "${SCRIPT_DIR}/setup_rclone.sh" 2>&1 | grep -v VBoxManage | tail -2

if [[ "${1:-}" == "--prepare" ]]; then
  echo "[3/3] skipped (--prepare): env, weights and rclone are staged; launch later"
  exit 0
fi
echo "[3/3] launching night queue under nohup${ONLY:+ (ONLY=$ONLY)}"
$RSH "$REMOTE" "env ONLY='${ONLY:-}' EVERY='${EVERY:-2}' bash -s" <<'EOF' 2>&1 | grep -v VBoxManage
S=/scratch0/$USER
CODE=$HOME/asvproject_project/ASVProject-ObstacleDetection
cp "$CODE/scripts/gpu_corpus/night_queue.sh" "$S/night_queue.sh"
if pgrep -f "$S/night_queue.sh" >/dev/null; then
  echo "queue already running"
else
  ONLY="$ONLY" EVERY="$EVERY" nohup bash "$S/night_queue.sh" >> "$S/logs/night_queue.log" 2>&1 &
  echo "launched pid $!"
fi
EOF
echo "poll with: bash scripts/gpu_corpus/run_night.sh --status"
