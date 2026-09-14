#!/usr/bin/env bash
# LOCAL: stage the RAW corpus (mp4/csv triplets), LaRS-as-MaSTr tree and the
# thermal inits to the GPU node, then launch the Track-B queue. Needs the
# ROS2_SSD mounted (data/captures etc. are symlinks onto it).
#
#   bash scripts/gpu_corpus/stage_raw.sh            # stage + launch
#   bash scripts/gpu_corpus/stage_raw.sh --dry-run  # sizes only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../gpu_seg/00_run_params.sh"
LOCAL="${LOCAL_PROJECT_ROOT}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=20"
# rsync with retries: a dead SSH pipe otherwise stalls the whole staging silently
# (observed 2026-08-27: 0 B/s for 30 min at 0% CPU). --partial resumes big mp4s.
rs() {
  for attempt in 1 2 3 4 5 6; do
    rsync -az --partial -e "ssh ${SSH_OPTS} -J ${JUMP}" "$@" && return 0
    echo "  rsync attempt $attempt failed (rc=$?), retrying in 20 s" >&2; sleep 20
  done
  return 1
}
JUMP="${REMOTE_USER}@${JUMP_HOST}"
REMOTE="${REMOTE_USER}@${GPU_NODE}"
RSH="ssh ${SSH_OPTS} -J ${JUMP}"
RAW=/scratch0/${REMOTE_USER}/asvproject_raw

[[ -d /Volumes/ROS2_SSD/asvproject/captures ]] || {
  echo "ROS2_SSD not mounted — plug it in first." >&2; exit 1; }

# 2026 corpus missions only (matches the dashboard corpus; excludes stability/
# bench/eval sets the baker skips).
MISSIONS=$(ls /Volumes/ROS2_SSD/asvproject/captures | grep '^2026' \
  | grep -vE 'stability|bench|_lab|pi_card_backup|eval_smoke')

echo "staging missions:"; echo "$MISSIONS" | sed 's/^/  /'
du -sh $(echo "$MISSIONS" | sed 's|^|/Volumes/ROS2_SSD/asvproject/captures/|') 2>/dev/null
[[ "${1:-}" == "--dry-run" ]] && exit 0

$RSH "$REMOTE" "mkdir -p $RAW/captures $RAW/lars_mastr $RAW/thermal_inits $RAW/labels" 2>&1 | grep -v VBoxManage || true

echo "[1/4] raw captures -> node scratch"
for m in $MISSIONS; do
  rs "/Volumes/ROS2_SSD/asvproject/captures/$m/" "$REMOTE:$RAW/captures/$m/"
  echo "  $m done"
done

echo "[2/4] lars_mastr (~180M)"
rs /Volumes/ROS2_SSD/asvproject/data/lars_mastr/ "$REMOTE:$RAW/lars_mastr/"

echo "[3/4] thermal inits + existing labels"
rs /Volumes/ROS2_SSD/asvproject/models/thermal_student_v0/ "$REMOTE:$RAW/thermal_inits/" || true
rs "$LOCAL/labels/" "$REMOTE:$RAW/labels/"

echo "[4/4] launching Track-B queue"
rsync -az -e "ssh ${SSH_OPTS} -J ${JUMP}" \
  --include='scripts/***' --include='configs/***' --exclude='*' --exclude='__pycache__' \
  "$LOCAL/" "$REMOTE:asvproject_project/ASVProject-ObstacleDetection/"
$RSH "$REMOTE" 'bash -s' <<'EOF' 2>&1 | grep -v VBoxManage
S=/scratch0/$USER
CODE=$HOME/asvproject_project/ASVProject-ObstacleDetection
cp "$CODE/scripts/gpu_corpus/night_queue_trackb.sh" "$S/night_queue_trackb.sh"
if pgrep -f "$S/night_queue_trackb.sh" >/dev/null; then
  echo "track B already running"
else
  nohup bash "$S/night_queue_trackb.sh" >> "$S/logs/night_queue_trackb.log" 2>&1 &
  echo "track B launched pid $!"
fi
EOF
echo "poll with: bash scripts/gpu_corpus/run_night.sh --status"
