#!/usr/bin/env bash
# Build the (tiny) staged inputs locally if missing, then sync:
#   - repo code            -> persistent AFS home (scripts + configs only)
#   - LaRS-as-MaSTr tree   -> scratch (resized 512x384; ~100 MB)
#   - undistorted frames   -> scratch (the clips we want masks for)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

# --- 0. Build staged inputs locally (idempotent) ---
if [[ ! -f "${LOCAL_MASTR_DIR}/train_images.txt" ]]; then
  echo "[local] Staging LaRS -> ${LOCAL_MASTR_DIR} (resize ${TRAIN_SIZE}) ..."
  ( cd "${LOCAL_PROJECT_ROOT}" && python3 -m scripts.lars.prepare \
      --lars-root "${LARS_LOCAL_ROOT}" --out "${LOCAL_MASTR_DIR}" \
      --size "${TRAIN_SIZE}" --splits train,val )
else
  echo "[local] LaRS-as-MaSTr tree present: ${LOCAL_MASTR_DIR}"
fi

echo "[local] Exporting undistorted frames for predict clips ..."
export_args=()
if [[ -n "${PREDICT_CLIPS}" ]]; then
  for clip in ${PREDICT_CLIPS}; do
    export_args+=(--triplet "data/${clip}")
  done
else
  # Expand MISSION -> every triplet under data/captures/<MISSION>/ within the
  # inclusive CLIP_FROM..CLIP_TO range (lexicographic == chronological).
  mission_dir="${LOCAL_PROJECT_ROOT}/data/captures/${MISSION}"
  for f in "${mission_dir}"/fisheye_*.mp4; do
    [[ -e "$f" ]] || continue
    ts="$(basename "$f")"; ts="${ts#fisheye_}"; ts="${ts%.mp4}"
    [[ -z "${CLIP_FROM}" || ! "$ts" < "${CLIP_FROM}" ]] || continue
    [[ -z "${CLIP_TO}"   || ! "$ts" > "${CLIP_TO}" ]]   || continue
    export_args+=(--triplet "data/captures/${MISSION}/${ts}")
  done
fi
echo "  ${#export_args[@]} triplet args (every=${PREDICT_EVERY})"
( cd "${LOCAL_PROJECT_ROOT}" && python3 -m scripts.eval.export_undistorted_frames \
    "${export_args[@]}" --out "${LOCAL_PRED_FRAMES_DIR}" --every "${PREDICT_EVERY}" )

REMOTE_HOME_DIR="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
REMOTE_CODE_DIR="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}"
echo "Remote home : ${REMOTE_HOME_DIR}"
echo "Code dir    : ${REMOTE_CODE_DIR}"
echo "Scratch     : ${SEG_SCRATCH}"

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" \
  "mkdir -p '${REMOTE_CODE_DIR}' '${SEG_MASTR}' '${SEG_PRED_IN}'"

echo ""
echo "[1/3] Syncing code to home ..."
rsync -avz \
  -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
  --exclude '.git' --exclude 'data' --exclude 'labels/cache' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'images' \
  --exclude 'CAD' --exclude '*.pdf' --exclude '*.mp4' --exclude 'results' \
  --exclude 'lars' --exclude 'models' \
  "${LOCAL_PROJECT_ROOT}/" \
  "${SSH_REMOTE}:${REMOTE_CODE_DIR}/"

echo ""
echo "[2/3] Syncing LaRS-as-MaSTr tree to scratch ..."
rsync -avzP \
  -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
  "${LOCAL_MASTR_DIR}/" \
  "${SSH_REMOTE}:${SEG_MASTR}/"

echo ""
echo "[3/3] Syncing undistorted predict frames to scratch ..."
rsync -avzP \
  -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
  "${LOCAL_PRED_FRAMES_DIR}/" \
  "${SSH_REMOTE}:${SEG_PRED_IN}/"

echo ""
echo "Sync complete."
