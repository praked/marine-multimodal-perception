#!/usr/bin/env bash
# Sync code (-> persistent home) and the mission's clip range (-> scratch).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

REMOTE_HOME_DIR="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
REMOTE_CODE_DIR="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}"

echo "Remote home : ${REMOTE_HOME_DIR}"
echo "Code dir    : ${REMOTE_CODE_DIR}"
echo "Data dir    : ${REMOTE_DATA_DIR}"
echo ""

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" \
  "mkdir -p '${REMOTE_CODE_DIR}' '${REMOTE_DATA_DIR}' '${REMOTE_OUT_DIR}'"

# --- Code to persistent home (scripts + configs only; no data, no .git) ---
echo "[1/2] Syncing code to home ..."
rsync -avz --progress \
  -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
  --exclude '.git' \
  --exclude 'data' \
  --exclude 'labels/cache' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'images' \
  --exclude 'CAD' \
  --exclude '*.pdf' \
  --exclude '*.mp4' \
  --exclude 'results' \
  "${LOCAL_PROJECT_ROOT}/" \
  "${SSH_REMOTE}:${REMOTE_CODE_DIR}/"

# --- Mission clip range to scratch (fisheye + thermal + mmwave triplets) ---
# iterate_triplet needs all three files of a triplet to resolve, even though
# only the fisheye is labelled, so sync the full triplet for each clip.
# Lexicographic compare is chronological for the YYYY-MM-DD_HH-MM-SS format.
echo ""
echo "[2/2] Syncing clip range ${CLIP_FROM} .. ${CLIP_TO} to scratch ..."
src_paths=()
clips=()
for f in "${LOCAL_DATA_DIR}"/fisheye_*.mp4; do
  [[ -e "$f" ]] || continue
  ts="$(basename "$f")"; ts="${ts#fisheye_}"; ts="${ts%.mp4}"
  [[ ! "$ts" < "$CLIP_FROM" ]] || continue   # ts >= CLIP_FROM
  [[ ! "$ts" > "$CLIP_TO" ]]   || continue   # ts <= CLIP_TO
  clips+=("$ts")
  for kind in "fisheye_${ts}.mp4" "thermal_${ts}.mp4" "mmwave_${ts}.csv"; do
    [[ -e "${LOCAL_DATA_DIR}/${kind}" ]] && src_paths+=("${LOCAL_DATA_DIR}/${kind}")
  done
done

if [[ ${#clips[@]} -eq 0 ]]; then
  echo "ERROR: no clips found in ${LOCAL_DATA_DIR} for range ${CLIP_FROM}..${CLIP_TO}"
  exit 1
fi
echo "  ${#clips[@]} clips: ${clips[*]}"

rsync -avzP \
  -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
  "${src_paths[@]}" \
  "${SSH_REMOTE}:${REMOTE_DATA_DIR}/"

echo ""
echo "Sync complete."
