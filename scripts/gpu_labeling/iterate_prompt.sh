#!/usr/bin/env bash
# Prompt-iteration loop: edit configs/qwen_label_prompt.yaml, then run this to
# smoke-test that prompt on the fixed frame set (labels/qwen/smoke_set.txt) and
# get a contact sheet to eyeball. Repeat with different prompts to compare.
#
#   1. edit  configs/qwen_label_prompt.yaml
#   2. bash scripts/gpu_labeling/iterate_prompt.sh [tag]
#   3. open results/qwen_audit/prompt_<tag>/contact_01.png
#
# Each run uses its own output + cache (keyed by tag AND by the prompt
# fingerprint), so prompts never mask each other via the cache.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

TAG="${1:-$(date +%Y%m%d-%H%M%S)}"
SMOKE_SET_REL="labels/qwen/smoke_set.txt"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new"
RSH="ssh ${SSH_OPTS} -J ${SSH_JUMP}"

REMOTE_HOME_DIR="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
REMOTE_CODE_DIR="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}"
REMOTE_SMOKE_OUT="${REMOTE_OUT_DIR}/smoke_${TAG}.jsonl"
REMOTE_SMOKE_CACHE="${REMOTE_OUT_DIR}/cache_smoke_${TAG}"
REMOTE_SMOKE_LOG="${REMOTE_OUT_DIR}/smoke_${TAG}.log"
LOCAL_OUT_DIR="${LOCAL_PROJECT_ROOT}/results/qwen_audit/prompt_${TAG}"

echo "=== prompt smoke test: tag=${TAG} model=${MODEL} ==="

# 1. Push the edited prompt + labeler + frame set (data is already on scratch).
echo "[1/4] syncing prompt + labeler + smoke set to node ..."
$RSH "${SSH_REMOTE}" "mkdir -p '${REMOTE_CODE_DIR}/configs' '${REMOTE_CODE_DIR}/labels/qwen' '${REMOTE_OUT_DIR}'"
rsync -az -e "${RSH}" "${LOCAL_PROJECT_ROOT}/configs/qwen_label_prompt.yaml" \
  "${SSH_REMOTE}:${REMOTE_CODE_DIR}/configs/qwen_label_prompt.yaml"
rsync -az -e "${RSH}" "${LOCAL_PROJECT_ROOT}/scripts/eval/qwen_batch_labeler.py" \
  "${SSH_REMOTE}:${REMOTE_CODE_DIR}/scripts/eval/qwen_batch_labeler.py"
rsync -az -e "${RSH}" "${LOCAL_PROJECT_ROOT}/${SMOKE_SET_REL}" \
  "${SSH_REMOTE}:${REMOTE_CODE_DIR}/${SMOKE_SET_REL}"

# 2. Launch the smoke run under nohup.
PIXFLAGS=""
[[ -n "${MIN_PIXELS:-}" ]] && PIXFLAGS+=" --min-pixels ${MIN_PIXELS}"
[[ -n "${MAX_PIXELS:-}" ]] && PIXFLAGS+=" --max-pixels ${MAX_PIXELS}"

echo "[2/4] launching smoke run on ${GPU_NODE} ... ${PIXFLAGS:+(res${PIXFLAGS})}"
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -s <<EOF
set -e
cat > '${REMOTE_OUT_DIR}/launch_smoke_${TAG}.sh' <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
source '${ACTIVATE_SHIM}'
cd '${REMOTE_CODE_DIR}'
export PYTHONUNBUFFERED=1
exec python -m scripts.eval.qwen_batch_labeler \
  --captures-dir '${REMOTE_DATA_DIR}' \
  --frames-file '${REMOTE_CODE_DIR}/${SMOKE_SET_REL}' \
  --backend ${BACKEND} --model '${MODEL}' \
  --out '${REMOTE_SMOKE_OUT}' --cache-dir '${REMOTE_SMOKE_CACHE}' \
  --batch-size ${BATCH_SIZE} --max-tokens ${MAX_TOKENS} \
  --gpu-mem-util ${GPU_MEM_UTIL} --max-model-len ${MAX_MODEL_LEN} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}${PIXFLAGS} --no-resume
LAUNCH
chmod +x '${REMOTE_OUT_DIR}/launch_smoke_${TAG}.sh'
: > '${REMOTE_SMOKE_LOG}'
nohup bash '${REMOTE_OUT_DIR}/launch_smoke_${TAG}.sh' >> '${REMOTE_SMOKE_LOG}' 2>&1 < /dev/null &
echo "started PID \$!"
EOF

# 3. Wait for completion (model load + ~36 frames).
echo "[3/4] waiting for smoke run to finish (tail: ${REMOTE_SMOKE_LOG}) ..."
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -c \
  "'for i in \$(seq 1 120); do grep -qE \"^done:|Traceback|Engine core initialization failed\" \"${REMOTE_SMOKE_LOG}\" 2>/dev/null && break; sleep 5; done; grep -E \"^done:|Traceback|Error\" \"${REMOTE_SMOKE_LOG}\" | head'" \
  2>&1 | grep -vE "VBoxManage|NS_ERROR|VirtualBox|tinderbox|Document is empty|Location:|Details:" || true

# 4. Pull results and render a contact sheet locally.
echo "[4/4] pulling results + rendering contact sheet ..."
mkdir -p "${LOCAL_OUT_DIR}"
rsync -az -e "${RSH}" "${SSH_REMOTE}:${REMOTE_SMOKE_OUT}" "${LOCAL_OUT_DIR}/smoke_${TAG}.jsonl"
rsync -az -e "${RSH}" "${SSH_REMOTE}:${REMOTE_SMOKE_LOG}" "${LOCAL_OUT_DIR}/smoke_${TAG}.log" || true

cp "${LOCAL_PROJECT_ROOT}/configs/qwen_label_prompt.yaml" "${LOCAL_OUT_DIR}/prompt.yaml"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
( cd "${LOCAL_PROJECT_ROOT}" && "${PYTHON}" -m scripts.eval.render_label_overlays \
    --labels "${LOCAL_OUT_DIR}/smoke_${TAG}.jsonl" \
    --captures-dir "${LOCAL_DATA_DIR}" \
    --out-dir "${LOCAL_OUT_DIR}" )

echo ""
echo "Done. Review:  ${LOCAL_OUT_DIR}/contact_01.png"
command -v open >/dev/null && open "${LOCAL_OUT_DIR}/contact_01.png" || true
