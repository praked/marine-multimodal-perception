#!/usr/bin/env bash
# Launch the Qwen batch labeler on the GPU node under nohup (survives logout).
# Re-runnable: the labeler resumes from whatever is already in the output JSONL.
#
# Robust quoting: we generate a launch script locally, write it to the node, and
# nohup that file: no multi-layer quote escaping through ssh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

REMOTE_HOME_DIR="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
REMOTE_CODE_DIR="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}"
REMOTE_LAUNCH="${REMOTE_OUT_DIR}/launch_labeling.sh"

MAX_FRAMES_FLAG=""
[[ -n "${MAX_FRAMES}" ]] && MAX_FRAMES_FLAG="--max-frames ${MAX_FRAMES}"
PIXFLAGS=""
[[ -n "${MIN_PIXELS:-}" ]] && PIXFLAGS+=" --min-pixels ${MIN_PIXELS}"
[[ -n "${MAX_PIXELS:-}" ]] && PIXFLAGS+=" --max-pixels ${MAX_PIXELS}"

# Generate the remote launch script locally (all params already expanded).
LAUNCH_CONTENT="$(cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
source '${ACTIVATE_SHIM}'
cd '${REMOTE_CODE_DIR}'
export PYTHONUNBUFFERED=1
exec python -m scripts.eval.qwen_batch_labeler \\
  --captures-dir '${REMOTE_DATA_DIR}' \\
  --from '${CLIP_FROM}' --to '${CLIP_TO}' \\
  --every ${EVERY} \\
  --backend ${BACKEND} \\
  --model '${MODEL}' \\
  --out '${REMOTE_OUT_JSONL}' \\
  --cache-dir '${REMOTE_CACHE_DIR}' \\
  --batch-size ${BATCH_SIZE} \\
  --max-tokens ${MAX_TOKENS} \\
  --gpu-mem-util ${GPU_MEM_UTIL} \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \\
  ${MAX_FRAMES_FLAG}${PIXFLAGS}
EOF
)"

echo "Writing launch script + nohup-ing it -> ${REMOTE_LAUNCH}"
echo "  out : ${REMOTE_OUT_JSONL}"
echo "  log : ${REMOTE_RUN_LOG}"
# GPU nodes default to tcsh; run everything under `bash -s`. The outer heredoc
# (unquoted) expands local vars; the inner (quoted) writes LAUNCH_CONTENT
# verbatim. \$! is escaped so it evaluates on the remote bash.
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -s <<EOF
set -e
mkdir -p '${REMOTE_OUT_DIR}'
cat > '${REMOTE_LAUNCH}' <<'LAUNCH_EOF'
${LAUNCH_CONTENT}
LAUNCH_EOF
chmod +x '${REMOTE_LAUNCH}'
nohup bash '${REMOTE_LAUNCH}' >> '${REMOTE_RUN_LOG}' 2>&1 < /dev/null &
echo "started PID \$!"
EOF

echo ""
echo "Monitor:"
echo "  ssh -i ${SSH_KEY_FILE} -J ${SSH_JUMP} ${SSH_REMOTE} \"tail -f '${REMOTE_RUN_LOG}'\""
echo "Count labels so far:"
echo "  ssh -i ${SSH_KEY_FILE} -J ${SSH_JUMP} ${SSH_REMOTE} \"wc -l '${REMOTE_OUT_JSONL}'\""
