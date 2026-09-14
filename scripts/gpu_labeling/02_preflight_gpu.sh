#!/usr/bin/env bash
# Preflight: local data present, SSH reachable, remote layout + GPU sane.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"
CHECK_TIMEOUT=12
FAILED=0

pass() { echo "[PASS] $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*"; FAILED=1; }

echo "=== Qwen labelling preflight ==="
echo "Mission : ${MISSION}  (${CLIP_FROM} .. ${CLIP_TO})"
echo "Model   : ${MODEL}  backend=${BACKEND}"
echo "Target  : ${SSH_REMOTE} (jump ${SSH_JUMP})"
echo ""

echo "[1/4] Local prerequisites"
[[ -f "${SSH_KEY_FILE}" ]] && pass "SSH key: ${SSH_KEY_FILE}" || fail "SSH key missing: ${SSH_KEY_FILE}"
[[ -d "${LOCAL_DATA_DIR}" ]] && pass "Local data dir: ${LOCAL_DATA_DIR}" || fail "Local data dir missing: ${LOCAL_DATA_DIR}"
n_local=$(ls "${LOCAL_DATA_DIR}"/fisheye_*.mp4 2>/dev/null | wc -l | tr -d ' ')
[[ "${n_local}" -gt 0 ]] && pass "Local fisheye clips found: ${n_local}" || fail "No local fisheye_*.mp4"

echo ""
echo "[2/4] SSH reachability"
REMOTE_HOME_DIR=""
if REMOTE_HOME_DIR="$(ssh ${SSH_OPTS} -o ConnectTimeout="${CHECK_TIMEOUT}" -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"' 2>/dev/null)"; then
  pass "SSH/jump connectivity to ${GPU_NODE}"
  pass "Remote home: ${REMOTE_HOME_DIR}"
else
  fail "Cannot reach ${GPU_NODE} via ${JUMP_HOST}. Check VPN + booking + key."
fi

echo ""
echo "[3/4] Remote layout + data"
if [[ -n "${REMOTE_HOME_DIR}" ]]; then
  REMOTE_CODE_DIR="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}/${REMOTE_REPO_DIRNAME}"
  remote_out="$(ssh ${SSH_OPTS} -o ConnectTimeout="${CHECK_TIMEOUT}" -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -s -- \
      "${REMOTE_CODE_DIR}" "${REMOTE_DATA_DIR}" "${REMOTE_SCRATCH_BASE}" "${ACTIVATE_SHIM}" <<'EOS'
set -u
code="$1"; data="$2"; scratch="$3"; shim="$4"
[[ -d "$code" ]]    && echo "[PASS] code synced: $code" || echo "[WARN] code not synced yet: $code (run 01_sync)"
[[ -f "$code/scripts/eval/qwen_batch_labeler.py" ]] && echo "[PASS] labeler present" || echo "[WARN] labeler missing (run 01_sync)"
n=$(ls "$data"/fisheye_*.mp4 2>/dev/null | wc -l | tr -d ' ')
[[ "$n" -gt 0 ]]    && echo "[PASS] remote fisheye clips: $n" || echo "[WARN] no remote clips yet: $data (run 01_sync)"
[[ -d "$scratch" ]] && echo "[PASS] scratch base: $scratch" || echo "[FAIL] scratch base missing: $scratch"
[[ -w "$scratch" ]] && echo "[PASS] scratch writable" || echo "[FAIL] scratch not writable"
[[ -f "$shim" ]]    && echo "[PASS] env shim present (setup done): $shim" || echo "[WARN] env shim missing (run 03_setup_gpu)"
EOS
)" || true
  echo "${remote_out}"
  echo "${remote_out}" | grep -q "\[FAIL\]" && FAILED=1 || true
fi

echo ""
echo "[4/4] Remote GPU"
if [[ -n "${REMOTE_HOME_DIR}" ]]; then
  if gpu="$(ssh ${SSH_OPTS} -o ConnectTimeout="${CHECK_TIMEOUT}" -J "${SSH_JUMP}" "${SSH_REMOTE}" \
        bash -c 'nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null')"; then
    [[ -n "${gpu}" ]] && pass "GPU: ${gpu}" || warn "nvidia-smi returned nothing; is this a GPU node?"
    echo "      (24 GB+: 30B-A3B-AWQ ok. 16 GB: use Qwen3-VL-8B/4B-Instruct.)"
  else
    warn "Could not query GPU."
  fi
fi

echo ""
if [[ "${FAILED}" -eq 0 ]]; then
  echo "Preflight: PASS"
  exit 0
fi
echo "Preflight: FAIL: fix the [FAIL] items above."
exit 1
