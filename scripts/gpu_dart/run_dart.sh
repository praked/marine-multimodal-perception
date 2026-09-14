#!/usr/bin/env bash
# LOCAL one-stop driver for the DART (SAM3) labeller on a GPU node.
# Knobs: scripts/gpu_dart/dart_params.sh (+ dart_params.local.sh); node/SSH from
# scripts/gpu_seg/00_run_params.sh (+ .local.sh). Every step is idempotent.
#
#   bash scripts/gpu_dart/run_dart.sh --prepare   # code sync + env + weights (gated: see below)
#   bash scripts/gpu_dart/run_dart.sh --stage     # fisheye videos -> node, export frames (needs the SSD)
#   bash scripts/gpu_dart/run_dart.sh --launch    # nohup the queue (smoke -> label all exported clips)
#   bash scripts/gpu_dart/run_dart.sh --status    # poll
#   bash scripts/gpu_dart/run_dart.sh --pull      # labels -> SSD archive + labels/dart (needs the SSD)
#   bash scripts/gpu_dart/run_dart.sh --all       # prepare + stage + launch
#
# macOS rsync (2.6.9) gotchas baked in: --rsh= (not -e) and --files-from (not
# --include chains); remote login shell is tcsh, so everything remote is piped
# through `bash -s` with `env VAR=x`.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/dart_params.sh"
LOCAL="${LOCAL_PROJECT_ROOT}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=25"
JUMPOPT="${JUMP_HOST:+-J ${REMOTE_USER}@${JUMP_HOST}}"; REMOTE="${REMOTE_USER}@${GPU_NODE}"
RSH="ssh ${SSH_OPTS} ${JUMPOPT}"
REMOTE_CODE_REL="${REMOTE_CODE_REL:-asvproject_project/ASVProject-ObstacleDetection}"
S="${DART_S}"
# env forwarded to every remote script
RENV="DART_S=${S} CODE=\$HOME/${REMOTE_CODE_REL} DART_REPO_URL=${DART_REPO_URL} DART_COMMIT=${DART_COMMIT} DART_PYTHON_VERSION=${DART_PYTHON_VERSION} DART_TORCH_INDEX=${DART_TORCH_INDEX} DART_TORCH_SPEC='${DART_TORCH_SPEC}' SAM3_REPO=${SAM3_REPO} SAM3_FILE=${SAM3_FILE} SAM3_MD5=${SAM3_MD5} DART_EVERY=${DART_EVERY} DART_CONFIG=${DART_CONFIG} DART_COMPILE=${DART_COMPILE} DART_SMOKE_CLIP=${DART_SMOKE_CLIP} DART_OUT_NAME=${DART_OUT_NAME}"
remote() { $RSH "$REMOTE" "env $RENV bash -s" 2>&1 | grep -v "VBoxManage\|VirtualBox"; }
rs() { local a; for a in 1 2 3 4 5 6; do rsync -az --partial --rsh="ssh ${SSH_OPTS} ${JUMPOPT}" "$@" && return 0; echo "  rsync attempt $a failed; 20 s" >&2; sleep 20; done; return 1; }
ts() { date +%H:%M:%S; }
ssd() { [[ -d "${DART_ARCHIVE_ROOT}/captures" ]] || { echo "SSD not mounted at ${DART_ARCHIVE_ROOT}" >&2; exit 1; }; }

sync_code() {
  echo "[$(ts)] code -> node home"
  $RSH "$REMOTE" "mkdir -p ${REMOTE_CODE_REL}" 2>&1 | grep -v "VBoxManage\|VirtualBox" || true
  rsync -az --rsh="ssh ${SSH_OPTS} ${JUMPOPT}" --exclude='__pycache__' "$LOCAL/scripts" "$LOCAL/configs" "$REMOTE:${REMOTE_CODE_REL}/"
}

prepare() {
  sync_code
  echo "[$(ts)] env on node (uv python, venv, DART)"
  remote < "${SCRIPT_DIR}/remote_setup.sh"
  # weights: gated repo -> laptop download + rsync when logged in here, else instructions
  if $RSH "$REMOTE" "test -s ${S}/asvproject_models/dart/${SAM3_FILE}" 2>/dev/null; then
    echo "[$(ts)] checkpoint already on node"
  elif hf auth whoami >/dev/null 2>&1; then
    echo "[$(ts)] downloading ${SAM3_REPO}/${SAM3_FILE} here (laptop HF login), then rsync"
    P=$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('${SAM3_REPO}','${SAM3_FILE}'))")
    rs --copy-links "$P" "$REMOTE:${S}/asvproject_models/dart/${SAM3_FILE}"
  else
    cat <<MSG
[$(ts)] checkpoint missing and this laptop is not logged into HF. Either:
  hf auth login            # then re-run --prepare (download here, rsync up; no token on the node)
or on the node, with your token (HF_HOME on scratch dies with the booking):
  ssh -i ${SSH_KEY_FILE} ${JUMPOPT} ${REMOTE} 'setenv HF_HOME ${S}/.cache/huggingface; ${S}/dart_venv/bin/hf download ${SAM3_REPO} ${SAM3_FILE} --local-dir ${S}/asvproject_models/dart --token <TOKEN>'
MSG
    return 2
  fi
}

stage() {
  ssd
  local C="${DART_ARCHIVE_ROOT}/captures" LIST TRIP
  LIST=$(mktemp); TRIP=$(mktemp)
  # triplet list: every fisheye chunk of DART_MISSIONS + extras + audit-plan clips
  for m in ${DART_MISSIONS}; do
    ( cd "$C" && find "$m" -maxdepth 3 -name 'fisheye_*.mp4' -not -path '*/recovered/*' -not -path '*/_*' ) \
      | sed 's|/fisheye_| |; s|\.mp4$||' | awk '{print "data/captures/"$1"/"$2}'
  done > "$TRIP"
  for t in ${DART_EXTRA_TRIPLETS}; do echo "$t"; done >> "$TRIP"
  if [[ -n "${DART_PLAN_CSV}" && -f "${DART_PLAN_CSV}" ]]; then
    tail -n +2 "${DART_PLAN_CSV}" | cut -d, -f2 | sort -u | python3 -c '
import sys, pathlib
C = pathlib.Path(sys.argv[1])
for clip in sys.stdin.read().split():
    scene, _, ts = clip.partition("__")
    # nested missions were flattened with "_" -> recover the on-disk path by probing
    parts = scene.split("_")
    for k in range(len(parts), 0, -1):
        cand = C / "_".join(parts[:k]) / "/".join(parts[k:]) if k < len(parts) else C / scene
        if (cand / f"fisheye_{ts}.mp4").exists() or (cand / "recovered" / f"fisheye_{ts}_trimmed.mp4").exists():
            print("data/captures/" + str(cand.relative_to(C)) + "/" + ts); break
    else:
        print(f"WARN plan clip not found on SSD: {clip}", file=sys.stderr)
' "$C" >> "$TRIP"
  fi
  sort -u "$TRIP" -o "$TRIP"
  # file list to stage: fisheye mp4 (+ recovered) + frames csv per triplet
  ( cd "$C" && while read -r t; do rel=${t#data/captures/}; d=$(dirname "$rel"); ts=$(basename "$rel")
      ls "$d/fisheye_$ts.mp4" "$d/frames_$ts.csv" "$d/recovered/fisheye_${ts}"*.mp4 2>/dev/null; done < "$TRIP" ) > "$LIST"
  echo "[$(ts)] staging $(wc -l < "$TRIP" | tr -d ' ') triplets / $(wc -l < "$LIST" | tr -d ' ') files ($(cd "$C" && cat "$LIST" | xargs du -ch 2>/dev/null | tail -1 | cut -f1))"
  # rsync --files-from does not create nested parents -> pre-create them
  $RSH "$REMOTE" "bash -c 'mkdir -p ${S}/asvproject_raw/captures; while read -r p; do mkdir -p ${S}/asvproject_raw/captures/\$(dirname \"\$p\"); done'" < "$LIST" 2>&1 | grep -v "VBoxManage\|VirtualBox" || true
  rs --files-from="$LIST" "$C/" "$REMOTE:${S}/asvproject_raw/captures/"
  rs "$TRIP" "$REMOTE:${S}/dart_triplets.txt"
  sync_code
  echo "[$(ts)] exporting frames on node"
  remote < "${SCRIPT_DIR}/remote_export.sh"
  rm -f "$LIST" "$TRIP"
}

launch() {
  sync_code
  rs "${SCRIPT_DIR}/remote_queue.sh" "$REMOTE:${S}/remote_queue.sh"
  $RSH "$REMOTE" "env $RENV bash -s" <<'EOS' 2>&1 | grep -v "VBoxManage\|VirtualBox"
S=${DART_S}
if pgrep -f "$S/remote_queue.sh" >/dev/null; then echo "queue already running"; else
  nohup env DART_S=$DART_S SAM3_FILE=$SAM3_FILE DART_OUT_NAME=$DART_OUT_NAME DART_CONFIG=$DART_CONFIG DART_EVERY=$DART_EVERY DART_COMPILE=$DART_COMPILE DART_SMOKE_CLIP=$DART_SMOKE_CLIP \
    bash $S/remote_queue.sh >> $S/logs/remote_queue.log 2>&1 &
  echo "launched pid $!"; fi
EOS
  echo "poll with: bash scripts/gpu_dart/run_dart.sh --status"
}

status() {
  $RSH "$REMOTE" "env $RENV bash -s" <<'EOS' 2>&1 | grep -v "VBoxManage\|VirtualBox"
S=${DART_S}; OUT=$S/corpus_out/${DART_OUT_NAME}
date +%H:%M:%S; echo "=== STATUS_dart ==="; tail -n 8 $S/logs/STATUS_dart 2>/dev/null
echo "records: $(cat $OUT/*.jsonl 2>/dev/null | wc -l)  files: $(ls $OUT 2>/dev/null | wc -l)  frames exported: $(find $S/native_frames -name '*.jpg' 2>/dev/null | wc -l)"
grep -h "rec/s\|labelled" $S/logs/dart_label.log 2>/dev/null | tail -2
grep -h "Traceback\|Error:" $S/logs/dart_label.log 2>/dev/null | tail -1
pgrep -fa "remote_queue|label_bundle_dart" | grep -v pgrep | cut -c1-80 || echo "(queue idle)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
EOS
}

pull() {
  ssd
  local ARCH="${DART_ARCHIVE_ROOT}/${GPU_NODE%%.*}_$(date +%Y-%m-%d)/${DART_OUT_NAME}"
  mkdir -p "$ARCH" "${DART_LOCAL_LABELS}"
  echo "[$(ts)] pull ${DART_OUT_NAME} -> $ARCH (+ logs, config)"
  rs "$REMOTE:${S}/corpus_out/${DART_OUT_NAME}/" "$ARCH/"
  rs "$REMOTE:${S}/logs/" "$(dirname "$ARCH")/logs/"
  cp "$LOCAL/${DART_CONFIG}" "$ARCH/"
  ssd
  echo "[$(ts)] place into ${DART_LOCAL_LABELS} (never labels/qwen)"
  rsync -a "$ARCH/" "${DART_LOCAL_LABELS}/" --exclude='*.yaml'
  echo "  $(ls "$ARCH"/*.jsonl | wc -l | tr -d ' ') files, $(cat "$ARCH"/*.jsonl | wc -l | tr -d ' ') records"
  echo "[$(ts)] PULL-DART DONE"
}

case "${1:-}" in
  --prepare) prepare;;
  --stage) stage;;
  --launch) launch;;
  --status) status;;
  --pull) pull;;
  --all) prepare && stage && launch;;
  *) sed -n 2,20p "$0"; exit 2;;
esac
