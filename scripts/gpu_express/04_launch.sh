#!/usr/bin/env bash
# 04: launch the selected queues on the node under nohup (survives laptop/SSD/ssh).
#     Track B (TRACKB_STAGES) -> DART -> YOLO, sequenced by express_queue.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_params.sh"
sync_code
# node-side scripts to scratch (the queues run from scratch copies, as always)
rs "${SCRIPT_DIR}/express_queue.sh" "${SCRIPT_DIR}/yolo_queue.sh" "${SCRIPT_DIR}/../gpu_dart/remote_queue.sh" "${REMOTE}:${S}/"
if [[ "${RUN_YOLO}" == 1 ]]; then
  need_ssd; echo "[$(ts)] YOLO dataset export -> node"
  YD=$(mktemp -d); ( cd "${LOCAL}" && "${LOCAL}/.venv/bin/python" -m scripts.eval.export_yolo --labels "${YOLO_LABELS}" --out "$YD" ${YOLO_PSEUDO:+--pseudo "$YOLO_PSEUDO" --pseudo-count "$YOLO_PSEUDO_COUNT"} )
  rs "$YD/" "${REMOTE}:${S}/yolo_dataset/"; rm -rf "$YD"
fi
# DART knobs come from gpu_dart/dart_params.sh
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../gpu_dart/dart_params.sh"
ENV="RUN_TRACKB=${RUN_TRACKB} TRACKB_STAGES='${TRACKB_STAGES}' THERMAL_SEEDS='${THERMAL_SEEDS}' THERMAL_EPOCHS=${THERMAL_EPOCHS} SCORER_TAG=${SCORER_TAG} RUN_DART=${RUN_DART} RUN_YOLO=${RUN_YOLO} YOLO_BASE=${YOLO_BASE} YOLO_EPOCHS=${YOLO_EPOCHS} YOLO_IMGSZ=${YOLO_IMGSZ} YOLO_BATCH=${YOLO_BATCH} DART_S=${S} SAM3_FILE=${SAM3_FILE} DART_OUT_NAME=${DART_OUT_NAME} DART_CONFIG=${DART_CONFIG} DART_EVERY=${DART_EVERY} DART_COMPILE=${DART_COMPILE} DART_SMOKE_CLIP=${DART_SMOKE_CLIP} STUDENT='${STUDENT:-}' YDET='${YDET:-}' SCORER_TRAIN_ARGS='${SCORER_TRAIN_ARGS:-}' ABLATE_HOLDOUTS='${ABLATE_HOLDOUTS:-}' ABLATE_ARGS='${ABLATE_ARGS:-}' TARGET_LABELS='${TARGET_LABELS:-}'"
$RSH "$REMOTE" "env ${RENV_BASE} $ENV bash -s" <<'EOS' 2>&1 | grep -v "VBoxManage\|VirtualBox"
S=${S:-/scratch0/$USER}
if pgrep -f "$S/express_queue.sh" >/dev/null; then echo "express queue already running"; else
  nohup env S="$S" CODE="$CODE" RUN_TRACKB="$RUN_TRACKB" TRACKB_STAGES="$TRACKB_STAGES" THERMAL_SEEDS="$THERMAL_SEEDS" THERMAL_EPOCHS="$THERMAL_EPOCHS" SCORER_TAG="$SCORER_TAG" \
    RUN_DART="$RUN_DART" RUN_YOLO="$RUN_YOLO" YOLO_BASE="$YOLO_BASE" YOLO_EPOCHS="$YOLO_EPOCHS" YOLO_IMGSZ="$YOLO_IMGSZ" YOLO_BATCH="$YOLO_BATCH" \
    DART_S="$DART_S" SAM3_FILE="$SAM3_FILE" DART_OUT_NAME="$DART_OUT_NAME" DART_CONFIG="$DART_CONFIG" DART_EVERY="$DART_EVERY" DART_COMPILE="$DART_COMPILE" DART_SMOKE_CLIP="$DART_SMOKE_CLIP" \
    STUDENT="$STUDENT" YDET="$YDET" SCORER_TRAIN_ARGS="$SCORER_TRAIN_ARGS" ABLATE_HOLDOUTS="$ABLATE_HOLDOUTS" ABLATE_ARGS="$ABLATE_ARGS" TARGET_LABELS="$TARGET_LABELS" \
    bash "$S/express_queue.sh" >> "$S/logs/express_queue.log" 2>&1 &
  echo "launched express queue pid $!"
fi
EOS
echo "poll with: bash scripts/gpu_express/05_status.sh"
