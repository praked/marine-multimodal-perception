#!/usr/bin/env bash
# Runs ON the node under nohup (from 04_launch.sh). Sequences the selected
# queues so they never fight for the GPU:  Track B (selected stages) -> DART -> YOLO.
# Each sub-queue keeps its own STATUS file; this one summarises in STATUS_express.
set -uo pipefail
S=${S:-/scratch0/$USER}; CODE=${CODE:-$HOME/asvproject_project/ASVProject-ObstacleDetection}; LOG=$S/logs; mkdir -p $LOG; export S CODE
status() { echo "$(date +%H:%M:%S) $*" >> $LOG/STATUS_express; }
status "express start: TRACKB=${RUN_TRACKB:-0} [${TRACKB_STAGES:-}] DART=${RUN_DART:-0} YOLO=${RUN_YOLO:-0}"

if [[ "${RUN_TRACKB:-0}" == 1 ]]; then
  status "track B: stages ${TRACKB_STAGES:-4 5 7 8 9 6}"
  cp "$CODE/scripts/gpu_corpus/night_queue_trackb.sh" "$S/night_queue_trackb.sh"
  TRACKB_STAGES="${TRACKB_STAGES:-}" THERMAL_SEEDS="${THERMAL_SEEDS:-0 1 2}" THERMAL_EPOCHS="${THERMAL_EPOCHS:-60}" SCORER_TAG="${SCORER_TAG:-night_v1}" \
    bash "$S/night_queue_trackb.sh" >> "$LOG/night_queue_trackb.log" 2>&1 || status "WARN track B queue exited non-zero"
  status "track B: $(tail -1 $LOG/STATUS 2>/dev/null)"
fi
if [[ "${RUN_DART:-0}" == 1 ]]; then
  status "DART labeller"
  bash "$S/remote_queue.sh" >> "$LOG/remote_queue.log" 2>&1 || status "WARN DART queue exited non-zero"
  status "DART: $(tail -1 $LOG/STATUS_dart 2>/dev/null)"
fi
if [[ "${RUN_YOLO:-0}" == 1 ]]; then
  status "YOLO fine-tune"
  bash "$S/yolo_queue.sh" >> "$LOG/yolo_queue.log" 2>&1 || status "WARN YOLO queue exited non-zero"
  status "YOLO: $(tail -1 $LOG/STATUS_yolo 2>/dev/null)"
fi
status "EXPRESS COMPLETE"
