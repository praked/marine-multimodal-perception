#!/usr/bin/env bash
# Runs ON the GPU node under nohup: the Track-A night queue (bundle-resolution
# pass over the whole corpus). Each stage logs to $S/logs/<stage>.log and the
# one-line state machine lives in $S/logs/STATUS.
#
#   Stage 1  finish the R2 corpus pull (idempotent rclone sync)
#   Stage 2  LRASPP student masks + YOLOv8n typed det + YOLOv8s-seg instances
#   Stage 3  GroundingDINO pseudo-labels (tuned detector_labeler recipe)
#
# Track B (raw mp4s from the SSD: full-res rerun, thermal training, fusion
# scorer) is appended by night_queue_trackb.sh when the raw data lands.
set -uo pipefail

S=/scratch0/$USER
CODE=$HOME/asvproject_project/ASVProject-ObstacleDetection
BUNDLE=$S/asvproject_corpus/bundle
OUT=$S/corpus_out
LOG=$S/logs
mkdir -p "$OUT" "$LOG"

status() { echo "$(date +%H:%M:%S) $*" >> "$LOG/STATUS"; }
# ONLY=<clip-key prefix> (env, optional) scopes stages 2-3 to matching clips
# (e.g. ONLY=2026-08-26_afloat for a fresh outing); unset = whole corpus.
ONLY="${ONLY:-}"
# EVERY=<n> (env, optional) frame stride for the GroundingDINO stage (default 2:
# lossless under the ±7-frame target dilation; 1 = every frame, 2x the time).
EVERY="${EVERY:-2}"
fail() { status "FAILED: $*"; exit 1; }

source "$S/activate_corpus.sh"
cd "$CODE"

status "queue start"

# --- Stage 1: corpus pull (idempotent; waits for/refreshes the early pull) ---
status "stage1 rclone sync"
"$S/bin/rclone" --config "$S/rclone.conf" sync r2:asvproject-clips "$BUNDLE" \
  --transfers 32 --checkers 16 --fast-list \
  >> "$LOG/rclone_pull.log" 2>&1 || fail "rclone sync"
N_CLIPS=$(ls "$BUNDLE" | wc -l)
N_FRAMES=$(find "$BUNDLE" -path '*/frames/*.jpg' | wc -l)
status "stage1 done: $N_CLIPS clips, $N_FRAMES frames"

# --- Stage 2: student seg + typed det + instance seg ---
STUDENT=$(find "$S/asvproject_models/lraspp-student" -name '*864*best*.pth' | head -1)
[[ -n "$STUDENT" ]] || STUDENT=$(find "$S/asvproject_models/lraspp-student" -name '*.pth' | head -1)
YDET=$(find "$S/asvproject_models/yolov8n-institutionone" -name '*.pt' | head -1)
YSEG=$(find "$S/asvproject_models/yolov8s-lars-seg" -name '*.pt' | head -1)
status "stage2 predict_bundle (student=$(basename "$STUDENT") det=$(basename "$YDET") seg=$(basename "$YSEG"))"
python scripts/gpu_corpus/predict_bundle.py \
  --bundle-root "$BUNDLE" --out-root "$OUT" \
  --student-pth "$STUDENT" --yolo-det "$YDET" --yolo-seg "$YSEG" \
  ${ONLY:+--only "$ONLY"} \
  --batch 8 >> "$LOG/predict_bundle.log" 2>&1 || fail "predict_bundle"
status "stage2 done: $(find "$OUT/seg" -name '*.png' | wc -l) masks, $(cat "$OUT"/done/*.json | python -c 'import json,sys; print(sum(json.loads(l)["det"] for l in sys.stdin))' 2>/dev/null || echo '?') dets"

# --- Stage 3: GroundingDINO labels ---
status "stage3 groundingdino labels"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/gpu_corpus/label_bundle_dino.py \
  --bundle-root "$BUNDLE" --out-dir "$OUT/labels_dino" \
  ${ONLY:+--only "$ONLY"} \
  --batch-size 4 --every "$EVERY" >> "$LOG/label_dino.log" 2>&1 \
  || python scripts/gpu_corpus/label_bundle_dino.py \
       --bundle-root "$BUNDLE" --out-dir "$OUT/labels_dino" \
       ${ONLY:+--only "$ONLY"} \
       --batch-size 2 --every "$EVERY" >> "$LOG/label_dino.log" 2>&1 \
  || fail "label_bundle_dino"
status "stage3 done: $(cat "$OUT"/labels_dino/*.jsonl | wc -l) label records"

status "TRACK A COMPLETE"
