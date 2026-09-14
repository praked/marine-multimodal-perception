#!/usr/bin/env bash
# RESUME-AFTER-TRAVEL driver. Once the YOLOv8-seg training has finished on the
# GPU, this pulls both detector weights and runs FULL inference over every
# uploaded InstitutionOne clip, populating the dashboard overlays:
#   - typed boxes  (YOLO det)   -> data/det/
#   - instance masks (YOLO-seg) -> data/det_seg/
#
# Safe to re-run. Reuses the undistorted frames already on scratch
# ($SEG/pred_frames): no re-upload. Prints per-class mAP for both models.
#
#   bash scripts/gpu_finetune/finish_annotations.sh
set -uo pipefail

SSH="-i ${HOME}/.ssh/cluster_key -o IdentitiesOnly=yes -o IdentityAgent=none"
JUMP="user@gpu-node.cluster.example.org"
RMT="user@gpu-node.cluster.example.org"
SEG=/scratch/user/asvproject_seg
YS=/scratch/user/asvproject_yolo
HOME_CODE=/cs/student/ug/2024/user/asvproject_project/ASVProject-ObstacleDetection
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"

echo "[0] push latest predictor scripts to node"
rsync -az -e "ssh $SSH -J $JUMP" scripts/gpu_finetune/yolo_predict_frames.py \
  scripts/gpu_finetune/yolo_predict_seg_frames.py \
  "$RMT:$HOME_CODE/scripts/gpu_finetune/"

echo "[1] check training state"
ssh $SSH -J "$JUMP" "$RMT" 'bash -s' <<INNER 2>&1 | grep -v -i vbox
det=$YS/yolo_runs/lars_det/weights/best.pt
seg=$YS/yolo_runs/lars_seg/weights/best.pt
echo "det best.pt: \$(ls -la \$det 2>/dev/null | awk '{print \$5}' || echo MISSING)"
echo "seg best.pt: \$(ls -la \$seg 2>/dev/null | awk '{print \$5}' || echo MISSING)"
echo "seg still training: \$(pgrep -f 'train_yolo.py.*lars_seg' | head -1 || echo no)"
INNER

echo "[2] run FULL det + seg inference over all InstitutionOne clips"
ssh $SSH -J "$JUMP" "$RMT" 'bash -s' <<INNER 2>&1 | grep -v -i vbox | tail -40
source $YS/activate_yolo.sh
DET=$YS/yolo_runs/lars_det/weights/best.pt
SEG_W=$YS/yolo_runs/lars_seg/weights/best.pt
if [ -f "\$DET" ]; then
  echo "=== det inference ==="
  python $HOME_CODE/scripts/gpu_finetune/yolo_predict_frames.py \
    --model "\$DET" --frames-root $SEG/pred_frames --out $YS/det \
    --conf 0.25 --imgsz 960 --batch 16 2>&1 | tail -16
fi
if [ -f "\$SEG_W" ]; then
  echo "=== seg inference ==="
  python $HOME_CODE/scripts/gpu_finetune/yolo_predict_seg_frames.py \
    --model "\$SEG_W" --frames-root $SEG/pred_frames --out $YS/det_seg \
    --conf 0.25 --imgsz 960 --batch 16 2>&1 | tail -16
else
  echo "seg best.pt missing; is training done? (skip seg)"
fi
INNER

echo "[3] pull detections + masks + weights"
mkdir -p data/det data/det_seg models
rsync -az -e "ssh $SSH -J $JUMP" "$RMT:$YS/det/"     data/det/
rsync -az -e "ssh $SSH -J $JUMP" "$RMT:$YS/det_seg/" data/det_seg/
rsync -az -e "ssh $SSH -J $JUMP" "$RMT:$YS/yolo_runs/lars_det/weights/best.pt" models/yolo_lars_det_best.pt 2>/dev/null || true
rsync -az -e "ssh $SSH -J $JUMP" "$RMT:$YS/yolo_runs/lars_seg/weights/best.pt" models/yolo_lars_seg_best.pt 2>/dev/null || true

echo "[4] per-class summary"
for run in lars_det lars_seg; do
  echo "=== $run results.csv (last row) ==="
  ssh $SSH -J "$JUMP" "$RMT" "bash -lc 'tail -n 1 $YS/yolo_runs/$run/results.csv 2>/dev/null'" 2>/dev/null | grep -v -i vbox
done
echo "det files: $(ls data/det/*.jsonl 2>/dev/null | wc -l)  seg files: $(ls data/det_seg/*.jsonl 2>/dev/null | wc -l)"
echo "Done. Open the dashboard and toggle 'Instance masks' / 'Typed detections'."
