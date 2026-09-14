#!/usr/bin/env bash
# Runs ON the node (from express_queue.sh). On-domain YOLO fine-tune on the
# exported dataset staged at $S/yolo_dataset (data.yaml), in the cu118 YOLO venv.
set -uo pipefail
S=${S:-/scratch0/$USER}; CODE=${CODE:-$HOME/asvproject_project/ASVProject-ObstacleDetection}; LOG=$S/logs; YS=$S/asvproject_yolo
status() { echo "$(date +%H:%M:%S) $*" >> $LOG/STATUS_yolo; }
[[ -f $S/yolo_dataset/data.yaml ]] || { status "FAILED: no $S/yolo_dataset/data.yaml (03_stage with RUN_YOLO=1 exports it)"; exit 1; }
source $YS/activate_yolo.sh 2>/dev/null || source $YS/yolo_venv/bin/activate
cd $CODE
status "train ${YOLO_BASE:-yolov8n.pt} epochs ${YOLO_EPOCHS:-100} imgsz ${YOLO_IMGSZ:-864}"
python scripts/gpu_finetune/train_yolo.py --data $S/yolo_dataset/data.yaml --model "${YOLO_BASE:-yolov8n.pt}" \
  --epochs "${YOLO_EPOCHS:-100}" --imgsz "${YOLO_IMGSZ:-864}" --batch "${YOLO_BATCH:-16}" \
  --project $YS/yolo_runs --name ondomain >> $LOG/yolo_train.log 2>&1 || { status "FAILED: train (see yolo_train.log)"; exit 1; }
status "YOLO DONE: $(ls $YS/yolo_runs/ondomain/weights/ 2>/dev/null | tr '\n' ' ')"
