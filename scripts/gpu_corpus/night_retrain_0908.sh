#!/usr/bin/env bash
# One shot: the night retrain of the fusion scorer on the 2026-09-08 lake session
# (InstitutionOne gpu-node, profile institutionone). Run from the laptop repo root AFTER the
# audit on /annotate is finished. Idempotent; each step prints what it did.
#
#   bash scripts/gpu_corpus/night_retrain_0908.sh            # export audits, stage, launch
#   bash scripts/gpu_express/05_status.sh --watch             # follow
#   bash scripts/gpu_express/06_pull.sh                       # when STATUS says done (models_out, sectors, features)
#   python -m scripts.eval.night_retrain_report --before results/sectors_night \
#       --after <pulled sectors_night> --logs <boat logs> --out results/closed_loop/2026-09-08/retrain_report.md
#
# What the node does (night_queue_trackb.sh stages 4 5 8 9, scoped to the 54
# chunks of the 17:10 session): export undistorted frames -> native student
# masks + typed boxes (leak-free v8n) + instance masks -> build_features (all
# staged triplets) + build_targets (audited labels, union as unaudited) ->
# train v1a/v1b with OUT-OF-FOLD calibration, night chunks 20:4x-21:1x held out
# -> ablate on that holdout and on the standing 2026-08-26 low-light holdout ->
# re-emit sectors for every chunk with the new bundle.
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "== 1/4 audit export -> labels/audited"
(cd dashboard && node tools/export_audits.mjs) | tail -3
echo "== 2/4 express params (institutionone, Track B stages 4 5 8 9)"
(cd scripts/gpu_express && ./use_profile.sh institutionone >/dev/null 2>&1 || true)
T=$(ls /Volumes/ROS2_SSD/asvproject/captures/2026-09-08_afloat/2026-09-08_17-10-55/fisheye_*.mp4 2>/dev/null | sed 's/.*fisheye_//; s/.mp4//' | sed 's|^|2026-09-08_afloat/2026-09-08_17-10-55/|' | paste -sd" " -)
if [[ -z "$T" ]]; then   # SSD not mounted: the raw session is already staged on gpu-node; list it from the node
  T=$(ssh -o BatchMode=yes user@10.0.0.1 'ls /home/user/asvproject_scratch/asvproject_raw/captures/2026-09-08_afloat/2026-09-08_17-10-55/fisheye_*.mp4' | sed 's/.*fisheye_//; s/.mp4//' | sed 's|^|2026-09-08_afloat/2026-09-08_17-10-55/|' | paste -sd" " -)
fi
cat > scripts/gpu_express/00_params.local.sh <<PARAMS
# 2026-09-12 night retrain (written by night_retrain_0908.sh)
RUN_TRACKB=1
TRACKB_STAGES="4 5 8 9"
RUN_DART=0
RUN_YOLO=0
MISSIONS="__none__"
EXTRA_TRIPLETS="$T"
STAGE_STREAMS="fisheye thermal mmwave imu frames gps"
AUDIT_PLAN=""
SCORER_TAG="night_v2_audited"
export STUDENT=/home/user/asvproject_scratch/asvproject_models/lraspp-student/student864t_best.pth
export YDET=/home/user/asvproject_scratch/asvproject_models/yolo-audited/weights/yolov8n_noholdout_best.pt
export SCORER_TRAIN_ARGS="--oof-calibration --holdout-clips 2026-09-08_20-4,2026-09-08_20-5,2026-09-08_21-0,2026-09-08_21-1"
export ABLATE_HOLDOUTS="clips:2026-09-08_20-4,2026-09-08_20-5,2026-09-08_21-0,2026-09-08_21-1 clips:2026-08-26_19-,2026-08-26_20-"
export ABLATE_ARGS="--seeds 0 1 2"
export TARGET_LABELS="labels/audited/*.jsonl labels/union/*.jsonl"
PARAMS
echo "   $(echo "$T" | wc -w | tr -d ' ') triplets scoped"
echo "== 3/4 stage (labels + code + any raw delta; skips if the SSD is absent and the raw is already on the node)"
if [[ -d /Volumes/ROS2_SSD/asvproject/captures ]]; then bash scripts/gpu_express/03_stage.sh | tail -4
else
  rsync -az --rsh="ssh -o ConnectTimeout=10" labels/ user@10.0.0.1:/home/user/asvproject_scratch/asvproject_raw/labels/
  rsync -az --rsh="ssh -o ConnectTimeout=10" --exclude __pycache__ scripts/ configs/ user@10.0.0.1:/home/user/asvproject_project/ASVProject-ObstacleDetection/ 2>/dev/null || { rsync -az --rsh="ssh -o ConnectTimeout=10" --exclude __pycache__ scripts/ user@10.0.0.1:/home/user/asvproject_project/ASVProject-ObstacleDetection/scripts/; rsync -az --rsh="ssh -o ConnectTimeout=10" configs/ user@10.0.0.1:/home/user/asvproject_project/ASVProject-ObstacleDetection/configs/; }
  ssh -o BatchMode=yes user@10.0.0.1 'cp -n /home/user/asvproject_project/ASVProject-ObstacleDetection/scripts/gpu_corpus/night_queue_trackb.sh /home/user/asvproject_project/ASVProject-ObstacleDetection/scripts/gpu_express/express_queue.sh /home/user/asvproject_scratch/ 2>/dev/null; cp /home/user/asvproject_project/ASVProject-ObstacleDetection/scripts/gpu_corpus/night_queue_trackb.sh /home/user/asvproject_project/ASVProject-ObstacleDetection/scripts/gpu_express/express_queue.sh /home/user/asvproject_scratch/; echo "   labels + code on the node"'
fi
echo "== 4/4 launch"
bash scripts/gpu_express/04_launch.sh | tail -3
echo "poll: bash scripts/gpu_express/05_status.sh --watch"
