#!/usr/bin/env bash
# Re-stage the thermal-student experiment on a (freshly wiped) GPU node and
# launch the LaRS-pretrain + fine-tune ladder under nohup.
#
# Bookings wipe /scratch0: this script rebuilds everything unattended in
# ~25 min: repo code + 2026-07-08 clips + quad-clip fisheye masks + student
# ONNX + LaRS-as-MaSTr tree up; venv + fisheye masks (3 clips) + warped
# thermal labels + ladder on the node. See
# docs/history/2026-07-09_thermal_tuning.md ("LaRS-gray pretraining").
#
# HOME-QUOTA RULE (2026-07-10): AFS home holds NOTHING from this workflow:
# all caches (pip/torch/XDG) are exported to scratch; ~/.cache/pip once hit
# 4.5 GB from a single torch install and tripped the quota.
#
# Usage:  bash scripts/gpu_seg/thermal_restage.sh            # full re-stage
#         bash scripts/gpu_seg/thermal_restage.sh --status   # poll the log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"
LOCAL="${DEFAULT_LOCAL_PROJECT_ROOT}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=no -o BatchMode=yes"
JUMP="${REMOTE_USER}@${JUMP_HOST}"
REMOTE="${REMOTE_USER}@${GPU_NODE}"
BASE=/scratch0/${REMOTE_USER}/asvproject_thermal
REPO=$BASE/repo

if [[ "${1:-}" == "--status" ]]; then
  ssh ${SSH_OPTS} -J "$JUMP" "$REMOTE" \
    "tail -n 15 $BASE/bootstrap.log 2>/dev/null; pgrep -f train_student >/dev/null && echo RUNNING || echo IDLE" \
    2>&1 | grep -v VBoxManage
  exit 0
fi

echo "[1/4] dirs"
ssh ${SSH_OPTS} -J "$JUMP" "$REMOTE" \
  "mkdir -p $REPO/data/captures/2026-07-08/recovered $REPO/data/seg $REPO/models/onnx $REPO/labels/qwen $BASE/runs"

echo "[2/4] code + data subset + student onnx"
rsync -az -e "ssh ${SSH_OPTS} -J $JUMP" \
  --include='scripts/***' --include='configs/***' --include='tests/***' \
  --include='requirements.txt' --exclude='*' "$LOCAL/" "$REMOTE:$REPO/"
cd "$LOCAL"
rsync -az -e "ssh ${SSH_OPTS} -J $JUMP" --relative \
  data/captures/2026-07-08/recovered/fisheye_2026-07-08_16-04-11.mp4 \
  data/captures/2026-07-08/recovered/thermal_2026-07-08_16-04-11.mp4 \
  data/captures/2026-07-08/recovered/fisheye_2026-07-08_16-18-12.mp4 \
  data/captures/2026-07-08/recovered/thermal_2026-07-08_16-18-12.mp4 \
  data/captures/2026-07-08/recovered/fisheye_2026-07-08_16-37-01_trimmed.mp4 \
  data/captures/2026-07-08/recovered/thermal_2026-07-08_16-37-01_trimmed.mp4 \
  data/captures/2026-07-08/recovered/fisheye_2026-07-08_16-44-56.mp4 \
  data/captures/2026-07-08/recovered/thermal_2026-07-08_16-44-56.mp4 \
  data/captures/2026-07-08/mmwave_2026-07-08_16-04-11.csv \
  data/captures/2026-07-08/mmwave_2026-07-08_16-18-12.csv \
  data/captures/2026-07-08/mmwave_2026-07-08_16-37-01.csv \
  data/captures/2026-07-08/mmwave_2026-07-08_16-44-56.csv \
  data/seg/2026-07-08__2026-07-08_16-37-01 \
  labels/qwen/det_2026-07-08.jsonl \
  "$REMOTE:$REPO/"
rsync -az -e "ssh ${SSH_OPTS} -J $JUMP" \
  models/onnx/student864t_864x648.onnx "$REMOTE:$REPO/models/onnx/"

echo "[3/4] lars_mastr (178M)"
rsync -az -e "ssh ${SSH_OPTS} -J $JUMP" "$LOCAL/data/lars_mastr/" "$REMOTE:$BASE/lars_mastr/"

echo "[4/4] remote bootstrap (nohup): venv -> masks -> labels -> ladder"
ssh ${SSH_OPTS} -J "$JUMP" "$REMOTE" 'bash -s' <<'EOF'
set -e
BASE=/scratch0/$USER/asvproject_thermal
cat > $BASE/bootstrap.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
BASE=/scratch0/$USER/asvproject_thermal
# All caches to scratch: NEVER home (see header).
export XDG_CACHE_HOME=$BASE/cache
export PIP_CACHE_DIR=$BASE/cache/pip
export TORCH_HOME=$BASE/cache/torch
mkdir -p $BASE/cache
cd $BASE
echo "== venv =="
python3 -m venv venv
./venv/bin/pip install --quiet --no-cache-dir --upgrade pip
./venv/bin/pip install --quiet --no-cache-dir torch torchvision \
  --index-url https://download.pytorch.org/whl/cu118
./venv/bin/pip install --quiet --no-cache-dir onnx onnxruntime \
  opencv-python-headless numpy pandas pyyaml pillow tqdm
P=$BASE/venv/bin/python
$P -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
cd $BASE/repo
echo "== fisheye masks (3 clips) =="
for clip in 2026-07-08_16-04-11 2026-07-08_16-18-12 2026-07-08_16-44-56; do
  $P -m scripts.eval.local_seg_masks --triplet data/captures/2026-07-08/$clip 2>&1 | tail -1
done
echo "== thermal labels =="
$P -m scripts.gpu_seg.thermal_labels \
  --triplet data/captures/2026-07-08/2026-07-08_16-04-11 \
  --triplet data/captures/2026-07-08/2026-07-08_16-18-12 \
  --triplet data/captures/2026-07-08/2026-07-08_16-37-01 \
  --triplet data/captures/2026-07-08/2026-07-08_16-44-56 \
  --out $BASE/dataset 2>&1 | tail -6
DS=$BASE/dataset
LARS=$BASE/lars_mastr
RUNS=$BASE/runs
TPAIRS="--pairs $DS/images/2026-07-08__2026-07-08_16-04-11:$DS/labels/2026-07-08__2026-07-08_16-04-11 --pairs $DS/images/2026-07-08__2026-07-08_16-18-12:$DS/labels/2026-07-08__2026-07-08_16-18-12 --pairs $DS/images/2026-07-08__2026-07-08_16-37-01:$DS/labels/2026-07-08__2026-07-08_16-37-01"
HOLD="--holdout-pairs $DS/images/2026-07-08__2026-07-08_16-44-56:$DS/labels/2026-07-08__2026-07-08_16-44-56"
COMMON="--size 160x120 --batch 64 --workers 4 --export-sizes 160x120 --gray"
echo "===== lars_pre ====="
$P -m scripts.gpu_seg.train_student --pairs $LARS/images:$LARS/masks $HOLD \
  $COMMON --epochs 40 --seed 0 \
  --aug-polarity 0.5 --aug-vshift 12 --aug-contrast 0.3 --out $RUNS/lars_pre
for seed in 0 2 3; do
  echo "===== ft_lars_s${seed} ====="
  $P -m scripts.gpu_seg.train_student $TPAIRS $HOLD $COMMON --epochs 60 --seed $seed \
    --aug-polarity 0.5 --aug-vshift 12 \
    --init-weights $RUNS/lars_pre/student_best.pth --out $RUNS/ft_lars_s${seed}
done
echo "LADDER DONE"
INNER
chmod +x $BASE/bootstrap.sh
nohup $BASE/bootstrap.sh > $BASE/bootstrap.log 2>&1 &
echo "bootstrap pid: $!"
EOF
echo "RESTAGE LAUNCHED: poll with: bash $0 --status"
