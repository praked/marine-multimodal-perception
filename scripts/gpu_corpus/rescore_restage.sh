#!/usr/bin/env bash
# One-command restage for the rescoring round on a FRESH booking (scratch
# wiped): finish the motion+typed feature rebuild, run the 2x2 ablation
# (base / +motion-context / +typed-evidence / +both), pick the winner by
# eval AP, and re-emit scored sectors for EVERY clip with the tracker on.
#
# Set the node first (gitignored scripts/gpu_seg/00_run_params.local.sh:
# GPU_NODE="<node>.cluster.example.org"), then:
#
#   bash scripts/gpu_corpus/rescore_restage.sh            # stage + launch
#   bash scripts/gpu_corpus/rescore_restage.sh --status   # poll
#
# Needs the ROS2_SSD mounted (raw captures + the gears_2026-08-22 staging
# pull are the sync sources). CPU-only round: the venv skips torch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../gpu_seg/00_run_params.sh"
LOCAL="${LOCAL_PROJECT_ROOT}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
JUMP="${REMOTE_USER}@${JUMP_HOST}"
REMOTE="${REMOTE_USER}@${GPU_NODE}"
RSH="ssh ${SSH_OPTS} -J ${JUMP}"
G=/Volumes/ROS2_SSD/asvproject/gears_2026-08-22
RAW=/scratch0/${REMOTE_USER}/asvproject_raw

if [[ "${1:-}" == "--status" ]]; then
  $RSH "$REMOTE" 'bash -s' <<'EOF' 2>&1 | grep -v VBoxManage
S=/scratch0/$USER
tail -n 12 $S/logs/rescore.log 2>/dev/null
pgrep -f rescore_chain >/dev/null && echo RUNNING || echo IDLE
EOF
  exit 0
fi

[[ -d /Volumes/ROS2_SSD/asvproject/captures ]] || {
  echo "ROS2_SSD not mounted" >&2; exit 1; }

echo "[1/5] code -> home"
rsync -az -e "ssh ${SSH_OPTS} -J ${JUMP}" \
  --include='scripts/***' --include='configs/***' --include='labels/***' \
  --exclude='*' --exclude='__pycache__' \
  "$LOCAL/" "$REMOTE:asvproject_project/ASVProject-ObstacleDetection/"

echo "[2/5] raw captures (1.6G) + lars-free data deps -> scratch"
$RSH "$REMOTE" "mkdir -p $RAW/captures /scratch0/${REMOTE_USER}/{logs,det,seg,features_motion,models_out,results}" 2>&1 | grep -v VBoxManage || true
for m in 2026-06-17_institutionone_day1 2026-07-08 2026-08-18_calibration \
         2026-08-18_pontoon 2026-08-19_afloat 2026-08-19_fx_walk; do
  rsync -az --partial -e "ssh ${SSH_OPTS} -J ${JUMP}" \
    "/Volumes/ROS2_SSD/asvproject/captures/$m/" "$REMOTE:$RAW/captures/$m/"
  echo "  $m"
done

echo "[3/5] seg masks + typed dets + partial features + motion yaml"
rsync -az --partial -e "ssh ${SSH_OPTS} -J ${JUMP}" \
  "$G/corpus_out/native/seg/" "$REMOTE:/scratch0/${REMOTE_USER}/seg/"
rsync -az -e "ssh ${SSH_OPTS} -J ${JUMP}" \
  "$G/corpus_out/native/det/" "$REMOTE:/scratch0/${REMOTE_USER}/det/"
rsync -az -e "ssh ${SSH_OPTS} -J ${JUMP}" \
  "$G/next_booking/canoe_fx__2026-08-19_15-58-20.jsonl" \
  "$REMOTE:/scratch0/${REMOTE_USER}/det/" 2>/dev/null || true
rsync -az --partial -e "ssh ${SSH_OPTS} -J ${JUMP}" \
  "$G/features_motion/" "$REMOTE:/scratch0/${REMOTE_USER}/features_motion/"
rsync -az -e "ssh ${SSH_OPTS} -J ${JUMP}" \
  "$G/next_booking/detection_motion.yaml" "$G/logs/trackb_triplets.txt" \
  "$REMOTE:/scratch0/${REMOTE_USER}/"

echo "[4/5] remote chain script"
$RSH "$REMOTE" 'bash -s' <<'EOF' 2>&1 | grep -v VBoxManage
S=/scratch0/$USER
cat > $S/rescore_chain.sh <<'INNER'
#!/usr/bin/env bash
set -uo pipefail
S=/scratch0/$USER
CODE=$HOME/asvproject_project/ASVProject-ObstacleDetection
export XDG_CACHE_HOME=$S/.cache PIP_CACHE_DIR=$S/.cache/pip TMPDIR=$S/tmp
mkdir -p $S/.cache/pip $S/tmp
log() { echo "$(date +%H:%M:%S) $*"; }
cd $CODE
log "[env] CPU-only venv (no torch)"
[[ -x $S/cpu_venv/bin/python ]] || python3 -m venv $S/cpu_venv
source $S/cpu_venv/bin/activate
pip install --no-cache-dir -q --upgrade pip
pip install --no-cache-dir -q -r requirements.txt scikit-learn pillow \
  || pip install --no-cache-dir -q opencv-python-headless numpy pandas pyyaml tqdm matplotlib scikit-learn pillow
# wire scratch data into the code tree (stale-home check: remove real dirs)
for l in data/captures:$S/asvproject_raw/captures data/seg:$S/seg \
         data/det:$S/det data/features:$S/features_motion \
         results:$S/results models:$S/models_out; do
  p=${l%%:*}; t=${l#*:}
  [[ -L $p ]] && rm $p
  [[ -e $p && ! -L $p ]] && mv "$p" "${p}.stale_$(date +%s)"
  mkdir -p "$(dirname $p)"; ln -s "$t" "$p"
done
# session/canoe seg aliases (node scene naming)
for d in $S/seg/2026-08-19_afloat_session__*; do
  [[ -d $d ]] || continue
  t=$S/seg/session__${d##*__}; [[ -e $t ]] || ln -s "$d" "$t"
done
c=$S/seg/2026-08-19_afloat_canoe_fx__2026-08-19_15-58-20
[[ -d $c ]] && { t=$S/seg/canoe_fx__2026-08-19_15-58-20; [[ -e $t ]] || ln -s "$c" "$t"; }
log "[features] finish the rebuild (skip existing)"
build() {
  local t=$1
  python - "$t" "$S/features_motion" <<'PYIN'
import sys, pathlib
sys.path.insert(0, ".")
from scripts.utils.datasets import resolve_triplet
tr = resolve_triplet(sys.argv[1], require=("fisheye", "mmwave"))
d = pathlib.Path(sys.argv[2]) / f"{tr.scene}__{tr.timestamp}"
sys.exit(0 if (d / "meta.json").exists() else 3)
PYIN
  local rc=$?
  if [[ $rc -eq 3 ]]; then
    python -m scripts.fusion_model.build_features --triplet "$t" \
      --detection $S/detection_motion.yaml --out $S/features_motion 2>&1 | tail -1
  elif [[ $rc -ne 0 ]]; then
    log "ERROR probing $t (rc=$rc) — not skipping silently"
  fi
}
while read -r t; do [[ -n "$t" ]] && build "$t"; done < $S/trackb_triplets.txt
for sub in session canoe_fx; do
  for m in $S/asvproject_raw/captures/2026-08-19_afloat/$sub/fisheye_*.mp4; do
    ts=$(basename "$m"); ts=${ts#fisheye_}; ts=${ts%.mp4}
    build "data/captures/2026-08-19_afloat/$sub/$ts"
  done
done
log "[features] dirs: $(ls $S/features_motion | wc -l)"
log "[targets]"
python -m scripts.fusion_model.build_targets --features $S/features_motion \
  --include-unaudited 2>&1 | tail -1
log "[ablation] 2x2 on identical tables"
for v in "base:" "motion:--motion-context" "typed:--typed-evidence" "both:--motion-context --typed-evidence"; do
  tag=${v%%:*}; flags=${v#*:}
  log "--- ab_$tag $flags"
  python -m scripts.fusion_model.train --features $S/features_motion \
    --tag ab_$tag --seeds 0 1 2 $flags 2>&1 | grep -E "AP per seed|holdout"
  cp models/fusion_scorer_v1a.json $S/models_out/fusion_scorer_ab_$tag.json
done
log "[winner] pick by eval AP (results/fusion_model/bakeoff_ab_*.json)"
WINNER=$(python - <<'PYW'
import json, pathlib
best, best_ap = "base", -1.0
for tag in ("base", "motion", "typed", "both"):
    try:
        r = json.loads(pathlib.Path(
            f"results/fusion_model/bakeoff_ab_{tag}.json").read_text())
        ap = max(float(v["average_precision"]) for k, v in r.items()
                 if isinstance(v, dict) and k.startswith("v1a")
                 and v.get("average_precision") is not None)
    except Exception:
        continue
    if ap > best_ap:
        best, best_ap = tag, ap
print(best)
PYW
)
log "[winner] ab_$WINNER"
cp $S/models_out/fusion_scorer_ab_$WINNER.json models/fusion_scorer_v1a.json
cp models/fusion_scorer_v1a.json $S/models_out/fusion_scorer_rescore_winner.json
log "[sectors] re-emit EVERY clip with the winner (tracker + targets on)"
mkdir -p $S/results/sectors_rescored
emit() {
  local t=$1 name=$2
  python -m scripts.sensor_processing.fusion --triplet "$t" \
    --detection $S/detection_motion.yaml \
    --out "$S/results/sectors_rescored/${name}.jsonl" --track --targets \
    --scorer models/fusion_scorer_v1a.json > /dev/null 2>&1 \
    || log "WARN sectors: $name"
}
while read -r t; do
  [[ -n "$t" ]] || continue
  emit "$t" "$(basename "$(dirname "$t")")__$(basename "$t")"
done < $S/trackb_triplets.txt
for sub in session canoe_fx imu_rock; do
  for m in $S/asvproject_raw/captures/2026-08-19_afloat/$sub/fisheye_*.mp4; do
    ts=$(basename "$m"); ts=${ts#fisheye_}; ts=${ts%.mp4}
    emit "data/captures/2026-08-19_afloat/$sub/$ts" \
         "2026-08-19_afloat_${sub}__${ts}"
  done
done
log "[sectors] $(ls $S/results/sectors_rescored | wc -l) files, $(find $S/results/sectors_rescored -size 0 | wc -l) empty"
log "RESCORE DONE"
INNER
chmod +x $S/rescore_chain.sh
nohup bash $S/rescore_chain.sh > $S/logs/rescore.log 2>&1 &
echo "rescore chain launched pid $!"
EOF
echo "[5/5] launched. Poll: bash scripts/gpu_corpus/rescore_restage.sh --status"
