#!/usr/bin/env bash
# LOCAL: pull a GPU node's corpus outputs for one mission into an SSD archive
# (+ optionally the repo data tree, rebake/enrich, publish).
#
#   bash scripts/gpu_corpus/pull_outputs.sh 2026-08-26_afloat                  # Track A layout, place + bake + publish
#   bash scripts/gpu_corpus/pull_outputs.sh 2026-08-26_afloat --native         # Track B layout
#   bash scripts/gpu_corpus/pull_outputs.sh 2026-08-26_afloat --native --archive-only --extras
#
# Flags:
#   --native        Track B layout (native full-res passes) instead of Track A
#   --archive-only  pull to the SSD archive only: no placement, bake or publish
#   --no-labels     place det/det_seg/seg but NOT labels (labels/qwen is the
#                   audit's label set; swapping it mid-sprint forks the audit)
#   --no-publish    place + bake + enrich, skip the R2/Supabase publish
#   --extras        also archive the Track-B training/scoring outputs:
#                   runs/thermal_joint_s*, models_out/, results/, features/,
#                   thermal_dataset/fit_report.json, logs/
# Env:
#   ARCHIVE_DIR     archive root (default /Volumes/ROS2_SSD/asvproject/cluster_<today>)
#
# First executed 2026-08-28 (Track B pull). Layouts on the node ($S/corpus_out):
#   Track A: det/<clip>.jsonl  det_seg/<clip>.jsonl  labels_dino/det_<clip>.jsonl  seg/<clip>/
#   Track B: native/det/<clip>.jsonl  native/det_seg/<clip>.jsonl
#            seg_native/<clip>/  labels_native/det_<clip>.jsonl
#   (Track B's predict_bundle writes under corpus_out/native/; only its seg
#   is copied up to seg_native by the queue — det/det_seg stay under native/.)
# Local placement: data/det, data/det_seg (symlinks onto the SSD), labels/qwen.
# Seg: Track A bundle-resolution masks are NOT placed (the local native
# local_seg_masks output is better); Track B native masks ARE placed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../gpu_seg/00_run_params.sh"
LOCAL="${LOCAL_PROJECT_ROOT}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=25"
RSH="ssh ${SSH_OPTS} -J ${REMOTE_USER}@${JUMP_HOST}"
REMOTE="${REMOTE_USER}@${GPU_NODE}"
S=/scratch0/${REMOTE_USER}
OUT=$S/corpus_out

M="${1:?mission (e.g. 2026-08-26_afloat)}"; shift || true
NATIVE=0; PUBLISH=1; PLACE=1; LABELS=1; EXTRAS=0
for a in "$@"; do case "$a" in
  --native) NATIVE=1;; --no-publish) PUBLISH=0;; --archive-only) PLACE=0; PUBLISH=0;;
  --no-labels) LABELS=0;; --extras) EXTRAS=1;;
  *) echo "unknown flag $a" >&2; exit 2;;
esac; done

ssd_ok() { [[ -d /Volumes/ROS2_SSD/asvproject/captures ]]; }
ssd_ok || { echo "ROS2_SSD not mounted" >&2; exit 1; }
ARCH="${ARCHIVE_DIR:-/Volumes/ROS2_SSD/asvproject/cluster_$(date +%Y-%m-%d)}/corpus_out"
mkdir -p "$ARCH"/{det,det_seg,labels,seg}
# retried rsync: the SSD cable and the lab SSH pipe both drop mid-transfer
rs() { local a; for a in 1 2 3 4; do rsync -az --partial -e "$RSH" "$@" && return 0
        echo "  rsync attempt $a failed, retrying in 20 s" >&2; sleep 20; ssd_ok || { echo "SSD GONE" >&2; exit 1; }; done; return 1; }
cnt() { ls "$1" 2>/dev/null | grep -c "$2" || true; }

if [[ $NATIVE == 1 ]]; then DET=$OUT/native/det; DSEG=$OUT/native/det_seg; SEG=$OUT/seg_native; LAB=$OUT/labels_native
else DET=$OUT/det; DSEG=$OUT/det_seg; SEG=$OUT/seg; LAB=$OUT/labels_dino; fi

echo "[1/4] pull ${M} outputs ($( [[ $NATIVE == 1 ]] && echo native || echo bundle )) -> $ARCH"
rs --include="${M}__*" --exclude="*" "$REMOTE:$DET/" "$ARCH/det/"
rs --include="${M}__*" --exclude="*" "$REMOTE:$DSEG/" "$ARCH/det_seg/"
rs --include="det_${M}__*" --exclude="*" "$REMOTE:$LAB/" "$ARCH/labels/"
rs --include="${M}__*/" --include="${M}__*/**" --exclude="*" "$REMOTE:$SEG/" "$ARCH/seg/"
echo "  det $(cnt "$ARCH/det" "^${M}__")  det_seg $(cnt "$ARCH/det_seg" "^${M}__")  labels $(cnt "$ARCH/labels" "^det_${M}__") files  seg clips $(cnt "$ARCH/seg" "^${M}__")"

if [[ $EXTRAS == 1 ]]; then
  X="$(dirname "$ARCH")"
  echo "[1b]  extras -> $X (runs, models_out, results, features, fit_report, logs)"
  mkdir -p "$X"/{runs,models_out,results,features,logs,thermal_dataset}
  rs --include="thermal_joint_s*/" --include="thermal_joint_s*/**" --exclude="*" "$REMOTE:$S/runs/" "$X/runs/" || true
  rs "$REMOTE:$S/models_out/" "$X/models_out/" || true
  rs "$REMOTE:$S/results/" "$X/results/" || true
  rs "$REMOTE:$S/features/" "$X/features/" || true
  rs "$REMOTE:$S/thermal_dataset/fit_report.json" "$X/thermal_dataset/" || true
  rs "$REMOTE:$S/logs/" "$X/logs/" || true
  ssd_ok || { echo "SSD GONE after extras" >&2; exit 1; }
  echo "  runs $(ls "$X/runs" | wc -l | tr -d ' ')  models_out $(ls "$X/models_out" | wc -l | tr -d ' ')  features $(ls "$X/features" | wc -l | tr -d ' ')"
fi

if [[ $PLACE == 0 ]]; then echo "[2/4] placement skipped (--archive-only)"; echo "PULL-DONE ${M} (archive only)"; exit 0; fi

echo "[2/4] place into the repo tree"
cp "$ARCH"/det/"${M}"__*.jsonl "$LOCAL/data/det/"
cp "$ARCH"/det_seg/"${M}"__*.jsonl "$LOCAL/data/det_seg/"
if [[ $LABELS == 1 ]]; then
  ls "$ARCH"/labels/det_"${M}"__*.jsonl >/dev/null 2>&1 && cp "$ARCH"/labels/det_"${M}"__*.jsonl "$LOCAL/labels/qwen/"
else echo "  labels NOT placed (--no-labels)"; fi
if [[ $NATIVE == 1 ]]; then
  for d in "$ARCH"/seg/"${M}"__*/; do rsync -a "$d" "$LOCAL/data/seg/$(basename "$d")/"; done
fi

echo "[3/4] rebake + enrich"
"$LOCAL/.venv/bin/python" "$LOCAL/dashboard/tools/bake_corpus.py" --force --only "$M"
"$LOCAL/.venv/bin/python" "$LOCAL/dashboard/tools/enrich_bundle.py"

if [[ $PUBLISH == 1 ]]; then
  echo "[4/4] publish"
  ( cd "$LOCAL/dashboard" && set -a && source .env.local && set +a && npx tsx tools/ingest/ingest_r2.ts --prune )
else
  echo "[4/4] publish skipped (--no-publish)"
fi
echo "PULL-DONE ${M}"
