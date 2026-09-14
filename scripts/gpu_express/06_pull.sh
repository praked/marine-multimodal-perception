#!/usr/bin/env bash
# 06: everything of value on scratch -> SSD archive ${ARCHIVE_ROOT}/<node>_<date>/ (+ optional
#     placement into the repo tree). Re-runnable; verifies the SSD is still mounted afterwards.
#     Pull BEFORE the booking ends: scratch is wiped.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_params.sh"
need_ssd
X="${ARCHIVE_ROOT}/${GPU_NODE%%.*}_$(date +%Y-%m-%d)"
mkdir -p "$X"/{corpus_out,runs,models_out,results,features,logs,thermal_dataset,thermal_dataset_0819,scripts}
pull() { local from="$1" to="$2"; echo "[$(ts)] $from"; rs "${REMOTE}:${S}/$from/" "$to/" || echo "  WARN: $from failed/absent"; need_ssd; }
pull corpus_out/native/det      "$X/corpus_out/det"
pull corpus_out/native/det_seg  "$X/corpus_out/det_seg"
pull corpus_out/seg_native      "$X/corpus_out/seg_native"
pull corpus_out/labels_native   "$X/corpus_out/labels_native"
pull corpus_out/labels_dino     "$X/corpus_out/labels_dino"
pull corpus_out/labels_dart     "$X/corpus_out/labels_dart"
pull runs                       "$X/runs"
pull models_out                 "$X/models_out"
pull results                    "$X/results"
pull features                   "$X/features"
pull thermal_dataset            "$X/thermal_dataset"
pull thermal_dataset_0819       "$X/thermal_dataset_0819"
pull logs                       "$X/logs"
pull asvproject_yolo/yolo_runs   "$X/yolo_runs"
[[ "${PULL_NATIVE_FRAMES}" == 1 ]] && pull native_frames "$X/native_frames"
rs "${REMOTE}:${S}/*.sh" "$X/scripts/" 2>/dev/null || true
echo "[$(ts)] archive:"; du -sh "$X"/* 2>/dev/null | sed 's/^/  /'
if [[ "${PULL_PLACE_LABELS_DART}" == 1 && -d "$X/corpus_out/labels_dart" ]]; then
  mkdir -p "${LOCAL}/labels/dart"; rsync -a "$X/corpus_out/labels_dart/" "${LOCAL}/labels/dart/"; echo "  placed labels/dart ($(ls "${LOCAL}/labels/dart" | wc -l | tr -d ' ') files) — never labels/qwen"
fi
if [[ "${PULL_PLACE_NATIVE_SEG}" == 1 && -d "$X/corpus_out/seg_native" ]]; then
  rsync -a "$X/corpus_out/seg_native/" "${LOCAL}/data/seg/"; echo "  placed native masks into data/seg"
fi
need_ssd; echo "[$(ts)] 06 PULL DONE -> $X"
