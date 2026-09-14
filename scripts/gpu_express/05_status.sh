#!/usr/bin/env bash
# 05: one-screen status of everything on the node (read-only). `--watch` refreshes every 60 s.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_params.sh"
show() { remote_env "" <<'EOF2'
date +%H:%M:%S; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
for f in STATUS_express STATUS STATUS_dart STATUS_yolo STATUS_0819 STATUS_thermal0819; do [ -f $S/logs/$f ] && { echo "== $f"; tail -n 6 $S/logs/$f; }; done
echo "== progress"
[ -d $S/native_frames ] && echo "  native frames: $(find $S/native_frames -name '*.jpg' | wc -l)"
[ -d $S/corpus_out/seg_native ] && echo "  native masks:  $(find $S/corpus_out/seg_native -name '*.png' | wc -l)"
for d in labels_native labels_dart; do [ -d $S/corpus_out/$d ] && echo "  $d: $(cat $S/corpus_out/$d/*.jsonl 2>/dev/null | wc -l) records / $(ls $S/corpus_out/$d | wc -l) files"; done
[ -d $S/features ] && echo "  feature tables: $(ls $S/features | wc -l)"; [ -d $S/results/sectors_night ] && echo "  sectors: $(ls $S/results/sectors_night | wc -l)"
grep -h "^epoch\|^HOLDOUT" $S/logs/thermal_train.log 2>/dev/null | tail -1 | cut -c1-100
grep -h "rec/s" $S/logs/dart_label.log $S/logs/label_native.log 2>/dev/null | tail -1 | cut -c1-100
echo "== running"; pgrep -fa "express_queue|night_queue|remote_queue|yolo_queue|label_bundle|train_student|train_yolo|build_features|fusion\.py|predict_bundle" | grep -v pgrep | cut -c1-90 || echo "  (idle)"
echo "== last errors"; for f in $S/logs/*.log; do grep -l "Traceback" "$f" 2>/dev/null; done | sed 's/^/  /' | tail -5
EOF2
}
if [[ "${1:-}" == "--watch" ]]; then while true; do clear; show; sleep 60; done; else show; fi
