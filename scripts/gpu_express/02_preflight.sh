#!/usr/bin/env bash
# 02: read-only look at the node: GPU, scratch, home quota, envs, staged inputs, running queues.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_params.sh"
echo "node: ${GPU_NODE}"
remote_env "RAW=${RAW}" <<'EOF2'
date; hostname; uptime | sed 's/.*load/load/'
echo "== GPU"; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "== scratch ($S)"; df -h $S | tail -1; du -sh $S 2>/dev/null
echo "== home (AFS quota!)"; du -sh $HOME 2>/dev/null
echo "== envs"; for v in corpus_venv dart_venv asvproject_yolo/yolo_venv; do [ -x $S/$v/bin/python ] && echo "  $v: $($S/$v/bin/python --version 2>&1)" || echo "  $v: missing"; done
echo "== weights"; ls $S/asvproject_models 2>/dev/null | sed 's/^/  /'; [ -s $S/asvproject_models/dart/sam3.pt ] && echo "  dart/sam3.pt present" || echo "  dart/sam3.pt MISSING"
echo "== staged raw"; for m in $(ls $RAW/captures 2>/dev/null); do echo "  $m: $(find $RAW/captures/$m -name 'fisheye_*.mp4' | wc -l) fisheye chunks"; done; [ -d $RAW/lars_mastr ] && echo "  lars_mastr: $(ls $RAW/lars_mastr/images 2>/dev/null | wc -l) images"; [ -d $RAW/thermal_inits ] && echo "  thermal_inits: $(ls $RAW/thermal_inits | wc -l) files"; [ -d $RAW/labels/qwen ] && echo "  labels/qwen: $(ls $RAW/labels/qwen/*.jsonl 2>/dev/null | wc -l) files"
echo "== outputs"; [ -d $S/native_frames ] && echo "  native_frames: $(ls $S/native_frames | wc -l) clips"; [ -d $S/corpus_out ] && du -sh $S/corpus_out/* 2>/dev/null | sed 's/^/  /'; [ -d $S/runs ] && echo "  runs: $(ls $S/runs | tr '\n' ' ')"
echo "== queues"; pgrep -fa "night_queue|remote_queue|label_bundle|train_student|train_yolo|build_features|fusion\.py" | grep -v pgrep | cut -c1-90 || echo "  (idle)"
echo "== STATUS files"; for f in $S/logs/STATUS $S/logs/STATUS_dart $S/logs/STATUS_yolo $S/logs/STATUS_express; do [ -f $f ] && { echo "  -- $(basename $f)"; tail -n 3 $f | sed 's/^/     /'; }; done
EOF2
