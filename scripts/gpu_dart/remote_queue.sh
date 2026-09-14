#!/usr/bin/env bash
# Runs ON the GPU node under nohup (launched by run_dart.sh --launch).
# smoke (5 frames, prints boxes) -> label every exported clip (resumable per
# frame: records already in det_<clip>.jsonl are skipped). Status file:
# $S/logs/STATUS_dart ; logs: $S/logs/dart_{smoke,label}.log
set -uo pipefail
S=${DART_S:-/scratch0/$USER}; CODE=${CODE:-$HOME/asvproject_project/ASVProject-ObstacleDetection}; cd $CODE
export XDG_CACHE_HOME=$S/.cache HF_HOME=$S/.cache/huggingface TORCH_HOME=$S/.cache/torch TMPDIR=$S/tmp TORCHINDUCTOR_CACHE_DIR=$S/.cache/inductor
CK=$S/asvproject_models/dart/${SAM3_FILE:-sam3.pt}; OUT=$S/corpus_out/${DART_OUT_NAME:-labels_dart}; LOG=$S/logs
CFG=${DART_CONFIG:-configs/dart_labeler.yaml}; EVERY=${DART_EVERY:-1}; COMPILE=${DART_COMPILE:-}
PY=$S/dart_venv/bin/python; DRIVER=scripts/gpu_corpus/label_bundle_dart.py
status() { echo "$(date +%H:%M:%S) $*" >> $LOG/STATUS_dart; }
[[ -s $CK ]] || { status "FAILED: no checkpoint at $CK"; exit 1; }
[[ -L data/captures && -e data/captures ]] || { rm -f data/captures; ln -s $S/asvproject_raw/captures data/captures; }
CFLAG=(); [[ -n "$COMPILE" ]] && CFLAG=(--compile "$COMPILE")
TXT=$S/dart_text_cache_$(basename ${CFG%.yaml}).pt

status "queue start: config $CFG stride $EVERY compile '${COMPILE:-none}'"
SMOKE=${DART_SMOKE_CLIP:-}; [[ -n "$SMOKE" ]] || SMOKE=$(ls $S/native_frames | head -1)
status "smoke on $SMOKE"
$PY $DRIVER --bundle-root $S/native_frames --out-dir $S/corpus_out/${DART_OUT_NAME:-labels_dart}_smoke --only "$SMOKE" \
  --max-frames 5 --checkpoint $CK --config $CFG "${CFLAG[@]}" > $LOG/dart_smoke.log 2>&1 || { status "SMOKE FAILED (see dart_smoke.log)"; exit 1; }
$PY - "$S/corpus_out/${DART_OUT_NAME:-labels_dart}_smoke" <<'PY' >> $LOG/dart_smoke.log
import json, glob, sys
for f in glob.glob(sys.argv[1] + "/*.jsonl"):
    for line in open(f):
        r = json.loads(line); print(r["frame_id"], len(r["fisheye_bboxes"]), [(b["cls"], b["prompt"], round(b["confidence"], 2), [round(v, 2) for v in b["xyxy"]]) for b in r["fisheye_bboxes"][:5]])
PY
status "smoke done"
status "labelling all exported clips -> $OUT"
$PY $DRIVER --bundle-root $S/native_frames --out-dir $OUT --every $EVERY --checkpoint $CK --config $CFG \
  --text-cache $TXT "${CFLAG[@]}" > $LOG/dart_label.log 2>&1 || status "WARN labelling run exited non-zero (resumable: relaunch)"
status "DART DONE: $(cat $OUT/*.jsonl 2>/dev/null | wc -l) records in $(ls $OUT 2>/dev/null | wc -l) files"
