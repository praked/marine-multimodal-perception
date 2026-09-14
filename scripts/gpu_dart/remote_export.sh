#!/usr/bin/env bash
# Runs ON the GPU node. Exports undistorted 864x648 fisheye frames for the
# staged triplets listed in $S/dart_triplets.txt (built by run_dart.sh --stage).
# Skips clips already exported. Uses the repo's export_undistorted_frames.
set -uo pipefail
S=${DART_S:-/scratch0/$USER}; CODE=${CODE:-$HOME/asvproject_project/ASVProject-ObstacleDetection}
RAW=$S/asvproject_raw; NATIVE=$S/native_frames; EVERY=${DART_EVERY:-1}
export XDG_CACHE_HOME=$S/.cache TMPDIR=$S/tmp
log() { echo "$(date +%H:%M:%S) $*"; }
cd $CODE; mkdir -p data $NATIVE
[[ -L data/captures && -e data/captures ]] || { rm -f data/captures; ln -s $RAW/captures data/captures; }
source $S/dart_venv/bin/activate
[[ -s $S/dart_triplets.txt ]] || { log "FAILED: no $S/dart_triplets.txt"; exit 1; }
log "$(wc -l < $S/dart_triplets.txt) triplets, stride $EVERY"
n=0; skipped=0; failed=0
while read -r t; do
  [[ -n "$t" ]] || continue
  # nested missions flatten to <mission>_<sub>__<ts> (datasets.resolve_triplet convention)
  rel=${t#data/captures/}; ts=$(basename "$rel"); scene=$(dirname "$rel" | tr '/' '_')
  if [[ -d $NATIVE/${scene}__${ts} ]]; then skipped=$((skipped+1)); continue; fi
  if python -m scripts.eval.export_undistorted_frames --triplet "$t" --out $NATIVE --every $EVERY >> $S/logs/export_dart.log 2>&1; then n=$((n+1)); else failed=$((failed+1)); log "WARN export $t"; fi
done < $S/dart_triplets.txt
log "exported $n, skipped $skipped (present), failed $failed; $(find $NATIVE -name '*.jpg' | wc -l) frames in $(ls $NATIVE | wc -l) dirs"
log "EXPORT-DART DONE"
