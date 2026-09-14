#!/usr/bin/env bash
# Runs ON the GPU node under nohup: Track B — the canonical full-resolution
# pass from RAW triplets, thermal training round, and the fusion scoring
# pipeline. Requires stage_raw.sh to have landed $S/asvproject_raw first.
# Waits for Track A to finish (shares the GPU) before its own GPU stages.
#
#   Stage 4  native undistorted frame export (CPU, all corpus triplets)
#   Stage 5  native student masks + YOLO det + instances (supersede Track A)
#   Stage 6  native GroundingDINO labels
#   Stage 7  thermal warped labels + JOINT training ladder (3 seeds)
#   Stage 8  fusion features -> targets -> v1a/v1b train -> evaluate
#   Stage 9  per-clip sector scores (fusion.py --scorer) for every triplet
set -uo pipefail

S=${S:-/scratch0/$USER}
CODE=${CODE:-$HOME/asvproject_project/ASVProject-ObstacleDetection}
RAW=$S/asvproject_raw
OUT=$S/corpus_out
NATIVE=$S/native_frames
LOG=$S/logs
mkdir -p "$LOG" "$NATIVE" "$OUT"

status() { echo "$(date +%H:%M:%S) $*" >> "$LOG/STATUS"; }
fail() { status "TRACKB FAILED: $*"; exit 1; }

source "$S/activate_corpus.sh"
cd "$CODE"

# Wire scratch data into the code tree (code in home, data on scratch).
# AFS home persists across bookings while scratch is wiped, so a link left
# by a previous booking is DANGLING here: `[[ -e ]]` is false on it, `ln -s`
# then fails silently on the existing name, and every consumer sees an empty
# tree (2026-08-28: thermal_labels "no confident horizon pairs" and
# build_features FileExistsError both came from this). A REAL dir at a link
# name (e.g. labels/qwen synced from the laptop) is moved aside, never
# deleted.
wire() {  # wire <link> <scratch target>
  local link=$1 target=$2
  mkdir -p "$target"
  if [[ -L "$link" ]]; then
    [[ "$(readlink "$link")" == "$target" && -e "$link" ]] && return 0
    rm -f "$link"
  elif [[ -e "$link" ]]; then
    mv "$link" "${link}.stale_afs_$(date +%Y%m%d%H%M)"
  fi
  ln -s "$target" "$link"
}
mkdir -p data labels
wire data/captures "$RAW/captures"
wire data/seg      "$OUT/seg_native"
wire data/det      "$OUT/native/det"      # typed boxes from stage 5 (predict_bundle --out-root $OUT/native) -> build_features --det-root
wire data/det_seg  "$OUT/native/det_seg"
wire labels/qwen   "$OUT/labels_native"
wire labels/audited "$RAW/labels/audited"    # human truth (03_stage rsyncs labels/ -> RAW/labels)
wire labels/union   "$RAW/labels/union"      # teacher union (unaudited pseudo-labels)
wire data/features "$S/features"
wire results       "$S/results"
wire models        "$S/models_out"
# seed labels_native with the laptop's curated labels + Track A's corpus-wide
# bundle-resolution DINO labels (the native relabel runs last as an upgrade)
cp -n "$RAW/labels/qwen/"*.jsonl "$OUT/labels_native/" 2>/dev/null || true
cp -n "$OUT/labels_dino/"*.jsonl "$OUT/labels_native/" 2>/dev/null || true

# Stage selection (gpu_express): TRACKB_STAGES="4 5 7 8 9 6" (default = all, in
# the historical order). want N -> true when stage N is selected.
STAGES="${TRACKB_STAGES:-4 5 7 8 9 6}"
want() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
THERMAL_SEEDS="${THERMAL_SEEDS:-0 1 2}"; THERMAL_EPOCHS="${THERMAL_EPOCHS:-60}"; SCORER_TAG="${SCORER_TAG:-night_v1}"
status "trackB start (stages: $STAGES; stage4 is CPU-only, overlaps Track A's GPU work)"

# --- Triplet list (every run; stages 4, 7 and 9 read it) ---
# list_triplets() does NOT descend nested missions (data/captures/<mission>/
# <sub>/<ts>, e.g. 2026-08-19_afloat/{session,canoe_fx,imu_rock}) — the
# 2026-08-28 run silently skipped that whole outing in every stage and it
# had to be filled in by hand. Append nested triplets explicitly; `_`-prefixed
# dirs (bench/mirror) and recovered/ are not missions.
TRIPLETS=$(python - <<'PY'
from pathlib import Path
from scripts.utils.datasets import list_triplets
seen = set()
for t in list_triplets():
    # derive the on-disk prefix from the fisheye path: nested per-boot sessions
    # flatten t.scene to <mission>_<sub> (2026-09-12), so the old
    # data/captures/<scene>/<ts> form would not exist for them
    p = str(Path(t.fisheye).parent / t.timestamp)
    seen.add(p); print(p)
import os, re
# calibration / bench missions also nest (2026-07-09_lab/<cfar variant>,
# 2026-08-18_calibration/<station>, 2026-08-19_fx_walk/...): not corpus.
skip = re.compile(os.environ.get("NESTED_SKIP", r"lab|calibration|fx_walk|bench"))
for m in sorted(Path("data/captures").glob("*/*/fisheye_*.mp4")):
    sub = m.parent
    if sub.name in ("recovered",) or sub.parent.name.startswith("_") or sub.name.startswith("_"):
        continue
    if skip.search(sub.parent.name):
        continue
    if sub.name in ("on", "off"):          # clutter_ab/{on,off}: radar A/B halves, not scenes
        continue
    p = f"data/captures/{sub.parent.name}/{sub.name}/{m.stem[len('fisheye_'):]}"
    if p not in seen:
        seen.add(p); print(p)
PY
)

# --- Stage 4: native frame export (CPU) ---
if want 4; then
status "stage4 native frame export"
echo "$TRIPLETS" > "$LOG/trackb_triplets.txt"
while read -r t; do
  [[ -n "$t" ]] || continue
  python -m scripts.eval.export_undistorted_frames --triplet "$t" \
    --out "$NATIVE" --every 1 >> "$LOG/export_frames.log" 2>&1 \
    || status "stage4 WARN: export failed for $t"
done <<< "$TRIPLETS"
status "stage4 done: $(find "$NATIVE" -name '*.jpg' | wc -l) native frames"
fi

# --- Stage 5: native inference (GPU) ---
if want 5; then
status "waiting for Track A GPU stages to free the card"
while pgrep -f "label_bundle_dino.*labels_dino" >/dev/null; do sleep 120; done
STUDENT="${STUDENT:-$(find "$S/asvproject_models/lraspp-student" -name '*864*best*.pth' | head -1)}"
YDET="${YDET:-$(find "$S/asvproject_models/yolov8n-institutionone" -name '*.pt' | head -1)}"   # override: the leak-free audited v8n
YSEG=$(find "$S/asvproject_models/yolov8s-lars-seg" -name '*.pt' | head -1)
status "stage5 native predict"
python scripts/gpu_corpus/predict_bundle.py \
  --bundle-root "$NATIVE" --out-root "$OUT/native" \
  --student-pth "$STUDENT" --yolo-det "$YDET" --yolo-seg "$YSEG" \
  --batch 8 >> "$LOG/predict_native.log" 2>&1 || fail "native predict"
# native masks become the canonical data/seg
cp -a "$OUT/native/seg/." "$OUT/seg_native/" 2>/dev/null || true
status "stage5 done: $(find "$OUT/seg_native" -name '*.png' | wc -l) native masks"
fi

# --- Stage 7: thermal warp + JOINT ladder ---
if want 7; then
status "stage7 thermal labels + joint ladder"
THERMAL_TRIPLETS=$(while read -r t; do
  d=$(dirname "$t"); ts=$(basename "$t")
  # one ls per pattern: `ls a* b*` fails when EITHER glob is unmatched, which
  # silently dropped every clip WITHOUT a recovered/ copy (2026-08-28: day-1
  # never entered the ladder; the 08-19 fill-in matched 9/28 by luck).
  # 0-byte thermal files (dead-thermal chunks) are not thermal.
  { ls "$d"/thermal_"$ts"*.mp4 2>/dev/null; ls "$d"/recovered/thermal_"$ts"*.mp4 2>/dev/null; } \
    | xargs -r -I{} sh -c 'test -s "{}" && echo ok' | grep -q ok && echo "$t"
done <<< "$TRIPLETS")
TL_ARGS=""; for t in $THERMAL_TRIPLETS; do TL_ARGS="$TL_ARGS --triplet $t"; done
DS=$S/thermal_dataset
if [[ -n "$TL_ARGS" ]]; then
  python -m scripts.gpu_seg.thermal_labels $TL_ARGS --out "$DS" \
    >> "$LOG/thermal_labels.log" 2>&1 || status "stage7 WARN: thermal_labels failed"
fi
if [[ -d "$DS/images" ]]; then
  TPAIRS=""; for d in "$DS"/images/*/; do
    n=$(basename "$d"); TPAIRS="$TPAIRS --pairs $DS/images/$n:$DS/labels/$n"
  done
  # gray-InstitutionOne pairs: native frames + native masks, per clip
  KPAIRS=""; for d in "$NATIVE"/*/; do
    n=$(basename "$d")
    [[ -d "$OUT/seg_native/$n" ]] && KPAIRS="$KPAIRS --pairs $NATIVE/$n:$OUT/seg_native/$n"
  done
  HOLD=""
  [[ -d "$DS/images/2026-07-08__2026-07-08_16-44-56" ]] && \
    HOLD="--holdout-pairs $DS/images/2026-07-08__2026-07-08_16-44-56:$DS/labels/2026-07-08__2026-07-08_16-44-56"
  COMMON="--size 160x120 --batch 64 --workers 4 --export-sizes 160x120 --gray"
  for seed in $THERMAL_SEEDS; do
    status "stage7 joint_all_s${seed}"
    python -m scripts.gpu_seg.train_student \
      --pairs "$RAW/lars_mastr/images:$RAW/lars_mastr/masks" $KPAIRS $TPAIRS $HOLD \
      $COMMON --epochs $THERMAL_EPOCHS --seed $seed --aug-polarity 0.5 --aug-vshift 12 \
      --out "$S/runs/thermal_joint_s${seed}" \
      >> "$LOG/thermal_train.log" 2>&1 || status "stage7 WARN: seed $seed failed"
  done
fi
status "stage7 done"
fi

# --- Stage 8: fusion features -> targets -> train -> evaluate ---
if want 8; then
status "stage8 fusion features (CPU, long)"
python -m scripts.fusion_model.build_features --all ${FEATURES_ARGS:---det-root data/det --skip-existing} \
  >> "$LOG/fusion_features.log" 2>&1 || fail "build_features"
status "stage8 targets"
python -m scripts.fusion_model.build_targets --include-unaudited ${TARGET_LABELS:+--labels $TARGET_LABELS} \
  >> "$LOG/fusion_targets.log" 2>&1 || fail "build_targets"
status "stage8 train v1a/v1b"
python -m scripts.fusion_model.train --tag "$SCORER_TAG" --seeds 0 1 2 ${SCORER_TRAIN_ARGS:-} \
  >> "$LOG/fusion_train.log" 2>&1 || fail "train"
# train.py's bakeoff_<tag>.md already scores v1a/v1a_cal/v1b against the
# incumbent on the holdout; evaluate.py only knows columns present in the
# feature tables (there is no p_<tag> column — 2026-08-28 run WARNed on it),
# so the standalone report here is the incumbent's stratified breakdown.
python -m scripts.fusion_model.evaluate --score score_legacy \
  --out "$S/results/fusion_model/eval_incumbent_${SCORER_TAG}.md" \
  >> "$LOG/fusion_eval.log" 2>&1 || status "stage8 WARN: evaluate failed"
for H in ${ABLATE_HOLDOUTS:-}; do            # e.g. "clips:2026-09-08_20-4,2026-09-08_20-5 clips:2026-08-26_19-,2026-08-26_20-"
  status "stage8 ablate holdout $H"
  python -m scripts.fusion_model.ablate --tag "${SCORER_TAG}_$(echo "$H" | tr -c 'A-Za-z0-9' '_' | cut -c1-40)" --holdout "$H" ${ABLATE_ARGS:-} \
    >> "$LOG/fusion_ablate.log" 2>&1 || status "stage8 WARN: ablate $H failed"
done
status "stage8 done"
fi

# --- Stage 9: per-clip sector scores ---
if want 9; then
status "stage9 sector scores"
# Re-emit with THIS run's v1b bundle (train.py writes models/fusion_scorer_<tag>_v1b.joblib;
# fusion.py's load_any_scorer dispatches on the suffix); v1a JSON only as a fallback.
MODEL_JSON="${SECTOR_SCORER:-$CODE/models/fusion_scorer_${SCORER_TAG}_v1b.joblib}"
[[ -f "$MODEL_JSON" ]] || MODEL_JSON="$CODE/models/fusion_scorer_v1a.json"
[[ -f "$MODEL_JSON" ]] || MODEL_JSON=""
status "stage9 scorer: ${MODEL_JSON:-legacy n/3}"
mkdir -p "$S/results/sectors_night"
emit_one() {  # one triplet -> one sector file (parallelised below; 128-core gpu-node)
  local t="$1"; [[ -n "$t" ]] || return 0
  local name; name="$(basename "$(dirname "$t")")__$(basename "$t")"
  python -m scripts.sensor_processing.fusion --triplet "$t" \
    --out "$S/results/sectors_night/${name}.jsonl" --track \
    ${MODEL_JSON:+--scorer "$MODEL_JSON"} \
    >> "$LOG/sectors_${name}.log" 2>&1 || echo "$(date +%H:%M:%S) stage9 WARN: $name failed" >> "$LOG/STATUS"
}
export -f emit_one; export S LOG MODEL_JSON
printf '%s\n' $TRIPLETS | OMP_NUM_THREADS=2 xargs -P "${SECTORS_PARALLEL:-12}" -I{} bash -c 'emit_one "$@"' _ {}
status "stage9 done: $(ls "$S/results/sectors_night" | wc -l) sector files"
fi

# --- Stage 6 (deferred quality upgrade): native GroundingDINO labels ---
if want 6; then
status "stage6 native dino labels"
# Scope: a marker file $S/dino_only (clip-key substring) limits the native
# relabel to new missions — relabelling the whole corpus is ~7 h of GPU for
# an undecided quality upgrade and would fork the label set mid-audit.
DINO_ONLY=""; [[ -f "$S/dino_only" ]] && DINO_ONLY="$(cat "$S/dino_only")"
[[ -n "$DINO_ONLY" ]] && status "stage6 scoped to --only $DINO_ONLY"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/gpu_corpus/label_bundle_dino.py \
  --bundle-root "$NATIVE" --out-dir "$OUT/labels_native" \
  ${DINO_ONLY:+--only "$DINO_ONLY"} --batch-size 4 --every 2 >> "$LOG/label_native.log" 2>&1 \
  || python scripts/gpu_corpus/label_bundle_dino.py \
       --bundle-root "$NATIVE" --out-dir "$OUT/labels_native" \
       ${DINO_ONLY:+--only "$DINO_ONLY"} --batch-size 2 --every 2 >> "$LOG/label_native.log" 2>&1 \
  || fail "native dino"
status "stage6 done: $(cat "$OUT"/labels_native/*.jsonl 2>/dev/null | wc -l) records"
fi


status "TRACK B COMPLETE"
