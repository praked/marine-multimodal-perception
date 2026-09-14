#!/usr/bin/env bash
# 03: inputs -> node scratch (needs the SSD): raw triplets for MISSIONS (+EXTRA_TRIPLETS,
#     + clips named in AUDIT_PLAN), LaRS-as-MaSTr tree (built locally if missing),
#     thermal inits, labels/ (seeds the node's label set), the audit plan, the DINO scope marker.
#     Resumable (rsync); safe to re-run with a wider MISSIONS list.
#     --dry-run prints sizes only.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_params.sh"
need_ssd
MISS="${MISSIONS}"
[[ -n "$MISS" ]] || MISS=$(ls "${CAPTURES}" | grep '^2026' | grep -vE 'stability|bench|_lab|pi_card_backup|eval_smoke' | tr '\n' ' ')
LIST=$(mktemp); TRIP=$(mktemp)

# --- triplet + file lists ---
pat=$(echo "${STAGE_STREAMS}" | tr ' ' '|')             # fisheye|thermal|...  (ERE alternation)
( cd "${CAPTURES}" && for m in $MISS; do
    find "$m" -maxdepth 3 -type f -not -path '*/_*' | grep -E "/(recovered/)?(${pat})_[0-9-]+(_trimmed)?\.(mp4|csv)$" || true
  done ) > "$LIST"
( cd "${CAPTURES}" && for m in $MISS; do find "$m" -maxdepth 3 -name 'fisheye_*.mp4' -not -path '*/recovered/*' -not -path '*/_*' || true; done ) \
  | sed 's|/fisheye_| |; s|\.mp4$||' | awk '{print "data/captures/"$1"/"$2}' > "$TRIP" || true
for t in ${EXTRA_TRIPLETS}; do echo "$t" >> "$TRIP"; done
if [[ -n "${AUDIT_PLAN}" && -f "${AUDIT_PLAN}" ]]; then
  tail -n +2 "${AUDIT_PLAN}" | cut -d, -f2 | sort -u | python3 -c '
import sys, pathlib
C = pathlib.Path(sys.argv[1])
for clip in sys.stdin.read().split():
    scene, _, ts = clip.partition("__"); parts = scene.split("_")
    for k in range(len(parts), 0, -1):
        cand = C / scene if k == len(parts) else C / "_".join(parts[:k]) / "/".join(parts[k:])
        if (cand / f"fisheye_{ts}.mp4").exists() or (cand / "recovered" / f"fisheye_{ts}_trimmed.mp4").exists():
            print("data/captures/" + str(cand.relative_to(C)) + "/" + ts); break
    else: print("WARN plan clip not on SSD: " + clip, file=sys.stderr)
' "${CAPTURES}" >> "$TRIP"
fi
sort -u "$TRIP" -o "$TRIP"
# files for plan/extra triplets outside MISSIONS
( cd "${CAPTURES}" && while read -r t; do rel=${t#data/captures/}; d=$(dirname "$rel"); x=$(basename "$rel")
    for s in ${STAGE_STREAMS}; do ls "$d/${s}_$x.csv" "$d/${s}_$x.mp4" "$d/recovered/${s}_${x}"*.mp4 2>/dev/null || true; done; done < "$TRIP" ) >> "$LIST"
sort -u "$LIST" -o "$LIST"
echo "[$(ts)] missions: $MISS"
echo "[$(ts)] $(wc -l < "$TRIP" | tr -d ' ') triplets, $(wc -l < "$LIST" | tr -d ' ') files, $(cd "${CAPTURES}" && cat "$LIST" | xargs du -ch 2>/dev/null | tail -1 | cut -f1)"
[[ "${1:-}" == "--dry-run" ]] && { rm -f "$LIST" "$TRIP"; exit 0; }

# --- LaRS-as-MaSTr tree (local build once) ---
if [[ ! -f "${LARS_MASTR_LOCAL}/train_images.txt" ]]; then
  echo "[$(ts)] building LaRS-as-MaSTr tree -> ${LARS_MASTR_LOCAL} (once, ~10 min)"
  ( cd "${LOCAL}" && "${LOCAL}/.venv/bin/python" -m scripts.lars.prepare --lars-root "${LARS_RAW_LOCAL}" --out "${LARS_MASTR_LOCAL}" --size 512x384 )
fi

# --- transfers ---
echo "[$(ts)] mkdirs on node"
$RSH "$REMOTE" "bash -c 'mkdir -p ${RAW}/captures ${RAW}/lars_mastr ${RAW}/thermal_inits ${RAW}/labels ${S}/logs; while read -r p; do mkdir -p ${RAW}/captures/\$(dirname \"\$p\"); done'" < "$LIST" 2>&1 | grep -v "$NOISE" || true
echo "[$(ts)] raw triplets (resumable)"; rs --files-from="$LIST" "${CAPTURES}/" "${REMOTE}:${RAW}/captures/"
echo "[$(ts)] lars_mastr"; rs "${LARS_MASTR_LOCAL}/" "${REMOTE}:${RAW}/lars_mastr/"
echo "[$(ts)] thermal inits"; rs "${THERMAL_INITS_LOCAL}/" "${REMOTE}:${RAW}/thermal_inits/" || true
echo "[$(ts)] labels/ (seeds labels_native), audit plan, triplet list, DINO scope"
rs "${LABELS_LOCAL}/" "${REMOTE}:${RAW}/labels/"
[[ -f "${AUDIT_PLAN}" ]] && rs "${AUDIT_PLAN}" "${REMOTE}:${S}/audit_plan.csv"
rs "$TRIP" "${REMOTE}:${S}/dart_triplets.txt"
$RSH "$REMOTE" "bash -c 'if [ -n \"${DINO_ONLY}\" ]; then printf %s \"${DINO_ONLY}\" > ${S}/dino_only; else rm -f ${S}/dino_only; fi'" 2>&1 | grep -v "$NOISE" || true
sync_code
rm -f "$LIST" "$TRIP"
echo "[$(ts)] 03 STAGE DONE"
