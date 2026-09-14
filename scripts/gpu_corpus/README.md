# Full-corpus night-run orchestration (GPU node)

Two-track overnight pipeline that runs every model over the whole 2026
corpus on a InstTwo GPU node, without needing the SSD for the first track.
Written for the 2026-08-21 "gpu-node" booking; full campaign record +
incident list: `docs/history/2026-08-22_gears_corpus_campaign.md`.

Shares `scripts/gpu_seg/00_run_params.sh` for the SSH topology — set the
per-booking node in the gitignored `00_run_params.local.sh` (`GPU_NODE=`).
House rules apply: code in AFS home, every venv/cache/dataset/output on
`/scratch0` (wiped at booking end — extract first), remote commands piped
to `bash -s` (login shell is tcsh).

## Track A — no local data needed

```bash
bash scripts/gpu_corpus/run_night.sh            # sync, env, weights, rclone, launch
ONLY=2026-08-26_afloat bash scripts/gpu_corpus/run_night.sh   # scope stages 2-3 to one mission
bash scripts/gpu_corpus/run_night.sh --status   # poll
```

`run_night.sh` also installs a static rclone under `$S/bin` and writes
`$S/rclone.conf` from the R2 creds in `dashboard/.env.local`
(`setup_rclone.sh`; 0600 on scratch, gone with the booking) — this used to
be a by-hand step on every fresh booking. `ONLY=<clip-key substring>`
scopes `predict_bundle` and `label_bundle_dino` (stage 1 still syncs the
whole bundle: it is the cheap part).

Stages (nohup on the node, STATUS state machine in `$S/logs/STATUS`):
1. rclone-pull the private R2 dashboard bundle to scratch (creds written
   to `$S/rclone.conf` 0600; the bundle's 432×324 frames are upscaled to
   native scale before inference).
2. `predict_bundle.py` — LRASPP-864 student masks + YOLOv8n typed det +
   YOLOv8s-seg instances, emitted in REPO format keyed by the real
   capture chunk (bundle meta `chunk` attribution; NB meta `ts` is
   colon-format, filenames dash — normalised in the lookup).
3. `label_bundle_dino.py` — the tuned GroundingDINO recipe
   (configs/detector_labeler.yaml) over bundle frames; batch 4 +
   `expandable_segments` (batch 8 OOMs on 16 GB), stride 2 (lossless
   under the ±7-frame target dilation), resumable per frame.

Weights come from the published HF collection (fetch_weights.sh); no SSD.

## Track B — raw triplets (needs the SSD mounted locally)

```bash
bash scripts/gpu_corpus/stage_raw.sh            # rsync raw captures + lars_mastr, launch
```

Stages: native undistorted frame export (every frame) → native re-run of
all three models (supersedes Track A) → thermal JOINT ladder (gray-LaRS +
gray-InstitutionOne + warped thermal, 3 seeds) → fusion features → targets →
v1a/v1b train/eval → per-clip `--scorer` sector JSONLs → native
GroundingDINO relabel (deferred last as a quality upgrade).

## Gotchas that will bite again

- **AFS home persists across bookings**: stale real dirs (e.g.
  `labels/qwen`) win a `[[ -e ]]` guard and silently shadow the corpus
  data; **dangling symlinks** from the previous booking do the opposite —
  `-e` is false, `ln -s` fails silently on the existing name, and every
  consumer sees an empty tree (2026-08-28: `thermal_labels` found no masks,
  `build_features` died on `FileExistsError`). `night_queue_trackb.sh`'s
  `wire()` now replaces dangling links and moves real dirs aside; keep
  using it rather than bare `ln -s`.
- `list_triplets` does not walk nested missions
  (`2026-08-19_afloat/session/…`) — pass explicit `--triplet` paths.
- `predict_bundle.py --bundle-root` also accepts a frames-root layout
  (`<scene>__<ts>/ts=*.jpg`, no meta.json) — used for the native pass.
- Extract before booking end: `corpus_out/`, `runs/`, `features/`,
  `results/`, `models_out/` → `ROS2_SSD/asvproject/gears_<date>/`.
