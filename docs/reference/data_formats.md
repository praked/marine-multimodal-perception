# Data-format spec: capture streams + fusion feature tables (2026-07-17)

The one-page reference promised by `docs/plans/capture_status_and_prereqs.md`
§3.B, extended with the Phase-0 feature-table schema
(`docs/plans/fusion_scoring_training.md` §4). Loaders named below
are the single entry points; do not re-parse these files by hand.

## 1. On-disk capture streams (one chunk = one `<ts>` = `YYYY-MM-DD_HH-MM-SS`)

| File | Loader | Columns / format | Notes |
|---|---|---|---|
| `fisheye_<ts>.mp4` | `iterate_triplet` | 864×648 BGR, nominal 3 fps | `recovered/<name>_trimmed.mp4` preferred when present |
| `thermal_<ts>.mp4` | `iterate_triplet` | 160×120 BGR (colourised) | lockstep with fisheye (one frame per loop) |
| `mmwave_<ts>.csv` | `datasets.load_mmwave_csv` | `Date,Time,X,Y,Z[,V][,SNR,NOISE]` | X lateral (right +), Y forward, Z up, metres; V radial Doppler m/s (since 2026-07-06, \|V\|>10 → NaN); SNR/NOISE dB (TLV-7, since 2026-07-09); \|X/Y/Z\|≥50 m rows dropped (spliced-TLV garbage); **sentinel rows** have empty X/Y/Z = "radar alive, zero objects" (timestamp heartbeat) |
| `imu_<ts>.csv` | `datasets.load_imu_csv` | `Date,Time,Yaw,Pitch,Roll[,Ax,Ay,Az]` (UART-RVC, deg / m·s⁻², since 2026-07-14) **or** legacy `Date,Time,W,X,Y,Z` quaternion (BLE, 2026-07-08 clips) | format auto-detected; trailing all-NaN row dropped; axis mapping applied at replay via `configs/imu.yaml replay:` |
| `frames_<ts>.csv` | `datasets.load_frames_csv` | `frame_index,Date,Time[,ExposureTime,AnalogueGain,DigitalGain,Lux]` (timestamps since 2026-07-16; exposure since 2026-08-28) | absent → synthesized `chunk_start + index/3 fps` (verified ≈1-frame accurate); one call, no special-casing. **Exposure columns** = picamera2's per-frame request metadata: ExposureTime in µs, analogue/digital gain, Lux estimate; empty when the camera gave none. Load-bearing for night work: at 3 fps auto-exposure integrates up to ~300 ms at high gain, so a frame that looks like evening can be a long exposure of near-darkness (the 2026-08-26 20:40 frames). `build_features` carries them as `exposure_time_us, analogue_gain, digital_gain, lux_est` (schema v4, NaN on older clips) |
| `gps_<ts>.csv` | `datasets.load_gps_csv` | `Date,Time,Lat,Lon,Fix,Source,FixTime` (since 2026-08-24) | own-boat position, **metadata not nav**: one row at chunk start (the position in force, so every clip is self-contained) plus one whenever the fix changes, on a ~30-min poll. `Date,Time` = when THIS box wrote the row (joins on RoundedTime like every other stream); `FixTime` = the source's own UTC timestamp, so a repeated row is recognisable as the same reading. `Fix` = NMEA GGA quality (4 = RTK fixed) or empty when unreported — empty ≠ 0. **`Source` is load-bearing**: `boat_log` = live fix read from boat1's autopilot log, `ble` = the BLE bridge, `fallback` = the fixed position in `configs/gps.yaml` and **not a measurement** — indistinguishable from a real fix by its coordinates alone, so anything deriving sun geometry or a weather lookup should say which it had |
| `radar_profile.cfg` | *(read by eye)* | the radar chirp/detection config the clip was captured under, verbatim, plus `% source:` and `% summary:` header lines (since 2026-08-19) | one per capture directory, not per chunk. The summary line carries `channelCfg`, `cfarCfg` and `clutterRemoval`, i.e. what field A/Bs vary. Written because the 2026-08-19 clutterRemoval A/B silently ran both halves on the default profile: a clip that cannot say how its radar was configured cannot be compared with another |

**Time.** All `Date,Time` values are the Pi system clock = **local time**
(Europe/Berlin), RTC-backed since 2026-07-16 (correct offline).
`RoundedTime` = `HH:MM:SS.f` truncated to 100 ms: the cross-stream join key
(matches the radar's 100 ms frame). `iterate_triplet` walks radar
RoundedTime groups as the master clock, one video frame per group; when the
radar CSV is empty it falls back to per-frame synthesized timestamps.
Ragged streams are expected: a stalled stream writes gaps (radar/IMU) while
frames keep their index.

**Radar point columns through the pipeline.** `iterate_triplet` yields
`(ts, fisheye, thermal, points)` with points Nx3 / Nx4 (+V) / Nx6
(+V,SNR,NOISE: appended only when *both* side-info columns exist, so
Doppler is always column 3). `process_mmwave` filters (y_min/y_max,
self-clutter zone) and exposes `points_xyz` (always Nx3), `snr_db`,
`noise_db` aligned with the filtered cloud.

## 2. Fusion-scorer feature tables (`scripts.fusion_model.build_features`)

Output: `data/features/<scene>__<ts>/{bins,frames}.{parquet|csv}` +
`meta.json` (schema_version 1). Gitignored, regenerable. Conventions:
missing modality ⇒ **NaN + `*_available=False`** (never zero-filled);
bearings degrees, right +, 0° = bow; ranges metres; binning is a
parameter of the exporter (defaults = detection.yaml fusion bins): the
underlying features are bearing-space.

### bins table: one row per (frame × bearing bin)

| Group | Columns | Meaning |
|---|---|---|
| ids | `clip_id, timestamp, frame_index, bin_index, bin_center_deg, bin_left_deg, bin_right_deg` | `timestamp` = RoundedTime (local) |
| incumbent | `hit_fisheye, hit_thermal, hit_mmwave, num_sensors, score_legacy, min_range_m, confirmed, per_bin_velocity_mps, per_bin_ttc_s` | the rule-based baseline-to-beat; `score_legacy ≡ (Σ hits)/num_sensors` is verified on every export (`check_score_reproduction`). Under default config `hit_fisheye` is the **retired legacy blob channel**: baseline reproduction only, excluded from every model feature set |
| e_fisheye (seg) | `seg_available, seg_obstacle_frac, seg_comp_count, seg_comp_max_size_px, free_space_m, free_space_blocked` | mask-derived evidence; `seg_obstacle_frac` = OBSTACLE-pixel fraction of the bin's column band (pinhole bearing→column); `free_space_m` NaN = open past max_range or no mask (disambiguate via `seg_available`); free-space columns use the legacy linear column↔bearing convention internally (`free_space_profile`) |
| fisheye detector | `fisheye_det_count, fisheye_det_max_size_px, fisheye_det_min_range_m, fisheye_det_min_raw_range_m, fisheye_det_votes, fisheye_det_radar_hits, fisheye_det_min_radar_range_m` | whatever `fisheye.detector` ran (blob under default config → legacy) |
| thermal detector | `thermal_det_*` (same shape) | the hot-blob channel; live evidence, stays |
| radar | `radar_n_points, radar_min_range_m, radar_median_range_m, radar_max_snr_db, radar_mean_snr_db, radar_median_noise_db, radar_n_velocity` | per-point cloud after pipeline filters; SNR/noise NaN pre-2026-07-09 |
| availability | `fisheye_available, thermal_available, mmwave_available, seg_available` | stream present this frame |
| reserved | `yolo_available (False), yolo_max_conf (NaN), yolo_top_cls ("")` | deferred typed-YOLO channel (plan §7): schema stable now, retrain later |
| targets (v3) | `target_present, target_range_m, target_closing_mps, target_v_tangential_mps, target_speed_mps, target_cpa_m, target_t_cpa_s, target_age_frames, target_motion_state` | per-bin tracked-target motion (`motion.targets` tracker; nearest-range target per bin). Reserved-style: all False/NaN/"" while the tracker is off (the default), filled when on — Phase-3 consumption is a retrain, not a migration. `target_motion_state` ∈ closing/crossing/diverging/static/unknown; signs per sector_protocol v1.2 (closing positive = approaching, tangential positive = rightward) |

### frames table: one row per frame (context features)

| Group | Columns |
|---|---|
| ids / split | `clip_id, scene, timestamp, frame_index, split, group_key` (split = frozen `scripts.data.splits` assignment, outing-level `group_hash`) |
| availability | `fisheye_available, thermal_available, mmwave_available, imu_available, seg_available` |
| fisheye context | `luminance_mean, luminance_p05, luminance_p95, is_dark` (mean < `fisheye.darkness_thresh` = 25, the SegWorker gate), `fisheye_horizon_conf`, `range_reference_fisheye` |
| thermal context | `thermal_quality_std, thermal_quality_dyn_range, thermal_quality_ok, thermal_horizon_conf, range_reference_thermal`: ⚠ Lepton AGC makes these unreliable *darkness* proxies (night water reads high-contrast); darkness comes from the fisheye |
| attitude | `attitude_roll_deg, attitude_pitch_deg, attitude_yaw_deg` (IMU replay; NaN before the first sample / without a sidecar) |
| radar frame stats | `radar_n_points, radar_n_points_raw, radar_velocity_source (doppler/mixed/tracker), radar_has_doppler, radar_has_snr, radar_median_snr_db` |
| sun | `sun_elevation_deg, sun_azimuth_deg` (NOAA `gps_boat1.sun_position`, elevation <0 = below horizon, azimuth clockwise from true N), `sun_lat_deg, sun_lon_deg, sun_fix_source ("fallback" until GPS sidecars exist), tz`: computed from the RTC timestamp converted local→UTC; midnight-crossing chunks roll the date forward |

`range_reference_*` = which attitude source won `range.attitude_priority`
for that frame's monocular ranges: `imu / water_edge / horizon / level`
(empty = range disabled). Radar-vs-mono provenance per detection is
derivable from `fisheye_det_min_radar_range_m` vs `*_min_range_m`.

### targets table (`scripts.fusion_model.build_targets`): one row per (labelled frame × bin)

| Column | Meaning |
|---|---|
| `clip_id, frame_index, bin_index, bin_center_deg` | join keys onto bins |
| `y_obstacle` | 1 iff a label bearing (pinhole, from the bbox centre, NOT the stored linear-convention `obstacle_bins_fisheye`) falls within half-bin + tolerance, in frames ±`dilation_frames` (default 7) |
| `y_relevant` | 0 when every supporting label's bbox-bottom mono range reads beyond `max_relevant_range_m` (30) or above the horizon: the perception-GT vs nav-GT split |
| `y_nav` | `y_obstacle AND y_relevant`: the nav-score training target |
| `from_dilation` | positive came only from a neighbouring frame |
| `label_classes` | "\|"-joined same-frame classes (person recall etc.) |
| `label_min_range_m` | nearest supporting label's mono range (NaN unknown) |
| `n_labels_frame, audited` | per-frame label count / any-audited flag |

Rows exist **only for labelled frames** (negatives on unlabelled frames are
unknown, not zero). Frames-table schema v2 adds
`fisheye_horizon_slope/intercept` + `fisheye_K`/`fisheye_camera_height_m`/
`image_size` in meta.json, which is what makes the label mono-range
computation identical to the pipeline's.

### meta.json

`schema_version, clip_id, n_frames, n_bin_rows, bin_edges_deg, num_sensors,
fisheye_detector, bearing_model{fisheye,thermal}, segmentation_enabled,
imu_replayed, split, group_key, sun{lat,lon,source,tz}`: enough to detect
config drift between exports; regenerate rather than mixing schema or
config generations in one training run.

## 3. Known caveats

- 2025-era radar CSVs: no V/SNR, ~30–40 % of points were UART-splice
  garbage (denormals): loader guards drop most; treat radar features from
  pre-2026-07-06 clips as low-trust.
- `frames.timestamp` for pre-2026-07-16 clips is the synthesized
  chunk-start+index/3 fps estimate (verified ≈1 frame accurate on real
  data); the radar-keyed iteration can also desync on capture FPS drift
  for the oldest clips (CLAUDE.md §7 time-alignment note).
- Sun position uses `build_features`' own fixed-lake fallback
  (47.66, 9.18) on every clip captured before 2026-08-24, which has no
  `gps_<ts>.csv` sidecar: fine for elevation/azimuth (<1° across Lake
  the lake). Clips WITH a sidecar are not yet wired into
  `build_features` (`--lat/--lon` still take a single value for the whole
  export); when they are, honour `Source` — a `fallback` row is that same
  fixed guess wearing a per-clip coordinate, not a measurement.
- One export = one config generation. `bins.csv` from different
  `detection.yaml` states (e.g. detector=blob vs segmentation) must not be
  concatenated blindly; compare `meta.json` first.

## 4. Scorer artefact + emitted probabilities (2026-08-22)

- `models/fusion_scorer_v1a.json` (GatedMixtureScorer) now embeds an
  **isotonic calibrator** (`calibrator: {x, yhat}`, thinned to 512
  quantile knots, fit on the best seed's own TRAIN predictions).
  `fusion.py --scorer` emits `predict_proba_calibrated` — a probability
  on the true base rate — with a raw fallback for pre-2026-08-22
  artefacts. Rationale: v1a trains with class-balanced BCE (positives
  ~5×), so the RAW output is calibrated to a ~balanced prior; without
  the embedded calibrator an evidence-free bin floors near p≈0.45
  instead of ~0.15 (the 2026-08-22 open-water misread).
- JSONLs written before 2026-08-22 carry the RAW score in
  `p_obstacle`; re-emit rather than mixing eras in one analysis.
- The `yolo_available / yolo_max_conf / yolo_top_cls` bins columns are
  populated only when `data/det/<scene>__<ts>.jsonl` files are present
  at build time — verify before using them (they were empty in the
  gpu-node-campaign tables).
- `p_obstacle_smooth` (2026-08-24, additive): persistence-smoothed
  `p_obstacle` — `max(p_raw, exp(-dt/decay_time_s) * p_prev)` per bin,
  dt from the record timestamps. Rises instantly (`>= p_obstacle`
  pointwise, zero onset delay by construction); decay time constant +
  gap-reset behaviour in `detection.yaml fusion.scorer_smoothing`
  (default OFF: field absent, JSONL byte-identical). Emit with
  `--scorer-smoothing TAU_S` or the config block; smoothing lives in
  `scripts/fusion_model/smoothing.py`, flicker/onset eval in
  `scripts.eval.scorer_smoothing_eval`
  (docs/history/2026-08-24_scorer_smoothing.md).

## 5. Dashboard bundle additions (2026-08-24)

- `meta.json` frame entries may carry `thermal_q: "low"` — thermal
  present but below the quality-guard contrast floors (std<25 or
  p99−p1<120, matching detection.yaml `thermal.quality_guard`); drives
  the viewer's amber "degraded" health state and is the seed for the
  degraded-sensor training ablations (present-but-useless ≠ absent).

## 6. Set curation: `configs/curation.yaml` (2026-08-28)

Source of truth is the dashboard's `public.sail_curation` table (soft
delete / restore + trim ranges per set, edited on `/clips` and in the
viewer; append-only history in `sail_curation_log`). `cd dashboard && pnpm
curation:export` writes it to the git-tracked **`configs/curation.yaml`**,
which `scripts/utils/curation.py` reads for every offline consumer:
`list_triplets` (deleted sets omitted), `iterate_triplet` (cut frames read
in lockstep but not yielded; `respect_curation=False` yields all),
`dashboard/tools/bake_corpus.py` (deleted chunks dropped before activity
grouping, cut frames not baked; `--ignore-curation`),
`scripts.fusion_model.build_features` (deleted sets refused, cut frames
dropped with the RAW `frame_index` kept; `--ignore-curation`) and
`scripts.eval.audit_frame_selector` (deleted sets / cut frames never
planned; `--ignore-curation`). The dashboard's own planner, data links
(`/api/export`: 410 for a deleted set, per-frame objects inside cuts
omitted), coverage map, replay and offline packs honour the same table
directly.

```yaml
version: 1
exported_at: "2026-08-28T20:12:35Z"
sets:                                    # only deleted (incl. purged) or trimmed sets
  "<scene>__<first_chunk_ts>":            # dashboard clip key (activity or lone chunk)
    deleted_at: "…" | null               # deleted := deleted_at set and not restored
    restored_at: "…" | null              #   since (restored_at < deleted_at), or purged
    purged_at: "…" | null                # bundle removed by `pnpm curation:prune --yes`
    chunks: ["<ts>", …]                  # member capture chunks (activity = successive
                                         #   chunks; from the catalogue, else the key's ts)
    cuts:                                # inclusive frame-id timestamp ranges
      - {start_ts: "HH:MM:SS.f", end_ts: "HH:MM:SS.f", note: "…"}
    note: "…"                            # free text (why)
    updated_by: "…"                      # actor / tool
```

Rules that matter downstream:

- **Keys resolve two ways**: a clip id (`scene/ts` or `scene__ts`) matches a
  set directly or through its `chunks` list — deleting a concatenated
  activity deletes every capture chunk it was built from, and a cut on the
  activity applies to whichever chunk holds those frames (cuts are
  wall-clock ranges, chunk-independent by construction).
- **Frame ids never change.** A cut removes frames from what consumers
  iterate; labels, audits, sector records and feature rows keyed by frame id
  keep lining up. `frame_index` in the feature tables stays the raw video
  index (gaps where frames were cut).
- **Nothing here deletes files.** Raw captures on the SSD are never touched;
  the only byte-removing path is `pnpm curation:prune --yes` on the derived
  R2 bundle + catalogue row of sets deleted for more than 30 days (dry run
  without `--yes`), after which the set stays in this file as `purged_at`.
- **A missing file is an empty curation** (pre-curation checkouts behave as
  before); `ASVPROJECT_CURATION=/path` overrides the location (tests).
- The first export carries the baker's former hard-coded
  `EXCLUDE_CHUNK_IDS` (AuthorTwo's 2026-08-21 per-chunk screening, 16 chunks)
  as deleted sets with `updated_by: migration:bake_corpus.EXCLUDE_CHUNK_IDS`,
  so nothing that was excluded before the table existed reappears.
- Re-export + commit after every curation session in the dashboard; the
  Python side is only as current as the committed file (`python -m
  scripts.utils.curation` prints what it holds).
