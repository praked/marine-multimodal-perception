# Sector Protocol v1

Provisional contract for the per-frame obstacle-sector messages emitted
by `scripts.sensor_processing.fusion --out <path>.jsonl`.

This is the input that the ASVProject navigation layer will consume.
v1 adds *motion* (per-bin velocity, time-to-collision) and an
*advisory heading recommendation* on top of v0's score + range vector,
so the navigation owner has the reasoning visible alongside the
number (see PLAN.md Branches D.2 + I.4.10).

## On-disk format

JSON Lines (`.jsonl`): one self-contained JSON object per frame,
newline-delimited.

```json
{
  "protocol": 1,
  "timestamp": "16:21:08.3",
  "clip_id": "Boats/2025-06-23_16-21-07",
  "bin_centers_deg": [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
  "scores":          [0.0,  0.0,  0.33, 0.66, 1.0,  0.66, 0.33, 0.0,  0.0,  0.0,  0.0],
  "min_range_m":     [null, null, 3.4,  3.2,  3.1,  3.3,  3.5,  null, null, null, null],
  "sensor_hits":     [[0,0,0],[0,0,0],[1,0,0],[1,1,0],[1,1,1],[1,1,0],[1,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
  "tracked":         false,

  "per_bin_velocity_mps": [null, null, 0.4, 0.6, 0.8, 0.7, 0.4, null, null, null, null],
  "per_bin_ttc_s":        [null, null, 8.5, 5.3, 3.9, 4.7, 8.8, null, null, null, null],

  "recommended_heading_deg": -22.0,
  "heading_reason": "ttc",
  "heading_candidates_deg": [-50.0, -49.0, ..., 50.0],
  "heading_cost_curve":     [0.18, 0.18, ..., 0.91]
}
```

## Field reference

| Field | Type | Meaning |
|-------|------|---------|
| `protocol` | int | Schema version. `0` (or absent) = legacy; `1` = motion + heading. |
| `timestamp` | string `HH:MM:SS.f` | Radar `RoundedTime` for the frame (100 ms granularity; matches mmWave config). |
| `clip_id` | string | `<scene>/<timestamp>` of the source triplet. Replay reference; nav can ignore in live mode. |
| `bin_centers_deg` | list[int] | 11 ten-degree bin centres from `-50..+50`. Same array every frame. |
| `scores` | list[float] | Fused confidence per bin. Range `0..1`. See *Score semantics* below. |
| `min_range_m` | list[float \| null] | Minimum radar Y (range, m) inside each bin. `null` if no radar hit. |
| `sensor_hits` | list[3-list[int]] | Per-bin sensor presence: `[fisheye, thermal, mmwave]`, 0/1. Provided so consumers can reason about *which* sensor fired (e.g. radar-only in fog). |
| `tracked` | bool | `true` if `--track` was passed: scores went through the SectorTracker before publish. |
| **v1 fields** | | |
| `per_bin_velocity_mps` | list[float \| null] | Radial closing speed per bin (m/s, positive = approaching). `null` when no matched radar point in that bin. Source: cross-frame greedy NN; see `scripts/sensor_processing/motion.py`. |
| `per_bin_ttc_s` | list[float \| null] | Time-to-collision (s), capped at 60 s. `null` when range or velocity unknown, or closing speed ≤ epsilon. |
| `recommended_heading_deg` | float \| null | Heading recommendation in degrees from bow (negative = port). `null` when the recommender abstained (ambiguous tied alternatives). |
| `heading_reason` | string | Dominant cost term at the chosen heading: `"block"`, `"close"`, `"ttc"`, `"course"`, `"nogo"`, `"clear"`, `"ambiguous"`, or `"all_blocked"`. |
| `heading_candidates_deg` | list[float] | Candidate-heading grid that `heading_cost_curve` is sampled on. |
| `heading_cost_curve` | list[float] | Aggregate cost per candidate; argmin is the recommendation (unless ambiguous). |
| `smoothed_heading_deg` | float \| null | Rolling-mean of `recommended_heading_deg` over the configured `smoothing_window` (default 9 frames). Holds the last smoothed value across ambiguous frames so the steering signal stays steady. **This is the value the autopilot should follow**; `recommended_heading_deg` is the unsmoothed per-frame argmin. `null` until `smoothing_min_samples` frames have been seen. |

### v1.1 extensions (2026-07-09: quad-clip stack; all optional)

Emitted by `fusion.py` when the corresponding source is active; absent on
older streams and when the source is off. Consumers must treat each as
optional.

| Field | Type | Meaning |
|-------|------|---------|
| `attitude` | object \| null | Replayed/live BNO085 attitude at the frame: `{roll_deg, pitch_deg, yaw_deg}` (camera-frame roll/pitch via the validated axis mapping, configs/imu.yaml `replay:`). `null` when no IMU sample within the staleness window. **Yaw is the absolute-heading reference**: compass-frame steer = `yaw_deg + sign * smoothed_heading_deg` (see `sector_consumer.absolute_heading`; the sign must be verified on-boat once: dock test against a landmark). |
| `free_space_m` | list[float \| null] | Per-bin navigable free-space distance from the water mask (segmentation.free_space_profile, EMA-smoothed). `null` = open past `range.max_range_m`. Present when `--seg --free-space-heading`. |
| `confirmed` | list[bool] | Per-bin object-level cross-sensor confirmation (`fusion.association`): the bin holds radar returns that were matched (padded-bbox gate) to a camera detection. Semantics since 2026-08-05: the pad is per-camera so the angular tolerance is equal (~4.3°: `max_px` 32 fisheye / `thermal_max_px` 12 thermal); the flag is anchored on the bins the MATCHED RETURNS occupy, per point (`confirm_anchor: radar`), and only marks anything when the detection's own bin is among them (`confirm_same_bin`), so a confirmed bin always contains radar evidence of the confirmed object (extended objects mark every bin they span), and a fully cross-bin match marks nothing. `min_range` in a confirmed bin tightens with the nearest matched return in that bin. Before 2026-08-05 the flag sat on the detection's centre bin with a shared 32 px pad (~13° on thermal): open-water bins could be "confirmed" by neighbour-bin returns; old JSONLs differ accordingly. Far stronger evidence than two sensors coincidentally sharing a bin: a consumer may treat `confirmed` blocked bins as higher-priority threats. Present with `--assoc`. |

### v1.2 extension (2026-08-24: per-target motion; optional)

Emitted by `fusion.py --targets` (or `motion.targets.enabled: true`).
Additive: per-bin consumers are unaffected; absent when the tracker is off.

| Field | Type | Meaning |
|-------|------|---------|
| `targets` | list[object] | One entry per **tracked radar object** confirmed this frame (`motion.targets`: clustered cloud, cross-frame association, min_hits gate). Sorted nearest first. Fields per target: `id` (track id, stable across frames), `bearing_deg`/`range_m`/`n_points`/`age_frames`, `v_radial_mps` (range-rate, **negative = approaching**, Doppler sign), `v_tangential_mps` (r x bearing-rate, positive = moving to the boat's right), `vx_mps`/`vy_mps`/`speed_mps`/`course_deg` (full 2-D velocity in the boat frame, rotation-corrected), `closing_mps` (= -v_radial, matching `per_bin_velocity_mps` sign), `cpa_m`/`t_cpa_s` (predicted closest point of approach from relative motion; null when speed is below the noise floor or CPA is in the past), `motion_state` (`closing` \| `crossing` \| `diverging` \| `static` \| `unknown`), `ego_corrected` (bearing-rate measured in the IMU-yaw-stabilised frame). Any velocity component may be null while its channel warms up (Doppler-free points / rate window unfilled). |

Motion semantics: the velocity is **relative** (ego translation deliberately
included: collision geometry lives in relative motion; a static buoy dead
ahead of a moving boat IS closing), but ego **rotation** is subtracted via
the IMU yaw so a turning boat does not read the world as crossing.
`crossing` targets are the COLREG give-way case the radial-only
`per_bin_velocity_mps` cannot see; consumers should treat a `crossing`
target with small `cpa_m`/`t_cpa_s` as a priority threat even when its
`closing_mps` is near zero.

### v1.3 extension (2026-08-25: cross-sensor track identity; optional)

Emitted by `fusion.py --identity` (or `motion.targets.identity.enabled:
true`; `--camera-rate` additionally enables
`motion.targets.camera_bearing_rate`). Additive: one extra field per
`targets` entry; absent on pre-identity streams (records without the key
are byte-identical to v1.2).

| Field | Type | Meaning |
|-------|------|---------|
| `targets[].camera_track_id` | int \| null | The camera (fisheye BBoxTracker) track this radar target is paired with, or null when no pairing is confirmed. Pairing is *physical*: it forms only after `pair_min_hits`-of-`pair_window` frames in which the radar returns matched to the camera detection (`fusion.association` pixel gate) intersect the target's own cluster points, and it breaks only on sustained co-observed disagreement — a radar or camera dropout holds the pair rather than divorcing it (see `scripts/sensor_processing/track_identity.py`). Stable across frames while the pair lives. |

With `camera_bearing_rate` on, a paired target additionally survives radar
dropouts on camera observations (bounded by
`motion.targets.camera_sustain_max_s`): such frames report the target with
`n_points: 0` — the honest marker that the range is dead-reckoned (held
Doppler) and the bearing/bearing-rate came from the paired camera track.
Consumers wanting radar-measured geometry only should skip `n_points == 0`
entries; consumers wanting continuity (the COLREG crossing case loses its
radar mid-transit on the 2026-08-19 GT clip for 4 s) should keep them.

### Shadow scorer extension (optional; additive)

Emitted by `fusion.py --scorer` (learned gated-mixture scorer, shadow
mode — see `docs/reference/data_formats.md` §4 for the artefact and
calibration-era caveats). Additive: absent when no scorer is loaded.

Which bundle produced a published sector file matters for `p_obstacle`
comparability: `results/sectors_v1b/` (2026-09-02/03) = live v1b bundle,
train-fit calibrator, pre-fix self-clutter zone; `results/sectors_v1b_oofcal_v2/`
(2026-09-05, what the dashboard serves) = OOF-calibrated v1b + the fixed zone
(`docs/history/2026-09-05_sector_reemission.md`). The OOF bundle was
promoted to `models/fusion_scorer_live_v1b.joblib` the same day (repo/SSD;
the box receives it at the next deploy). The on-boat shadow service beside
capture emits with the radar-only bundle instead (tail mode has no camera
evidence; `deploy/asvproject-shadow.service`).

| Field | Type | Meaning |
|-------|------|---------|
| `p_obstacle` | list[float] | Calibrated per-bin obstacle probability from the learned scorer (`predict_proba_calibrated`; raw fallback for pre-2026-08-22 artefacts). Scored per frame, memoryless. |
| `threat` | list[float] | `p_obstacle` × range/TTC/CPA urgency (`threat_from_score`), severity placeholder pending the D.2 table. |
| `p_obstacle_smooth` | list[float] | Persistence-smoothed `p_obstacle` (2026-08-24, additive): `p_smooth = max(p_raw, exp(-dt/decay_time_s) · p_prev)`. **Rises instantly** — `p_obstacle_smooth ≥ p_obstacle` every frame, so every threshold crossing happens on the same frame as the raw series (zero onset delay); only the decay is damped (default time constant `fusion.scorer_smoothing.decay_time_s`, dt from the record timestamps). State resets across timestamp gaps > `gap_reset_s` (chunk rolls) and backwards jumps. Present only when `fusion.scorer_smoothing.enabled` (or `--scorer-smoothing TAU_S`); default off. Consumers wanting a steady per-bin threat signal should prefer this over `p_obstacle`; the raw field remains the per-frame evidence. |

## Score semantics

- `0.0`, no sensor fired in this bin.
- `0.33`, exactly one sensor fired.
- `0.66`: two sensors fired.
- `1.0`: all three sensors fired (highest possible at v0).

The score is `num_sensor_hits / num_sensors`. It is **not a probability**;
treat it as a coarse agreement strength.

## Motion semantics

- `per_bin_velocity_mps` is the *radial closing speed* relative to the
  own boat's frame; positive means the obstacle is getting closer.
  Since 2026-07-06 the capture writes the TI radar's per-point Doppler
  as a `V` CSV column; on such captures the per-point velocity comes
  from Doppler directly (`MMWaveResult.velocity_source` =
  `"doppler"`, or `"mixed"` when the cross-frame tracker fills NaN
  gaps). On older captures without the column it falls back to
  cross-frame greedy nearest-neighbour association (`"tracker"`).
  The field's shape is identical in all three cases.
- The tracker has no knowledge of own-boat motion. While the sailboat
  is moving forward at ~1 m/s, static obstacles read as approaching at
  ~1 m/s. Once the IMU/GPS bridge lands (V.3), own-velocity will be
  subtracted before publishing.
- `per_bin_ttc_s` is `min_range_m / per_bin_velocity_mps` clamped at 60 s.
  `null` whenever the closing speed is at or below the configured
  `ttc_eps_mps` (i.e. opening or near-static motion).

## Heading recommendation

- Advisory, not commanded. The autopilot owns the final action,
  including any COLREG translation and wind / no-go-zone overlay.
- Computed by sweeping candidate headings across the bin span at 1°
  resolution and summing weighted cost terms (see
  `scripts/sensor_processing/heading.py`):
  - `block`: Gaussian-falloff "threat field" amplitude from per-bin
    scores
  - `close`: same field, amplified by 1/(range + scale)
  - `ttc`: same field, amplified by 1/(ttc + scale)
  - `course`: penalty for deviation from `current_heading_deg`
  - `nogo`: stubbed at 0 until V.3 wind sensor wiring lands
- The recommender returns `null` rather than picking arbitrarily when
  the cost minimum is tied across distinct directions (e.g. obstacle
  exactly ahead). "No recommendation" is the safer signal than
  "swerve confidently in a random direction"; see PLAN.md §III.1.
- `smoothed_heading_deg` is the rolling-mean version intended for
  the autopilot. The autopilot should *follow the smoothed value*
  and treat the per-frame `recommended_heading_deg` as diagnostic
  only: a single noisy frame can swing the 1°-resolution argmin
  by tens of degrees, and steering on that would oscillate. The
  smoothed value holds across `null` recommended-heading frames so
  brief abstentions don't drop the line.

## Bearing convention

- `0°` is straight ahead (boat's bow direction).
- Positive bearings are to **starboard / right**.
- Negative bearings are to **port / left**.
- This matches the radar's TI convention (`atan2(X, Y)`) and the camera
  pixel convention `(x - cx) / pix_deg_ratio`.

The 11-bin layout (`-55..+55`, step `10`) covers the radar's reliable
azimuth resolution. Beyond `±55°` is currently out-of-spec; the cameras
can see further but `cx`/`pix_deg_ratio` are sanity-checked only in the
central FOV.

## Time

`timestamp` is the radar frame's wall-clock time at 100 ms granularity.
There is **no clock-domain mapping yet** between this timestamp and any
external nav clock; that's a v2 task. For now, sequential frames are
~100 ms apart on average.

## Consumption recipe

A 10-line Python stub that converts the stream into "blocked-sector
+ recommended-heading" decisions at a 0.66 threshold:

```python
import json
BLOCKED = 0.66
for line in open("results/sectors_xxx.jsonl"):
    rec = json.loads(line)
    blocked = [b for b, s in zip(rec["bin_centers_deg"], rec["scores"]) if s >= BLOCKED]
    closest = min((r for r in rec["min_range_m"] if r is not None), default=None)
    head = rec.get("recommended_heading_deg")  # may be missing or null
    print(rec["timestamp"], "blocked", blocked, "closest_m", closest, "head", head)
```

`scripts/eval/sector_consumer.py` is the reference consumer; it
handles both v0 and v1 records and surfaces the new fields in its
line output.

## Versioning

Each record carries an explicit `protocol` integer. v0 records lacked
the field; readers should default to `0` when absent. New consumers
should reject unknown protocol versions explicitly rather than
silently ignore them.

## What v1 deliberately does NOT include

- Per-bin **type** (boat / duck / debris / unknown).
- A **hard-stop** bin (obstacle within emergency-stop range): the
  reference consumer derives it from `min_range_m` instead.
- **Track IDs** for multi-frame persistence; comes with Branch B.3.
- **Sensor-health** flag; comes with B.2 / C.2.
- **Hysteresis state** on the heading recommendation: the nav
  consumer is responsible for any flicker filtering.
- A **clock-domain hash** linking this stream to GPS/IMU sidecars.

## v0 → v1 migration notes

- The v0 fields are unchanged; v1 strictly *adds* keys.
- A v0 reader can consume v1 records safely if it ignores unknown
  keys (the reference consumer does).
- A v1 reader must handle missing or null heading fields; they
  signal `--no-heading` mode or recommender abstention.
- `fusion.py --out` writes v1 by default; pass `--no-heading` to
  emit v0 records for compatibility with old readers.

## BLE transport (2026-09-02)

The box→autopilot link: `scripts/data_collection/sector_ble_tx.py` on the
obstacle box (SensorBox) tails the shadow service's JSONL
(`shadow_<stamp>.jsonl`, or any `fusion.py --out` stream) and re-emits each
record over a BLE GATT notify characteristic; boat1 runs the matching
central (`sector_rx.py` on the boatv1-boat-b `sector-rx` branch) and logs
`sector_rx_<stamp>.jsonl` beside its boat logs. **Transport only** — the
autopilot does not yet consume the records. Codec source of truth:
`scripts/data_collection/sector_codec.py` (vendored verbatim into
boatv1-boat-b; never imported across repos).

**Roles + UUIDs.** The box is the GATT peripheral/server, advertising as
`ASVProjectSector`, service `b5ec70a0-5a11-4a5a-8d2a-6f1e0c9a0001` with one
notify-only characteristic `…-6f1e0c9a0002`. boat1 is the central
(subscriber). Both radios are the Pis' onboard BLE — same vessel, metres
apart; the NORA-B120 boat↔boat radio is not involved.

**Consumer status (2026-09-05).** boat1 logs the stream; for the final
sessions AuthorOne's heading policy on boat1 consumes it for a slow closed-loop
pilot on known targets (the paper's Table IV). The interface is unchanged.

**Advertisement, as deployed (2026-09-03).** The receiver matches the box by
**service UUID first**, local name second — deliberately, because on the box's
kernel (Raspberry Pi 6.18.34, raspberrypi/linux#7473) bluetoothd cannot
register an advertisement at all and `sector_ble_tx` falls back to a legacy
MGMT advertisement that carries the service UUID but no local name. The box
controller is LE-only (`ControllerMode = le`) so the advertisement flags
"BR/EDR Not Supported"; without that boat1's BlueZ attempts a classic
connection. The box keeps advertising while connected (several centrals can
subscribe; each gets every notification), stops advertising cleanly on exit
and disconnects its centrals first, so a transmitter restart shows up on the
receiver as a disconnect → re-scan → resubscribe, `seq` restarting from 0.
The receiver forgets the box in BlueZ after any connection error (its GATT
handles move on a restart and bleak's cache would otherwise hold the old
service paths) and drops a link that carries no record for `--stale-s`
(15 s) — an idle transmitter therefore cycles the connection, which is
harmless. Measured box → boat1: 0.30 s median / 0.37 s p95 from frame
timestamp to receipt (both clocks NTP-synced), 3 notifications per record at
the 20-byte floor.

**What travels** (the essentials only, everything else stays on disk):
timestamp, the uniform bin grid, per-bin `scores`, per-bin `p_obstacle`
when the record has one, per-bin `min_range_m`, and the shadow health bits
(`scorer_ok`, `seg_fresh`, `tick_ms`, `source_latency_s`). Heading,
targets, attitude, sensor_hits etc. are deliberately not on the wire at v1
of the transport.

**Record packet** (little-endian; `sector_codec.VERSION = 1`):

| off | size | field |
|----:|-----:|-------|
| 0 | 2 | magic `b"SB"` |
| 2 | 1 | codec version (1) |
| 3 | 2 | `seq` uint16, wraps (receiver gap detection) |
| 5 | 1 | flags: bit0 HAS_P_OBSTACLE, bit1 SCORER_OK, bit2 SEG_FRESH, bit3 HAS_LATENCY |
| 6 | 4 | timestamp, uint32 **deciseconds since midnight** (= the repo's 100 ms RoundedTime, exactly) |
| 10 | 1 | `n_bins` |
| 11 | 1 | first bin centre, int8 degrees |
| 12 | 1 | bin step, uint8 degrees (grid must be uniform — D.2 default: −45, step 15, n 7) |
| 13 | 2 | `tick_ms` uint16 |
| 15 | 2 | source latency, uint16 centiseconds, `0xFFFF` = absent |
| 17 | n | `scores`, uint8 each = round(score × 200) → 0.005 resolution |
| 17+n | 2n | `min_range_m`, uint16 centimetres each; `0xFFFF` = null, clamped at `0xFFFE` (655.34 m) |
| … | n | `p_obstacle`, uint8 × 200 (present iff flag bit0) |
| end | 1 | checksum: whole packet sums to 0 mod 256 |

Sizes at 7 bins: **46 bytes** with `p_obstacle`, 39 without. Quantisation
tolerances (test-asserted): scores/p ±0.0025, range ±0.005 m.

**Fragmentation.** Every packet travels as ≥1 notification fragments of
≤`--notify-bytes` (default 20, the ATT-MTU-23 floor → 3 notifications per
record at 3 fps): 2-byte prefix `[seq & 0xFF, frag_index | 0x80-if-final]`,
then packet bytes. Notifications on one connection are ordered, so the
reassembler only handles gaps (reconnects): any discontinuity drops the
partial packet and waits for the next index-0 fragment; the checksum guards
the rest. Once the BlueZ↔BlueZ negotiated MTU (typically 517) is confirmed
on the boat, `--notify-bytes 180` makes it one notification per record.

**Versioning.** The codec version is independent of the JSONL `protocol`
integer. Receivers reject a mismatched magic/version/checksum per packet
(`CodecError`) and keep listening — a format change bumps
`sector_codec.VERSION` and requires re-vendoring the codec on boat1.

**BLE packet flag bit 4 = YOLO_FRESH (2026-09-07, VERSION unchanged):** set when the tick had fresh typed-detection boxes; boat1's `sector_policy.has_camera_evidence` counts it alongside SEG_FRESH. Older decoders ignore the bit.
