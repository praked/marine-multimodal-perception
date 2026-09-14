# On-box timing: every model, every stage, every combination (Raspberry Pi 4)

Measured 2026-09-03 on SensorBox (Pi 4 Model B Rev 1.5, 1.8 GHz, Debian 13,
onnxruntime 1.29, scikit-learn 1.9.0), on the boat feed, dockside. Raw reports:
`results/pi_timing/pi_timing_2026-09-03{,_beside,_run1_cpufreq_probe}.json`
(SSD). Tool: `scripts.eval.pi_timing_bench`; figures:
`scripts.eval.pi_timing_figures` → `images/pi_timing/`. Reproduce (capture
service stopped for the clean numbers):

```bash
ssh vesselauser@<pi> 'cd ~/ASVProject-ObstacleDetection && python3 -m scripts.eval.pi_timing_bench \
    --triplet ~/captures/2026-09-03_12-13-15/2026-09-03_12-28-33 \
    --out ~/results/pi_timing_<date>.json --frames 60 --iters 30 --threads 2 --cool-c 74 --run-budget 12'
python -m scripts.eval.pi_timing_figures results/pi_timing/pi_timing_<date>.json \
    --beside results/pi_timing/pi_timing_<date>_beside.json --out images/pi_timing
```

## 0. How to read every number here (the two throttles)

The box throttles in two independent ways on the boat, and both silently
corrupt naive benchmarks:

1. **Under-voltage**: the boat feed sags under sustained CPU load and the
   firmware drops the ARM clock 1.8 GHz → 600 MHz for 2–8 s at a time
   (`get_throttled` bit 0). The same `process_fisheye` call on the same frame
   reads **245 ms at 1.8 GHz and 640 ms at 600 MHz**, alternating in blocks.
2. **Temperature**: the sealed enclosure idles at 73–78 °C in a warm lab; the
   firmware caps the clock from 80 °C (soft-limit bit at ~82 °C, mild: a
   single-core load draws it to 1.73 GHz; the heavy CNNs run 4–5 % slower).

Neither is visible in `scaling_cur_freq` (the governor's set-point, which kept
reporting 1.8 GHz through 600 MHz episodes — the `run1` report is the
evidence). The bench therefore reads the **firmware clock** (`vcgencmd
measure_clock arm`) and the SoC temperature around every model iteration and
every pipeline frame, and reports:

- **full-clock median** — clock at maximum AND SoC < 80 °C on both reads: *what
  the code costs*. This is the number to compare code changes and boards with.
- **all-sample median + throttled fraction** — *what the boat feed delivers*.
- Hatched bars / "≥ 80 °C" rows: the model heated the box past 80 °C on its
  very first iterations, so only the all-sample median exists. Where a July
  measurement of the same model exists (open box, ≤ 70 °C) it is within 3 %:
  eWaSR-512 int8 779 vs 766 ms, eWaSR-864 int8 2873 vs 2700 ms.

Each model/combination starts only once the SoC is < 74 °C (cool-down gate),
and a model's timed loop is capped at ~12 s. Validation of the method: the
student ladder reproduces the July unthrottled numbers exactly (512 int8 218 /
221 ms vs 217.5; 640 → 3.02 fps vs 3.05; 864 → 1.67 vs 1.68).

## 1. Every model alone (`images/pi_timing/models_fps.png`)

onnxruntime CPU, **2 intra-op threads** (the deployment setting: the teacher is
slower at 4, the student scales to 5.75 fps at 4 but the other two cores are
the capture service's). Real frame input; medians.

| model | input | file | full-clock ms | fps | note |
|---|---|---|---:|---:|---|
| **thermal seg student v3 s1, fp32** | 160×120 | 12.9 MB | **39.9** | **25.1** | 8× the camera rate; int8 not needed (and fails its accuracy gate) |
| eWaSR teacher 256×192 int8 | 256×192 | 60.8 MB | 212.7 | 4.70 | fastest teacher point; low-res masks |
| **LRASPP student 512×384 int8** | 512×384 | 3.7 MB | **220.7** | **4.53** | deployed streaming tier |
| LRASPP student 640×480 int8 | 640×480 | 3.7 MB | 330.6 | 3.02 | exactly the camera rate, no headroom |
| LRASPP student 512×384 fp32 | 512×384 | 12.9 MB | 343.1 | 2.91 | int8 buys 1.55× at 99.4 % pixel agreement |
| **LRASPP student 864×648 int8** | 864×648 | 3.7 MB | 599.6 | 1.67 | full-res tier (crisper thin structures, same accuracy) |
| YOLOv8n audited, fp32 | 640×640 | 12.3 MB | 675 (run 1: 613) | 1.48 | graph only, NMS excluded |
| eWaSR teacher 512×384 int8 | 512×384 | 60.8 MB | 779 ≥80 °C | 1.28 | July: 766 |
| LRASPP student 864×648 fp32 | 864×648 | 12.9 MB | 1028 ≥80 °C | 0.97 | |
| YOLOv8n audited, fp32 | 864×864 | 12.4 MB | 1222 ≥80 °C (run 1: 1159) | 0.82 | trained size |
| YOLOv8s audited, fp32 | 640×640 | 44.8 MB | 2060 ≥80 °C (run 1: 1932) | 0.49 | the confirm-gate model (§4g) |
| eWaSR teacher 864×648 int8 | 864×648 | 64.1 MB | 2873 ≥80 °C | 0.35 | July: 2700 |
| YOLOv8s audited, fp32 | 864×864 | 44.9 MB | 3978 ≥80 °C (run 1: 3599) | 0.25 | |

YOLO exports: `models/onnx_yolo/` (SSD) from `aime_2026-09-02/yolo_runs/*_audited/
weights/best.pt`, ultralytics 8.4, opset 17, fp32, no NMS; also archived in
`aime_2026-09-02/yolo_onnx/`. No int8 YOLO exists yet (quantising + re-gating it
is the obvious next step if typed evidence ever goes live: the student's int8
gain was 1.55×).

## 2. The fusion pipeline, stage by stage (`stage_budget.png`)

Replay of the dock chunk `12-28-33` (≈65 detections/frame — a *busy* scene;
open water is cheaper), as the shadow service runs it, 60 frames per
configuration, full-clock medians. "tick" = `process_frame`; the scorer runs
after it in `_record`, so **reactive cost per frame = tick + scorer**.

| configuration | fisheye | thermal | radar | fusion/misc | scorer | **per frame** | fps (full clock) | fps on the boat feed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| classical, no scorer | 230 | 6 | 18 | 13 | 0 | **267** | 3.74 | 1.88 (55 % throttled) |
| classical + v1a scorer | 231 | 6 | 18 | 13 | 24 | 292 | 3.43 | 1.87 |
| **classical + v1b scorer** (live) | 230 | 6 | 18 | 11 | 43 | **308** | 3.25 | 1.48 (77 %) |
| classical + v1b, association off | 231 | 6 | 18 | 4 | 43 | 301 | 3.32 | 1.58 |
| + seg student 512 int8 (async) | 331 | 6 | 18 | 13 | 145 | **513** | 1.95 | 1.01 (80 %) |
| + seg student 864 int8 (async) | 466 | 8 | 21 | 27 | 58 | 579 | 1.73 | 1.11 (93 %; 4 full-clock frames) |
| + seg 512 + target motion | 341 | 6 | 17 | 20 | 147 | 530 | 1.89 | 1.10 |
| classical + v1b **beside the capture service** | — | — | — | — | — | 311 | 2.79 (wall 1.26; 92 % throttled) | |
| + seg 512 **beside the capture service** | — | — | — | — | — | 625 all-sample | 100 % throttled; wall 0.91 | |

What the columns mean:

- **fisheye classical 230 ms** is the tick. Inside it, `detect_horizon` (RANSAC
  on Canny edges, 200 iterations, `rng.choice(N, 2, replace=False)` = a full
  permutation of all N edge points per iteration) is ~120 ms on the undistorted
  fisheye (+ ~10 ms on the thermal); glint inpaint, Sobel, blob detection,
  undistortion make up the rest. On a raw edge-dense frame one horizon call
  alone is 241 ms. **Post-freeze fix (changes RANSAC draws → corpus-wide
  horizon shift, so after 09-13): an O(1) pair draw.** Expected tick ≈ 140 ms.
- **thermal 6 ms, radar 18 ms** (incl. self-clutter zone, tracker, Doppler
  velocities), **fusion/association/veto/bins ~11–13 ms** (association itself
  ≈ 7 ms: the "association off" row).
- **scorer**: v1a gated mixture 24 ms, **v1b GBT 43 ms** — and **145 ms once seg
  masks are present**, because `bin_rows_for_frame` computes
  `free_space_profile` from the mask for the scorer's free-space features
  (~100 ms). That is the second post-freeze target: cache/stride the profile
  (the pipeline already computes free space for the record; compute once).
- **with the seg worker alive** the fisheye stage itself rises 230 → 331 ms
  (mask-consuming path + CPU contention with the worker's two threads), and
  the **worker achieves ~1.0 Hz although its inference is 313 ms** (4.5 fps
  alone): the tick, the worker and — on the boat — the throttling share four
  cores. Masks were fresh (< 2 s) on 58/60 frames at 512, 37/60 at 864.
- **target motion** (clusters, tracks, CPA): +7 ms in fusion, ~+17 ms per
  frame overall (the rest is run-to-run noise in the fisheye stage).

## 3. Communications (`latency_chain.png`)

| link | measured |
|---|---|
| capture → sidecar CSVs on disk | **0.10 s median / 0.18 s p95** (per-loop flush; was 12–50 s) |
| tail-mode shadow tick (radar + IMU evidence, v1b) | **66 ms p50 / 112 ms p95**, 0/400 over budget; capture untouched (900/900) |
| sector codec, per record | encode 63–159 µs, fragment 10–30 µs, decode 32–92 µs (clock-dependent) — negligible |
| wire | 46-byte packet, 3 notifications of 20 B per record, 3 Hz → ~9 notifications/s |
| **frame timestamp → record on boat1** | **0.30 s median / 0.37 s p95** (NTP-synced clocks); laptop stand-in 0.45 s |
| transmitter restart → boat1 resubscribed | ~30 s (disconnect → re-scan → forget → connect) |

This is the deployed chain (systemd on both Pis). It carries no camera
evidence — the shadow tails capture's CSVs, and the mp4s have no moov atom
while growing.

## 4. Combinations: baseline and trade-offs

**The honest "everything at once" budget.** The pipeline only *consumes* the
fisheye seg mask live; the thermal student and YOLO have no runtime consumer
yet, so their combination with the rest can only be costed, not measured
end-to-end. Costing at full clock, 2 threads, per camera frame at 3 fps, in
core-seconds per second of wall time (4 cores available, ~2.5 of them after
the capture service and the OS):

| stack | reactive tick / frame | async workers per frame | core-s per s at 3 fps | verdict on a Pi 4 |
|---|---:|---|---:|---|
| A. **deployed**: tail shadow (radar+IMU) + BLE | 66 ms | — | 0.2 | 3 fps with 80 % headroom |
| B. classical cameras + v1b | 308 ms | — | 0.9 | 3 fps at full clock only; 1.3–1.5 fps on the boat feed |
| C. B + seg student 512 int8 | 513 ms | seg 0.44 | 2.9 | measured: 1.9 fps tick, masks at 1 Hz |
| D. B + seg student 864 int8 | 579 ms | seg 1.2 | 5.3 | measured: 1.7 fps tick, masks at 1 Hz, 37/60 fresh |
| E. C + thermal student (in-tick, +40 ms) | ≈ 553 ms | seg 0.44 | 3.0 | affordable — thermal is the cheap model |
| F. E + YOLOv8n 640 (async) | ≈ 553 ms | seg 0.44 + yolo 1.35 | 7.1 | over budget: YOLO would get ≤ 0.5 Hz beside everything else |
| G. "best of everything": seg 864 + thermal + YOLOv8s 864 + v1b | ≈ 619 ms | seg 1.2 + yolo 8.0 | 29 | not a Pi 4 workload (YOLOv8s-864 alone is 0.25 fps) |

Trade-offs that are worth having:

- **seg 512 int8 vs 864 int8**: 2.7× the mask cost for the same accuracy and
  crisper thin structures; both land at ~1 Hz beside the tick today, but 512
  keeps masks fresh (58/60 vs 37/60). **512 stays the streaming tier.**
- **int8 vs fp32 student**: 1.55× for 99.4 % agreement — keep int8 for fisheye.
  For thermal it is moot (25 fps fp32) and int8 fails the gate anyway.
- **v1b vs v1a scorer**: +19 ms/frame for +0.15–0.24 AP on audited night truth
  (§40). Worth it. The 145 ms with masks is a code cost, not a model cost.
- **association**: 7 ms for radar-confirmed bins and the range hierarchy. Keep.
- **target motion**: 17 ms for closing/crossing/CPA. Keep when enabled.
- **YOLO on the box**: v8n-640 at 1.5 fps alone is the only viable candidate
  and only as a low-rate async confirm-gate (the night motor-as-person fix,
  §4g); v8s is offline-only on a Pi 4. An int8 export would be the first thing
  to try (untested).
- **thermal student**: essentially free (40 ms). The missing piece is the
  consumer (thermal water edge / free space in the pipeline), not compute.

## 4b. The learned primaries — MEASURED 2026-09-07 (bench, capture stopped)

`configs/detection_shadow_seg.yaml` (`horizon.lazy` + seg detector + target
motion) with the student 512 int8 worker, and the same plus the async
YOLOv8n-640 worker (`pi_timing_bench` combos "SEG-PRIMARY …"; JSON
`results/pi_timing/pi_timing_2026-09-07_segprimary_v2.json`, dockside clip
`2026-09-03_12-28-33`, 60 frames each, cool-down gate 74 °C, no heatsink).

| combo | tick p50 full clock | p95 | wall rate | masks fresh | throttled frames | SoC | typed worker |
|---|---|---|---|---|---|---|---|
| seg-primary (seg 512, RANSAC-free) | **171 ms** | 182 | **2.26 fps** (= seg 2.26 Hz) | 58/60 | 22 % | 73→79 °C | — |
| seg-primary + YOLOv8n-640 | **236 ms** | 440 | **1.06 fps** (seg 1.05 Hz) | 56/60 | **82 %** | 73→77 °C | 1.17 s/frame, **0.48 Hz** |

Stage medians at full clock (seg-primary): fisheye 114 ms (undistort + seg
contact/free-space, no RANSAC), scorer 144 ms (the `free_space_profile`
recomputation, §5 item 3), radar 18, thermal 6. The first run of the day
measured 298 / 505 ms instead: `horizon.lazy` never fired because
`horizon_from_water` reads confidence ~0.1 on a dockside boundary (piers and
moored boats are not a straight line), so the RANSAC ran twice per frame for
a mask the seg detector never uses. Fixed the same hour: with the seg
detector the RANSAC is skipped whenever a mask exists (whole-frame mask, low
confidence reported so the range reference goes water_edge → IMU → level);
the remaining `horizon` calls (62/60 frames) are the thermal one.

Reading the second row: the typed worker itself is the throttle. Its
1.2 s inference beside the tick pushed the board to 82 % throttled frames and
the under-voltage bit (`0x50000`) even on the bench supply, so the wall rate
is 1.06 fps where the full-clock tick would allow 4. The heatsink and the
supply are the levers; on a Pi 5 the same worker runs at the camera rate.

## 4c. Beside capture on the boat supply — the deployment modes (2026-09-07, no heatsink)

Shadow service logs, 5 min each, capture running at 3 Hz throughout
(`results/pi_timing/shadow_modes_2026-09-07.md`, figure
`images/pi_timing/shadow_modes.png`; records carry the firmware clock,
SoC temperature and throttle flags):

| mode (`shadow_mode.sh`) | sector rate | tick p50 / p95 | ARM clock | throttled | SoC max | mask fresh | typed boxes fresh |
|---|---|---|---|---|---|---|---|
| `tail` (radar+IMU, radar-only bundle) | **3.00 Hz** | 67 / 122 ms | 1800 MHz | never | 69.1 °C | — | — |
| `full` (seg-primary, live OOF bundle) | **1.24 Hz** | 717 / 935 ms | **600 MHz** (392/404 ticks) | `0x50005` | 76.9 °C | **99 %** | — |
| `typed` (seg + YOLOv8n + typed bundle) | 0.57 Hz | 1706 / 1981 ms | 600 MHz | `0x50005` | 76.0 °C | 5 % | 99 % |

Reading it: beside capture on the boat supply the board sits at the 600 MHz
under-voltage clock for the whole run in both camera modes. Seg-primary still
keeps a fresh mask on every tick at 1.24 Hz; the typed worker's 1.2 s
inference (2.4 s at 600 MHz) starves the seg worker past its 2 s freshness
window, so `typed` degrades to radar + boxes at half a hertz. Decision
(AuthorTwo, 2026-09-07): `full` for the closed-loop bursts and static blocks,
`tail` when sailing, `typed` on the Pi 5. The tail source latency in the
report (35 s median) is the chunk-head backlog the tail reads at start; after
it the median is 0.11 s (p95 0.17 s).

## 5. What limits the box, in order

1. **Power.** Every configuration heavier than A spends 50–100 % of its frames
   at 600 MHz on the boat feed: full-clock 3.25 fps becomes 1.3–1.5 fps
   (classical+v1b), 1.95 becomes 1.0 (with seg). Capture itself is unaffected
   (900/900 in every chunk during every run). Measure the rail at pins 2/6
   under a replay bench before any board decision.
2. **Heat.** 73–78 °C idle in the sealed box; heavy models cross 80 °C within
   seconds. A Pi 5 makes both worse without a supply and cooling change.
3. **Two known code costs** (post-freeze for the classical path, byte-identity
   NOT preserved for the first): the horizon RANSAC draw (~120 ms of the
   230 ms fisheye stage; already skipped in seg-primary mode since
   2026-09-07) and the scorer's `free_space_profile` recomputation (~100 ms
   with masks, 144 ms measured in the seg-primary scorer stage).
4. **Pi 5** (A76 @ 2.4 GHz, ~2.5× on this code): B ≈ 120 ms, C ≈ 200 ms — the
   full camera stack at 3 fps with margin, *if* the feed holds ~2× the draw.

## 6. Provenance

Run 2 (`pi_timing_2026-09-03.json`) is the canonical report; run 1 used the
cpufreq probe and is kept as the evidence that it misses under-voltage;
`_beside.json` is the two configurations measured with all three box services
running. Models were staged to `~/ewasr/` (students int8 + fp32, teachers,
thermal v3 s1, `yolo/`), 2026-09-03. History: `docs/history/
2026-09-03_pi_realtime_bench.md`.
