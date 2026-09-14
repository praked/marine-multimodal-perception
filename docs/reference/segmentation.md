# Water segmentation for monocular range (LaRS / eWaSR)

Maintainer-facing summary of the water-segmentation feature. Design rationale
and the overnight build log live in `docs/lars_segmentation_plan.md` (internal,
gitignored).

## Why

Monocular range (`scripts/utils/geometry.py`,
`scripts/sensor_processing/pipeline.py`) estimates distance by back-projecting a
detection's waterline-contact pixel onto a flat water plane `camera_height_m`
below the camera. It used the **bbox bottom-centre** as that pixel. When the boat
heels (tilted horizon) or the box is loose, the box bottom dips into the water
below the hull, so the object back-projects as **too close**.

A learned **water/sky/obstacle segmentation** fixes this: it yields the true
**water edge** (a robust horizon / up-vector, replacing the tilt-fragile
RANSAC-Canny line) and, per detection, the true **waterline-contact point** (the
lowest obstacle pixel bordering water inside the box).

## Pieces

| Where | What |
|-------|------|
| `scripts/utils/segmentation.py` | mask IO, `water_edge_contact`, `horizon_from_water`, `seg_obstacle_detections`, `SegProvider` |
| `scripts/utils/geometry.py` | `range_from_contact_point` (water-plane range from an explicit contact pixel) |
| `scripts/sensor_processing/pipeline.py` | consumes a per-frame mask (default off) for range up-vector + contact point; optional seg-driven detector |
| `configs/detection.yaml` → `segmentation:` | enable flag + tuning; `fisheye.detector: blob\|segmentation` |
| `scripts/lars/` | stage LaRS into eWaSR's MaSTr format |
| `scripts/gpu_seg/` | train eWaSR on LaRS + predict masks on the InstTwo GPU (see its README) |
| `scripts/eval/seg_range_ablation.py` | measure bbox-bottom vs contact-point range |

Masks live at `data/seg/<scene>__<triplet_ts>/ts=<HH-MM-SS.f>.png` (label-encoded
0=obstacle/1=water/2=sky), in the **undistorted fisheye** coordinate space, keyed
to dashboard `frame_id`s. They are produced on the GPU and pulled back by
`scripts/gpu_seg/06_extract.sh`.

## Enabling it

1. Train + predict + extract masks: see `scripts/gpu_seg/README.md`.
2. Turn on in `configs/detection.yaml`:
   ```yaml
   segmentation:
     enabled: true
   ```
   With `enabled: false` (the default) or no masks present, the fisheye
   range/horizon path is byte-identical to before: all existing tests stay green.
3. Measure the effect:
   ```bash
   python -m scripts.eval.seg_range_ablation --seg-root data/seg \
       --out results/seg_range_ablation.csv
   ```

## Seeing it in the dashboard

`scripts/eval/dashboard.py` auto-detects masks under the seg root (`data/seg`).
When present it builds the pipeline with a `SegProvider` (so the fisheye ranges
shown already use the waterline-contact point) and adds a **"Segmentation"**
toggle to the display panel: on by default when masks exist, disabled when they
don't. The overlay draws, on the fisheye panel:

- a **per-class tint** of the full segmentation (eWaSR palette): **amber =
  obstacle**, **blue = water**, **purple = sky**, so all three classes are
  visible, not just water,
- a **green line** for the water-edge "horizon" (the range up-vector source),
- a **green dot** at each detection's true waterline-contact point (vs the red
  blob marker / bbox bottom it replaces).

Note the obstacle tint is *semantic* (the WaSR "non-water/non-sky = obstacle"
class, static + dynamic); it is not per-object instances or types. Discrete
objects still come from the blob/YOLO detector (red markers), or from the
connected-component seg detector when `fisheye.detector: segmentation` is set.

Masks are matched per frame by `frame_id`; clips/frames without a mask simply
show no overlay (the range path falls back to the bbox bottom). Only the fisheye
panel is covered; LaRS/eWaSR is RGB, so thermal keeps the RANSAC horizon path.

### Typed object detections (LaRS YOLO)

A second, independent layer shows the LaRS-trained YOLO typed detector
(boat/row_boats/paddle_board/buoy/swimmer/animal/float/other). Because there is
no local torch, inference runs on the GPU
(`scripts/gpu_finetune/yolo_predict_frames.py`) over the same undistorted frames,
writing one dashboard-schema JSONL per clip to `data/det/<scene>__<ts>.jsonl`.
The dashboard's `DetProvider` (`scripts/utils/detections.py`) reads them and the
**"Typed detections"** toggle draws per-class coloured boxes with confidences.
This is distinct from the semantic obstacle tint (which has no instances/types)
and from the `--pseudo` labelling-audit flow. Train + predict:
`scripts/gpu_finetune/setup_scratch_yolo.sh` + `train_yolo.py` →
`yolo_predict_frames.py`.

> Caveat (see plan §integration risks): this is an offline/replay visualization.
> The masks are precomputed per-clip on the GPU and frozen to the current
> intrinsics + clip_overrides; there is no on-device runtime yet, and the seg
> range can err in the *unsafe* (farther) direction when the water edge is
> mis-segmented (night/glint). Keep it out of the live nav/heading path until a
> fail-safe range-fusion rule and an on-Pi runtime exist.

## Range up-vector priority

IMU (BNO085) → water-edge segmentation → RANSAC horizon (gated) → level.
The water edge sits just below the IMU because it is far more robust than the
Canny-RANSAC line (which locks onto shorelines/docks).

## Notes / future

- eWaSR is trained on LaRS (pinhole/USV) and applied to our *undistorted* fisheye
  frames (centre well-behaved); validate qualitatively on our clips.
- The non-IMU `ewasr_resnet18` is used (Pi-portable). An IMU-variant (eWaSR's
  horizon-prior mask) is a future improvement now that the BNO085 is coming up.
- ONNX export (`export.py` in eWaSR) + a Pi-4 runtime benchmark are the
  deployment follow-ups; tonight targets offline range accuracy.
- The eWaSR stack needs old pins (PL 1.9.5, setuptools<81, albumentations 1.3.1,
  torch 2.0.1, all-zeros IMU masks); see `scripts/gpu_seg/README.md`.

## Roadmap / direction (agreed 2026-06-29)

Strategy: treat the **3-class segmentation as the primary navigation primitive**
and build outward from it, fusing typed detection + motion later.

**Phase 1: segmentation-driven nav (now, fisheye-only, no new data):**
1. **Free-space layer.** Derive a per-azimuth navigable-distance profile from the
   water mask (water edge → free-space curve) and feed the sector/heading layer
   behind a flag, A/B against the current blob-bin fusion. This is the "where can
   I go" primitive segmentation is genuinely good enough for.
2. **Demote RANSAC to fallback.** Make the seg water-edge the primary range
   up-vector / horizon; keep RANSAC (and level) only as the no-mask / low-water
   fallback. Effectively "use only segmentation" when a mask exists.
3. **Quantify the win we *can* measure now.** Seg water-edge vs RANSAC:
   frame-to-frame jitter, tilt-robustness, % frames RANSAC was rejected/wrong.
   NOTE: absolute *range accuracy* can't be validated without ground truth
   (radar pending, or a surveyed target), so Phase 1 proves *horizon stability*,
   not absolute metres.
4. **Temporal smoothing** on the water edge / mask (per-frame seg flickers).

**Phase 1b: Pi-4 runtime (now actionable: Pi in hand):** ONNX-export eWaSR,
benchmark on the Pi 4, then quantise / distil / reduce cadence until it runs at a
useful rate. This gates the whole "segmentation streaming for nav" thesis;
do it in parallel with Phase 1. Iterate finetune↔quantise↔benchmark on hardware.

**Phase 2: fuse type + motion (later):** combine the free-space layer with the
typed detector (and instance masks if fine-tuned up) to classify
stationary-vs-moving and apply per-type kinematic priors (sailboat vs row boat vs
swimmer → plausible speed/heading) for prediction and COLREG-aware path planning.
Dependencies: (a) an **on-domain fine-tune** of the typed detector: LaRS-domain
misses our scenes (e.g. moored boats against shore), and (b) a **velocity
source**: radar Doppler (pending) is natural; vision tracker + per-frame range
is a noisier stopgap.

**Blocked on incoming data (don't over-invest yet):** absolute range accuracy
(radar / survey), velocity / stationary-vs-moving at quality (radar Doppler),
night & low-vis (thermal; seg is RGB-only).

**Verdict carried forward:** segmentation = the dependable, navigation-grade
layer; typed detectors = visualization until an on-domain fine-tune; keep the
seg range OUT of the live nav path until there's a Pi runtime + a fail-safe
range-fusion rule (the error currently points the unsafe/farther way).

**Why this is low-risk (option value).** Once radar + thermal are fused, this
vision-heavy layer may prove overkill for nav; that's fine. The
GroundingDINO-distilled **YOLOv8n deployment detector is already saved and
Pi-4-runnable** (`models/`, see fisheye_context §"YOLO fine-tune"), so it stays
the proven fallback. Pushing the segmentation/instance direction costs us nothing
to revert and yields a stronger nav primitive if it pans out: an asymmetric bet.

## Published models (HuggingFace · hf-handle)

Collection: **https://huggingface.co/collections/hf-handle/asvproject-obstacle-detection-6a42591fe0f1a1866eb92083**

| Model | Repo | Notes |
|---|---|---|
| eWaSR (LaRS) | `hf-handle/asvproject-ewasr-lars` | water/sky/obstacle segmentation |
| YOLOv8n (InstitutionOne) | `hf-handle/asvproject-yolov8n-institutionone` | Pi deployment detector (F1 0.673) |
| YOLOv8s-seg (LaRS) | `hf-handle/asvproject-yolov8s-lars-seg` | typed instance masks |
| GroundingDINO labeller | `hf-handle/asvproject-groundingdino-labeler` | config-only teacher recipe (F1 0.701) |
| LRASPP student (distilled) | `hf-handle/asvproject-lraspp-student` | real-time Pi-4 seg (4.6 fps @512×384, 1.7 fps @864×648 full-res, 3 MB int8); 512-trained + 864-trained variants, fp32 + int8 ONNX + weights + training logs (2026-07-07) |
| LRASPP **thermal** student (cross-modal) | `hf-handle/asvproject-lraspp-thermal` | Lepton 160×120 water/sky/obstacle; `v1_day_2026-08-22/` = the gpu-node JOINT ladder (3 seeds, pinned holdout 0.908 ± 0.001, fp32 pth + ONNX; int8 fails the gate); `v2_night_2026-08-28/` (0.9135 ± 0.0003) and `v3_night_2026-09-02/` (0.9148 ± 0.0001; seed 1 = the box's fp32 ONNX) |
| Fusion scorer v1a | `hf-handle/asvproject-fusion-scorer-v1a` | the live calibrated gated-mixture artefact (`d2_base`, 15° bins, AP 0.982 vs incumbent 0.945 on the saturated pre-audit corpus) + standalone `models.py`, `fusion_model.yaml`, bake-off report (2026-08-28) |
| Fusion scorer v1b (LIVE) | `hf-handle/asvproject-fusion-scorer-v1b` | the three deployed GBT bundles (OOF-calibrated full, radar-only tail, typed) + `models.py`, `fusion_model.yaml`, ablation tables; audited_v2 holdout AP 0.869/0.811/0.890 vs incumbent 0.673 (2026-09-09) |
| YOLO on-domain fine-tunes | `hf-handle/asvproject-yolo-audited` | v8n/v8s/11n/11s on the audited frames (v8n mAP50 0.454, v8s 0.462) + the leak-free v8n (0.428 on the held-out recordings) behind the typed-evidence channel, + the ONNX exports the box runs (2026-09-09) |

Staging + upload script: `models/hf_staging/publish.sh` (needs `hf auth login` or `HF_TOKEN`; excludes `__pycache__`).
LaRS-derived cards (eWaSR, YOLOv8s-seg, LRASPP student, thermal student) carry the LaRS
citation (Žust et al., ICCV 2023). YOLO repos are AGPL-3.0; GroundingDINO
base is Apache-2.0; the student's torchvision backbone is BSD-3. Public:
confirm LaRS terms permit redistribution of the LaRS-derived weight repos.

## Phase 1: delivered (2026-06-29)

- **Range fix measured:** seg waterline-contact vs bbox-bottom on YOLO boat
  boxes = **+2.16 m mean / +2.14 m median farther, 83% of detections**
  (`scripts/eval/range_ab_boats.py`).
- **Navigable free-space:** `segmentation.free_space_profile` (per-bearing
  distance, RANSAC-free) + `FreeSpaceSmoother` (EMA). Feeds the heading
  recommender behind `heading.use_free_space`; **live "Free-space heading"
  dashboard checkbox** for one-click A/B. Viz: `scripts/eval/freespace_viz.py`.
- **FP filter:** `segmentation.fp_filter` drops water-only detections (glint/night FPs).
- **Pi runtime:** eWaSR → ONNX (fixed-size) → static INT8 = **229→57 MB,
  99.7%/99.4% pixel-identical to fp32**. `scripts/gpu_seg/{export_onnx,
  quantize_onnx,pi_benchmark,pi_predict}.py`; runbook: `docs/guides/pi_model_runtime.md`.
- **Full annotation:** 10,731 masks across 12 InstitutionOne clips (`12:41–13:41`).

All Phase-1 additions are config-gated default-off; full suite green. Progress
report for the project lead: `docs/progress_report_2026-06-29.pdf` (gitignored).
