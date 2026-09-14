# Documentation index

Four kinds of document, in four directories. The distinction is about
**maintenance**, not topic:

| | | |
|---|---|---|
| **`guides/`** | living | How to do a thing. Kept current; edit these when reality changes. |
| **`reference/`** | living | What a thing *is*: formats, protocols, measurements, hardware. |
| **`plans/`** | forward-looking | Agreed direction, not yet built. Superseded when built. |
| **`history/`** | frozen | Dated session logs and measurements. **Never edited**; they record what was true on a date. |

Two documents sit at the top level because they are read on their own, as PDFs:

- **`quick_card.pdf`**: two pages to run capture, the trial stack and the avoidance experiment (source `quick_card.tex`, `latexmk -pdf docs/quick_card.tex`).
- **`field_reference.pdf`**: the operational cheat-sheet. Every command for
  connecting, capturing, testing, syncing, replaying and labelling, each with
  the files it writes. **Current Pi addresses live here.** Source:
  `field_reference.tex` (`latexmk -pdf docs/field_reference.tex`).
- **`annotation_manual.pdf`**: labelling conventions, class taxonomy, and the
  dashboard/audit workflow. Read before labelling anything.

---

## Start here

| I want to… | Read |
|---|---|
| **Pick the project up for the first time** | `guides/start_here.md` (then `plans/handover_2026-09-09.md`) |
| Set up a Pi from a blank card | `guides/pi_setup.md` |
| Reassemble and bring up the box | `guides/box_reassembly.md` |
| Image a box for safekeeping, or swap its compute board | `guides/box_duplication.md` |
| Run a pontoon session | `guides/pontoon_runbook.md` |
| Finish the calibration afloat | `guides/afloat_runbook.md` |
| Prove the closed-loop chain on land | `guides/land_test_runbook.md` |
| Run the closed-loop trials on the lake | `guides/closed_loop_water_runbook.md` |
| Run the label-audit sprint | `guides/audit_sprint.md` |
| Run a capture day | `guides/capture_runbook.md` + `field_reference.pdf` |
| Capture with the fisheye only | `guides/capture_runbook.md` §Fisheye-only capture |
| Get own-boat GPS working on the box | `guides/box_reassembly.md` §5 (GPS) + `configs/gps.yaml` |
| Understand a file the box wrote | `reference/data_formats.md` |
| Run models on the Pi | `guides/pi_model_runtime.md` |
| Train something on the GPU | `guides/gpu_nodes.md` |
| Consume the fusion output | `reference/sector_protocol.md` |
| Debug a power symptom on the box | `reference/power_and_supply.md` |
| Find the captured footage | `reference/data_storage.md` |

---

## `guides/`: living how-to

| File | Contents |
|---|---|
| `start_here.md` | **Entry point for the InstitutionOne team**: run a capture, a closed-loop launch, an ingest and a retrain; where data, models and labels live; the credentials-and-ownership table; and the ten things that fail silently. |
| `pi_setup.md` | Blank card → capturing box: imager settings, `config.txt`, external RTC, software stack, network, verification, on-Pi file tree, rebuild checklist. Automated by `deploy/pi_setup.sh`. |
| `afloat_runbook.md` | The one session that closes the remaining calibration: camera height, the canoe pass for thermal fx, the rocking clip for IMU pitch, and the clutterRemoval A/B, each with its analysis step. |
| `land_test_runbook.md` | The trolley session that proves box → boat1 → rudder before the water: link and evidence check, heading sign, decision repeatability, rudder-follow, abort, optional typed burst, and the log pull afterwards. |
| `closed_loop_water_runbook.md` | The lake trials that fill Table IV: pre-departure checks incl. the starboard radar return, one launch with several trials driven by the RC switch (MANUAL to reposition with real throttle, AUTO to hold/decide/go), the rules on the water, and the evening log pull. |
| `pontoon_runbook.md` | Offline, solo procedure for a pontoon trip: pre-departure checks, the IMU-horizon and radar exercises with commands, the shutdown rule, and what such a session cannot close. |
| `box_reassembly.md` | Putting the sensor box back together: power harness check, full Pi-4 pinout, assembly order, bring-up verification, and the calibration sequence that follows. |
| `box_duplication.md` | Three separable operations: **A** take a golden image of the working box (do before a campaign), **B** swap its compute Pi 4 → Pi 5 (deferred until after), **C** build a second fisheye-only box. Covers the dual-board `config.txt`, passwords, re-identifying a clone, and the fingerprint diff that makes "nothing changed" checkable. |
| `capture_runbook.md` | Capture-day pre-flight gates, sensor-enable flags (incl. fisheye-only), radar A/B protocol, thermal wet-cover check, troubleshooting. |
| `pi_model_runtime.md` | Deploying and running the segmentation models on the Pi: which model, how many threads, benchmark/predict/two-tier worker. |
| `gpu_nodes.md` | InstTwo GPU access, the numbered sync/train/extract pipelines, the AFS home-quota rule, extraction discipline. |
| `qualitative_pass.md` | The human-eye review pass over detector output (+ `qualitative_template.csv`). |

## `reference/`: stable specs

| File | Contents |
|---|---|
| `data_storage.md` | Where field captures are archived (the external `ROS2_SSD` volume), what is in each session directory, and what is still on the Pi. |
| `data_formats.md` | Every on-disk stream the box writes and every feature table the fusion scorer reads, with the loader that owns it. |
| `sector_protocol.md` | The per-frame sector JSONL contract the navigation layer consumes. |
| `extrinsics.md` | Measured sensor offsets (photogrammetry, ±1 mm) and the remount table. |
| `imu_uart_rvc.md` | BNO085 wiring, UART-RVC protocol, axis validation, and the SPI post-mortem. |
| `power_and_supply.md` | The boat-power blocker: measurements, root cause (reversed polarity), the fix, and the fisheye-only low-draw mode. |
| `segmentation.md` | The LaRS/eWaSR water-segmentation layer and the distilled student: what they are, how they are gated, what they are trusted for. |
| `pi_timing.md` | On-box timing of every model, every pipeline stage and every configuration (Pi 4, 2026-09-03), the two throttles that corrupt naive benchmarks, and the trade-off/budget tables. |

## `plans/`: agreed but not built

| File | Contents |
|---|---|
| `fusion_scoring_training.md` | The learned per-bin fusion scorer: feature design, bake-off, phases. Phases 0–2 are shipped; Phase 3 is data-gated. |
| `capture_status_and_prereqs.md` | State of the capture stack + the ordered prerequisites for the training pipeline. |
| `dataset_reframing.md` | Proposal to reframe the corpus as a multi-platform robotics dataset (for AuthorOne's review). |
| `handover_2026-09-09.md` | Last InstitutionOne day timeline, 10–15 Sep milestones, the four-agent split with prompts, credentials table. |
| `boatv1_sector_experiment_pr.md` | PR description for the gh-handle-one/boatv1 `sector-experiment` branch. |
| `d2_protocol_decisions.md` | D.2 autopilot-handoff decision list for the conversation with AuthorOne: bin width, value semantics, thresholds, severity table. |
| `campaign_checklist_2026-09-07.md` | Everything still open to the ICRA deadline, phase by phase in gating order (laptop → box bench → boat1 dock → lake → annotation → retrain → paper) with the gate each phase must pass; mirrors the interactive checklist page. |
| `teacher_comparison.md` | DART/SAM3 vs GroundingDINO as labelling teacher: audit protocol, tranche-A verdict (DART wins), costed failure modes, prompt-v1 design. |
| `swarm_self_recognition.md` | Branch D.1: deciding whether a detected vessel is a ASVProject boat (cooperative) or a COLREG stranger. Agreed direction 2026-08-25, nothing implemented; gates on AuthorOne for physical changes. |

## `history/`: frozen session logs

Chronological record of what was measured, when, and on what hardware. Cite
these; don't edit them. Index: `history/README.md`.

---

## Other writeups

Longer LaTeX narratives (`fisheye_context`, `radar_diagnosis`,
`thermal_water_problem`, `fusion_pipeline_overview`, `progress_report_*`) build
to PDFs alongside their sources. Most are gitignored; they are sendable
writeups for the project lead rather than repository documentation. Build any of
them with `latexmk -pdf docs/<name>.tex`.
