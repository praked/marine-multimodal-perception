# Documentation index

This snapshot ships the reference set: what a thing *is* (formats, protocols,
measurements, hardware timing). Operational runbooks, session logs and plans
are kept out of the snapshot because they name people, places and addresses.

| File | Contents |
|---|---|
| `reference/data_formats.md` | Every on-disk stream the box writes (`fisheye_*.mp4`, `thermal_*.mp4`, `mmwave_*.csv` with Doppler and SNR, `imu_*.csv`, `frames_*.csv` with per-frame exposure, `gps_*.csv`) and every feature table the fusion scorer reads, with the loader that owns it. |
| `reference/sector_protocol.md` | The per-frame sector JSONL contract the navigation layer consumes, the 46-byte Bluetooth LE packet, and the protocol history. |
| `reference/extrinsics.md` | Measured sensor offsets (photogrammetry, ±1 mm) and the remount table. |
| `reference/segmentation.md` | The LaRS/eWaSR water-segmentation layer and the distilled LR-ASPP student: what they are, how they are gated, what they are trusted for. |
| `reference/pi_timing.md` | On-box timing of every model, every pipeline stage and every configuration (Raspberry Pi 4), the two throttles that corrupt naive benchmarks, and the trade-off tables. |

Tool-level documentation lives next to the code: `scripts/gpu_express/README.md`
(GPU staging, training, labelling), `scripts/gpu_corpus/README.md`,
`scripts/gpu_dart/README.md`, `scripts/gpu_seg/README.md`, and
`dashboard/README.md`.
