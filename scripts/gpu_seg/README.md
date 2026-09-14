# InstTwo-GPU eWaSR water-segmentation pipeline

Train **eWaSR** (`tersekmatija/eWaSR`, ResNet-18, 3-class water/sky/obstacle) on
**LaRS**, then predict masks for our undistorted fisheye clips so the local
range pipeline can use the true **waterline-contact point** instead of the bbox
bottom (the tilted-horizon "too close" bug). Full rationale:
`docs/lars_segmentation_plan.md` (internal, not distributed with the repo);
the maintainer-facing summary is `docs/reference/segmentation.md`.

Mirrors `scripts/gpu_labeling/`: a sourced `00_run_params.sh` drives numbered
steps. Code lives in the small AFS home; the venv, dataset, weights and outputs
all live on `/scratch0` (wiped when the booking ends: **extract first**).

## One-time / per-booking

`00_run_params.sh` is set for node **gpu-node** (`GPU_NODE`), user `user`,
jump host `gpu-node`, key `~/.ssh/cluster_key`. When the booking moves to another
node, set `GPU_NODE` (or override in a gitignored `00_run_params.local.sh`).
SSH uses the shared-AFS key (already authorized); the GPU login shell is tcsh,
so every remote step pipes a bash script to `bash -s`.

## Run order

```bash
cd scripts/gpu_seg
bash 01_sync_to_gpu.sh     # build staged inputs locally, rsync code+data to node
bash 02_preflight.sh       # nvidia-smi, scratch space, staged-input sanity
bash 03_setup_gpu.sh       # venv + torch(cu118) + eWaSR deps, clone eWaSR, pretrained
bash 04_train.sh           # launch eWaSR LaRS training (nohup); poll: 04_train.sh --status
bash 05_predict.sh         # predict clean label masks for every uploaded clip
bash 06_extract.sh         # pull masks -> data/seg/, weights -> models/  (DO THIS BEFORE BOOKING ENDS)
```

`01_sync` runs `scripts.lars.prepare` (LaRS -> resized MaSTr tree, ~100 MB) and
`scripts.eval.export_undistorted_frames` (the clips to segment) locally if their
outputs are missing, then rsyncs. `05_predict` auto-picks weights:
explicit `WEIGHTS=...` > LaRS-trained `weights.pth` > pretrained MaSTr.

After extract, enable segmentation locally:

```yaml
# configs/detection.yaml
segmentation:
  enabled: true        # masks now in data/seg/
```

then quantify the fix:

```bash
python -m scripts.eval.seg_range_ablation --seg-root data/seg \
    --out results/seg_range_ablation.csv
```

## eWaSR version pins (why)

eWaSR (2023) needs an old stack; see `requirements-gpu-seg.txt` and
`setup_scratch_seg.sh`:

- `pytorch-lightning==1.9.5` (uses `pl.Trainer(gpus=)`, removed in PL 2.0);
  torch `2.0.1+cu118` to match.
- `setuptools<81` (PL 1.9's `lightning_fabric` calls
  `pkg_resources.declare_namespace`, gone in setuptools 81; `uv venv` seeds none).
- `albumentations==1.3.1` (>=1.4 `np.stack`s the heterogeneous masks list and
  fails: eWaSR passes a one-hot seg mask + an imu mask together).
- **No** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (torch 2.1+ only).
- eWaSR's `WaSR.forward` reads `imu_mask` unconditionally even for the non-IMU
  model, so `prepare.py` ships all-zeros IMU masks and `predict_masks.py` feeds a
  zeros tensor (the SIM block ignores them when `imu=False`).
