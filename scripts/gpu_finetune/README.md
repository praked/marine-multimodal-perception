# GPU fine-tune pipeline (YOLO detector)

Fine-tune a small YOLO detector on the audited training set, on a InstTwo GPU node
(e.g. gpu-node). Mirrors `gpu_labeling/`: **only scripts live in the NFS home;
everything heavy (venv, caches, model downloads, dataset, outputs) lives on
`/scratch0`**: the home quota is small and easy to overrun.

## One-time / per-node

```bash
# from the repo on your laptop: copy the scripts to the node home
rsync -a -e "ssh -i ~/.ssh/cluster_key -J user@gpu-node.cluster.example.org" \
    scripts/gpu_finetune/ user@gpu-node.cluster.example.org:asvproject-gpu-finetune/
```

`env.sh` creates the scratch venv on first source and redirects all caches
(`XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `TORCH_HOME`, `HF_HOME`, `YOLO_CONFIG_DIR`,
`TMPDIR`) to `/scratch0/$USER`. Install once:

```bash
ssh ... user@gpu-node.cluster.example.org
bash                              # nodes default to tcsh
source ~/asvproject-gpu-finetune/env.sh
pip install -q ultralytics       # torch CUDA + ultralytics, all onto scratch
```

## Each iteration (after more labelling)

```bash
# 1. export the audited labels -> portable YOLO dataset (undistorted frames)
python -m scripts.eval.export_yolo \
    --labels labels/training_frames.jsonl --out data/yolo_finetune
#    (val = every 5th frame; --val-every N to change)

# 2. sync the dataset to scratch (delete stale files)
rsync -a --delete -e "ssh -i ~/.ssh/cluster_key -J user@gpu-node.cluster.example.org" \
    data/yolo_finetune/ user@gpu-node.cluster.example.org:/scratch/user/yolo_finetune/

# 3. train (on the node)
source ~/asvproject-gpu-finetune/env.sh
python ~/asvproject-gpu-finetune/train_yolo.py \
    --data /scratch/user/yolo_finetune/data.yaml \
    --project /scratch/user/yolo_runs --name ft --device 0

# 4. pull results back to look at (results/ is gitignored)
rsync -a -e "ssh ..." \
    user@gpu-node.cluster.example.org:/scratch/user/yolo_runs/ft/ results/yolo_ft/
```

## Caveat on the metrics

The export's val split is 1-in-5 of the *same* clips, so frames are highly
correlated with training (nearby timestamps, same scenes); mAP is **optimistic**.
The real generalisation test is the held-out `eval_smoke36` clip; wire that in as
the val set once the training pool is large enough to spare it. 50 frames is a
smoke test, not a deployable model.
