#!/usr/bin/env bash
# One-command retrain of the /crops boat classifier on the CURRENT labels,
# then republish predictions (the page picks them up on refresh — no deploy).
# Usage:  bash dashboard/tools/swarm_crops/retrain.sh    (from the repo root)
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root
EMB=/Volumes/ROS2_SSD/asvproject/swarm_crops/2026-08-26/clip_vitl14_embeds.npz
[ -f "$EMB" ] || { echo "SSD not mounted (need $EMB)"; exit 1; }
python3 dashboard/tools/swarm_crops/train_classifier.py --embeddings "$EMB"
( cd dashboard && node --env-file=.env.local tools/swarm_crops/upload_predictions.mjs )
echo "RETRAIN COMPLETE — refresh /crops to get the new queue."
