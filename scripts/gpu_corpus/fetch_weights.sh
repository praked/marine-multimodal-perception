#!/usr/bin/env bash
# Runs ON the GPU node. Pulls the published ASVProject model weights from
# HuggingFace (hf-handle collection) into /scratch0 — no SSD needed.
set -euo pipefail

S=${S:-/scratch0/$USER}
export S
source "$S/activate_corpus.sh"
M=$S/asvproject_models
mkdir -p "$M"

python - <<'PY'
from huggingface_hub import snapshot_download
repos = {
    "lraspp-student": "hf-handle/asvproject-lraspp-student",
    "yolov8n-institutionone": "hf-handle/asvproject-yolov8n-institutionone",
    "yolov8s-lars-seg": "hf-handle/asvproject-yolov8s-lars-seg",
    "ewasr-lars": "hf-handle/asvproject-ewasr-lars",
}
import os
base = os.environ.get("S", os.path.expandvars("/scratch0/$USER")) + "/asvproject_models"
for name, repo in repos.items():
    path = snapshot_download(repo_id=repo, local_dir=f"{base}/{name}")
    print(name, "->", path)
PY

echo "--- inventory ---"
find "$M" -name '*.pth' -o -name '*.pt' -o -name '*.onnx' | sort
echo "FETCH OK"
