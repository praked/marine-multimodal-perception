#!/usr/bin/env bash
# Runs ON the GPU node (pipe via `bash -s`). Builds the corpus-inference venv
# on /scratch0 with the pinned cu126 torch stack (requirements-gpu.txt rule:
# torch<2.8 — unpinned resolves pull cu130 wheels that fall back to CPU).
#
# HOME-QUOTA RULE: AFS home holds code only; every cache (pip/torch/HF/XDG)
# is exported to scratch and pip runs --no-cache-dir.
set -euo pipefail

S=${S:-/scratch0/$USER}
VENV=$S/corpus_venv
CACHE=$S/.cache
mkdir -p "$CACHE/pip" "$CACHE/torch" "$CACHE/hf" "$S/logs"

export XDG_CACHE_HOME="$CACHE"
export PIP_CACHE_DIR="$CACHE/pip"
export TORCH_HOME="$CACHE/torch"
export HF_HOME="$CACHE/hf"
export TMPDIR="$S/tmp"
mkdir -p "$TMPDIR"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --no-cache-dir -q --upgrade pip wheel

pip install --no-cache-dir -q \
  "torch==2.7.1" "torchvision==0.22.1" \
  --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

pip install --no-cache-dir -q \
  ultralytics transformers opencv-python-headless numpy pandas pyyaml \
  tqdm pillow "huggingface_hub[cli]" onnx   # onnx: train_student's final export imports it lazily (2026-08-28 seed 0 lost its ONNX without it)

python - <<'PY'
import torch, torchvision, transformers, ultralytics
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("torchvision", torchvision.__version__,
      "| transformers", transformers.__version__,
      "| ultralytics", ultralytics.__version__)
PY

# Activation shim for later stages (nohup shells re-source this).
cat > "$S/activate_corpus.sh" <<SHIM
export XDG_CACHE_HOME="$CACHE"
export PIP_CACHE_DIR="$CACHE/pip"
export TORCH_HOME="$CACHE/torch"
export HF_HOME="$CACHE/hf"
export TMPDIR="$S/tmp"
export PYTHONUNBUFFERED=1
source "$VENV/bin/activate"
SHIM
echo "SETUP OK"
