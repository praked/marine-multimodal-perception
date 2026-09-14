#!/usr/bin/env bash
# Runs ON THE GPU NODE. Builds a scratch venv with torch (cu118) + ultralytics
# for YOLO training (train_yolo.py). Caches on scratch. Idempotent.
#
#   YOLO_SCRATCH=/scratch0/$USER/asvproject_yolo \
#     bash ~/asvproject_project/ASVProject-ObstacleDetection/scripts/gpu_finetune/setup_scratch_yolo.sh
set -euo pipefail

YOLO_SCRATCH="${YOLO_SCRATCH:-/scratch0/${USER}/asvproject_yolo}"
VENV_DIR="${YOLO_SCRATCH}/yolo_venv"
CACHE_DIR="${YOLO_SCRATCH}/.cache"
SHIM="${YOLO_SCRATCH}/activate_yolo.sh"

echo "Scratch : ${YOLO_SCRATCH}"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
# Keep ultralytics' settings/datasets dir on scratch (not the tiny AFS home).
export YOLO_CONFIG_DIR="${CACHE_DIR}/ultralytics"
mkdir -p "${VENV_DIR}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}" "${TORCH_HOME}" \
         "${XDG_CACHE_HOME}" "${YOLO_CONFIG_DIR}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "ERROR: nvidia-smi failed"; exit 1; }

if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | \
    env INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR="${YOLO_SCRATCH}/bin" sh
  export PATH="${YOLO_SCRATCH}/bin:${PATH}"
fi

if command -v uv &>/dev/null; then
  [[ -x "${VENV_DIR}/bin/python" ]] || uv venv "${VENV_DIR}" --python 3.11
  source "${VENV_DIR}/bin/activate"
  uv pip install --upgrade pip wheel
  uv pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
  uv pip install ultralytics
  # ultralytics pulls numpy 2.x, but torch 2.0.1 was built against numpy 1.x
  # ("RuntimeError: Numpy is not available" in torch.from_numpy). Hold it back.
  uv pip install "numpy<2"
else
  [[ -x "${VENV_DIR}/bin/python" ]] || python3 -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
  pip install --upgrade pip wheel
  pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
  pip install ultralytics
  pip install "numpy<2"
fi

python - <<'PY'
import torch, ultralytics
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("ultralytics:", ultralytics.__version__)
PY

cat > "${SHIM}" <<EOF
#!/usr/bin/env bash
export PATH="${YOLO_SCRATCH}/bin:\$PATH"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export YOLO_CONFIG_DIR="${CACHE_DIR}/ultralytics"
source "${VENV_DIR}/bin/activate"
echo "YOLO env activated (venv: ${VENV_DIR})."
EOF
chmod +x "${SHIM}"
echo "Setup complete. Shim: ${SHIM}"
