#!/usr/bin/env bash
# ASVProject GPU fine-tune environment for InstTwo nodes (e.g. gpu-node).
#
# The NFS home (/cs/student/.../$USER) is quota-limited and easy to overrun;
# /scratch0 is large and local. This script redirects EVERYTHING heavy --- the
# venv, pip/torch/HF/XDG caches, tmp, model downloads --- to scratch, so only
# scripts ever live in home. Source it (with bash) before any GPU work:
#
#     bash                       # nodes default to tcsh; drop into bash first
#     source env.sh
#
# Override the scratch root with ASVPROJECT_SCRATCH if /scratch0 differs.
set -u

SCRATCH="${ASVPROJECT_SCRATCH:-/scratch0/$USER}"
export ASVPROJECT_SCRATCH="$SCRATCH"

# All caches -> scratch (XDG covers pip build, matplotlib, fontconfig, ...).
export XDG_CACHE_HOME="$SCRATCH/.cache"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export TMPDIR="$SCRATCH/tmp"
export TORCH_HOME="$SCRATCH/.cache/torch"
export HF_HOME="$SCRATCH/.cache/huggingface"
export YOLO_CONFIG_DIR="$SCRATCH/.config/Ultralytics"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$TORCH_HOME" "$HF_HOME" "$YOLO_CONFIG_DIR"

# Venv on scratch (torch is ~2 GB: must not live in home).
VENV="$SCRATCH/yolovenv"
if [ ! -x "$VENV/bin/python" ]; then   # python is executable; activate is only sourced
    echo "[env] creating venv at $VENV"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Pretrained weights download to the CWD; keep that on scratch too.
cd "$SCRATCH" 2>/dev/null || true

echo "[env] scratch=$SCRATCH  venv=$VENV  $(python --version 2>&1)"
echo "[env] caches -> $SCRATCH/.cache ; tmp -> $TMPDIR"
