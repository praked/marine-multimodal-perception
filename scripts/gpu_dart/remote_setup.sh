#!/usr/bin/env bash
# Runs ON the GPU node (piped via `bash -s`, env from run_dart.sh). Idempotent.
# Builds: uv-managed Python, $S/dart_venv (torch cu126 + DART -e .), $S/DART.
# Caches on scratch; nothing but code touches AFS home.
set -uo pipefail
S=${DART_S:-/scratch0/$USER}
export XDG_CACHE_HOME=$S/.cache PIP_CACHE_DIR=$S/.cache/pip TORCH_HOME=$S/.cache/torch \
       HF_HOME=$S/.cache/huggingface TMPDIR=$S/tmp UV_CACHE_DIR=$S/.cache/uv \
       UV_INSTALL_DIR=$S/bin UV_PYTHON_INSTALL_DIR=$S/.cache/uv/python
mkdir -p $S/logs $S/tmp $S/bin $S/.cache/pip $S/corpus_out $S/native_frames $S/asvproject_models/dart $S/asvproject_raw/captures
log() { echo "$(date +%H:%M:%S) $*"; }

log "[1] uv + python ${DART_PYTHON_VERSION}"
[[ -x $S/bin/uv ]] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
PY=$($S/bin/uv python find "${DART_PYTHON_VERSION}" 2>/dev/null || true)
[[ -x "$PY" ]] || { $S/bin/uv python install "${DART_PYTHON_VERSION}" >/dev/null 2>&1; PY=$($S/bin/uv python find "${DART_PYTHON_VERSION}"); }
[[ -x "$PY" ]] || { log "FAILED: no python ${DART_PYTHON_VERSION} (uv)"; exit 1; }
$PY --version

log "[2] DART clone${DART_COMMIT:+ @ $DART_COMMIT}"
if [[ ! -d $S/DART/.git ]]; then git clone -q "${DART_REPO_URL}" $S/DART || { log "FAILED: clone"; exit 1; }; fi
if [[ -n "${DART_COMMIT:-}" ]]; then (cd $S/DART && git fetch -q --depth 1 origin "$DART_COMMIT" 2>/dev/null; git checkout -q "$DART_COMMIT") || { log "FAILED: checkout $DART_COMMIT"; exit 1; }; fi
(cd $S/DART && git rev-parse --short HEAD)

log "[3] venv"
[[ -x $S/dart_venv/bin/python ]] || $PY -m venv $S/dart_venv
source $S/dart_venv/bin/activate
pip install --no-cache-dir -q --upgrade pip wheel
if ! python -c "import torch, sam3" 2>/dev/null; then
  pip install --no-cache-dir -q ${DART_TORCH_SPEC} --index-url "${DART_TORCH_INDEX}" || { log "FAILED: torch"; exit 1; }
  (cd $S/DART && pip install --no-cache-dir -q -e .) || { log "FAILED: DART -e ."; exit 1; }
  pip install --no-cache-dir -q opencv-python-headless pandas pyyaml "numpy<2" huggingface_hub[cli] || { log "FAILED: extras"; exit 1; }
fi
python -c "import torch, torchvision, numpy, sam3; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU', 'np', numpy.__version__)" || { log "FAILED: import check"; exit 1; }

log "[4] weights"
CK=$S/asvproject_models/dart/${SAM3_FILE}
if [[ -s $CK ]]; then
  if [[ -n "${SAM3_MD5:-}" ]]; then got=$(md5sum $CK | cut -c1-32); [[ "$got" == "$SAM3_MD5" ]] && log "checkpoint md5 OK" || log "WARN checkpoint md5 $got != $SAM3_MD5"; fi
  log "checkpoint present: $(du -h $CK | cut -f1)"
else
  log "checkpoint MISSING: $CK  (gated ${SAM3_REPO} — run_dart.sh --prepare handles it)"
fi
du -sh $S/dart_venv $HOME 2>/dev/null | sed 's/^/  /'
log "SETUP-DART DONE"
