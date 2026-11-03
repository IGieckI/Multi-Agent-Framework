#!/bin/bash
# One-time setup on the giano.cs.unibo.it hpc, that moves stuff from home to scratch folder
set -euo pipefail
GRID="${1:?usage: bash scripts/slurm/setup_giano.sh <grid.yaml>}"
cd "$(dirname "$0")/../.."
[ -f "${GRID}" ] || { echo "no grid ${GRID} (a path from the repo root, e.g. configs/experiments/pilot.yaml)"; exit 1; }
source scripts/slurm/env.sh
mkdir -p "${MAS_SCRATCH}" "${HF_HOME}" "${XDG_CACHE_HOME}"

if [ ! -x "${MAS_VENV}/bin/python" ]; then
  if command -v python3.11 >/dev/null; then
    python3.11 -m venv "${MAS_VENV}"
  else
    python3 -m venv "${MAS_SCRATCH}/venvs/uv"
    "${MAS_SCRATCH}/venvs/uv/bin/pip" install uv
    "${MAS_SCRATCH}/venvs/uv/bin/uv" venv --seed --python 3.11 "${MAS_VENV}"
  fi
fi
PY="${MAS_VENV}/bin/python"
"${PY}" -m pip install --upgrade pip
"${PY}" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
"${PY}" -m pip install -e ".[text]"

[ -f data/alchemy2.json ] || "${PY}" scripts/fetch_data.py
"${PY}" scripts/download_models.py "${GRID}"
"${PY}" scripts/sweep.py plan "${GRID}"
echo "space used: $(du -sh "${MAS_SCRATCH}" | cut -f1) in ${MAS_SCRATCH}; home: $(du -sh ~ 2>/dev/null | cut -f1) of 400 MB"
