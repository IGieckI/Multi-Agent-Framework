# Env variables to move stuff from the home directory to the scratch directory (which has more space)
export MAS_SCRATCH="${MAS_SCRATCH:-/scratch.hpc/${USER%@*}}"
export MAS_VENV="${MAS_VENV:-${MAS_SCRATCH}/venvs/mas}"
export HF_HOME="${HF_HOME:-${MAS_SCRATCH}/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${MAS_SCRATCH}/cache}"
export PIP_NO_CACHE_DIR=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-${MAS_SCRATCH}/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${MAS_SCRATCH}/python}"
