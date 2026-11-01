#!/bin/bash
# Runs an experiment grid as a Slurm job array
# For evaluation: python scripts/sweep.py evaluate configs/experiments/pilot.yaml
#
#SBATCH --job-name=sweep
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=giacomo.antonelli4@studio.unibo.it
#SBATCH --partition=l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail

GRID="${1:?usage: sbatch --array=0-N scripts/slurm/sweep.sbatch <grid.yaml>}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
COUNT="${SWEEP_SHARDS:-$((${SLURM_ARRAY_TASK_MAX:-0} + 1))}"

cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm/env.sh
source "${MAS_VENV}/bin/activate"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "node=$(hostname) job=${SLURM_JOB_ID} shard=${TASK}/${COUNT}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/sweep.py run "${GRID}" --shard "${TASK}/${COUNT}"
