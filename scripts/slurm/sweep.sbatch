#!/bin/bash
# Runs an experiment grid as a Slurm job array: task i of n runs block i/n of the grid (scripts/sweep.py).
#
#   1. on a login node (internet), once:
#        export HF_HOME=/shared/hf HF_TOKEN=...
#        python scripts/download_models.py configs/experiments/pilot.yaml
#        python scripts/sweep.py plan configs/experiments/pilot.yaml        # prints the number of units
#   2. submit, with as many array tasks as you want shards (n <= number of units); the array starts at 0:
#        sbatch --array=0-9 scripts/slurm/sweep.sbatch configs/experiments/pilot.yaml
#      Re-submitting the same command resumes: runs that ended on their own are skipped. To redo only the failed
#      shards, say how many shards the grid was split in: SWEEP_SHARDS=10 sbatch --array=3,7 ... (sbatch exports
#      the environment to the job)
#   3. when it is done:  python scripts/sweep.py evaluate configs/experiments/pilot.yaml
#
# Adjust the partition, GPU type, time and environment lines to your cluster.
#SBATCH --job-name=sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail

GRID="${1:?usage: sbatch --array=0-N scripts/slurm/sweep.sbatch <grid.yaml>}"
COUNT="${SWEEP_SHARDS:-$((SLURM_ARRAY_TASK_MAX + 1))}"

cd "${SLURM_SUBMIT_DIR}"
# source ~/venvs/mas/bin/activate          # the environment with pip install -e .
export HF_HOME="${HF_HOME:?set HF_HOME to the cache that scripts/download_models.py filled}"
export HF_HUB_OFFLINE=1                    # compute nodes need no internet: everything is in the cache
export TOKENIZERS_PARALLELISM=false

echo "node=$(hostname) array_task=${SLURM_ARRAY_TASK_ID} shard=${SLURM_ARRAY_TASK_ID}/${COUNT}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/sweep.py run "${GRID}" --shard "${SLURM_ARRAY_TASK_ID}/${COUNT}"
