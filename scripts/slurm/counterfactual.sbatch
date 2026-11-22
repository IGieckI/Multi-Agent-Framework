#!/bin/bash
# Scores logged turns of finished runs under edited prompts (scripts/counterfactual.py): sbatch counterfactual.sbatch <model> <grid folders...>
#
#SBATCH --job-name=counterfactual
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=giacomo.antonelli4@studio.unibo.it
#SBATCH --partition=l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

MODEL="${1:?usage: sbatch scripts/slurm/counterfactual.sbatch <model> <grid folder>...}"
shift

cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm/env.sh
source "${MAS_VENV}/bin/activate"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "node=$(hostname) job=${SLURM_JOB_ID} model=${MODEL} grids=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/counterfactual.py "${MODEL}" "$@"
