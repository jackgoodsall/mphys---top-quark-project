#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/model_improvement_%x_%j.out
#SBATCH --chdir=/net/scratch/b58521jg/transformers

set -euo pipefail

CONFIG=${1:?"usage: sbatch scripts/submit_model_improvement.sbatch config/model_improvement/E_arch01_similarity.yaml"}
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }

module purge
module load libs/cuda

export UV_PROJECT_ENVIRONMENT=.transformer_env
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="/net/scratch/b58521jg/transformers:/net/scratch/b58521jg/transformers/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/model-improvement-mpl-${SLURM_JOB_ID:-local}"
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR"

source .transformer_env/bin/activate

echo "Job ID: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-local}"
echo "Config: $CONFIG"
echo "Start: $(date)"
uv run src/main.py --config "$CONFIG"
echo "Completed: $(date)"
