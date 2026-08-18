#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=exp_stageD_v2
#SBATCH --output=slurm_outputs/exp_stageD_v2_%j.out
#SBATCH --chdir=/net/scratch/b58521jg/transformers

# Full Stage-D training on v2 data (config/exp_stageD_v2.yaml).
# Requires scripts/preprocess_v2.sh to have produced masked_targets_combined_v2 first.
echo "BLOCKED: Stage-D v2 predates the G-1/G0 contract." >&2
exit 2
module purge
module load libs/cuda

export CUDA_VISIBLE_DEVICES=0,1
export UV_PROJECT_ENVIRONMENT=.transformer_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# main.py uses `from models...` (needs src/) while modules use `from src.models...`
# (needs repo root). Put both on the path.
export PYTHONPATH="/net/scratch/b58521jg/transformers:/net/scratch/b58521jg/transformers/src:${PYTHONPATH:-}"

source .transformer_env/bin/activate

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
uv run src/main.py --config config/exp_stageD_v2.yaml
echo "Completed at: $(date)"
