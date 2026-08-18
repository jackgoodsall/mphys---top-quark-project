#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=exp_bundle_v1
#SBATCH --output=slurm_outputs/exp_bundle_v1_%j.out
#SBATCH --chdir=/net/scratch/b58521jg/transformers

# Recommended-bundle training on existing v1 data (config/exp_bundle_v1.yaml).
echo "BLOCKED: v1 data has no manifest or stable event IDs." >&2
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
uv run src/main.py --config config/exp_bundle_v1.yaml
echo "Completed at: $(date)"
