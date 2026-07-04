#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=exp_bundle_v1
#SBATCH --output=slurm_outputs/exp_bundle_v1_%j.out

# Recommended-bundle training on existing v1 data (config/exp_bundle_v1.yaml).
module purge
module load libs/cuda

export CUDA_VISIBLE_DEVICES=0,1
export UV_PROJECT_ENVIRONMENT=.transformer_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$(dirname "$0")/.."
source .transformer_env/bin/activate

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
uv run src/main.py --config config/exp_bundle_v1.yaml
echo "Completed at: $(date)"
