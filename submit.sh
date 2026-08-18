#!/bin/bash --login
#SBATCH -p gpuL           # Partition: A100 GPU / V100 GPU
#SBATCH --gres=gpu:2          # Matches model_training.devices: 2
#SBATCH --time=3-00:00:00     # Wall time: 3 days
#SBATCH --ntasks=1            # One task
#SBATCH --cpus-per-task=8     # 16 CPU cores / 8 CPU cores

echo "BLOCKED: legacy training entrypoint is not v3-gated." >&2
exit 2

# Load CUDA module
.transformer_env/bin/python scripts/plan_gate.py training --config config/top_reconstruction_config.yaml
module purge
module load libs/cuda

# Slurm sets CUDA_VISIBLE_DEVICES for the allocation; do not override it.
export UV_PROJECT_ENVIRONMENT=.transformer_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export UV_PROJECT_ENVIRONMENT=.transformer_env

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_CPUS_PER_TASK CPU core(s)"

# Activate UV environment
source .transformer_env/bin/activate

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Run Python program
echo "Running src/main.py..."
uv run src/main.py

echo "Program completed at: $(date)"
