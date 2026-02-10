#!/bin/bash --login
#SBATCH -p gpuA               # Partition: A100 GPU / V100 GPU
#SBATCH --gres=gpu:1          # Request 2 GPUs / 1 GPU
#SBATCH --time=3-00:00:00     # Wall time: 3 days
#SBATCH --ntasks=1            # One task
#SBATCH --cpus-per-task=8     # 16 CPU cores / 8 CPU cores

# Load CUDA module
module purge
module load libs/cuda

export CUDA_VISsIBLE_DEVICES=0,1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

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
