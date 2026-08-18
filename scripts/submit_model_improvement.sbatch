#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=model_improvement_ddp
#SBATCH --output=slurm_outputs/model_improvement_%x_%j.out
#SBATCH --chdir=/net/scratch/b58521jg/transformers

set -euo pipefail
echo "BLOCKED: G2 model/head/calibration gates are not yet complete." >&2
exit 2

CONFIG=${1:?"usage: sbatch scripts/submit_model_improvement.sbatch CONFIG [CHECKPOINT]"}
RESUME_FROM=${2:-}
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
.transformer_env/bin/python scripts/plan_gate.py training --config "$CONFIG"
if [[ -n "$RESUME_FROM" && ! -f "$RESUME_FROM" ]]; then
    echo "Missing checkpoint: $RESUME_FROM" >&2
    exit 2
fi

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

if [[ "${SLURM_NTASKS:-1}" -ne 2 ]]; then
    echo "Expected two DDP tasks, got ${SLURM_NTASKS:-unset}" >&2
    exit 3
fi

RUN=(uv run src/main.py --config "$CONFIG")
if [[ -n "$RESUME_FROM" ]]; then
    RUN+=(--resume-from "$RESUME_FROM")
fi

srun --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1 "${RUN[@]}"
echo "Completed: $(date)"
