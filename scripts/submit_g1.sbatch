#!/bin/bash --login
# ─────────────────────────────────────────────────────────────────────────────
# G-track G1 training (contract v3 processed data).
#
#   sbatch scripts/submit_g1.sbatch config/top_reconstruction_g1.yaml [CHECKPOINT]
#
# DDP contract: Lightning's SLURMEnvironment maps each srun task to
# local_rank = SLURM_LOCALID and calls set_device(local_rank). Every task must
# therefore see ALL `model_training.devices` GPUs — do NOT use --gpu-bind=single:1,
# which leaves rank 1 with only cuda:0 visible and crashes on set_device(1).
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p gpuL
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --job-name=g1_train
#SBATCH --output=slurm_outputs/g1_%x_%j.out
#SBATCH --chdir=/net/scratch/b58521jg/transformers

set -euo pipefail

CONFIG=${1:?"usage: sbatch scripts/submit_g1.sbatch CONFIG [CHECKPOINT]"}
RESUME_FROM=${2:-}
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }

.transformer_env/bin/python scripts/preflight_receipt.py check training --config "$CONFIG"
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/g1-mpl-${SLURM_JOB_ID:-local}"
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR" slurm_outputs

source .transformer_env/bin/activate

echo "Job ID: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-local}"
echo "Config: $CONFIG"
echo "Start: $(date)"

# Visible-GPU count must equal trainer.devices, and srun tasks must equal it too.
DEVICES=$(.transformer_env/bin/python - "$CONFIG" <<'PY'
import sys
sys.path[:0] = ["/net/scratch/b58521jg/transformers", "/net/scratch/b58521jg/transformers/src"]
from utils.utils import load_any_config
cfg = load_any_config(sys.argv[1])["model_training"]
assert int(cfg.get("num_nodes", 1)) == 1, "this script is single-node"
print(int(cfg["devices"]))
PY
)
VISIBLE=$(nvidia-smi --list-gpus | wc -l)
echo "model_training.devices=$DEVICES  visible GPUs=$VISIBLE  SLURM_NTASKS=${SLURM_NTASKS:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
[[ "$VISIBLE" -eq "$DEVICES" ]] || { echo "GPU visibility ($VISIBLE) != model_training.devices ($DEVICES)" >&2; exit 3; }
[[ "${SLURM_NTASKS:-1}" -eq "$DEVICES" ]] || { echo "SLURM_NTASKS (${SLURM_NTASKS:-unset}) != model_training.devices ($DEVICES)" >&2; exit 3; }

RUN=(uv run src/main.py --config "$CONFIG")
if [[ -n "$RESUME_FROM" ]]; then
    RUN+=(--resume-from "$RESUME_FROM")
fi

# Each task keeps both GPUs visible; Lightning selects cuda:SLURM_LOCALID.
srun --ntasks="$DEVICES" bash -c '
  echo "rank ${SLURM_PROCID} localid ${SLURM_LOCALID}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} sees $(nvidia-smi --list-gpus | wc -l) GPUs"
  exec "$@"' _ "${RUN[@]}"

echo "Completed: $(date)"
