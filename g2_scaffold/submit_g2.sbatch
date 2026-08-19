#!/bin/bash --login
# Independent G2-from-scratch pilot.  This file does not modify or resume G1.
# Usage: sbatch g2_scaffold/submit_g2.sbatch g2_scaffold/g2_pilot.yaml
#SBATCH -p gpuL
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --job-name=g2_pilot
#SBATCH --output=slurm_outputs/g2_%x_%j.out
#SBATCH --chdir=/net/scratch/b58521jg/transformers

set -euo pipefail
CONFIG=${1:?'usage: sbatch g2_scaffold/submit_g2.sbatch CONFIG'}
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }

module purge
module load libs/cuda
export PYTHONPATH="/net/scratch/b58521jg/transformers:/net/scratch/b58521jg/transformers/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPLBACKEND=Agg

source .transformer_env/bin/activate
DEVICES=$(.transformer_env/bin/python - "$CONFIG" <<'PY'
import sys
sys.path[:0] = ["/net/scratch/b58521jg/transformers"]
from utils.utils import load_any_config
print(int(load_any_config(sys.argv[1])["model_training"]["devices"]))
PY
)
VISIBLE=$(nvidia-smi --list-gpus | wc -l)
echo "G2 devices=$DEVICES visible GPUs=$VISIBLE SLURM_NTASKS=${SLURM_NTASKS:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
[[ "$VISIBLE" -eq "$DEVICES" ]] || exit 3
[[ "${SLURM_NTASKS:-1}" -eq "$DEVICES" ]] || exit 3

srun --ntasks="$DEVICES" bash -c '
  echo "rank ${SLURM_PROCID} localid ${SLURM_LOCALID}: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
  exec .transformer_env/bin/python g2_scaffold/train.py --config "$1"' _ "$CONFIG"
