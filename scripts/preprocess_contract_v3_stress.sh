#!/bin/bash --login
# ─────────────────────────────────────────────────────────────────────────────
# Contract v3 stress-split preprocessing (inclusive test, 636,263 events).
# Transform only: the frozen G1 train scaler is copied in verbatim, so the
# resulting scaler_hash matches contract_v3/processed by construction.
# Run only after contract_v3/processed exists.
#
#   sbatch scripts/preprocess_contract_v3_stress.sh
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p multicore_small
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# multicore_small caps memory at 5345 MB per CPU; 8 CPUs safely allow 40G.
#SBATCH --mem=40G
#SBATCH --job-name=preprocess_contract_v3_stress
#SBATCH --output=slurm_outputs/preprocess_contract_v3_stress_%j.out
#SBATCH --chdir=/net/scratch/b58521jg/transformers

set -euo pipefail

export UV_PROJECT_ENVIRONMENT=.transformer_env
# preprocessing.py imports both `from src.data_utils...` and `from kinematics...`.
export PYTHONPATH="/net/scratch/b58521jg/transformers:/net/scratch/b58521jg/transformers/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/preprocess-v3-mpl-${SLURM_JOB_ID:-local}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/preprocess-v3-cache-${SLURM_JOB_ID:-local}"
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" slurm_outputs

source .transformer_env/bin/activate

echo "=== contract v3 stress preprocessing ==="
echo "Start: $(date)   Node: ${SLURM_NODELIST:-local}"

.transformer_env/bin/python src/data/preprocessing.py \
  --config config/preprocessing_contract_v3_stress.yaml

echo "=== Done: $(date) ==="
ls -lh data/topquarkreconstruction/contract_v3/processed_stress/
