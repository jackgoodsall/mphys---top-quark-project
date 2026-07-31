#!/bin/bash --login
# ─────────────────────────────────────────────────────────────────────────────
# Stage D v2 preprocessing (hadronic pipeline, preprocessing.py).
# Rebuilds masked_targets_combined_v2 with: jet mass scaling, jet_p4_raw (for the
# invariant-mass loss), and per-feature interaction scaling. v1 stays untouched.
#
#   sbatch scripts/preprocess_v2.sh      # or: bash scripts/preprocess_v2.sh
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p multicore_small
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --job-name=preprocess_v2
#SBATCH --output=slurm_outputs/preprocess_v2_%j.out
#SBATCH --chdir=/net/scratch/b58521jg/transformers

set -euo pipefail

export UV_PROJECT_ENVIRONMENT=.transformer_env
# preprocessing.py imports both `from src.data_utils...` and `from kinematics...`.
export PYTHONPATH="/net/scratch/b58521jg/transformers:/net/scratch/b58521jg/transformers/src:${PYTHONPATH:-}"

echo "=== Stage D v2 preprocessing ==="
echo "Start: $(date)   Node: ${SLURM_NODELIST:-local}"

mkdir -p slurm_outputs data/topquarkreconstruction/masked_targets_combined_v2

source .transformer_env/bin/activate

uv run src/data/preprocessing.py --config config/preprocessing_config_v2.yaml

echo "=== Done: $(date) ==="
ls -lh data/topquarkreconstruction/masked_targets_combined_v2/
