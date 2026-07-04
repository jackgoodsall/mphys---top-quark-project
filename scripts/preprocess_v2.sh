#!/bin/bash --login
# ─────────────────────────────────────────────────────────────────────────────
# Stage D v2 preprocessing (hadronic pipeline, preprocessing.py).
# Rebuilds masked_targets_combined_v2 with: jet mass scaling, jet_p4_raw (for the
# invariant-mass loss), and per-feature interaction scaling. v1 stays untouched.
#
#   sbatch scripts/preprocess_v2.sh      # or: bash scripts/preprocess_v2.sh
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p serial
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --job-name=preprocess_v2
#SBATCH --output=slurm_outputs/preprocess_v2_%j.out

set -euo pipefail
cd "$(dirname "$0")/.."

export UV_PROJECT_ENVIRONMENT=.transformer_env

echo "=== Stage D v2 preprocessing ==="
echo "Start: $(date)   Node: ${SLURM_NODELIST:-local}"

mkdir -p slurm_outputs data/topquarkreconstruction/masked_targets_combined_v2

uv run src/data/preprocessing.py --config config/preprocessing_config_v2.yaml

echo "=== Done: $(date) ==="
ls -lh data/topquarkreconstruction/masked_targets_combined_v2/
