#!/bin/bash --login
#SBATCH -p multicore_small
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
# multicore_small caps memory at 5345 MB per CPU; 8 CPUs safely allow 40G.
#SBATCH --mem=40G
#SBATCH --job-name=top_contract_convert
#SBATCH --output=slurm_outputs/top_contract_convert_%j.out
#SBATCH --chdir=/net/scratch/b58521jg/transformers

set -euo pipefail

source .transformer_env/bin/activate
export PYTHONPATH=/net/scratch/b58521jg/transformers
export MPLCONFIGDIR="${TMPDIR:-/tmp}/top-contract-convert-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/top-contract-convert-cache-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" slurm_outputs

.transformer_env/bin/python scripts/plan_gate.py audit \
  --contract config/top_reconstruction_data_contract.yaml

.transformer_env/bin/python src/data/convert_fixed_sources.py \
  --config config/top_reconstruction_data_contract.yaml
