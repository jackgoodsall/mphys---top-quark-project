#!/bin/bash --login
#SBATCH -p multicore_small
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=top_contract_audit
#SBATCH --output=slurm_outputs/top_contract_audit_%j.out
#SBATCH --chdir=/net/scratch/b58521jg/transformers

set -euo pipefail
.transformer_env/bin/python scripts/preflight_receipt.py check audit
source .transformer_env/bin/activate
export PYTHONPATH=/net/scratch/b58521jg/transformers
export MPLCONFIGDIR="${TMPDIR:-/tmp}/top-contract-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" slurm_outputs

.transformer_env/bin/python src/data/root_to_h5.py \
  --config config/top_reconstruction_data_contract.yaml \
  --max-events 200000 \
  --output data/topquarkreconstruction/contract_v3/audit/ttbar_contract_audit.h5 \
  --manifest data/topquarkreconstruction/contract_v3/audit/manifest.json
