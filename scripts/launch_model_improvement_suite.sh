#!/bin/bash --login
set -euo pipefail

ROOT=/net/scratch/b58521jg/transformers
cd "$ROOT"

echo "BLOCKED: superseded architecture suite does not satisfy G-1/G0." >&2
echo "Use scripts/submit_contract_audit.sh to start the gated v3 pipeline." >&2
exit 2

configs=(
  config/model_improvement/E_arch01_similarity.yaml
  config/model_improvement/E_arch02_pair_capacity.yaml
  config/model_improvement/E_arch03_structured_decoder.yaml
  config/model_improvement/E_arch04_combined.yaml
)

for config in "${configs[@]}"; do
  experiment=${config##*/}
  experiment=${experiment%.yaml}
  sbatch --job-name="$experiment" scripts/submit_model_improvement.sbatch "$config"
done
