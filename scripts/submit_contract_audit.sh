#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
scripts/preflight_top_reconstruction.sh audit
sbatch scripts/contract_audit.sbatch
