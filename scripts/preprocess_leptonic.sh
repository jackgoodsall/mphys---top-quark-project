#!/bin/bash --login
# ─────────────────────────────────────────────────────────────────────────────
# Preprocess combined hadronic + semi-leptonic dataset
#
# Runs preprocess_combined.py which:
#   1. Fits shared scalers on both hadronic + semi-leptonic training data
#   2. Processes all splits into a unified HDF5 schema
#   3. Writes output to data/topquarkreconstruction/leptonic_combined/
#
# Submit as SLURM job:
#   sbatch scripts/preprocess_leptonic.sh
#
# Or run interactively:
#   bash scripts/preprocess_leptonic.sh
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p cpu               # CPU partition — preprocessing needs no GPU
#SBATCH --time=4:00:00       # Wall time: 4 hours (generous for ~30M events)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=preprocess_leptonic
#SBATCH --output=slurm_outputs/preprocess_leptonic_%j.out

set -euo pipefail
cd "$(dirname "$0")/.."   # always run from repo root

export UV_PROJECT_ENVIRONMENT=.transformer_env

echo "=== Leptonic combined dataset preprocessing ==="
echo "Start: $(date)"
echo "Node: ${SLURM_NODELIST:-local}"

# ── Input paths ───────────────────────────────────────────────────────────────
HAD_DIR="data/topquarkreconstruction/h5py_data/combined"
SLEP_DIR="data/semi_leptonic_ttbar"
OUT_DIR="data/topquarkreconstruction/leptonic_combined"

HAD_TRAIN="${HAD_DIR}/ttbar_h5py_raw_train.h5"
HAD_VAL="${HAD_DIR}/ttbar_h5py_raw_val.h5"
HAD_TEST="${HAD_DIR}/ttbar_h5py_raw_test_old.h5"      # all-hadronic test set
SLEP_TRAIN="${SLEP_DIR}/training_mass_variation.h5"   # mass-variation for training
SLEP_TEST="${SLEP_DIR}/testing_sm.h5"                  # SM-only for evaluation

echo ""
echo "Hadronic sources:"
echo "  train : ${HAD_TRAIN}"
echo "  val   : ${HAD_VAL}"
echo "  test  : ${HAD_TEST}"
echo ""
echo "Semi-leptonic sources:"
echo "  train : ${SLEP_TRAIN}"
echo "  test  : ${SLEP_TEST}"
echo ""
echo "Output directory: ${OUT_DIR}"
echo ""

mkdir -p "${OUT_DIR}" slurm_outputs

# ── Run preprocessing ─────────────────────────────────────────────────────────
uv run src/data/preprocess_combined.py \
    --had_train  "${HAD_TRAIN}" \
    --had_val    "${HAD_VAL}" \
    --had_test   "${HAD_TEST}" \
    --slep_train "${SLEP_TRAIN}" \
    --slep_test  "${SLEP_TEST}" \
    --output_dir "${OUT_DIR}"

echo ""
echo "=== Done: $(date) ==="
echo ""
echo "Output files:"
ls -lh "${OUT_DIR}/"

echo ""
echo "Next: train with config/leptonic_config.yaml"
echo "  sbatch submit.sh   (after updating config data path if needed)"
