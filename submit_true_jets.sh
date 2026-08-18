#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=true_jets_inference

echo "BLOCKED: legacy true-jet artifacts have no stable event identity." >&2
exit 2
module purge
module load libs/cuda

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

source .transformer_env/bin/activate

CKPT="/mnt/iusers01/fse-ugpgt01/phy01/b58521jg/scratch/transformers/masked_reconstruction/lightning_logs/version_11796339/checkpoints/epoch=epoch=42-val_loss=val_loss=1.5476.ckpt"
HPARAMS="/mnt/iusers01/fse-ugpgt01/phy01/b58521jg/scratch/transformers/masked_reconstruction/lightning_logs/version_11796339/hparams.yaml"
DATA="data/topquarkreconstruction/masked_targets_all_events/ttbar_preprocessed_test.h5"
OUT="masked_reconstruction/true_jets_analysis"

echo "Running true-jets inference..."
uv run src/analysis/true_jets_inference.py \
    --config  "$HPARAMS" \
    --ckpt    "$CKPT" \
    --data_file "$DATA" \
    --out_dir   "$OUT" \
    --batch_size 2024 \
    --device cuda

echo "Running evaluation..."
uv run src/analysis/evaluate.py --run_dir "$OUT" --prior top=3 W=2 --plot

echo "Done at: $(date)"
