# MPhys Project: Transformer-Based Top Quark Reconstruction

MPhys masters project (2025–2026) at the University of Manchester. A DETR-style **MaskedFormer** transformer architecture for semi-leptonic ttbar top quark reconstruction, using Hungarian matching and multi-task learning to simultaneously reconstruct top quarks and W bosons from jet collections.

## Overview

The model takes a variable-length jet collection as input and predicts which jets belong to each top quark and W boson decay product — framed as a set-prediction problem with learnable query tokens, bipartite matching, and per-particle binary mask outputs.

Key features:
- **MaskedFormer architecture**: encoder–decoder transformer with learnable object queries
- **Hungarian matching**: optimal bipartite assignment between predictions and targets at training time
- **Multi-object reconstruction**: simultaneous top and W boson reconstruction with separate query banks
- **Partially-matched events**: handles variable numbers of reconstructable tops per event
- **Layer-wise supervision**: auxiliary losses at each decoder layer, weighted by depth
- **Mask-only pretraining**: two-phase curriculum (masks first, then objectness)
- **Particle gating**: auxiliary task to suppress background jet assignment

## Repository structure

```
src/
  main.py                             # Entry point — wires config, model, trainer
  models/
    particle_transformer.py           # MaskedReconstructionPart, ParticleEmbedder, InteractionEmbedder
    components/
      attention_layers.py             # ParticleAttentionBlock, DecoderAttentionBlock
      masked_former_tasks.py          # TaskRegistry, BaseTask, all task implementations
      matcher.py                      # Hungarian matcher (from hepattn — do not modify)
  data/
    datamodule.py                     # MaskedFormerTopsWsDataModule (PyTorch Lightning)
    root_to_h5.py                     # ROOT → HDF5 conversion
    preprocessing.py                  # Feature scaling, dataset preparation
    prepare_combined_dataset.py       # Combine tops+Ws into unified HDF5
    kinematics.py                     # Kinematic feature helpers
    convert_interactions_fp16.py      # fp16 compression for interaction matrices
    split_raw_h5.py                   # Train/val/test splitting
    analyse_unique_tags.py            # Tag uniqueness analysis
    fix_matching_unique_tags.py       # Fix tag matching issues
    root_unique_tags.py               # Extract unique tags from ROOT files
  analysis/
    evaluate.py                       # Main evaluation: efficiency, purity, IoU, plots, threshold sweep
    evaluate_chain.py                 # Chain-query (hierarchical) model evaluation
    compare_models.py                 # Side-by-side model comparison
    true_jets_inference.py            # Signal-jets-only inference for intrinsic performance
    loss_by_multiplicity.py           # Loss breakdown by jet multiplicity
    objectness_score_distribution.py  # Objectness score histograms
    report_plots.py                   # Publication-quality result plots
  trainers/
    top_reconstruction_trainers.py    # ReconstructionTrainer (LightningModule)
    loss_functions.py                 # LogMinMax scaling, set-invariant loss
  data_utls/
    scalers.py                        # LogMinMax, PhiTransformer scalers

config/
  top_reconstruction_config.yaml     # Main config — all hyperparameters and task weights
  preprocessing_config.yaml          # Data preprocessing config
  vanilla_3m_config.yaml             # Vanilla (no interaction) 3M-param baseline config

notebooks/
  event_reconstruction_analysis.ipynb      # Event-level reconstruction visualisation
  tops_ws_reconstruction_analysis.ipynb    # Tops+Ws joint reconstruction analysis
  explore_inclusive_root.ipynb             # Raw ROOT data exploration

submit.sh                            # Main SLURM job submission
submit_lr_find.sh                    # LR finder job
submit_true_jets.sh                  # Signal-jets-only inference job
requirements.lock                    # Pinned dependencies
```

## Running

**Local:**
```bash
uv run src/main.py
```

**GPU cluster (SLURM, A100):**
```bash
sbatch submit.sh
```

Configuration is driven entirely by `config/top_reconstruction_config.yaml`. No `setup.py` or `pyproject.toml` — the project uses `uv` with a manually managed virtualenv at `.transformer_env/` (Python 3.9, CUDA-enabled PyTorch).

### Environment setup

```bash
uv venv .transformer_env --python 3.9
uv pip install -r requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match \
    --python .transformer_env/bin/python
```

## Data pipeline

Raw ROOT files → HDF5 → preprocessed HDF5 → training.

```bash
# 1. Convert ROOT to raw HDF5
uv run src/data/root_to_h5.py

# 2. Preprocess (scaling, target masks, interaction features)
uv run src/data/preprocessing.py

# 3. (Optional) Combine tops+Ws into unified dataset
uv run src/data/prepare_combined_dataset.py
```

Data is stored in `data/topquarkreconstruction/` (gitignored). The model expects HDF5 files with keys: `jet`, `src_mask`, `interactions`, `masks_tops`, `masks_Ws`, `kinematics_tops`, `kinematics_Ws`.

## Evaluation

```bash
# Full efficiency + purity report with plots
uv run src/analysis/evaluate.py --run_dir lightning_logs/version_X --plot

# Also save plain-text summary
uv run src/analysis/evaluate.py --run_dir lightning_logs/version_X --plot --save_summary

# Threshold sweep to find optimal binarisation threshold
uv run src/analysis/evaluate.py --run_dir lightning_logs/version_X --threshold_sweep

# Top-k prior binarisation (e.g. always assign exactly 3 jets to top, 2 to W)
uv run src/analysis/evaluate.py --run_dir lightning_logs/version_X --prior top=3 W=2
```

Metrics reported: top/W/ttbar reconstruction efficiency and purity (per-multiplicity and overall), objectness precision/recall/F1, type accuracy, mask IoU.

## Architecture

### Forward pass

1. **ParticleEmbedder**: raw jet features `[B, N, 7]` → embeddings `[B, N, D]`
2. **InteractionEmbedder**: pairwise features `[B, N, N, 4]` → interaction maps `[B, C, N, N]`
3. **Encoder** (`ParticleAttentionBlock` stack): self-attention + interaction-modulated cross-attention → memory `[B, N, D]`
4. **Decoder** (`nn.TransformerDecoderLayer` stack): learnable query tokens cross-attend to encoder memory; outputs collected at each decoder layer for layer-wise supervision
5. **Prediction heads**: built dynamically from `TaskRegistry` output specs (mask logits, objectness, type classification)
6. **Hungarian matching** (`Matcher`): bipartite match predictions to targets; permute outputs to match
7. **Loss**: `TaskRegistry.compute_total_loss()` aggregates per-task losses with per-layer weights

### Task registry

Tasks implement `BaseTask` with `compute_cost()` (for matching cost matrix) and `compute_loss()`. Adding a new task: subclass `BaseTask`, register in config — prediction heads are built automatically from task output specs.

Active tasks: `MaskReconstructionTask`, `ObjectnessTask`, `ObjectTypeTask`, `BackgroundSuppressionTask`, `ParticleGatingTask`, `IoUPredictionTask`, `MaskOverlapTask`.

## Branch structure

| Branch | Description |
|---|---|
| `masked_former` | Main stable branch |
| `mask_improvements` | Active development |
| `semester_1_work` | Semester 1 — pre-MaskedFormer classifier architecture |
| `archive/week5-masked-former` | Week 5 checkpoint snapshot |
| `archive/week6-masked-former` | Week 6 checkpoint snapshot |

## Dependencies

PyTorch, PyTorch Lightning, torch_optimizer, scikit-learn, scipy, numpy, h5py, matplotlib, torchmetrics, vector, pydantic, PyYAML, uproot.
