# Top Quark Reconstruction with Transformers

> MPhys project (2025–2026) — applying transformer-based set prediction to semi-leptonic tt̄ events at the LHC.

## Overview

This project develops a **DETR-style MaskedFormer** model for reconstructing top quark decay chains from jet-level inputs. The model uses learnable query tokens and Hungarian matching to assign jets to decay products (W bosons and top quarks) in a permutation-invariant way, without relying on explicit jet ordering.

Key features:
- **Hierarchical chain-query decoding** — a single query token first reconstructs the W, then the parent top from the same token, enforcing the decay topology
- **Multi-task learning** — mask prediction, objectness scoring, and kinematic regression trained jointly with layer-wise supervision
- **Interaction-modulated attention** — pairwise jet features (ΔR, k_T, z, m²) bias the encoder attention maps
- **Hungarian bipartite matching** — permutation-invariant set loss with GPU-accelerated brute-force solver
- **Partially-matched event support** — events where fewer than all tops are reconstructable are handled natively via null query slots

## Architecture

```
Jets [B, N, 7]
    │
    ├─ ParticleEmbedder  →  [B, N, 64]
    └─ InteractionEmbedder  →  [B, C, N, N]
            │
      Encoder (8× ParticleAttentionBlock)
            │  memory [B, N, D]
      Decoder (8× TransformerDecoderLayer)
            │  chain-query: layers 0-3 → W phase, layers 4-7 → top phase
      Prediction heads (mask logits, objectness, kinematics)
            │
      Hungarian matching  →  permuted loss
```

## Repository Layout

```
.
├── src/
│   ├── main.py                          # Entry point
│   ├── models/
│   │   ├── particle_transformer.py      # Main model, embedders
│   │   └── components/
│   │       ├── masked_former_tasks.py   # Task registry & loss functions
│   │       ├── attention_layers.py      # ParticleAttentionBlock
│   │       └── matcher.py              # Hungarian matcher (hepattn)
│   ├── data/
│   │   ├── top_quark_reconstruction.py  # Data modules (HDF5)
│   │   ├── dataset_prepper_2/3.py       # Preprocessing pipelines
│   │   └── root_to_h5py.py             # ROOT → HDF5 conversion
│   ├── trainers/
│   │   ├── top_reconstruction_trainers.py  # LightningModule
│   │   └── loss_functions.py            # LogMinMax scaling, set losses
│   └── analysis/
│       ├── evaluate.py                  # Full evaluation pipeline
│       └── evaluate_predictions.py      # Prediction metrics
├── config/
│   └── top_reconstruction_config.yaml   # All hyperparameters
├── docs/
│   ├── particle_gating_plan.md
│   └── code_quality_audit.md
├── pyproject.toml
├── requirements.lock
└── submit.sh                            # SLURM job script
```

## Getting Started

### Requirements

- Python ≥ 3.9
- CUDA 12.x (tested on A100)
- [`uv`](https://github.com/astral-sh/uv) for environment management

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd transformers

# Create the virtual environment and install dependencies
uv venv .transformer_env
source .transformer_env/bin/activate
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu129
```

### Data

Data is stored as HDF5 files under `data/` (not tracked in git). Each file exposes the following keys:

| Key | Shape | Description |
|---|---|---|
| `jet` | `[N, 7]` | Jet features: pt, η, φ, E, m, b-tag, ... |
| `src_mask` | `[N]` | Padding mask |
| `interactions` | `[N, N, 4]` | Pairwise features: ΔR, k_T, z, m² |
| `masks_tops` | `[n_tops, N]` | Binary jet-assignment masks for tops |
| `masks_Ws` | `[n_Ws, N]` | Binary jet-assignment masks for Ws |
| `kinematics_tops` | `[n_tops, D]` | Target 4-vectors for tops |
| `kinematics_Ws` | `[n_Ws, D]` | Target 4-vectors for Ws |

### Training

```bash
# Local (CPU/single GPU)
uv run src/main.py

# GPU cluster (SLURM — A100)
sbatch submit.sh
```

All hyperparameters live in `config/top_reconstruction_config.yaml`. The `inference.mode` field controls behaviour:

| Mode | Description |
|---|---|
| `train` | Train from scratch |
| `resume` | Resume from `checkpoint_path` |
| `test` | Run evaluation only |
| `lr_find` | Run Lightning LR finder |

## Configuration Highlights

```yaml
model_parameters:
  transformer:
    n_encoder_layers: 8
    n_decoder_layers: 8        # layers 0-3: W phase, 4-7: top phase
    n_heads: 8
    embedding_size: 64
    chain_queries: true         # W→top hierarchical decoding

model_training:
  learning_rate: 5e-4
  precision: "32-true"
  scheduler:
    type: "warmup_cosine"
    warmup_epochs: 3
    T_max: 50
```

## Datasets

Three event topologies are included:

| Dataset | Description |
|---|---|
| `semi_leptonic_ttbar` | Standard Model tt̄ (training baseline) |
| `semi_leptonic_ttH` | tt̄H associated production |
| `semi_leptonic_zprime` | Z′ → tt̄ BSM signal at 500/700/900 GeV |

## Dependencies

See [`requirements.lock`](requirements.lock) for pinned versions. Core stack:

| Package | Version |
|---|---|
| PyTorch | 2.8.0+cu129 |
| PyTorch Lightning | 2.5.5 |
| torchmetrics | 1.8.2 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| h5py | 3.11.0 |

## Cluster (SLURM)

Jobs run on the `gpuA` partition (A100s). Outputs go to `slurm_outputs/`, checkpoints to `checkpoints/`, Lightning logs to `masked_reconstruction/lightning_logs/`.

```bash
sbatch submit.sh
squeue -u $USER          # check job status
tail -f slurm_outputs/<job_id>.out
```
