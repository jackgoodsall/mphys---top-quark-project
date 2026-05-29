"""
Compute per-event losses grouped by jet multiplicity (6, 7, 8+).

Usage:
    cd /net/scratch/b58521jg/transformers
    uv run src/analysis/loss_by_multiplicity.py \
        --config config/top_reconstruction_config.yaml \
        --split val
"""
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import create_default_task_registry
from models.particle_transformer import (
    ParticleEmbedder, InteractionEmbedder, MaskedReconstructionPart,
)
from data.datamodule import MaskedFormerTopsWsDataModule, masked_former_collate_fn
from trainers.top_reconstruction_trainers import ReconstructionTrainer
from utils.utils import load_any_config


def per_event_mask_dice(pred_logits, target_masks, obj_valid, valid_mask=None, eps=1e-6):
    """
    Compute per-event mean Dice score for matched real objects.

    Args:
        pred_logits: [B, Q, P] raw logits
        target_masks: [B, Q, P] matched targets
        obj_valid: [B, Q] bool — which query slots are real
        valid_mask: [B, P] bool — which particles are real (optional)
    Returns:
        dice_per_event: [B] mean Dice per event (NaN for events with 0 real objects)
    """
    B, Q, P = pred_logits.shape
    pred_probs = pred_logits.sigmoid()

    if valid_mask is not None:
        vm = valid_mask.unsqueeze(1).expand(B, Q, P)
        pred_probs = pred_probs * vm
        target_masks = target_masks.float() * vm

    intersection = (pred_probs * target_masks.float()).sum(dim=-1)  # [B, Q]
    pred_sum = pred_probs.sum(dim=-1)                                # [B, Q]
    tgt_sum = target_masks.float().sum(dim=-1)                       # [B, Q]
    dice = (2 * intersection) / (pred_sum + tgt_sum + eps)           # [B, Q]

    # Mask out null queries
    dice = dice * obj_valid.float()                                  # [B, Q]
    n_real = obj_valid.float().sum(dim=-1).clamp(min=1)              # [B]
    return dice.sum(dim=-1) / n_real                                 # [B]


def per_event_mask_bce(pred_logits, target_masks, obj_valid, valid_mask=None):
    """
    Compute per-event mean BCE for matched real objects.

    Returns:
        bce_per_event: [B]
    """
    B, Q, P = pred_logits.shape
    target_float = target_masks.float()

    bce = F.binary_cross_entropy_with_logits(pred_logits, target_float, reduction='none')  # [B, Q, P]

    if valid_mask is not None:
        vm = valid_mask.unsqueeze(1).expand(B, Q, P)
        bce = bce * vm
        per_query = bce.sum(dim=-1) / vm.sum(dim=-1).clamp(min=1)  # [B, Q]
    else:
        per_query = bce.mean(dim=-1)  # [B, Q]

    per_query = per_query * obj_valid.float()
    n_real = obj_valid.float().sum(dim=-1).clamp(min=1)
    return per_query.sum(dim=-1) / n_real  # [B]


@torch.no_grad()
def analyse(config, split, device, max_batches=None):
    # Build model + task registry
    particle_embedder = ParticleEmbedder(**config["model_parameters"]["particle_embedder"])
    interaction_embedder = InteractionEmbedder(**config["model_parameters"]["interaction_embedder"])
    task_registry = create_default_task_registry(config)

    transformer_model = MaskedReconstructionPart(
        particle_embedder=particle_embedder,
        interaction_embedder=interaction_embedder,
        task_registry=task_registry,
        **config["model_parameters"]["transformer"],
    )

    ckpt_path = config["inference"]["checkpoint_path"]
    print(f"Loading checkpoint: {ckpt_path}")

    # Auto-detect vanilla vs custom encoder from checkpoint keys
    ckpt_sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    has_vanilla = any("encoder_stack.0.self_attn" in k for k in ckpt_sd)
    has_custom = any("encoder_stack.0.qkv" in k for k in ckpt_sd)
    if has_vanilla and not has_custom:
        cfg_vanilla = config["model_parameters"]["transformer"].get("use_vanilla_attention", False)
        if not cfg_vanilla:
            print("  Checkpoint uses vanilla attention — overriding config")
            config["model_parameters"]["transformer"]["use_vanilla_attention"] = True
            # Rebuild model with corrected config
            transformer_model = MaskedReconstructionPart(
                particle_embedder=particle_embedder,
                interaction_embedder=interaction_embedder,
                task_registry=task_registry,
                **config["model_parameters"]["transformer"],
            )
    elif has_custom and not has_vanilla:
        cfg_vanilla = config["model_parameters"]["transformer"].get("use_vanilla_attention", False)
        if cfg_vanilla:
            print("  Checkpoint uses custom attention — overriding config")
            config["model_parameters"]["transformer"]["use_vanilla_attention"] = False
            transformer_model = MaskedReconstructionPart(
                particle_embedder=particle_embedder,
                interaction_embedder=interaction_embedder,
                task_registry=task_registry,
                **config["model_parameters"]["transformer"],
            )
    del ckpt_sd

    lightning_model = ReconstructionTrainer.load_from_checkpoint(
        ckpt_path,
        model=transformer_model,
        task_registry=task_registry,
        config=config,
    )
    lightning_model.eval()
    lightning_model.to(device)
    model = lightning_model.model
    print("Model loaded and moved to device.", flush=True)

    # Load data — force lazy loading to avoid OOM on login nodes
    config["data_modules"]["lazy"] = True
    config["data_modules"]["load_interactions"] = False
    print(f"Loading {split} data (lazy, no interactions)...", flush=True)
    dm = MaskedFormerTopsWsDataModule(config)
    dm.setup("fit" if split in ("train", "val") else "test")
    ds = {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}[split]
    print(f"Dataset loaded: {len(ds)} events", flush=True)
    dl_cfg = config["data_modules"][split if split != "train" else "val"]
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=masked_former_collate_fn,
    )
    print(f"DataLoader ready: {len(loader)} batches", flush=True)

    # Accumulators: dict from n_jets → list of per-event metrics
    records = []  # list of (n_jets, dice_top, dice_W, bce_top, bce_W, total_loss)

    chain_queries = config["model_parameters"]["transformer"].get("chain_queries", False)

    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = {k: v.to(device) for k, v in targets.items()}
        inputs["targets"] = targets

        outputs = model(inputs, last_output_only=True)

        # Jet multiplicity: number of real particles per event
        n_jets = inputs["src_mask"].sum(dim=-1).cpu().numpy()  # [B]

        # Get final layer outputs (only layer present with last_output_only=True)
        final_layer_id = max(outputs.keys())
        layer_out = outputs[final_layer_id]

        if "__targets__" in layer_out:
            matched_targets = layer_out["__targets__"]
            preds = {k: v for k, v in layer_out.items() if k != "__targets__"}
        else:
            matched_targets = targets
            preds = layer_out

        obj_valid = matched_targets.get("obj_valid_mask")
        valid_mask = matched_targets.get("jet_valid_mask")

        B = n_jets.shape[0]

        # -- Top mask metrics --
        if "mask_predictions" in preds:
            pred_top = preds["mask_predictions"]  # [B, Q, P]
            tgt_top = matched_targets["jet_mask_true"]  # [B, Q, P]

            if obj_valid is not None:
                ov_top = obj_valid
            else:
                Q_t = tgt_top.shape[1]
                ov_top = torch.ones(B, pred_top.shape[1], dtype=torch.bool, device=device)
                ov_top[:, Q_t:] = False

            dice_top = per_event_mask_dice(pred_top, tgt_top, ov_top, valid_mask).cpu().numpy()
            bce_top = per_event_mask_bce(pred_top, tgt_top, ov_top, valid_mask).cpu().numpy()
        else:
            dice_top = np.full(B, np.nan)
            bce_top = np.full(B, np.nan)

        # -- W mask metrics (chain_queries mode) --
        if chain_queries and "mask_W" in preds:
            pred_W = preds["mask_W"]
            tgt_W = matched_targets.get("jet_mask_true_W")
            if tgt_W is not None:
                obj_valid_W = matched_targets.get("obj_valid_mask_W", obj_valid)
                if obj_valid_W is None:
                    obj_valid_W = torch.ones(B, pred_W.shape[1], dtype=torch.bool, device=device)
                dice_W = per_event_mask_dice(pred_W, tgt_W, obj_valid_W, valid_mask).cpu().numpy()
                bce_W = per_event_mask_bce(pred_W, tgt_W, obj_valid_W, valid_mask).cpu().numpy()
            else:
                dice_W = np.full(B, np.nan)
                bce_W = np.full(B, np.nan)
        else:
            dice_W = np.full(B, np.nan)
            bce_W = np.full(B, np.nan)

        for i in range(B):
            records.append((int(n_jets[i]), dice_top[i], dice_W[i], bce_top[i], bce_W[i]))

        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            print(f"  batch {batch_idx + 1}/{total_batches}", flush=True)

    # Convert to arrays
    all_njets = np.array([r[0] for r in records])
    all_dice_top = np.array([r[1] for r in records])
    all_dice_W = np.array([r[2] for r in records])
    all_bce_top = np.array([r[3] for r in records])
    all_bce_W = np.array([r[4] for r in records])

    has_W = not np.all(np.isnan(all_dice_W))

    # --- Per-multiplicity stats ---
    unique_mults = sorted(set(all_njets))
    mult_labels = []
    mult_counts = []
    dice_top_means, dice_top_stds = [], []
    bce_top_means = []
    dice_W_means, dice_W_stds = [], []
    bce_W_means = []

    print(f"\n{'='*70}")
    print(f"  Loss breakdown by jet multiplicity ({split} set)")
    print(f"{'='*70}")

    for m in unique_mults:
        mask = all_njets == m
        n = mask.sum()
        dt = np.nanmean(all_dice_top[mask])
        dt_std = np.nanstd(all_dice_top[mask])
        bt = np.nanmean(all_bce_top[mask])

        mult_labels.append(str(m))
        mult_counts.append(n)
        dice_top_means.append(dt)
        dice_top_stds.append(dt_std)
        bce_top_means.append(bt)

        line = f"  {m:>2d} jets: {n:>7d} events   Top Dice={dt:.4f}+-{dt_std:.4f}   BCE={bt:.4f}"
        if has_W:
            dw = np.nanmean(all_dice_W[mask])
            dw_std = np.nanstd(all_dice_W[mask])
            bw = np.nanmean(all_bce_W[mask])
            dice_W_means.append(dw)
            dice_W_stds.append(dw_std)
            bce_W_means.append(bw)
            line += f"   W Dice={dw:.4f}+-{dw_std:.4f}   W BCE={bw:.4f}"
        print(line)

    # --- Plot ---
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    n_metrics = 3 if has_W else 2  # dice_top, bce_top, [dice_W]
    fig, axes = plt.subplots(1, n_metrics + 1, figsize=(5 * (n_metrics + 1), 5))

    x = np.arange(len(mult_labels))

    # Panel 1: Event count per multiplicity
    ax = axes[0]
    ax.bar(x, mult_counts, color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(mult_labels)
    ax.set_xlabel("Number of jets")
    ax.set_ylabel("Number of events")
    ax.set_title("Event distribution")
    for i, c in enumerate(mult_counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=8)

    # Panel 2: Top Dice score
    ax = axes[1]
    ax.errorbar(x, dice_top_means, yerr=dice_top_stds, fmt="o-",
                color="crimson", capsize=4, label="Top Dice")
    ax.set_xticks(x)
    ax.set_xticklabels(mult_labels)
    ax.set_xlabel("Number of jets")
    ax.set_ylabel("Dice score")
    ax.set_title("Top mask Dice score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Panel 3: Top BCE
    ax = axes[2]
    ax.plot(x, bce_top_means, "s-", color="darkorange", label="Top BCE")
    ax.set_xticks(x)
    ax.set_xticklabels(mult_labels)
    ax.set_xlabel("Number of jets")
    ax.set_ylabel("BCE loss")
    ax.set_title("Top mask BCE")
    ax.grid(True, alpha=0.3)

    # Panel 4: W Dice (if applicable)
    if has_W:
        ax = axes[3]
        ax.errorbar(x, dice_W_means, yerr=dice_W_stds, fmt="D-",
                    color="forestgreen", capsize=4, label="W Dice")
        ax.set_xticks(x)
        ax.set_xticklabels(mult_labels)
        ax.set_xlabel("Number of jets")
        ax.set_ylabel("Dice score")
        ax.set_title("W mask Dice score")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Loss by jet multiplicity ({split} set)", fontsize=14, y=1.02)
    fig.tight_layout()
    plot_path = out_dir / f"loss_by_multiplicity_{split}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/top_reconstruction_config.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap number of batches (for quick testing)")
    args = parser.parse_args()

    config = load_any_config(args.config)
    torch.set_float32_matmul_precision(
        config.get("model_training", {}).get("matmul_precision", "highest")
    )

    analyse(config, args.split, args.device, max_batches=args.max_batches)
