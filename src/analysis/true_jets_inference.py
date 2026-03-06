"""
True-jets-only multiplicity analysis: run inference on signal jets only.

For each test event, feed the model only jets that are true decay products
(appear in at least one valid top/W mask), then save outputs in evaluate.py-
compatible format for direct comparison against full-multiplicity baseline.

Usage:
    uv run src/analysis/true_jets_inference.py \\
        --config  config/top_reconstruction_config.yaml \\
        --ckpt    <checkpoint.ckpt> \\
        --data_file <ttbar_preprocessed_test.h5> \\
        --out_dir  <output_directory> \\
        [--batch_size 2024] [--device cuda]
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Set up import path so src/ modules are importable
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR))
# Also add the project root so 'src.*' imports work (some modules use src.X)
_PROJ_DIR = _SRC_DIR.parent
sys.path.insert(0, str(_PROJ_DIR))

from models.particle_transformer import (
    ParticleEmbedder, InteractionEmbedder, MaskedReconstructionPart,
)
from data.top_quark_reconstruction import (
    MaskedFormerDataSet, masked_former_collate_fn, merge_object_types,
)
from trainers.top_reconstruction_trainers import ReconstructionTrainer
from utils.utils import load_any_config
import yaml


def _load_config(path: Path) -> dict:
    """Load config from a plain YAML or a Lightning hparams.yaml (where config is nested under 'config:')."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Lightning hparams.yaml wraps the config under a 'config' key
    if isinstance(raw, dict) and set(raw.keys()) == {"config"}:
        return raw["config"]
    return raw

# Import create_default_task_registry and _build_layer_weights from main.py
import importlib.util
_main_spec = importlib.util.spec_from_file_location(
    "main_module",
    str(_SRC_DIR / "main.py"),
)
_main_mod = importlib.util.module_from_spec(_main_spec)
_main_spec.loader.exec_module(_main_mod)
create_default_task_registry = _main_mod.create_default_task_registry
_build_layer_weights = _main_mod._build_layer_weights


# ---------------------------------------------------------------------------
# Signal-jets dataset builder
# ---------------------------------------------------------------------------

def build_signal_jets_dataset(h5_path: Path, config: dict):
    """
    Load an HDF5 test file and return a dataset containing only the signal
    jets (jets appearing in at least one valid top or W mask).

    Returns:
        dataset:        MaskedFormerDataSet with compacted signal jets
        original_mult:  int array [N] — original jet multiplicity per event
        signal_mult:    int array [N] — signal jet multiplicity per event
    """
    with h5py.File(h5_path, "r") as f:
        jet          = f["jet"][()]           # [N, P, 7]
        src_mask     = f["src_mask"][()]      # [N, P]
        interactions = f["interactions"][()]  # [N, P, P, 4]
        masks_tops   = f["masks_tops"][()]    # [N, T_top, P]
        tops_kins    = f["kinematics_tops"][()] # [N, T_top, D]

        if "masks_Ws" in f:
            masks_Ws = f["masks_Ws"][()]
            ws_kins  = f["kinematics_Ws"][()]
        else:
            masks_Ws = np.zeros((jet.shape[0], 0, jet.shape[1]), dtype=masks_tops.dtype)
            ws_kins  = np.zeros((jet.shape[0], 0, tops_kins.shape[-1]), dtype=tops_kins.dtype)

        valid_tops = f["valid_tops"][()] if "valid_tops" in f else np.ones(
            (jet.shape[0], masks_tops.shape[1]), dtype=bool
        )
        valid_Ws   = f["valid_Ws"][()] if "valid_Ws" in f else np.ones(
            (jet.shape[0], masks_Ws.shape[1]), dtype=bool
        )

    N, P = src_mask.shape

    # Vectorised signal mask: OR of all valid-object particle masks
    vt = valid_tops.astype(bool)[:, :, None]           # [N, T_top, 1]
    tops_sig = (masks_tops * vt).any(axis=1)            # [N, P]

    vw = valid_Ws.astype(bool)[:, :, None]              # [N, T_w, 1]
    ws_sig   = (masks_Ws * vw).any(axis=1)              # [N, P]

    signal_mask = (tops_sig | ws_sig) & src_mask.astype(bool)  # [N, P]

    original_mult = src_mask.astype(bool).sum(axis=1).astype(int)   # [N]
    signal_mult   = signal_mask.sum(axis=1).astype(int)              # [N]

    # Per-event repack: compact signal jets to front; keep array shape [N, P, ...]
    new_jet          = np.zeros_like(jet)
    new_src_mask     = np.zeros_like(src_mask)
    new_interactions = np.zeros_like(interactions)
    new_masks_tops   = np.zeros_like(masks_tops)
    new_masks_Ws     = np.zeros_like(masks_Ws)

    for i in range(N):
        idx = np.where(signal_mask[i])[0]   # signal jet indices in original order
        k   = len(idx)
        if k == 0:
            continue
        new_jet[i, :k]              = jet[i, idx]
        new_src_mask[i, :k]         = True
        new_interactions[i, :k, :k] = interactions[i][np.ix_(idx, idx)]
        new_masks_tops[i, :, :k]    = masks_tops[i][:, idx]
        if masks_Ws.shape[1] > 0:
            new_masks_Ws[i, :, :k]  = masks_Ws[i][:, idx]

    # Merge object types into unified arrays
    masks, kins, classes, object_valid = merge_object_types(
        new_masks_tops, tops_kins,
        new_masks_Ws if masks_Ws.shape[1] > 0 else None,
        ws_kins if masks_Ws.shape[1] > 0 else None,
        valid_tops, valid_Ws,
    )

    has_partial = not object_valid.all()

    dataset = MaskedFormerDataSet(
        jet=new_jet,
        interactions=new_interactions,
        src_mask=new_src_mask,
        targets=masks,
        target_kinematics=kins,
        target_mass=None,
        mass_with_kinematics=False,
        classes=classes,
        object_valid=object_valid if has_partial else None,
    )

    return dataset, original_mult, signal_mult


# ---------------------------------------------------------------------------
# H5 output file creation
# ---------------------------------------------------------------------------

def create_output_h5s(out_dir: Path, N: int, Q: int, P: int):
    """Create the three evaluate.py-compatible H5 output files."""
    mask_f = h5py.File(out_dir / "test_outputs_mask.h5", "w")
    mask_f.create_dataset("target_masks",           shape=(N, Q, P), dtype="float32")
    mask_f.create_dataset("predicted_masks_logits", shape=(N, Q, P), dtype="float32")
    mask_f.create_dataset("predicted_masks_prob",   shape=(N, Q, P), dtype="float32")
    mask_f.create_dataset("jet_valid_mask",         shape=(N, P),    dtype="float32")

    obj_f = h5py.File(out_dir / "test_outputs_objectness.h5", "w")
    obj_f.create_dataset("target_objectness",          shape=(N, Q), dtype="float32")
    obj_f.create_dataset("predicted_objectness_logit", shape=(N, Q), dtype="float32")
    obj_f.create_dataset("predicted_objectness_prob",  shape=(N, Q), dtype="float32")

    type_f = h5py.File(out_dir / "test_outputs_object_type.h5", "w")
    type_f.create_dataset("target_classes",       shape=(N, Q), dtype="int32")
    type_f.create_dataset("predicted_type_logit", shape=(N, Q), dtype="float32")
    type_f.create_dataset("predicted_type_prob",  shape=(N, Q), dtype="float32")

    return mask_f, obj_f, type_f


# ---------------------------------------------------------------------------
# Write one batch to H5
# ---------------------------------------------------------------------------

def write_batch(mask_f, obj_f, type_f, predictions, save_targets, start, Q, P):
    """Write one batch of predictions/targets into the open H5 file handles."""
    B   = predictions["mask_predictions"].shape[0]
    end = start + B

    # ---- Mask ----
    pred_logit = predictions["mask_predictions"].float().cpu().numpy()   # [B, Q_model, P_model]
    pred_prob  = torch.sigmoid(predictions["mask_predictions"]).float().cpu().numpy()
    tgt_masks  = save_targets["jet_mask_true"].float().cpu().numpy()

    if tgt_masks.ndim == 2:
        tgt_masks = tgt_masks[:, None, :]  # [B, 1, P]

    # Truncate query and particle dims to stored shapes
    pred_logit = pred_logit[:, :Q, :P]
    pred_prob  = pred_prob[:, :Q, :P]
    tgt_masks  = tgt_masks[:, :Q, :P]

    mask_f["target_masks"][start:end]           = tgt_masks
    mask_f["predicted_masks_logits"][start:end] = pred_logit
    mask_f["predicted_masks_prob"][start:end]   = pred_prob

    jvm = save_targets.get("jet_valid_mask")
    if jvm is not None:
        mask_f["jet_valid_mask"][start:end] = jvm.float().cpu().numpy()[:, :P]

    # ---- Objectness ----
    obj_logit = predictions["objectness_logit"].squeeze(-1)  # [B, Q_model]
    obj_prob  = torch.sigmoid(obj_logit)

    obj_valid = save_targets.get("obj_valid_mask")
    if obj_valid is not None:
        tgt_obj = obj_valid.float()
    else:
        tgt_obj = torch.ones_like(obj_logit)

    obj_f["target_objectness"][start:end]          = tgt_obj[:, :Q].float().cpu().numpy()
    obj_f["predicted_objectness_logit"][start:end] = obj_logit[:, :Q].float().cpu().numpy()
    obj_f["predicted_objectness_prob"][start:end]  = obj_prob[:, :Q].float().cpu().numpy()

    # ---- Object type ----
    type_logit = predictions["type_logit"].squeeze(-1)  # [B, Q_model]
    type_prob  = torch.sigmoid(type_logit)

    classes = save_targets.get(
        "classes", torch.zeros(B, Q, dtype=torch.long, device=type_logit.device)
    )

    type_f["target_classes"][start:end]       = classes[:, :Q].cpu().numpy().astype("int32")
    type_f["predicted_type_logit"][start:end] = type_logit[:, :Q].float().cpu().numpy()
    type_f["predicted_type_prob"][start:end]  = type_prob[:, :Q].float().cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="True-jets-only inference for multiplicity analysis."
    )
    parser.add_argument("--config",    required=True,  type=Path)
    parser.add_argument("--ckpt",      required=True,  type=Path)
    parser.add_argument("--data_file", required=True,  type=Path)
    parser.add_argument("--out_dir",   required=True,  type=Path)
    parser.add_argument("--batch_size", type=int, default=2024)
    parser.add_argument("--device",    type=str, default="cuda")
    args = parser.parse_args()

    # Validate inputs
    for p, name in [(args.config, "--config"), (args.ckpt, "--ckpt"),
                    (args.data_file, "--data_file")]:
        if not p.exists():
            sys.exit(f"ERROR: {name} path not found: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Build model architecture from config
    # ------------------------------------------------------------------
    config = _load_config(args.config)

    particle_embedder    = ParticleEmbedder(**config["model_parameters"]["particle_embedder"])
    interactions_embedder = InteractionEmbedder(**config["model_parameters"]["interaction_embedder"])
    task_registry        = create_default_task_registry(config)

    transformer_model = MaskedReconstructionPart(
        particle_embedder=particle_embedder,
        interaction_embedder=interactions_embedder,
        task_registry=task_registry,
        **config["model_parameters"]["transformer"],
        use_hungarian_matching=config.get("use_hungarian_matching", True),
        matching_solver=config.get("matching_solver", "gpu_bruteforce"),
        max_targets=config.get("max_targets", 5),
    )

    # ------------------------------------------------------------------
    # 2. Load checkpoint
    # ------------------------------------------------------------------
    lightning_model = ReconstructionTrainer.load_from_checkpoint(
        str(args.ckpt),
        model=transformer_model,
        task_registry=task_registry,
        config=config,
        strict=False,
    )
    # Zero out type_embeddings so randomly-initialised weights don't corrupt
    # target_tokens for checkpoints trained without type embeddings.
    lightning_model.model.type_embeddings.weight.data.zero_()
    lightning_model.eval()
    lightning_model.to(device)
    print(f"Loaded checkpoint: {args.ckpt}")

    # ------------------------------------------------------------------
    # 3. Build signal-jets dataset
    # ------------------------------------------------------------------
    print(f"Building signal-jets dataset from {args.data_file} ...")
    dataset, original_mult, signal_mult = build_signal_jets_dataset(
        args.data_file, config
    )
    N = len(dataset)
    P = dataset.src_mask.shape[1]   # particle dim (same as original data)
    Q = config["max_objects"]
    print(f"  Events: {N:,}   P_max: {P}   Q: {Q}")
    print(f"  Original mult: mean={original_mult.mean():.2f}")
    print(f"  Signal mult:   mean={signal_mult.mean():.2f}")

    # ------------------------------------------------------------------
    # 4. Save multiplicities
    # ------------------------------------------------------------------
    np.savez(
        args.out_dir / "event_multiplicities.npz",
        original_mult=original_mult,
        signal_mult=signal_mult,
    )
    print(f"Saved event_multiplicities.npz")

    # ------------------------------------------------------------------
    # 5. Create H5 output files
    # ------------------------------------------------------------------
    mask_f, obj_f, type_f = create_output_h5s(args.out_dir, N, Q, P)

    # ------------------------------------------------------------------
    # 6. Inference loop
    # ------------------------------------------------------------------
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=masked_former_collate_fn,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    start_idx = 0
    print(f"Running inference over {len(dataloader)} batches ...")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move to device
            inputs  = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) if torch.is_tensor(v) else v
                       for k, v in targets.items()}

            # Inject targets for Hungarian matching (mirrors training_step)
            inputs["targets"] = targets

            outputs = lightning_model.model(inputs, last_output_only=True)

            final_layer = max(outputs.keys())
            layer_dict  = outputs[final_layer]

            if "__targets__" in layer_dict:
                save_targets = layer_dict["__targets__"]
                predictions  = {k: v for k, v in layer_dict.items()
                                if k != "__targets__"}
            else:
                save_targets = targets
                predictions  = layer_dict

            write_batch(mask_f, obj_f, type_f, predictions, save_targets,
                        start_idx, Q, P)
            start_idx += predictions["mask_predictions"].shape[0]

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(dataloader)} done")

    # ------------------------------------------------------------------
    # 7. Close H5 files
    # ------------------------------------------------------------------
    mask_f.close()
    obj_f.close()
    type_f.close()
    print("H5 files closed.")

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    orig_dist   = Counter(original_mult.tolist())
    signal_dist = Counter(signal_mult.tolist())
    n_zero      = int((signal_mult == 0).sum())

    print("\n--- Multiplicity Summary ---")
    print("Original mult distribution:", dict(sorted(orig_dist.items())))
    print("Signal mult distribution:  ", dict(sorted(signal_dist.items())))
    print(f"Events with 0 signal jets: {n_zero} (excluded from efficiency metrics)")
    print(f"\nOutputs written to: {args.out_dir.resolve()}")
    print(
        f"\nTo evaluate true-jets performance:\n"
        f"  python src/analysis/evaluate.py --run_dir {args.out_dir} --plot\n"
        f"\nTo load multiplicity mapping:\n"
        f"  import numpy as np\n"
        f"  d = np.load('{args.out_dir}/event_multiplicities.npz')\n"
        f"  original_mult, signal_mult = d['original_mult'], d['signal_mult']"
    )


if __name__ == "__main__":
    main()
