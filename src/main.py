import os
import torch
import lightning as pl
from pathlib import Path
from typing import Dict, Optional
from lightning.pytorch.loggers import TensorBoardLogger
from models.particle_transformer import (
    ParticleEmbedder, InteractionEmbedder, MaskedReconstructionPart,
)
from data.datamodule import MaskedFormerTopsWsDataModule
from trainers.top_reconstruction_trainers import (
    ReconstructionTrainer, train_reconstruction_model,
)
from models.components.masked_former_tasks import (
    TaskRegistry, TaskConfig,
    MaskReconstructionTask, ObjectnessTask, ObjectTypeTask,
    BackgroundSuppressionTask, ParticleGatingTask,
    ChainTypeTask, NeutrinoRegressionTask,
    ExclusiveAssignmentTask, MaskHierarchyConsistencyTask,
    InvariantMassTask,
)
from utils.utils import load_and_split_config, load_any_config



def find_latest_checkpoint(log_dir: str):
    """Return the path to last.ckpt in the most recent Lightning version dir."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return None
    version_dirs = sorted(log_path.glob("version_*"),
                          key=lambda p: int(p.name.split("_")[1]))
    for version_dir in reversed(version_dirs):
        last_ckpt = version_dir / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            return str(last_ckpt)
    return None




def create_default_task_registry(config: dict) -> TaskRegistry:
    """
    Create task registry from config file.
    
    Args:
        config: Full configuration dict containing task specifications
    
    Returns:
        task_registry: Configured TaskRegistry with all tasks
    """
    task_registry = TaskRegistry()

    # Get general settings
    max_objects = config.get("max_objects", 2)
    n_decoder_layers = config["model_parameters"]["transformer"]["n_decoder_layers"]
    chain_queries = config["model_parameters"]["transformer"].get("chain_queries", False)

    # Get task-specific configurations
    task_configs = config.get("tasks", {})
    
    # ========================================
    # Mask Reconstruction Task
    # ========================================
    mask_config = task_configs.get("mask", {})

    mask_loss_weights = {
        'dice': mask_config.get('dice_weight', 1.0),
        'bce': mask_config.get('bce_weight', 0.5),
    }
    
    # Layer weights
    mask_layer_weights = _build_layer_weights(
        layer_config=mask_config.get('layer_weights'),
        strategy=mask_config.get('layer_weight_strategy', "uniform"),
        strategy_params=mask_config.get('layer_weight_params', {}),
        n_layers=n_decoder_layers
    )
    
    mask_task = MaskReconstructionTask(
        TaskConfig(
            name='mask',
            output_names=['mask_predictions'],
            output_dims={},
            cost_weights={'mask': mask_config.get('cost_weight', 1.0)},
            loss_weights=mask_loss_weights,
            max_objects=max_objects,
            layer_weights=mask_layer_weights,
            head_norm=mask_config.get('head_norm', False),
            mask_embed_head=mask_config.get('mask_embed_head', False),
        ),
        null_mask_penalty=mask_config.get('null_mask_penalty', 0.1),
        bce_pos_weight=mask_config.get('bce_pos_weight', False),
        cost_bce_weight=mask_config.get('cost_bce_weight', 0.0),
    )
    task_registry.register_task(mask_task)

    # ========================================
    # W Mask Task (chain_queries mode only)
    # Each chain query also predicts the W mask in the W-decoder phase.
    # Uses 'mask_W' output (W layers) and 'jet_mask_true_W' target (top-half split).
    # ========================================
    if chain_queries:
        mask_W_config = task_configs.get("mask_W", mask_config)
        mask_W_loss_weights = {
            'dice': mask_W_config.get('dice_weight', mask_config.get('dice_weight', 1.0)),
            'bce':  mask_W_config.get('bce_weight',  mask_config.get('bce_weight',  0.5)),
        }
        mask_W_layer_weights = _build_layer_weights(
            layer_config=mask_W_config.get('layer_weights'),
            strategy=mask_W_config.get('layer_weight_strategy', 'uniform'),
            strategy_params=mask_W_config.get('layer_weight_params', {}),
            n_layers=n_decoder_layers,
        )
        mask_W_task = MaskReconstructionTask(
            TaskConfig(
                name='mask_W',
                output_names=['mask_W'],
                output_dims={},
                cost_weights={'mask': mask_W_config.get('cost_weight', mask_config.get('cost_weight', 1.0))},
                loss_weights=mask_W_loss_weights,
                max_objects=max_objects,
                layer_weights=mask_W_layer_weights,
                head_norm=mask_W_config.get('head_norm', False),
                mask_embed_head=mask_W_config.get('mask_embed_head', mask_config.get('mask_embed_head', False)),
            ),
            null_mask_penalty=mask_W_config.get('null_mask_penalty', mask_config.get('null_mask_penalty', 0.1)),
            bce_pos_weight=mask_W_config.get('bce_pos_weight', mask_config.get('bce_pos_weight', False)),
            cost_bce_weight=mask_W_config.get('cost_bce_weight', mask_config.get('cost_bce_weight', 0.0)),
            pred_key='mask_W',
            target_key='jet_mask_true_W',
        )
        task_registry.register_task(mask_W_task)

    # ========================================
    # Objectness Task (is this query a real object?)
    # Single head — in chain_queries mode a valid top requires a valid W,
    # so one binary objectness per query is sufficient.
    # ========================================
    obj_config = task_configs.get("objectness", {})

    # Conditional registration (DDP-safe on/off): default enabled = baseline.
    # NOTE: old checkpoints were trained WITH objectness — set enabled: true to
    # load / test such a run.
    if obj_config.get('enabled', True):
        obj_layer_weights = _build_layer_weights(
            layer_config=obj_config.get('layer_weights'),
            strategy=obj_config.get('layer_weight_strategy'),
            strategy_params=obj_config.get('layer_weight_params', {}),
            n_layers=n_decoder_layers
        )

        objectness_task = ObjectnessTask(
            TaskConfig(
                name='objectness',
                output_names=['objectness_logit'],
                output_dims={'objectness_logit': 1},
                cost_weights={'objectness': obj_config.get('cost_weight', 1.0)},
                loss_weights={'objectness': obj_config.get('loss_weight', 1.0)},
                max_objects=max_objects,
                layer_weights=obj_layer_weights,
                head_norm=obj_config.get('head_norm', False),
            ),
        )
        task_registry.register_task(objectness_task)

    # ========================================
    # Object Type Task (top vs W classification)
    # Not used in chain_queries mode — all queries are the same "chain" type.
    # ========================================
    if not chain_queries:
        type_config = task_configs.get("object_type", {})

        type_layer_weights = _build_layer_weights(
            layer_config=type_config.get('layer_weights'),
            strategy=type_config.get('layer_weight_strategy'),
            strategy_params=type_config.get('layer_weight_params', {}),
            n_layers=n_decoder_layers
        )

        object_type_task = ObjectTypeTask(
            TaskConfig(
                name='object_type',
                output_names=['type_logit'],
                output_dims={'type_logit': 1},
                cost_weights={'type': type_config.get('cost_weight', 1.0)},
                loss_weights={'type': type_config.get('loss_weight', 1.0)},
                max_objects=max_objects,
                layer_weights=type_layer_weights,
                head_norm=type_config.get('head_norm', False),
            ),
            top_weight=type_config.get('top_weight', 1.0)
        )
        task_registry.register_task(object_type_task)

    # ========================================
    # Leptonic-extension tasks (chain_queries + enable_leptonic mode)
    # ========================================
    enable_leptonic = config["model_parameters"]["transformer"].get("enable_leptonic", False)
    if chain_queries and enable_leptonic:

        # ChainTypeTask: per-chain hadronic/leptonic binary classification
        ct_config = task_configs.get("chain_type", {})
        chain_type_task = ChainTypeTask(
            TaskConfig(
                name='chain_type',
                output_names=['is_leptonic_logit'],
                output_dims={'is_leptonic_logit': 1},
                cost_weights={'chain_type': ct_config.get('cost_weight', 0.5)},
                loss_weights={'chain_type': ct_config.get('loss_weight', 1.0)},
                max_objects=max_objects,
                layer_weights=_build_layer_weights(
                    layer_config=ct_config.get('layer_weights'),
                    strategy=ct_config.get('layer_weight_strategy', 'final_only'),
                    strategy_params=ct_config.get('layer_weight_params', {}),
                    n_layers=n_decoder_layers,
                ),
                head_norm=ct_config.get('head_norm', False),
            ),
        )
        task_registry.register_task(chain_type_task)

        # NeutrinoRegressionTask: pz regression for leptonic chains
        nu_config = task_configs.get("neutrino", {})
        if nu_config:
            nu_dim = nu_config.get('output_dim', 1)
            neutrino_task = NeutrinoRegressionTask(
                TaskConfig(
                    name='neutrino',
                    output_names=['neutrino_pz'],
                    output_dims={'neutrino_pz': nu_dim},
                    cost_weights={},
                    loss_weights={'neutrino': nu_config.get('loss_weight', 1.0)},
                    max_objects=max_objects,
                    layer_weights=_build_layer_weights(
                        layer_config=nu_config.get('layer_weights'),
                        strategy=nu_config.get('layer_weight_strategy', 'final_only'),
                        strategy_params=nu_config.get('layer_weight_params', {}),
                        n_layers=n_decoder_layers,
                    ),
                    head_norm=nu_config.get('head_norm', False),
                ),
                mw_gev=nu_config.get('mw_gev', 80.379),
            )
            task_registry.register_task(neutrino_task)

    # ========================================
    # Exclusive-Assignment CE Tasks (per-particle cross-chain exclusivity)
    # Registered only when the config block is present (headless, no params).
    # ========================================
    for exc_name, exc_pred, exc_tgt in (
        ('exclusive_ce', 'mask_predictions', 'jet_mask_true'),
        ('exclusive_ce_W', 'mask_W', 'jet_mask_true_W'),
    ):
        # mask_W / jet_mask_true_W only exist in chain_queries mode.
        if exc_pred == 'mask_W' and not chain_queries:
            continue
        exc_config = task_configs.get(exc_name)
        if exc_config:
            exc_task = ExclusiveAssignmentTask(
                TaskConfig(
                    name=exc_name,
                    output_names=[exc_pred],   # reuses existing mask head; no new params
                    output_dims={},
                    cost_weights={},
                    loss_weights={'exclusive': exc_config.get('loss_weight', 0.25)},
                    max_objects=max_objects,
                    layer_weights=_build_layer_weights(
                        layer_config=exc_config.get('layer_weights'),
                        strategy=exc_config.get('layer_weight_strategy', 'final_only'),
                        strategy_params=exc_config.get('layer_weight_params', {}),
                        n_layers=n_decoder_layers,
                    ),
                ),
                pred_key=exc_pred,
                target_key=exc_tgt,
                loss_key='exclusive',
                background=exc_config.get('background', 'zero'),
            )
            task_registry.register_task(exc_task)

    # ========================================
    # Mask Hierarchy Consistency Task (soft W-in-top; chain_queries only)
    # ========================================
    cons_config = task_configs.get("mask_consistency")
    if cons_config and chain_queries:
        cons_task = MaskHierarchyConsistencyTask(
            TaskConfig(
                name='mask_consistency',
                output_names=['mask_predictions', 'mask_W'],  # forces both into layer map
                output_dims={},
                cost_weights={},
                loss_weights={'consistency': cons_config.get('loss_weight', 0.1)},
                max_objects=max_objects,
                layer_weights=_build_layer_weights(
                    layer_config=cons_config.get('layer_weights'),
                    strategy=cons_config.get('layer_weight_strategy', 'final_only'),
                    strategy_params=cons_config.get('layer_weight_params', {}),
                    n_layers=n_decoder_layers,
                ),
            ),
            margin=cons_config.get('margin', 0.0),
        )
        task_registry.register_task(cons_task)

    # ========================================
    # Invariant-Mass Task (soft m(top)/m(W) constraint; chain_queries only)
    # ========================================
    im_config = task_configs.get("invariant_mass")
    if im_config and chain_queries:
        im_task = InvariantMassTask(
            TaskConfig(
                name='invariant_mass',
                output_names=['mask_predictions', 'mask_W'],
                output_dims={},
                cost_weights={},
                loss_weights={'invariant_mass': im_config.get('loss_weight', 0.05)},
                max_objects=max_objects,
                layer_weights=_build_layer_weights(
                    layer_config=im_config.get('layer_weights'),
                    strategy=im_config.get('layer_weight_strategy', 'final_only'),
                    strategy_params=im_config.get('layer_weight_params', {}),
                    n_layers=n_decoder_layers,
                ),
            ),
            m_w=im_config.get('m_w', 80.4),
            m_top=im_config.get('m_top', 172.5),
            width_w=im_config.get('width_w', 15.0),
            width_top=im_config.get('width_top', 25.0),
            huber_delta=im_config.get('huber_delta', 1.0),
        )
        task_registry.register_task(im_task)

    # ========================================
    # Background Suppression Task
    # ========================================
    bg_config = task_configs.get("background_suppression", {})
    if bg_config:
        bg_task = BackgroundSuppressionTask(
            TaskConfig(
                name='background_suppression',
                output_names=['mask_predictions'],   # reuses existing head, no new head needed
                output_dims={},
                cost_weights={},
                loss_weights={'bg_suppress': bg_config.get('loss_weight', 1.0)},
                max_objects=max_objects,
                layer_weights=_build_layer_weights(
                    layer_config=bg_config.get('layer_weights'),
                    strategy=bg_config.get('layer_weight_strategy', 'uniform'),
                    strategy_params=bg_config.get('layer_weight_params', {}),
                    n_layers=n_decoder_layers,
                ),
            )
        )
        task_registry.register_task(bg_task)

    # ========================================
    # Particle Gating Task
    # ========================================
    gate_config = task_configs.get("particle_gating", {})
    if gate_config:
        gate_task = ParticleGatingTask(
            TaskConfig(
                name='particle_gating',
                output_names=['mask_predictions'],  # no new head; uses gate_relevance key
                output_dims={},
                cost_weights={},
                loss_weights={'gate': gate_config.get('loss_weight', 0.5)},
                max_objects=max_objects,
                layer_weights=_build_layer_weights(
                    layer_config=gate_config.get('layer_weights'),
                    strategy=gate_config.get('layer_weight_strategy', 'uniform'),
                    strategy_params=gate_config.get('layer_weight_params', {}),
                    n_layers=n_decoder_layers,
                ),
            )
        )
        task_registry.register_task(gate_task)

    return task_registry


def _build_layer_weights(
    layer_config: Optional[dict],
    strategy: Optional[str],
    strategy_params: dict,
    n_layers: int
) -> Optional[Dict[int, float]]:
    """
    Build layer weights from config.
    
    Priority:
        1. Explicit layer_config dict
        2. Strategy-based generation
        3. None (uniform weighting)
    
    Args:
        layer_config: Explicit layer weights dict {0: 0.5, 1: 1.0, ...}
        strategy: Strategy name ('uniform', 'exponential', 'linear', 'final_only')
        strategy_params: Parameters for strategy (e.g., {'base': 2})
        n_layers: Number of decoder layers
    
    Returns:
        layer_weights: Dict or None for uniform
    """
    # Priority 1: Explicit layer weights
    if layer_config is not None:
        return layer_config
    
    # Priority 2: Strategy-based
    if strategy is not None:
        return create_layer_weights(strategy, n_layers, **strategy_params)
    
    # Priority 3: Uniform (None)
    return None


def create_layer_weights(strategy: str, n_layers: int, **kwargs) -> Optional[Dict[int, float]]:
    """
    Helper to create layer weight patterns.
    
    Args:
        strategy: 'uniform', 'exponential', 'linear', 'final_only'
        n_layers: Number of decoder layers
        **kwargs: Strategy-specific parameters
            - base: for exponential (default 2)
            - start, end: for linear (default 0.5, 2.0)
    
    Returns:
        layer_weights: Dict mapping layer_id to weight, or None for uniform
    """
    if strategy == 'uniform':
        return None  # None means all 1.0
    
    elif strategy == 'exponential':
        base = kwargs.get('base', 2)
        return {i: base ** i for i in range(n_layers)}
    
    elif strategy == 'linear':
        start = kwargs.get('start', 0.5)
        end = kwargs.get('end', 2.0)
        weights = {}
        for i in range(n_layers):
            alpha = i / (n_layers - 1) if n_layers > 1 else 1.0
            weights[i] = start * (1 - alpha) + end * alpha
        return weights
    
    elif strategy == 'final_only':
        return {i: (1.0 if i == n_layers - 1 else 0.0) for i in range(n_layers)}
    
    else:
        raise ValueError(f"Unknown layer weight strategy: {strategy}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/top_reconstruction_config.yaml")
    args, _ = parser.parse_known_args()
    config = load_any_config(args.config)

    # FP32 matmul precision (TF32 on Ampere+ GPUs)
    matmul_precision = config.get("model_training", {}).get("matmul_precision", "highest")
    torch.set_float32_matmul_precision(matmul_precision)

    # Reproducibility
    seed = config.get("model_training", {}).get("seed", None)
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    # Create embedders
    particle_embedder = ParticleEmbedder(**config["model_parameters"]["particle_embedder"])
    interactions_embedder = InteractionEmbedder(**config["model_parameters"]["interaction_embedder"])
    
    # Create task registry with defaults
    task_registry = create_default_task_registry(
       config
    )
    
    # Create model
    transformer_model = MaskedReconstructionPart(
        particle_embedder=particle_embedder,
        interaction_embedder=interactions_embedder,
        task_registry=task_registry,
        **config["model_parameters"]["transformer"],
    )
    
    # DataModule
    topantitopquark = MaskedFormerTopsWsDataModule(config)

    # --- Mode dispatch ---
    inf_cfg = config.get("inference", {})
    mode = inf_cfg.get("mode", "train")
    ckpt_path = inf_cfg.get("checkpoint_path", None)
    log_dir = config.get("model_artefacts", {}).get("log_dir", "lightning_logs")

    if mode == "train":
        trainer, model = train_reconstruction_model(
            model=transformer_model,
            task_registry=task_registry,
            data_module=topantitopquark,
            config=config,
        )
        trainer.test(model, datamodule=topantitopquark)

    elif mode == "test":
        assert ckpt_path is not None, (
            "inference.checkpoint_path must be set in config for test mode"
        )
        lightning_model = ReconstructionTrainer.load_from_checkpoint(
            ckpt_path,
            model=transformer_model,
            task_registry=task_registry,
            config=config,
        )
        slurm_id = os.environ.get("SLURM_JOB_ID")
        version = int(slurm_id) if slurm_id else None
        logger = TensorBoardLogger(log_dir, version=version)
        trainer = pl.Trainer(default_root_dir=log_dir, logger=logger)
        trainer.test(lightning_model, datamodule=topantitopquark)

    elif mode == "resume":
        if ckpt_path is None:
            ckpt_path = find_latest_checkpoint(log_dir)
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"No last.ckpt found in {log_dir}. "
                    "Set inference.checkpoint_path explicitly."
                )
            print(f"Auto-detected checkpoint: {ckpt_path}")
        trainer, model = train_reconstruction_model(
            model=transformer_model,
            task_registry=task_registry,
            data_module=topantitopquark,
            config=config,
            ckpt_path=ckpt_path,
        )
        trainer.test(model, datamodule=topantitopquark)

    elif mode == "lr_find":
        from lightning.pytorch.tuner import Tuner

        # Minimal trainer — no early stopping, no checkpointing
        lr_trainer = pl.Trainer(
            num_nodes=1,
            devices=1,
            accelerator=config.get("model_training", {}).get("accelerator", "auto"),
            precision=config.get("model_training", {}).get("precision", "32-true"),
            max_epochs=1,
            default_root_dir=log_dir,
            enable_checkpointing=False,
            logger=False,
        )
        lightning_model = ReconstructionTrainer(transformer_model, task_registry, config)

        tuner = Tuner(lr_trainer)
        lr_finder = tuner.lr_find(
            lightning_model,
            datamodule=topantitopquark,
            min_lr=1e-6,
            max_lr=1e-1,
            num_training=200,        # steps: more steps → smoother curve
            mode="exponential",      # log-scale sweep
            early_stop_threshold=4,  # stop if loss 4x worse than best
        )

        suggestion = lr_finder.suggestion()
        print(f"\nSuggested LR: {suggestion:.2e}")
        print("(Set model_training.learning_rate in config to this value)\n")

        plot_path = Path(log_dir) / "lr_find_result.png"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fig = lr_finder.plot(suggest=True)
        fig.savefig(plot_path, dpi=150)
        print(f"LR range test plot saved to: {plot_path}")

    else:
        raise ValueError(f"Unknown inference mode: {mode}. Use 'train', 'test', 'resume', or 'lr_find'.")


