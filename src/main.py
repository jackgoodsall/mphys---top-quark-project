from models.particle_transformer import  *
from data.top_quark_reconstruction import *
from trainers.top_reconstruction_trainers import *
from utils.utils import load_and_split_config, load_any_config




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
    
    # Get task-specific configurations
    task_configs = config.get("tasks", {})
    
    # Get legacy loss parameters for backward compatibility
    legacy_loss_params = config.get("loss_parameters", {})
    
    # ========================================
    # Mask Reconstruction Task
    # ========================================
    mask_config = task_configs.get("mask", {})
    
    # Loss weights: try new format first, fall back to legacy
    mask_loss_weights = {
        'dice': mask_config.get('dice_weight', 
                               legacy_loss_params.get('dice_weight', 1.0)),
        'bce': mask_config.get('bce_weight',
                              legacy_loss_params.get('bce_weight', 0.5))
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
            layer_weights=mask_layer_weights
        )
    )
    task_registry.register_task(mask_task)
    
    # ========================================
    # Kinematic Regression Task
    # ========================================
    kin_config = task_configs.get("kinematics", {})
    
    # Loss weights: support different loss types
    kin_loss_weights = {}
    if 'l1_weight' in kin_config:
        kin_loss_weights['l1'] = kin_config['l1_weight']
    elif 'mse_weight' in kin_config:
        kin_loss_weights['mse'] = kin_config['mse_weight']
    else:
        # Default to smooth_l1
        kin_loss_weights['smooth_l1'] = kin_config.get('smooth_l1_weight', 1.0)
    
    # Layer weights
    kin_layer_weights = _build_layer_weights(
        layer_config=kin_config.get('layer_weights'),
        strategy=kin_config.get('layer_weight_strategy'),
        strategy_params=kin_config.get('layer_weight_params', {}),
        n_layers=n_decoder_layers
    )
    
    kinematics_task = KinematicRegressionTask(
        TaskConfig(
            name='kinematics',
            output_names=['object_kinematics'],
            output_dims={'object_kinematics': 4},
            cost_weights={'kinematics': kin_config.get('cost_weight', 1.0)},
            loss_weights=kin_loss_weights,
            max_objects=max_objects,
            layer_weights=kin_layer_weights
        )
    )
    #task_registry.register_task(kinematics_task)
    
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
    config = load_any_config("config/top_reconstruction_config.yaml")
    
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
        use_hungarian_matching=config.get("use_hungarian_matching", True)
    )
    
    # DataModule
    topantitopquark = MaskedFormer2(config)
    
    # Train
    trainer, model = train_reconstruction_model(
        model=transformer_model,
        task_registry=task_registry,
        data_module=topantitopquark,
        config=config
    )
    
    # Test
    trainer.test(model, datamodule=topantitopquark)


