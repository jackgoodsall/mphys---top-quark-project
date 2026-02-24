from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

# Object class constants
CLASS_NULL = 0
CLASS_TOP = 1
CLASS_W = 2


@dataclass
class TaskConfig:
    """Configuration for a task"""
    name: str
    output_names: list[str]
    output_dims: Dict[str, int]
    cost_weights: Dict[str, float]  # For matching cost
    loss_weights: Dict[str, float]  # For loss computation
    max_objects: int
    layer_weights: Optional[Dict[int, float]] = None
    
    def get_layer_weight(self, layer_id: int) -> float:
        """Get weight for a specific layer"""
        if self.layer_weights is None:
            return 1.0
        return self.layer_weights.get(layer_id, 1.0)
    
    def get_loss_weight(self, component: str) -> float:
        """Get weight for a specific loss component"""
        return self.loss_weights.get(component, 1.0)


class BaseTask(ABC, nn.Module):
    """
    Base task class that defines how to compute costs and losses.
    """
    
    def __init__(self, config: TaskConfig):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute cost matrix for matching.
        
        Returns:
            cost_matrix: [B, num_queries, num_targets]
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss for this task.
        
        Returns:
            loss: Scalar loss
        """
        pass
    
    def create_test_datasets(self, file: h5py.File, number_events: int):
        """
        Create HDF5 datasets for test predictions.
        Override in subclass to define what to save.
        
        Args:
            file: HDF5 file handle
            number_events: Number of test samples
        """
        # Default: do nothing
        pass
    
    def save_test_predictions(
        self,
        file: h5py.File,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        start_idx: int,
        batch_size: int
    ):
        """
        Save test predictions to HDF5.
        Override in subclass to define what to save.
        
        Args:
            file: HDF5 file handle
            predictions: Model predictions
            targets: Ground truth
            start_idx: Starting index in HDF5 file
            batch_size: Size of current batch
        """
        # Default: do nothing
        pass


class TaskRegistry(nn.Module):
    """Registry for managing multiple tasks"""

    def __init__(self):
        super().__init__()
        self.tasks = nn.ModuleDict()
        # None = all tasks active; set of names = only those tasks active.
        # Plain Python attribute so it is NOT saved in checkpoints — reconstructed
        # from config on every run, making checkpoint resume correct automatically.
        self._active_tasks: Optional[set] = None

    def register_task(self, task: 'BaseTask'):
        """Register a new task"""
        self.tasks[task.config.name] = task

    def set_active_tasks(self, task_names: Optional[list]):
        """
        Control which tasks contribute to cost and loss computation.

        Task parameters remain in the model and optimizer throughout — only
        their loss/cost contributions are gated. This preserves optimizer
        state and avoids initialisation shocks at the phase transition.

        Args:
            task_names: List of task names to activate, or None for all tasks.
        """
        if task_names is None:
            self._active_tasks = None
        else:
            unknown = set(task_names) - set(self.tasks.keys())
            if unknown:
                raise ValueError(f"Unknown tasks requested: {unknown}")
            self._active_tasks = set(task_names)

    def _is_active(self, task_name: str) -> bool:
        return self._active_tasks is None or task_name in self._active_tasks

    def compute_total_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total cost across all active tasks.

        Returns:
            cost_matrix: [B, num_queries, num_targets]
        """
        total_cost = None

        for task_name, task in self.tasks.items():
            if not self._is_active(task_name):
                continue
            task_cost = task.compute_cost(predictions, targets)

            if total_cost is None:
                total_cost = task_cost
            else:
                total_cost += task_cost

        return total_cost

    def compute_total_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
        layer_id: Optional[int] = None,
        is_final_layer: bool = True,
    ) -> tuple:
        """
        Compute total loss across all active tasks.

        Returns:
            (total_loss, per_task_losses): total scalar loss and dict of per-task tensor losses
        """
        total_loss = 0.0
        per_task_losses: Dict[str, torch.Tensor] = {}

        for task_name, task in self.tasks.items():
            if not self._is_active(task_name):
                continue

            # Determine layer weight up-front so we can skip zero-weight tasks
            if layer_id is not None:
                layer_weight = task.config.get_layer_weight(layer_id)
                if layer_weight == 0:
                    continue
            else:
                layer_weight = 1.0

            # Gate stat updates to final decoder layer only (intermediate layers
            # give misleading stats since matching is performed on the final layer)
            task._stats_enabled = is_final_layer
            task_loss = task.compute_loss(predictions, targets, valid_mask)
            task._stats_enabled = True  # reset to safe default

            task_loss = layer_weight * task_loss

            per_task_losses[task_name] = task_loss.detach() if torch.is_tensor(task_loss) else torch.tensor(float(task_loss))
            total_loss += task_loss

        return total_loss, per_task_losses


class MaskReconstructionTask(BaseTask):
    """Task for mask reconstruction"""

    def __init__(self, config: TaskConfig, null_mask_penalty: float = 0.1):
        super().__init__(config)
        self.eps = 1e-8
        self.null_mask_penalty = null_mask_penalty
        # Set True during mask-only pretraining to remove "predict nothing" signal
        self.suppress_null_penalty = False
    
    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute mask cost using Dice coefficient"""
        pred_masks = predictions['mask_predictions'].sigmoid()
        target_masks = targets['jet_mask_true'].float()
        
        # Handle 2D targets
        if target_masks.ndim == 2:
            target_masks = target_masks.unsqueeze(1)
        
        B, num_queries, N = pred_masks.shape
        num_targets = target_masks.shape[1]
        
        # Compute pairwise Dice
        pred_expanded = pred_masks.unsqueeze(2)
        target_expanded = target_masks.unsqueeze(1)
        
        intersection = (pred_expanded * target_expanded).sum(dim=-1)
        pred_sizes = pred_masks.sum(dim=-1, keepdim=True)
        target_sizes = target_masks.sum(dim=-1).unsqueeze(1)
        
        dice = (2 * intersection) / (pred_sizes + target_sizes + self.eps)
        cost = self.config.cost_weights['mask'] * (1 - dice)
        
        return cost
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute mask loss (Dice + BCE)"""
        pred_masks = predictions['mask_predictions']
        target_masks = targets['jet_mask_true']

        if target_masks.ndim == 2:
            target_masks = target_masks.unsqueeze(1)

        B, num_queries, N = pred_masks.shape
        obj_valid = targets.get('obj_valid_mask')

        if obj_valid is not None:
            # --- New path: use obj_valid_mask for variable T ---
            total_loss = pred_masks.new_tensor(0.0)

            if obj_valid.any():
                real_pred_logits = pred_masks[obj_valid]       # [N_real, N]
                real_tgt = target_masks[obj_valid]             # [N_real, N]

                real_pred_probs = real_pred_logits.sigmoid()
                real_tgt_float = real_tgt.float()

                if valid_mask is not None:
                    vm_expanded = valid_mask.unsqueeze(1).expand(B, num_queries, N)
                    real_vm = vm_expanded[obj_valid]            # [N_real, N]
                    real_pred_probs = real_pred_probs * real_vm
                    real_tgt_float = real_tgt_float * real_vm

                # Dice loss
                intersection = (real_pred_probs * real_tgt_float).sum(dim=-1)
                pred_sum = real_pred_probs.sum(dim=-1)
                target_sum = real_tgt_float.sum(dim=-1)
                dice = (2 * intersection) / (pred_sum + target_sum + self.eps)
                dice_loss = (1 - dice).mean()

                # BCE loss
                bce = F.binary_cross_entropy_with_logits(
                    real_pred_logits, real_tgt_float, reduction='none'
                )
                if valid_mask is not None:
                    bce = bce * real_vm
                    num_valid = real_vm.sum(dim=-1).clamp(min=1)
                    bce_loss = (bce.sum(dim=-1) / num_valid).mean()
                else:
                    bce_loss = bce.mean()

                dice_weight = self.config.get_loss_weight('dice')
                bce_weight = self.config.get_loss_weight('bce')
                total_loss = dice_weight * dice_loss + bce_weight * bce_loss

            # Null mask penalty: encourage unmatched queries to predict empty masks.
            # Adaptive scaling: reduce penalty when most queries are null so the
            # "predict empty" signal doesn't overwhelm real-object learning.
            # Suppressed during mask-only pretraining to eliminate "predict nothing" signal.
            if self.null_mask_penalty > 0 and not self.suppress_null_penalty and (~obj_valid).any():
                null_logits = pred_masks[~obj_valid]           # [N_null, N]
                null_targets = torch.zeros_like(null_logits)
                null_loss = F.binary_cross_entropy_with_logits(
                    null_logits, null_targets, reduction='mean'
                )
                n_real = obj_valid.sum().float()
                n_total = obj_valid.numel()
                adaptive_scale = (n_real / n_total).clamp(min=0.01)
                total_loss = total_loss + self.null_mask_penalty * adaptive_scale * null_loss

            return total_loss
        else:
            # --- Backward compatible path: use [:num_targets] slicing ---
            num_targets = target_masks.shape[1]
            pred_masks = pred_masks[:, :num_targets, :]

            pred_probs = pred_masks.sigmoid()
            target_float = target_masks.float()

            if valid_mask is not None:
                valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(pred_probs)
                pred_probs = pred_probs * valid_mask_expanded
                target_float = target_float * valid_mask_expanded

            pred_probs_flat = pred_probs.reshape(-1, N)
            target_float_flat = target_float.reshape(-1, N)
            pred_masks_flat = pred_masks.reshape(-1, N)

            # Dice loss
            intersection = (pred_probs_flat * target_float_flat).sum(dim=-1)
            pred_sum = pred_probs_flat.sum(dim=-1)
            target_sum = target_float_flat.sum(dim=-1)
            dice = (2 * intersection) / (pred_sum + target_sum + self.eps)
            dice_loss = 1 - dice

            # BCE loss
            bce_per_particle = F.binary_cross_entropy_with_logits(
                pred_masks_flat, target_float_flat, reduction='none'
            )

            if valid_mask is not None:
                valid_mask_flat = valid_mask.unsqueeze(1).expand(-1, num_targets, -1).reshape(-1, N)
                bce_per_particle = bce_per_particle * valid_mask_flat
                num_valid_per_sample = valid_mask_flat.sum(dim=-1)
                num_valid_per_sample = torch.clamp(num_valid_per_sample, min=1)
                bce_loss = bce_per_particle.sum(dim=-1) / num_valid_per_sample
            else:
                bce_loss = bce_per_particle.mean(dim=-1)

            dice_weight = self.config.get_loss_weight('dice')
            bce_weight = self.config.get_loss_weight('bce')

            total_loss = dice_weight * dice_loss.mean() + bce_weight * bce_loss.mean()
            return total_loss
    
    def create_test_datasets(self, file: h5py.File, number_events: int):
        """Create HDF5 datasets for mask predictions"""
        N_particles = 20  # Adjust to your actual particle count if needed
        
        file.create_dataset(
            "target_masks",
            shape=(number_events, self.config.max_objects, N_particles),
            dtype='float32'
        )
        file.create_dataset(
            "predicted_masks_logits",
            shape=(number_events, self.config.max_objects, N_particles),
            dtype='float32'
        )
        file.create_dataset(
            "predicted_masks_prob",
            shape=(number_events, self.config.max_objects, N_particles),
            dtype='float32'
        )
        file.create_dataset(
            "jet_valid_mask",
            shape=(number_events, N_particles),
            dtype='float32'
        )
    
    def save_test_predictions(
        self,
        file: h5py.File,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        start_idx: int,
        batch_size: int
    ):
        """Save mask predictions to HDF5"""
        pred_masks_logits = predictions['mask_predictions'].float().cpu().numpy()
        pred_masks_prob = predictions['mask_predictions'].sigmoid().float().cpu().numpy()
        target_masks = targets['jet_mask_true'].float().cpu().numpy()
        jet_valid_mask = targets.get('jet_valid_mask')
        
        # Handle 2D targets
        if target_masks.ndim == 2:
            target_masks = target_masks[:, None, :]
        
        # Truncate to max_objects (predictions may have Q > max_objects after padding)
        M = self.config.max_objects
        pred_masks_logits = pred_masks_logits[:, :M, :]
        pred_masks_prob = pred_masks_prob[:, :M, :]
        target_masks = target_masks[:, :M, :]

        # Save to HDF5
        end_idx = start_idx + batch_size
        file["target_masks"][start_idx:end_idx] = target_masks
        file["predicted_masks_logits"][start_idx:end_idx] = pred_masks_logits
        file["predicted_masks_prob"][start_idx:end_idx] = pred_masks_prob
        if jet_valid_mask is not None:
            file["jet_valid_mask"][start_idx:end_idx] = jet_valid_mask.float().cpu().numpy()


class KinematicRegressionTask(BaseTask):
    """Task for kinematic regression"""
    
    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute kinematic cost using L1 distance"""
        pred_kin = predictions['object_kinematics']
        target_kin = targets['kinematics']
        
        # Handle 2D targets
        if target_kin.ndim == 2:
            target_kin = target_kin.unsqueeze(1)
        
        pred_exp = pred_kin.unsqueeze(2)
        target_exp = target_kin.unsqueeze(1)
        l1_distance = torch.abs(pred_exp - target_exp).sum(dim=-1)
        
        return self.config.cost_weights['kinematics'] * l1_distance
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute kinematic loss"""
        pred_kin = predictions['object_kinematics']
        target_kin = targets.get('kinematics', targets.get('target_kinematics'))

        # Handle 2D targets
        if target_kin.ndim == 2:
            target_kin = target_kin.unsqueeze(1)

        obj_valid = targets.get('obj_valid_mask')

        if obj_valid is not None:
            # New path: use obj_valid_mask for variable T
            pred_real = pred_kin[obj_valid]       # [N_real, D]
            target_real = target_kin[obj_valid]   # [N_real, D]

            if pred_real.numel() == 0:
                return torch.tensor(0.0, device=pred_kin.device)
        else:
            # Backward compatible path: use [:num_targets] slicing
            num_targets = target_kin.shape[1]
            pred_real = pred_kin[:, :num_targets, :].reshape(-1, pred_kin.shape[-1])
            target_real = target_kin.reshape(-1, target_kin.shape[-1])

        # Get loss type and weight
        if 'l1' in self.config.loss_weights:
            loss = F.l1_loss(pred_real, target_real)
            weight = self.config.get_loss_weight('l1')
        elif 'mse' in self.config.loss_weights:
            loss = F.mse_loss(pred_real, target_real)
            weight = self.config.get_loss_weight('mse')
        else:
            loss = F.smooth_l1_loss(pred_real, target_real)
            weight = self.config.get_loss_weight('smooth_l1')

        return weight * loss
    
    def create_test_datasets(self, file: h5py.File, number_events: int):
        """Create HDF5 datasets for kinematic predictions"""
        D = 4  # (pt, eta, phi, mass)
        
        file.create_dataset(
            "target_kinematics",
            shape=(number_events, self.config.max_objects, D),
            dtype='float32'
        )
        file.create_dataset(
            "predicted_kinematics",
            shape=(number_events, self.config.max_objects, D),
            dtype='float32'
        )
    
    def save_test_predictions(
        self,
        file: h5py.File,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        start_idx: int,
        batch_size: int
    ):
        """Save kinematic predictions to HDF5"""
        pred_kin = predictions['object_kinematics'].float().cpu().numpy()
        target_kin = (targets.get('kinematics') or targets.get('target_kinematics')).float().cpu().numpy()
        
        # Handle 2D targets
        if target_kin.ndim == 2:
            target_kin = target_kin[:, None, :]
        
        # Truncate to max_objects (predictions may have Q > max_objects after padding)
        M = self.config.max_objects
        pred_kin = pred_kin[:, :M, :]
        target_kin = target_kin[:, :M, :]

        # Save to HDF5
        end_idx = start_idx + batch_size
        file["target_kinematics"][start_idx:end_idx] = target_kin
        file["predicted_kinematics"][start_idx:end_idx] = pred_kin


class ClassificationTask(BaseTask):
    """Task for object classification"""
    
    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute classification cost"""
        if 'class_logits' not in predictions or 'classes' not in targets:
            B, num_queries = predictions[list(predictions.keys())[0]].shape[:2]
            num_targets = targets[list(targets.keys())[0]].shape[1]
            return torch.zeros(B, num_queries, num_targets, 
                             device=predictions[list(predictions.keys())[0]].device)
        
        pred_logits = predictions['class_logits']
        target_classes = targets['classes']
        
        pred_probs = pred_logits.log_softmax(dim=-1)
        
        B, num_queries, num_classes = pred_logits.shape
        num_targets = target_classes.shape[1]
        
        target_expanded = target_classes.unsqueeze(1).expand(B, num_queries, num_targets)
        pred_expanded = pred_probs.unsqueeze(2).expand(B, num_queries, num_targets, num_classes)
        
        cost = -torch.gather(pred_expanded, dim=3, index=target_expanded.unsqueeze(-1)).squeeze(-1)
        
        return self.config.cost_weights['class'] * cost
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute classification loss"""
        if 'class_logits' not in predictions or 'classes' not in targets:
            return torch.tensor(0.0, device=predictions[list(predictions.keys())[0]].device)
        
        pred_logits = predictions['class_logits']
        target_classes = targets['classes']
        
        if valid_mask is not None:
            pred_logits = pred_logits[valid_mask]
            target_classes = target_classes[valid_mask]
        
        return F.cross_entropy(pred_logits, target_classes)
    
    def create_test_datasets(self, file: h5py.File, number_events: int):
        """Create HDF5 datasets for classification predictions"""
        num_classes = 2  # Adjust based on your task
        
        file.create_dataset(
            "target_classes",
            shape=(number_events, self.config.max_objects),
            dtype='int32'
        )
        file.create_dataset(
            "predicted_class_logits",
            shape=(number_events, self.config.max_objects, num_classes),
            dtype='float32'
        )
        file.create_dataset(
            "predicted_classes",
            shape=(number_events, self.config.max_objects),
            dtype='int32'
        )
    
    def save_test_predictions(
        self,
        file: h5py.File,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        start_idx: int,
        batch_size: int
    ):
        """Save classification predictions to HDF5"""
        if 'class_logits' not in predictions or 'classes' not in targets:
            return
        
        pred_logits = predictions['class_logits'].float().cpu().numpy()
        pred_classes = predictions['class_logits'].argmax(dim=-1).float().cpu().numpy()
        target_classes = targets['classes'].float().cpu().numpy()
        
        # Truncate to max_objects (predictions may have Q > max_objects after padding)
        M = self.config.max_objects
        pred_logits = pred_logits[:, :M, :]
        pred_classes = pred_classes[:, :M]
        target_classes = target_classes[:, :M]

        # Save to HDF5
        end_idx = start_idx + batch_size
        file["target_classes"][start_idx:end_idx] = target_classes
        file["predicted_class_logits"][start_idx:end_idx] = pred_logits
        file["predicted_classes"][start_idx:end_idx] = pred_classes



class MulticlassClassificationTask(BaseTask):
    """
    Task for multiclass (single-label, mutually exclusive) object classification.

    Each object is assigned exactly one class from C possibilities.
    At inference, the predicted class is argmax over logits.

    Predictions:
        predictions['class_logits']: [B, Q, C]  (raw, pre-softmax)

    Targets:
        targets['classes']: [B, T]  (integer class indices in [0, C))

    Cost (for Hungarian matching):
        Negative log-probability of the target class under log-softmax.
        This is the standard DETR classification cost.

    Loss (after matching):
        Weighted combination of:
          - 'ce'      : cross-entropy with optional label smoothing and class weights
          - 'focal'   : focal loss variant for class imbalance

    Config keys (all in TaskConfig.loss_weights):
        'ce'            : weight for cross-entropy loss          (default 1.0)
        'focal'         : weight for focal loss                  (default 0.0)
        'focal_gamma'   : focusing parameter γ                   (default 2.0)
        'label_smooth'  : smoothing factor ε ∈ [0, 1)           (default 0.0)

    Config keys (TaskConfig.output_dims):
        'class_logits'  : number of classes C  (required for HDF5 I/O)

    Config keys (TaskConfig.cost_weights):
        'class'         : scalar weight on the matching cost     (required)

    Class weights for imbalance (optional):
        Pass a 1-D float tensor of shape [C] via the constructor argument
        `class_weights`. These are applied to both CE and focal losses.

    Accuracy tracking:
        Call task.get_accuracy_stats() after training steps to retrieve
        top-1 and top-k accuracy accumulated since the last reset.
        Call task.reset_accuracy_stats() to clear accumulators.

    Example config:
        TaskConfig(
            name="multiclass",
            output_names=["class_logits"],
            output_dims={"class_logits": 10},
            cost_weights={"class": 1.0},
            loss_weights={
                "ce": 1.0,
                "focal": 0.0,
                "focal_gamma": 2.0,
                "label_smooth": 0.1,
            },
            max_objects=10,
        )
    """

    def __init__(
        self,
        config: TaskConfig,
        class_weights: Optional[torch.Tensor] = None,
        topk: int = 3,
    ):
        """
        Args:
            config:        TaskConfig as described above.
            class_weights: Optional [C] float tensor for class imbalance.
                           Will be registered as a buffer so it moves with .to(device).
            topk:          k for top-k accuracy tracking. Must satisfy k < C.
        """
        super().__init__(config)

        if "class" not in config.cost_weights:
            raise ValueError("TaskConfig.cost_weights must contain 'class'.")
        if "class_logits" not in config.output_dims:
            raise ValueError("TaskConfig.output_dims must contain 'class_logits' (= C).")

        self.num_classes: int = config.output_dims["class_logits"]
        self.topk: int = topk

        if class_weights is not None:
            if class_weights.shape != (self.num_classes,):
                raise ValueError(
                    f"class_weights must have shape ({self.num_classes},), "
                    f"got {tuple(class_weights.shape)}"
                )
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

        # Accuracy accumulators — GPU buffers, .item() deferred to getter
        self.register_buffer('_correct_top1_buf', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('_correct_topk_buf', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('_total_buf', torch.zeros(1, dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------
    # Cost (used by Hungarian matcher)
    # ------------------------------------------------------------------

    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Pairwise negative log-probability cost.

        For each (query q, target t) pair:
            cost[b, q, t] = -log_softmax(logits[b, q])[class[b, t]]

        This is the standard DETR matching cost for classification.

        Args:
            predictions['class_logits']: [B, Q, C]
            targets['classes']:          [B, T]  int64

        Returns:
            cost_matrix: [B, Q, T]
        """
        if "class_logits" not in predictions or "classes" not in targets:
            # No-op: return zero cost so other tasks drive matching
            first = next(iter(predictions.values()))
            B, Q = first.shape[:2]
            T = next(iter(targets.values())).shape[1]
            return torch.zeros(B, Q, T, device=first.device)

        pred_logits = predictions["class_logits"]          # [B, Q, C]
        target_classes = targets["classes"].long()         # [B, T]

        B, Q, C = pred_logits.shape
        T = target_classes.shape[1]

        log_probs = pred_logits.log_softmax(dim=-1)        # [B, Q, C]

        # Gather log-prob of each target class for every query
        # target_expanded: [B, Q, T]
        target_expanded = (
            target_classes.unsqueeze(1)                    # [B, 1, T]
            .expand(B, Q, T)
        )
        # log_probs_expanded: [B, Q, T, C]  → gather on last dim → [B, Q, T]
        cost = -torch.gather(
            log_probs.unsqueeze(2).expand(B, Q, T, C),
            dim=3,
            index=target_expanded.unsqueeze(-1),
        ).squeeze(-1)

        return self.config.cost_weights["class"] * cost

    # ------------------------------------------------------------------
    # Loss (used after matching)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multiclass classification loss and update accuracy stats.

        Args:
            predictions['class_logits']: [B, Q, C]
            targets['classes']:          [B, T]  int64, matched order
            valid_mask:                  [B, T]  bool — which objects are real
                                         (as opposed to padding). If None, all
                                         objects are treated as valid.

        Returns:
            Scalar loss tensor.
        """
        if "class_logits" not in predictions or "classes" not in targets:
            first = next(iter(predictions.values()))
            return torch.tensor(0.0, device=first.device)

        pred_logits = predictions["class_logits"]   # [B, Q, C]
        target_cls  = targets["classes"].long()     # [B, T]

        B, Q, C = pred_logits.shape
        T = target_cls.shape[1]

        # Align queries to matched targets
        pred_logits = pred_logits[:, :T, :]         # [B, T, C]

        # Build validity mask: [B, T] bool
        if valid_mask is not None:
            # valid_mask may be [B, N_particles]; take first T entries or adapt
            mask = valid_mask[:, :T].bool()         # [B, T]
        else:
            mask = torch.ones(B, T, dtype=torch.bool, device=pred_logits.device)

        # Flatten for loss computation
        pred_flat   = pred_logits[mask]             # [N_valid, C]
        target_flat = target_cls[mask]              # [N_valid]

        if pred_flat.numel() == 0:
            return pred_logits.new_tensor(0.0)

        # --- Update accuracy stats (no_grad, doesn't affect graph) ----------
        with torch.no_grad():
            self._update_accuracy(pred_flat, target_flat)

        # --- Loss components -------------------------------------------------
        ce_weight    = self.config.get_loss_weight("ce")
        focal_weight = self.config.get_loss_weight("focal")
        smooth_eps   = float(self.config.loss_weights.get("label_smooth", 0.0))

        total_loss = pred_logits.new_tensor(0.0)

        if ce_weight > 0.0:
            ce_loss = self._ce_loss(pred_flat, target_flat, smooth_eps)
            total_loss = total_loss + ce_weight * ce_loss

        if focal_weight > 0.0:
            focal_loss = self._focal_loss(pred_flat, target_flat)
            total_loss = total_loss + focal_weight * focal_loss

        return total_loss

    # ------------------------------------------------------------------
    # Private loss helpers
    # ------------------------------------------------------------------

    def _ce_loss(
        self,
        pred_logits: torch.Tensor,   # [N, C]
        target_cls:  torch.Tensor,   # [N]
        smooth_eps:  float,
    ) -> torch.Tensor:
        """
        Cross-entropy with optional label smoothing and class weights.

        Label smoothing (Szegedy et al., 2016) replaces the hard target
        distribution with:
            y_smooth = (1 - ε) * y_hard + ε / C

        which is equivalent to PyTorch's built-in label_smoothing parameter.
        """
        return F.cross_entropy(
            pred_logits,
            target_cls,
            weight=self.class_weights,      # None or [C]
            label_smoothing=smooth_eps,
        )

    def _focal_loss(
        self,
        pred_logits: torch.Tensor,   # [N, C]
        target_cls:  torch.Tensor,   # [N]
    ) -> torch.Tensor:
        """
        Multiclass focal loss (Lin et al., 2017).

            FL(p_t) = -(1 - p_t)^γ * log(p_t)

        where p_t is the softmax probability of the true class.

        Optionally weighted by class_weights if provided.

        References:
            Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
            https://arxiv.org/abs/1708.02002
        """
        gamma = float(self.config.loss_weights.get("focal_gamma", 2.0))

        log_probs  = F.log_softmax(pred_logits, dim=-1)        # [N, C]
        probs      = log_probs.exp()                            # [N, C]

        # p_t: probability of the true class per sample
        log_p_t = log_probs.gather(1, target_cls.unsqueeze(1)).squeeze(1)  # [N]
        p_t     = probs.gather(1, target_cls.unsqueeze(1)).squeeze(1)      # [N]

        focal_factor = (1.0 - p_t) ** gamma                    # [N]
        focal_elem   = -focal_factor * log_p_t                 # [N]

        # Apply class weights if present
        if self.class_weights is not None:
            w = self.class_weights[target_cls]                  # [N]
            focal_elem = focal_elem * w

        return focal_elem.mean()

    # ------------------------------------------------------------------
    # Accuracy tracking
    # ------------------------------------------------------------------

    def _update_accuracy(
        self,
        pred_logits: torch.Tensor,   # [N, C]  — detached by caller's no_grad
        target_cls:  torch.Tensor,   # [N]
    ):
        """Accumulate top-1 and top-k correct counts (no .item() — deferred to getter)."""
        if not getattr(self, '_stats_enabled', True):
            return
        N = target_cls.shape[0]

        # Top-1
        top1_preds = pred_logits.argmax(dim=-1)                 # [N]
        self._correct_top1_buf += (top1_preds == target_cls).sum()

        # Top-k (only meaningful if k < C)
        if self.topk < self.num_classes:
            topk_preds = pred_logits.topk(self.topk, dim=-1).indices  # [N, k]
            target_exp = target_cls.unsqueeze(1).expand_as(topk_preds)
            self._correct_topk_buf += (topk_preds == target_exp).any(dim=-1).sum()
        else:
            # k >= C means top-k accuracy is always 1.0
            self._correct_topk_buf += N

        self._total_buf += N

    def get_accuracy_stats(self) -> Dict[str, float]:
        """
        Return accumulated top-1 and top-k accuracy since last reset.
        (.item() called here, once per epoch)

        Returns:
            {
                'top1_accuracy': float in [0, 1],
                'topk_accuracy': float in [0, 1],
                'topk':          int (the k used),
                'total_samples': int,
            }
        """
        total = self._total_buf.item()
        if total == 0:
            return {
                "top1_accuracy": 0.0,
                "topk_accuracy": 0.0,
                "topk": self.topk,
                "total_samples": 0,
            }
        return {
            "top1_accuracy": self._correct_top1_buf.item() / total,
            "topk_accuracy": self._correct_topk_buf.item() / total,
            "topk": self.topk,
            "total_samples": total,
        }

    def reset_accuracy_stats(self):
        """Reset accuracy accumulators. Call at the start of each epoch."""
        self._correct_top1_buf.zero_()
        self._correct_topk_buf.zero_()
        self._total_buf.zero_()

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------

    def create_test_datasets(self, file: h5py.File, number_events: int):
        """
        Create HDF5 datasets for classification predictions.

        Datasets:
          - target_classes              [N, max_objects]     int32
          - predicted_class_logits      [N, max_objects, C]  float32
          - predicted_class_probs       [N, max_objects, C]  float32
          - predicted_classes           [N, max_objects]     int32
        """
        C = self.num_classes
        M = self.config.max_objects

        file.create_dataset(
            "target_classes",
            shape=(number_events, M),
            dtype="int32",
        )
        file.create_dataset(
            "predicted_class_logits",
            shape=(number_events, M, C),
            dtype="float32",
        )
        file.create_dataset(
            "predicted_class_probs",
            shape=(number_events, M, C),
            dtype="float32",
        )
        file.create_dataset(
            "predicted_classes",
            shape=(number_events, M),
            dtype="int32",
        )

    def save_test_predictions(
        self,
        file: h5py.File,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        start_idx: int,
        batch_size: int,
    ):
        """Save classification predictions to HDF5."""
        if "class_logits" not in predictions or "classes" not in targets:
            return

        pred_logits = predictions["class_logits"]           # [B, Q, C]
        pred_probs  = pred_logits.softmax(dim=-1)           # [B, Q, C]
        pred_cls    = pred_logits.argmax(dim=-1)            # [B, Q]
        target_cls  = targets["classes"].long()             # [B, T]

        # Truncate to max_objects (predictions may have Q > max_objects after padding)
        M = self.config.max_objects
        pred_logits = pred_logits[:, :M, :].float().cpu().numpy()
        pred_probs  = pred_probs[:, :M, :].float().cpu().numpy()
        pred_cls    = pred_cls[:, :M].float().cpu().numpy()
        target_cls  = target_cls[:, :M].float().cpu().numpy()

        end_idx = start_idx + batch_size
        file["target_classes"][start_idx:end_idx]         = target_cls
        file["predicted_class_logits"][start_idx:end_idx] = pred_logits
        file["predicted_class_probs"][start_idx:end_idx]  = pred_probs
        file["predicted_classes"][start_idx:end_idx]      = pred_cls


class ObjectnessTask(BaseTask):
    """
    Binary classification: is this query a real object?
    Runs over ALL Q queries (this is the object-count signal).

    Predictions: "objectness_logit" [B, Q, 1]
    Targets: derived from targets["classes"] — real=1.0, null=0.0
    """

    def __init__(self, config: TaskConfig, null_weight: float = 0.1):
        super().__init__(config)
        self.null_weight = null_weight
        # Detection stats accumulators — GPU buffers so .item() is deferred to
        # get_detection_stats() (called once per epoch, not per step)
        self.register_buffer('_tp_buf', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('_fp_buf', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('_fn_buf', torch.zeros(1, dtype=torch.long), persistent=False)

    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Cost: -log p(real) broadcast to [B, Q, T].
        All real targets have objectness=1 so cost is uniform across T columns.
        Biases the matcher toward queries already predicting "real".
        """
        if 'objectness_logit' not in predictions:
            first = next(iter(predictions.values()))
            B, Q = first.shape[:2]
            T = next(iter(targets.values())).shape[1]
            return torch.zeros(B, Q, T, device=first.device)

        pred_logit = predictions['objectness_logit']  # [B, Q, 1]
        prob_real = pred_logit.squeeze(-1).sigmoid()   # [B, Q]
        neg_log_prob = -torch.log(prob_real.clamp(min=1e-8))  # [B, Q]

        # Determine T from targets
        first_target = next(iter(targets.values()))
        T = first_target.shape[1]

        # Broadcast: [B, Q] -> [B, Q, T]
        cost = neg_log_prob.unsqueeze(-1).expand(-1, -1, T)

        return self.config.cost_weights.get('objectness', 1.0) * cost

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Weighted BCE over ALL Q queries.
        Real slots (obj_valid=True)  -> weight = 1.0
        Null slots (obj_valid=False) -> weight = null_weight
        """
        if 'objectness_logit' not in predictions:
            first = next(iter(predictions.values()))
            return torch.tensor(0.0, device=first.device)

        pred_logit = predictions['objectness_logit'].squeeze(-1)  # [B, Q]

        obj_valid = targets.get('obj_valid_mask')
        if obj_valid is None:
            # Backward compat: all queries are real
            target_obj = torch.ones_like(pred_logit)
            weight = torch.ones_like(pred_logit)
        else:
            target_obj = obj_valid.float()  # [B, Q]
            # Adaptive null weighting: scale null weight by n_real/Q per event
            # so the total null contribution stays proportional to the real signal.
            # T_i=4,Q=5 → ratio=0.8; T_i=1,Q=5 → ratio=0.2 (5× reduction)
            n_real = obj_valid.sum(dim=1, keepdim=True).float()  # [B, 1]
            Q = obj_valid.shape[1]
            adaptive_null_w = self.null_weight * (n_real / Q).clamp(min=0.01)  # [B, 1]
            weight = torch.where(obj_valid, torch.ones_like(pred_logit), adaptive_null_w.expand_as(pred_logit))

        loss = F.binary_cross_entropy_with_logits(
            pred_logit, target_obj, weight=weight, reduction='mean'
        )

        # Update detection stats
        with torch.no_grad():
            self._update_detection_stats(pred_logit, obj_valid if obj_valid is not None else target_obj.bool())

        loss_weight = self.config.get_loss_weight('objectness')
        return loss_weight * loss

    def _update_detection_stats(self, pred_logit: torch.Tensor, obj_valid: torch.Tensor):
        """Accumulate TP, FP, FN for precision/recall/F1 (no .item() — deferred to getter)."""
        if not getattr(self, '_stats_enabled', True):
            return
        pred_real = (pred_logit.sigmoid() > 0.5)
        self._tp_buf += (pred_real & obj_valid).sum()
        self._fp_buf += (pred_real & ~obj_valid).sum()
        self._fn_buf += (~pred_real & obj_valid).sum()

    def get_detection_stats(self) -> Dict[str, float]:
        """Return precision, recall, F1 since last reset (.item() called here, once per epoch)."""
        tp = self._tp_buf.item()
        fp = self._fp_buf.item()
        fn = self._fn_buf.item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }

    def reset_detection_stats(self):
        """Reset detection stats. Call at the start of each epoch."""
        self._tp_buf.zero_()
        self._fp_buf.zero_()
        self._fn_buf.zero_()

    def create_test_datasets(self, file: h5py.File, number_events: int):
        """Create HDF5 datasets for objectness predictions."""
        M = self.config.max_objects
        file.create_dataset("predicted_objectness_logit", shape=(number_events, M), dtype='float32')
        file.create_dataset("predicted_objectness_prob", shape=(number_events, M), dtype='float32')
        file.create_dataset("target_objectness", shape=(number_events, M), dtype='float32')

    def save_test_predictions(
        self,
        file: h5py.File,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        start_idx: int,
        batch_size: int
    ):
        """Save objectness predictions to HDF5."""
        if 'objectness_logit' not in predictions:
            return

        pred_logit = predictions['objectness_logit'].squeeze(-1)  # [B, Q]
        pred_prob = pred_logit.sigmoid()

        obj_valid = targets.get('obj_valid_mask')
        if obj_valid is not None:
            target_obj = obj_valid.float()
        else:
            target_obj = torch.ones_like(pred_logit)

        M = self.config.max_objects
        end_idx = start_idx + batch_size
        file["predicted_objectness_logit"][start_idx:end_idx] = pred_logit[:, :M].float().cpu().numpy()
        file["predicted_objectness_prob"][start_idx:end_idx] = pred_prob[:, :M].float().cpu().numpy()
        file["target_objectness"][start_idx:end_idx] = target_obj[:, :M].float().cpu().numpy()


class ObjectTypeTask(BaseTask):
    """
    Binary classification: is this real object a top (1) or a W (0)?
    Runs on REAL (matched) slots ONLY. Null slots are excluded entirely.

    Predictions: "type_logit" [B, Q, 1]
    Targets: derived from targets["classes"][obj_valid] — top=1.0, W=0.0
    """

    def __init__(self, config: TaskConfig, top_weight: float = 1.0):
        super().__init__(config)
        self.top_weight = top_weight
        # Accuracy accumulators — GPU buffers, .item() deferred to getter
        self.register_buffer('_correct_buf', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('_total_buf', torch.zeros(1, dtype=torch.long), persistent=False)

    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        For each (query, target) pair, compute BCE between query's type_logit
        and the target's type label (top=1, W=0).
        Called before null padding so targets["classes"] is [B, T] with real objects only.
        Returns [B, Q, T].
        """
        if 'type_logit' not in predictions or 'classes' not in targets:
            first = next(iter(predictions.values()))
            B, Q = first.shape[:2]
            T = next(iter(targets.values())).shape[1]
            return torch.zeros(B, Q, T, device=first.device)

        pred_logit = predictions['type_logit'].squeeze(-1)  # [B, Q]
        target_classes = targets['classes']                  # [B, T]

        B, Q = pred_logit.shape
        T = target_classes.shape[1]

        # Target type labels: top (CLASS_TOP=1) -> 1.0, W (CLASS_W=2) -> 0.0
        type_labels = (target_classes == CLASS_TOP).float()  # [B, T]

        # Expand for pairwise cost: [B, Q, T]
        pred_expanded = pred_logit.unsqueeze(2).expand(B, Q, T)  # [B, Q, T]
        target_expanded = type_labels.unsqueeze(1).expand(B, Q, T)  # [B, Q, T]

        cost = F.binary_cross_entropy_with_logits(
            pred_expanded, target_expanded, reduction='none'
        )

        return self.config.cost_weights.get('type', 1.0) * cost

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Weighted BCE on real slots only.
        top slots -> weight = top_weight
        W slots   -> weight = 1.0
        """
        if 'type_logit' not in predictions:
            first = next(iter(predictions.values()))
            return torch.tensor(0.0, device=first.device)

        pred_logit = predictions['type_logit'].squeeze(-1)  # [B, Q]

        obj_valid = targets.get('obj_valid_mask')
        if obj_valid is None:
            return torch.tensor(0.0, device=pred_logit.device)

        if not obj_valid.any():
            return torch.tensor(0.0, device=pred_logit.device)

        classes = targets['classes']  # [B, Q] (padded, CLASS_NULL for unmatched)

        # Extract real slots only
        real_logits = pred_logit[obj_valid]     # [N_real]
        real_classes = classes[obj_valid]        # [N_real]

        # Type labels: top (CLASS_TOP=1) -> 1.0, W (CLASS_W=2) -> 0.0
        type_labels = (real_classes == CLASS_TOP).float()

        # Per-sample weights
        weight = torch.where(real_classes == CLASS_TOP, self.top_weight, 1.0)

        loss = F.binary_cross_entropy_with_logits(
            real_logits, type_labels, weight=weight, reduction='mean'
        )

        # Update accuracy stats
        with torch.no_grad():
            self._update_accuracy(real_logits, type_labels)

        loss_weight = self.config.get_loss_weight('type')
        return loss_weight * loss

    def _update_accuracy(self, pred_logits: torch.Tensor, type_labels: torch.Tensor):
        """Accumulate binary accuracy on real objects (no .item() — deferred to getter)."""
        if not getattr(self, '_stats_enabled', True):
            return
        preds = (pred_logits.sigmoid() > 0.5).float()
        self._correct_buf += (preds == type_labels).sum().long()
        self._total_buf += type_labels.numel()

    def get_accuracy_stats(self) -> Dict[str, float]:
        """Return accuracy since last reset (.item() called here, once per epoch)."""
        total = self._total_buf.item()
        if total == 0:
            return {'accuracy': 0.0, 'total_samples': 0}
        return {
            'accuracy': self._correct_buf.item() / total,
            'total_samples': total,
        }

    def reset_accuracy_stats(self):
        """Reset accuracy accumulators."""
        self._correct_buf.zero_()
        self._total_buf.zero_()

    def create_test_datasets(self, file: h5py.File, number_events: int):
        """Create HDF5 datasets for type predictions."""
        M = self.config.max_objects
        file.create_dataset("predicted_type_logit", shape=(number_events, M), dtype='float32')
        file.create_dataset("predicted_type_prob", shape=(number_events, M), dtype='float32')
        file.create_dataset("target_type", shape=(number_events, M), dtype='float32')
        file.create_dataset("target_classes", shape=(number_events, M), dtype='int32')

    def save_test_predictions(
        self,
        file: h5py.File,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        start_idx: int,
        batch_size: int
    ):
        """Save type predictions to HDF5."""
        if 'type_logit' not in predictions:
            return

        pred_logit = predictions['type_logit'].squeeze(-1)  # [B, Q]
        pred_prob = pred_logit.sigmoid()

        obj_valid = targets.get('obj_valid_mask')
        classes = targets.get('classes', torch.zeros_like(pred_logit, dtype=torch.long))

        type_labels = (classes == CLASS_TOP).float()

        M = self.config.max_objects
        end_idx = start_idx + batch_size
        file["predicted_type_logit"][start_idx:end_idx] = pred_logit[:, :M].float().cpu().numpy()
        file["predicted_type_prob"][start_idx:end_idx] = pred_prob[:, :M].float().cpu().numpy()
        file["target_type"][start_idx:end_idx] = type_labels[:, :M].float().cpu().numpy()
        file["target_classes"][start_idx:end_idx] = classes[:, :M].float().cpu().numpy()