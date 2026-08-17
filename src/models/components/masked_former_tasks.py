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
    head_norm: bool = False  # LayerNorm before prediction head MLP
    mask_embed_head: bool = False  # mask outputs: learned MLP on queries before the einsum
    validity_key: Optional[str] = None  # Per-task object validity, e.g. top_valid or w_valid
    bce_reference_particles: Optional[float] = None  # Fixed BCE normalization reference
    rank_weight: float = 0.0  # Boundary-ranking surrogate for exact fixed-cardinality match
    rank_margin: float = 1.0
    rank_temperature: float = 0.5
    
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
        # Per-task loss scales for smooth phase transitions (0.0 → 1.0 ramp).
        # Not persisted — reconstructed each epoch by the trainer.
        self._loss_scales: Dict[str, float] = {}

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

    def set_loss_scale(self, task_name: str, scale: float):
        """Set a multiplicative loss scale for a task (used for smooth phase transitions)."""
        self._loss_scales[task_name] = scale

    def get_loss_scale(self, task_name: str) -> float:
        """Get the current loss scale for a task (default 1.0)."""
        return self._loss_scales.get(task_name, 1.0)

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

            loss_scale = self.get_loss_scale(task_name)
            task_loss = layer_weight * loss_scale * task_loss

            per_task_losses[task_name] = task_loss.detach() if torch.is_tensor(task_loss) else torch.tensor(float(task_loss))
            total_loss += task_loss

        return total_loss, per_task_losses


class MaskReconstructionTask(BaseTask):
    """Task for mask reconstruction"""

    def __init__(
        self,
        config: TaskConfig,
        null_mask_penalty: float = 0.1,
        pred_key: str = 'mask_predictions',
        target_key: str = 'jet_mask_true',
        bce_pos_weight: bool = False,
        cost_bce_weight: float = 0.0,
        validity_key: Optional[str] = None,
        bce_reference_particles: Optional[float] = None,
        rank_weight: Optional[float] = None,
        rank_margin: Optional[float] = None,
        rank_temperature: Optional[float] = None,
    ):
        super().__init__(config)
        self.pred_key = pred_key
        self.target_key = target_key
        self.eps = 1e-6  # Increased from 1e-8 for better numerical stability in Dice loss
        self.null_mask_penalty = null_mask_penalty
        self.bce_pos_weight = bce_pos_weight
        # Weight on a BCE term added to the Hungarian matching cost (0.0 = Dice-only,
        # bitwise-identical to previous behaviour). Complements the Dice cost with a
        # per-particle assignment signal so matching is less degenerate on small masks.
        self.cost_bce_weight = cost_bce_weight
        # Chain mode supplies separate validity for top and W slots.  Infer the
        # conventional key for the two built-in mask targets, while allowing
        # callers to provide another per-task validity field.
        self.validity_key = validity_key or config.validity_key
        if self.validity_key is None:
            self.validity_key = {
                'jet_mask_true': 'top_valid',
                'jet_mask_true_W': 'w_valid',
            }.get(target_key)
        bce_reference_particles = (
            bce_reference_particles
            if bce_reference_particles is not None
            else config.bce_reference_particles
        )
        if bce_reference_particles is not None and bce_reference_particles <= 0:
            raise ValueError("bce_reference_particles must be positive when set")
        # None preserves the historical mean over valid particles.  A fixed
        # reference makes BCE scale comparable when batches have different
        # amounts of padding or different particle multiplicities.
        self.bce_reference_particles = bce_reference_particles
        self.rank_weight = config.rank_weight if rank_weight is None else float(rank_weight)
        self.rank_margin = config.rank_margin if rank_margin is None else float(rank_margin)
        self.rank_temperature = (
            config.rank_temperature if rank_temperature is None else float(rank_temperature)
        )
        if self.rank_weight < 0:
            raise ValueError("rank_weight must be non-negative")
        if self.rank_temperature <= 0:
            raise ValueError("rank_temperature must be positive")
        # 0.0 = suppressed (during mask-only pretraining), 1.0 = full penalty.
        # Ramped from 0→1 during phase transition to avoid "predict nothing" snap-on.
        self.null_penalty_scale = 1.0

        # Per-class Dice accumulators — GPU buffers, .item() deferred to getter
        self.register_buffer('_top_dice_sum', torch.zeros(1), persistent=False)
        self.register_buffer('_top_dice_count', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('_w_dice_sum', torch.zeros(1), persistent=False)
        self.register_buffer('_w_dice_count', torch.zeros(1, dtype=torch.long), persistent=False)

    def _particle_validity(
        self,
        targets: Dict[str, torch.Tensor],
        batch_size: int,
        num_particles: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return [B, N] particle validity, or None for legacy callers."""
        valid = targets.get('jet_valid_mask')
        if valid is None:
            return None
        if valid.ndim != 2 or valid.shape != (batch_size, num_particles):
            raise ValueError(
                f"jet_valid_mask must have shape {(batch_size, num_particles)}, "
                f"got {tuple(valid.shape)}"
            )
        return valid.to(device=device).bool()

    def _object_validity(
        self,
        targets: Dict[str, torch.Tensor],
        batch_size: int,
        num_queries: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return this task's [B, Q] real-slot mask, including null padding."""
        valid = None
        if self.validity_key is not None:
            valid = targets.get(self.validity_key)
        if valid is None:
            valid = targets.get('obj_valid_mask')
        if valid is None:
            return None
        if valid.ndim != 2 or valid.shape[0] != batch_size:
            raise ValueError(
                f"{self.validity_key or 'obj_valid_mask'} must be a [B, Q] mask, "
                f"got {tuple(valid.shape)}"
            )
        valid = valid.to(device=device).bool()
        if valid.shape[1] < num_queries:
            valid = F.pad(valid, (0, num_queries - valid.shape[1]), value=False)
        return valid[:, :num_queries]

    def _bce_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stable fp32 BCE with optional particle masking/reference denominator."""
        logits_f = logits.float()
        target_f = target.float()
        bce = F.binary_cross_entropy_with_logits(
            logits_f,
            target_f,
            pos_weight=None if pos_weight is None else pos_weight.float(),
            reduction='none',
        )
        if valid is None:
            if self.bce_reference_particles is None:
                return bce.mean()
            return (
                bce.sum(dim=-1) / logits_f.new_tensor(self.bce_reference_particles)
            ).mean()

        valid_f = valid.float()
        bce = bce * valid_f
        if self.bce_reference_particles is None:
            denom = valid_f.sum(dim=-1).clamp(min=1.0)
        else:
            denom = logits_f.new_tensor(self.bce_reference_particles)
        # Normalize each mask independently, then average masks. This avoids
        # large events dominating solely because they contain more valid slots.
        return (bce.sum(dim=-1) / denom).mean()

    def _ranking_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        particle_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Smoothly enforce min-positive score > max-negative score.

        This is an exact-match-aligned surrogate for fixed-cardinality decoding.
        It treats positive jets as an unordered set: only the lowest positive
        and highest negative scores determine the boundary.
        """
        if self.rank_weight == 0.0:
            return logits.sum() * 0.0
        if particle_valid is None:
            particle_valid = torch.ones_like(target, dtype=torch.bool)
        else:
            particle_valid = particle_valid.bool()

        target = target > 0.5
        has_pos = (target & particle_valid).any(dim=-1)
        has_neg = ((~target) & particle_valid).any(dim=-1)
        active = has_pos & has_neg
        if not active.any().item():
            return logits.sum() * 0.0

        tau = logits.new_tensor(self.rank_temperature)
        pos_scores = (-logits).masked_fill(~(target & particle_valid), float('-inf'))
        neg_scores = logits.masked_fill(~((~target) & particle_valid), float('-inf'))
        smooth_min_pos = -tau * torch.logsumexp(pos_scores / tau, dim=-1)
        smooth_max_neg = tau * torch.logsumexp(neg_scores / tau, dim=-1)
        losses = F.softplus(
            logits.new_tensor(self.rank_margin) + smooth_max_neg - smooth_min_pos
        )
        return losses[active].mean()

    @staticmethod
    def _align_target_slots(
        target_masks: torch.Tensor,
        num_queries: int,
    ) -> torch.Tensor:
        """Pad legacy [B, T, N] targets to Q without changing their values."""
        if target_masks.shape[1] < num_queries:
            return F.pad(target_masks, (0, 0, 0, num_queries - target_masks.shape[1]))
        return target_masks[:, :num_queries]
    
    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute mask cost using Dice coefficient (+ optional BCE term)"""
        logits = predictions[self.pred_key]
        pred_masks = logits.sigmoid()
        target_masks = targets[self.target_key].float()

        # Handle 2D targets
        if target_masks.ndim == 2:
            target_masks = target_masks.unsqueeze(1)

        B, num_queries, N = pred_masks.shape
        if target_masks.shape[0] != B or target_masks.shape[-1] != N:
            raise ValueError(
                f"{self.target_key} must have shape [B, T, {N}], "
                f"got {tuple(target_masks.shape)}"
            )
        particle_valid = self._particle_validity(targets, B, N, logits.device)
        if particle_valid is not None:
            particle_valid_f = particle_valid[:, None, None, :].float()
        else:
            particle_valid_f = 1.0

        # Compute pairwise Dice
        pred_expanded = pred_masks.unsqueeze(2)
        target_expanded = target_masks.unsqueeze(1)

        intersection = (pred_expanded * target_expanded * particle_valid_f).sum(dim=-1)
        pred_sizes = (pred_masks * particle_valid_f.squeeze(2)).sum(dim=-1, keepdim=True)
        target_sizes = (target_masks * particle_valid_f.squeeze(1)).sum(dim=-1).unsqueeze(1)

        dice = (2 * intersection) / (pred_sizes + target_sizes + self.eps)
        w_dice = self.config.cost_weights['mask']
        cost = w_dice * (1 - dice)

        # Optional per-particle BCE matching term. Runs under no_grad (matching);
        # cast to float to avoid bf16 saturation of logsigmoid. Normalised by the
        # number of valid particles so it scales O(1) like the Dice term.
        if self.cost_bce_weight > 0:
            logits_f = logits.float()
            valid = (particle_valid if particle_valid is not None
                     else logits_f.new_ones(B, N))
            valid = valid.float()
            tgt = target_masks                                  # [B, T, N]
            pos = -F.logsigmoid(logits_f) * valid[:, None, :]   # [B, Q, N]
            neg = -F.logsigmoid(-logits_f) * valid[:, None, :]  # [B, Q, N]
            bce_cost = (torch.einsum('bqn,btn->bqt', pos, tgt)
                        + torch.einsum('bqn,btn->bqt', neg, (1 - tgt) * valid[:, None, :]))
            if self.bce_reference_particles is None:
                bce_denom = valid.sum(-1).clamp(min=1)
            else:
                bce_denom = logits_f.new_tensor(self.bce_reference_particles).expand(B)
            bce_cost = bce_cost / bce_denom[:, None, None]
            cost = cost + self.cost_bce_weight * bce_cost

        # A chain can be matchable because only the other half is present.
        # Keep a missing top/W half neutral in the shared permutation cost;
        # treating its zero mask as a real target would prefer empty queries.
        task_valid = targets.get(self.validity_key) if self.validity_key else None
        if task_valid is None:
            task_valid = targets.get('obj_valid_mask')
        if task_valid is not None:
            task_valid = task_valid.to(device=cost.device).bool()
            num_targets = target_masks.shape[1]
            if task_valid.ndim != 2 or task_valid.shape[0] != B:
                raise ValueError(
                    f"{self.validity_key or 'obj_valid_mask'} must have shape [B, T], "
                    f"got {tuple(task_valid.shape)}"
                )
            if task_valid.shape[1] < num_targets:
                task_valid = F.pad(task_valid, (0, num_targets - task_valid.shape[1]), value=False)
            cost = cost * task_valid[:, None, :num_targets].float()

        return cost
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute mask loss (Dice + BCE)"""
        pred_masks = predictions[self.pred_key]
        target_masks = targets[self.target_key]

        if target_masks.ndim == 2:
            target_masks = target_masks.unsqueeze(1)

        B, num_queries, N = pred_masks.shape
        particle_valid = self._particle_validity(targets, B, N, pred_masks.device)
        if particle_valid is None and valid_mask is not None:
            if valid_mask.ndim != 2 or valid_mask.shape != (B, N):
                raise ValueError(
                    f"valid_mask must have shape {(B, N)}, got {tuple(valid_mask.shape)}"
                )
            particle_valid = valid_mask.to(device=pred_masks.device).bool()
        obj_valid = self._object_validity(targets, B, num_queries, pred_masks.device)

        if obj_valid is not None:
            # --- Variable-T path: use this task's validity key for real slots ---
            total_loss = pred_masks.sum() * 0.0
            target_masks = self._align_target_slots(target_masks, num_queries)

            if obj_valid.any().item():
                real_pred_logits = pred_masks[obj_valid]       # [N_real, N]
                real_tgt = target_masks[obj_valid]             # [N_real, N]

                real_pred_probs = real_pred_logits.sigmoid()
                real_tgt_float = real_tgt.float()

                if particle_valid is not None:
                    real_vm = particle_valid.unsqueeze(1).expand(B, num_queries, N)[obj_valid]
                    real_pred_probs = real_pred_probs * real_vm
                    real_tgt_float = real_tgt_float * real_vm
                else:
                    real_vm = None

                # Dice loss
                intersection = (real_pred_probs * real_tgt_float).sum(dim=-1)
                pred_sum = real_pred_probs.sum(dim=-1)
                target_sum = real_tgt_float.sum(dim=-1)
                dice = (2 * intersection) / (pred_sum + target_sum + self.eps)
                dice_loss = (1 - dice).mean()

                # BCE loss — optional per-mask positive weighting to
                # counteract signal dilution in high-multiplicity events.
                # pos_weight = n_neg / n_pos per mask so that signal and
                # background particles contribute equally to the gradient.
                if self.bce_pos_weight:
                    n_pos = real_tgt_float.sum(dim=-1, keepdim=True).clamp(min=1)  # [N_real, 1]
                    if real_vm is not None:
                        n_total = real_vm.sum(dim=-1, keepdim=True).clamp(min=1)
                    else:
                        n_total = real_tgt_float.new_tensor(N)
                    pw = ((n_total - n_pos) / n_pos).expand_as(real_tgt_float)     # [N_real, N]
                    bce_loss = self._bce_loss(real_pred_logits, real_tgt_float, real_vm, pw)
                else:
                    bce_loss = self._bce_loss(real_pred_logits, real_tgt_float, real_vm)

                dice_weight = self.config.get_loss_weight('dice')
                bce_weight = self.config.get_loss_weight('bce')
                total_loss = dice_weight * dice_loss + bce_weight * bce_loss
                total_loss = total_loss + self.rank_weight * self._ranking_loss(
                    real_pred_logits, real_tgt_float, real_vm
                )

                # Track per-class Dice (top vs W) for monitoring
                if getattr(self, '_stats_enabled', True) and 'classes' in targets:
                    classes = targets['classes']
                    if classes.ndim != 2 or classes.shape[0] != B:
                        raise ValueError(
                            f"classes must have shape [B, Q], got {tuple(classes.shape)}"
                        )
                    if classes.shape[1] < num_queries:
                        classes = F.pad(classes, (0, num_queries - classes.shape[1]), value=CLASS_NULL)
                    classes_real = classes[:, :num_queries][obj_valid]  # [N_real]
                    with torch.no_grad():
                        top_mask_cls = (classes_real == CLASS_TOP)
                        w_mask_cls = (classes_real == CLASS_W)
                        if top_mask_cls.any():
                            self._top_dice_sum += dice[top_mask_cls].sum()
                            self._top_dice_count += top_mask_cls.sum()
                        if w_mask_cls.any():
                            self._w_dice_sum += dice[w_mask_cls].sum()
                            self._w_dice_count += w_mask_cls.sum()
                elif getattr(self, '_stats_enabled', True):
                    # Chain mode removes ``classes`` after splitting the
                    # targets; the task's validity key still identifies which
                    # per-task Dice accumulator should receive this batch.
                    with torch.no_grad():
                        if self.validity_key == 'top_valid':
                            self._top_dice_sum += dice.sum()
                            self._top_dice_count += dice.numel()
                        elif self.validity_key == 'w_valid':
                            self._w_dice_sum += dice.sum()
                            self._w_dice_count += dice.numel()

            # Null mask penalty: encourage unmatched queries to predict empty masks.
            # Adaptive scaling: reduce penalty when most queries are null so the
            # "predict empty" signal doesn't overwhelm real-object learning.
            # Suppressed during mask-only pretraining to eliminate "predict nothing" signal.
            if (self.validity_key is None and self.null_mask_penalty > 0 and self.null_penalty_scale > 0
                    and (~obj_valid).any().item()):
                null_logits = pred_masks[~obj_valid]           # [N_null, N]
                null_targets = torch.zeros_like(null_logits)
                if particle_valid is not None:
                    null_valid = particle_valid.unsqueeze(1).expand(B, num_queries, N)[~obj_valid]
                else:
                    null_valid = None
                null_loss = self._bce_loss(null_logits, null_targets, null_valid)
                n_real = obj_valid.sum().float()
                n_total = obj_valid.numel()
                adaptive_scale = (n_real / n_total).clamp(min=0.01)
                total_loss = total_loss + self.null_mask_penalty * self.null_penalty_scale * adaptive_scale * null_loss

            return total_loss
        else:
            # --- Backward compatible path: use [:num_targets] slicing ---
            num_targets = target_masks.shape[1]
            pred_masks = pred_masks[:, :num_targets, :]

            pred_probs = pred_masks.sigmoid()
            target_float = target_masks.float()

            if particle_valid is not None:
                valid_mask_expanded = particle_valid.unsqueeze(1).expand_as(pred_probs)
                pred_probs = pred_probs * valid_mask_expanded
                target_float = target_float * valid_mask_expanded

            pred_probs_flat = pred_probs.reshape(-1, N)
            target_float_flat = target_float.reshape(-1, N)
            pred_masks_flat = pred_masks.reshape(-1, N)
            valid_mask_flat = (
                particle_valid.unsqueeze(1).expand(-1, num_targets, -1).reshape(-1, N)
                if particle_valid is not None else None
            )

            # Dice loss
            intersection = (pred_probs_flat * target_float_flat).sum(dim=-1)
            pred_sum = pred_probs_flat.sum(dim=-1)
            target_sum = target_float_flat.sum(dim=-1)
            dice = (2 * intersection) / (pred_sum + target_sum + self.eps)
            dice_loss = 1 - dice

            # BCE loss
            if self.bce_pos_weight:
                n_pos = target_float_flat.sum(dim=-1, keepdim=True).clamp(min=1)
                n_total_flat = target_float_flat.new_tensor(N)
                pw = ((n_total_flat - n_pos) / n_pos).expand_as(target_float_flat)
                bce_loss = self._bce_loss(
                    pred_masks_flat, target_float_flat, valid_mask_flat, pw
                )
            else:
                bce_loss = self._bce_loss(
                    pred_masks_flat, target_float_flat, valid_mask_flat
                )

            dice_weight = self.config.get_loss_weight('dice')
            bce_weight = self.config.get_loss_weight('bce')

            total_loss = dice_weight * dice_loss.mean() + bce_weight * bce_loss
            total_loss = total_loss + self.rank_weight * self._ranking_loss(
                pred_masks_flat, target_float_flat, valid_mask_flat
            )
            return total_loss
    
    def get_detection_stats(self) -> Dict[str, float]:
        """Return per-class Dice since last reset (.item() called here, once per epoch)."""
        stats = {}
        top_count = self._top_dice_count.item()
        w_count = self._w_dice_count.item()
        if top_count > 0:
            stats['top_dice'] = self._top_dice_sum.item() / top_count
        else:
            stats['top_dice'] = 0.0
        if w_count > 0:
            stats['w_dice'] = self._w_dice_sum.item() / w_count
        else:
            stats['w_dice'] = 0.0
        return stats

    def reset_detection_stats(self):
        """Reset per-class Dice accumulators. Call at the start of each epoch."""
        self._top_dice_sum.zero_()
        self._top_dice_count.zero_()
        self._w_dice_sum.zero_()
        self._w_dice_count.zero_()

    def create_test_datasets(
        self, file: h5py.File, number_events: int, num_particles: Optional[int] = None
    ):
        """Create HDF5 datasets for mask predictions"""
        if num_particles is None or int(num_particles) <= 0:
            raise ValueError(
                "Mask test output shape requires the runtime particle count; "
                "pass num_particles from the test dataset."
            )
        N_particles = int(num_particles)
        M = self.config.max_objects

        file.create_dataset(
            "target_masks",
            shape=(number_events, M, N_particles),
            dtype='float32'
        )
        file.create_dataset(
            "predicted_masks_logits",
            shape=(number_events, M, N_particles),
            dtype='float32'
        )
        file.create_dataset(
            "predicted_masks_prob",
            shape=(number_events, M, N_particles),
            dtype='float32'
        )
        file.create_dataset(
            "jet_valid_mask",
            shape=(number_events, N_particles),
            dtype='float32'
        )
        # Per-type slot validity (chain mode): which slots have a real top / real W
        file.create_dataset(
            "slot_valid",
            shape=(number_events, M),
            dtype='bool'
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
        pred_masks_logits = predictions[self.pred_key].float().cpu().numpy()
        pred_masks_prob = predictions[self.pred_key].sigmoid().float().cpu().numpy()
        target_masks = targets[self.target_key].float().cpu().numpy()
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

        # Per-type slot validity: top mask task → top_valid, W mask task → w_valid
        validity_key = 'top_valid' if self.target_key == 'jet_mask_true' else 'w_valid'
        slot_valid = targets.get(validity_key)
        if slot_valid is not None:
            file["slot_valid"][start_idx:end_idx] = slot_valid[:, :M].bool().cpu().numpy()
        else:
            # Fallback: use chain-level obj_valid_mask
            obj_valid = targets.get('obj_valid_mask')
            if obj_valid is not None:
                file["slot_valid"][start_idx:end_idx] = obj_valid[:, :M].bool().cpu().numpy()


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

    Predictions: config.output_names[0] (e.g. "objectness_logit" or "objectness_W_logit") [B, Q, 1]
    Targets: derived from targets["classes"] — real=1.0, null=0.0
    """

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.pred_key = config.output_names[0]  # e.g. 'objectness_logit' or 'objectness_W_logit'
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
        if self.pred_key not in predictions:
            first = next(iter(predictions.values()))
            B, Q = first.shape[:2]
            T = next(iter(targets.values())).shape[1]
            return torch.zeros(B, Q, T, device=first.device)

        pred_logit = predictions[self.pred_key]  # [B, Q, 1]
        prob_real = pred_logit.squeeze(-1).sigmoid()   # [B, Q]
        neg_log_prob = -torch.log(prob_real.clamp(min=1e-8))  # [B, Q]

        # Determine T from targets
        first_target = next(iter(targets.values()))
        T = first_target.shape[1]

        # Broadcast: [B, Q] -> [B, Q, T]
        cost = neg_log_prob.unsqueeze(-1).expand(-1, -1, T)

        # Do not let padded/null target columns influence query permutation.
        # In chain mode this is the union validity mask after target compaction.
        obj_valid = targets.get('obj_valid_mask')
        if obj_valid is not None:
            obj_valid = obj_valid.to(device=cost.device).bool()
            if obj_valid.ndim != 2 or obj_valid.shape[0] != cost.shape[0]:
                raise ValueError(
                    f"obj_valid_mask must have shape [B, T], got {tuple(obj_valid.shape)}"
                )
            if obj_valid.shape[1] < T:
                obj_valid = F.pad(obj_valid, (0, T - obj_valid.shape[1]), value=False)
            cost = cost * obj_valid[:, None, :T].float()

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
        if self.pred_key not in predictions:
            first = next(iter(predictions.values()))
            return torch.tensor(0.0, device=first.device)

        pred_logit = predictions[self.pred_key].squeeze(-1)  # [B, Q]

        obj_valid = targets.get('obj_valid_mask')
        if obj_valid is None:
            target_obj = torch.ones_like(pred_logit)
        else:
            target_obj = obj_valid.float()  # [B, Q]

        loss = F.binary_cross_entropy_with_logits(
            pred_logit, target_obj, reduction='mean'
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
        prefix = f"predicted_{self.config.name}"
        file.create_dataset(f"{prefix}_logit", shape=(number_events, M), dtype='float32')
        file.create_dataset(f"{prefix}_prob", shape=(number_events, M), dtype='float32')
        file.create_dataset(f"target_{self.config.name}", shape=(number_events, M), dtype='float32')

    def save_test_predictions(
        self,
        file: h5py.File,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        start_idx: int,
        batch_size: int
    ):
        """Save objectness predictions to HDF5."""
        if self.pred_key not in predictions:
            return

        pred_logit = predictions[self.pred_key].squeeze(-1)  # [B, Q]
        pred_prob = pred_logit.sigmoid()

        obj_valid = targets.get('obj_valid_mask')
        if obj_valid is not None:
            target_obj = obj_valid.float()
        else:
            target_obj = torch.ones_like(pred_logit)

        M = self.config.max_objects
        prefix = f"predicted_{self.config.name}"
        end_idx = start_idx + batch_size
        file[f"{prefix}_logit"][start_idx:end_idx] = pred_logit[:, :M].float().cpu().numpy()
        file[f"{prefix}_prob"][start_idx:end_idx] = pred_prob[:, :M].float().cpu().numpy()
        file[f"target_{self.config.name}"][start_idx:end_idx] = target_obj[:, :M].float().cpu().numpy()


class ChainStateTask(BaseTask):
    """Three-state detection: absent, W-only, or full top chain."""

    ABSENT, W_ONLY, FULL_TOP = 0, 1, 2

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.pred_key = config.output_names[0]

    @staticmethod
    def target_states(targets, width=None):
        top = targets["top_valid"].bool()
        w = targets["w_valid"].bool()
        if top.shape != w.shape:
            raise ValueError("top_valid and w_valid must have identical shape")
        if (top & ~w).any().item():
            raise ValueError("a full-top target cannot be valid when its W is invalid")
        states = w.long() + top.long()
        if width is not None:
            states = F.pad(states, (0, max(0, width - states.shape[1])), value=0)[:, :width]
        return states

    def compute_cost(self, predictions, targets):
        logits = predictions[self.pred_key]
        states = self.target_states(targets)
        log_probs = logits.log_softmax(dim=-1)
        cost = -log_probs[:, :, None, :].expand(-1, -1, states.shape[1], -1).gather(
            -1, states[:, None, :, None].expand(-1, logits.shape[1], -1, 1)
        ).squeeze(-1)
        valid = (targets["top_valid"] | targets["w_valid"]).to(cost.device)
        return self.config.cost_weights.get("chain_state", 1.0) * cost * valid[:, None, :]

    def compute_loss(self, predictions, targets, valid_mask=None):
        logits = predictions[self.pred_key]
        states = self.target_states(targets, logits.shape[1]).to(logits.device)
        return self.config.get_loss_weight("chain_state") * F.cross_entropy(
            logits.transpose(1, 2), states
        )

    def create_test_datasets(self, file, number_events):
        q = self.config.max_objects
        file.create_dataset("predicted_chain_state_logits", shape=(number_events, q, 3), dtype="f4")
        file.create_dataset("target_chain_state", shape=(number_events, q), dtype="u1")

    def save_test_predictions(self, file, predictions, targets, start_idx, batch_size):
        logits = predictions[self.pred_key]
        states = self.target_states(targets, logits.shape[1])
        stop = start_idx + batch_size
        file["predicted_chain_state_logits"][start_idx:stop] = logits.float().cpu().numpy()
        file["target_chain_state"][start_idx:stop] = states.to(torch.uint8).cpu().numpy()


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


class ChainTypeTask(BaseTask):
    """
    Per-chain binary classification: hadronic (0) vs leptonic (1).
    Used in chain_queries + enable_leptonic mode only.

    Contributes a SOFT matching cost so the matcher preferentially assigns
    chains to same-type targets while keeping free matching (no hard penalty).

    Predictions: 'is_leptonic_logit' [B, Q, 1]
    Targets: 'chain_type' [B, T] int (0=hadronic, 1=leptonic)
    """

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.register_buffer('_correct_buf', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('_total_buf', torch.zeros(1, dtype=torch.long), persistent=False)

    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if 'is_leptonic_logit' not in predictions or 'chain_type' not in targets:
            first = next(iter(predictions.values()))
            B, Q = first.shape[:2]
            T = next(iter(targets.values())).shape[1]
            return torch.zeros(B, Q, T, device=first.device)

        pred_logit = predictions['is_leptonic_logit'].squeeze(-1)  # [B, Q]
        chain_type = targets['chain_type'].float()                  # [B, T]
        B, Q = pred_logit.shape
        T = chain_type.shape[1]

        pred_exp   = pred_logit.unsqueeze(2).expand(B, Q, T)
        target_exp = chain_type.unsqueeze(1).expand(B, Q, T)
        cost = F.binary_cross_entropy_with_logits(pred_exp, target_exp, reduction='none')
        return self.config.cost_weights.get('chain_type', 1.0) * cost

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if 'is_leptonic_logit' not in predictions or 'chain_type' not in targets:
            first = next(iter(predictions.values()))
            return torch.tensor(0.0, device=first.device)

        pred_logit = predictions['is_leptonic_logit'].squeeze(-1)  # [B, Q]
        chain_type = targets['chain_type'].float()                  # [B, Q] post-padding
        obj_valid  = targets.get('obj_valid_mask')

        if obj_valid is None or not obj_valid.any():
            return pred_logit.new_tensor(0.0)

        real_logits = pred_logit[obj_valid]
        real_labels = chain_type[obj_valid]

        loss = F.binary_cross_entropy_with_logits(real_logits, real_labels, reduction='mean')

        with torch.no_grad():
            if getattr(self, '_stats_enabled', True):
                preds = (real_logits.sigmoid() > 0.5).float()
                self._correct_buf += (preds == real_labels).sum().long()
                self._total_buf   += real_labels.numel()

        return self.config.get_loss_weight('chain_type') * loss

    def get_accuracy_stats(self) -> Dict[str, float]:
        total = self._total_buf.item()
        if total == 0:
            return {'accuracy': 0.0, 'total_samples': 0}
        return {'accuracy': self._correct_buf.item() / total, 'total_samples': total}

    def reset_accuracy_stats(self):
        self._correct_buf.zero_()
        self._total_buf.zero_()

    def create_test_datasets(self, file: h5py.File, number_events: int):
        M = self.config.max_objects
        file.create_dataset("predicted_is_leptonic_logit", shape=(number_events, M), dtype='float32')
        file.create_dataset("predicted_is_leptonic_prob",  shape=(number_events, M), dtype='float32')
        file.create_dataset("target_chain_type",           shape=(number_events, M), dtype='int32')

    def save_test_predictions(
        self,
        file: h5py.File,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        start_idx: int,
        batch_size: int,
    ):
        if 'is_leptonic_logit' not in predictions:
            return
        logit = predictions['is_leptonic_logit'].squeeze(-1)  # [B, Q]
        prob  = logit.sigmoid()
        ct    = targets.get('chain_type', torch.zeros_like(logit, dtype=torch.long))
        M     = self.config.max_objects
        end   = start_idx + batch_size
        file["predicted_is_leptonic_logit"][start_idx:end] = logit[:, :M].float().cpu().numpy()
        file["predicted_is_leptonic_prob"][start_idx:end]  = prob[:, :M].float().cpu().numpy()
        file["target_chain_type"][start_idx:end]           = ct[:, :M].float().cpu().numpy()


class NeutrinoRegressionTask(BaseTask):
    """
    Smooth-L1 regression of neutrino truth kinematics for leptonic chains.
    Only fires on chains where chain_type == 1 (leptonic) AND obj_valid == True.

    Optionally adds a soft m(ℓν) ≈ m_W constraint when 'lepton_p4' is provided
    in targets (future extension — currently only regression loss is active).

    Predictions: 'neutrino_pz'  [B, Q, K]  (K=1 for pz-only, or more)
    Targets:     'neutrino_truth' [B, Q, K]  (post-padding)
                 'chain_type'     [B, Q]     (0=had, 1=lep)
    """

    def __init__(self, config: TaskConfig, mw_gev: float = 80.379):
        super().__init__(config)
        self.mw_gev = mw_gev

    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # No direct matching contribution — chain assignment driven by masks + chain_type.
        first = next(iter(predictions.values()))
        B, Q = first.shape[:2]
        T = next(iter(targets.values())).shape[1]
        return torch.zeros(B, Q, T, device=first.device)

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if 'neutrino_pz' not in predictions or 'neutrino_truth' not in targets:
            first = next(iter(predictions.values()))
            return torch.tensor(0.0, device=first.device)

        pred_pz    = predictions['neutrino_pz']         # [B, Q, K]
        nu_truth   = targets['neutrino_truth']           # [B, Q, K]
        chain_type = targets.get('chain_type')           # [B, Q] or None
        obj_valid  = targets.get('obj_valid_mask')       # [B, Q] or None

        B, Q, K = pred_pz.shape

        # Leptonic mask: valid AND leptonic chains
        if chain_type is not None and obj_valid is not None:
            lep_mask = (chain_type.float() > 0.5).bool() & obj_valid  # [B, Q]
        elif obj_valid is not None:
            lep_mask = obj_valid
        else:
            lep_mask = torch.ones(B, Q, dtype=torch.bool, device=pred_pz.device)

        if not lep_mask.any():
            return pred_pz.new_tensor(0.0)

        pred_real  = pred_pz[lep_mask]    # [N_lep, K]
        truth_real = nu_truth[lep_mask]   # [N_lep, K]

        loss = F.smooth_l1_loss(pred_real, truth_real)
        return self.config.get_loss_weight('neutrino') * loss

    def create_test_datasets(self, file: h5py.File, number_events: int):
        K = self.config.output_dims.get('neutrino_pz', 1)
        M = self.config.max_objects
        file.create_dataset("predicted_neutrino_pz",  shape=(number_events, M, K), dtype='float32')
        file.create_dataset("target_neutrino_truth",  shape=(number_events, M, K), dtype='float32')

    def save_test_predictions(
        self,
        file: h5py.File,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        start_idx: int,
        batch_size: int,
    ):
        if 'neutrino_pz' not in predictions or 'neutrino_truth' not in targets:
            return
        pred = predictions['neutrino_pz']                # [B, Q, K]
        truth = targets['neutrino_truth']                 # [B, Q, K]
        M = self.config.max_objects
        end = start_idx + batch_size
        file["predicted_neutrino_pz"][start_idx:end]  = pred[:, :M].float().cpu().numpy()
        file["target_neutrino_truth"][start_idx:end]  = truth[:, :M].float().cpu().numpy()


class ExclusiveAssignmentTask(BaseTask):
    """
    Per-particle exclusive-assignment cross-entropy.

    Each valid particle is softmax-assigned to at most one chain (or a
    background class), enforcing cross-chain exclusivity that the independent
    per-mask Dice/BCE losses never impose. Headless: reuses the mask logits
    (``pred_key``) as class scores over the Q chains plus one background class.
    GT masks per type are chain-exclusive by truthtag construction, so a hard
    argmax label is well defined.

    ``background="zero"`` (default) appends a constant-0 background column, so a
    particle is assigned to a chain only when that chain's logit beats 0. A
    learned per-particle background head (``"learned"``) is deferred to a later
    ablation: expose a memory-derived scalar via ``MEMORY_OUTPUT_NAMES`` and pass
    it through ``NON_QUERY_OUTPUTS`` (as ``gate_relevance`` is), then use it as
    the background column here.

    Post-matching (queries already permuted to target order). Two instances are
    registered: chains vs jet_mask_true, and chains vs jet_mask_true_W.
    All Q slots stay in the softmax (fixed shapes); null slots are pushed down —
    partially redundant with ``null_mask_penalty``, kept for exclusivity.
    """

    def __init__(
        self,
        config: TaskConfig,
        pred_key: str = 'mask_predictions',
        target_key: str = 'jet_mask_true',
        loss_key: str = 'exclusive',
        background: str = 'zero',
        validity_key: Optional[str] = None,
    ):
        super().__init__(config)
        self.pred_key = pred_key
        self.target_key = target_key
        self.loss_key = loss_key
        self.background = background
        self.validity_key = validity_key

    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # No influence on Hungarian matching.
        B, Q, _ = predictions[self.pred_key].shape
        T = next(iter(targets.values())).shape[1]
        return torch.zeros(B, Q, T, device=predictions[self.pred_key].device)

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = predictions[self.pred_key]        # [B, Q, N]
        tgt = targets[self.target_key]             # [B, Q, N]
        if tgt.ndim == 2:
            tgt = tgt.unsqueeze(1)

        B, Q, N = logits.shape

        obj_valid = targets.get(self.validity_key) if self.validity_key else targets.get('obj_valid_mask')
        if obj_valid is not None:
            tgt = tgt * obj_valid.float().unsqueeze(-1)   # drop null-slot targets
            logits = logits.masked_fill(~obj_valid.bool().unsqueeze(-1), -1e4)

        tgt_bool = tgt > 0.5
        has_sig = tgt_bool.any(dim=1)                             # [B, N]
        chain_idx = tgt_bool.float().argmax(dim=1)               # [B, N] in [0, Q)
        bg_label = torch.full((B, N), Q, dtype=torch.long, device=logits.device)
        label = torch.where(has_sig, chain_idx, bg_label)        # [B, N], bg class = Q

        # Background column (constant 0) → class Q. cross_entropy wants [B, C, N].
        bg_col = logits.new_zeros(B, 1, N)
        logits_full = torch.cat([logits, bg_col], dim=1)          # [B, Q+1, N]

        ce = F.cross_entropy(logits_full, label, reduction='none')  # [B, N]

        if valid_mask is not None:
            vm = valid_mask.float()
            denom = vm.sum().clamp(min=1)
            loss = (ce * vm).sum() / denom
        else:
            loss = ce.mean()

        return self.config.get_loss_weight(self.loss_key) * loss


class MaskHierarchyConsistencyTask(BaseTask):
    """
    Soft W-in-top consistency: within a chain, the W-mask should be a subset of
    the top-mask, so sigmoid(w_logit) <= sigmoid(top_logit) per particle. Any
    excess is penalised quadratically:

        viol = relu(sigmoid(w) - sigmoid(top) + margin) ** 2

    Headless (no params, zero matching cost). Restricted to valid chains
    (obj_valid) × valid particles. The stronger b-union structural formulation
    (W ⊂ top by construction) is deferred (user decision).
    """

    def __init__(self, config: TaskConfig, margin: float = 0.0,
                 top_key: str = 'mask_predictions', w_key: str = 'mask_W'):
        super().__init__(config)
        self.margin = margin
        self.top_key = top_key
        self.w_key = w_key

    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        B, Q, _ = predictions[self.top_key].shape
        T = next(iter(targets.values())).shape[1]
        return torch.zeros(B, Q, T, device=predictions[self.top_key].device)

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.top_key not in predictions or self.w_key not in predictions:
            # Layer where one head is inactive — nothing to constrain.
            ref = next(iter(predictions.values()))
            return ref.new_tensor(0.0)

        top_logit = predictions[self.top_key]   # [B, Q, N]
        w_logit = predictions[self.w_key]        # [B, Q, N]
        B, Q, N = top_logit.shape

        viol = F.relu(w_logit.sigmoid() - top_logit.sigmoid() + self.margin) ** 2  # [B, Q, N]

        # Selection mask over valid chains × valid particles.
        sel = top_logit.new_ones(B, Q, 1, dtype=torch.bool)
        obj_valid = targets.get('obj_valid_mask')
        if obj_valid is not None:
            sel = sel & obj_valid.bool().unsqueeze(-1)          # [B, Q, 1]
        sel = sel.expand(B, Q, N)
        if valid_mask is not None:
            sel = sel & valid_mask.bool().unsqueeze(1)          # [B, Q, N]

        denom = sel.float().sum()
        if denom < 1:
            return top_logit.new_tensor(0.0)
        loss = (viol * sel.float()).sum() / denom.clamp(min=1)
        return self.config.get_loss_weight('consistency') * loss


class InvariantMassTask(BaseTask):
    """
    Soft invariant-mass loss (headless, no params, zero matching cost).

    Reconstructs each chain's top and W 4-vectors as a soft (sigmoid-weighted)
    sum of the raw particle 4-vectors ``jet_p4_raw`` [B, N, 4] = (E, px, py, pz),
    computes the invariant mass, and applies a width-normalised Huber loss toward
    m_top / m_W. Hadronic chains only (leptonic chains lack the neutrino, so their
    reconstructed mass is meaningless). Runs in an fp32 island for numerical safety.
    """

    def __init__(self, config: TaskConfig, m_w: float = 80.4, m_top: float = 172.5,
                 width_w: float = 15.0, width_top: float = 25.0, huber_delta: float = 1.0):
        super().__init__(config)
        self.m_w = m_w
        self.m_top = m_top
        self.width_w = width_w
        self.width_top = width_top
        self.huber_delta = huber_delta

    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        ref = predictions.get('mask_predictions', next(iter(predictions.values())))
        B, Q, _ = ref.shape
        T = next(iter(targets.values())).shape[1]
        return torch.zeros(B, Q, T, device=ref.device)

    def _mass_term(self, mask_logits, p4, jet_valid, sel, m_target, width):
        # mask_logits [B,Q,N]; p4 [B,N,4]; jet_valid [B,N]; sel [B,Q] bool
        probs = mask_logits.sigmoid().float() * jet_valid[:, None, :]     # [B,Q,N]
        P4 = torch.einsum('bqn,bnk->bqk', probs, p4)                      # [B,Q,4]
        E, px, py, pz = P4[..., 0], P4[..., 1], P4[..., 2], P4[..., 3]
        m2 = (E * E - px * px - py * py - pz * pz).clamp(min=0.0)
        m = torch.sqrt(m2 + 1e-6)
        z = (m - m_target) / width                                       # [B,Q]
        z_sel = z[sel]
        if z_sel.numel() == 0:
            return mask_logits.new_tensor(0.0), 0
        loss = F.huber_loss(z_sel, torch.zeros_like(z_sel),
                            delta=self.huber_delta, reduction='sum')
        return loss, z_sel.numel()

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if 'mask_predictions' not in predictions or 'mask_W' not in predictions:
            ref = next(iter(predictions.values()))
            return ref.new_tensor(0.0)
        p4 = targets.get('jet_p4_raw')
        if p4 is None:
            return predictions['mask_predictions'].new_tensor(0.0)

        top_logits = predictions['mask_predictions']
        w_logits = predictions['mask_W']
        B, Q, N = top_logits.shape

        jet_valid = valid_mask if valid_mask is not None else targets.get('jet_valid_mask')
        jet_valid = top_logits.new_ones(B, N) if jet_valid is None else jet_valid.float()

        obj_valid = targets.get('obj_valid_mask')
        if obj_valid is not None:
            sel = obj_valid.bool()
        else:
            sel = torch.ones(B, Q, dtype=torch.bool, device=top_logits.device)
        ct = targets.get('chain_type')
        if ct is not None:
            sel = sel & (ct == 0)   # hadronic chains only

        with torch.autocast(device_type=top_logits.device.type, enabled=False):
            p4f = p4.float()
            top_loss, n_top = self._mass_term(top_logits, p4f, jet_valid, sel, self.m_top, self.width_top)
            w_loss, n_w = self._mass_term(w_logits, p4f, jet_valid, sel, self.m_w, self.width_w)

        n = n_top + n_w
        if n == 0:
            return top_logits.new_tensor(0.0)
        total = (top_loss + w_loss) / n
        return self.config.get_loss_weight('invariant_mass') * total


class BackgroundSuppressionTask(BaseTask):
    """
    Penalises high mask logits for background particles (not in any real GT mask)
    across all real query slots. This provides Q× stronger gradient than the
    per-matched-slot BCE in MaskReconstructionTask.
    """

    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # No influence on Hungarian matching
        B, Q, _ = predictions['mask_predictions'].shape
        T = next(iter(targets.values())).shape[1]
        return torch.zeros(B, Q, T, device=predictions['mask_predictions'].device)

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pred_masks    = predictions['mask_predictions']   # [B, Q, N]
        jet_mask_true = targets['jet_mask_true']          # [B, Q, N]
        obj_valid     = targets.get('obj_valid_mask')     # [B, Q] bool or None

        if jet_mask_true.ndim == 2:
            jet_mask_true = jet_mask_true.unsqueeze(1)

        B, Q, N = pred_masks.shape

        # Signal particle: appears in ANY real object's GT mask
        if obj_valid is not None:
            real_masks = jet_mask_true * obj_valid.float().unsqueeze(-1)  # [B, Q, N]
        else:
            real_masks = jet_mask_true
        signal = real_masks.any(dim=1)   # [B, N]  True = signal
        bg     = ~signal                 # [B, N]  True = background

        # Exclude padding positions
        if valid_mask is not None:
            bg = bg & valid_mask.bool()

        if not bg.any():
            return pred_masks.new_tensor(0.0)

        # Only fire on real query slots (null slots covered by null_mask_penalty)
        if obj_valid is not None:
            real_query = obj_valid                        # [B, Q]
        else:
            real_query = torch.ones(B, Q, dtype=torch.bool, device=pred_masks.device)

        # suppress_mask: [B, Q, N] — positions where we apply the loss
        suppress_mask = real_query.unsqueeze(-1) & bg.unsqueeze(1)  # [B, Q, N]

        bce = F.binary_cross_entropy_with_logits(
            pred_masks, torch.zeros_like(pred_masks), reduction='none'
        )                                                # [B, Q, N]
        bce = bce * suppress_mask.float()

        n = suppress_mask.float().sum().clamp(min=1)
        loss = bce.sum() / n

        return self.config.get_loss_weight('bg_suppress') * loss


class ParticleGatingTask(BaseTask):
    """
    Binary BCE loss on gate relevance scores: signal particles → 1, background → 0.
    Only fires at the final decoder layer where gate_relevance is injected.
    """

    def compute_cost(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        B, Q, _ = predictions['mask_predictions'].shape
        T = next(iter(targets.values())).shape[1]
        return torch.zeros(B, Q, T, device=predictions['mask_predictions'].device)

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if 'gate_relevance' not in predictions:
            return predictions['mask_predictions'].new_tensor(0.0)

        relevance = predictions['gate_relevance']   # [B, N]
        jet_mask_true = targets['jet_mask_true']    # [B, Q, N]
        obj_valid = targets.get('obj_valid_mask')   # [B, Q]

        if jet_mask_true.ndim == 2:
            jet_mask_true = jet_mask_true.unsqueeze(1)

        if obj_valid is not None:
            real = jet_mask_true * obj_valid.float().unsqueeze(-1)
        else:
            real = jet_mask_true
        particle_target = real.any(dim=1).float()  # [B, N]  1=signal, 0=background

        if valid_mask is not None:
            vm = valid_mask.float()
            n = vm.sum().clamp(min=1)
            loss = (F.binary_cross_entropy(relevance.clamp(1e-6, 1 - 1e-6),
                                           particle_target, reduction='none') * vm).sum() / n
        else:
            loss = F.binary_cross_entropy(relevance.clamp(1e-6, 1 - 1e-6),
                                          particle_target, reduction='mean')

        return self.config.get_loss_weight('gate') * loss
