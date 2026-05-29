import os
import math
import lightning
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


class WarmupEarlyStopping(EarlyStopping):
    """EarlyStopping that ignores the monitored metric for the first `warmup_epochs` epochs."""

    def __init__(self, warmup_epochs: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_epochs:
            return
        super().on_validation_epoch_end(trainer, pl_module)
import h5py
import torch_optimizer
from src.models.components.masked_former_tasks import *
from constants import TARGETS_KEY


def _check_for_nans( name, x):
    """Recursively check tensors or dicts for non-finite values."""
    if isinstance(x, dict):
        for k, v in x.items():
            _check_for_nans(f"{name}.{k}", v)
        return

    if not torch.is_tensor(x):
        return

    if not torch.isfinite(x).all():
        bad = x[~torch.isfinite(x)]
        print(f"\n🚨 NON-FINITE DETECTED in {name}!")
        print(f"   count = {bad.numel()}")
        print(f"   min   = {bad.min().item() if bad.numel()>0 else 'n/a'}")
        print(f"   max   = {bad.max().item() if bad.numel()>0 else 'n/a'}")
        print(f"   sample values = {bad[:10]}")


def _as_float(value, key_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Config '{key_name}' must be numeric, got: {value!r}") from exc
    raise TypeError(f"Config '{key_name}' must be numeric, got type: {type(value).__name__}")


def _as_int(value, key_name: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError as exc:
            raise ValueError(f"Config '{key_name}' must be integer-like, got: {value!r}") from exc
    raise TypeError(f"Config '{key_name}' must be integer-like, got type: {type(value).__name__}")

class ReconstructionTrainer(lightning.LightningModule):
    """
    Base trainer that is completely task-agnostic.
    Layer weights are defined per-task in TaskConfig.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        task_registry: TaskRegistry,
        config: dict,
        *args, 
        **kwargs
    ):
        super().__init__()
        
        self.model = model
        self.task_registry = task_registry
        self.config = config
        
        # Training config
        training_config = config["model_training"]
        self.lr = _as_float(training_config.get("learning_rate", 1e-4), "model_training.learning_rate")
        self.weight_decay = _as_float(training_config.get("weight_decay", 5e-4), "model_training.weight_decay")
        self.use_lookahead = training_config.get("use_lookahead", False)
        
        # History tracking
        self.train_loss_history = []
        self.val_loss_history = []
        self.lr_history = []
        self.test_metrics = {}

        # Multi-phase pretraining config
        pretrain_cfg = config.get("pretraining", {})
        self.transition_ramp_epochs = pretrain_cfg.get("transition_ramp_epochs", 0)

        if "phases" in pretrain_cfg:
            # New multi-phase format: list of {epochs, tasks, cost_only_tasks?}
            self._pretrain_phases = pretrain_cfg["phases"]
            cumulative = 0
            self._phase_boundaries = []
            for p in self._pretrain_phases:
                cumulative += p["epochs"]
                self._phase_boundaries.append(cumulative)
            self.mask_pretrain_epochs = self._phase_boundaries[-1]
        else:
            # Legacy single-phase
            self._pretrain_phases = None
            self._phase_boundaries = None
            self.mask_pretrain_epochs = pretrain_cfg.get("mask_pretrain_epochs", 0)
            self.mask_pretrain_tasks = pretrain_cfg.get("tasks", ["mask"])

        self._pretrain_phase_active = self.mask_pretrain_epochs > 0

        self.save_hyperparameters(ignore=["model", "task_registry"])

    @property
    def _sync_dist(self) -> bool:
        """Only use sync_dist when running on multiple devices."""
        return getattr(self.trainer, 'num_devices', 1) > 1

    def forward(self, batch, last_output_only=False):
        """Forward pass through model"""
        return self.model(batch, last_output_only=last_output_only)
    
    def training_step(self, batch, batch_idx):
        """Task-agnostic training step"""
        inputs, targets = batch
        inputs['targets'] = targets

        outputs = self(inputs)
        total_loss, task_losses = self._compute_loss(outputs, targets)

        self.log('train_loss', total_loss, on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=self._sync_dist)
        for task_name, task_loss in task_losses.items():
            self.log(f'train_loss_{task_name}', task_loss, on_step=False,
                     on_epoch=True, prog_bar=False, sync_dist=self._sync_dist)

        # GPU memory (MB) — local device query, no cross-GPU sync needed
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1e6
            mem_reserved = torch.cuda.memory_reserved() / 1e6
            self.log('gpu_mem_alloc_mb', mem_alloc, on_step=True, on_epoch=False,
                     prog_bar=False, sync_dist=False)
            self.log('gpu_mem_reserved_mb', mem_reserved, on_step=True, on_epoch=False,
                     prog_bar=False, sync_dist=False)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Task-agnostic validation step"""
        inputs, targets = batch
        inputs['targets'] = targets

        outputs = self(inputs)
        total_loss, task_losses = self._compute_loss(outputs, targets)

        self.log('val_loss', total_loss, on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=self._sync_dist)
        for task_name, task_loss in task_losses.items():
            self.log(f'val_loss_{task_name}', task_loss, on_step=False,
                     on_epoch=True, prog_bar=False, sync_dist=self._sync_dist)

        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Task-agnostic test step"""
        inputs, targets = batch
        inputs['targets'] = targets

        outputs = self(inputs, last_output_only=True)
        total_loss, task_losses = self._compute_loss(outputs, targets)

        self.log('test_loss', total_loss, on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=self._sync_dist)
        for task_name, task_loss in task_losses.items():
            self.log(f'test_loss_{task_name}', task_loss, on_step=False,
                     on_epoch=True, prog_bar=False, sync_dist=self._sync_dist)

        self._save_test_predictions(outputs, targets)

        return total_loss
    
    def _compute_loss(
        self,
        outputs: Dict[int, Dict[str, torch.Tensor]],
        targets
    ):
        """
        Compute loss across all layers.
        Each task applies its own layer weights.

        If __targets__ is injected into layer dicts (new matching path),
        extract and use those. Otherwise fall back to the original targets.

        Returns:
            (total_loss, task_loss_accum): total scalar and per-task accumulated losses (tensors)
        """
        total_loss = 0.0
        task_loss_accum: Dict[str, torch.Tensor] = {}

        final_layer_id = max(outputs.keys())

        for layer_id, layer_predictions in outputs.items():
            if TARGETS_KEY in layer_predictions:
                layer_targets = layer_predictions[TARGETS_KEY]
                valid_mask = layer_targets.get("jet_valid_mask")
                preds = {k: v for k, v in layer_predictions.items()
                         if k != TARGETS_KEY}
            else:
                layer_targets = targets
                preds = layer_predictions
                valid_mask = targets.get('valid_mask') if isinstance(targets, dict) else None

            layer_loss, per_task = self.task_registry.compute_total_loss(
                predictions=preds,
                targets=layer_targets,
                valid_mask=valid_mask,
                layer_id=layer_id,
                is_final_layer=(layer_id == final_layer_id),
            )
            total_loss += layer_loss

            for task_name, task_val in per_task.items():
                prev = task_loss_accum.get(task_name, 0.0)
                task_loss_accum[task_name] = prev + task_val

        # Stop training on NaN/Inf — replace with zero to keep DDP ranks in
        # sync (setting should_stop on one rank desynchronises collectives).
        if torch.is_tensor(total_loss) and not torch.isfinite(total_loss):
            bad_tasks = [
                k for k, v in task_loss_accum.items()
                if not (torch.isfinite(v).all() if torch.is_tensor(v) else (v == v))
            ]
            print(f"\n[Rank {self.global_rank}] Non-finite total_loss "
                  f"— replacing with 0 and signalling stop.")
            if bad_tasks:
                print(f"   Tasks with NaN: {bad_tasks}")
            # Multiply by 0 (not zeros_like) to preserve grad_fn so backward() succeeds
            total_loss = total_loss * 0.0
            self.trainer.should_stop = True

        return total_loss, task_loss_accum
    
    def configure_optimizers(self):
        """Optimizer configuration with config-driven scheduler selection"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.use_lookahead:
            import torch_optimizer
            optimizer = torch_optimizer.Lookahead(optimizer)
            # torch_optimizer.Lookahead doesn't initialise hooks dicts
            # that PyTorch >=2.6 expects when calling state_dict()
            if not hasattr(optimizer, '_optimizer_state_dict_pre_hooks'):
                optimizer._optimizer_state_dict_pre_hooks = {}
            if not hasattr(optimizer, '_optimizer_state_dict_post_hooks'):
                optimizer._optimizer_state_dict_post_hooks = {}
            if not hasattr(optimizer, '_optimizer_load_state_dict_pre_hooks'):
                optimizer._optimizer_load_state_dict_pre_hooks = {}
            if not hasattr(optimizer, '_optimizer_load_state_dict_post_hooks'):
                optimizer._optimizer_load_state_dict_post_hooks = {}

        sched_cfg = self.config.get("model_training", {}).get("scheduler", {})
        sched_type = sched_cfg.get("type", "step")

        if sched_type == "step":
            interval = sched_cfg.get("update_interval", "epoch")
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=_as_int(sched_cfg.get("step_size", 2000), "model_training.scheduler.step_size"),
                gamma=_as_float(sched_cfg.get("gamma", 0.7), "model_training.scheduler.gamma"),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": interval}}

        elif sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=_as_int(sched_cfg.get("T_max", 50), "model_training.scheduler.T_max"),
                eta_min=_as_float(sched_cfg.get("eta_min", 1e-6), "model_training.scheduler.eta_min"),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

        elif sched_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=_as_float(sched_cfg.get("factor", 0.5), "model_training.scheduler.factor"),
                patience=_as_int(sched_cfg.get("patience", 5), "model_training.scheduler.patience"),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": sched_cfg.get("plateau_monitor", "val_loss"),
                },
            }

        elif sched_type == "warmup_cosine":
            warmup_epochs = _as_int(sched_cfg.get("warmup_epochs", 5), "model_training.scheduler.warmup_epochs")
            T_max = _as_int(sched_cfg.get("T_max", 50), "model_training.scheduler.T_max")
            eta_min = _as_float(sched_cfg.get("eta_min", 1e-6), "model_training.scheduler.eta_min")
            interval = sched_cfg.get("update_interval", "step")
            base_lr = self.lr

            # LR re-warmup at phase transition
            rewarmup_epochs = _as_int(sched_cfg.get("lr_rewarmup_epochs", 0), "model_training.scheduler.lr_rewarmup_epochs")
            rewarmup_frac = _as_float(sched_cfg.get("lr_rewarmup_fraction", 0.3), "model_training.scheduler.lr_rewarmup_fraction")
            transition_epoch = self.mask_pretrain_epochs  # 0 if no pretraining

            if interval == "step":
                steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
                warmup_units = warmup_epochs * steps_per_epoch
                total_units = T_max * steps_per_epoch
                transition_unit = transition_epoch * steps_per_epoch
                rewarmup_units = rewarmup_epochs * steps_per_epoch
            else:
                warmup_units = warmup_epochs
                total_units = T_max
                transition_unit = transition_epoch
                rewarmup_units = rewarmup_epochs

            min_factor = eta_min / base_lr

            def lr_lambda(t):
                # Two-phase schedule when re-warmup is configured
                if rewarmup_units > 0 and transition_unit > 0 and t >= transition_unit:
                    t_post = t - transition_unit
                    post_total = total_units - transition_unit
                    if t_post < rewarmup_units:
                        # Re-warmup: ramp from rewarmup_frac → 1.0
                        return rewarmup_frac + (1.0 - rewarmup_frac) * (t_post / rewarmup_units)
                    # Post re-warmup cosine decay
                    decay_start = rewarmup_units
                    decay_total = max(1, post_total - rewarmup_units)
                    progress = (t_post - decay_start) / decay_total
                    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                    return min_factor + (1.0 - min_factor) * cosine

                # Pre-transition (or no re-warmup): original warmup + cosine
                if t < warmup_units:
                    return max(1e-8, t / max(1, warmup_units))
                if transition_unit > 0 and rewarmup_units > 0:
                    # Cosine scoped to pre-transition period
                    pre_decay_total = max(1, transition_unit - warmup_units)
                    progress = (t - warmup_units) / pre_decay_total
                else:
                    progress = (t - warmup_units) / max(1, total_units - warmup_units)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                return min_factor + (1.0 - min_factor) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": interval},
            }

        elif sched_type == "warmup_step":
            warmup_epochs = _as_int(sched_cfg.get("warmup_epochs", 3), "model_training.scheduler.warmup_epochs")
            step_size = _as_int(sched_cfg.get("step_size", 10), "model_training.scheduler.step_size")
            gamma = _as_float(sched_cfg.get("gamma", 0.5), "model_training.scheduler.gamma")
            eta_min = _as_float(sched_cfg.get("eta_min", 1e-6), "model_training.scheduler.eta_min")
            base_lr = self.lr
            min_factor = eta_min / base_lr

            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return max(1e-8, epoch / max(1, warmup_epochs))
                # Number of steps taken since warmup ended
                n_steps = (epoch - warmup_epochs) // step_size
                return max(min_factor, gamma ** n_steps)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        elif sched_type == "none":
            return optimizer

        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
    
    def on_train_epoch_end(self):
        """Track training loss history, log task metrics, and log peak GPU memory"""
        cm = self.trainer.callback_metrics
        train_loss = self._grab_metric(cm, ["train_loss", "train_loss_epoch"])
        if train_loss is not None:
            self.train_loss_history.append(train_loss)

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            self.log('gpu_peak_mem_mb', peak_mb, on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=False)

        # Log and reset task-level metrics
        for task_name, task in self.task_registry.tasks.items():
            if hasattr(task, 'get_detection_stats'):
                stats = task.get_detection_stats()
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.log(f'train_{task_name}_{stat_name}', float(value),
                                 prog_bar=False, sync_dist=self._sync_dist)
                task.reset_detection_stats()

            if hasattr(task, 'get_accuracy_stats'):
                stats = task.get_accuracy_stats()
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.log(f'train_{task_name}_{stat_name}', float(value),
                                 prog_bar=False, sync_dist=self._sync_dist)
                task.reset_accuracy_stats()

    def on_validation_epoch_end(self):
        """Track validation loss history and log task metrics"""
        cm = self.trainer.callback_metrics
        val_loss = self._grab_metric(cm, ["val_loss", "val_loss_epoch"])
        if val_loss is not None:
            self.val_loss_history.append(val_loss)

        # Log and reset task-level metrics
        for task_name, task in self.task_registry.tasks.items():
            if hasattr(task, 'get_detection_stats'):
                stats = task.get_detection_stats()
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.log(f'val_{task_name}_{stat_name}', float(value),
                                 prog_bar=False, sync_dist=self._sync_dist)
                task.reset_detection_stats()

            if hasattr(task, 'get_accuracy_stats'):
                stats = task.get_accuracy_stats()
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.log(f'val_{task_name}_{stat_name}', float(value),
                                 prog_bar=False, sync_dist=self._sync_dist)
                task.reset_accuracy_stats()
    
    def _grab_metric(self, cm, keys):
        """Helper to extract metric from callback metrics"""
        for k in keys:
            if k in cm:
                v = cm[k]
                return float(v.item() if hasattr(v, "item") else v)
        return None
    
    def on_test_start(self):
        """Initialize HDF5 files for test predictions (rank 0 only)"""
        super().on_test_start()

        # Always evaluate with all tasks fully active regardless of which
        # pretraining phase training ended in.  Without this, the matching
        # cost used during the forward pass can differ from Phase 1 (e.g.
        # objectness cost added when its cost_weight > 0), misaligning saved
        # HDF5 targets with mask predictions and collapsing efficiency.
        self.task_registry.set_active_tasks(None)
        self.task_registry.set_cost_only_tasks(None)
        # Restore null penalty so the mask loss is computed correctly during test
        for task in self.task_registry.tasks.values():
            if hasattr(task, 'null_penalty_scale'):
                task.null_penalty_scale = 1.0

        self._test_h5_files: Dict[str, h5py.File] = {}

        if self.global_rank == 0:
            out_dir = Path(self.trainer.logger.log_dir)
            test_loaders = self.trainer.test_dataloaders
            number_events = len(test_loaders.dataset)

            for task_name, task in self.task_registry.tasks.items():
                h5_filename = f"test_outputs_{task_name}.h5"
                fh = h5py.File(out_dir / h5_filename, "w")
                task.create_test_datasets(fh, number_events)
                self._test_h5_files[task_name] = fh

        self.test_start_idx = 0

    def on_test_end(self):
        """Close HDF5 file handles opened during testing"""
        for fh in self._test_h5_files.values():
            fh.close()
        self._test_h5_files.clear()
    
    def _save_test_predictions(
        self,
        outputs: Dict[int, Dict[str, torch.Tensor]],
        targets,
    ):
        """Save test predictions to HDF5 (rank 0 only)"""
        if self.global_rank != 0:
            return

        final_layer = max(outputs.keys())
        layer_dict = outputs[final_layer]

        # Use matched/padded targets if available, else original targets
        if TARGETS_KEY in layer_dict:
            save_targets = layer_dict[TARGETS_KEY]
            predictions = {k: v for k, v in layer_dict.items()
                          if k != TARGETS_KEY}
        else:
            save_targets = targets
            predictions = layer_dict

        batch_size = predictions[list(predictions.keys())[0]].shape[0]

        for task_name, task in self.task_registry.tasks.items():
            task.save_test_predictions(
                file=self._test_h5_files[task_name],
                predictions=predictions,
                targets=save_targets,
                start_idx=self.test_start_idx,
                batch_size=batch_size
            )

        self.test_start_idx += batch_size
    
    def on_fit_start(self):
        """Log model parameter counts and optionally compile the model."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.global_rank == 0:
            print(f"\nModel parameters: {total:,} total, {trainable:,} trainable\n")

        compile_cfg = self.config.get("model_training", {}).get("compile", False)
        if compile_cfg:
            mode = compile_cfg if isinstance(compile_cfg, str) else "default"
            if self.global_rank == 0:
                print(f"[Compile] torch.compile(mode='{mode}') — first batch will be slow (tracing).")
            self.model = torch.compile(self.model, mode=mode)
        if self.logger:
            self.logger.experiment.add_scalar('model/total_params', float(total), global_step=0)
            self.logger.experiment.add_scalar('model/trainable_params', float(trainable), global_step=0)

    def on_train_epoch_start(self):
        """Reset peak GPU memory counter, log current learning rate, and apply pretraining phase."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for opt in optimizers:
            # Lookahead wraps the optimizer — access inner param_groups
            param_groups = getattr(opt, 'optimizer', opt).param_groups
            for j, pg in enumerate(param_groups):
                suffix = f'_pg{j}' if len(param_groups) > 1 else ''
                self.log(f'lr{suffix}', pg['lr'], on_step=False,
                         on_epoch=True, prog_bar=False, sync_dist=False)
            # Track LR of first param group for plotting
            if param_groups:
                self.lr_history.append(param_groups[0]['lr'])

        # Multi-phase pretraining transitions
        if self._pretrain_phase_active:
            epoch = self.trainer.current_epoch
            ramp = self.transition_ramp_epochs

            if self._pretrain_phases is not None:
                # --- New multi-phase path ---
                current_phase_idx = None
                for i, boundary in enumerate(self._phase_boundaries):
                    if epoch < boundary:
                        current_phase_idx = i
                        break

                if current_phase_idx is not None:
                    # Still in a pretrain phase
                    phase = self._pretrain_phases[current_phase_idx]
                    phase_tasks = phase["tasks"]
                    cost_only = phase.get("cost_only_tasks", [])
                    trainable_heads = phase.get("trainable_heads", None)
                    phase_start = self._phase_boundaries[current_phase_idx - 1] if current_phase_idx > 0 else 0

                    self.task_registry.set_active_tasks(phase_tasks)
                    self.task_registry.set_cost_only_tasks(cost_only)
                    # Suppress null penalty for ALL mask tasks during pretraining
                    for task in self.task_registry.tasks.values():
                        if hasattr(task, 'null_penalty_scale'):
                            task.null_penalty_scale = 0.0

                    # Freeze backbone if this phase only trains specific heads
                    if epoch == phase_start:
                        if trainable_heads is not None:
                            for param in self.model.parameters():
                                param.requires_grad = False
                            for head_name, head_module in self.model.prediction_heads.items():
                                if head_name in trainable_heads:
                                    for param in head_module.parameters():
                                        param.requires_grad = True
                            if self.global_rank == 0:
                                frozen    = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
                                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                                print(f"\n[Pretraining] Phase {current_phase_idx + 1}: backbone frozen. "
                                      f"Trainable heads: {trainable_heads} "
                                      f"({trainable:,} params, {frozen:,} frozen)")
                        else:
                            # No head restriction — unfreeze everything
                            for param in self.model.parameters():
                                param.requires_grad = True

                        if self.global_rank == 0:
                            print(f"\n[Pretraining] Phase {current_phase_idx + 1}/{len(self._pretrain_phases)}: "
                                  f"epochs {phase_start}–{self._phase_boundaries[current_phase_idx] - 1}, "
                                  f"active tasks: {phase_tasks}"
                                  + (f", cost-only: {cost_only}" if cost_only else "")
                                  + (f", trainable heads: {trainable_heads}" if trainable_heads else ""))
                else:
                    # Final phase: all tasks active, unfreeze everything
                    self.task_registry.set_active_tasks(None)
                    self.task_registry.set_cost_only_tasks(None)
                    if epoch == self.mask_pretrain_epochs:
                        for param in self.model.parameters():
                            param.requires_grad = True
                    epochs_since = epoch - self.mask_pretrain_epochs
                    alpha = min(1.0, (epochs_since + 1) / ramp) if ramp > 0 and epochs_since < ramp else 1.0

                    # Ramp all tasks that weren't in the last pretrain phase
                    last_phase_tasks = set(self._pretrain_phases[-1]["tasks"])
                    for task_name in self.task_registry.tasks:
                        if task_name not in last_phase_tasks:
                            self.task_registry.set_loss_scale(task_name, alpha)

                    # Ramp null penalty for ALL mask tasks together
                    for task in self.task_registry.tasks.values():
                        if hasattr(task, 'null_penalty_scale'):
                            task.null_penalty_scale = alpha

                    if epoch == self.mask_pretrain_epochs and self.global_rank == 0:
                        print(f"\n[Pretraining] Final phase: full multi-task "
                              f"(epoch {epoch}+), all tasks active"
                              + (f", ramp over {ramp} epochs" if ramp > 0 else ""))
                    if ramp > 0 and epochs_since < ramp and self.global_rank == 0:
                        print(f"[Ramp] epoch {epoch}: alpha={alpha:.3f} for new tasks + null penalty")

            else:
                # --- Legacy single-phase path ---
                pretrain_set = set(self.mask_pretrain_tasks)

                if epoch < self.mask_pretrain_epochs:
                    self.task_registry.set_active_tasks(self.mask_pretrain_tasks)
                    self.task_registry.set_cost_only_tasks(None)
                    for task in self.task_registry.tasks.values():
                        if hasattr(task, 'null_penalty_scale'):
                            task.null_penalty_scale = 0.0
                    if epoch == 0 and self.global_rank == 0:
                        print(f"\n[Pretraining] Phase 1: mask-only "
                              f"(epochs 0–{self.mask_pretrain_epochs - 1}), "
                              f"active tasks: {self.mask_pretrain_tasks}")
                else:
                    self.task_registry.set_active_tasks(None)
                    self.task_registry.set_cost_only_tasks(None)
                    epochs_since = epoch - self.mask_pretrain_epochs
                    alpha = min(1.0, (epochs_since + 1) / ramp) if ramp > 0 and epochs_since < ramp else 1.0

                    for task_name in self.task_registry.tasks:
                        if task_name not in pretrain_set:
                            self.task_registry.set_loss_scale(task_name, alpha)

                    for task in self.task_registry.tasks.values():
                        if hasattr(task, 'null_penalty_scale'):
                            task.null_penalty_scale = alpha

                    if epoch == self.mask_pretrain_epochs and self.global_rank == 0:
                        print(f"\n[Pretraining] Phase 2: full multi-task "
                              f"(epoch {epoch}+), all tasks active"
                              + (f", ramp over {ramp} epochs" if ramp > 0 else ""))
                    if ramp > 0 and epochs_since < ramp and self.global_rank == 0:
                        print(f"[Ramp] epoch {epoch}: alpha={alpha:.3f} for new tasks + null penalty")

    # Gradient norm is already computed by Lightning for gradient clipping.
    # Re-computing it manually every step was redundant — removed.
    # Enable Lightning's built-in norm logging via Trainer(log_every_n_steps=...)
    # if needed for debugging.

    def on_train_end(self):
        """Plot loss curves and learning rate schedule (rank 0 only)"""
        if self.global_rank != 0:
            return
        if self.trainer.logger is None:
            return
        out_dir = Path(self.trainer.logger.log_dir)

        fig_path = out_dir / "loss_curves.png"
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_history, label="train")
        plt.plot(self.val_loss_history, label="val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

        if self.lr_history:
            lr_path = out_dir / "lr_schedule.png"
            plt.figure(figsize=(10, 4))
            plt.plot(self.lr_history)
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.yscale("log")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(lr_path, dpi=150)
            plt.close()


def train_reconstruction_model(
        model,
        task_registry,
        data_module,
        config,
        ckpt_path=None,
        logger=None,
    ):
    """
    Train the reconstruction model with config-driven callbacks and scheduler.

    Args:
        model: The particle transformer model
        task_registry: TaskRegistry with registered tasks
        data_module: Lightning DataModule
        config: Full config dict
        ckpt_path: Optional checkpoint path to resume training from
        logger: Optional Lightning logger
    """
    callbacks = []

    # --- Config-driven callbacks ---
    cb_cfg = config.get("training_callbacks", {})

    # Early stopping — skip checks during pretraining phase.
    # Support both legacy mask_pretrain_epochs and new multi-phase format.
    es_cfg = cb_cfg.get("early_stopping", {})
    pretrain_cfg = config.get("pretraining", {})
    if "phases" in pretrain_cfg:
        pretrain_warmup = sum(p["epochs"] for p in pretrain_cfg["phases"])
    else:
        pretrain_warmup = pretrain_cfg.get("mask_pretrain_epochs", 0)
    callbacks.append(WarmupEarlyStopping(
        warmup_epochs=pretrain_warmup,
        monitor=es_cfg.get("monitor", "val_loss"),
        patience=es_cfg.get("patience", 10),
        min_delta=es_cfg.get("min_delta", 0.0001),
        mode=es_cfg.get("mode", "min"),
    ))

    # Model checkpoint (no dirpath — Lightning places it in default_root_dir/version_N/checkpoints/)
    ckpt_cfg = cb_cfg.get("checkpoint", {})
    callbacks.append(ModelCheckpoint(
        save_top_k=ckpt_cfg.get("save_top_k", 3),
        monitor=ckpt_cfg.get("monitor", "val_loss"),
        mode=ckpt_cfg.get("mode", "min"),
        save_weights_only=ckpt_cfg.get("save_weights_only", False),
        filename=ckpt_cfg.get("filename", "epoch={epoch}-val_loss={val_loss:.4f}"),
        save_last=ckpt_cfg.get("save_last", True),
    ))

    # Log directory — checkpoints co-located with Lightning logs
    log_dir = config.get("model_artefacts", {}).get("log_dir", "lightning_logs")
    grad_clip = config.get("model_training", {}).get("gradient_clip_val", 1.0)
    precision = config.get("model_training", {}).get("precision", "32-true")

    if logger is None:
        slurm_id = os.environ.get("SLURM_JOB_ID")
        version = int(slurm_id) if slurm_id else None
        logger = TensorBoardLogger(log_dir, version=version)

    train_cfg = config["model_training"]
    lightning_trainer = lightning.Trainer(
        num_nodes=train_cfg.get("num_nodes", 1),
        devices=train_cfg.get("devices", "auto"),
        strategy=train_cfg.get("strategy", "auto"),
        accelerator=train_cfg.get("accelerator", "auto"),
        precision=precision,
        min_epochs=config["model_training"]["min_epochs"],
        max_epochs=config["model_training"]["max_epochs"],
        check_val_every_n_epoch=train_cfg.get("check_val_every_n_epoch", 1),
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=grad_clip,
        default_root_dir=log_dir,
    )

    lightning_model = ReconstructionTrainer(model, task_registry, config)
    lightning_trainer.fit(lightning_model, datamodule=data_module, ckpt_path=ckpt_path)
    return lightning_trainer, lightning_model






