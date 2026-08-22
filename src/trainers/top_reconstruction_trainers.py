import os
import math
import lightning
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self._val_exact_counts = {}

        # Mask-only pretraining config
        pretrain_cfg = config.get("pretraining", {})
        self.mask_pretrain_epochs = pretrain_cfg.get("mask_pretrain_epochs", 0)
        self.mask_pretrain_tasks = pretrain_cfg.get("tasks", ["mask"])
        self._pretrain_phase_active = self.mask_pretrain_epochs > 0
        self.transition_ramp_epochs = pretrain_cfg.get("transition_ramp_epochs", 0)

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
        raw_outputs = self(inputs)
        outputs = self.model.match_for_loss(raw_outputs, targets)
        total_loss, task_losses = self._compute_loss(outputs, targets)
        margin = getattr(getattr(self.model, "matcher", None), "last_cost_margin", None)
        if margin is not None:
            self.log('train_matching_cost_margin', margin.mean(), on_step=False,
                     on_epoch=True, prog_bar=False, sync_dist=self._sync_dist)

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
        raw_outputs = self(inputs)
        outputs = self.model.match_for_loss(raw_outputs, targets)
        total_loss, task_losses = self._compute_loss(outputs, targets)
        matched_targets = outputs[max(outputs)].get('__targets__')
        self._accumulate_validation_exact(raw_outputs, matched_targets)

        self.log('val_loss', total_loss, on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=self._sync_dist)
        for task_name, task_loss in task_losses.items():
            self.log(f'val_loss_{task_name}', task_loss, on_step=False,
                     on_epoch=True, prog_bar=False, sync_dist=self._sync_dist)

        return total_loss

    def on_validation_epoch_start(self):
        self._val_exact_counts = {}
        self._val_collapse_counts = {}

    def _accumulate_validation_exact(self, outputs, matched_targets=None):
        """Accumulate exact raw-query counts under both S2 permutations."""
        final = outputs[max(outputs)]
        matched = matched_targets if matched_targets is not None else final.get('__targets__')
        if matched is None or 'mask_predictions' not in final or 'mask_W' not in final:
            return

        jet_valid = matched.get('jet_valid_mask')
        if jet_valid is not None:
            self._accumulate_collapse_metrics(
                final['mask_predictions'], final['mask_W'], jet_valid)

        top_valid = matched.get('top_valid', matched.get('obj_valid_mask'))
        w_valid = matched.get('w_valid', matched.get('obj_valid_mask'))
        if jet_valid is None or top_valid is None or w_valid is None:
            return

        jet_valid = jet_valid.bool()
        top_valid = top_valid.bool()
        w_valid = w_valid.bool()

        def aligned_target(target, prediction):
            """Align legacy target tensors to the prediction [B, Q, P] shape."""
            target = target.float()
            target = target[:, :prediction.shape[1], :prediction.shape[2]]
            if target.shape[1] < prediction.shape[1]:
                target = F.pad(target, (0, 0, 0, prediction.shape[1] - target.shape[1]))
            if target.shape[2] < prediction.shape[2]:
                target = F.pad(target, (0, prediction.shape[2] - target.shape[2]))
            return target

        def exact_mask(prediction, target):
            target = aligned_target(target, prediction)
            mismatch = ((prediction > 0) != (target > 0.5)) & jet_valid[:, None, :prediction.shape[2]]
            return ~mismatch.any(dim=-1)

        if final['mask_predictions'].shape[1] != 2:
            raise ValueError("validation S2 scoring requires exactly two raw query slots")
        top_target = aligned_target(matched['jet_mask_true'], final['mask_predictions'])
        w_target = aligned_target(matched['jet_mask_true_W'], final['mask_W'])
        permutations = torch.tensor([[0, 1], [1, 0]], device=jet_valid.device)
        top_candidates, w_candidates = [], []
        for permutation in permutations:
            top_candidates.append(exact_mask(final['mask_predictions'][:, permutation], top_target))
            w_candidates.append(exact_mask(final['mask_W'][:, permutation], w_target))
        top_candidates = torch.stack(top_candidates, dim=1)
        w_candidates = torch.stack(w_candidates, dim=1)
        identifiable = top_valid | w_valid
        chain_candidates = (
            (~top_valid[:, None] | top_candidates)
            & (~w_valid[:, None] | w_candidates)
            & identifiable[:, None]
        )
        fully_matchable = (top_valid & w_valid).all(dim=1)
        event_candidates = fully_matchable[:, None] & chain_candidates.all(dim=2)
        partial_candidates = (~identifiable[:, None] | chain_candidates).all(dim=2)
        component_errors = (
            ((~top_candidates) & top_valid[:, None]).sum(dim=2)
            + ((~w_candidates) & w_valid[:, None]).sum(dim=2)
        )
        rank = (
            event_candidates.long() * 1_000_000
            + partial_candidates.long() * 100_000
            + chain_candidates.sum(dim=2) * 1_000
            - component_errors
        )
        best = rank.argmax(dim=1)
        rows = torch.arange(rank.shape[0], device=rank.device)
        top_exact = top_candidates[rows, best]
        w_exact = w_candidates[rows, best]

        top_num = (top_exact & top_valid).sum().float()
        top_den = top_valid.sum().float()
        w_num = (w_exact & w_valid).sum().float()
        w_den = w_valid.sum().float()
        chain_valid = top_valid & w_valid
        # Full-event efficiency needs an event-level conjunction; chain
        # efficiency counts complete chain slots only.
        chain_exact = top_exact & w_exact
        event_valid = fully_matchable
        event_exact = (top_exact & w_exact & chain_valid).all(dim=1)

        values = {
            'top_eff': (top_num, top_den),
            'W_eff': (w_num, w_den),
            'chain_eff': ((chain_exact & chain_valid).sum().float(), chain_valid.sum().float()),
            'ttbar_eff': (event_exact[event_valid].sum().float(), event_valid.sum().float()),
        }
        for name, (num, den) in values.items():
            if name not in self._val_exact_counts:
                self._val_exact_counts[name] = [num.detach(), den.detach()]
            else:
                self._val_exact_counts[name][0] += num.detach()
                self._val_exact_counts[name][1] += den.detach()

    def _accumulate_collapse_metrics(self, top_pred, w_pred, jet_valid):
        """Accumulate query-collapse diagnostics on raw (unmatched) predictions.

        Healthy models keep the two chain queries distinct (<1% identical masks,
        padding logits near -10); collapsed runs converge to one shared solution
        (78-88% identical masks) with mask logits bloating into padding (+0.7).
        """
        jet_valid = jet_valid.bool()
        batch_values = {}
        for name, pred in (('top', top_pred), ('w', w_pred)):
            if pred.dim() != 3 or pred.shape[1] != 2 or pred.shape[2] > jet_valid.shape[1]:
                continue
            valid = jet_valid[:, None, :pred.shape[2]]
            identical = (((pred[:, 0] > 0) == (pred[:, 1] > 0)) | ~valid).all(dim=-1)
            batch_values[f'query_identical_{name}'] = (
                identical.sum().float(),
                torch.full((), identical.numel(), device=identical.device,
                           dtype=torch.float32))
            padding = ~jet_valid[:, :pred.shape[2]]
            pad_count = padding.sum().float() * pred.shape[1]
            if pad_count > 0:
                batch_values[f'pad_logit_{name}'] = (
                    (pred * padding[:, None, :]).sum().float(), pad_count)

        for key, (num, den) in batch_values.items():
            if key not in self._val_collapse_counts:
                self._val_collapse_counts[key] = [num.detach(), den.detach()]
            else:
                self._val_collapse_counts[key][0] += num.detach()
                self._val_collapse_counts[key][1] += den.detach()

    def _log_collapse_metrics(self):
        for key, (num, den) in self._val_collapse_counts.items():
            pair = torch.stack([num, den])
            if self._sync_dist:
                pair = self.all_gather(pair).reshape(-1, 2).sum(dim=0)
            if pair[1] <= 0:
                continue
            value = pair[0] / pair[1]
            self.log(f'val_{key}', value, on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=False)
        self._val_collapse_counts = {}

    def test_step(self, batch, batch_idx):
        """Task-agnostic test step"""
        inputs, targets = batch
        raw_outputs = self(inputs, last_output_only=True)
        outputs = self.model.match_for_loss(raw_outputs, targets)
        total_loss, task_losses = self._compute_loss(outputs, targets)

        self.log('test_loss', total_loss, on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=self._sync_dist)
        for task_name, task_loss in task_losses.items():
            self.log(f'test_loss_{task_name}', task_loss, on_step=False,
                     on_epoch=True, prog_bar=False, sync_dist=self._sync_dist)

        final = outputs[max(outputs)]
        save_targets = final.get("__targets__", targets)
        event_ids = inputs.get("event_id")
        if event_ids is None:
            raise ValueError("test artifacts require stable event_id values from the dataset")
        self._save_test_predictions(raw_outputs, save_targets, event_ids)

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
            if "__targets__" in layer_predictions:
                layer_targets = layer_predictions["__targets__"]
                valid_mask = layer_targets.get("jet_valid_mask")
                preds = {k: v for k, v in layer_predictions.items()
                         if k != "__targets__"}
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
            # Replace loss with zero so backward + allreduce still runs on all ranks
            total_loss = torch.zeros_like(total_loss)
            self.trainer.should_stop = True

        return total_loss, task_loss_accum
    
    def configure_optimizers(self):
        """Optimizer configuration with config-driven scheduler selection"""
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            name_lower = name.lower()
            is_query_token = any(
                token_name in name_lower
                for token_name in ("target_token", "query_token")
            )
            if (
                param.ndim <= 1
                or name_lower.endswith(".bias")
                or "norm" in name_lower
                or is_query_token
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
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
            # Fractions above one make a "rewarmup" decay from above the base LR.
            rewarmup_frac = min(1.0, max(0.0, rewarmup_frac))
            transition_epoch = self.mask_pretrain_epochs  # 0 if no pretraining

            if interval == "step":
                estimated_steps = max(1, int(self.trainer.estimated_stepping_batches))
                max_epochs = max(1, int(self.trainer.max_epochs))
                steps_per_epoch = max(1, math.ceil(estimated_steps / max_epochs))
                warmup_units = warmup_epochs * steps_per_epoch
                total_units = T_max * steps_per_epoch
                transition_unit = transition_epoch * steps_per_epoch
                rewarmup_units = rewarmup_epochs * steps_per_epoch
            else:
                warmup_units = warmup_epochs
                total_units = T_max
                transition_unit = transition_epoch
                rewarmup_units = rewarmup_epochs

            total_units = max(1, total_units)
            warmup_units = min(max(0, warmup_units), total_units)
            transition_unit = max(0, transition_unit)
            rewarmup_units = min(
                max(0, rewarmup_units),
                max(0, total_units - transition_unit),
            )

            min_factor = min(1.0, max(0.0, eta_min / base_lr))

            def cosine_factor(progress):
                progress = min(1.0, max(0.0, progress))
                return min_factor + (1.0 - min_factor) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )

            def lr_lambda(t):
                # Two-phase schedule when re-warmup is configured
                has_rewarmup = (
                    rewarmup_units > 0
                    and 0 < transition_unit < total_units
                )
                if has_rewarmup and t >= transition_unit:
                    t_post = t - transition_unit
                    if t_post < rewarmup_units:
                        # Re-warmup: ramp from rewarmup_frac → 1.0.
                        return rewarmup_frac + (1.0 - rewarmup_frac) * (
                            (t_post + 1) / rewarmup_units
                        )
                    # Post re-warmup cosine decay
                    decay_start = rewarmup_units
                    decay_total = max(1, total_units - transition_unit - decay_start)
                    return cosine_factor((t_post - decay_start) / decay_total)

                # Pre-transition (or no re-warmup): original warmup + cosine
                if t < warmup_units:
                    # LambdaLR evaluates t=0 before the first optimizer step;
                    # use the first warmup fraction there instead of near-zero LR.
                    return (t + 1) / max(1, warmup_units)
                if has_rewarmup:
                    # Cosine scoped to pre-transition period
                    pre_decay_total = max(1, transition_unit - warmup_units)
                    return cosine_factor((t - warmup_units) / pre_decay_total)
                else:
                    return cosine_factor(
                        (t - warmup_units) / max(1, total_units - warmup_units)
                    )

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
        """Track validation metrics, exact-match rates, and task metrics."""
        for name, (num, den) in self._val_exact_counts.items():
            pair = torch.stack([num, den])
            if self._sync_dist:
                pair = self.all_gather(pair).reshape(-1, 2).sum(dim=0)
            rate = pair[0] / pair[1].clamp(min=1.0)
            # This hook runs once per validation epoch, so Lightning can expose
            # the value to ModelCheckpoint/EarlyStopping without re-averaging
            # batch-local rates.
            self.log(f'val_{name}', rate, on_step=False, on_epoch=True,
                     prog_bar=name == 'ttbar_eff', sync_dist=False)
        self._val_exact_counts = {}
        self._log_collapse_metrics()

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
        """Initialize HDF5 files for single-device test predictions."""
        super().on_test_start()
        self._test_h5_files: Dict[str, h5py.File] = {}
        self._test_output_enabled = getattr(self.trainer, "world_size", 1) == 1

        if not self._test_output_enabled:
            if self.global_rank == 0:
                print(
                    "[Test outputs] HDF5 prediction writing disabled for distributed "
                    "test: batches do not carry original event indices."
                )
            self.test_start_idx = 0
            return

        if self.global_rank == 0:
            out_dir = Path(self.trainer.logger.log_dir)
            test_loaders = self.trainer.test_dataloaders
            test_loader = test_loaders[0] if isinstance(test_loaders, (list, tuple)) else test_loaders
            dataset = test_loader.dataset
            number_events = len(dataset)
            self._test_expected_events = number_events
            self._test_output_paths = []
            particle_count = None
            for attr in ("jet", "_jet"):
                jet_array = getattr(dataset, attr, None)
                if jet_array is not None:
                    particle_count = int(jet_array.shape[1])
                    break
            if particle_count is None:
                if number_events == 0:
                    raise ValueError("Cannot create test outputs for an empty dataset")
                particle_count = int(dataset[0][0]["jet"].shape[0])

            for task_name, task in self.task_registry.tasks.items():
                h5_filename = f"test_outputs_{task_name}.h5"
                fh = h5py.File(out_dir / h5_filename, "w")
                self._test_output_paths.append(out_dir / h5_filename)
                if isinstance(task, MaskReconstructionTask):
                    task.create_test_datasets(fh, number_events, particle_count)
                else:
                    task.create_test_datasets(fh, number_events)
                fh.create_dataset("event_id", shape=(number_events,), dtype="u8")
                self._test_h5_files[task_name] = fh

        self.test_start_idx = 0

    def on_test_end(self):
        """Close HDF5 file handles opened during testing"""
        for fh in self._test_h5_files.values():
            fh.close()
        self._test_h5_files.clear()
        expected = getattr(self, "_test_expected_events", self.test_start_idx)
        if self._test_output_enabled and self.global_rank == 0 and self.test_start_idx != expected:
            for path in getattr(self, "_test_output_paths", []):
                path.unlink(missing_ok=True)
            raise RuntimeError(
                "refusing malformed partial test artifacts: wrote "
                f"{self.test_start_idx} of {expected} events; run the complete test split"
            )
    
    def _save_test_predictions(
        self,
        outputs: Dict[int, Dict[str, torch.Tensor]],
        targets,
        event_ids,
    ):
        """Save test predictions to HDF5 (rank 0 only)"""
        if not getattr(self, "_test_output_enabled", False) or self.global_rank != 0:
            return

        final_layer = max(outputs.keys())
        layer_dict = outputs[final_layer]

        # Predictions are always raw learned-query order. Targets retain their
        # canonical truth-chain order; S2 alignment belongs in evaluation.
        save_targets = targets
        predictions = {k: v for k, v in layer_dict.items() if k != "__targets__"}

        batch_size = predictions[list(predictions.keys())[0]].shape[0]

        for task_name, task in self.task_registry.tasks.items():
            task.save_test_predictions(
                file=self._test_h5_files[task_name],
                predictions=predictions,
                targets=save_targets,
                start_idx=self.test_start_idx,
                batch_size=batch_size
            )
            self._test_h5_files[task_name]["event_id"][
                self.test_start_idx:self.test_start_idx + batch_size
            ] = event_ids.detach().to(dtype=torch.uint64, device="cpu").numpy()

        self.test_start_idx += batch_size
    
    def on_fit_start(self):
        """Log model parameter counts at the start of training"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.global_rank == 0:
            print(f"\nModel parameters: {total:,} total, {trainable:,} trainable\n")
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

        # Masked cross-attention warm-up (Stage C): epoch-gated, rank-identical, DDP-safe.
        if getattr(self.model, 'masked_cross_attention', False):
            active = self.trainer.current_epoch >= self.model.masked_attention_start_epoch
            if active != self.model.masked_attention_active and self.global_rank == 0:
                print(f"[MaskedAttn] epoch {self.trainer.current_epoch}: "
                      f"masked cross-attention {'ON' if active else 'OFF'}")
            self.model.masked_attention_active = active

        # Mask-only pretraining phase transitions
        if self._pretrain_phase_active:
            epoch = self.trainer.current_epoch
            mask_tasks = [
                task for task in self.task_registry.tasks.values()
                if hasattr(task, 'null_penalty_scale')
            ]
            pretrain_set = set(self.mask_pretrain_tasks)
            ramp = self.transition_ramp_epochs

            if epoch < self.mask_pretrain_epochs:
                # Phase 1: mask only — no objectness/type cost or loss, no null penalty
                self.task_registry.set_active_tasks(self.mask_pretrain_tasks)
                for mask_task in mask_tasks:
                    mask_task.null_penalty_scale = 0.0
                if epoch == 0 and self.global_rank == 0:
                    print(f"\n[Pretraining] Phase 1: mask-only "
                          f"(epochs 0–{self.mask_pretrain_epochs - 1}), "
                          f"active tasks: {self.mask_pretrain_tasks}")
            else:
                # Phase 2+: all tasks enabled, with optional ramp
                self.task_registry.set_active_tasks(None)
                epochs_since = epoch - self.mask_pretrain_epochs
                if ramp > 0 and epochs_since < ramp:
                    alpha = min(1.0, (epochs_since + 1) / ramp)
                else:
                    alpha = 1.0

                # Ramp loss scales for non-pretrain tasks
                for task_name in self.task_registry.tasks:
                    if task_name not in pretrain_set:
                        self.task_registry.set_loss_scale(task_name, alpha)

                # Ramp null penalty scale on every registered mask task.  W-only
                # pretraining must not accidentally receive the null target.
                for mask_task in mask_tasks:
                    mask_task.null_penalty_scale = alpha

                if epoch == self.mask_pretrain_epochs and self.global_rank == 0:
                    print(f"\n[Pretraining] Phase 2: full multi-task "
                          f"(epoch {epoch}+), all tasks active"
                          f"{f', ramp over {ramp} epochs' if ramp > 0 else ''}")
                if ramp > 0 and epochs_since < ramp and self.global_rank == 0:
                    print(f"[Ramp] epoch {epoch}: alpha={alpha:.3f} "
                          f"for new tasks + null penalty")

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm before optimizer step (after clipping)"""
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if grads:
            total_norm = torch.linalg.vector_norm(torch.stack([
                g.detach().float().norm(2) for g in grads
            ]))
        else:
            total_norm = next(self.parameters()).new_zeros(())
        self.log('grad_norm', total_norm, on_step=True, on_epoch=False,
                 prog_bar=False, sync_dist=False)

    def on_train_end(self):
        """Plot loss curves and learning rate schedule (rank 0 only)"""
        if self.global_rank != 0:
            return
        if self.trainer.logger is None:
            return
        import matplotlib.pyplot as plt

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

    # Early stopping — skip checks during pretraining phase
    es_cfg = cb_cfg.get("early_stopping", {})
    pretrain_warmup = config.get("pretraining", {}).get("mask_pretrain_epochs", 0)
    if es_cfg.get("enabled", True):
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
        auto_insert_metric_name=ckpt_cfg.get("auto_insert_metric_name", False),
        save_last=ckpt_cfg.get("save_last", True),
    ))

    # Keep loss-selected checkpoints for backward comparisons while also
    # retaining the checkpoint selected by the primary exact-efficiency metric.
    # This is opt-in because older tasks/configurations may not expose it.
    eff_ckpt_cfg = cb_cfg.get("efficiency_checkpoint", {})
    if eff_ckpt_cfg.get("enabled", False):
        callbacks.append(ModelCheckpoint(
            save_top_k=eff_ckpt_cfg.get("save_top_k", 1),
            monitor=eff_ckpt_cfg.get("monitor", "val_ttbar_eff"),
            mode=eff_ckpt_cfg.get("mode", "max"),
            save_weights_only=eff_ckpt_cfg.get("save_weights_only", False),
            filename=eff_ckpt_cfg.get(
                "filename", "epoch{epoch:03d}-val_ttbar_eff{val_ttbar_eff:.4f}"
            ),
            auto_insert_metric_name=eff_ckpt_cfg.get("auto_insert_metric_name", False),
            save_last=eff_ckpt_cfg.get("save_last", False),
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
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=grad_clip,
        default_root_dir=log_dir,
        # ponytail: pilot knobs only; default 1.0 keeps full-run behaviour identical.
        limit_train_batches=train_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=train_cfg.get("limit_val_batches", 1.0),
    )

    lightning_model = ReconstructionTrainer(model, task_registry, config)
    lightning_trainer.fit(lightning_model, datamodule=data_module, ckpt_path=ckpt_path)
    return lightning_trainer, lightning_model
