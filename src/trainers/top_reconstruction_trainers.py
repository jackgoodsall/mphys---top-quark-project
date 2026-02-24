import lightning
from lightning import Trainer
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from dataclasses import field
from torchmetrics.functional import roc, precision_recall_curve, auroc
import h5py
from .loss_functions import *
import torch_optimizer
from utils.utils import generate_reconstruction_report
import joblib
from src.models.components.masked_former_tasks import * 


def _to_1d(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)

def _safe_all_gather(self, t: torch.Tensor) -> torch.Tensor:
    # Works in single or multi-GPU; returns concatenated tensor on every rank
    gathered = self.all_gather(t)
    return gathered.reshape(-1, *t.shape[1:]).cpu()

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
        self.lr = training_config.get("learning_rate", 1e-4)
        self.weight_decay = training_config.get("weight_decay", 5e-4)
        self.use_lookahead = training_config.get("use_lookahead", False)
        
        # History tracking
        self.train_loss_history = []
        self.val_loss_history = []
        self.test_metrics = {}
        
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

        out_dir = Path(self.trainer.logger.log_dir)
        self._save_test_predictions(outputs, targets, out_dir)

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
            (total_loss, task_loss_accum): total scalar and per-task accumulated losses
        """
        total_loss = 0.0
        task_loss_accum: Dict[str, float] = {}

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
                layer_id=layer_id
            )
            total_loss += layer_loss

            for task_name, task_val in per_task.items():
                task_loss_accum[task_name] = task_loss_accum.get(task_name, 0.0) + task_val

        # Stop training on NaN/Inf
        if torch.is_tensor(total_loss) and not torch.isfinite(total_loss):
            bad_tasks = [k for k, v in task_loss_accum.items() if not (v == v)]  # NaN check
            print(f"\nNon-finite total_loss={total_loss.item():.4f} — stopping training.")
            if bad_tasks:
                print(f"   Tasks with NaN: {bad_tasks}")
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
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.get("step_size", 2000),
                gamma=sched_cfg.get("gamma", 0.7),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

        elif sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_cfg.get("T_max", 50),
                eta_min=sched_cfg.get("eta_min", 1e-6),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

        elif sched_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": sched_cfg.get("plateau_monitor", "val_loss"),
                },
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
        """Initialize HDF5 files for test predictions"""
        super().on_test_start()
        out_dir = Path(self.trainer.logger.log_dir)
        
        test_loaders = self.trainer.test_dataloaders
        number_events = len(test_loaders.dataset)
        
        for task_name, task in self.task_registry.tasks.items():
            h5_filename = f"test_outputs_{task_name}.h5"
            with h5py.File(out_dir / h5_filename, "w") as file:
                task.create_test_datasets(file, number_events)
        
        self.test_start_idx = 0
    
    def _save_test_predictions(
        self,
        outputs: Dict[int, Dict[str, torch.Tensor]],
        targets,
        out_dir: Path
    ):
        """Save test predictions to HDF5"""
        final_layer = max(outputs.keys())
        layer_dict = outputs[final_layer]

        # Use matched/padded targets if available, else original targets
        if "__targets__" in layer_dict:
            save_targets = layer_dict["__targets__"]
            predictions = {k: v for k, v in layer_dict.items()
                          if k != "__targets__"}
        else:
            save_targets = targets
            predictions = layer_dict

        batch_size = predictions[list(predictions.keys())[0]].shape[0]

        for task_name, task in self.task_registry.tasks.items():
            h5_filename = f"test_outputs_{task_name}.h5"
            with h5py.File(out_dir / h5_filename, "r+") as file:
                task.save_test_predictions(
                    file=file,
                    predictions=predictions,
                    targets=save_targets,
                    start_idx=self.test_start_idx,
                    batch_size=batch_size
                )

        self.test_start_idx += batch_size
    
    def on_fit_start(self):
        """Log model parameter counts at the start of training"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel parameters: {total:,} total, {trainable:,} trainable\n")
        if self.logger:
            self.logger.experiment.add_scalar('model/total_params', float(total), global_step=0)
            self.logger.experiment.add_scalar('model/trainable_params', float(trainable), global_step=0)

    def on_train_epoch_start(self):
        """Reset peak GPU memory counter and log current learning rate"""
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

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm before optimizer step (after clipping)"""
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if grads:
            total_norm = torch.norm(
                torch.stack([g.detach().norm(2) for g in grads])
            ).item()
        else:
            total_norm = 0.0
        self.log('grad_norm', total_norm, on_step=True, on_epoch=False,
                 prog_bar=False, sync_dist=False)

    def on_train_end(self):
        """Plot loss curves"""
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

    # Early stopping
    es_cfg = cb_cfg.get("early_stopping", {})
    callbacks.append(EarlyStopping(
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

    lightning_trainer = lightning.Trainer(
        num_nodes=1,
        precision=precision,
        min_epochs=config["model_training"]["min_epochs"],
        max_epochs=config["model_training"]["max_epochs"],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=grad_clip,
        default_root_dir=log_dir,
    )

    lightning_model = ReconstructionTrainer(model, task_registry, config)
    lightning_trainer.fit(lightning_model, datamodule=data_module, ckpt_path=ckpt_path)
    return lightning_trainer, lightning_model






