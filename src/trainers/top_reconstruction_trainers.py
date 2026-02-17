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
    
    def forward(self, batch, last_output_only=False):
        """Forward pass through model"""
        return self.model(batch, last_output_only=last_output_only)
    
    def training_step(self, batch, batch_idx):
        """Task-agnostic training step"""
        inputs, targets = batch
        inputs['targets'] = targets

        outputs = self(inputs)
        total_loss = self._compute_loss(outputs, targets)
        
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, 
                prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Task-agnostic validation step"""
        inputs, targets = batch
        inputs['targets'] = targets
        
        outputs = self(inputs)
        total_loss = self._compute_loss(outputs, targets)
        
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, 
                prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Task-agnostic test step"""
        inputs, targets = batch
        inputs['targets'] = targets
        
        outputs = self(inputs, last_output_only=True)
        total_loss = self._compute_loss(outputs, targets)
        
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, 
                prog_bar=True, sync_dist=True)
        
        out_dir = Path(self.trainer.logger.log_dir)
        self._save_test_predictions(outputs, targets, out_dir)
        
        return total_loss
    
    def _compute_loss(
        self, 
        outputs: Dict[int, Dict[str, torch.Tensor]], 
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss across all layers.
        Each task applies its own layer weights.
        """
        total_loss = 0.0
        valid_mask = targets.get('valid_mask')
        
        for layer_id, layer_predictions in outputs.items():
            # TaskRegistry applies per-task layer weights internally
            layer_loss = self.task_registry.compute_total_loss(
                predictions=layer_predictions,
                targets=targets,
                valid_mask=valid_mask,
                layer_id=layer_id  # Pass layer_id for task-specific weighting
            )
            total_loss += layer_loss
        
        return total_loss
    
    def configure_optimizers(self):
        """Optimizer configuration"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        if self.use_lookahead:
            import torch_optimizer
            optimizer = torch_optimizer.Lookahead(optimizer)
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=2000,
            gamma=0.7,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            }
        }
    
    def on_train_epoch_end(self):
        """Track training loss history"""
        cm = self.trainer.callback_metrics
        train_loss = self._grab_metric(cm, ["train_loss", "train_loss_epoch"])
        if train_loss is not None:
            self.train_loss_history.append(train_loss)
    
    def on_validation_epoch_end(self):
        """Track validation loss history"""
        cm = self.trainer.callback_metrics
        val_loss = self._grab_metric(cm, ["val_loss", "val_loss_epoch"])
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
    
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
        targets: Dict[str, torch.Tensor],
        out_dir: Path
    ):
        """Save test predictions to HDF5"""
        final_layer = max(outputs.keys())
        predictions = outputs[final_layer]
        
        batch_size = predictions[list(predictions.keys())[0]].shape[0]
        
        for task_name, task in self.task_registry.tasks.items():
            h5_filename = f"test_outputs_{task_name}.h5"
            with h5py.File(out_dir / h5_filename, "r+") as file:
                task.save_test_predictions(
                    file=file,
                    predictions=predictions,
                    targets=targets,
                    start_idx=self.test_start_idx,
                    batch_size=batch_size
                )
        
        self.test_start_idx += batch_size
    
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


## Function to train a binary classifier
def train_reconstruction_model(
        model,
        task_registry,
        data_module,
        config,
        use_lr_finder = False,
        use_early_stopping = True,
        early_stopping_params = None,
        logger = None,
        *args,
        **kwargs
    ):
    callbacks = []
    ## Uses default dict to ensure construction
    ## To do actually finish EarlyStopping addition
    if use_early_stopping:
        if early_stopping_params:
            callbacks.append(EarlyStopping(**early_stopping_params))
        else:
            callbacks.append(EarlyStopping(monitor = "val_loss"))
    ## Adds a model checkpointer so that it only saves weights of the model.
    ## To do make the dirpath inside the lightning log
    checkpoint_cb = ModelCheckpoint(
        filename="{epoch}-{val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True, 
    )

    callbacks.append(checkpoint_cb)
    
    # Create Trainer
    print("loading")
    lightning_trainer = lightning.Trainer(
        num_nodes = 1,
        min_epochs=config["model_training"]["min_epochs"],
        max_epochs=config["model_training"]["max_epochs"],
        logger = logger,
        callbacks= callbacks,
        gradient_clip_val=1,
        default_root_dir="./masked_reconstruction"
    )
    print("training")
    # Fit trainer
    ## Passes in the whole config object as allows for easier saving
    model = ReconstructionTrainer(model,task_registry, config, **kwargs)
    print("fitting")
    lightning_trainer.fit(model, datamodule=data_module)
    return lightning_trainer, model






