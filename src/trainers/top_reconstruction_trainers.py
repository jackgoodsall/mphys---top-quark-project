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
    ### Lightning Module for training a binary classifier
    def __init__(self, model, config, *args, **kwargs):
        super(ReconstructionTrainer, self).__init__()

        self.model = model
        self.config = config
        training_config = config["model_training"]
        self.lr =  training_config.get("learning_rate", 1e-4)
        self.weight_decay = training_config.get("weight_decay", 5e-4)
        self.use_lookahead = training_config.get("use_lookahead", False)
        self.train_loss_history = []
        self.val_loss_history = []

        self.test_metrics = {}

        self.reconstruct_Ws = kwargs.get("reconstruct_Ws", False)
        self.use_hungarian = config.get("use_hungarian_matching", False)

        
        self.save_hyperparameters(ignore = ["model"])
    
    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
       
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, targets)

        _check_for_nans("outputs", outputs)
        _check_for_nans("targets", targets)

        _check_for_nans("loss", loss)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, targets)


        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_loss_step', loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
       
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, targets)

        out_dir = Path(self.trainer.logger.log_dir)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
       
        self.test_metrics["test_loss"] = loss

        targets = targets["jet_mask_true"]

        with h5py.File(out_dir / "test_outputs.h5", "r+" ) as file:
            file["targets"][self.start_idx: self.start_idx + len(targets)] = targets.view(-1, 1, 20).cpu().numpy()
            file["predicted"][self.start_idx: self.start_idx + len(targets)] = outputs.cpu().numpy()

            self.start_idx += len(targets)


        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                    lr= self.lr,
                                    weight_decay = self.weight_decay)
        
        if self.use_lookahead:
            optimizer = torch_optimizer.Lookahead(
                optimizer,

            )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2000,  # every 1000 steps
        gamma=0.5,
        )

        return {
            "optimizer":optimizer,
            "lr_scheduler":{
                "scheduler"  : scheduler,
                #"interval" : "step",
                #"monitor": "val_loss_step"

            }
            }
    
    def loss_function(self, outputs, targets):


        masked_loss = total_masked_loss(outputs, targets)
        return masked_loss["loss_total"]
    
    def on_train_epoch_end(self):

        cm = self.trainer.callback_metrics
        def grab(keys):
            for k in keys:
                if k in cm:
                    v = cm[k]
                    return float(v.item() if hasattr(v, "item") else v)
            return None

        tr = grab(["train_loss", "train_loss_epoch"])
 
        if tr is not None:
            self.train_loss_history.append(tr)

    def on_validation_epoch_end(self):
        cm = self.trainer.callback_metrics
        def grab(keys):
            for k in keys:
                if k in cm:
                    v = cm[k]
                    return float(v.item() if hasattr(v, "item") else v)
            return None

        vl = grab(["val_loss", "val_loss_epoch"])
        if vl is not None:
            self.val_loss_history.append(vl)

    def on_test_start(self):
        super().on_test_start()
        out_dir = Path(self.trainer.logger.log_dir)

        test_loaders = self.trainer.test_dataloaders
        number_events = len(test_loaders.dataset)
        

        with h5py.File(out_dir / "test_outputs.h5", "w" ) as file:
            file.create_dataset("targets",
                                   shape = (number_events, 1, 20) )

            file.create_dataset("predicted",
                                  shape = (number_events, 1, 20))

        self.start_idx = 0
        self.W_start_idx = 0

    def on_test_end(self):
        out_dir = Path(self.trainer.logger.log_dir)


        
        

    def on_train_end(self):
        out_dir = Path(self.trainer.logger.log_dir)

        fig_path = out_dir / "loss_curves.png"
        plt.figure()
        plt.plot(self.train_loss_history, label="train")
        plt.plot(self.val_loss_history, label="val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

## Function to train a binary classifier
def train_reconstruction_model(
        model,
        data_module,
        config,
        min_epochs,
        max_epochs,
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
        dirpath="checkpoints",
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
    model = ReconstructionTrainer(model, config, **kwargs)
    print("fitting")
    lightning_trainer.fit(model, datamodule=data_module)
    return lightning_trainer, model






