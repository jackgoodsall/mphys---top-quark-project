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
        self.reverse_transformers = joblib.load( Path(self.config["data_modules"]["input_path"], "target_transforms.joblib"))
        self.val_loss_history = []


        self.test_metrics = {}

        self.reconstruct_Ws = kwargs.get("reconstruct_Ws", False)


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
        if self.reconstruct_Ws:
            W_targets = targets["W"]
            W_outputs = outputs["W"]
            targets = targets["top"]
            outputs = outputs["top"]
        out_dir = Path(self.trainer.logger.log_dir)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
       
        self.test_metrics["test_loss"] = loss
        
        with h5py.File(out_dir / "test_outputs.h5", "r+" ) as file:
            file["targets"][self.start_idx: self.start_idx + len(targets)] = targets.cpu().numpy()
            file["predicted"][self.start_idx: self.start_idx + len(targets)] = outputs.cpu().numpy()

            self.start_idx += len(targets)
        
        if self.reconstruct_Ws:
            with h5py.File(out_dir / "W_boson_output.h5", "r+" ) as file:
                file["targets"][self.W_start_idx: self.W_start_idx + len(targets)] = W_targets.cpu().numpy()
                file["predicted"][self.W_start_idx: self.W_start_idx + len(targets)] = W_outputs.cpu().numpy()

            self.W_start_idx += len(W_targets)

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

        if self.reconstruct_Ws: 
            return W_boson_loss_function(outputs, targets) + invariant_mass_loss(outputs["top"], targets["inv_mass"], self.reverse_transformers[0], self.reverse_transformers[1], self.reverse_transformers[3], mass_mean = 6.352097, mass_std = 0.30358553)
        
        return set_invariant_loss(outputs, targets)
    
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
        _, tops = next(iter(test_loaders))
        if self.reconstruct_Ws:
            num_particles, num_features = tops["top"].shape[1:]
        else:
            num_particles, num_features = tops.shape[1:]
        with h5py.File(out_dir / "test_outputs.h5", "w" ) as file:
            file.create_dataset("targets",
                                   shape = (number_events, num_particles, num_features) )

            file.create_dataset("predicted",
                                  shape = (number_events, num_particles, num_features))
        if self.reconstruct_Ws:
            with h5py.File(out_dir / "W_boson_output.h5", "w" ) as file:
                file.create_dataset("targets",
                                    shape = (number_events, num_particles, num_features) )

                file.create_dataset("predicted",
                                    shape = (number_events, num_particles, num_features))
        self.start_idx = 0
        self.W_start_idx = 0

    def on_test_end(self):
        out_dir = Path(self.trainer.logger.log_dir)
        ## To do: add in the reverse tranformer logic into here so that can just plot straight away.

        generate_reconstruction_report(
            out_dir / "test_outputs.h5",
            out_dir / "top_plots",
            Path(self.config["data_modules"]["input_path"], "target_transforms.joblib")
        )
        if self.reconstruct_Ws:
            generate_reconstruction_report(
                out_dir / "W_boson_output.h5",
                out_dir / "W_boson_plots",
                Path(self.config["data_modules"]["input_path"], "W_target_transforms.joblib")
            , pid = 24)
        
        

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
    )
    print("training")
    # Fit trainer
    ## Passes in the whole config object as allows for easier saving
    model = ReconstructionTrainer(model, config, **kwargs)
    print("fitting")
    lightning_trainer.fit(model, datamodule=data_module)
    return lightning_trainer, model






