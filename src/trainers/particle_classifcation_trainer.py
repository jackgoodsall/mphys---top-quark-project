import lightning
from lightning import Trainer
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from lightning.pytorch.callbacks import EarlyStopping
from dataclasses import field
from lightning.pytorch.strategies import FSDPStrategy

class BinaryClassifierTrainer(lightning.LightningModule):
    ### Lightning Module for training a binary classifier
    def __init__(self, model, config, *args, **kwargs):
        super(BinaryClassifierTrainer, self).__init__()

        self.model = model
        self.lr =  config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 1e-4)

        self.accuracy_metric = BinaryAccuracy()
        self.train_loss_history = []
        self.train_acc_history = []

        self.val_loss_history = []
        self.val_acc_history = []
        
        self.test_metrics = {}

        self.save_hyperparameters(ignore = ["model"])
    
    def forward(self, batch):
        return self.model(batch)


    def training_step(self, batch, batch_idx):
       
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, targets)
        acc = self.accuracy_metric(outputs, targets)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, targets)
        acc = self.accuracy_metric(outputs, targets)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, targets)
        acc = self.accuracy_metric(outputs, targets)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.test_metrics["test_loss"] = loss
        self.test_metrics["test_acc"] = acc

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                    lr= self.lr,
                                    weight_decay = self.weight_decay)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
        )
        return {
            "optimizer":optimizer,
            "lr_schedular":{
                "schedular"  : schedular,
                "interval" : "epoch"

            }
            }
    
    
    def loss_function(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets)
    
    def on_train_epoch_end(self):

        cm = self.trainer.callback_metrics
        def grab(keys):
            for k in keys:
                if k in cm:
                    v = cm[k]
                    return float(v.item() if hasattr(v, "item") else v)
            return None

        tr = grab(["train_loss", "train_loss_epoch"])
        acc = grab(["train_acc"])
        if tr is not None:
            self.train_loss_history.append(tr)
            self.train_acc_history.append(acc)

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
            self.val_acc_history.append(grab(["val_acc"]))

    def on_test_end(self):
        return
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

        fig_path = out_dir / "acc_curves.png"
        plt.figure()
        plt.plot(self.train_acc_history, label="train")
        plt.plot(self.val_acc_history, label="val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Training and Validation Accuracy")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

        metrics = self.trainer.callback_metrics
        val_acc = float(metrics.get("val_acc", torch.tensor(float("nan"))))
        train_acc = float(metrics.get("train_acc", torch.tensor(float("nan"))))
        print(f"Final validation accuracy: {val_acc:.3f}, train loss: {train_acc:.3f}")


## Function to train a binary classifier
def train_binary_classifier_model(
        model,
        data_module,
        config,
        min_epoches = 1,
        max_epoches= 2,
        use_lr_finder = False,
        use_early_stopping = True,
        early_stopping_params = None,
        logger = None,
        *args,
        **kwargs
    ):
    callbacks = []
    ## Uses default dict to ensure construction
    if use_early_stopping:
        if early_stopping_params:
            callbacks.append(EarlyStopping(**early_stopping_params))
        else:
            callbacks.append(EarlyStopping(monitor = "val_loss"))
    # Create Trainer
    lightning_trainer = lightning.Trainer(
        num_nodes = 1,
        min_epochs=min_epoches,
        max_epochs=max_epoches,
        logger = logger
    )
    # Fit trainer
    model = BinaryClassifierTrainer(model, config)
    lightning_trainer.fit(model, datamodule=data_module)
    return lightning_trainer
