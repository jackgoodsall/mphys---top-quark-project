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
from torchmetrics.functional import roc, precision_recall_curve, auroc
import h5py


def _to_1d(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)

def _safe_all_gather(self, t: torch.Tensor) -> torch.Tensor:
    # Works in single or multi-GPU; returns concatenated tensor on every rank
    gathered = self.all_gather(t)
    return gathered.reshape(-1, *t.shape[1:]).cpu()


class BinaryClassifierTrainer(lightning.LightningModule):
    ### Lightning Module for training a binary classifier
    def __init__(self, model, config, *args, **kwargs):
        super(BinaryClassifierTrainer, self).__init__()

        self.model = model
        self.lr =  config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 1e-4)

        self.class_weights = config.get("class_weights", None)
        assert len(self.class_weights) == 2, "Class weights length must be number of classes"
        
        self.class_weights = torch.Tensor(self.class_weights)

        self.accuracy_metric = BinaryAccuracy()
        self.train_loss_history = []
        self.train_acc_history = []

        self.val_loss_history = []
        self.val_acc_history = []
        
        self.test_metrics = {}

        self._test_logits = []
        self._test_targets = []

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
        acc = self.accuracy_metric  (outputs, targets)

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

        # store for epoch-end plots
        self._test_logits.append(outputs.detach())
        self._test_targets.append(targets.detach())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                    lr= self.lr,
                                    weight_decay = self.weight_decay)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
        )
        return {
            "optimizer":optimizer,
            "lr_scheduler":{
                "scheduler"  : schedular,
                "interval" : "epoch",
                "monitor": "val_loss"

            }
            }
    
    
    def loss_function(self, outputs, targets):
        return nn.BCEWithLogitsLoss(pos_weight = self.class_weights[0]/self.class_weights[1])(outputs, targets)
    
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

    def on_test_epoch_end(self):
        # Concatenate local tensors
        logits = torch.cat(self._test_logits, dim=0) if self._test_logits else torch.empty(0)
        targets = torch.cat(self._test_targets, dim=0) if self._test_targets else torch.empty(0)

        # DDP: gather from all ranks
        if self.trainer.world_size > 1:
            logits = _safe_all_gather(self, logits)
            targets = _safe_all_gather(self, targets)

        logits = _to_1d(logits)
        targets = _to_1d(targets).float()

        if logits.numel() == 0:
            return  # nothing to plot

        probs = torch.sigmoid(logits)  # scores in [0,1]

        out_dir = Path(self.trainer.logger.log_dir) if self.trainer.logger else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Per-class score histograms (positives vs negatives)
        fig = plt.figure()
        plt.hist(probs[targets == 1].cpu().numpy(), bins=40, alpha=0.6, label="pos (y=1)")
        plt.hist(probs[targets == 0].cpu().numpy(), bins=40, alpha=0.6, label="neg (y=0)")
        plt.xlabel("Predicted probability")
        plt.ylabel("Count")
        plt.title("Test score distribution by class")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "test_score_hist_by_class.png", dpi=150)
        plt.close(fig)

        # 2) ROC curve + AUC
        # torchmetrics.functional.roc returns fpr, tpr, thresholds for binary when pred is probs
        fpr, tpr, _ = roc(probs, targets.int(), task = "binary")
        roc_auc = auroc(probs, targets.int(), task = "binary")
        fig = plt.figure()
        plt.plot(fpr.cpu().numpy(), tpr.cpu().numpy(), label=f"ROC AUC = {roc_auc.item():.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC (test)")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "test_roc.png", dpi=150)
        plt.close(fig)

        # 3) Precision-Recall curve + AUC
        prec, rec, _ = precision_recall_curve(probs, targets.int(), task = "binary")
        fig = plt.figure()
        plt.plot(rec.cpu().numpy(), prec.cpu().numpy())
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (test)")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "test_pr.png", dpi=150)
        plt.close(fig)

        # 4) Threshold sweep (F1 vs threshold) to see where you’re paying the price
        thresholds = torch.linspace(0.0, 1.0, steps=201)
        best_f1, best_t = -1.0, 0.5
        f1_values = []
        for t in thresholds:
            preds = (probs >= t).int()
            tp = ((preds == 1) & (targets == 1)).sum().item()
            fp = ((preds == 1) & (targets == 0)).sum().item()
            fn = ((preds == 0) & (targets == 1)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1_values.append(f1)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t.item())

        fig = plt.figure()
        plt.plot(thresholds.cpu().numpy(), f1_values)
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.title(f"F1 vs threshold (best={best_f1:.3f} @ t={best_t:.2f})")
        plt.tight_layout()
        fig.savefig(out_dir / "test_f1_vs_threshold.png", dpi=150)
        plt.close(fig)

        # 5) Confusion matrix at the best threshold (optional, quick text dump)
        preds_best = (probs >= best_t).int()
        tn = ((preds_best == 0) & (targets == 0)).sum().item()
        tp = ((preds_best == 1) & (targets == 1)).sum().item()
        fp = ((preds_best == 1) & (targets == 0)).sum().item()
        fn = ((preds_best == 0) & (targets == 1)).sum().item()
        self.print(f"[test] Confusion @ t={best_t:.2f} — TP:{tp} FP:{fp} TN:{tn} FN:{fn}")
        
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
    if use_early_stopping:
        if early_stopping_params:
            callbacks.append(EarlyStopping(**early_stopping_params))
        else:
            callbacks.append(EarlyStopping(monitor = "val_loss"))
    # Create Trainer
    lightning_trainer = lightning.Trainer(
        num_nodes = 1,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger = logger
    )
    # Fit trainer
    model = BinaryClassifierTrainer(model, config)
    lightning_trainer.fit(model, datamodule=data_module)
    return lightning_trainer, model


class ReconstructionTrainer(lightning.LightningModule):
    ### Lightning Module for training a binary classifier
    def __init__(self, model, config, *args, **kwargs):
        super(ReconstructionTrainer, self).__init__()

        self.model = model
        self.lr =  config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 5e-4)

        self.train_loss_history = []

        self.val_loss_history = []

        
        self.test_metrics = {}


        self.save_hyperparameters(ignore = ["model"])
    
    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
       
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, targets)
 

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, targets)


        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
       
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, targets)

        out_dir = Path(self.trainer.logger.log_dir)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
       
        self.test_metrics["test_loss"] = loss
        
        with h5py.File(out_dir / "test_outputs.h5", "r+" ) as file:
            file["targets"][self.start_idx: self.start_idx + len(targets)] = targets.cpu().numpy()
            file["predicted"][self.start_idx: self.start_idx + len(targets)] = outputs.cpu().numpy()

            self.start_idx += len(targets)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                    lr= self.lr,
                                    weight_decay = self.weight_decay)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
        )
        return {
            "optimizer":optimizer,
            "lr_scheduler":{
                "scheduler"  : schedular,
                "interval" : "epoch",
                "monitor": "val_loss"

            }
            }
    
    
    def loss_function(self, outputs, targets):
        return nn.MSELoss()(outputs, targets)
    
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
        num_particles, num_features = tops.shape[1:]
        with h5py.File(out_dir / "test_outputs.h5", "w" ) as file:
            file.create_dataset("targets",
                                   shape = (number_events, num_particles, num_features) )

            file.create_dataset("predicted",
                                  shape = (number_events, num_particles, num_features))

        self.start_idx = 0

    def on_test_end(self):
        out_dir = Path(self.trainer.logger.log_dir)

        with h5py.File(out_dir / "test_outputs.h5", "r") as file:
            predicted_outputs = file["predicted"][()]
            target_outputs = file["targets"][()]

        
        for top in (0, 1):
            for variable in (0,1,2,3):
                plt.figure()
                plt.scatter(predicted_outputs[: , top, variable]
                            ,target_outputs[: , top, variable] )
                plt.savefig(f"particle_{top}_variable_{variable}.png")



        


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
    if use_early_stopping:
        if early_stopping_params:
            callbacks.append(EarlyStopping(**early_stopping_params))
        else:
            callbacks.append(EarlyStopping(monitor = "val_loss"))
    # Create Trainer
    print("loading")
    lightning_trainer = lightning.Trainer(
        num_nodes = 1,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger = logger
    )
    print("training")
    # Fit trainer
    model = ReconstructionTrainer(model, config["model_training"])
    print("fitting")
    lightning_trainer.fit(model, datamodule=data_module)
    return lightning_trainer, model
