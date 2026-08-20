#!/usr/bin/env python
"""Standalone G2-from-scratch training entrypoint.

It reuses the repository encoder and data module but owns its Lightning module,
loss, output directory, and configuration.  The live G1 training path is not
imported as a trainer and no G1 checkpoint is required.
"""

import argparse
import os
import sys
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from data.datamodule import MaskedFormerTopsWsDataModule  # noqa: E402
from main import create_default_task_registry  # noqa: E402
from models.particle_transformer import (  # noqa: E402
    InteractionEmbedder,
    MaskedReconstructionPart,
    ParticleEmbedder,
)
from utils.utils import load_any_config  # noqa: E402

from g2_scaffold.losses import hierarchical_loss  # noqa: E402
from g2_scaffold.decoder import decode_batch  # noqa: E402
from g2_scaffold.metrics import METRIC_KEYS, merge_counts, rates, score_batch  # noqa: E402
from g2_scaffold.targets import targets_to_g2  # noqa: E402
from g2_scaffold.data import G2DataModule  # noqa: E402


def _candidate_outputs(raw_outputs):
    final = raw_outputs[max(raw_outputs)]
    return {
        "state_logits": final["chain_state_logits"],
        "w_pair_logits": final["w_pair_scores"],
        "b_extension_logits": final["b_extension_scores"],
    }


class G2Trainer(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        g2_cfg = config.get("g2", {})
        self.loss_mode = g2_cfg.get("loss_mode", "hard_min")
        self.loss_temperature = float(g2_cfg.get("loss_temperature", 1.0))
        self.validation_metric_batches = int(g2_cfg.get("validation_metric_batches", 2))
        self._val_counts = {key: 0 for key in METRIC_KEYS}
        train_cfg = config["model_training"]
        self.learning_rate = float(train_cfg.get("learning_rate", 2e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.01))
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch):
        return _candidate_outputs(self.model(batch))

    def _loss(self, batch):
        inputs, raw_targets = batch
        targets = targets_to_g2(raw_targets, event_ids=inputs.get("event_id"))
        outputs = self(inputs)
        loss = hierarchical_loss(
            outputs,
            targets["state_targets"],
            targets["w_targets"],
            targets["b_targets"],
            mode=self.loss_mode,
            temperature=self.loss_temperature,
            valid_particles=targets["valid_particles"],
        )
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, _ = self._loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self._loss(batch)
        if batch_idx < self.validation_metric_batches:
            targets = targets_to_g2(batch[1], event_ids=batch[0].get("event_id"))
            decoded = decode_batch(outputs, targets["valid_particles"])
            merge_counts(self._val_counts, score_batch(decoded, targets))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_counts = {key: 0 for key in METRIC_KEYS}

    def on_validation_epoch_end(self):
        counts = torch.tensor(
            [self._val_counts[key] for key in METRIC_KEYS],
            dtype=torch.float64,
            device=self.device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        reduced = {key: int(counts[index].item()) for index, key in enumerate(METRIC_KEYS)}
        for name, value in rates(reduced).items():
            if name in {"state_accuracy", "w_exact", "full_top_exact", "event_exact"}:
                self.log(f"val_{name}", value, on_step=False, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )


def build_model(config):
    particle_embedder = ParticleEmbedder(**config["model_parameters"]["particle_embedder"])
    interaction_embedder = InteractionEmbedder(**config["model_parameters"]["interaction_embedder"])
    task_registry = create_default_task_registry(config)
    model = MaskedReconstructionPart(
        particle_embedder=particle_embedder,
        interaction_embedder=interaction_embedder,
        task_registry=task_registry,
        **config["model_parameters"]["transformer"],
    )
    return model


def cpu_smoke(config):
    """Run one CPU forward/backward without opening the dataset."""
    torch.set_num_threads(1)
    model = build_model(config)
    model.train()
    B, P = 2, 20
    inputs = {
        "jet": torch.randn(B, P, 7),
        "src_mask": torch.ones(B, P, dtype=torch.bool),
        "interactions": torch.randn(B, P, P, 4),
    }
    masks = torch.zeros(B, 4, P)
    masks[:, 0, 0:3] = 1.0
    masks[:, 1, 6:9] = 1.0
    masks[:, 2, 1:3] = 1.0
    masks[:, 3, 7:9] = 1.0
    targets = {
        "jet_mask_true": masks,
        "jet_valid_mask": inputs["src_mask"],
        "target_valid_mask": torch.ones(B, 4, dtype=torch.bool),
        "classes": torch.tensor([[1, 1, 2, 2]] * B),
    }
    loss, outputs = G2Trainer(model, config)._loss((inputs, targets))
    assert torch.isfinite(loss), loss
    loss.backward()
    missing = [name for name, param in model.named_parameters()
               if param.requires_grad and param.grad is None]
    if missing:
        raise RuntimeError(f"G2 CPU smoke found parameters without gradients: {missing}")
    print(f"G2 CPU smoke OK: loss={loss.item():.5f}, outputs={sorted(outputs)}")


def train(config):
    seed = config.get("model_training", {}).get("seed")
    if seed is not None:
        pl.seed_everything(seed, workers=True)
    model = build_model(config)
    data_module = G2DataModule(config)
    data_module.setup("fit")
    checked = 0
    for inputs, raw_targets in data_module.val_dataloader():
        targets_to_g2(raw_targets, event_ids=inputs.get("event_id"))
        checked += raw_targets["jet_mask_true"].shape[0]
    if checked != len(data_module.val_dataset):
        raise RuntimeError(f"G2 validation guard checked {checked} of {len(data_module.val_dataset)} selected events")
    print(f"G2 validation target guard OK: {checked} selected events")
    train_cfg = config["model_training"]
    artefact_cfg = config["model_artefacts"]
    log_dir = artefact_cfg["log_dir"]
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(log_dir, "checkpoints"),
            filename="epoch{epoch:03d}-val_loss{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        )
    ]
    logger = TensorBoardLogger(log_dir, name="", version=int(os.environ["SLURM_JOB_ID"])
                               if os.environ.get("SLURM_JOB_ID") else None)
    trainer = pl.Trainer(
        accelerator=train_cfg.get("accelerator", "gpu"),
        devices=train_cfg.get("devices", 2),
        num_nodes=train_cfg.get("num_nodes", 1),
        strategy=train_cfg.get("strategy", "ddp"),
        precision=train_cfg.get("precision", "bf16-mixed"),
        min_epochs=train_cfg.get("min_epochs", 1),
        max_epochs=train_cfg.get("max_epochs", 2),
        logger=logger,
        callbacks=callbacks,
        default_root_dir=log_dir,
        limit_train_batches=train_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=train_cfg.get("limit_val_batches", 1.0),
    )
    trainer.fit(G2Trainer(model, config), datamodule=data_module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="g2_scaffold/g2_pilot.yaml")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    config = load_any_config(args.config)
    torch.set_float32_matmul_precision(config.get("model_training", {}).get("matmul_precision", "high"))
    if args.cpu_smoke:
        cpu_smoke(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
