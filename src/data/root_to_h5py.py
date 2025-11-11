import numpy as np 
import pandas as pd 

import os

from torch.utils.data import Dataset
import awkward as ak

import h5py
from torch.utils.data import random_split
import uproot
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.utils import  load_and_split_config, load_any_config

config_file_path = "config/transformer_classifier_config.yaml"
reco_config_file_path = "config/top_reconstruction_config.yaml"

   
class TopAntiTopRegressionDataSetPrep:
    max_particles = 20
    n_top_quarks = 2
    particle_features = [
        "jet_pt",
        "jet_eta",
        "jet_phi",
        "jet_e",
        "jet_m",
        "jet_bTag",
    ]
    target_features = [
        "particle_pt",
        "particle_eta",
        "particle_phi",
        "particle_m",
        "particle_pid",
    ]
    event_features = [
        "njet",
        "nbTagged",
        "allMatchedEvent",
    ]

    def __init__(self, config: dict):
        root_path = config["root_path"]
        train_name = config["train_file_name"]
        val_name = config["val_file_name"]
        test_name = config["test_file_name"]
        save_dir = config["save_path"]
        save_name = config["save_file_prefix"]
        step_size = config.get("step_size", 50_000)

        os.makedirs(save_dir, exist_ok=True)

        for name, stage in zip((train_name, val_name, test_name),
                               ("train", "val", "test")):
            root_file = os.path.join(root_path, name) + ".root"
            h5_path = os.path.join(save_dir, save_name+  stage + ".h5")
            self._save_dataset(
                h5_path,
                self.parse_root_file(
                    self.max_particles,
                    self.n_top_quarks,
                    root_file,
                    step_size=step_size,
                ),
            )

    def parse_root_file(
        self,
        max_particles: int,
        n_top_quarks: int,
        file_path: str,
        step_size: int = 50_000,
    ):
        needed = list(
            set(self.particle_features + self.target_features + self.event_features)
        )

        # Open explicitly; avoid wildcard tree names
        with uproot.open(file_path) as f:
            tree = f["Delphes;1"] if "Delphes;1" in f else f["Delphes"]

            for arrays in tree.iterate(expressions=needed, step_size=step_size):
                # Jets -> (B, max_particles=20, F=6)
                bdict = {k: arrays[k] for k in self.particle_features}
                base = ak.concatenate(
                    [bdict[name][..., None] for name in self.particle_features],
                    axis=-1,
                )
                base = ak.pad_none(base, max_particles, axis=1, clip=True)
                base = ak.fill_none(base, np.nan)
                jet_np = ak.to_numpy(base).astype(np.float32, copy=False)

                # Targets -> (B, 4, 5)
                tdict = {k: arrays[k] for k in self.target_features}
                tarr = ak.concatenate(
                    [tdict[name][..., None] for name in self.target_features],
                    axis=-1,
                )
                targets_np = ak.to_numpy(tarr).astype(np.float32, copy=False)
                if targets_np.ndim == 2:
                    # If flattened (B, 20) => reshape to (B, 4, 5)
                    targets_np = targets_np.reshape(targets_np.shape[0], 4, 5)

                # Event globals -> (B, 3)
                edict = {k: arrays[k] for k in self.event_features}
                earr = ak.concatenate(
                    [edict[name][..., None] for name in self.event_features],
                    axis=-1,
                )
                event_np = ak.to_numpy(earr).astype(np.float32, copy=False)

                yield jet_np, event_np, targets_np

    def _save_dataset(self, h5_path: str, batches):
        with h5py.File(h5_path, "w") as f:
            part_d = None
            event_d = None
            targ_d = None

            for jet_np, event_np, targets_np in batches:
                if part_d is None:
                    # /jet: (N, 20, 6)
                    part_d = f.create_dataset(
                        "jet",
                        data=jet_np,
                        maxshape=(None, jet_np.shape[1], jet_np.shape[2]),
                        chunks=(
                            min(2048, max(1, jet_np.shape[0])),
                            jet_np.shape[1],
                            jet_np.shape[2],
                        ),
                        compression="gzip",
                        compression_opts=4,
                    )
                    # /event: (N, 3)
                    event_d = f.create_dataset(
                        "event",
                        data=event_np,
                        maxshape=(None, event_np.shape[1]),
                        chunks=(min(4096, max(1, event_np.shape[0])), event_np.shape[1]),
                        compression="gzip",
                        compression_opts=4,
                    )
                    # /targets: (N, 4, 5)
                    targ_d = f.create_dataset(
                        "targets",
                        data=targets_np,
                        maxshape=(None, targets_np.shape[1], targets_np.shape[2]),
                        chunks=(
                            min(2048, max(1, targets_np.shape[0])),
                            targets_np.shape[1],
                            targets_np.shape[2],
                        ),
                        compression="gzip",
                        compression_opts=4,
                    )
                else:
                    # append
                    n0, n1 = part_d.shape[0], part_d.shape[0] + jet_np.shape[0]
                    part_d.resize((n1, part_d.shape[1], part_d.shape[2]))
                    part_d[n0:n1] = jet_np

                    n0, n1 = event_d.shape[0], event_d.shape[0] + event_np.shape[0]
                    event_d.resize((n1, event_d.shape[1]))
                    event_d[n0:n1] = event_np

                    n0, n1 = targ_d.shape[0], targ_d.shape[0] + targets_np.shape[0]
                    targ_d.resize((n1, targ_d.shape[1], targ_d.shape[2]))
                    targ_d[n0:n1] = targets_np     

if __name__ == "__main__":
    """ config = load_and_split_config(config_file_path)
    data_set = TopClassiferDataSetPrepaper(config)
    data_set._save_dataset(config.data_pipeline["root_to_h5py"]["file_save_name"])
    """
    config = load_any_config(reco_config_file_path)
    data_set = TopAntiTopRegressionDataSetPrep(config["root_dataset_prepper"])
