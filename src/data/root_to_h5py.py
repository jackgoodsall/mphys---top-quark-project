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


PARTICLE_PREFIXES = {
    "jet" : 0,
    "el" : 1,
    "mu" : 2
    
}

class TopClassiferDataSetPrepaper:

    def __init__(self, config):
        self.config = config.data_pipeline["root_to_h5py"]
        raw_data_dir = self.config["raw_data_dir"]
        self.target_file_dict = { **{ raw_data_dir + file : 1 for file in  self.config["signal_files"] },
                                 **{raw_data_dir + file : 0 for file in self.config["background_files"] 
        }}
        (
         particle_data, 
         global_data, 
         src_mask,
         targets
        ) = self.parse_root_file(
             max_particles = self.config["max_particles"],
             particle_features = self.config["particle_features"],
             global_features = self.config["global_features"]
                )  
        self.particle_data = particle_data
        self.global_data = global_data
        self.targets = targets
        
    
    def parse_root_file(self, max_particles, particle_features, global_features):
        per_file_events, per_file_targets, per_file_globals = [], [], []

        for path, y in self.target_file_dict.items():
            reco = uproot.open(path)["Reco;1"]
            blocks = []

            gdict = reco.arrays(global_features, how=dict)
            gstack = ak.concatenate([gdict[name][..., None] for name in global_features], axis=-1)
            per_file_globals.append(gstack)

            for prefix in PARTICLE_PREFIXES.keys():
                feats = [f"{prefix}{fe}" for fe in particle_features]
                bdict = reco.arrays(feats, how=dict)
                base  = ak.concatenate([bdict[name][..., None] for name in feats], axis=-1)
                # Comment charge out for now
                if prefix in ("el", "mu"):
                    chd = reco.arrays(f"{prefix}_charge", how=dict)
                    ch  = ak.concatenate([chd[k][..., None] for k in chd], axis=-1)
                else:
                    ch = ak.zeros_like(base[..., :1])

                if prefix == "jet":
                    btd = reco.arrays(f"{prefix}_btag", how=dict)
                    bt  = ak.concatenate([btd[k][..., None] for k in btd], axis=-1)
                else:
                    bt = ak.zeros_like(base[..., :1])

                blocks.append(ak.concatenate([base,  bt], axis=-1))

            events = ak.concatenate(blocks, axis=1)
            per_file_events.append(events)
            per_file_targets.append(torch.full((len(events), 1), int(y), dtype=torch.long))

        global_arr = ak.concatenate(per_file_globals, axis=0)
        global_arr = torch.from_numpy(ak.to_numpy(global_arr).astype(np.float32, copy=False))

        arr = ak.concatenate(per_file_events, axis=0)                  
        arr = ak.pad_none(arr, max_particles, axis=1, clip=True)

        pad_mask_np   = ak.to_numpy(ak.is_none(arr, axis=-1))          
        arr           = ak.fill_none(arr, np.nan)                     
        src_mask = allnan_mask_np = ak.to_numpy(ak.all(np.isnan(arr), axis=-1))   


        dense_np = ak.to_numpy(arr).astype(np.float32, copy=False)
        ## Commented out to save file in h5oy format with nans
        #np.nan_to_num(dense_np, copy=False, nan=-1010, posinf=-1010, neginf=-1010)
        set_array = torch.from_numpy(dense_np)

        target_array = torch.cat(per_file_targets, dim=0)
        return set_array, global_arr, src_mask, target_array

    def _save_dataset(self, save_file_name: str):
        ## Assumes the data set is already created tbh cba coding in the checks
        with h5py.File(save_file_name, "w") as f:
            dset = f.create_dataset("particle_data", data = self.particle_data)
            glo_dset = f.create_dataset("global_features" , data = self.global_data)
            target_dset = f.create_dataset("targets", data = self.targets)


   
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
