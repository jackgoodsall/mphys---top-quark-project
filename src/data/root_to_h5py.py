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
from src.utils.utils import  load_and_split_config

config_file_path = "config/transformer_classifier_config.yaml"



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


if __name__ == "__main__":
    config = load_and_split_config(config_file_path)
    data_set = TopClassiferDataSetPrepaper(config)
    data_set._save_dataset(config.data_pipeline["root_to_h5py"]["file_save_name"])
