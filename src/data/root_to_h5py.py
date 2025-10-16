import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from torch.utils.data import Dataset
import awkward as ak

import h5py
from torch.utils.data import random_split
import uproot
import torch

DATA_DIR = "data/fourtopvsthree/rootfiles/"


class FileConfig:
    files = {DATA_DIR + "3tJ_LO_final.root" : 0,
              DATA_DIR+ "3tWm_LO_final.root" : 0,
             DATA_DIR +   "3tWp_LO_final.root" : 0, 
             DATA_DIR +    "4top_2LSS_April18.root" : 1}

PARTICLE_PREFIXES = {
    "jet" : 0,
    "el" : 1,
    "mu" : 2
    
}

class TopClassiferDataSetPrepaper:
    file_config = FileConfig
    def __init__(self):
        (
         particle_data, 
         global_data, 
         src_mask,
         targets
        ) = self.parse_root_file(
             max_particles = 25,
             particle_features = ["_pt", "_eta", "_phi", "_mass"],
             global_features = ["met_met", "met_eta", "met_phi"]
                )  
        self.particle_data = particle_data
        self.global_data = global_data
        self.targets = targets
    
    def parse_root_file(self, max_particles, particle_features, global_features):
        per_file_events, per_file_targets, per_file_globals = [], [], []

        for path, y in self.file_config.files.items():
            reco = uproot.open(path)["Reco;1"]
            blocks = []

            gdict = reco.arrays(global_features, how=dict)
            gstack = ak.concatenate([gdict[name][..., None] for name in global_features], axis=-1)
            per_file_globals.append(gstack)

            for prefix in PARTICLE_PREFIXES.keys():
                feats = [f"{prefix}{fe}" for fe in particle_features]
                bdict = reco.arrays(feats, how=dict)
                base  = ak.concatenate([bdict[name][..., None] for name in feats], axis=-1)

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

                blocks.append(ak.concatenate([base, ch, bt], axis=-1))

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
            dset = f.create_dataset("particle_data", data = np.array(self.particle_data))
            glo_dset = f.create_dataset("global_features" , data = np.array(self.global_data))
            target_dset = f.create_dataset("targets", data = np.array(self.targets))


data_set = TopClassiferDataSetPrepaper()
data_set._save_dataset("all_combined_raw")