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



class TopMulitplicityClassifierDataSet(Dataset):
    ### Torch module for Dataset, allows easy dataloader creation
    def __init__(self, particle_features, global_features, src_mask, target_labels):
        self.particle_features = particle_features
        self.global_features = global_features
        self.src_mask = src_mask
        self.target_labels = target_labels
    def __len__(self):
        return self.particle_features.shape[0]
    def __getitem__(self, idx):
        return {"particle_features": self.particle_features[idx],
               "global_features": self.global_features[idx],
               "src_mask": self.src_mask[idx]}, self.target_labels[idx]


class TopTensorDatasetFromH5py:
    def __init__(self, config):

        self.config = config
        self.pad_value = config["pad_value"]

        self._get_datas()
        self._selection_cuts()

        self.src_mask = np.all(np.isnan(self.particle_data), axis = -1)
        

        self._load_into_tensordataset()
        self.split_data()

        self.scale_global_data()
        self.fit_scale_particle_data()
        self.transform_scale_particle_data()
        
        self.particle_data = np.nan_to_num(self.particle_data, nan = self.pad_value)
        self._save_splits(self.config["processed_file_name"])

    ## Function for any event selection cuts -> currently only implements abs(charge) == 2 
    def _selection_cuts(self):
        total_charge = np.sum( np.nan_to_num(self.particle_data[... , 4], nan = 0), axis = 1)
        print(total_charge)
        cut_events = np.where(abs(total_charge) == 2, True, False)
        print(cut_events)

        self.particle_data = self.particle_data[cut_events]
        self.global_data = self.global_data[cut_events]
        self.targets = self.targets[cut_events]

    def fit_scale_particle_data(self):
        B, M, N = self.particle_data[self.train.indices].shape
        PT, ETA, PHI, MASS = 0, 1, 2, 3 

        x = self.particle_data[self.train.indices].reshape(B * M, N)
        mask_indicies = self.src_mask[self.train.indices].reshape(B* M)
        x = x[~mask_indicies]
        
        # Apply your scalers directly on slices (vectorized):
        self.pt_scaler   = LogScaler().fit(x[..., PT].reshape(-1, 1))
        self.eta_scaler =    ArctanScaler().fit(x[..., ETA].reshape(-1, 1))
        self.mass_scaler = LogScaler().fit(x[..., MASS].reshape(-1, 1))
        self.phi_scaler = PhiTransformer().fit(x[..., PHI].reshape(-1, 1)) 

    def transform_scale_particle_data(self):
        B, M, N = self.particle_data.shape
        PT, ETA, PHI, MASS = 0, 1, 2, 3
        
        pt = self.particle_data[..., PT].reshape(B * M, 1)
        eta = self.particle_data[..., ETA].reshape(B * M, 1)
        phi = self.particle_data[..., PHI].reshape(B*M, 1)
        mass = self.particle_data[..., MASS].reshape(B*M , 1)
        rest = self.particle_data[..., MASS + 1 : ]

        scaled_pt = self.pt_scaler.transform(pt).reshape(B, M, 1)
        scaled_eta = self.eta_scaler.transform(eta).reshape(B, M, 1)
        scaled_phi = self.phi_scaler.transform(phi).reshape(B, M , 2)
        scaled_mass = self.mass_scaler.transform(mass).reshape(B, M , 1)

        self.particle_data = np.concatenate((scaled_pt, scaled_eta, scaled_phi, scaled_mass,
                                           rest), axis = 2 )
        print(self.particle_data.shape)

    def scale_global_data(self):
        """
        Function to scale the global data
        """
        scaler = StandardScaler()
        self.global_data = scaler.fit_transform(self.global_data.reshape(-1, 3)).reshape(-1, 1 , 3)
        
    def _get_datas(self):
        """
        Gets path from the file defined in the config.
        Sets particle, global and target data of the object.
        """
        raw_file_path = Path(self.config["raw_file_dir"] , self.config["raw_file_name"])
        (self.particle_data, 
        self.global_data,
        self.targets) = self._load_file(raw_file_path)
    
    def _load_file(self, file_name):
        with h5py.File(file_name, "r") as f:
            part_data = np.array(f["particle_data"])
            glob_data = np.array(f["global_features"])
            targets = np.array(f["targets"])
        return part_data, glob_data, targets
        
    def _save_splits(self, stem: str, pad_value: float = -1010.0):
        """
        Save train/val/test into separate files:
          {stem}_train.h5, {stem}_val.h5, {stem}_test.h5
        Uses Subset.indices to slice original numpy arrays in one shot.
        """
        splits = {"train": self.train, "val": self.val, "test": self.test}

        for name, subset in splits.items():
            idx = np.array(subset.indices, dtype=np.int64)  # indices into original arrays

            part = self.particle_data[idx]                 # [N_split, Nmax, Fp]
            pmask = self.src_mask[idx]                     # [N_split, Nmax]
            glob = self.global_data[idx]                   # [N_split, ...]
            y = self.targets[idx]                          # [N_split]

            out_path = f"{stem}_{name}.h5"
            with h5py.File(out_path, "w") as f:
                f.create_dataset("particle_features", data=part,
                                 compression="gzip", compression_opts=4, chunks=True)
                f.create_dataset("particle_mask", data=pmask.astype(np.bool_),
                                 compression="gzip", compression_opts=4, chunks=True)
                f.create_dataset("global_features", data=glob,
                                 compression="gzip", compression_opts=4, chunks=True)
                f.create_dataset("targets", data=y.astype(np.int64),
                                 compression="gzip", compression_opts=4, chunks=True)
                f.attrs["pad_value"] = float(pad_value)
                f.attrs["class_weights"] = self.class_weights
            print(f"Saved {name} -> {out_path}")

    def split_data(self, splits = None):
        if splits is None:
            splits = self.config["split_sizes"]
        self.train, self.val, self.test = random_split(self.dataset, splits)

        idx = np.array(self.train.indices)
        labels = np.unique(self.targets[idx])
        cls_weights = []
        for label in labels:
            cls_weights.append(np.sum(self.targets[idx] == label) / len(self.train))
        self.class_weights = np.array(cls_weights)
        print(self.class_weights)

    def _load_into_tensordataset(self):
        self.dataset = TopMulitplicityClassifierDataSet(self.particle_data, self.global_data,
                                                       self.src_mask, self.targets)
