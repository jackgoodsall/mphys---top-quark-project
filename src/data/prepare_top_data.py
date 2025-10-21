import numpy as np 
import pandas as pd
import os
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import StandardScaler
import h5py
from torch.utils.data import random_split
import torch
from sklearn.compose import ColumnTransformer
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_utls.scalers import *
from src.utils.utils import load_and_split_config

config_file_path  = "config/transformer_classifier_config.yaml"

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

        self._get_datas()
        self.scale_global_data()
        self.scale_particle_data()
        self.src_mask = np.all(np.isnan(self.particle_data), axis = -1)
        self.particle_data = np.nan_to_num(self.particle_data, nan = -1010)
        self._load_into_tensordataset()
        self.split_data()
        self._save_splits(self.config["processed_file_name"])

    def scale_particle_data(self):
        B, M, N = self.particle_data.shape
        PT, ETA, PHI, MASS = 0, 1, 2, 3 

        x = self.particle_data.copy()

        # Apply your scalers directly on slices (vectorized):
        x[..., PT]   = LogScaler().fit_transform(x[..., PT].reshape(-1, 1)).reshape(B, M)
        x[..., ETA]  = ArctanScaler().fit_transform(x[..., ETA].reshape(-1, 1)).reshape(B, M)
        x[..., MASS] = LogScaler().fit_transform(x[..., MASS].reshape(-1, 1)).reshape(B, M)

        # Phi expands to two features (e.g., sin/cos) – insert them right after PHI to preserve per-particle order
        phi_out = PhiTransformer().fit_transform(x[..., PHI].reshape(-1, 1))  # shape (B*M, 2)
        phi_out = phi_out.reshape(B, M, 2)

        # Rebuild features with phi split in place
        x = np.concatenate([x[..., :PHI], phi_out, x[..., PHI+1:]], axis=-1)  

        self.particle_data = x

    def scale_global_data(self):
        scaler = StandardScaler()
        self.global_data = scaler.fit_transform(self.global_data.reshape(-1, 3)).reshape(-1, 1 , 3)
        
    def _get_datas(self):
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
        

    def _load_into_tensordataset(self):
        self.dataset = TopMulitplicityClassifierDataSet(self.particle_data, self.global_data,
                                                       self.src_mask, self.targets)


if __name__ == "__main__":
    config = load_and_split_config(config_file_path)
    config = config.data_pipeline["data_preprocessing"]
    top_dataset = TopTensorDatasetFromH5py(config)       
    