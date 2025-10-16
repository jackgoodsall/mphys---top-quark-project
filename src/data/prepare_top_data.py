import numpy as np 
import pandas as pd
import os
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import StandardScaler
import h5py
from torch.utils.data import random_split
import torch
from sklearn.compose import ColumnTransformer
from ..data_utls.scalers import *

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
    raw_file_name = "/kaggle/input/threetop4toph5py/raw_data"
    processed_file_name =  "processed_data"
    def __init__(self):
        self._get_datas()
        self.scale_global_data()
        self.scale_particle_data()
        self.src_mask = np.all(np.isnan(self.particle_data), axis = -1)
        self.particle_data = np.nan_to_num(self.particle_data, nan = -1010)
        
    def scale_particle_data(self):
        pt_indices = np.arange(0, 25*6, 6)
        eta_indicies = pt_indices + 1
        phi_indices = pt_indices + 2
        mass_indices = pt_indices + 3 
        pipeline = ColumnTransformer([
            ("pt", LogScaler(), pt_indices),
            ("eta", ArctanScaler(), eta_indicies)
            ("phi", PhiTransformer, phi_indices),
            ("mass", LogScaler(), mass_indices)
        ], remainder="passthrough")

        m, n = self.particle_data.shape[1] , self.particle_data.shape[2]
        print(self.particle_data.shape)
        data = self.particle_data.reshape(-1, m * n )

        self.particle_data = pipeline.fit_transform(data).reshape(-1, m, n + 1)

    def scale_global_data(self):
        scaler = StandardScaler()
        self.global_data = scaler.fit_transform(self.global_data.reshape(-1, 3)).reshape(-1, 1 , 3)
        
    def _get_datas(self):
        (self.particle_data, 
        self.global_data,
        self.targets) = self._load_file(self.raw_file_name)
    
    def _load_file(self, file_name):
        with h5py.File(file_name, "r") as f:
            part_data = np.array(f["particle_features"]["all"])
            glob_data = np.array(f["global_data"]["all"])
            targets = np.array(f["targets"]["all"])
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
            print(f"Saved {name} -> {out_path}")

    def split_data(self, splits = [0.8, 0.1, 0.1]):
        self.train, self.val, self.test = random_split(self.dataset, splits)
        print(self.train[0])
    def _load_into_tensordataset(self):
        self.dataset = TopMulitplicityClassifierDataSet(self.particle_data, self.global_data,
                                                       self.src_mask, self.targets)

top_dataset = TopTensorDatasetFromH5py()       