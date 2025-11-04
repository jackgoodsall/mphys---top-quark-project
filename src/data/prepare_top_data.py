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
from src.utils.utils import load_and_split_config, load_any_config
from utils import apply_mask

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


class TopReconstructionDatasetFromH5:
    def __init__(self, config):

        ## Use the save information from the step prior for ease.
        self.raw_file_config = config["root_dataset_prepper"]
        self.preprocessing_config = config.get("preprocessing", None)


        raw_file_prefix_and_path = os.path.join(self.raw_file_config["save_path"],
                                                 self.raw_file_config["save_file_prefix"])
        
        save_file_prefix_and_path = os.path.join(self.preprocessing_config["save_path"],
                                                 self.preprocessing_config["save_file_prefix"])
        
    
        stream_size = self.preprocessing_config["stream_size"]

        

        (raw_train,  
        raw_val,
        raw_test) = (raw_file_prefix_and_path +"train.h5",
                    raw_file_prefix_and_path + "val.h5",
                    raw_file_prefix_and_path + "test.h5") 
        

        (save_train,
         save_val,
         save_test) = (
                save_file_prefix_and_path +"train.h5",
                save_file_prefix_and_path + "val.h5",
                save_file_prefix_and_path + "test.h5"
         )

        ## Temp is the filtered train set
        temp_file = "temp_file.h5"
        temp =  save_file_prefix_and_path + temp_file
        self.last_temp_idx = 0

        temp_exits = Path(temp)
        temp_exits.touch(exist_ok= True)
        print(temp_exits)
        self._init_transformers()

        read_file_order = (raw_train, temp, raw_val, raw_test)
        write_file_order = (temp, save_train, save_val, save_test)

        for read, write in zip(read_file_order, write_file_order) :
            with h5py.File(read, "r") as read_file, h5py.File(write, "w") as write_file:
                file_len = read_file["jet"].shape[0]
                TopReconstructionDatasetFromH5._create_datagroups(write_file)
                ## Go through the batches
                for i in range(0, file_len, stream_size):
                    ## Read the batches
                    jet_chunk = read_file["jet"][i: i + stream_size]
                    event_chunk = read_file["event"][i: i + stream_size]
                    targets_chunk = read_file["targets"][i: i + stream_size]
          
                    ## Select top quarks 
                    if read == temp:
                        print(targets_chunk.shape)
                    top_quark_selection = self._select_top_quarks(targets_chunk)
                    targets_chunk = targets_chunk[top_quark_selection].reshape(-1, 2, 5)

                    ## Get mask of events to cut
                    mask = self._selection_cuts(
                        jet_chunk,
                        event_chunk,
                        targets_chunk
                    ).squeeze()

                    ## Apply the mask using helper function
                    jet_chunk, event_chunk, targets_chunk = apply_mask((jet_chunk, event_chunk, targets_chunk), mask)

                    if read == raw_train:
                        ## Transform and save temp data
                        self._fit_transformers(jet_chunk, targets_chunk)
                        self._save_data_chunks(write_file, jet_chunk, event_chunk, jet_chunk[..., 0],targets_chunk)
                    else:
                        ## Transform and save data
                        jet_chunk, targets_chunk = self._transform(jet_chunk, targets_chunk)
                        jet_chunk, src_mask_chunk = self._pad_and_src_mask(jet_chunk)
                        self._save_data_chunks(write_file, jet_chunk, event_chunk,src_mask_chunk ,targets_chunk)
                        if read == temp:
                            self.last_temp_idx += jet_chunk.shape[0]
                
    def _init_transformers(self):
        self.jet_transformers = (
            LogMinMaxScaler(),
            ArctanScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
            LogMinMaxScaler(),
        )
        self.target_transformers = (
            LogMinMaxScaler(),
            ArctanScaler(),
            StandardScaler(),
            LogMinMaxScaler()
        )
    
    def _selection_cuts(self, 
                        jet,
                        event,
                        target):
        event_selection_mask = (event[:, 2] == 1).reshape(-1, 1)
        return event_selection_mask
    
    def _select_top_quarks(
            self,
            target
    ):
        target_pid = target[..., 4]
        top_quark_mask = (abs(target_pid) == 6)
        return top_quark_mask

    def _fit_transformers(self, jet, targets):
        
        N, P, F = jet.shape
        pt, eta, phi, mass, e = 0, 1, 2, 3, 4
        ## Remove pid information don't need it 
        targets = targets[...,0:4]
        jet = jet.reshape(N* P, F)
        targets = targets.reshape(N * 2, 4)

        jet_variables = (jet_pt,jet_et,jet_phi,jet_mass,jet_e) = jet[:, pt], jet[:, eta],jet[:, phi], jet[:, mass], jet[:, e]
        target_variables = (target_pt, target_et, target_phi, target_mass) = targets[:, pt], targets[:,eta],targets[:, phi], targets[:, mass]

        for variable, transformation in zip(jet_variables, self.jet_transformers):
            transformation.partial_fit(variable.reshape(-1, 1))
            
        for variable, transformation in zip(target_variables, self.target_transformers):
            transformation.partial_fit(variable.reshape(-1, 1))
    
    def _save_data_chunks(self, file, jet_chunk, event_chunk,src_mask_chunk, targets_chunk):

        jet_ds = file["jet"]
        event_ds = file["event"]
        target_ds = file["targets"]
        src_ds = file["src_mask"]
        _, N, F = jet_chunk.shape
        _, M, P = targets_chunk.shape
        print(src_mask_chunk.shape)
        cur_len = file["jet"].shape[0]
        n0, n1 = cur_len, cur_len + jet_chunk.shape[0]

        jet_ds.resize((n1 , N, F))
        src_ds.resize((n1 , N))
        event_ds.resize((n1 , 3) )
        target_ds.resize((n1 , M, P))
        
        jet_ds[n0 : n1] = jet_chunk
        src_ds[n0 : n1] = src_mask_chunk
        event_ds[n0 : n1] = event_chunk
        target_ds[n0 : n1] = targets_chunk


    def _transform(self, jet, targets):

        ## Same process of the fit transform function - get the in the right shape and seperate it.
        N, P, F = jet.shape
        pt, eta, phi, mass, e = 0, 1, 2, 3, 4
        ## Remove pid information don't need it 
        targets = targets[...,0:4]
        jet = jet.reshape(N* P, F)
        targets = targets.reshape(N * 2, 4)
        # Sepererate and store in a tuple for easy iteration
        jet_variables = (jet_pt,jet_et,jet_phi,
                         jet_mass,jet_e) = jet[:, pt], jet[:, eta],jet[:, phi], jet[:, mass], jet[:, e]
        target_variables = (target_pt, target_et, 
                            target_phi, target_mass) = targets[:, pt], targets[:,eta], targets[:, phi], targets[:, mass]

        jet_transformed = []
        target_transformed = []

        # Transformed the data and append to list
        for variable, transformation in zip(jet_variables, self.jet_transformers):
            jet_transformed.append(transformation.transform(variable.reshape(-1, 1)).reshape(N, P, -1 ))
        for variable, transformation in zip(target_variables, self.target_transformers):
            target_transformed.append(transformation.transform(variable.reshape(-1, 1)).reshape(N, 2, -1))
        
        jet = np.concatenate(jet_transformed, axis = -1)
        target = np.concatenate(target_transformed, axis = -1)

        
        return jet, target
        
    def _pad_and_src_mask(self, jet_chunk, pad_value  = -99):
        self.pad_value = pad_value

        src_mask = np.all(np.isnan(jet_chunk), axis = -1)
        
        jet_chunk = np.nan_to_num(jet_chunk, pad_value)

        return jet_chunk, src_mask
    

    @staticmethod
    def _create_datagroups(file):
        ## Creates the following datasets in a h5py file object
        jet_ds = file.create_dataset(
            "jet",
            shape = (0,0,0),
            maxshape = (None, None,None)
        )
        event_ds = file.create_dataset(
            "event",
            shape = (0,0),
            maxshape = (None, None)
        )
        target_ds = file.create_dataset(
            "targets",
            shape = (0,0,0),
            maxshape = (None, None,None)
        )
        file.create_dataset(
            "src_mask",
            shape = (0,0),
            maxshape = (None,None)
        )


if __name__ == "__main__":
    config = load_any_config("config/top_reconstruction_config.yaml")
    top_reconstruction_dataset = TopReconstructionDatasetFromH5(config)

""" 

if __name__ == "__main__":
    config = load_and_split_config(config_file_path)
    config = config.data_pipeline["data_preprocessing"]
    top_dataset = TopTensorDatasetFromH5py(config)       
     """