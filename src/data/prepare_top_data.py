import numpy as np 
import pandas as pd
import os
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import StandardScaler
import h5py
import torch
from sklearn.compose import ColumnTransformer
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_utls.scalers import *
from src.utils.utils import load_and_split_config, load_any_config
from utils import apply_mask, calculate_energy_value

config_file_path  = "config/transformer_classifier_config.yaml"
import joblib



class TopReconstructionDatasetFromH5:
    def __init__(self, config):

        ## Use the save information from the step prior for ease.
        self.raw_file_config = config["root_dataset_prepper"]
        self.preprocessing_config = config.get("preprocessing", None)


        raw_file_prefix_and_path = os.path.join(self.raw_file_config["save_path"],
                                                 self.raw_file_config["save_file_prefix"])
        

        save_dir = Path(self.preprocessing_config["save_path"])
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
                if write == save_train:
                    joblib.dump(self.target_transformers, save_dir / "target_transforms.joblib")
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
                    targets_chunk[..., 3] = calculate_energy_value(targets_chunk[..., :])
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
            StandardScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
            LogMinMaxScaler(),
        )
        self.target_transformers = (
            LogMinMaxScaler(),
            StandardScaler(),
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
        jet_btag = jet[..., 5 ].reshape(N, P, 1)
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
        
        jet = np.concatenate(jet_transformed, axis = - 1)
        jet = np.concatenate((jet, jet_btag), axis = 2 )
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