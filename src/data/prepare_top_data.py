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
from utils import apply_mask, calculate_energy_value, convert_polar_to_cartesian, create_interaction_matrix

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
                    if read != temp:
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
        
        jet_ds[n0 : n1] = jet_chunk.astype("float32")
        src_ds[n0 : n1] = src_mask_chunk.astype("float32")
        event_ds[n0 : n1] = event_chunk.astype("float32")
        target_ds[n0 : n1] = targets_chunk.astype("float32")


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
            maxshape = (None, None,None),
             compression="gzip",                
            compression_opts=4,  
        )
        event_ds = file.create_dataset(
            "event",
            shape = (0,0),        
            compression="gzip",                
            compression_opts=4,   
            maxshape = (None, None)
        )
        target_ds = file.create_dataset(
            "targets",
            shape = (0,0,0),
            maxshape = (None, None,None),     
            compression="gzip", 
            compression_opts=4,               
        )
        file.create_dataset(
            "src_mask",
            shape = (0,0),
            maxshape = (None,None),         
            compression="gzip",                 
            compression_opts=4,   
        )


class CartesianTopReconstructionDatasetFromH5:
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
                    if read != temp:
                        targets_chunk[..., :4] = convert_polar_to_cartesian(targets_chunk)
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
            StandardScaler()
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
        
        jet_ds[n0 : n1] = jet_chunk.astype("float32")
        src_ds[n0 : n1] = src_mask_chunk.astype("float32")
        event_ds[n0 : n1] = event_chunk.astype("float32")
        target_ds[n0 : n1] = targets_chunk.astype("float32")


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
            maxshape = (None, None,None),
             compression="gzip",                
            compression_opts=4,  
        )
        event_ds = file.create_dataset(
            "event",
            shape = (0,0),        
            compression="gzip",                
            compression_opts=4,   
            maxshape = (None, None)
        )
        target_ds = file.create_dataset(
            "targets",
            shape = (0,0,0),
            maxshape = (None, None,None),     
            compression="gzip", 
            compression_opts=4,               
        )
        file.create_dataset(
            "src_mask",
            shape = (0,0),
            maxshape = (None,None),         
            compression="gzip",                 
            compression_opts=4,   
        )


class InteractionTopReconstructionDatasetFromH5:
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
        Path(self.preprocessing_config["save_path"]).mkdir(exist_ok = True)
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
                
                file_len, N, P = read_file["jet"].shape
                _ , M, Q = read_file["targets"].shape
                
                self._create_datagroups(write_file, file_len, N, M, P, Q)
                ## Go through the batches
                for i in range(0, file_len, stream_size):
                    ## Read the batches
                    jet_chunk = read_file["jet"][i: i + stream_size]
                    event_chunk = read_file["event"][i: i + stream_size]
                    targets_chunk = read_file["targets"][i: i + stream_size]
                    if read == temp:
                        interaction_chunk = read_file["interactions"][i: i + stream_size]
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
                    if read != temp:
                        targets_chunk[..., 3] = calculate_energy_value(targets_chunk[..., :])
                        interaction_chunk = create_interaction_matrix(jet_chunk)
                    if read == raw_train:
                        ## Transform and save temp data
                        self._fit_transformers(jet_chunk, targets_chunk, interaction_chunk)
                        self._save_data_chunks(write_file, jet_chunk, interaction_chunk,event_chunk, jet_chunk[..., 0],targets_chunk)
                    else:
                        ## Transform and save data
                        jet_chunk, targets_chunk, interaction_chunk = self._transform(jet_chunk, targets_chunk, interaction_chunk)
                        jet_chunk, src_mask_chunk, interaction_chunk = self._pad_and_src_mask(jet_chunk, interaction_chunk)
                        self._save_data_chunks(write_file, jet_chunk, interaction_chunk,event_chunk,src_mask_chunk ,targets_chunk)
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
            PhiTransformer(),
            LogMinMaxScaler()
        )

        self._interaction_transformers = LogMinMaxScaler()
    
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

    def _fit_transformers(self, jet, targets, interactions):
        
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

        self._interaction_transformers.partial_fit(interactions.reshape(N * P * P, -1))

    def _save_data_chunks(self, file, jet_chunk, interaction_chunk,event_chunk,src_mask_chunk, targets_chunk):

        jet_ds = file["jet"]
        event_ds = file["event"]
        target_ds = file["targets"]
        src_ds = file["src_mask"]
        interaction_ds = file["interactions"]

        _, N, F = jet_chunk.shape
        _, M, P = targets_chunk.shape
        print(src_mask_chunk.shape)
        cur_len = file["jet"].shape[0]
        n0, n1 = cur_len, cur_len + jet_chunk.shape[0]

        jet_ds.resize((n1,) + jet_chunk.shape[ 1:])
        src_ds.resize((n1 ,) + src_ds.shape[1:])
        event_ds.resize((n1 , 3) )
        target_ds.resize((n1 , M, P))
        interaction_ds.resize((n1, N, N, 4))
        
        jet_ds[n0 : n1] = jet_chunk.astype("float32")
        src_ds[n0 : n1] = src_mask_chunk.astype("float32")
        event_ds[n0 : n1] = event_chunk.astype("float32")
        target_ds[n0 : n1] = targets_chunk.astype("float32")
        interaction_ds[n0 : n1] = interaction_chunk.astype("float32")

    def _transform(self, jet, targets, interactions):
        
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

        interactions = self._interaction_transformers.transform(interactions.reshape(N * P * P, -1)).reshape(N, P, P, -1)

        
        return jet, target, interactions
        
    def _pad_and_src_mask(self, jet_chunk,interaction_chunk, pad_value  = -99):
        self.pad_value = pad_value

        src_mask = np.all(np.isnan(jet_chunk), axis = -1)
        
        jet_chunk = np.nan_to_num(jet_chunk, pad_value)
        interaction_chunk = np.nan_to_num(interaction_chunk, -np.inf)

        return jet_chunk, src_mask, interaction_chunk
    

    def _create_datagroups(self, file, *args):
        ## Creates the following datasets in a h5py file object
        file_len, N, M, P, Q = args


        jet_ds = file.create_dataset(
            "jet",
            shape = (0,N,P),
            maxshape = (None, N, 7),
             compression="gzip",                
            compression_opts=4,  
        )
        event_ds = file.create_dataset(
            "event",
            shape = (0,0),        
            compression="gzip",                
            compression_opts=4,   
            maxshape = (None, None)
        )
        target_ds = file.create_dataset(
            "targets",
            shape = (0,2,Q),
            maxshape = (None, 2,Q + 1),     
            compression="gzip", 
            compression_opts=4,               
        )
        interaction_ds = file.create_dataset(
            "interactions",
            shape = (0,N,N,4),
            maxshape = (None, N, N, 4),
            compression = "gzip",
            compression_opts = 4,
        )
        file.create_dataset(
            "src_mask",
            shape = (0,N),
            maxshape = (None,N),         
            compression="gzip",                 
            compression_opts=4,   
        )


class WBosonInteractionTopReconstructionDatasetFromH5:
    def __init__(self, config):

        self.raw_file_config = config["root_dataset_prepper"]
        self.preprocessing_config = config.get("preprocessing", None)

        raw_file_prefix_and_path = os.path.join(
            self.raw_file_config["save_path"],
            self.raw_file_config["save_file_prefix"],
        )

        save_dir = Path(self.preprocessing_config["save_path"])
        save_file_prefix_and_path = os.path.join(
            self.preprocessing_config["save_path"],
            self.preprocessing_config["save_file_prefix"],
        )

        self.stream_size = self.preprocessing_config["stream_size"]

        self.raw_train = raw_file_prefix_and_path + "train.h5"
        self.raw_val   = raw_file_prefix_and_path + "val.h5"
        self.raw_test  = raw_file_prefix_and_path + "test.h5"

        self.save_train = save_file_prefix_and_path + "train.h5"
        self.save_val   = save_file_prefix_and_path + "val.h5"
        self.save_test  = save_file_prefix_and_path + "test.h5"

        Path(self.preprocessing_config["save_path"]).mkdir(exist_ok=True)

        self._init_transformers()

        # 1) Fit transformers on train only
        self._fit_on_file(self.raw_train)

        # Save transformers (tops and Ws separately)
        joblib.dump(self.target_transformers, save_dir / "target_transforms.joblib")
        joblib.dump(self.W_target_transformers, save_dir / "W_target_transforms.joblib")

        # 2) Transform and save train/val/test
        self._transform_file(self.raw_train, self.save_train)
        self._transform_file(self.raw_val,   self.save_val)
        self._transform_file(self.raw_test,  self.save_test)

    # -----------------------------
    # Init and selection logic
    # -----------------------------
    def _init_transformers(self):
        self.jet_transformers = (
            LogMinMaxScaler(),
            StandardScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
            LogMinMaxScaler(),
        )
        # Top quark targets
        self.target_transformers = (
            LogMinMaxScaler(),
            StandardScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
        )
        # W boson targets – same structure but independent instances
        self.W_target_transformers = (
            LogMinMaxScaler(),
            StandardScaler(),
            PhiTransformer(),
            LogMinMaxScaler(),
        )

        self._interaction_transformers = LogMinMaxScaler()

    def _selection_cuts(self, jet, event, target):
        # Currently: require event[:, 2] == 1
        event_selection_mask = (event[:, 2] == 1).reshape(-1)
        return event_selection_mask

    # -----------------------------
    # High-level file passes
    # -----------------------------
    def _fit_on_file(self, raw_path):
        """First pass on train file: fit transformers only, no writing."""
        with h5py.File(raw_path, "r") as f:
            jets    = f["jet"]
            events  = f["event"]
            targets = f["targets"]

            file_len = jets.shape[0]

            for i in range(0, file_len, self.stream_size):
                jet_chunk    = jets[i: i + self.stream_size]
                event_chunk  = events[i: i + self.stream_size]
                targets_chunk = targets[i: i + self.stream_size]

                # Select valid events and extract tops + Ws
                selection = self._select_events_and_targets(jet_chunk, event_chunk, targets_chunk)
                if selection is None:
                    continue

                jet_chunk, event_chunk, top_targets_chunk, W_targets_chunk = selection

                # Compute energies
                top_targets_chunk[..., 3] = calculate_energy_value(top_targets_chunk[..., :])
                W_targets_chunk[..., 3]   = calculate_energy_value(W_targets_chunk[..., :])

                # Interactions from jets
                interaction_chunk = create_interaction_matrix(jet_chunk)

                # Fit transformers
                self._fit_transformers(jet_chunk, top_targets_chunk, W_targets_chunk, interaction_chunk)

    def _transform_file(self, raw_path, save_path):
        """Second pass (or first for val/test): transform and write out."""
        with h5py.File(raw_path, "r") as read_file, h5py.File(save_path, "w") as write_file:
            jets    = read_file["jet"]
            events  = read_file["event"]
            targets = read_file["targets"]

            file_len = jets.shape[0]
            datasets_created = False

            for i in range(0, file_len, self.stream_size):
                jet_chunk    = jets[i: i + self.stream_size]
                event_chunk  = events[i: i + self.stream_size]
                targets_chunk = targets[i: i + self.stream_size]

                # Select valid events and extract tops + Ws
                selection = self._select_events_and_targets(jet_chunk, event_chunk, targets_chunk)
                if selection is None:
                    continue

                jet_chunk, event_chunk, top_targets_chunk, W_targets_chunk = selection

                # Compute energies
                top_targets_chunk[..., 3] = calculate_energy_value(top_targets_chunk[..., :])
                W_targets_chunk[..., 3]   = calculate_energy_value(W_targets_chunk[..., :])

                # Interactions
                interaction_chunk = create_interaction_matrix(jet_chunk)

                # Transform
                jet_chunk, top_targets_chunk, W_targets_chunk, interaction_chunk = self._transform(
                    jet_chunk,
                    top_targets_chunk,
                    W_targets_chunk,
                    interaction_chunk,
                )

                # Pad and src_mask
                jet_chunk, src_mask_chunk, interaction_chunk = self._pad_and_src_mask(
                    jet_chunk,
                    interaction_chunk,
                )

                # Create datasets on first non-empty chunk
                if not datasets_created:
                    self._create_datagroups(
                        write_file,
                        jet_chunk.shape,
                        event_chunk.shape,
                        top_targets_chunk.shape,
                        W_targets_chunk.shape,
                        interaction_chunk.shape,
                    )
                    datasets_created = True

                # Save chunk
                self._save_data_chunks(
                    write_file,
                    jet_chunk,
                    interaction_chunk,
                    event_chunk,
                    src_mask_chunk,
                    top_targets_chunk,
                    W_targets_chunk,
                )

    # -----------------------------
    # Event-wise selection
    # -----------------------------
    def _select_events_and_targets(self, jet_chunk, event_chunk, targets_chunk):
        """
        Take raw chunk (B, ...) and:
          - apply event-level cuts
          - require exactly 2 tops (PID ±6) and 2 Ws (PID ±24) per event
          - return filtered jets/events and (tops, Ws) as (B', 2, Q)
        """
  

        pid = targets_chunk[..., 4]  # (B, M)
        top_mask = np.abs(pid) == 6
        W_mask   = np.abs(pid) == 24

        num_tops = top_mask.sum(axis=1)  # (B,)
        num_Ws   = W_mask.sum(axis=1)    # (B,)

        # Base event cuts
        base_mask = self._selection_cuts(jet_chunk, event_chunk, targets_chunk)  # (B,)

        # Require exactly 2 tops and 2 Ws and pass base mask
        keep = (num_tops == 2) & (num_Ws == 2) & base_mask

 

        jet_chunk    = jet_chunk[keep]
        event_chunk  = event_chunk[keep]
        targets_chunk = targets_chunk[keep]
        top_mask     = top_mask[keep]
        W_mask       = W_mask[keep]

        Q = targets_chunk.shape[-1]

        # Extract tops and Ws per event; 2 of each per event => safe reshape
        top_targets_chunk = targets_chunk[top_mask].reshape(-1, 2, Q)
        W_targets_chunk   = targets_chunk[W_mask].reshape(-1, 2, Q)

        return jet_chunk, event_chunk, top_targets_chunk, W_targets_chunk

    # -----------------------------
    # Fitting and transforming
    # -----------------------------
    def _fit_transformers(self, jet, top_targets, W_targets, interactions):
        N, P, F = jet.shape
        pt, eta, phi, mass, e = 0, 1, 2, 3, 4

        # Remove pid (assumed last) from targets
        top_targets = top_targets[..., 0:4]
        W_targets   = W_targets[..., 0:4]

        jet_flat        = jet.reshape(N * P, F)
        top_flat        = top_targets.reshape(N * 2, 4)
        W_flat          = W_targets.reshape(N * 2, 4)
        interactions_flat = interactions.reshape(N * P * P, -1)

        jet_variables = (jet_pt, jet_et, jet_phi, jet_mass, jet_e) = (
            jet_flat[:, pt], jet_flat[:, eta], jet_flat[:, phi], jet_flat[:, mass], jet_flat[:, e]
        )
        top_variables = (top_pt, top_et, top_phi, top_mass) = (
            top_flat[:, pt], top_flat[:, eta], top_flat[:, phi], top_flat[:, mass]
        )
        W_variables = (W_pt, W_et, W_phi, W_mass) = (
            W_flat[:, pt], W_flat[:, eta], W_flat[:, phi], W_flat[:, mass]
        )

        for variable, transformation in zip(jet_variables, self.jet_transformers):
            transformation.partial_fit(variable.reshape(-1, 1))

        for variable, transformation in zip(top_variables, self.target_transformers):
            transformation.partial_fit(variable.reshape(-1, 1))

        for variable, transformation in zip(W_variables, self.W_target_transformers):
            transformation.partial_fit(variable.reshape(-1, 1))

        self._interaction_transformers.partial_fit(interactions_flat)

    def _transform(self, jet, top_targets, W_targets, interactions):
        N, P, F = jet.shape
        jet_btag = jet[..., 5].reshape(N, P, 1)  # keep btag unscaled
        pt, eta, phi, mass, e = 0, 1, 2, 3, 4

        # Remove pid from targets
        top_targets = top_targets[..., 0:4]
        W_targets   = W_targets[..., 0:4]

        jet_flat        = jet.reshape(N * P, F)
        top_flat        = top_targets.reshape(N * 2, 4)
        W_flat          = W_targets.reshape(N * 2, 4)
        interactions_flat = interactions.reshape(N * P * P, -1)

        jet_variables = (jet_pt, jet_et, jet_phi, jet_mass, jet_e) = (
            jet_flat[:, pt], jet_flat[:, eta], jet_flat[:, phi], jet_flat[:, mass], jet_flat[:, e]
        )
        top_variables = (top_pt, top_et, top_phi, top_mass) = (
            top_flat[:, pt], top_flat[:, eta], top_flat[:, phi], top_flat[:, mass]
        )
        W_variables = (W_pt, W_et, W_phi, W_mass) = (
            W_flat[:, pt], W_flat[:, eta], W_flat[:, phi], W_flat[:, mass]
        )

        jet_transformed = []
        top_transformed = []
        W_transformed   = []

        # Transform jet variables
        for variable, transformation in zip(jet_variables, self.jet_transformers):
            jet_transformed.append(
                transformation.transform(variable.reshape(-1, 1)).reshape(N, P, -1)
            )

        # Transform top targets
        for variable, transformation in zip(top_variables, self.target_transformers):
            top_transformed.append(
                transformation.transform(variable.reshape(-1, 1)).reshape(N, 2, -1)
            )

        # Transform W targets
        for variable, transformation in zip(W_variables, self.W_target_transformers):
            W_transformed.append(
                transformation.transform(variable.reshape(-1, 1)).reshape(N, 2, -1)
            )

        jet = np.concatenate(jet_transformed, axis=-1)
        jet = np.concatenate((jet, jet_btag), axis=2)  # append btag

        top_target = np.concatenate(top_transformed, axis=-1)
        W_target   = np.concatenate(W_transformed, axis=-1)

        interactions = self._interaction_transformers.transform(
            interactions_flat
        ).reshape(N, P, P, -1)

        return jet, top_target, W_target, interactions

    # -----------------------------
    # Padding & saving
    # -----------------------------
    def _pad_and_src_mask(self, jet_chunk, interaction_chunk, pad_value=-99):
        self.pad_value = pad_value

        src_mask = np.all(np.isnan(jet_chunk), axis=-1)

        jet_chunk = np.nan_to_num(jet_chunk, nan=pad_value)
        interaction_chunk = np.nan_to_num(interaction_chunk, nan=-np.inf)

        return jet_chunk, src_mask, interaction_chunk

    def _create_datagroups(self, file, jet_shape, event_shape, top_shape, W_shape, interaction_shape):
        """
        jet_shape         = (B, N_jets, jet_features)
        event_shape       = (B, event_features)
        top_shape / W_shape = (B, 2, target_features)
        interaction_shape = (B, N_jets, N_jets, interaction_features)
        """
        _, N_jets, jet_features      = jet_shape
        _, event_features            = event_shape
        _, M_targets, target_features = top_shape
        _, _, _, interaction_features = interaction_shape

        file.create_dataset(
            "jet",
            shape=(0, N_jets, jet_features),
            maxshape=(None, N_jets, jet_features),
            compression="gzip",
            compression_opts=4,
        )
        file.create_dataset(
            "event",
            shape=(0, event_features),
            maxshape=(None, event_features),
            compression="gzip",
            compression_opts=4,
        )
        file.create_dataset(
            "targets",  # tops
            shape=(0, M_targets, target_features),
            maxshape=(None, M_targets, target_features),
            compression="gzip",
            compression_opts=4,
        )
        file.create_dataset(
            "W_targets",  # Ws
            shape=(0, M_targets, target_features),
            maxshape=(None, M_targets, target_features),
            compression="gzip",
            compression_opts=4,
        )
        file.create_dataset(
            "interactions",
            shape=(0, N_jets, N_jets, interaction_features),
            maxshape=(None, N_jets, N_jets, interaction_features),
            compression="gzip",
            compression_opts=4,
        )
        file.create_dataset(
            "src_mask",
            shape=(0, N_jets),
            maxshape=(None, N_jets),
            compression="gzip",
            compression_opts=4,
        )

    def _save_data_chunks(self, file, jet_chunk, interaction_chunk,
                          event_chunk, src_mask_chunk, targets_chunk, W_targets_chunk):

        jet_ds         = file["jet"]
        event_ds       = file["event"]
        target_ds      = file["targets"]
        W_target_ds    = file["W_targets"]
        src_ds         = file["src_mask"]
        interaction_ds = file["interactions"]

        cur_len = jet_ds.shape[0]
        n0, n1 = cur_len, cur_len + jet_chunk.shape[0]

        jet_ds.resize((n1,) + jet_ds.shape[1:])
        event_ds.resize((n1,) + event_ds.shape[1:])
        target_ds.resize((n1,) + target_ds.shape[1:])
        W_target_ds.resize((n1,) + W_target_ds.shape[1:])
        interaction_ds.resize((n1,) + interaction_ds.shape[1:])
        src_ds.resize((n1,) + src_ds.shape[1:])

        jet_ds[n0:n1]         = jet_chunk.astype("float32")
        event_ds[n0:n1]       = event_chunk.astype("float32")
        target_ds[n0:n1]      = targets_chunk.astype("float32")
        W_target_ds[n0:n1]    = W_targets_chunk.astype("float32")
        interaction_ds[n0:n1] = interaction_chunk.astype("float32")
        src_ds[n0:n1]         = src_mask_chunk.astype("float32")

if __name__ == "__main__":
    config = load_any_config("config/top_reconstruction_config.yaml")
    top_reconstruction_dataset = WBosonInteractionTopReconstructionDatasetFromH5(config)

""" 

if __name__ == "__main__":
    config = load_and_split_config(config_file_path)
    config = config.data_pipeline["data_preprocessing"]
    top_dataset = TopTensorDatasetFromH5py(config)       
     """