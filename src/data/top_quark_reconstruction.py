from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py

from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import h5py
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.utils import load_any_config

class TopReconstructionDataset(Dataset):
    def __init__(self, jet, src_mask, targets):
        # ensure numpy arrays
        self.jet = np.asarray(jet)
        self.src_mask   = np.asarray(src_mask)
        self.targets     = np.asarray(targets)

    def __len__(self):
        return int(self.jet.shape[0])

    def __getitem__(self, idx):
        sample = {
            "jet": torch.from_numpy(self.jet[idx]).float(),
            "src_mask":     torch.from_numpy(self.src_mask[idx]).bool(),
        }
        target = torch.from_numpy(self.targets[idx]).float()
        return sample, target


class TopReconstruction(LightningDataModule):
    def __init__(self, config):

        super().__init__()
        
        self.config  = config["data_modules"]

        self.train_config = self.config["train"]
        self.test_config = self.config["test"]
        self.val_config = self.config["val"]

        self.input_path = self.config["input_path"]
        self.input_prefix = self.config["input_prefix"]

        self.data_prefix = Path(self.input_path, self.input_prefix)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def _load_split(self, name: str) -> TopReconstructionDataset:
        path = Path(f"{self.data_prefix}{name}.h5")
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with h5py.File(path, "r") as f:
            jet = f["jet"][()]   # load to memory
            src_mask = f["src_mask"][()]
            targ = f["targets"][()]
        ds = TopReconstructionDataset(jet,  src_mask, targ)
        return ds

    def setup(self, stage):
        # Lightning may call with stage=None (setup everything) and/or "fit"/"validate"/"test"
        if stage in (None, "fit", "validate"):
            self.train_dataset = self._load_split("train")
            self.val_dataset   = self._load_split("val")
            print(f"[DM] train len={len(self.train_dataset)}  val len={len(self.val_dataset)}")
        if stage in (None, "test"):
            self.test_dataset  = self._load_split("test")
            print(f"[DM] test  len={len(self.test_dataset)}")

    def train_dataloader(self):
        assert self.train_dataset is not None, "train_dataset not set (did setup() run?)"
        return DataLoader(self.train_dataset, batch_size=self.train_config["batch_size"], shuffle=self.train_config["shuffle"],
                          num_workers=self.train_config["num_workers"], pin_memory=self.train_config["pin_memory"],
                          drop_last = True)

    def val_dataloader(self):
        assert self.val_dataset is not None, "val_dataset not set (did setup() run?)"
        return DataLoader(self.val_dataset, batch_size=self.val_config["batch_size"], shuffle=self.val_config["shuffle"],
                          num_workers=self.val_config["num_workers"], pin_memory=self.val_config["pin_memory"])

    def test_dataloader(self):
        assert self.test_dataset is not None, "test_dataset not set (did setup() run?)"
        return DataLoader(self.test_dataset, batch_size=self.test_config["batch_size"], shuffle=self.test_config["shuffle"],
                          num_workers=self.test_config["num_workers"], pin_memory=self.test_config["pin_memory"])



class ParTDataset(Dataset):
    def __init__(self, jet, interactions,src_mask, targets):
        # ensure numpy arrays
        self.jet = np.asarray(jet)
        self.src_mask   = np.asarray(src_mask)
        self.targets     = np.asarray(targets)
        self.interactions = np.asarray(interactions)
    def __len__(self):
        return int(self.jet.shape[0])

    def __getitem__(self, idx):
        sample = {
            "jet": torch.from_numpy(self.jet[idx]).float(),
            "src_mask":     torch.from_numpy(self.src_mask[idx]).bool(),
            "interactions": torch.from_numpy(self.interactions[idx])
        }
        target = torch.from_numpy(self.targets[idx]).float()
        return sample, target


class TopReconstructionInteractions(LightningDataModule):
    def __init__(self, config):

        super().__init__()
        
        self.config  = config["data_modules"]

        self.train_config = self.config["train"]
        self.test_config = self.config["test"]
        self.val_config = self.config["val"]

        self.input_path = self.config["input_path"]
        self.input_prefix = self.config["input_prefix"]

        self.data_prefix = Path(self.input_path, self.input_prefix)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def _load_split(self, name: str) -> ParTDataset:
        path = Path(f"{self.data_prefix}{name}.h5")
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with h5py.File(path, "r") as f:
            jet = f["jet"][()]   # load to memory
            src_mask = f["src_mask"][()]
            targ = f["targets"][()]
            interactions = f["interactions"][()]
        ds = ParTDataset(jet, interactions ,src_mask, targ)
        return ds

    def setup(self, stage):
        # Lightning may call with stage=None (setup everything) and/or "fit"/"validate"/"test"
        if stage in (None, "fit", "validate"):
            self.train_dataset = self._load_split("train")
            self.val_dataset   = self._load_split("val")
            print(f"[DM] train len={len(self.train_dataset)}  val len={len(self.val_dataset)}")
        if stage in (None, "test"):
            self.test_dataset  = self._load_split("test")
            print(f"[DM] test  len={len(self.test_dataset)}")

    def train_dataloader(self):
        assert self.train_dataset is not None, "train_dataset not set (did setup() run?)"
        return DataLoader(self.train_dataset, batch_size=self.train_config["batch_size"], shuffle=self.train_config["shuffle"],
                          num_workers=self.train_config["num_workers"], pin_memory=self.train_config["pin_memory"],
                          drop_last = True)

    def val_dataloader(self):
        assert self.val_dataset is not None, "val_dataset not set (did setup() run?)"
        return DataLoader(self.val_dataset, batch_size=self.val_config["batch_size"], shuffle=self.val_config["shuffle"],
                          num_workers=self.val_config["num_workers"], pin_memory=self.val_config["pin_memory"])

    def test_dataloader(self):
        assert self.test_dataset is not None, "test_dataset not set (did setup() run?)"
        return DataLoader(self.test_dataset, batch_size=self.test_config["batch_size"], shuffle=self.test_config["shuffle"],
                          num_workers=self.test_config["num_workers"], pin_memory=self.test_config["pin_memory"])



class ParTWDataset(Dataset):
    def __init__(self, jet, interactions,src_mask, targets):
        # ensure numpy arrays
        self.jet = np.asarray(jet)
        self.src_mask   = np.asarray(src_mask)
        self.targets     = np.asarray(targets)
        self.interactions = np.asarray(interactions)
 
    def __len__(self):
        return int(self.jet.shape[0])

    def __getitem__(self, idx):
        sample = {
            "jet": torch.from_numpy(self.jet[idx]).float(),
            "src_mask":     torch.from_numpy(self.src_mask[idx]).bool(),
            "interactions": torch.from_numpy(self.interactions[idx])
        }
        target = {
            "jet_mask_true" : self.targets[idx],
            "jet_valid_mask" : self.src_mask[idx]
        }
        return sample, target


class TopandWReconstuctionDataModule(LightningDataModule):
    def __init__(self, config):

        super().__init__()
        
        self.config  = config["data_modules"]

        self.train_config = self.config["train"]
        self.test_config = self.config["test"]
        self.val_config = self.config["val"]

        self.input_path = self.config["input_path"]
        self.input_prefix = self.config["input_prefix"]

        self.data_prefix = Path(self.input_path, self.input_prefix)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def _load_split(self, name: str) -> ParTDataset:
        path = Path(f"{self.data_prefix}{name}.h5")
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with h5py.File(path, "r") as f:
            jet = f["jet"][()]   # load to memory
            src_mask = f["src_mask"][()]
            targ = f["targets"][()]
            interactions = f["interactions"][()]

        ds = ParTWDataset(jet, interactions ,src_mask, targ)
        return ds

    def setup(self, stage):
        # Lightning may call with stage=None (setup everything) and/or "fit"/"validate"/"test"
        if stage in (None, "fit", "validate"):
            self.train_dataset = self._load_split("train")
            self.val_dataset   = self._load_split("val")
            print(f"[DM] train len={len(self.train_dataset)}  val len={len(self.val_dataset)}")
        if stage in (None, "test"):
            self.test_dataset  = self._load_split("test")
            print(f"[DM] test  len={len(self.test_dataset)}")

    def train_dataloader(self):
        assert self.train_dataset is not None, "train_dataset not set (did setup() run?)"
        return DataLoader(self.train_dataset, batch_size=self.train_config["batch_size"], shuffle=self.train_config["shuffle"],
                          num_workers=self.train_config["num_workers"], pin_memory=self.train_config["pin_memory"],
                          drop_last = True)

    def val_dataloader(self):
        assert self.val_dataset is not None, "val_dataset not set (did setup() run?)"
        return DataLoader(self.val_dataset, batch_size=self.val_config["batch_size"], shuffle=self.val_config["shuffle"],
                          num_workers=self.val_config["num_workers"], pin_memory=self.val_config["pin_memory"])

    def test_dataloader(self):
        assert self.test_dataset is not None, "test_dataset not set (did setup() run?)"
        return DataLoader(self.test_dataset, batch_size=self.test_config["batch_size"], shuffle=self.test_config["shuffle"],
                          num_workers=self.test_config["num_workers"], pin_memory=self.test_config["pin_memory"])




class MaskedFormerDataSet(Dataset):
    def __init__(self, jet, interactions,src_mask, targets, target_kinematics, target_mass, mass_with_kinematics = False):
        # Flag for if to include mass with the kinematics
        self.mass_with_kinematics = mass_with_kinematics
        # ensure numpy arrays
        self.jet = np.asarray(jet)
        self.src_mask   = np.asarray(src_mask)
        self.targets     = np.asarray(targets)
        self.interactions = np.asarray(interactions)
        self.target_kinematics = np.asarray(target_kinematics)
        self.inv_mass = np.asarray(target_mass)

        if self.mass_with_kinematics:
            self.inv_mass = self.inv_mass.reshape(-1, 1 , 1)
            self.target_kinematics = np.concatenate((self.target_kinematics, self.inv_mass), axis = 2)
 
    def __len__(self):
        return int(self.jet.shape[0])

    def __getitem__(self, idx):
        sample = {
            "jet": torch.from_numpy(self.jet[idx]).float(),
            "src_mask":     torch.from_numpy(self.src_mask[idx]).bool(),
            "interactions": torch.from_numpy(self.interactions[idx])
        }
        if not self.mass_with_kinematics:
            target = {
                "jet_mask_true" : self.targets[idx],
                "jet_valid_mask" : self.src_mask[idx],
                "target_kinematics" : self.target_kinematics[idx],
                "inv_mass" : self.inv_mass[idx]
            }
        else: 
            target = {
                "jet_mask_true" : self.targets[idx],
                "jet_valid_mask" : self.src_mask[idx],
                "target_kinematics" : self.target_kinematics[idx],
            }

        return sample, target


class MaskedFormerDataModule(LightningDataModule):
    def __init__(self, config):

        super().__init__()
        
        self.config  = config["data_modules"]

        self.train_config = self.config["train"]
        self.test_config = self.config["test"]
        self.val_config = self.config["val"]

        self.input_path = self.config["input_path"]
        self.input_prefix = self.config["input_prefix"]

        self.mass_with_kinematics = self.config["mass_with_kinematics"]

        self.data_prefix = Path(self.input_path, self.input_prefix)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def _load_split(self, name: str) -> ParTDataset:
        path = Path(f"{self.data_prefix}{name}.h5")
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with h5py.File(path, "r") as f:
            jet = f["jet"][()]   # load to memory
            src_mask = f["src_mask"][()]
            targ = f["targets"][()]
            targ_kinematics = f["target_kinematics"][()]
            interactions = f["interactions"][()]
            target_mass = f["target_mass"][()]


        ds = MaskedFormerDataSet(jet, interactions ,src_mask, targ, targ_kinematics, target_mass, self.mass_with_kinematics)
        return ds

    def setup(self, stage):
        # Lightning may call with stage=None (setup everything) and/or "fit"/"validate"/"test"
        if stage in (None, "fit", "validate"):
            self.train_dataset = self._load_split("train")
            self.val_dataset   = self._load_split("val")
            print(f"[DM] train len={len(self.train_dataset)}  val len={len(self.val_dataset)}")
        if stage in (None, "test"):
            self.test_dataset  = self._load_split("test")
            print(f"[DM] test  len={len(self.test_dataset)}")

    def train_dataloader(self):
        assert self.train_dataset is not None, "train_dataset not set (did setup() run?)"
        return DataLoader(self.train_dataset, batch_size=self.train_config["batch_size"], shuffle=self.train_config["shuffle"],
                          num_workers=self.train_config["num_workers"], pin_memory=self.train_config["pin_memory"],
                          drop_last = True)

    def val_dataloader(self):
        assert self.val_dataset is not None, "val_dataset not set (did setup() run?)"
        return DataLoader(self.val_dataset, batch_size=self.val_config["batch_size"], shuffle=self.val_config["shuffle"],
                          num_workers=self.val_config["num_workers"], pin_memory=self.val_config["pin_memory"])

    def test_dataloader(self):
        assert self.test_dataset is not None, "test_dataset not set (did setup() run?)"
        return DataLoader(self.test_dataset, batch_size=self.test_config["batch_size"], shuffle=self.test_config["shuffle"],
                          num_workers=self.test_config["num_workers"], pin_memory=self.test_config["pin_memory"])








if __name__ == "__main__":
    config = load_any_config("config/top_reconstruction_config.yaml")
    top = TopReconstruction(config)._load_split("test")
    print(top)