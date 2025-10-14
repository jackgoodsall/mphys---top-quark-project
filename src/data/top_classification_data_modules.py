from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py

class TopMulitplicityClassifierDataSet(Dataset):
    ### Dictionary for first return type makes writing reproducable wrappers easier, 
    ### will change in future if it is slower
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


class FourvsThreeTopDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "../data/processed_data", batch_size = 16384):
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def _setup_one_set(self, name: str):
        file_name = self.data_dir + name 
        with h5py.File(file_name, "r") as f:
            global_data =  f["global_features"]
            particle_data = f["particle_features"]
            src_mask = f["particle_mask"]
            target_labels = f["targets"]
        return TopMulitplicityClassifierDataSet(particle_data
                , global_data, src_mask, target_labels)
        

    def setup(self, stage: str):
        if stage == "fit":
            train_dataset, val_dataset = self._setup_one_set("train"), self._setup_one_set("val")
        elif stage == "test":
            pass
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          self.batch_size)
    def test_dataloader(self):
        return 
    