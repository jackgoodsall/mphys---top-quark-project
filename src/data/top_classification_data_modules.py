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

class TopMultiplicityClassifierDataset(Dataset):
    def __init__(self, particle_features, global_features, particle_mask, target_labels):
        # ensure numpy arrays
        self.particle_features = np.asarray(particle_features)
        self.global_features   = np.asarray(global_features)
        self.particle_mask     = np.asarray(particle_mask)
        self.target_labels     = np.asarray(target_labels)

    def __len__(self):
        return int(self.particle_features.shape[0])

    def __getitem__(self, idx):
        sample = {
            "particle_features": torch.from_numpy(self.particle_features[idx]).float(),
            "global_features":   torch.from_numpy(self.global_features[idx]).float(),
            "particle_mask":     torch.from_numpy(self.particle_mask[idx]).bool(),
        }
        target = torch.from_numpy(self.target_labels[idx]).float()
        return sample, target


class FourvsThreeTopDataModule(LightningDataModule):
    def __init__(self, data_prefix: str = "data/fourtopvsthree/processed_data/two_charge/fourtop3top_2_charge_events_functional_transformed_", batch_size: int = 512,
                 num_workers: int = 1, pin_memory: bool = True,
                 *args, **kwargs):
        super().__init__()
        self.data_prefix = data_prefix  # expects files like processed_data_train.h5, _val.h5, _test.h5
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

    def _load_split(self, name: str) -> TopMultiplicityClassifierDataset:
        path = Path(f"{self.data_prefix}{name}.h5")
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with h5py.File(path, "r") as f:
            part = f["particle_features"][()]   # load to memory
            glob = f["global_features"][()]
            pmask = f["particle_mask"][()]
            targ = f["targets"][()]
        ds = TopMultiplicityClassifierDataset(part, glob, pmask, targ)
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers > 0,
                          drop_last = True)

    def val_dataloader(self):
        assert self.val_dataset is not None, "val_dataset not set (did setup() run?)"
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        assert self.test_dataset is not None, "test_dataset not set (did setup() run?)"
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers > 0)

