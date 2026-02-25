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

# Object class constants
CLASS_NULL = 0
CLASS_TOP = 1
CLASS_W = 2


def _load_object_type(f: h5py.File, mask_key: str, kin_key: str):
    """
    Load masks and kinematics for one object type from an open HDF5 file.
    Returns (None, None) if the key is absent or has shape[1] == 0.
    """
    if mask_key not in f:
        return None, None
    masks = f[mask_key][()]
    if masks.shape[1] == 0:
        return None, None
    kins = f[kin_key][()]
    return masks, kins


def merge_object_types(tops_masks, tops_kins, ws_masks, ws_kins,
                       valid_tops=None, valid_Ws=None):
    """
    Concatenate tops and Ws (tops first) into unified arrays,
    stamp per-object integer class labels, and build a validity mask.

    Args:
        valid_tops: [N, T_top] bool/uint8 — which top objects are reconstructable.
                    Defaults to all True if None.
        valid_Ws:   [N, T_w]  bool/uint8 — which W objects are reconstructable.
                    Defaults to all True if None.

    Returns:
        masks:        [N, n_obj, P]
        kins:         [N, n_obj, D]
        classes:      [N, n_obj] int64, values CLASS_TOP or CLASS_W
        object_valid: [N, n_obj] bool — True for reconstructable objects
    """
    all_masks = []
    all_kins = []
    all_classes = []
    all_valid = []

    if tops_masks is not None:
        N, T_top = tops_masks.shape[0], tops_masks.shape[1]
        all_masks.append(tops_masks)
        all_kins.append(tops_kins)
        all_classes.append(np.full((N, T_top), CLASS_TOP, dtype=np.int64))
        vt = valid_tops if valid_tops is not None else np.ones((N, T_top), dtype=bool)
        all_valid.append(vt.astype(bool))

    if ws_masks is not None:
        N, T_w = ws_masks.shape[0], ws_masks.shape[1]
        all_masks.append(ws_masks)
        all_kins.append(ws_kins)
        all_classes.append(np.full((N, T_w), CLASS_W, dtype=np.int64))
        vw = valid_Ws if valid_Ws is not None else np.ones((N, T_w), dtype=bool)
        all_valid.append(vw.astype(bool))

    masks = np.concatenate(all_masks, axis=1)
    kins = np.concatenate(all_kins, axis=1)
    classes = np.concatenate(all_classes, axis=1)
    object_valid = np.concatenate(all_valid, axis=1)  # [N, n_obj]
    return masks, kins, classes, object_valid


def masked_former_collate_fn(batch):
    """
    Custom collate function for MaskedFormer with variable-length targets.

    Stacks sample tensors normally (jet, src_mask, interactions have fixed shape).
    Pre-pads and stacks targets on CPU so _collate_targets() is skipped on GPU,
    eliminating ~12,000 CUDA kernel launches per batch.

    Returns:
        batched_samples: Dict[str, Tensor]
        batched_targets: Dict[str, Tensor] including 'target_valid_mask' [B, T_max]
    """
    samples = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack sample dicts (all have fixed shape)
    batched_samples = {}
    for key in samples[0]:
        batched_samples[key] = torch.stack([s[key] for s in samples])

    B = len(targets)
    T_per_event = [t['jet_mask_true'].shape[0] for t in targets]
    T_max = max(T_per_event) if T_per_event else 0

    # Edge case: every event has zero reconstructable objects
    if T_max == 0:
        P = targets[0]['jet_valid_mask'].shape[0]
        batched_targets = {
            'jet_mask_true': torch.zeros(B, 0, P),
            'jet_valid_mask': torch.stack([t['jet_valid_mask'] for t in targets]),
            'target_valid_mask': torch.zeros(B, 0, dtype=torch.bool),
        }
        if 'target_kinematics' in targets[0]:
            D = targets[0]['target_kinematics'].shape[-1]
            tk = torch.zeros(B, 0, D)
            batched_targets['target_kinematics'] = tk
            batched_targets['kinematics'] = tk
        if 'classes' in targets[0]:
            batched_targets['classes'] = torch.zeros(B, 0, dtype=torch.long)
        return batched_samples, batched_targets

    # Pre-allocate and fill jet_mask_true [B, T_max, P]
    P = targets[0]['jet_valid_mask'].shape[0]
    jmt = torch.zeros(B, T_max, P)
    for i, t in enumerate(targets):
        T_i = T_per_event[i]
        if T_i > 0:
            jmt[i, :T_i] = t['jet_mask_true']

    # Build target_valid_mask [B, T_max]
    tvm = torch.zeros(B, T_max, dtype=torch.bool)
    for i, T_i in enumerate(T_per_event):
        tvm[i, :T_i] = True

    batched_targets = {
        'jet_mask_true': jmt,
        'jet_valid_mask': torch.stack([t['jet_valid_mask'] for t in targets]),
        'target_valid_mask': tvm,
    }

    # Pre-allocate and fill target_kinematics [B, T_max, D]
    if 'target_kinematics' in targets[0]:
        D = targets[0]['target_kinematics'].shape[-1]
        tk = torch.zeros(B, T_max, D)
        for i, t in enumerate(targets):
            T_i = T_per_event[i]
            if T_i > 0:
                tk[i, :T_i] = t['target_kinematics']
        batched_targets['target_kinematics'] = tk
        batched_targets['kinematics'] = tk

    # Pre-allocate and fill classes [B, T_max]
    if 'classes' in targets[0]:
        cls = torch.zeros(B, T_max, dtype=torch.long)
        for i, t in enumerate(targets):
            T_i = T_per_event[i]
            if T_i > 0:
                cls[i, :T_i] = t['classes']
        batched_targets['classes'] = cls

    return batched_samples, batched_targets


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
    def __init__(self, jet, interactions, src_mask, targets, target_kinematics,
                 target_mass=None, mass_with_kinematics=False, classes=None,
                 object_valid=None):
        # Flag for if to include mass with the kinematics
        self.mass_with_kinematics = mass_with_kinematics
        # ensure numpy arrays
        self.jet = np.asarray(jet)
        self.src_mask = np.asarray(src_mask)
        self.targets = np.asarray(targets)
        self.interactions = np.asarray(interactions)
        self.target_kinematics = np.asarray(target_kinematics)
        self.inv_mass = np.asarray(target_mass) if target_mass is not None else None
        self.classes = np.asarray(classes) if classes is not None else None
        # Per-event object validity mask [N, n_obj] bool.
        # If set, __getitem__ filters objects so T_i varies per event.
        # If None, all objects are returned (backward-compatible behaviour).
        self.object_valid = np.asarray(object_valid, dtype=bool) if object_valid is not None else None

        if self.mass_with_kinematics:
            if target_mass is not None:
                self.inv_mass = self.inv_mass.reshape(-1, 1, 1)
                self.target_kinematics = np.concatenate(
                    (self.target_kinematics, self.inv_mass), axis=2
                )

    def __len__(self):
        return int(self.jet.shape[0])

    def __getitem__(self, idx):
        sample = {
            "jet": torch.from_numpy(self.jet[idx]).float(),
            "src_mask": torch.from_numpy(self.src_mask[idx]).bool(),
            "interactions": torch.from_numpy(self.interactions[idx])
        }

        # Apply per-event validity filter so T_i varies when object_valid is set
        if self.object_valid is not None:
            valid = self.object_valid[idx]  # [n_obj] bool
            jet_mask = self.targets[idx][valid]           # [T_i, P]
            kin = self.target_kinematics[idx][valid]      # [T_i, D]
            cls = self.classes[idx][valid] if self.classes is not None else None
        else:
            jet_mask = self.targets[idx]
            kin = self.target_kinematics[idx]
            cls = self.classes[idx] if self.classes is not None else None

        target = {
            "jet_mask_true": torch.from_numpy(np.asarray(jet_mask)).float(),
            "jet_valid_mask": torch.from_numpy(
                np.asarray(self.src_mask[idx])
            ).bool(),
            "target_kinematics": torch.from_numpy(np.asarray(kin)).float(),
        }
        if not self.mass_with_kinematics and self.inv_mass is not None:
            target["inv_mass"] = torch.from_numpy(
                np.asarray(self.inv_mass[idx])
            ).float()
        if cls is not None:
            target["classes"] = torch.from_numpy(np.asarray(cls)).long()

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


class MaskedFormer2(LightningDataModule):
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
            targ = f["masks_tops"][()]
            targ_kinematics = f["kinematics_tops"][()]
            interactions = f["interactions"][()]
            #target_mass = f["target_mass"][()]
            target_mass = None

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






class MaskedFormerTopsWsDataModule(LightningDataModule):
    """
    DataModule that loads both tops and Ws from HDF5, merges them
    into unified targets with class labels, and uses a custom collate
    function that returns targets as List[Dict].
    """

    def __init__(self, config):
        super().__init__()

        self.config = config["data_modules"]

        self.train_config = self.config["train"]
        self.test_config = self.config["test"]
        self.val_config = self.config["val"]

        self.input_path = self.config["input_path"]
        self.input_prefix = self.config["input_prefix"]
        self.data_prefix = Path(self.input_path, self.input_prefix)

        # HDF5 key names (configurable with defaults)
        self.tops_mask_key = self.config.get("tops_mask_key", "masks_tops")
        self.tops_kin_key = self.config.get("tops_kin_key", "kinematics_tops")
        self.ws_mask_key = self.config.get("ws_mask_key", "masks_Ws")
        self.ws_kin_key = self.config.get("ws_kin_key", "kinematics_Ws")

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def _load_split(self, name: str) -> MaskedFormerDataSet:
        path = Path(f"{self.data_prefix}{name}.h5")
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

        with h5py.File(path, "r") as f:
            jet = f["jet"][()]
            src_mask = f["src_mask"][()]
            interactions = f["interactions"][()]

            tops_masks, tops_kins = _load_object_type(
                f, self.tops_mask_key, self.tops_kin_key
            )
            ws_masks, ws_kins = _load_object_type(
                f, self.ws_mask_key, self.ws_kin_key
            )

            # Load validity arrays if present (produced by dataset_prepper_3.py
            # with min_objects < 4). Falls back to all-True (old behaviour) if absent.
            valid_tops = f["valid_tops"][()] if "valid_tops" in f else None
            valid_Ws = f["valid_Ws"][()] if "valid_Ws" in f else None

        masks, kins, classes, object_valid = merge_object_types(
            tops_masks, tops_kins, ws_masks, ws_kins, valid_tops, valid_Ws
        )

        # Only pass object_valid when it's not all-True (i.e. when some objects
        # are actually invalid) to avoid unnecessary filtering overhead.
        has_partial = (object_valid is not None and not object_valid.all())

        ds = MaskedFormerDataSet(
            jet=jet,
            interactions=interactions,
            src_mask=src_mask,
            targets=masks,
            target_kinematics=kins,
            target_mass=None,
            mass_with_kinematics=False,
            classes=classes,
            object_valid=object_valid if has_partial else None,
        )
        return ds

    def setup(self, stage):
        if stage in (None, "fit", "validate"):
            self.train_dataset = self._load_split("train")
            self.val_dataset = self._load_split("val")
            print(f"[DM TopsWs] train len={len(self.train_dataset)}  val len={len(self.val_dataset)}")
        if stage in (None, "test"):
            self.test_dataset = self._load_split("test")
            print(f"[DM TopsWs] test  len={len(self.test_dataset)}")

    def train_dataloader(self):
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config["batch_size"],
            shuffle=self.train_config["shuffle"],
            num_workers=self.train_config["num_workers"],
            pin_memory=self.train_config["pin_memory"],
            drop_last=True,
            collate_fn=masked_former_collate_fn,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_config["batch_size"],
            shuffle=self.val_config["shuffle"],
            num_workers=self.val_config["num_workers"],
            pin_memory=self.val_config["pin_memory"],
            collate_fn=masked_former_collate_fn,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_config["batch_size"],
            shuffle=self.test_config["shuffle"],
            num_workers=self.test_config["num_workers"],
            pin_memory=self.test_config["pin_memory"],
            collate_fn=masked_former_collate_fn,
        )


if __name__ == "__main__":
    config = load_any_config("config/top_reconstruction_config.yaml")
    top = TopReconstruction(config)._load_split("test")
    print(top)