from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
import torch
import numpy as np

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


class MaskedFormerDataSet(Dataset):
    """In-memory dataset. Used by analysis scripts and as a fallback."""

    def __init__(self, jet, interactions, src_mask, targets, target_kinematics,
                 classes=None, object_valid=None):
        self.jet = np.asarray(jet)
        self.src_mask = np.asarray(src_mask)
        self.targets = np.asarray(targets)
        self.interactions = np.asarray(interactions) if interactions is not None else None
        self._zero_interactions = None  # lazily created single zero row
        self.target_kinematics = np.asarray(target_kinematics)
        self.classes = np.asarray(classes) if classes is not None else None
        self.object_valid = np.asarray(object_valid, dtype=bool) if object_valid is not None else None

    def __len__(self):
        return int(self.jet.shape[0])

    def __getitem__(self, idx):
        if self.interactions is not None:
            interactions = self.interactions[idx]
        else:
            if self._zero_interactions is None:
                P = self.jet.shape[1]
                self._zero_interactions = np.zeros((P, P, 4), dtype=np.float32)
            interactions = self._zero_interactions

        sample = {
            "jet": torch.from_numpy(self.jet[idx]).float(),
            "src_mask": torch.from_numpy(self.src_mask[idx]).bool(),
            "interactions": torch.from_numpy(interactions).float(),
        }

        if self.object_valid is not None:
            valid = self.object_valid[idx]
            jet_mask = self.targets[idx][valid]
            kin = self.target_kinematics[idx][valid]
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
        if cls is not None:
            target["classes"] = torch.from_numpy(np.asarray(cls)).long()

        return sample, target


class LazyHDF5Dataset(Dataset):
    """
    Lazy-loading dataset that reads from HDF5 on the fly.

    Only small metadata (length, validity, classes) is held in RAM.
    The large arrays (jet, interactions, masks, kinematics) are read
    per-event from disk via h5py indexing.

    Each DataLoader worker opens its own file handle on first access
    (h5py files cannot be shared across forked processes).
    """

    def __init__(self, h5_path, tops_mask_key, tops_kin_key,
                 ws_mask_key, ws_kin_key):
        self.h5_path = str(h5_path)

        # Read only small arrays + metadata into RAM
        with h5py.File(self.h5_path, "r") as f:
            self._length = f["jet"].shape[0]
            self._has_interactions = "interactions" in f

            # HDF5 dataset keys for tops/Ws
            self._tops_mask_key = tops_mask_key if tops_mask_key in f else None
            self._tops_kin_key = tops_kin_key if tops_kin_key in f else None
            self._ws_mask_key = ws_mask_key if ws_mask_key in f else None
            self._ws_kin_key = ws_kin_key if ws_kin_key in f else None

            # Load small validity arrays into RAM (~38 MB each for 19M events)
            valid_tops = f["valid_tops"][()] if "valid_tops" in f else None
            valid_Ws = f["valid_Ws"][()] if "valid_Ws" in f else None

        # Pre-compute per-event classes and object_valid (small arrays).
        # These stay in RAM; everything else is read lazily.
        N = self._length
        classes_parts = []
        valid_parts = []

        if self._tops_mask_key is not None:
            T_top = 2  # always 2 tops
            classes_parts.append(np.full((N, T_top), CLASS_TOP, dtype=np.int64))
            vt = valid_tops if valid_tops is not None else np.ones((N, T_top), dtype=bool)
            valid_parts.append(vt.astype(bool))

        if self._ws_mask_key is not None:
            T_w = 2  # always 2 Ws
            classes_parts.append(np.full((N, T_w), CLASS_W, dtype=np.int64))
            vw = valid_Ws if valid_Ws is not None else np.ones((N, T_w), dtype=bool)
            valid_parts.append(vw.astype(bool))

        self.classes = np.concatenate(classes_parts, axis=1)        # [N, n_obj]
        self.object_valid = np.concatenate(valid_parts, axis=1)     # [N, n_obj]
        self._has_partial = not self.object_valid.all()

        # File handle opened lazily per worker process
        self._file = None

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self._length

    def __del__(self):
        if self._file is not None:
            self._file.close()

    def __getitem__(self, idx):
        self._ensure_open()

        jet = self._file["jet"][idx]
        src_mask = self._file["src_mask"][idx]

        if self._has_interactions:
            interactions = self._file["interactions"][idx]
        else:
            P = jet.shape[0]
            interactions = np.zeros((P, P, 4), dtype=np.float32)

        sample = {
            "jet": torch.from_numpy(jet).float(),
            "src_mask": torch.from_numpy(src_mask).bool(),
            "interactions": torch.from_numpy(interactions).float(),
        }

        # Read and concatenate tops + Ws masks/kinematics for this event
        masks_parts = []
        kins_parts = []
        if self._tops_mask_key is not None:
            masks_parts.append(self._file[self._tops_mask_key][idx])
            kins_parts.append(self._file[self._tops_kin_key][idx])
        if self._ws_mask_key is not None:
            masks_parts.append(self._file[self._ws_mask_key][idx])
            kins_parts.append(self._file[self._ws_kin_key][idx])

        masks = np.concatenate(masks_parts, axis=0)   # [n_obj, P]
        kins = np.concatenate(kins_parts, axis=0)      # [n_obj, D]
        cls = self.classes[idx]                         # [n_obj]

        # Apply validity filter
        if self._has_partial:
            valid = self.object_valid[idx]
            masks = masks[valid]
            kins = kins[valid]
            cls = cls[valid]

        target = {
            "jet_mask_true": torch.from_numpy(masks).float(),
            "jet_valid_mask": torch.from_numpy(src_mask).bool(),
            "target_kinematics": torch.from_numpy(kins).float(),
            "classes": torch.from_numpy(cls).long(),
        }

        return sample, target


class MemmapDataset(Dataset):
    """
    Memory-mapped dataset backed by .npy files converted once from HDF5.

    Per-sample access (arr[idx]) reads only the bytes for that row via the OS
    page cache — zero upfront RAM cost, near in-memory speed after warm-up.

    To create the .npy files from an existing .h5 file, call
    MemmapDataset.prepare(h5_path, npy_dir) once before training.
    """

    @staticmethod
    def npy_dir(h5_path: Path) -> Path:
        """Canonical directory for the .npy files derived from h5_path."""
        return h5_path.parent / (h5_path.stem + "_memmap")

    @staticmethod
    def prepare(h5_path: Path, tops_mask_key, tops_kin_key,
                ws_mask_key, ws_kin_key, load_interactions: bool = True):
        """
        Convert all arrays in h5_path to .npy files in a sibling directory.
        Skips keys whose .npy file already exists.
        """
        npy_dir = MemmapDataset.npy_dir(h5_path)
        npy_dir.mkdir(exist_ok=True)
        keys_to_save = ["jet", "src_mask"]
        if load_interactions:
            keys_to_save.append("interactions")
        for k in [tops_mask_key, tops_kin_key, ws_mask_key, ws_kin_key,
                  "valid_tops", "valid_Ws"]:
            if k:
                keys_to_save.append(k)

        with h5py.File(h5_path, "r") as f:
            for key in keys_to_save:
                if key not in f:
                    continue
                out = npy_dir / f"{key}.npy"
                if out.exists():
                    continue
                print(f"[memmap] converting {key} → {out} ...", flush=True)
                np.save(str(out), f[key][()])
        return npy_dir

    def __init__(self, npy_dir: Path, tops_mask_key, tops_kin_key,
                 ws_mask_key, ws_kin_key, load_interactions: bool = True):
        self.npy_dir = npy_dir
        self.load_interactions = load_interactions

        def _mmap(key):
            p = npy_dir / f"{key}.npy"
            return np.load(str(p), mmap_mode="r") if p.exists() else None

        self._jet = _mmap("jet")
        self._src_mask = _mmap("src_mask")
        self._interactions = _mmap("interactions") if load_interactions else None

        self._tops_masks = _mmap(tops_mask_key) if tops_mask_key else None
        self._tops_kins = _mmap(tops_kin_key) if tops_kin_key else None
        self._ws_masks = _mmap(ws_mask_key) if ws_mask_key else None
        self._ws_kins = _mmap(ws_kin_key) if ws_kin_key else None

        valid_tops = _mmap("valid_tops")
        valid_Ws = _mmap("valid_Ws")

        N = len(self._jet)
        classes_parts, valid_parts = [], []
        if self._tops_masks is not None:
            T_top = self._tops_masks.shape[1]
            classes_parts.append(np.full((N, T_top), CLASS_TOP, dtype=np.int64))
            vt = valid_tops if valid_tops is not None else np.ones((N, T_top), dtype=bool)
            valid_parts.append(vt.astype(bool))
        if self._ws_masks is not None:
            T_w = self._ws_masks.shape[1]
            classes_parts.append(np.full((N, T_w), CLASS_W, dtype=np.int64))
            vw = valid_Ws if valid_Ws is not None else np.ones((N, T_w), dtype=bool)
            valid_parts.append(vw.astype(bool))

        self.classes = np.concatenate(classes_parts, axis=1)
        self.object_valid = np.concatenate(valid_parts, axis=1)
        self._has_partial = not self.object_valid.all()
        self._zero_interactions = None

    def __len__(self):
        return len(self._jet)

    def __getitem__(self, idx):
        jet = np.array(self._jet[idx])
        src_mask = np.array(self._src_mask[idx])

        if self._interactions is not None:
            interactions = np.array(self._interactions[idx])
        else:
            if self._zero_interactions is None:
                P = jet.shape[0]
                self._zero_interactions = np.zeros((P, P, 4), dtype=np.float32)
            interactions = self._zero_interactions

        sample = {
            "jet": torch.from_numpy(jet).float(),
            "src_mask": torch.from_numpy(src_mask).bool(),
            "interactions": torch.from_numpy(interactions).float(),
        }

        masks_parts, kins_parts = [], []
        if self._tops_masks is not None:
            masks_parts.append(np.array(self._tops_masks[idx]))
            kins_parts.append(np.array(self._tops_kins[idx]))
        if self._ws_masks is not None:
            masks_parts.append(np.array(self._ws_masks[idx]))
            kins_parts.append(np.array(self._ws_kins[idx]))

        masks = np.concatenate(masks_parts, axis=0)
        kins = np.concatenate(kins_parts, axis=0)
        cls = self.classes[idx]

        if self._has_partial:
            valid = self.object_valid[idx]
            masks, kins, cls = masks[valid], kins[valid], cls[valid]

        target = {
            "jet_mask_true": torch.from_numpy(masks).float(),
            "jet_valid_mask": torch.from_numpy(src_mask).bool(),
            "target_kinematics": torch.from_numpy(kins).float(),
            "classes": torch.from_numpy(cls).long(),
        }
        return sample, target


class MaskedFormerTopsWsDataModule(LightningDataModule):
    """
    DataModule that loads both tops and Ws from HDF5, merges them
    into unified targets with class labels, and uses a custom collate
    function that returns targets as List[Dict].

    Uses lazy HDF5 loading by default to avoid OOM on large datasets.
    Set data_modules.lazy: false in config to force in-memory loading.
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

        self.lazy = self.config.get("lazy", False)   # False | True | "memmap"
        self.load_interactions = self.config.get("load_interactions", True)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def _load_split(self, name: str) -> Dataset:
        path = Path(f"{self.data_prefix}{name}.h5")
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

        if self.lazy == "memmap":
            return self._load_split_memmap(path)
        if self.lazy:
            return self._load_split_lazy(path)
        return self._load_split_eager(path)

    def _load_split_lazy(self, path: Path) -> LazyHDF5Dataset:
        ds = LazyHDF5Dataset(
            h5_path=path,
            tops_mask_key=self.tops_mask_key,
            tops_kin_key=self.tops_kin_key,
            ws_mask_key=self.ws_mask_key,
            ws_kin_key=self.ws_kin_key,
        )
        return ds

    def _load_split_memmap(self, path: Path) -> MemmapDataset:
        npy_dir = MemmapDataset.prepare(
            path,
            tops_mask_key=self.tops_mask_key,
            tops_kin_key=self.tops_kin_key,
            ws_mask_key=self.ws_mask_key,
            ws_kin_key=self.ws_kin_key,
            load_interactions=self.load_interactions,
        )
        return MemmapDataset(
            npy_dir=npy_dir,
            tops_mask_key=self.tops_mask_key,
            tops_kin_key=self.tops_kin_key,
            ws_mask_key=self.ws_mask_key,
            ws_kin_key=self.ws_kin_key,
            load_interactions=self.load_interactions,
        )

    def _load_split_eager(self, path: Path) -> MaskedFormerDataSet:
        with h5py.File(path, "r") as f:
            jet = f["jet"][()]
            src_mask = f["src_mask"][()]

            if self.load_interactions and "interactions" in f:
                interactions = f["interactions"][()].astype(np.float16)
            else:
                interactions = None

            tops_masks, tops_kins = _load_object_type(
                f, self.tops_mask_key, self.tops_kin_key
            )
            ws_masks, ws_kins = _load_object_type(
                f, self.ws_mask_key, self.ws_kin_key
            )

            valid_tops = f["valid_tops"][()] if "valid_tops" in f else None
            valid_Ws = f["valid_Ws"][()] if "valid_Ws" in f else None

        masks, kins, classes, object_valid = merge_object_types(
            tops_masks, tops_kins, ws_masks, ws_kins, valid_tops, valid_Ws
        )

        has_partial = (object_valid is not None and not object_valid.all())

        ds = MaskedFormerDataSet(
            jet=jet,
            interactions=interactions,
            src_mask=src_mask,
            targets=masks,
            target_kinematics=kins,
            classes=classes,
            object_valid=object_valid if has_partial else None,
        )
        return ds

    def setup(self, stage):
        if stage in (None, "fit", "validate"):
            self.train_dataset = self._load_split("train")
            self.val_dataset = self._load_split("val")
            mode = {False: "in-memory", True: "lazy", "memmap": "memmap"}.get(self.lazy, str(self.lazy))
            print(f"[DM TopsWs] train len={len(self.train_dataset)}  val len={len(self.val_dataset)}  {mode}")
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
