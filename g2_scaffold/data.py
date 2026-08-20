"""G2-only filtered views over the repository's existing HDF5 datasets."""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from torch.utils.data import Dataset

from data.datamodule import MaskedFormerTopsWsDataModule


_REQUIRED_SPLITS = ("train", "val", "calibration", "test", "stress")


def _index_path(manifest_path: Path, entry, split: str) -> Path:
    if isinstance(entry, str):
        value = entry
    elif isinstance(entry, dict):
        value = next(
            (entry[key] for key in ("path", "file", "index_path", "eligible_index", "eligible_indices") if key in entry),
            None,
        )
        if value is None:
            raise ValueError(f"eligibility manifest entry for {split!r} has no .npy path")
    else:
        raise ValueError(f"eligibility manifest entry for {split!r} must be a path")
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def load_eligibility_manifest(path: Union[str, Path]) -> dict[str, np.ndarray]:
    """Load and validate per-split integer row indices without copying HDF5."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"G2 eligibility manifest is required: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid G2 eligibility manifest JSON: {manifest_path}") from exc

    entries = payload.get("splits", payload)
    if not isinstance(entries, dict):
        raise ValueError("G2 eligibility manifest must contain a split mapping")

    result = {}
    for split in _REQUIRED_SPLITS:
        if split not in entries:
            raise ValueError(f"G2 eligibility manifest missing required split {split!r}")
        index_path = _index_path(manifest_path, entries[split], split)
        if index_path.suffix != ".npy" or not index_path.is_file():
            raise FileNotFoundError(f"missing G2 {split} eligibility index: {index_path}")
        indices = np.load(index_path, allow_pickle=False)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"G2 {split} eligibility index must be a 1-D integer .npy array: {index_path}")
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            raise ValueError(f"G2 {split} eligibility index is empty: {index_path}")
        if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
            raise ValueError(f"G2 {split} eligibility index contains negative or duplicate rows: {index_path}")
        result[split] = indices
    return result


class G2FilteredDataset(Dataset):
    """A row-indexed view; the underlying HDF5/memmap dataset is untouched."""

    def __init__(self, dataset: Dataset, indices: np.ndarray, split: str):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.split = split
        if np.any(self.indices >= len(dataset)):
            bad = int(self.indices[self.indices >= len(dataset)][0])
            raise IndexError(f"G2 {split} eligibility row {bad} exceeds dataset length {len(dataset)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]


class G2DataModule(MaskedFormerTopsWsDataModule):
    """Existing data module plus mandatory G2 eligibility views."""

    def __init__(self, config, test_split: str = "test"):
        manifest = config.get("g2", {}).get("eligibility_manifest")
        if not manifest:
            raise ValueError("G2 training/evaluation requires g2.eligibility_manifest")
        if test_split not in ("test", "stress"):
            raise ValueError("test_split must be 'test' or 'stress'")
        self.eligible_indices = load_eligibility_manifest(manifest)
        self.test_split = test_split
        self._setup_stages = set()
        super().__init__(config)

    def setup(self, stage):
        stage_key = stage or "all"
        if stage_key in self._setup_stages:
            return
        super().setup(stage)
        if stage in (None, "fit", "validate"):
            self.train_dataset = G2FilteredDataset(self.train_dataset, self.eligible_indices["train"], "train")
            self.val_dataset = G2FilteredDataset(self.val_dataset, self.eligible_indices["val"], "val")
        if stage in (None, "test"):
            self.test_dataset = G2FilteredDataset(self.test_dataset, self.eligible_indices[self.test_split], self.test_split)
        if stage in (None, "calibrate"):
            self.calibration_dataset = G2FilteredDataset(self.calibration_dataset, self.eligible_indices["calibration"], "calibration")
        self._setup_stages.add(stage_key)
