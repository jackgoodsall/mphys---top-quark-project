import tempfile
import unittest
from pathlib import Path
import sys

import h5py
import numpy as np

# Match the repository's existing ``src``-on-PYTHONPATH execution model.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.datamodule import (
    LazyHDF5Dataset,
    MemmapDataset,
    _dataloader_worker_kwargs,
)


class DataModuleTest(unittest.TestCase):
    def test_worker_kwargs_skip_worker_options_without_workers(self):
        config = {
            "num_workers": 0,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
        self.assertEqual(_dataloader_worker_kwargs(config), {"num_workers": 0})
        self.assertEqual(
            _dataloader_worker_kwargs({
                "num_workers": 2,
                "persistent_workers": True,
                "prefetch_factor": 4,
            }),
            {"num_workers": 2, "persistent_workers": True, "prefetch_factor": 4},
        )

    def test_lazy_and_memmap_keep_object_shapes_and_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "split.h5"
            with h5py.File(h5_path, "w") as f:
                f["jet"] = np.zeros((2, 3, 4), dtype=np.float32)
                f["src_mask"] = np.ones((2, 3), dtype=np.uint8)
                f["masks_tops"] = np.zeros((2, 1, 3), dtype=np.float32)
                f["kinematics_tops"] = np.zeros((2, 1, 5), dtype=np.float32)
                f["masks_Ws"] = np.zeros((2, 3, 3), dtype=np.float32)
                f["kinematics_Ws"] = np.zeros((2, 3, 5), dtype=np.float32)
                f["valid_tops"] = np.array([[1], [0]], dtype=np.uint8)
                f["valid_Ws"] = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)

            expected_valid = [True, True, False, True]
            lazy = LazyHDF5Dataset(
                h5_path, "masks_tops", "kinematics_tops", "masks_Ws", "kinematics_Ws"
            )
            memmap_dir = MemmapDataset.prepare(
                h5_path, "masks_tops", "kinematics_tops", "masks_Ws", "kinematics_Ws",
                load_interactions=False,
            )
            memmap = MemmapDataset(
                memmap_dir, "masks_tops", "kinematics_tops", "masks_Ws", "kinematics_Ws",
                load_interactions=False,
            )

            for dataset in (lazy, memmap):
                _, target = dataset[0]
                self.assertEqual(tuple(target["jet_mask_true"].shape), (4, 3))
                self.assertEqual(tuple(target["target_kinematics"].shape), (4, 5))
                self.assertEqual(target["classes"].tolist(), [1, 2, 2, 2])
                self.assertEqual(target["object_valid"].tolist(), expected_valid)


if __name__ == "__main__":
    unittest.main()
