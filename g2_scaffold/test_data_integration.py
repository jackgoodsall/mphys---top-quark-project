import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Match the repository's src-on-PYTHONPATH execution model.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from g2_scaffold.data import G2FilteredDataset, load_eligibility_manifest
from g2_scaffold.targets import targets_to_g2


class _Rows(Dataset):
    def __init__(self, size):
        self.rows = list(range(size))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class G2DataIntegrationTest(unittest.TestCase):
    def test_filtered_view_indexes_without_copying_base_dataset(self):
        base = _Rows(5)
        view = G2FilteredDataset(base, np.array([4, 1], dtype=np.int64), "train")
        self.assertIs(view.dataset, base)
        self.assertEqual(len(view), 2)
        self.assertEqual([view[0], view[1]], [4, 1])

    def test_manifest_loads_all_split_index_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = {}
            for split in ("train", "val", "calibration", "test", "stress"):
                filename = f"{split}_eligible.npy"
                np.save(root / filename, np.array([0, 2], dtype=np.int64))
                entries[split] = filename
            (root / "manifest.json").write_text(json.dumps({"splits": entries}), encoding="utf-8")
            result = load_eligibility_manifest(root / "manifest.json")
            self.assertEqual(sorted(result), ["calibration", "stress", "test", "train", "val"])
            np.testing.assert_array_equal(result["stress"], [0, 2])

    def test_target_guard_reports_event_and_does_not_truncate_three_jet_w(self):
        masks = torch.zeros(1, 4, 6)
        masks[0, 0, [0, 1, 2]] = 1
        masks[0, 1, [3, 4, 5]] = 1
        masks[0, 2, [0, 1, 2]] = 1
        masks[0, 3, [3, 4]] = 1
        targets = {
            "jet_mask_true": masks,
            "jet_valid_mask": torch.ones(1, 6, dtype=torch.bool),
            "target_valid_mask": torch.ones(1, 4, dtype=torch.bool),
            "classes": torch.tensor([[1, 1, 2, 2]]),
        }
        with self.assertRaisesRegex(ValueError, r"exactly two particles.*event_id=123"):
            targets_to_g2(targets, event_ids=torch.tensor([123], dtype=torch.uint64))


if __name__ == "__main__":
    unittest.main()
