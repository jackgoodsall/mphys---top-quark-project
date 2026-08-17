import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.data_contract import content_hash
from src.data.split_raw_h5 import split_file


class GroupSplitTest(unittest.TestCase):
    def test_all_rows_and_whole_groups_are_preserved(self):
        cfg = {
            "seed": 4,
            "generator_group_size": 4,
            "fractions": {"train": .5, "val": .2, "calibration": .15, "test": .15},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.h5"
            entries = np.arange(40, dtype=np.uint64)
            with h5py.File(source, "w") as handle:
                handle.create_dataset("event_id", data=entries + 100)
                handle.create_dataset("source_file_id", data=np.ones(40, dtype=np.uint16))
                handle.create_dataset("source_entry", data=entries)
                handle.create_dataset("generator_group", data=entries // 4)
                handle.create_dataset("jet", data=np.arange(80).reshape(40, 2))
                vlen = h5py.vlen_dtype(np.dtype("f4"))
                variable = handle.create_dataset("incidence", shape=(40,), dtype=vlen)
                for row in range(40):
                    variable[row] = np.arange(6, dtype=np.float32) + row

            report = split_file(source, root / "out", "part_", cfg, chunk_size=7)
            self.assertEqual(sum(report["event_counts"].values()), 40)

            ids = []
            groups = {}
            for split in cfg["fractions"]:
                with h5py.File(root / "out" / f"part_{split}.h5", "r") as handle:
                    ids.extend(handle["event_id"][:].tolist())
                    self.assertEqual(handle["incidence"].shape[0], handle["event_id"].shape[0])
                    for group in np.unique(handle["generator_group"][:]):
                        self.assertNotIn(int(group), groups)
                        groups[int(group)] = split
            self.assertEqual(sorted(ids), list(range(100, 140)))

    def test_refuses_legacy_files_without_identity(self):
        cfg = {
            "seed": 1,
            "generator_group_size": 4,
            "fractions": {"train": .5, "val": .2, "calibration": .15, "test": .15},
        }
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy.h5"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("jet", data=np.zeros((2, 3)))
            with self.assertRaisesRegex(ValueError, "identity keys"):
                split_file(source, Path(tmp) / "out", "x_", cfg)

    def test_refuses_stale_or_unhashed_contract_input(self):
        cfg = {
            "seed": 1,
            "generator_group_size": 4,
            "fractions": {"train": .5, "val": .2, "calibration": .15, "test": .15},
        }
        contract = {
            "schema_version": "v3",
            "selection": {"pt": 25},
            "matching": {"version": "exact"},
        }
        # content_hash covers the whole contract, including the split config in production.
        contract["split"] = cfg
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.h5"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("event_id", data=np.array([1], dtype=np.uint64))
                handle.create_dataset("source_file_id", data=np.array([1], dtype=np.uint16))
                handle.create_dataset("source_entry", data=np.array([0], dtype=np.uint64))
                handle.create_dataset("generator_group", data=np.array([0], dtype=np.uint64))
                handle.attrs["schema_version"] = "v3"
                handle.attrs["contract_hash"] = "stale"
                handle.attrs["selection_hash"] = content_hash(contract["selection"])
                handle.attrs["matcher_hash"] = content_hash(contract["matching"])
                handle.attrs["source_hash"] = "not-computed-audit-only"
            with self.assertRaisesRegex(ValueError, "do not match"):
                split_file(source, Path(tmp) / "out", "x_", cfg, contract=contract)

            with h5py.File(source, "a") as handle:
                handle.attrs["contract_hash"] = content_hash(contract)
            with self.assertRaisesRegex(ValueError, "source hash"):
                split_file(source, Path(tmp) / "out", "x_", cfg, contract=contract)


if __name__ == "__main__":
    unittest.main()
