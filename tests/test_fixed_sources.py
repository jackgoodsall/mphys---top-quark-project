import copy
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.convert_fixed_sources import _record, convert
from src.data.data_contract import load_contract


class FixedSourceTest(unittest.TestCase):
    def test_partial_source_is_preserved_and_gets_stable_identity(self):
        contract = copy.deepcopy(load_contract("config/top_reconstruction_data_contract.yaml"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "inclusive.h5"
            jet = np.full((2, 20, 7), np.nan, dtype=np.float32)
            jet[0, :4] = [
                [80, 0, 0, 80, 5, 1, 1], [70, .1, .1, 71, 5, 0, 2],
                [60, .2, .2, 61, 5, 0, 3], [50, .3, .3, 51, 5, 0, 4],
            ]
            jet[1, :2] = [[40, 0, 0, 40, 5, 0, 1], [35, .1, .1, 35, 5, 0, 2]]
            with h5py.File(source, "w") as handle:
                handle.create_dataset("jet", data=jet)
                handle.create_dataset("event", data=np.zeros((2, 3), dtype=np.float32))

            contract["source"]["sources"] = [{
                "name": "synthetic", "path": str(source), "format": "legacy_raw_h5",
                "source_file_id": 99, "split": "train",
            }]
            contract["output"]["split_dir"] = str(root / "splits")
            contract["output"]["manifest_path"] = str(root / "manifest.json")
            manifest = convert(contract, skip_source_hash=True)

            self.assertEqual(manifest["cutflow"]["train_selected"], 2)
            with h5py.File(root / "splits" / "ttbar_contract_raw_train.h5", "r") as handle:
                self.assertEqual(handle["event_id"][0], (99 << 48))
                self.assertEqual(handle["source_entry"][1], 1)
                self.assertEqual(handle["selected_njets"][1], 2)
                self.assertEqual(handle["selection_pass"][1], 1)
                np.testing.assert_array_equal(handle["post_truncation_matchable"][0], [1, 1, 1, 1, 0, 0])

    def test_tagged_record_handles_nan_padding(self):
        contract = load_contract("config/top_reconstruction_data_contract.yaml")
        record, _ = _record(
            [40, np.nan], [0, np.nan], [0, np.nan], [40, np.nan], [5, np.nan],
            [1, 0], [1, np.nan], 3, 0, contract,
        )
        self.assertEqual(record["selected_njets"], 1)
        self.assertEqual(record["selected_parton_jet"][0], 0)


if __name__ == "__main__":
    unittest.main()
