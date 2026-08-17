import tempfile
import unittest
import copy
from pathlib import Path

import numpy as np

from src.data.data_contract import (
    content_hash,
    load_contract,
    split_names,
    stable_event_id,
)


class DataContractTest(unittest.TestCase):
    def test_repository_contract_is_valid_and_stable(self):
        contract = load_contract("config/top_reconstruction_data_contract.yaml")
        self.assertEqual(contract["schema_version"], "top-reconstruction-v3")
        self.assertEqual(content_hash(contract), content_hash(dict(contract)))

    def test_event_id_round_trip_layout(self):
        ids = stable_event_id(np.array([1, 2]), np.array([7, 9]))
        self.assertEqual((ids >> np.uint64(48)).tolist(), [1, 2])
        self.assertEqual((ids & np.uint64(2**48 - 1)).tolist(), [7, 9])

    def test_split_assignment_keeps_groups_together_and_is_order_independent(self):
        cfg = {
            "seed": 42,
            "generator_group_size": 10,
            "fractions": {"train": .8, "val": .1, "calibration": .05, "test": .05},
        }
        entries = np.array([0, 9, 10, 19, 20, 29])
        splits = split_names(1, entries, cfg)
        self.assertEqual(splits[0], splits[1])
        self.assertEqual(splits[2], splits[3])
        rev = split_names(1, entries[::-1], cfg)[::-1]
        np.testing.assert_array_equal(splits, rev)

    def test_negative_split_fraction_is_rejected(self):
        from src.data.data_contract import validate_contract
        contract = copy.deepcopy(load_contract("config/top_reconstruction_data_contract.yaml"))
        contract["split"]["fractions"] = {
            "train": 1.1, "val": -.1, "calibration": 0.0, "test": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "negative"):
            validate_contract(contract)

    def test_unsupported_semantics_are_rejected_instead_of_ignored(self):
        from src.data.data_contract import validate_contract

        changes = (
            ("population", "truth_decay", "anything_else"),
            ("matching", "objective", "greedy"),
            ("matching", "tie_break", "random"),
            ("matching", "exact_cardinality", False),
            ("truncation", "order", "pt"),
        )
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                contract = copy.deepcopy(load_contract("config/top_reconstruction_data_contract.yaml"))
                contract[section][key] = value
                with self.assertRaises(ValueError):
                    validate_contract(contract)


if __name__ == "__main__":
    unittest.main()
