import copy
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.data_contract import load_contract
from src.data.root_to_h5 import (
    TRUTH_DILEPTONIC,
    TRUTH_HADRONIC,
    TRUTH_SEMILEPTONIC,
    process_event,
    truth_decay_channel,
    _append,
    _create_datasets,
)


def event(jet_pt=None, jet_eta=None, jet_phi=None, jet_btag=None):
    pt = np.asarray(jet_pt if jet_pt is not None else [80, 70, 60, 55, 50, 45], dtype=np.float32)
    return {
        "jet_pt": pt,
        "jet_eta": np.asarray(jet_eta if jet_eta is not None else [0, .1, .2, 1, 1.1, 1.2]),
        "jet_phi": np.asarray(jet_phi if jet_phi is not None else [3.13, .1, .2, 1, 1.1, 1.2]),
        "jet_mass": np.ones(len(pt), dtype=np.float32) * 5,
        "jet_btag": np.asarray(jet_btag if jet_btag is not None else [1, 0, 0, 1, 0, 0]),
        "el_pt": np.array([]),
        "mu_pt": np.array([]),
        "event_number": 99,
    }


def truth(ids=(-1, 2, 1, -2)):
    return {
        "top_id": np.array([6, -6]),
        "w_id": np.array([24, -24]),
        "b_id": np.array([5, -5]),
        "b_eta": np.array([0, 1]),
        "b_phi": np.array([-3.13, 1]),
        "w_decay_eta": np.array([.1, .2, 1.1, 1.2]),
        "w_decay_phi": np.array([.1, .2, 1.1, 1.2]),
        "w_decay_id": np.array(ids),
    }


class RootContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract("config/top_reconstruction_data_contract.yaml")

    def test_truth_decay_categories(self):
        self.assertEqual(truth_decay_channel([-1, 2, 1, -2]), TRUTH_HADRONIC)
        self.assertEqual(truth_decay_channel([-11, 12, 1, -2]), TRUTH_SEMILEPTONIC)
        self.assertEqual(truth_decay_channel([-11, 12, 13, -14]), TRUTH_DILEPTONIC)

    def test_selection_boundaries_and_local_wrapped_matching(self):
        record, bits = process_event(event(), truth(), 1, 7, self.contract)
        self.assertIsNotNone(record)
        self.assertTrue(all(bits.values()))
        self.assertEqual(record["source_entry"], 7)
        self.assertTrue(np.all(record["selected_parton_jet"] >= 0))
        self.assertIn("second_closest_local_raw_jet", record)
        self.assertIn("second_closest_local_distance", record)
        self.assertNotIn("second_best_raw_jet", record)
        np.testing.assert_array_equal(record["jet"][:6, 6], np.arange(1, 7))
        self.assertAlmostEqual(record["selected_parton_distance"][0], 2 * np.pi - 6.26, places=5)

        at_boundary = event(jet_pt=[25] * 6, jet_eta=[2.5, 0, 0, 0, 0, 0])
        record, bits = process_event(at_boundary, truth(), 1, 8, self.contract)
        self.assertTrue(bits["selection_min_jets"])

        below = event(jet_pt=[24.99, 30, 30, 30, 30, 30])
        record, bits = process_event(below, truth(), 1, 9, self.contract)
        self.assertIsNone(record)
        self.assertFalse(bits["selection_min_jets"])

    def test_truth_and_reco_population_bits_are_independent(self):
        reco = event()
        reco["el_pt"] = np.array([20.0])
        record, bits = process_event(reco, truth((-11, 12, 1, -2)), 1, 0, self.contract)
        self.assertIsNone(record)
        self.assertFalse(bits["selection_truth_decay"])
        self.assertFalse(bits["selection_reco_lepton_veto"])

    def test_truncation_is_reported_and_changes_post_stage_matchability(self):
        contract = copy.deepcopy(self.contract)
        contract["truncation"]["max_particles"] = 6
        reco = event(
            jet_pt=[100, 90, 80, 70, 60, 50, 40],
            jet_eta=[0, .1, .2, 1, 1.1, 2.0, 1.2],
            jet_phi=[-3.13, .1, .2, 1, 1.1, 2.0, 1.2],
            jet_btag=[1, 0, 0, 1, 0, 0, 0],
        )
        record, _ = process_event(reco, truth(), 1, 2, contract)
        self.assertEqual(record["truncated_at_max_particles"], 1)
        self.assertEqual(record["pre_acceptance_matchable"].sum(), 6)
        self.assertEqual(record["post_truncation_matchable"].sum(), 5)

    def test_vlen_matching_arrays_round_trip_through_hdf5(self):
        record, _ = process_event(event(), truth(), 1, 7, self.contract)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contract.h5"
            with h5py.File(path, "w") as handle:
                _create_datasets(handle, 20, "gzip", 1)
                _append(handle, [record])
            with h5py.File(path, "r") as handle:
                np.testing.assert_array_equal(
                    handle["match_incidence_raw"][0], record["match_incidence_raw"]
                )
                np.testing.assert_allclose(
                    handle["match_delta_r_raw"][0], record["match_delta_r_raw"]
                )


if __name__ == "__main__":
    unittest.main()
