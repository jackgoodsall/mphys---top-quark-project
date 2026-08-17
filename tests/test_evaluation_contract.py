import unittest

import numpy as np

from src.analysis.evaluation_contract import (
    assert_aligned_event_ids,
    minimal_scorecard,
    score_s2,
    wilson_interval,
)


class EvaluationContractTest(unittest.TestCase):
    def test_swap_invariant_scoring_and_denominators(self):
        truth_top = np.array([[[1, 1, 0, 0], [0, 0, 1, 1]]], dtype=bool)
        truth_w = np.array([[[0, 1, 0, 0], [0, 0, 0, 1]]], dtype=bool)
        score = score_s2(
            truth_top[:, ::-1], truth_w[:, ::-1], truth_top, truth_w,
            np.ones((1, 4), bool), np.ones((1, 2), bool), np.ones((1, 2), bool),
        )
        self.assertEqual(score.permutation.tolist(), [1])
        self.assertTrue(score.event_exact[0])
        rows = {row["metric"]: row for row in minimal_scorecard(score, [123])}
        self.assertEqual((rows["M2"]["numerator"], rows["M2"]["denominator"]), (1, 1))
        self.assertEqual((rows["M4"]["numerator"], rows["M4"]["denominator"]), (2, 2))

    def test_partial_chain_censors_missing_component(self):
        zeros = np.zeros((1, 2, 3), dtype=bool)
        truth_w = zeros.copy()
        truth_w[0, 0, :2] = True
        pred_w = truth_w.copy()
        pred_top = np.ones_like(zeros)  # ignored because no top is identifiable
        score = score_s2(
            pred_top, pred_w, zeros, truth_w, np.ones((1, 3), bool),
            np.zeros((1, 2), bool), np.array([[True, False]]),
        )
        self.assertTrue(score.chain_exact[0, 0])
        self.assertFalse(score.fully_matchable[0])
        rows = {row["metric"]: row for row in minimal_scorecard(score, [1])}
        self.assertEqual(rows["M2"]["denominator"], 0)
        self.assertEqual((rows["M3"]["numerator"], rows["M3"]["denominator"]), (1, 1))

    def test_component_cost_tie_prefers_more_exact_chains(self):
        truth_top = np.zeros((1, 2, 3), dtype=bool)
        truth_w = np.array([[[0, 0, 0], [0, 0, 1]]], dtype=bool)
        # Identity has two component errors split over both chains; swapping
        # makes one complete chain exact at the same component-error cost.
        pred_top = np.array([[[0, 0, 0], [0, 0, 1]]], dtype=bool)
        pred_w = np.array([[[0, 0, 1], [0, 0, 1]]], dtype=bool)
        score = score_s2(
            pred_top, pred_w, truth_top, truth_w, np.ones((1, 3), bool),
            np.ones((1, 2), bool), np.ones((1, 2), bool),
        )
        self.assertEqual(score.chain_exact.sum(), 1)

    def test_artifact_identity_mismatch_and_duplicates_fail(self):
        np.testing.assert_array_equal(assert_aligned_event_ids(a=[1, 2], b=[1, 2]), [1, 2])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            assert_aligned_event_ids(a=[1, 2], b=[2, 1])
        with self.assertRaisesRegex(ValueError, "unique"):
            assert_aligned_event_ids(a=[1, 1])

    def test_wilson_counts_are_validated(self):
        low, high = wilson_interval(5, 10)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)


if __name__ == "__main__":
    unittest.main()
