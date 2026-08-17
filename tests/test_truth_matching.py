import unittest

import numpy as np

from src.data.truth_matching import (
    compare_stage_matchability,
    match_truth_to_jets,
    pairwise_delta_r,
    summarize_matchability,
)


class TruthMatchingTest(unittest.TestCase):
    def test_pairwise_delta_r_wraps_phi_and_vectorizes(self):
        distances = pairwise_delta_r(
            [0.0, 1.0], [np.pi - 0.1, 0.0],
            [0.0, 2.0], [-np.pi + 0.1, 0.0],
        )
        self.assertEqual(distances.shape, (2, 2))
        self.assertAlmostEqual(distances[0, 0], 0.2)
        self.assertAlmostEqual(distances[1, 1], 1.0)

    def test_global_assignment_maximizes_matches_before_distance(self):
        result = match_truth_to_jets(
            [0.0, 0.01], [0.0, 0.0],
            [0.011, -0.05], [0.0, 0.0],
            radius=0.055,
        )
        np.testing.assert_array_equal(result.selected_jet, [1, 0])
        self.assertEqual(len(set(result.selected_jet)), 2)

    def test_equal_cost_assignment_has_deterministic_lexicographic_tie_break(self):
        expected = np.array([0, 1])
        for _ in range(5):
            result = match_truth_to_jets(
                [0.0, 0.0], [0.0, 0.0],
                [0.0, 0.0], [0.0, 0.0],
                radius=0.1,
            )
            np.testing.assert_array_equal(result.selected_jet, expected)

    def test_preserves_candidates_alternatives_and_ambiguity_flags(self):
        result = match_truth_to_jets(
            [0.0, 0.02], [0.0, 0.0],
            [0.0, 0.01, 1.0], [0.0, 0.0, 0.0],
            radius=0.05,
        )
        np.testing.assert_array_equal(
            result.incidence,
            [[True, True, False], [True, True, False]],
        )
        np.testing.assert_array_equal(result.second_closest_local_jet, [1, 0])
        np.testing.assert_array_equal(result.parton_ambiguous, [True, True])
        np.testing.assert_array_equal(result.parton_collision, [True, True])
        np.testing.assert_array_equal(result.jet_collision, [True, True, False])
        self.assertTrue(result.has_ambiguity)
        self.assertTrue(result.has_collision)

    def test_second_closest_local_jet_is_not_a_global_assignment_alternative(self):
        result = match_truth_to_jets(
            [0.0, 0.01], [0.0, 0.0],
            [0.011, -0.05], [0.0, 0.0],
            radius=0.055,
        )
        self.assertEqual(result.selected_jet[0], 1)
        self.assertEqual(result.second_closest_local_jet[0], 1)
        self.assertAlmostEqual(result.second_closest_local_distance[0], 0.05)

    def test_stage_mask_keeps_full_incidence_and_reports_matchability_loss(self):
        args = ([0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0])
        pre = match_truth_to_jets(*args, radius=0.1)
        post = match_truth_to_jets(*args, radius=0.1, jet_mask=[True, False])

        np.testing.assert_array_equal(pre.incidence, post.incidence)
        np.testing.assert_array_equal(post.stage_incidence[:, 1], [False, False])
        stages = compare_stage_matchability(pre, post)
        self.assertTrue(stages.pre.fully_matchable)
        self.assertFalse(stages.post.fully_matchable)
        np.testing.assert_array_equal(stages.lost_partons, [False, True])

    def test_required_parton_subset_and_empty_inputs(self):
        summary = summarize_matchability([0, -1, -1], [True, False, False])
        self.assertEqual(summary.matched_count, 1)
        self.assertEqual(summary.required_count, 1)
        self.assertTrue(summary.fully_matchable)

        empty = match_truth_to_jets([], [], [], [], radius=0.4)
        self.assertEqual(empty.distances.shape, (0, 0))
        self.assertTrue(empty.matchability().fully_matchable)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            pairwise_delta_r([0.0], [0.0, 1.0], [], [])
        with self.assertRaises(ValueError):
            match_truth_to_jets([0.0], [0.0], [0.0], [0.0], radius=-1)
        with self.assertRaises(ValueError):
            match_truth_to_jets([0.0], [0.0], [0.0], [0.0], jet_mask=[])


if __name__ == "__main__":
    unittest.main()
