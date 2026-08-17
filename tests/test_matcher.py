import unittest

import torch

from src.models.components.matcher import create_matcher


class MatcherValidityTest(unittest.TestCase):
    def test_non_contiguous_valid_target_is_compacted(self):
        costs = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
        valid = torch.tensor([[False, True]])

        for solver in ("scipy", "gpu_bruteforce"):
            matcher = create_matcher(solver, num_queries=2, max_targets=2)
            matched = matcher(costs, object_valid_mask=valid)
            self.assertEqual(matched.shape, (1, 2))
            self.assertEqual(matched[0, 0].item(), 0)
            self.assertEqual(sorted(matched[0].tolist()), [0, 1])

    def test_gpu_matcher_fallback_preserves_valid_lengths(self):
        costs = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
        valid = torch.tensor([[False, True]])
        matcher = create_matcher("gpu_bruteforce", num_queries=2, max_targets=1)
        matched = matcher(costs, object_valid_mask=valid)
        self.assertEqual(matched.tolist(), [[0, 1]])

    def test_more_valid_targets_than_queries_fails_clearly(self):
        matcher = create_matcher("gpu_bruteforce", num_queries=2, max_targets=10)
        with self.assertRaisesRegex(ValueError, "more valid targets than queries"):
            matcher(torch.zeros(1, 2, 3), object_valid_mask=torch.ones(1, 3, dtype=torch.bool))

    def test_binary_assignment_margin_is_exposed(self):
        matcher = create_matcher("gpu_bruteforce", num_queries=2, max_targets=2)
        matcher(torch.tensor([[[0.0, 3.0], [3.0, 0.0]]]))
        self.assertEqual(matcher.last_cost_margin.shape, (1,))
        self.assertAlmostEqual(matcher.last_cost_margin.item(), 6.0)

    def test_margin_ignores_duplicate_padded_target_assignments(self):
        matcher = create_matcher("gpu_bruteforce", num_queries=2, max_targets=2)
        costs = torch.tensor([[[0.0, 99.0], [3.0, 99.0]]])
        matcher(costs, object_valid_mask=torch.tensor([[True, False]]))
        self.assertAlmostEqual(matcher.last_cost_margin.item(), 3.0)

    def test_margin_is_reset_for_empty_targets(self):
        matcher = create_matcher("gpu_bruteforce", num_queries=2, max_targets=2)
        matcher(torch.tensor([[[0.0], [3.0]]]))
        matcher(torch.empty(1, 2, 0))
        self.assertEqual(matcher.last_cost_margin.shape, (1,))
        self.assertTrue(torch.isinf(matcher.last_cost_margin).all())

    def test_fallback_exposes_exact_margin(self):
        matcher = create_matcher("gpu_bruteforce", num_queries=2, max_targets=1)
        with self.assertWarns(RuntimeWarning):
            matcher(torch.tensor([[[0.0, 3.0], [3.0, 0.0]]]))
        self.assertAlmostEqual(matcher.last_cost_margin.item(), 6.0)


if __name__ == "__main__":
    unittest.main()
