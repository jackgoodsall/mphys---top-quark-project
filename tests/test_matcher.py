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


if __name__ == "__main__":
    unittest.main()
