import unittest

import torch

from src.models.components.hierarchical_candidate_head import HierarchicalCandidateHead


class HierarchicalCandidateHeadTest(unittest.TestCase):
    def test_shapes_symmetry_and_gradients(self):
        torch.set_num_threads(1)
        head = HierarchicalCandidateHead(embedding_size=8, hidden_size=4)
        queries = torch.randn(2, 2, 8, requires_grad=True)
        particles = torch.randn(2, 5, 8, requires_grad=True)
        w_pair, b_extension = head(queries, particles)
        self.assertEqual(tuple(w_pair.shape), (2, 2, 5, 5))
        self.assertEqual(tuple(b_extension.shape), (2, 2, 5, 5, 5))
        torch.testing.assert_close(w_pair, w_pair.transpose(-1, -2))
        torch.testing.assert_close(b_extension, b_extension.transpose(-1, -2))
        (w_pair.mean() + b_extension.mean()).backward()
        self.assertIsNotNone(queries.grad)
        self.assertIsNotNone(particles.grad)


if __name__ == "__main__":
    unittest.main()
