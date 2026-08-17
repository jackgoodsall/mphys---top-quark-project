import inspect
import itertools
import time
import unittest

import numpy as np

from src.analysis.hierarchical_decoder import ABSENT, FULL_TOP, W_ONLY, decode_hierarchical


class HierarchicalDecoderTest(unittest.TestCase):
    def scores(self, particles=4):
        return (
            np.zeros((1, 2, 3)),
            np.zeros((1, 2, particles, particles)),
            np.zeros((1, 2, particles, particles, particles)),
            np.ones((1, particles), dtype=bool),
        )

    def test_signature_cannot_receive_truth(self):
        names = inspect.signature(decode_hierarchical).parameters
        self.assertFalse({"targets", "truth", "n_matchable"} & set(names))

    def test_absent_w_only_and_full_top_support(self):
        state, w_pair, b_ext, valid = self.scores()
        state[0, 0] = [5, 0, 0]
        state[0, 1] = [0, 4, 3]
        w_pair[0, 1, 1, 2] = 2
        decoded = decode_hierarchical(state, w_pair, b_ext, valid)
        self.assertEqual(decoded.state.tolist(), [[ABSENT, W_ONLY]])
        self.assertEqual(decoded.w_mask[0, 1].sum(), 2)
        self.assertEqual(decoded.top_mask[0, 1].sum(), 2)

        state[0, 0] = [0, 0, 10]
        w_pair[0, 0, 0, 1] = 4
        b_ext[0, 0, 2, 0, 1] = 3
        decoded = decode_hierarchical(state, w_pair, b_ext, valid)
        self.assertEqual(decoded.state[0, 0], FULL_TOP)
        self.assertEqual(decoded.top_mask[0, 0].sum(), 3)
        self.assertEqual(decoded.w_mask[0, 0].sum(), 2)

    def test_exact_decode_enforces_all_legality_invariants(self):
        rng = np.random.default_rng(4)
        state, w_pair, b_ext, valid = self.scores(5)
        state[:] = rng.normal(size=state.shape)
        w_pair[:] = rng.normal(size=w_pair.shape)
        b_ext[:] = rng.normal(size=b_ext.shape)
        decoded = decode_hierarchical(state, w_pair, b_ext, valid)
        self.assertFalse(np.any(decoded.top_mask[:, 0] & decoded.top_mask[:, 1]))
        self.assertTrue(np.all(~decoded.w_mask | decoded.top_mask))
        for query in range(2):
            if decoded.state[0, query] == W_ONLY:
                self.assertEqual(decoded.top_mask[0, query].sum(), 2)
            elif decoded.state[0, query] == FULL_TOP:
                self.assertEqual(decoded.top_mask[0, query].sum(), 3)
                self.assertEqual(decoded.w_mask[0, query].sum(), 2)
        self.assertGreaterEqual(decoded.margin[0], 0)

    def test_padded_jets_and_nonfinite_scores(self):
        args = list(self.scores())
        args[3][0, 3] = False
        args[1][0, :, 3, :] = 100
        decoded = decode_hierarchical(*args)
        self.assertFalse(decoded.top_mask[..., 3].any())
        args[0][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            decode_hierarchical(*args)

    def test_declared_n20_support_avoids_python_cartesian_product(self):
        rng = np.random.default_rng(9)
        state = rng.normal(size=(1, 2, 3))
        w_pair = rng.normal(size=(1, 2, 20, 20))
        b_extension = rng.normal(size=(1, 2, 20, 20, 20))
        start = time.perf_counter()
        decoded = decode_hierarchical(
            state, w_pair, b_extension, np.ones((1, 20), dtype=bool)
        )
        elapsed = time.perf_counter() - start
        self.assertTrue(np.isfinite(decoded.score).all())
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()
