import unittest

import numpy as np

from src.analysis.evaluate_chain import compute_efficiencies


class EvaluateChainS2Test(unittest.TestCase):
    def test_swapped_raw_queries_are_perfect(self):
        truth_top = np.array([[[1, 1, 0, 0], [0, 0, 1, 1]]], dtype=np.float32)
        truth_w = np.array([[[0, 1, 0, 0], [0, 0, 0, 1]]], dtype=np.float32)
        result = compute_efficiencies(
            truth_top[:, ::-1], truth_top,
            truth_w[:, ::-1], truth_w,
            np.ones((1, 4), dtype=bool),
            target_obj_top=None,
            slot_valid_top=np.ones((1, 2), dtype=bool),
            slot_valid_W=np.ones((1, 2), dtype=bool),
            use_probs=True,
            threshold=0.5,
            event_ids=np.array([99], dtype=np.uint64),
        )
        self.assertEqual(result["exact_match"], 1.0)
        rows = {row["metric"]: row for row in result["scorecard"]}
        self.assertEqual(rows["M2"]["numerator_event_ids"].tolist(), [99])


if __name__ == "__main__":
    unittest.main()
