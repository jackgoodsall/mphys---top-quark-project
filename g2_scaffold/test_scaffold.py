import unittest

import torch

from g2_scaffold.decoder import FULL_TOP, W_ONLY, decode_event
from g2_scaffold.losses import hierarchical_loss
from g2_scaffold.model import G2CandidateScorer
from g2_scaffold.targets import targets_to_g2


class G2ScaffoldTest(unittest.TestCase):
    def test_scorer_shapes_symmetry_and_gradients(self):
        torch.manual_seed(3)
        scorer = G2CandidateScorer(8, 4)
        queries = torch.randn(2, 2, 8, requires_grad=True)
        particles = torch.randn(2, 5, 8, requires_grad=True)
        outputs = scorer(queries, particles)
        self.assertEqual(tuple(outputs["state_logits"].shape), (2, 2, 3))
        self.assertEqual(tuple(outputs["w_pair_logits"].shape), (2, 2, 5, 5))
        self.assertEqual(tuple(outputs["b_extension_logits"].shape), (2, 2, 5, 5, 5))
        torch.testing.assert_close(
            outputs["w_pair_logits"], outputs["w_pair_logits"].transpose(-1, -2)
        )
        torch.testing.assert_close(
            outputs["b_extension_logits"], outputs["b_extension_logits"].transpose(-1, -2)
        )
        sum(value.sum() for value in outputs.values()).backward()
        self.assertIsNotNone(queries.grad)
        self.assertIsNotNone(particles.grad)

    def test_decoder_emits_partial_and_full_chains_without_overlap(self):
        state = torch.full((2, 3), -10.0)
        state[0, W_ONLY] = 10.0
        state[1, FULL_TOP] = 10.0
        w_pair = torch.full((2, 6, 6), -10.0)
        w_pair[0, 0, 1] = w_pair[0, 1, 0] = 8.0
        w_pair[1, 2, 3] = w_pair[1, 3, 2] = 8.0
        b_extension = torch.full((2, 6, 6, 6), -10.0)
        b_extension[1, 4, 2, 3] = 8.0
        decoded = decode_event(state, w_pair, b_extension, torch.ones(6, dtype=torch.bool))
        self.assertEqual(decoded.chains[0].state, W_ONLY)
        self.assertEqual(decoded.chains[0].w_pair, (0, 1))
        self.assertEqual(decoded.chains[1].state, FULL_TOP)
        self.assertEqual(decoded.chains[1].w_pair, (2, 3))
        self.assertEqual(decoded.chains[1].b_jet, 4)
        self.assertFalse(decoded.chains[0].jets & decoded.chains[1].jets)

    def test_decoder_rejects_overlap_in_favour_of_legal_choice(self):
        state = torch.full((2, 3), -10.0)
        state[:, FULL_TOP] = 10.0
        w_pair = torch.full((2, 6, 6), -10.0)
        b_extension = torch.full((2, 6, 6, 6), -10.0)
        for q in range(2):
            w_pair[q, 0, 1] = 8.0
            b_extension[q, 2, 0, 1] = 8.0
        w_pair[1, 3, 4] = 4.0
        b_extension[1, 5, 3, 4] = 4.0
        decoded = decode_event(state, w_pair, b_extension, torch.ones(6, dtype=torch.bool))
        self.assertFalse(decoded.chains[0].jets & decoded.chains[1].jets)
        self.assertEqual(decoded.chains[1].w_pair, (3, 4))

    def test_loss_is_chain_swap_invariant_and_finite(self):
        torch.manual_seed(4)
        outputs = {
            "state_logits": torch.randn(1, 2, 3, requires_grad=True),
            "w_pair_logits": torch.randn(1, 2, 5, 5, requires_grad=True),
            "b_extension_logits": torch.randn(1, 2, 5, 5, 5, requires_grad=True),
        }
        states = torch.tensor([[W_ONLY, FULL_TOP]])
        w_targets = torch.tensor([[[0, 1], [2, 3]]])
        b_targets = torch.tensor([[-1, 4]])
        swapped = (states.flip(1), w_targets.flip(1), b_targets.flip(1))
        for mode in ("hard_min", "marginal"):
            first = hierarchical_loss(outputs, states, w_targets, b_targets, mode=mode)
            second = hierarchical_loss(outputs, *swapped, mode=mode)
            torch.testing.assert_close(first, second)
            self.assertTrue(torch.isfinite(first))
            first.backward(retain_graph=True)

    def test_loss_handles_censored_states_and_padding(self):
        torch.manual_seed(5)
        outputs = {
            "state_logits": torch.randn(2, 2, 3),
            "w_pair_logits": torch.randn(2, 2, 5, 5),
            "b_extension_logits": torch.randn(2, 2, 5, 5, 5),
        }
        states = torch.tensor([[0, W_ONLY], [FULL_TOP, 0]])
        w_targets = torch.tensor([[[-1, -1], [1, 2]], [[0, 1], [-1, -1]]])
        b_targets = torch.tensor([[-1, -1], [3, -1]])
        valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.bool)
        for mode in ("hard_min", "marginal"):
            loss = hierarchical_loss(
                outputs, states, w_targets, b_targets, mode=mode, valid_particles=valid
            )
            self.assertTrue(torch.isfinite(loss))

    def test_target_conversion_preserves_three_states(self):
        masks = torch.zeros(1, 4, 8)
        masks[0, 0, [0, 1, 2]] = 1  # full top 0; W0 is [1, 2]
        masks[0, 1, [3, 4, 5]] = 1  # top 1 invalid
        masks[0, 2, [1, 2]] = 1
        masks[0, 3, [3, 4]] = 1  # W-only 1
        converted = targets_to_g2({
            "jet_mask_true": masks,
            "jet_valid_mask": torch.ones(1, 8, dtype=torch.bool),
            "target_valid_mask": torch.tensor([[True, False, True, True]]),
            "classes": torch.tensor([[1, 1, 2, 2]]),
        })
        torch.testing.assert_close(converted["state_targets"], torch.tensor([[FULL_TOP, W_ONLY]]))
        torch.testing.assert_close(converted["w_targets"], torch.tensor([[[1, 2], [3, 4]]]))
        torch.testing.assert_close(converted["b_targets"], torch.tensor([[0, -1]]))


if __name__ == "__main__":
    unittest.main()
