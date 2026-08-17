import unittest

import torch

from src.models.components.masked_former_tasks import (
    MaskReconstructionTask,
    TaskConfig,
    TaskRegistry,
)
from src.models.particle_transformer import (
    InteractionEmbedder,
    MaskedReconstructionPart,
    ParticleEmbedder,
)


def build_model():
    registry = TaskRegistry()
    for name, pred, target, validity in (
        ("mask", "mask_predictions", "jet_mask_true", "top_valid"),
        ("mask_W", "mask_W", "jet_mask_true_W", "w_valid"),
    ):
        registry.register_task(MaskReconstructionTask(
            TaskConfig(
                name=name, output_names=[pred], output_dims={}, cost_weights={"mask": 1.0},
                loss_weights={"dice": 1.0, "bce": .5}, max_objects=2,
                validity_key=validity,
            ), pred_key=pred, target_key=target, validity_key=validity,
        ))
    return MaskedReconstructionPart(
        particle_embedder=ParticleEmbedder(7, [8], 8, 0.0),
        interaction_embedder=InteractionEmbedder(4, [4], 2, 0.0),
        task_registry=registry,
        embedding_size=8, n_encoder_layers=1, n_decoder_layers=2, n_heads=2,
        p_dropout=0.0, dim_ff=16, num_query_tokens=2,
        hierarchical_decoding=True, chain_queries=True, matching_solver="scipy",
    )


class TargetFreeInferenceTest(unittest.TestCase):
    def test_raw_outputs_are_bitwise_independent_of_targets(self):
        torch.manual_seed(3)
        model = build_model()
        model.eval()
        samples = {
            "jet": torch.randn(1, 3, 7),
            "src_mask": torch.ones(1, 3, dtype=torch.bool),
            "interactions": torch.rand(1, 3, 3, 4),
        }
        targets = {"sentinel": torch.randn(1, 2, 3)}
        contaminated = dict(samples)
        contaminated["targets"] = targets
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.no_grad():
                reference = model(samples)
                alternative = model(contaminated)
        finally:
            torch.set_num_threads(old_threads)
        self.assertEqual(reference.keys(), alternative.keys())
        for layer in reference:
            self.assertEqual(reference[layer].keys(), alternative[layer].keys())
            for key in reference[layer]:
                self.assertTrue(torch.equal(reference[layer][key], alternative[layer][key]), key)


if __name__ == "__main__":
    unittest.main()
