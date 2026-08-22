import unittest
from unittest import mock

import torch

from src.trainers.top_reconstruction_trainers import ReconstructionTrainer
from src.models.components.masked_former_tasks import (
    MaskReconstructionTask,
    TaskConfig,
    TaskRegistry,
)


class ValidationMetricTest(unittest.TestCase):
    def test_boundary_ranking_loss_prefers_correct_set_ordering(self):
        config = TaskConfig(
            name="mask",
            output_names=["mask_predictions"],
            output_dims={"mask_predictions": 4},
            cost_weights={"mask": 0.0},
            loss_weights={"dice": 0.0, "bce": 0.0},
            max_objects=2,
            rank_weight=1.0,
        )
        task = MaskReconstructionTask(config, validity_key="top_valid")
        target = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0]]], dtype=torch.float32)
        targets = {
            "jet_mask_true": target,
            "jet_valid_mask": torch.ones(1, 4, dtype=torch.bool),
            "top_valid": torch.ones(1, 2, dtype=torch.bool),
        }
        good = torch.tensor([[[5.0, -5.0, -5.0, -5.0], [-5.0, 5.0, -5.0, -5.0]]], requires_grad=True)
        bad = torch.tensor([[[-5.0, 5.0, -5.0, -5.0], [5.0, -5.0, -5.0, -5.0]]], requires_grad=True)
        good_loss = task.compute_loss({"mask_predictions": good}, targets)
        bad_loss = task.compute_loss({"mask_predictions": bad}, targets)
        self.assertTrue(torch.isfinite(good_loss))
        self.assertLess(good_loss.item(), bad_loss.item())
        good_loss.backward()

    def test_exact_counts_use_per_type_validity(self):
        trainer = ReconstructionTrainer.__new__(ReconstructionTrainer)
        trainer._val_exact_counts = {}
        trainer._val_collapse_counts = {}

        targets = {
            "jet_valid_mask": torch.ones(1, 4, dtype=torch.bool),
            "jet_mask_true": torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0]]], dtype=torch.float32),
            "jet_mask_true_W": torch.tensor([[[1, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.float32),
            # Top is valid for both slots, W only for slot 0.
            "top_valid": torch.tensor([[True, True]]),
            "w_valid": torch.tensor([[True, False]]),
        }
        outputs = {
            0: {
                "mask_predictions": torch.tensor([[[4, -4, -4, -4], [-4, 4, -4, -4]]], dtype=torch.float32),
                "mask_W": torch.tensor([[[4, -4, -4, -4], [-4, -4, -4, -4]]], dtype=torch.float32),
                "__targets__": targets,
            }
        }

        trainer._accumulate_validation_exact(outputs)
        self.assertEqual([x.item() for x in trainer._val_exact_counts["top_eff"]], [2.0, 2.0])
        self.assertEqual([x.item() for x in trainer._val_exact_counts["W_eff"]], [1.0, 1.0])
        # Only the complete slot contributes to chain efficiency.
        self.assertEqual([x.item() for x in trainer._val_exact_counts["chain_eff"]], [1.0, 1.0])
        self.assertEqual([x.item() for x in trainer._val_exact_counts["ttbar_eff"]], [0.0, 0.0])

    def test_w_dice_stats_use_w_phase_final_layer(self):
        registry = TaskRegistry()
        task = MaskReconstructionTask(
            TaskConfig(
                name="mask_W",
                output_names=["mask_W"],
                output_dims={},
                cost_weights={"mask": 1.0},
                loss_weights={"dice": 1.0, "bce": 0.5},
                max_objects=2,
                layer_weights={1: 0.5, 2: 1.0, 5: 0.0},
                validity_key="w_valid",
            ),
            pred_key="mask_W",
            target_key="jet_mask_true_W",
            validity_key="w_valid",
        )
        registry.register_task(task)
        targets = {
            "jet_mask_true_W": torch.tensor([[[1.0, 0.0]]]),
            "jet_valid_mask": torch.ones(1, 2, dtype=torch.bool),
            "w_valid": torch.ones(1, 1, dtype=torch.bool),
            "obj_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        }
        predictions = {"mask_W": torch.tensor([[[4.0, -4.0]]])}
        registry.compute_total_loss(predictions, targets, layer_id=1, is_final_layer=False)
        self.assertEqual(task._w_dice_count.item(), 0)
        registry.compute_total_loss(predictions, targets, layer_id=2, is_final_layer=False)
        self.assertEqual(task._w_dice_count.item(), 1)

    def test_collapse_metrics_detect_identical_queries_and_padding_bloat(self):
        trainer = ReconstructionTrainer.__new__(ReconstructionTrainer)
        trainer._val_exact_counts = {}
        trainer._val_collapse_counts = {}

        # 3 jets per event; jet 3 is padding.
        jet_valid = torch.tensor([[True, True, True, False]])
        collapsed_top = torch.tensor([[[4.0, 5.0, -4.0, 9.0],
                                       [4.0, 5.0, -4.0, -9.0]]])
        distinct_w = torch.tensor([[[4.0, -4.0, -4.0, -4.0],
                                    [-4.0, 4.0, -4.0, -4.0]]])
        trainer._accumulate_collapse_metrics(collapsed_top, distinct_w, jet_valid)

        identical = trainer._val_collapse_counts['query_identical_top']
        # The two top queries agree on all valid jets (padding excluded).
        self.assertEqual(identical[0].item(), 1.0)
        self.assertEqual(identical[1].item(), 1.0)
        w_identical = trainer._val_collapse_counts['query_identical_w']
        self.assertEqual(w_identical[0].item(), 0.0)

        pad_top = trainer._val_collapse_counts['pad_logit_top']
        # Only the padding jet contributes: query 0 logit 9, query 1 -9 -> mean 0.
        self.assertEqual(pad_top[1].item(), 2.0)
        self.assertAlmostEqual(pad_top[0].item(), 0.0)

    def test_log_collapse_metrics_skips_empty_counts(self):
        trainer = ReconstructionTrainer.__new__(ReconstructionTrainer)
        trainer._val_collapse_counts = {}
        with mock.patch.object(ReconstructionTrainer, '_sync_dist',
                               new_callable=mock.PropertyMock, return_value=False), \
                mock.patch.object(ReconstructionTrainer, 'log') as log_mock:
            trainer._log_collapse_metrics()
            log_mock.assert_not_called()

            trainer._val_collapse_counts = {
                'query_identical_top': [torch.tensor(0.25), torch.tensor(4.0)],
                'pad_logit_top': [torch.tensor(-21.0), torch.tensor(4.0)],
            }
            trainer._log_collapse_metrics()
        logged = {call.args[0]: call.args[1].item() for call in log_mock.call_args_list}
        self.assertAlmostEqual(logged['val_query_identical_top'], 0.0625)
        self.assertAlmostEqual(logged['val_pad_logit_top'], -5.25)


if __name__ == "__main__":
    unittest.main()
