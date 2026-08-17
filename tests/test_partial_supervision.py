import unittest

import torch

from src.models.components.masked_former_tasks import (
    ChainStateTask,
    ExclusiveAssignmentTask,
    MaskReconstructionTask,
    TaskConfig,
)


def mask_task(name, pred_key, target_key, validity):
    return MaskReconstructionTask(
        TaskConfig(
            name=name, output_names=[pred_key], output_dims={}, cost_weights={"mask": 1.0},
            loss_weights={"dice": 1.0, "bce": .5}, max_objects=2, validity_key=validity,
        ),
        pred_key=pred_key, target_key=target_key, validity_key=validity,
        null_mask_penalty=1.0,
    )


class PartialSupervisionTest(unittest.TestCase):
    def setUp(self):
        self.targets = {
            "jet_mask_true": torch.zeros(1, 2, 4),
            "jet_mask_true_W": torch.tensor([[[1., 1, 0, 0], [0, 0, 0, 0]]]),
            "jet_valid_mask": torch.ones(1, 4, dtype=torch.bool),
            "obj_valid_mask": torch.tensor([[True, False]]),
            "top_valid": torch.tensor([[False, False]]),
            "w_valid": torch.tensor([[True, False]]),
        }

    def test_w_only_has_w_gradient_and_exactly_zero_top_gradient(self):
        top_logits = torch.randn(1, 2, 4, requires_grad=True)
        w_logits = torch.randn(1, 2, 4, requires_grad=True)
        top_loss = mask_task("mask", "mask_predictions", "jet_mask_true", "top_valid").compute_loss(
            {"mask_predictions": top_logits}, self.targets
        )
        w_loss = mask_task("mask_W", "mask_W", "jet_mask_true_W", "w_valid").compute_loss(
            {"mask_W": w_logits}, self.targets
        )
        (top_loss + w_loss).backward()
        self.assertEqual(torch.count_nonzero(top_logits.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(w_logits.grad).item(), 0)

    def test_chain_state_vocab_and_loss(self):
        task = ChainStateTask(TaskConfig(
            name="chain_state", output_names=["chain_state_logits"],
            output_dims={"chain_state_logits": 3}, cost_weights={"chain_state": 1.0},
            loss_weights={"chain_state": 1.0}, max_objects=2,
        ))
        self.assertEqual(task.target_states(self.targets).tolist(), [[1, 0]])
        logits = torch.tensor([[[0., 4., 0.], [4., 0., 0.]]], requires_grad=True)
        loss = task.compute_loss({"chain_state_logits": logits}, self.targets)
        self.assertLess(loss.item(), .1)
        loss.backward()

    def test_impossible_top_without_w_is_rejected(self):
        bad = dict(self.targets)
        bad["top_valid"] = torch.tensor([[True, False]])
        bad["w_valid"] = torch.tensor([[False, False]])
        with self.assertRaisesRegex(ValueError, "cannot be valid"):
            ChainStateTask.target_states(bad)

    def test_exclusive_top_ce_has_no_gradient_for_censored_query(self):
        task = ExclusiveAssignmentTask(
            TaskConfig(
                name="exclusive", output_names=["mask_predictions"], output_dims={},
                cost_weights={}, loss_weights={"exclusive": 1.0}, max_objects=2,
            ), validity_key="top_valid",
        )
        logits = torch.randn(1, 2, 4, requires_grad=True)
        loss = task.compute_loss({"mask_predictions": logits}, self.targets, self.targets["jet_valid_mask"])
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()
