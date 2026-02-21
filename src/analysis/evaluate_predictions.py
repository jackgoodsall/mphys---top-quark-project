"""
Post-run evaluation script.

Reads HDF5 test outputs saved by the task registry during trainer.test()
and computes metrics: objectness precision/recall/F1, type accuracy, mask IoU.

Usage:
    uv run src/analysis/evaluate_predictions.py lightning_logs/version_0
"""
import argparse
import h5py
import numpy as np
from pathlib import Path


def evaluate_objectness(h5_path: Path) -> dict:
    """Compute precision, recall, F1 for objectness predictions."""
    with h5py.File(h5_path, 'r') as f:
        logits = f['predicted_objectness_logit'][:]   # [N, M]
        targets = f['target_objectness'][:]           # [N, M]

    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs > 0.5).astype(int)
    tgts = targets.astype(int).flatten()
    preds_flat = preds.flatten()

    tp = int(((preds_flat == 1) & (tgts == 1)).sum())
    fp = int(((preds_flat == 1) & (tgts == 0)).sum())
    fn = int(((preds_flat == 0) & (tgts == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'objectness_precision': precision,
        'objectness_recall': recall,
        'objectness_f1': f1,
        'objectness_tp': tp,
        'objectness_fp': fp,
        'objectness_fn': fn,
    }


def evaluate_type(h5_path: Path) -> dict:
    """Compute binary accuracy for top-vs-W type predictions on real objects."""
    with h5py.File(h5_path, 'r') as f:
        logits = f['predicted_type_logit'][:]   # [N, M]
        targets = f['target_type'][:]           # [N, M]
        classes = f['target_classes'][:]        # [N, M]

    # Only evaluate on real objects (class != 0)
    real_mask = classes > 0
    if not real_mask.any():
        return {'type_accuracy': 0.0, 'type_n_real': 0}

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(float)

    real_preds = preds[real_mask]
    real_targets = targets[real_mask]

    accuracy = float((real_preds == real_targets).mean())
    return {
        'type_accuracy': accuracy,
        'type_n_real': int(real_mask.sum()),
    }


def evaluate_masks(h5_path: Path) -> dict:
    """Compute mean IoU between predicted and target masks."""
    with h5py.File(h5_path, 'r') as f:
        pred_logits = f['predicted_masks_logits'][:]   # [N, M, P]
        targets = f['target_masks'][:]                 # [N, M, P]

    pred_bin = (pred_logits > 0).astype(float)
    intersection = (pred_bin * targets).sum(axis=-1)       # [N, M]
    union = np.clip(pred_bin + targets, 0, 1).sum(axis=-1) # [N, M]

    # Only compute IoU where there are actual target particles
    has_particles = targets.sum(axis=-1) > 0  # [N, M]
    if not has_particles.any():
        return {'mask_mean_iou': 0.0}

    iou = intersection[has_particles] / (union[has_particles] + 1e-8)
    return {
        'mask_mean_iou': float(iou.mean()),
        'mask_n_objects': int(has_particles.sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate test predictions from a Lightning run"
    )
    parser.add_argument('log_dir', help='Path to lightning_logs/version_N/')
    args = parser.parse_args()
    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"Directory not found: {log_dir}")
        return

    results = {}

    # Map task name -> (h5 filename, evaluation function)
    evaluators = {
        'objectness':  ('test_outputs_objectness.h5',  evaluate_objectness),
        'object_type': ('test_outputs_object_type.h5', evaluate_type),
        'mask':        ('test_outputs_mask.h5',        evaluate_masks),
    }

    for task_name, (h5_name, fn) in evaluators.items():
        h5_path = log_dir / h5_name
        if h5_path.exists():
            task_results = fn(h5_path)
            results.update(task_results)
            print(f"  [{task_name}] evaluated {h5_path.name}")
        else:
            print(f"  [{task_name}] skipped — {h5_name} not found")

    print("\n=== Evaluation Summary ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    out = log_dir / 'evaluation_summary.txt'
    with open(out, 'w') as f:
        for k, v in results.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
