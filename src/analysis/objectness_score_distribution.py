"""
Objectness score distribution plot.

Loads predicted objectness logits and truth labels from the HDF5 test output,
applies sigmoid, and plots the score distributions for true=0 and true=1 separately
to help identify an optimal cut threshold.

Usage:
    uv run src/analysis/objectness_score_distribution.py lightning_logs/version_0
    uv run src/analysis/objectness_score_distribution.py lightning_logs/version_0 --bins 100
    uv run src/analysis/objectness_score_distribution.py lightning_logs/version_0 --no-valid-filter
"""
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def compute_roc_stats(scores_neg, scores_pos):
    """Compute precision, recall, F1, and TPR/FPR across thresholds."""
    all_scores = np.concatenate([scores_neg, scores_pos])
    thresholds = np.linspace(0.0, 1.0, 201)

    tpr_list, fpr_list, f1_list, prec_list = [], [], [], []
    for t in thresholds:
        tp = (scores_pos >= t).sum()
        fn = (scores_pos < t).sum()
        fp = (scores_neg >= t).sum()
        tn = (scores_neg < t).sum()

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * tpr / max(prec + tpr, 1e-8)

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        f1_list.append(f1)
        prec_list.append(prec)

    return thresholds, np.array(tpr_list), np.array(fpr_list), np.array(f1_list), np.array(prec_list)


def plot_objectness_distribution(h5_path: Path, bins: int = 80, valid_filter: bool = True, out_dir: Path = None):
    with h5py.File(h5_path, 'r') as f:
        logits  = f['predicted_objectness_logit'][:]  # [N, Q]
        targets = f['target_objectness'][:]           # [N, Q]
        # obj_valid_mask marks real query slots (excludes padding from null events)
        valid_mask = f['obj_valid_mask'][:] if ('obj_valid_mask' in f and valid_filter) else None

    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid → [0, 1]

    flat_probs   = probs.flatten()
    flat_targets = targets.flatten().astype(int)

    if valid_mask is not None:
        flat_valid = valid_mask.flatten().astype(bool)
        flat_probs   = flat_probs[flat_valid]
        flat_targets = flat_targets[flat_valid]
        n_total = flat_valid.sum()
        print(f"Valid query slots: {n_total:,} / {valid_mask.size:,}")
    else:
        n_total = flat_probs.size

    scores_pos = flat_probs[flat_targets == 1]
    scores_neg = flat_probs[flat_targets == 0]

    print(f"  True positives (label=1): {len(scores_pos):,}")
    print(f"  True negatives (label=0): {len(scores_neg):,}")

    thresholds, tpr, fpr, f1, prec = compute_roc_stats(scores_neg, scores_pos)
    best_idx  = np.argmax(f1)
    best_cut  = thresholds[best_idx]
    best_f1   = f1[best_idx]
    best_tpr  = tpr[best_idx]
    best_prec = prec[best_idx]

    print(f"\n  Best F1={best_f1:.4f} at cut={best_cut:.3f}  "
          f"(precision={best_prec:.4f}, recall/TPR={best_tpr:.4f})")

    # ------------------------------------------------------------------ #
    # Figure 1: Score distributions
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bin_edges = np.linspace(0, 1, bins + 1)

    # --- Normalised density (area=1) ---
    ax = axes[0]
    ax.hist(scores_pos, bins=bin_edges, density=True, alpha=0.65,
            color='steelblue', label=f'True real (label=1)  n={len(scores_pos):,}')
    ax.hist(scores_neg, bins=bin_edges, density=True, alpha=0.65,
            color='tomato',   label=f'True null (label=0)  n={len(scores_neg):,}')
    ax.axvline(best_cut, color='black', lw=1.5, linestyle='--',
               label=f'Best F1 cut = {best_cut:.3f}')
    ax.axvline(0.5, color='grey', lw=1.0, linestyle=':',
               label='Default cut = 0.5')
    ax.set_xlabel('Objectness score (sigmoid)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Objectness Score Distribution', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Log-scale version ---
    ax = axes[1]
    ax.hist(scores_pos, bins=bin_edges, density=True, alpha=0.65,
            color='steelblue', label=f'True real (label=1)  n={len(scores_pos):,}')
    ax.hist(scores_neg, bins=bin_edges, density=True, alpha=0.65,
            color='tomato',   label=f'True null (label=0)  n={len(scores_neg):,}')
    ax.axvline(best_cut, color='black', lw=1.5, linestyle='--',
               label=f'Best F1 cut = {best_cut:.3f}')
    ax.axvline(0.5, color='grey', lw=1.0, linestyle=':',
               label='Default cut = 0.5')
    ax.set_yscale('log')
    ax.set_xlabel('Objectness score (sigmoid)', fontsize=12)
    ax.set_ylabel('Density (log)', fontsize=12)
    ax.set_title('Objectness Score Distribution (log scale)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    # ------------------------------------------------------------------ #
    # Figure 2: Precision / Recall / F1 vs threshold
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(thresholds, tpr,  color='steelblue', label='Recall (TPR)')
    ax2.plot(thresholds, prec, color='darkorange', label='Precision')
    ax2.plot(thresholds, f1,   color='green',     label='F1')
    ax2.plot(thresholds, fpr,  color='tomato',    linestyle='--', label='FPR')
    ax2.axvline(best_cut, color='black', lw=1.5, linestyle='--',
                label=f'Best F1 cut = {best_cut:.3f}')
    ax2.axvline(0.5, color='grey', lw=1.0, linestyle=':',
                label='Default cut = 0.5')
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Precision / Recall / F1 vs Objectness Threshold', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    fig2.tight_layout()

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    if out_dir is None:
        out_dir = h5_path.parent

    p1 = out_dir / 'objectness_score_distribution.png'
    p2 = out_dir / 'objectness_threshold_scan.png'
    fig.savefig(p1, dpi=150)
    fig2.savefig(p2, dpi=150)
    plt.close('all')
    print(f"\nSaved:\n  {p1}\n  {p2}")

    return best_cut, best_f1


def main():
    parser = argparse.ArgumentParser(
        description="Plot objectness score distributions and threshold scan"
    )
    parser.add_argument('log_dir', help='Path to lightning_logs/version_N/')
    parser.add_argument('--bins', type=int, default=80,
                        help='Number of histogram bins (default: 80)')
    parser.add_argument('--no-valid-filter', action='store_true',
                        help='Do not filter by obj_valid_mask even if present')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    h5_path = log_dir / 'test_outputs_objectness.h5'

    if not h5_path.exists():
        print(f"HDF5 not found: {h5_path}")
        print("Run trainer.test() first to generate test outputs.")
        return

    print(f"Loading {h5_path}")
    plot_objectness_distribution(
        h5_path,
        bins=args.bins,
        valid_filter=not args.no_valid_filter,
        out_dir=log_dir,
    )


if __name__ == '__main__':
    main()
