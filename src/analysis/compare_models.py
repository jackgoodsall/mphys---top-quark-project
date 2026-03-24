"""
Compare reconstruction efficiency across models (chain and non-chain).

Produces a tabulated comparison of top/W/ttbar efficiency in bins
6 jets, 7 jets, >=8 jets, and inclusive (All).

Auto-detects whether each run is chain_queries or standard (type-based)
by checking for test_outputs_mask_W.h5 (chain) vs test_outputs_object_type.h5.

Usage:
    python analysis/compare_models.py \\
        --runs "Model A"=/path/to/version_X "Model B"=/path/to/version_Y \\
        --data_file /path/to/test_data.h5

    python analysis/compare_models.py \\
        --runs baseline=lightning_logs/version_100 new=lightning_logs/version_200 \\
        --prior top=3 W=2 --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import evaluate_chain
import evaluate as evaluate_std


def _agg_bin(br: dict, mult_keys):
    """Aggregate efficiency across a set of multiplicity keys using weighted average."""
    n_top = sum(br[m]["n_top"]   for m in mult_keys if m in br)
    n_W   = sum(br[m]["n_W"]     for m in mult_keys if m in br)
    n_tt  = sum(br[m]["n_ttbar"] for m in mult_keys if m in br)
    top_eff   = sum(br[m]["top_eff"]   * br[m]["n_top"]   for m in mult_keys if m in br) / max(n_top, 1)
    W_eff     = sum(br[m]["W_eff"]     * br[m]["n_W"]     for m in mult_keys if m in br) / max(n_W,   1)
    ttbar_eff = sum(br[m]["ttbar_eff"] * br[m]["n_ttbar"] for m in mult_keys if m in br) / max(n_tt,  1)
    n_events  = sum(br[m]["n_events"]  for m in mult_keys if m in br)
    return {"top_eff": top_eff, "W_eff": W_eff, "ttbar_eff": ttbar_eff, "n_events": n_events}


def _detect_run_type(run_dir: Path) -> str:
    """Detect whether a run is chain_queries or standard."""
    has_mask_W = (run_dir / "test_outputs_mask_W.h5").exists()
    has_type   = (run_dir / "test_outputs_object_type.h5").exists()
    if has_mask_W:
        return "chain"
    if has_type:
        return "standard"
    # Fallback: if only mask + objectness, assume chain
    if (run_dir / "test_outputs_mask.h5").exists():
        return "chain"
    sys.exit(f"ERROR: cannot detect run type for {run_dir} — no recognised output files found")


def _parse_runs(run_args):
    """Parse NAME=PATH pairs from --runs."""
    runs = []
    for item in run_args:
        if "=" not in item:
            p = Path(item).resolve()
            runs.append((p.name, p))
        else:
            name, path_str = item.split("=", 1)
            runs.append((name, Path(path_str).resolve()))
    return runs


def _load_and_evaluate_chain(run_dir, data_file, use_probs, threshold,
                              prior_top, prior_W, strict, joint,
                              require_top_for_w=False):
    """Load and evaluate a chain_queries run."""
    (pred_scores_top, target_masks_top, pred_scores_W, target_masks_W,
     jet_valid, target_obj_top, pred_obj_top, target_obj_W, pred_obj_W,
     original_mult) = evaluate_chain.load_run_data(run_dir, data_file, use_probs=use_probs)

    return evaluate_chain.compute_efficiencies(
        pred_scores_top, target_masks_top, pred_scores_W, target_masks_W,
        jet_valid, target_obj_top,
        pred_obj_top=pred_obj_top, target_obj_W=target_obj_W,
        pred_obj_W=pred_obj_W, prior_top=prior_top, prior_W=prior_W,
        use_probs=use_probs, threshold=threshold,
        strict=strict, joint=joint, original_mult=original_mult,
        require_top_for_w=require_top_for_w,
    )


def _load_and_evaluate_std(run_dir, data_file, use_probs, threshold,
                            prior_top, prior_W, strict):
    """Load and evaluate a standard (type-based) run."""
    (pred_scores, target_masks, jet_valid, target_obj, target_cls,
     pred_obj, pred_type, original_mult) = evaluate_std.load_run_data(
        run_dir, data_file, use_probs=use_probs
    )

    # Convert prior_top/prior_W to the priors dict format used by evaluate.py
    priors = {}
    if prior_top is not None:
        priors[evaluate_std.TYPE_IDS["top"]] = prior_top
    if prior_W is not None:
        priors[evaluate_std.TYPE_IDS["W"]] = prior_W

    return evaluate_std.compute_efficiencies(
        pred_scores, target_masks, jet_valid, target_obj, target_cls,
        pred_obj=pred_obj, pred_type=pred_type, priors=priors or None,
        use_probs=use_probs, threshold=threshold,
        strict=strict, original_mult=original_mult,
    )


def build_comparison_table(all_results):
    """
    Build a list of rows for the comparison table.

    Returns list of dicts with keys: model, type, metric, 6, 7, 8+, All
    """
    metrics = [
        ("Top eff",   "top_eff"),
        ("W eff",     "W_eff"),
        ("ttbar eff", "ttbar_eff"),
    ]

    rows = []
    for model_name, run_type, results in all_results:
        br = results["breakdown"]
        all_mults = sorted(br.keys())
        ge8_keys = [m for m in all_mults if m >= 8]

        bins = {
            6:     _agg_bin(br, [6]),
            7:     _agg_bin(br, [7]),
            "8+":  _agg_bin(br, ge8_keys),
            "All": {
                "top_eff":   results["top_eff"],
                "W_eff":     results["W_eff"],
                "ttbar_eff": results["ttbar_eff"],
                "n_events":  results["N"],
            },
        }

        for metric_label, metric_key in metrics:
            row = {
                "model": model_name,
                "type": run_type,
                "metric": metric_label,
            }
            for bin_key in [6, 7, "8+", "All"]:
                row[bin_key] = bins[bin_key][metric_key]
            rows.append(row)

    return rows


def print_table(rows):
    """Print a formatted comparison table."""
    bin_labels = [6, 7, "8+", "All"]
    bin_headers = ["6 jets", "7 jets", ">=8 jets", "All"]

    # Find column widths
    model_w = max(len(r["model"]) for r in rows)
    type_w = max(len(r["type"]) for r in rows)
    metric_w = max(len(r["metric"]) for r in rows)
    val_w = 9

    # Header
    header = f"  {'Model':<{model_w}}   {'Type':<{type_w}}   {'Metric':<{metric_w}}"
    for bh in bin_headers:
        header += f"   {bh:>{val_w}}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    # Rows, grouped by model with separator lines
    prev_model = None
    for r in rows:
        if prev_model is not None and r["model"] != prev_model:
            print("  " + "─" * (len(header) - 2))
        prev_model = r["model"]

        line = f"  {r['model']:<{model_w}}   {r['type']:<{type_w}}   {r['metric']:<{metric_w}}"
        for bk in bin_labels:
            line += f"   {r[bk]*100:>{val_w-1}.2f}%"
        print(line)

    print()


def make_comparison_plot(all_results, output_path: Path):
    """Bar chart comparing models across bins."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available — skipping plot.")
        return

    metrics = [
        ("Top eff",   "top_eff",   "#4c72b0"),
        ("W eff",     "W_eff",     "#dd8452"),
        ("ttbar eff", "ttbar_eff", "#55a868"),
    ]
    bin_labels = ["6 jets", "7 jets", ">=8 jets", "All"]
    bin_keys = [6, 7, "8+", "All"]

    n_models = len(all_results)
    n_metrics = len(metrics)
    n_bins = len(bin_labels)

    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)

    for ax, (metric_label, metric_key, base_color) in zip(axes[0], metrics):
        x = np.arange(n_bins)
        bar_width = 0.8 / n_models

        for i, (model_name, run_type, results) in enumerate(all_results):
            br = results["breakdown"]
            all_mults = sorted(br.keys())
            ge8_keys = [m for m in all_mults if m >= 8]
            bins = {
                6:     _agg_bin(br, [6]),
                7:     _agg_bin(br, [7]),
                "8+":  _agg_bin(br, ge8_keys),
                "All": {"top_eff": results["top_eff"], "W_eff": results["W_eff"],
                         "ttbar_eff": results["ttbar_eff"]},
            }

            label = f"{model_name} ({run_type})"
            offsets = x + (i - (n_models - 1) / 2) * bar_width
            vals = [bins[bk][metric_key] * 100 for bk in bin_keys]
            bars = ax.bar(offsets, vals, width=bar_width, label=label, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.set_ylabel("Efficiency (%)")
        ax.set_title(metric_label)
        ax.set_ylim(0, 115)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Comparison: Reconstruction Efficiency", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_table_image(rows, output_path: Path, binarisation: str = ""):
    """Render the comparison table as a blocky, high-contrast image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
    except ImportError:
        print("WARNING: matplotlib not available — skipping table image.")
        return

    bin_labels = [6, 7, "8+", "All"]
    col_headers = ["Model", "Metric", "6 jets", "7 jets", ">=8 jets", "All"]
    n_cols = len(col_headers)

    # Build cell text and track model grouping
    cell_text = []
    model_indices = []
    model_names_seen = []
    for r in rows:
        model_name = r["model"]
        if model_name not in model_names_seen:
            model_names_seen.append(model_name)
        model_indices.append(model_names_seen.index(model_name))
        cell_text.append([
            model_name,
            r["metric"],
        ] + [f"{r[bk]*100:.2f}%" for bk in bin_labels])

    n_rows = len(cell_text)
    fig_width = 11
    fig_height = 1.6 + n_rows * 0.55

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    # Strong model-group colours
    model_colors = [
        ("#d6eaf8", "#2c3e50"),  # light blue bg, dark text
        ("#fadbd8", "#78281f"),  # light red bg, dark red text
        ("#d5f5e3", "#1e8449"),  # light green bg, dark green text
        ("#fdebd0", "#784212"),  # light orange bg, dark orange text
    ]

    # Style header - thick, dark
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#1a252f")
        cell.set_text_props(color="white", fontweight="bold", fontsize=12)
        cell.set_edgecolor("white")
        cell.set_linewidth(2.5)

    # Style data rows - bold borders, strong alternating colours per model
    for i in range(n_rows):
        midx = model_indices[i]
        bg, fg = model_colors[midx % len(model_colors)]
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("white")
            cell.set_linewidth(2.5)
            if j >= 2:  # value columns
                cell.set_text_props(fontweight="bold", fontsize=11, color=fg)
            else:
                cell.set_text_props(fontweight="bold", fontsize=11, color="#2c3e50")

    # Find best value per metric+bin and highlight it
    from collections import defaultdict
    metric_bins = defaultdict(list)
    for idx, r in enumerate(rows):
        for bk in bin_labels:
            metric_bins[(r["metric"], bk)].append((r[bk], idx))

    for (metric, bk), vals in metric_bins.items():
        best_val = max(v for v, _ in vals)
        col_j = bin_labels.index(bk) + 2
        for v, row_idx in vals:
            if v == best_val and len(set(v for v, _ in vals)) > 1:
                cell = table[row_idx + 1, col_j]
                midx = model_indices[row_idx]
                _, fg = model_colors[midx % len(model_colors)]
                cell.set_text_props(fontweight="bold", fontsize=12, color=fg,
                                    fontstyle="normal")
                # Slightly darker bg to highlight winner
                from matplotlib.colors import to_rgb
                r_c, g_c, b_c = to_rgb(model_colors[midx % len(model_colors)][0])
                cell.set_facecolor((r_c * 0.85, g_c * 0.85, b_c * 0.85))

    title = "Model Comparison: Reconstruction Efficiency"
    if binarisation:
        title += f"\n(binarisation: {binarisation})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare reconstruction efficiency across models (chain and non-chain)."
    )
    parser.add_argument(
        "--runs", required=True, nargs="+", metavar="[NAME=]PATH",
        help="Model runs to compare. Format: 'Label=/path/to/version_X' or just '/path/to/version_X'"
    )
    parser.add_argument(
        "--data_file", default=None, type=Path,
        help="HDF5 data file with 'src_mask' (if not embedded in outputs)"
    )
    parser.add_argument(
        "--use_probs", action="store_true",
        help="Use sigmoid probabilities instead of raw logits"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Binarisation threshold (overrides default)"
    )
    parser.add_argument(
        "--prior", nargs="+", metavar="TYPE=K", default=[],
        help="Per-type top-k prior, e.g. --prior top=3 W=2"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Require objectness prediction to mark slot as real"
    )
    parser.add_argument(
        "--joint", action="store_true",
        help="(Chain only) Top correct only if associated W is also correct"
    )
    parser.add_argument(
        "--require_top_for_w", action="store_true",
        help="Only count a chain as real if the top target also has particles (no orphan Ws)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save comparison bar chart"
    )
    parser.add_argument(
        "--table", action="store_true",
        help="Save comparison table as an image"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for plot/table (default: comparison_*.png in first run dir)"
    )
    args = parser.parse_args()

    # Parse priors
    prior_top, prior_W = None, None
    for item in args.prior:
        if "=" not in item:
            sys.exit(f"ERROR: --prior entries must be TYPE=K, got '{item}'")
        name, k_str = item.split("=", 1)
        if name not in ("top", "W"):
            sys.exit(f"ERROR: unknown type '{name}' in --prior. Valid: top, W")
        try:
            k = int(k_str)
        except ValueError:
            sys.exit(f"ERROR: K must be an integer, got '{k_str}'")
        if name == "top":
            prior_top = k
        else:
            prior_W = k

    runs = _parse_runs(args.runs)
    for name, path in runs:
        if not path.is_dir():
            sys.exit(f"ERROR: run directory not found: {path}")

    # Compute efficiencies for each model, auto-detecting run type
    all_results = []
    for model_name, run_dir in runs:
        run_type = _detect_run_type(run_dir)
        print(f"Loading {model_name} ({run_dir}) [detected: {run_type}]...")

        if run_type == "chain":
            results = _load_and_evaluate_chain(
                run_dir, args.data_file, args.use_probs, args.threshold,
                prior_top, prior_W, args.strict, args.joint,
                require_top_for_w=args.require_top_for_w,
            )
        else:
            results = _load_and_evaluate_std(
                run_dir, args.data_file, args.use_probs, args.threshold,
                prior_top, prior_W, args.strict,
            )

        all_results.append((model_name, run_type, results))

    # Print comparison
    prior_parts = []
    if prior_top is not None:
        prior_parts.append(f"top={prior_top}")
    if prior_W is not None:
        prior_parts.append(f"W={prior_W}")
    binarisation = ", ".join(prior_parts) if prior_parts else f"threshold={args.threshold if args.threshold is not None else ('0.5' if args.use_probs else '0.0')}"

    print(f"\n=== Model Comparison ===")
    print(f"Binarisation: {binarisation}")
    if args.strict:
        print(f"Mode: strict (objectness gated)")
    if args.joint:
        print(f"Mode: joint (top requires W correct, chain only)")
    print()

    rows = build_comparison_table(all_results)
    print_table(rows)

    if args.plot:
        output_path = args.output or runs[0][1] / "comparison_efficiency.png"
        make_comparison_plot(all_results, output_path)

    if args.table:
        table_path = args.output or runs[0][1] / "comparison_table.png"
        make_table_image(rows, table_path, binarisation)


if __name__ == "__main__":
    main()
