"""
Report-ready evaluation plots for chain_queries models.

Generates two publication-quality figures:
  1. Efficiency vs jet multiplicity   → report_eff_vs_mult.pdf / .png
  2. Efficiency–purity trade-off      → report_eff_vs_purity.pdf / .png
     (parametric curve sweeping the objectness threshold)

Usage:
    python src/analysis/report_plots.py --run_dir lightning_logs/version_X
    python src/analysis/report_plots.py --run_dir lightning_logs/version_X --use_probs
    python src/analysis/report_plots.py --run_dir lightning_logs/version_X \\
        --data_file /path/to/test.h5 --out_dir plots/

Options that affect which events are counted:
    --use_probs            use sigmoid probabilities instead of raw logits
    --threshold T          mask binarisation threshold (default 0.0 logits / 0.5 probs)
    --obj_threshold T      objectness threshold for strict/purity metrics (default 0.5 prob)
    --prior top=K [W=K]    top-k particle prior instead of thresholding
    --require_complete_truth   only count real tops/Ws with full particle multiplicity
    --require_top_for_w        only count a W as real when its parent top is also real
    --joint                count top as correct only if its W is also correct

Sweep options:
    --sweep_steps N        number of objectness threshold steps (default: 200)
    --no_pdf               save only PNG (skip PDF)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ── matplotlib style ─────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Publication-quality defaults
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "lines.linewidth":    1.8,
    "lines.markersize":   6,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# Colourblind-friendly palette (Wong 2011)
C_TOP   = "#0072B2"   # blue
C_W     = "#E69F00"   # amber
C_TTBAR = "#009E73"   # green

# ── reuse evaluate_chain logic ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_chain import (
    load_run_data,
    compute_efficiencies,
    sweep_obj_threshold,
    binarise_predictions,
    _f1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Efficiency vs jet multiplicity
# ─────────────────────────────────────────────────────────────────────────────

def plot_eff_vs_multiplicity(results: dict, out_dir: Path, save_pdf: bool = True):
    """
    Line plot: Top / W / ttbar efficiency as a function of signal jet multiplicity.
    One panel per multiplicity type (signal jets / original jets) side by side when
    both are available.
    """
    br      = results.get("breakdown", {})
    br_orig = results.get("breakdown_orig", {})

    panels = []
    if br:
        panels.append((br,      "Number of signal jets fed to model",  "signal"))
    if br_orig:
        panels.append((br_orig, "Number of jets in full event",         "original"))

    if not panels:
        print("  No per-multiplicity breakdown available — skipping multiplicity plot.")
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.2), squeeze=False)

    for col, (table, xlabel, _tag) in enumerate(panels):
        ax   = axes[0, col]
        mults = sorted(table.keys())
        xs    = np.array(mults)

        top_effs   = np.array([table[m]["top_eff"]   * 100 for m in mults])
        W_effs     = np.array([table[m]["W_eff"]     * 100 for m in mults])
        ttbar_effs = np.array([table[m]["ttbar_eff"] * 100 for m in mults])
        counts     = np.array([table[m]["n_events"]         for m in mults])

        ax.plot(xs, top_effs,   "o-", color=C_TOP,   label=r"Top ($t$)",         zorder=3)
        ax.plot(xs, W_effs,     "s-", color=C_W,     label=r"$W$ boson",         zorder=3)
        ax.plot(xs, ttbar_effs, "^-", color=C_TTBAR, label=r"$t\bar{t}$ (both tops)", zorder=3)

        ax.set_xticks(xs)
        ax.set_xticklabels([str(m) for m in mults])

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Efficiency (%)")
        ax.set_ylim(0, 102)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        ax.legend(loc="upper right")

    fig.tight_layout()

    _save(fig, out_dir, "report_eff_vs_mult", save_pdf)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Efficiency–purity trade-off (parametric objectness sweep)
# ─────────────────────────────────────────────────────────────────────────────

def plot_eff_vs_purity(sweep: dict, out_dir: Path, save_pdf: bool = True,
                       n_annotate: int = 5):
    """
    Parametric efficiency–purity (recall–precision) curve sweeping the objectness
    threshold.  Each of Top, W, and ttbar is a separate line.  A subset of
    threshold values are annotated as dots with labels.
    """
    thresholds   = sweep["thresholds"]
    n_thresholds = len(thresholds)

    # Indices to annotate (spread evenly, always include best F1)
    best_top_idx   = int(np.argmax(sweep["top_f1"]))
    best_W_idx     = int(np.argmax(sweep["W_f1"]))
    best_ttbar_idx = int(np.argmax(sweep["ttbar_f1"]))

    # Evenly-spaced annotation indices + the best-F1 points
    step = max(1, n_thresholds // (n_annotate - 1))
    base_annot = list(range(0, n_thresholds, step))
    # Make sure last point is included
    if n_thresholds - 1 not in base_annot:
        base_annot.append(n_thresholds - 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    configs = [
        ("top",   r"Top ($t$)",             C_TOP,   best_top_idx),
        ("W",     r"$W$ boson",             C_W,     best_W_idx),
        ("ttbar", r"$t\bar{t}$ (both tops)", C_TTBAR, best_ttbar_idx),
    ]

    for ax, (prefix, label, color, best_idx) in zip(axes, configs):
        eff = sweep[f"{prefix}_eff"]   * 100
        pur = sweep[f"{prefix}_purity"] * 100
        f1  = sweep[f"{prefix}_f1"]    * 100

        # Main parametric curve
        ax.plot(pur, eff, "-", color=color, lw=2.0, label=label, zorder=2)

        # Direction arrow: show that threshold increases left → right
        # (higher threshold → fewer predicted real → higher purity, lower efficiency)
        mid = n_thresholds // 2
        ax.annotate(
            "", xy=(pur[mid + 3], eff[mid + 3]),
            xytext=(pur[mid], eff[mid]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
            zorder=3,
        )

        # Annotate selected threshold values
        annot_idxs = sorted(set(base_annot) | {best_idx})
        for idx in annot_idxs:
            t_val = float(thresholds[idx])
            ax.plot(pur[idx], eff[idx], "o", color=color, ms=5, zorder=4)
            # Alternate label positions to reduce overlap
            va = "bottom" if idx % 2 == 0 else "top"
            offset = (3, 4) if va == "bottom" else (3, -4)
            ax.annotate(
                f"{t_val:.2f}",
                xy=(pur[idx], eff[idx]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.5,
                color="dimgrey",
                va=va,
            )

        # Mark best-F1 point with a star
        ax.plot(pur[best_idx], eff[best_idx], "*", color="black", ms=10, zorder=5,
                label=f"Best F1 = {f1[best_idx]:.1f}%\n(thr = {thresholds[best_idx]:.2f})")

        ax.set_xlabel("Purity (precision, %)")
        ax.set_ylabel("Efficiency (recall, %)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _save(fig, out_dir, "report_eff_vs_purity", save_pdf)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Efficiency AND purity vs multiplicity (combined, 2-panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_eff_and_purity_vs_mult(results: dict, out_dir: Path, save_pdf: bool = True):
    """
    Two-panel figure: left = efficiency vs multiplicity, right = purity vs multiplicity.
    Only generated when purity data is available in the breakdown dict.
    """
    br = results.get("breakdown", {})
    if not br:
        return

    first = next(iter(br.values()))
    if "top_purity" not in first:
        return   # no purity data

    mults = sorted(br.keys())
    xs    = np.array(mults)
    counts = np.array([br[m]["n_events"] for m in mults])

    top_effs   = np.array([br[m]["top_eff"]   * 100 for m in mults])
    W_effs     = np.array([br[m]["W_eff"]     * 100 for m in mults])
    ttbar_effs = np.array([br[m]["ttbar_eff"] * 100 for m in mults])

    top_purs   = np.array([br[m]["top_purity"]   * 100 for m in mults])
    W_purs     = np.array([br[m]["W_purity"]     * 100 for m in mults])
    ttbar_purs = np.array([br[m]["ttbar_purity"] * 100 for m in mults])

    fig, (ax_eff, ax_pur) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    for ax, data_triples, ylabel in [
        (ax_eff,
         [(top_effs, "o-", r"Top ($t$)"), (W_effs, "s-", r"$W$ boson"), (ttbar_effs, "^-", r"$t\bar{t}$")],
         "Efficiency (%)"),
        (ax_pur,
         [(top_purs, "o-", r"Top ($t$)"), (W_purs, "s-", r"$W$ boson"), (ttbar_purs, "^-", r"$t\bar{t}$")],
         "Purity (%)"),
    ]:
        colors = [C_TOP, C_W, C_TTBAR]
        for (vals, fmt, lbl), col in zip(data_triples, colors):
            ax.plot(xs, vals, fmt, color=col, label=lbl, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(m) for m in mults])
        ax.set_xlabel("Number of signal jets fed to model")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 102)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        ax.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, out_dir, "report_eff_purity_vs_mult", save_pdf)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Mask-only efficiency vs multiplicity (no objectness gating)
# ─────────────────────────────────────────────────────────────────────────────

def plot_mask_eff_vs_multiplicity(results: dict, out_dir: Path, save_pdf: bool = True):
    """
    Efficiency vs multiplicity using only mask correctness — no objectness gate.
    Works for any model (with or without objectness head).
    Uses the 'breakdown' dict from compute_efficiencies (strict=False).
    """
    br      = results.get("breakdown", {})
    br_orig = results.get("breakdown_orig", {})

    panels = []
    if br:
        panels.append((br,      "Number of signal jets fed to model",  "signal"))
    if br_orig:
        panels.append((br_orig, "Number of jets in full event",         "original"))

    if not panels:
        print("  No per-multiplicity breakdown available — skipping mask-only plot.")
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.2), squeeze=False)

    for col, (table, xlabel, _tag) in enumerate(panels):
        ax    = axes[0, col]
        mults = sorted(table.keys())
        xs    = np.array(mults)

        top_effs   = np.array([table[m]["top_eff"]   * 100 for m in mults])
        W_effs     = np.array([table[m]["W_eff"]     * 100 for m in mults])
        ttbar_effs = np.array([table[m]["ttbar_eff"] * 100 for m in mults])

        ax.plot(xs, top_effs,   "o-", color=C_TOP,   label=r"Top ($t$)",              zorder=3)
        ax.plot(xs, W_effs,     "s-", color=C_W,     label=r"$W$ boson",              zorder=3)
        ax.plot(xs, ttbar_effs, "^-", color=C_TTBAR, label=r"$t\bar{t}$ (both tops)", zorder=3)

        ax.set_xticks(xs)
        ax.set_xticklabels([str(m) for m in mults])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Efficiency (%)")
        ax.set_ylim(0, 102)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        ax.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, out_dir, "report_mask_eff_vs_mult", save_pdf)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, out_dir: Path, stem: str, save_pdf: bool):
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path)
    print(f"  Saved: {png_path}")
    if save_pdf:
        pdf_path = out_dir / f"{stem}.pdf"
        fig.savefig(pdf_path)
        print(f"  Saved: {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate report-ready efficiency plots from chain_queries test outputs."
    )
    parser.add_argument("--run_dir",  required=True, type=Path)
    parser.add_argument("--data_file", default=None, type=Path)
    parser.add_argument("--out_dir",  default=None, type=Path,
                        help="Where to save plots (default: same as --run_dir)")
    parser.add_argument("--use_probs", action="store_true")
    parser.add_argument("--threshold", type=float, default=None, metavar="T")
    parser.add_argument("--obj_threshold", type=float, default=None, metavar="T")
    parser.add_argument("--prior", nargs="+", metavar="TYPE=K", default=[])
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--require_complete_truth", action="store_true")
    parser.add_argument("--require_top_for_w", action="store_true")
    parser.add_argument("--sweep_steps", type=int, default=200,
                        help="Objectness threshold sweep steps (default: 200)")
    parser.add_argument("--no_pdf", action="store_true",
                        help="Skip PDF output, save PNG only")
    args = parser.parse_args()

    # Parse --prior top=3 W=2
    prior_top, prior_W = None, None
    for item in args.prior:
        if "=" not in item:
            sys.exit(f"ERROR: --prior entries must be TYPE=K, got '{item}'")
        name, k_str = item.split("=", 1)
        if name not in ("top", "W"):
            sys.exit(f"ERROR: unknown type '{name}' in --prior. Valid: top, W")
        k = int(k_str)
        if name == "top":
            prior_top = k
        else:
            prior_W = k

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        sys.exit(f"ERROR: --run_dir not found: {run_dir}")

    out_dir = (args.out_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_pdf = not args.no_pdf

    print(f"Loading run data from: {run_dir}")
    (pred_scores_top, target_masks_top,
     pred_scores_W,   target_masks_W,
     jet_valid,
     target_obj_top, pred_obj_top,
     target_obj_W,   pred_obj_W,
     original_mult,
     valid_tops_truth, valid_Ws_truth,
     slot_valid_top, slot_valid_W) = load_run_data(
        run_dir, args.data_file, use_probs=args.use_probs
    )

    print("Computing efficiencies…")
    results = compute_efficiencies(
        pred_scores_top, target_masks_top,
        pred_scores_W,   target_masks_W,
        jet_valid, target_obj_top,
        pred_obj_top=pred_obj_top,
        target_obj_W=target_obj_W,
        pred_obj_W=pred_obj_W,
        prior_top=prior_top, prior_W=prior_W,
        use_probs=args.use_probs,
        threshold=args.threshold,
        obj_threshold=args.obj_threshold,
        strict=False,
        joint=args.joint,
        original_mult=original_mult,
        valid_tops_truth=valid_tops_truth,
        valid_Ws_truth=valid_Ws_truth,
        slot_valid_top=slot_valid_top,
        slot_valid_W=slot_valid_W,
        require_complete_truth=args.require_complete_truth,
        require_top_for_w=args.require_top_for_w,
    )

    print("\n── Figure 1: Efficiency vs multiplicity ─────────────────────────")
    plot_eff_vs_multiplicity(results, out_dir, save_pdf)

    print("\n── Figure 2: Mask-only efficiency vs multiplicity (no objectness) ─")
    plot_mask_eff_vs_multiplicity(results, out_dir, save_pdf)

    if pred_obj_top is not None:
        print("\n── Figure 3: Efficiency–purity trade-off (objectness sweep) ────")
        print(f"   Sweeping objectness threshold in probability space "
              f"({args.sweep_steps} steps)…")
        sweep = sweep_obj_threshold(
            pred_scores_top, target_masks_top,
            pred_scores_W,   target_masks_W,
            jet_valid, pred_obj_top,
            pred_obj_W=pred_obj_W,
            use_probs=args.use_probs,
            threshold=args.threshold,
            prior_top=prior_top, prior_W=prior_W,
            slot_valid_top=slot_valid_top,
            slot_valid_W=slot_valid_W,
            require_complete_truth=args.require_complete_truth,
            require_top_for_w=args.require_top_for_w,
            joint=args.joint,
            n_steps=args.sweep_steps,
        )
        plot_eff_vs_purity(sweep, out_dir, save_pdf)

        print("\n── Figure 4: Efficiency & purity vs multiplicity ───────────────")
        plot_eff_and_purity_vs_mult(results, out_dir, save_pdf)
    else:
        print("\nWARNING: no objectness predictions found — "
              "skipping efficiency–purity trade-off plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
